# 026 SSCA Retry: `traj1 + summary + traj2` 实验总结

日期：2026-06-29  
实验线：SSCA retry / self-summary credit assignment  
代码位置：`recipe/SSCA/`  
主日志：`logs/ssca_retry_grpo_qwen25_15b_alfworld_full_seed2026_ssca_vimpo_full_20260626_233243_20260626_233243.log`

## 1. 一句话结论

这条 `traj1 + sum + traj2` 路线已经工程跑通：模型先执行一次 ALFWorld 轨迹，随后根据第一次轨迹和结果生成一个 summary/retry memory，再从初始环境重置后用 `原始任务 prompt + summary` 执行第二次轨迹；summary 本身作为 policy action 进入训练，并通过 retry 结果获得 credit。

但是当前 seed2026 full 结果显示：summary 的格式质量明显学起来了，整体 validation 成功率也从初始的约 6.6% 提升到约 30.9%，但核心假设没有成立：`traj2` 没有优于 `traj1`。最终 validation 中 `traj1_success=0.336`，`traj2_success=0.281`，retry 的退化率高于改善率，平均 reward delta 仍为负。

## 2. 设计目标

最初目标是验证一种“自生成 retry memory”的两阶段轨迹训练：

```text
traj1:  task prompt -> actions -> reward1
summary: compact(traj1, reward1, feedback) -> retry memory
traj2:  reset env -> task prompt + retry memory -> actions -> reward2
```

关键假设：

1. `traj1` 只依赖原始任务 prompt，不看 summary。
2. `summary` 是 actor 生成的 action，不是后处理文本，因此必须进入 policy training。
3. `traj2` 不能直接看到 `traj1` 的完整轨迹，只能看到 summary/retry memory 和当前 live observation。
4. 如果 summary 有用，`traj2` 应该比 paired `traj1` 更容易成功，或者至少提高 reward。
5. 训练上需要让 summary 的梯度来自后续 `traj2` 的 outcome，而不是只学一个漂亮格式。

## 3. 当前实现结构

### 3.1 Rollout collector

核心 collector 是 `SSCA2TrajCollector`，定义在 `recipe/SSCA/rollout_loop.py:53`。

它的注释中明确当前链路：

- `traj1`: ALFWorld task prompt -> first-attempt actions。
- `summary`: compact observation trace + outcome feedback + summary prompt -> retry memory。
- `traj2`: reset to initial task -> ALFWorld task prompt + retry memory -> retry actions。

训练数据组织为两组：

- `traj1` rows 使用第一次尝试的 outcome reward。
- `summary + traj2` rows 共享一个 retry `uid/traj_uid`，使用第二次 retry outcome reward。

这个设计的目的，是让 summary response tokens 通过后续 retry policy 的质量被优化，而不是只作为不可训练的外部 memory。

### 3.2 Summary prompt

当前 summary instruction 在 `recipe/SSCA/rollout_loop.py:15`。它要求模型输出固定五行：

```text
Task: ...
Known: object=...; receptacle=...; inventory=...
Attempt1: tried=...; bad=...; errors=...
Plan: first=...; backup=...; after_object=...
Rule: one admissible command per action; no or/and-then/list actions; no invented names
```

几个重要约束：

- summary 必须只基于 task、compact first attempt trace、outcome feedback，不允许发明未观察对象或位置。
- retry 从初始场景开始，因此 summary 要写成“retry memory”，而不是继续执行当前状态。
- plan verbs 限制为 ALFWorld 风格，例如 `go to`、`open`、`take`、`put`、`clean`、`heat` 等，避免模型输出不可执行的自然语言计划。
- 最多约 90 words，停止在 `Rule` 行之后。

### 3.3 Summary 输入上下文

summary prompt 的构造在 `recipe/SSCA/rollout_loop.py:491` 附近。当前默认使用 `compact_obs_trace`，不是完整轨迹全文。

每条 summary 输入由三部分组成：

1. `SUMMARY_INSTRUCTION`。
2. `[Compact first attempt trace]`：从 traj1 中挑选最多若干步，包含 observation、action、reward、validity、env feedback 等压缩证据。
3. `[Outcome feedback]`：包含 attempt status、final environment reward、episode length、invalid action count、won flag、goal-condition success rate、final observation 等。

summary prompt 最后明确要求：`Write only the five-line retry memory. Stop after the Rule line.`

### 3.4 Retry prompt / traj2 输入

traj2 的 prompt 拼接在 `recipe/SSCA/rollout_loop.py:525`。

当前形式是：

```text
<current ALFWorld observation / original task prompt>

[Retry memory generated from the first attempt]
<summary text>

Retry context contract: use the current ALFWorld observation plus this retry memory only. Treat the retry as starting from the initial scene; choose exactly one admissible action.
```

注意：traj2 看到的是 live observation + summary，不直接看到 traj1 的完整 history。summary 需要承载“第一次尝试的经验压缩”。

### 3.5 Rollout 执行顺序

主流程在 `recipe/SSCA/rollout_loop.py:655`：

1. `envs.reset()` 得到初始 observation。
2. `_run_env_episode(... phase="traj1")` 执行第一次完整 episode。
3. `_build_summary_obs(...)` 根据 traj1 history、reward1、final info 构造 summary prompt。
4. `_generate_from_obs(...)` 由 actor 生成 summary。
5. 对 summary 做结构质量评分，并在训练时添加小的 summary quality bonus。
6. `envs.reset(retry_same_seed=True)` 回到初始任务。
7. `_prepend_summary_to_obs(...)` 把 summary 放入 retry prompt。
8. `_run_env_episode(... phase="traj2")` 执行第二次完整 episode。
9. 将 `traj1` rows 与 `summary + traj2` rows 合并成训练 batch。

### 3.6 Summary 质量评分

summary quality parser 在 `recipe/SSCA/rollout_loop.py:163`。

它主要检查：

- 五个 label 是否都出现。
- 是否正好按五行顺序输出。
- word count 是否不超过上限。
- 是否没有 `<think>`、`<action>`、markdown wrapper、bullet list 等。
- `Plan:` 是否包含 `first=`、`backup=`、`after_object=` 等字段。
- `first=` 是否以允许的 ALFWorld command prefix 开头。
- 是否避免 `check/inspect/search/verify/try to` 等非动作词。
- task terms 和 summary 是否有一定 overlap。

最终 score 是一个结构化启发式分数。当前训练时 summary row 会获得 `summary_quality_reward_coef * score` 的小 bonus；validation 时不加 bonus。对应逻辑在 `recipe/SSCA/rollout_loop.py:716`。

### 3.7 Credit assignment / reward 组织

合并训练数据的逻辑在 `recipe/SSCA/rollout_loop.py:760`。

当前数据组织：

- `traj1` batch list 使用 `reward1`、`traj_uid_first`。
- `summary + traj2` batch list 使用 `reward2`、`traj_uid_retry`。
- summary row 被放在 retry branch 的第一行，随后是 traj2 的 action rows。
- `summary + traj2` 共享 retry uid，因此 GRPO 会把 summary 作为 retry branch 的 action 之一。

关键 metrics 在 `recipe/SSCA/rollout_loop.py:811`：

- `ssca_traj1_success_rate`
- `ssca_traj2_success_rate`
- `ssca_retry_improved_success_rate`
- `ssca_retry_degraded_success_rate`
- `ssca_metric/retry/reward_delta_mean`
- `ssca_metric/summary/quality_mean`
- `ssca_metric/summary/quality_reward_corr`

### 3.8 VIMPO-style value loss

为了让 `traj2` policy 显式优于 `traj1`，后来加了一个 VIMPO-style auxiliary value loss。

目标构造在 `recipe/SSCA/vimpo_trainer.py:29`：

```text
reward_delta = reward2 - reward1
branch_target = (reward_delta + margin) / reward_scale
```

当前默认：

- `reward_scale = 10.0`
- `margin = 0.1`
- `target_clip = 2.0`
- `target_distribution = split_by_retry_rows`

`split_by_retry_rows` 表示把 branch-level target 分摊到 summary row 和 traj2 action rows 上，而不是每一行都吃完整 target。

VIMPO loss 计算在 `recipe/SSCA/vimpo_core.py:54`：

```text
0.5 * [sum_t beta * (log pi - log pi_ref - sg[KL_t]) - target]^2
```

其中：

- loss 只在 `summary/traj2` rows 上启用。
- `pi_ref` 是同一 state/action 上的 reference log prob，不是直接拿 traj1 的 action 分布，因为 traj1 和 traj2 的 state/action 不同，不能逐 token 对齐。
- paired `traj1` reward 只作为 outcome baseline，用于构造 `reward2 - reward1` target。
- 当前实现没有启用独立 value head；`SSCA_VALUE_HEAD_ENABLE=False`。

actor 更新时，这个 loss 被加到原 policy loss 上。FSDP actor 路径见 `verl/workers/actor/step_ppo_actor.py:630`，DP actor 路径见 `verl/workers/actor/dp_actor.py:461`。

## 4. Full 脚本配置

主脚本：`scripts/run_ssca_retry_grpo_alfworld_1p5b_full.sh`  
三 seed 脚本：`scripts/run_ssca_retry_grpo_alfworld_1p5b_full_3seeds.sh`

当前 full 默认配置要点：

| 项 | 当前值 |
|---|---:|
| model | `Qwen/Qwen2.5-1.5B-Instruct` |
| task | `alfworld/AlfredTWEnv` |
| GPUs | 4 |
| rollout engine | `vllm` |
| tensor model parallel | 2 |
| train data size | 16 |
| val data size | 128 |
| train batch size | 16 |
| group size | 8 |
| max env steps | 50 |
| total epochs / steps | 150 |
| test freq | 5 |
| lr | `1e-6` |
| KL loss | enabled, coef `0.01` |
| invalid action penalty | enabled, coef `0.1` |
| flash attention | `flash_attention_2` |
| Ray CPUs | 60 |
| env worker CPUs | 0.1 |
| heavy RL trace | disabled |
| rollout text dump | enabled/lightweight |

SSCA-specific 默认：

| 项 | 当前值 |
|---|---:|
| summary context mode | `compact_obs_trace` |
| summary trace max steps | 6 |
| summary step obs chars | 240 |
| summary feedback chars | 240 |
| summary max prompt chars | 1200 |
| summary max chars | 1200 |
| summary quality reward coef | 0.2 |
| summary quality max words | 110 |
| retry same seed | True |
| VIMPO enable | True |
| VIMPO beta | 0.01 |
| VIMPO value loss coef | 0.05 |
| VIMPO target distribution | `split_by_retry_rows` |
| value head | disabled |

## 5. 实际运行记录

目前保留下来的 SSCA full 运行主要有两条：

| run | seed | 状态 | 日志 |
|---|---:|---|---|
| early SSCA retry | 2026 | 跑到 step 56/150 后中断 | `logs/ssca_retry_grpo_qwen25_15b_alfworld_full_seed2026_20260626_164746.log` |
| SSCA + VIMPO full | 2026 | 跑满 150/150；final validation 后 DataLoader worker 被 kill | `logs/ssca_retry_grpo_qwen25_15b_alfworld_full_seed2026_ssca_vimpo_full_20260626_233243_20260626_233243.log` |

没有看到 seed2077、seed2501 的完整 SSCA full 日志，因此下面的结论只基于 seed2026，不能作为 3-seed 稳健结论。

## 6. Validation 曲线

下表来自 SSCA + VIMPO full seed2026 日志。

| step | val/text/test_score | val/success_rate | val/traj1_success | val/traj2_success | retry_improved | retry_degraded | reward_delta_mean | summary_quality | five_label_rate | plan_ok_rate | quality_reward_corr |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.328 | 0.066 | 0.086 | 0.047 | 0.023 | 0.062 | -0.391 | 0.296 | 0.148 | 0.094 | 0.113 |
| 5 | 0.264 | 0.062 | 0.086 | 0.039 | 0.039 | 0.086 | -0.469 | 0.493 | 0.547 | 0.219 | -0.093 |
| 10 | 0.604 | 0.109 | 0.133 | 0.086 | 0.078 | 0.125 | -0.469 | 0.539 | 0.711 | 0.297 | -0.114 |
| 15 | 0.307 | 0.082 | 0.109 | 0.055 | 0.047 | 0.102 | -0.547 | 0.605 | 0.758 | 0.125 | 0.032 |
| 25 | 0.492 | 0.121 | 0.203 | 0.039 | 0.031 | 0.195 | -1.641 | 0.625 | 0.750 | 0.305 | -0.057 |
| 50 | 1.107 | 0.230 | 0.281 | 0.180 | 0.109 | 0.211 | -1.016 | 0.705 | 0.695 | 0.758 | 0.026 |
| 75 | 1.124 | 0.254 | 0.297 | 0.211 | 0.125 | 0.211 | -0.859 | 0.749 | 0.742 | 0.961 | 0.002 |
| 100 | 1.022 | 0.223 | 0.258 | 0.188 | 0.102 | 0.172 | -0.703 | 0.751 | 0.734 | 0.969 | 0.005 |
| 125 | 0.839 | 0.180 | 0.219 | 0.141 | 0.109 | 0.188 | -0.781 | 0.760 | 0.734 | 0.992 | -0.133 |
| 150 | 1.203 | 0.309 | 0.336 | 0.281 | 0.172 | 0.227 | -0.547 | 0.759 | 0.734 | 0.992 | -0.061 |

最终 validation metrics 出现在 `logs/ssca_retry_grpo_qwen25_15b_alfworld_full_seed2026_ssca_vimpo_full_20260626_233243_20260626_233243.log:4078` 和 `logs/ssca_retry_grpo_qwen25_15b_alfworld_full_seed2026_ssca_vimpo_full_20260626_233243_20260626_233243.log:4079`。

## 7. 最终 step 详细指标

### 7.1 整体任务指标

最终 step150：

| metric | value |
|---|---:|
| `val/text/test_score` | 1.203 |
| `val/success_rate` | 0.309 |
| `episode/success_rate` | 0.605 |
| `episode/reward/mean` | 6.071 |
| `response_length/mean` | 110.699 |
| `prompt_length/mean` | 617.534 |
| `timing_s/step` | 610.902 |
| `timing_s/testing` | 274.708 |
| `perf/throughput` | 2212.467 |

说明：`actor/lr` 在 console 中显示为 `0.000`，这是三位小数格式导致，实际脚本 lr 是 `1e-6`。

### 7.2 traj1 vs traj2

最终 validation：

| metric | value |
|---|---:|
| `val/ssca_traj1_success_rate` | 0.336 |
| `val/ssca_traj2_success_rate` | 0.281 |
| `val/ssca_retry_improved_success_rate` | 0.172 |
| `val/ssca_retry_degraded_success_rate` | 0.227 |
| `val/ssca_metric/traj1/reward_mean` | 3.359 |
| `val/ssca_metric/traj2/reward_mean` | 2.812 |
| `val/ssca_metric/retry/reward_delta_mean` | -0.547 |
| `val/ssca_metric/traj1/length_mean` | 37.219 |
| `val/ssca_metric/traj2/length_mean` | 41.195 |
| `val/ssca_metric/traj1/valid_action_ratio` | 0.998 |
| `val/ssca_metric/traj2/valid_action_ratio` | 0.994 |
| `val/ssca_metric/traj1/invalid_count_mean` | 0.109 |
| `val/ssca_metric/traj2/invalid_count_mean` | 0.250 |

最终 train/episode 侧：

| metric | value |
|---|---:|
| `episode/ssca_traj1_success_rate` | 0.797 |
| `episode/ssca_traj2_success_rate` | 0.414 |
| `episode/ssca_retry_improved_success_rate` | 0.070 |
| `episode/ssca_retry_degraded_success_rate` | 0.453 |
| `episode/ssca_metric/traj1/reward_mean` | 7.969 |
| `episode/ssca_metric/traj2/reward_mean` | 4.141 |
| `episode/ssca_metric/retry/reward_delta_mean` | -3.828 |

这说明在训练 batch 上，traj1 已经非常强，而 summary-conditioned traj2 反而显著弱于 traj1。

### 7.3 Summary 质量指标

最终 validation：

| metric | value |
|---|---:|
| `val/ssca_metric/summary/quality_mean` | 0.759 |
| `val/ssca_metric/summary/quality_std` | 0.079 |
| `val/ssca_metric/summary/label_count_mean` | 4.453 |
| `val/ssca_metric/summary/five_label_rate` | 0.734 |
| `val/ssca_metric/summary/ordered_five_line_rate` | 0.000 |
| `val/ssca_metric/summary/plan_ok_rate` | 0.992 |
| `val/ssca_metric/summary/word_count_mean` | 89.898 |
| `val/ssca_metric/summary/bonus_mean` | 0.000 |
| `val/ssca_metric/summary/quality_reward_corr` | -0.061 |

解读：

- `quality_mean` 从 step0 的 0.296 提升到 0.759，summary 格式确实被优化了。
- `five_label_rate` 从 0.148 提升到 0.734，说明模型学会了输出五个指定 label。
- `plan_ok_rate` 从 0.094 提升到 0.992，说明 plan 行的动作风格约束基本学会了。
- `ordered_five_line_rate=0.000` 说明 strict parser 认为 summary 并非“恰好五行且每行按 label 开头”。这可能是 parser 过严，也可能是模型仍输出多余换行/额外文本。
- `quality_reward_corr=-0.061` 是最关键的负信号：summary 的结构质量和最终 retry reward 没有形成正相关。

### 7.4 VIMPO-style value loss 指标

最终 step150：

| metric | value |
|---|---:|
| `actor/ssca_vimpo_value_loss` | 0.000 |
| `actor/ssca_vimpo_value_loss_raw` | 0.002 |
| `actor/ssca_vimpo_value_loss_x1e4` | 0.761 |
| `actor/ssca_vimpo_value_loss_raw_x1e4` | 15.214 |
| `actor/ssca_vimpo_active_row_ratio` | 0.570 |
| `actor/ssca_vimpo_terminal_value_mean_active` | -0.018 |
| `actor/ssca_vimpo_terminal_target_mean_active` | -0.019 |
| `actor/ssca_vimpo_rmse_active` | 0.070 |
| `actor/ssca_vimpo_mae_active` | 0.049 |
| `actor/ssca_vimpo_value_target_corr_active` | 0.003 |
| `actor/ssca_vimpo_sign_acc_active` | 0.488 |
| `actor/ssca_vimpo_target_positive_rate_active` | 0.414 |
| `actor/ssca_vimpo_value_positive_rate_active` | 0.540 |

解读：

- VIMPO loss 数值很小，不是训练崩溃信号。
- 但 `value_target_corr_active=0.003`、`sign_acc_active=0.488`，几乎等于随机，说明 policy-implied value 没有有效学到 “retry 是否比 traj1 好”。
- target 本身被 `split_by_retry_rows` 分摊，加上 `beta=0.01`、`value_loss_coef=0.05`，最终监督可能过弱。
- 训练最终 `reward_delta_mean` 明显为负，value loss 的 target 多数是在告诉模型 retry branch 不如 first branch，但它没有转化为“生成更有用 summary 并让 traj2 变好”的行为改进。

## 8. 和 GraphGPO 的参照

同一个 repo 下有 seed2026 的 GraphGPO ALFWorld full 完整日志：

`logs/graphgpo_qwen25_15b_alfworld_full_seed2026_20260622_051648.log`

其最终 step150 约为：

| run | step | val/success_rate | episode/success_rate | timing_s/step |
|---|---:|---:|---:|---:|
| GraphGPO seed2026 | 150 | 0.883 | 0.906 | 235.068 |
| SSCA + VIMPO seed2026 | 150 | 0.309 | 0.605 | 610.902 |

这不是严格的算法公平比较，因为 SSCA rollout 每个 prompt 要执行 `traj1 + summary + traj2`，训练/验证成本更高，且 reward 和 batch 组织也不同。但作为工程结果参照，当前 SSCA 远弱于已跑通的 GraphGPO。

## 9. 主要问题分析

### 9.1 Summary 学到了格式，但没有学到 outcome-useful memory

最明显的现象是：

- `summary_quality_mean` 明显提升。
- `plan_ok_rate` 接近 1。
- 但 `quality_reward_corr` 接近 0 或为负。
- `traj2_success` 长期低于 `traj1_success`。

所以当前 summary reward 主要在优化“结构正确”，不是优化“帮助 retry 成功”。这解释了为什么 summary 看起来变规范，但没有推动 traj2。

### 9.2 traj2 的输入分布更复杂，可能反而干扰 action policy

traj2 prompt 比 traj1 多了 retry memory，平均 prompt length 更长；最终 `traj2_invalid_count_mean=0.250` 高于 `traj1_invalid_count_mean=0.109`，valid action ratio 也略低。

这说明 summary 可能在部分场景中引入额外干扰：模型既要读 live observation，又要读 memory，还要避免把 summary 中的 plan 当成可直接复制的动作序列。如果 summary 中出现过时、错误或不够 grounded 的信息，traj2 会更容易走偏。

### 9.3 “成功 traj1 后再 retry” 会自然制造退化样本

当前每个 prompt 都执行 retry，不管 traj1 成功还是失败。随着训练进行，traj1 本身越来越强，很多 paired 样本中 traj1 已经成功；此时 traj2 必须重复成功才不退化。

最终 train 侧 `traj1_success=0.797`，`traj2_success=0.414`，这会产生大量负 delta。也就是说，当前机制在后期更多是在训练“不要比已经成功的 traj1 差”，而不是专注于“从失败经验中改善”。

### 9.4 VIMPO target 信号可能过弱或方向不够直接

当前 VIMPO target 是 `reward2 - reward1`，并按 retry rows 分摊。summary row 与 traj2 action rows 共同承担 target。

问题是：

- summary 是 retry branch 的关键条件变量，但 target 被分摊到很多 action rows 后，summary token 获得的直接监督变弱。
- `beta=0.01` 和 `value_loss_coef=0.05` 使辅助 loss 相对小。
- final `value_target_corr_active=0.003` 表明 VIMPO implied value 与 target 没建立相关性。
- 如果 paired reward delta 多为负，value loss 可能更像“压低 retry branch log-ratio”，而不是明确告诉模型如何产生更好的 summary。

### 9.5 validation 通过，但结束不干净

SSCA + VIMPO full 已经跑满 150 step 并打印 final validation metrics，但最后有：

`RuntimeError: DataLoader worker (pid ...) is killed by signal: Killed.`

该错误出现在 final validation metrics 和 progress 100% 之后，因此不影响已记录指标；但说明运行结束阶段仍有资源/worker 清理问题。这个问题可能和长时间运行后的 DataLoader/Ray worker 资源回收有关。

## 10. 当前结论

当前版本可以认为：

1. 工程链路成立：能跑 `traj1 -> summary -> traj2`，summary 作为 action 进入 batch 和 actor loss。
2. summary prompt v3 的格式控制有效：五标签、plan 约束、word limit 等指标明显改善。
3. 但核心算法目标未达成：`traj2` 没有比 `traj1` 更好，retry 平均 delta 仍为负。
4. VIMPO-style value loss 已接入，但没有显示出有效学习 paired improvement 的迹象。
5. 不建议直接扩大到 3 seed 当作正式结果；更合理的是先改机制，再重新 smoke/full。

## 11. 建议的下一版方向

### 11.1 只对失败或低分 traj1 启动 retry 训练

当前所有样本都 retry，会让成功 traj1 生成大量退化风险。建议下一版：

- 对 `reward1 <= threshold` 的样本启用 summary+traj2 主训练。
- 对成功 traj1，可选择不 retry，或只做“preserve plan”辅助分析，不进入主要 policy loss。
- metrics 分开报告 failed-traj1 subset 的 retry improvement。

这样目标会更接近“失败后总结经验并改善”。

### 11.2 把 summary 的 outcome credit 单独加强

当前 summary 和 traj2 action rows 共享 retry reward，但 summary token 的监督被稀释。可以考虑：

- summary row 单独吃完整 `reward2 - reward1` target。
- traj2 rows 继续吃 GRPO reward2。
- 或者 summary row 使用 pairwise objective：成功改善的 summary 增强，导致退化的 summary 抑制。

### 11.3 降低结构 bonus，增加 outcome-aligned summary reward

当前 summary quality 学得快，但和 reward 不相关。可以改为：

- 保留最小格式约束 bonus，例如 label 完整即可。
- summary quality bonus 不再直接按结构分数给高权重。
- 加入 outcome-aligned signal，例如 `bonus = alpha * max(reward2 - reward1, 0)` 或基于 improvement label 的 binary reward。

### 11.4 改 retry prompt，避免 summary 干扰动作选择

当前 summary 直接附在 observation 后，可能导致模型复制 plan 或被错误 memory 牵引。下一版可尝试：

- 把 summary 放入更明确的 system/developer style memory section。
- 强化“当前 admissible actions 优先，memory 只用于定位策略”的约束。
- 在 action prompt 前重新列出 admissible-action constraint，避免 plan 变成非法自然语言动作。

### 11.5 把指标按 traj1 outcome 分桶

现有总指标不足以诊断。建议新增：

- `retry_given_traj1_failed_success_rate`
- `retry_given_traj1_success_success_rate`
- `reward_delta_given_traj1_failed_mean`
- `reward_delta_given_traj1_success_mean`
- `summary_quality_given_improved_mean`
- `summary_quality_given_degraded_mean`
- `summary_contains_target_object_rate`
- `summary_contains_target_receptacle_rate`
- `summary_plan_first_is_admissible_rate`

这样能回答：summary 到底是否帮助失败样本，还是只在成功样本上造成退化。

## 12. 复现实验入口

单 seed full：

```bash
bash /prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/scripts/run_ssca_retry_grpo_alfworld_1p5b_full.sh
```

三 seed 串行脚本：

```bash
bash /prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/scripts/run_ssca_retry_grpo_alfworld_1p5b_full_3seeds.sh
```

不过基于当前 seed2026 结果，我建议先不要直接跑 3 seed full，而是先实现上面的失败样本 gating 和 summary credit 强化，再跑短 smoke/full 验证。
