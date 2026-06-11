# 022 Soft-GiGPO 前期 Trace 采集记录

## 目的

本次前期实验服务于 `021_soft_gigpo_state_matching_theory` 中的状态软匹配想法：先不改训练算法，只用已经训练好的 checkpoint 在 WebShop train 子集上采集 step-level trace，观察哪些信号可能用于 step-level group 或 soft neighbor 选择。

核心问题是：GiGPO 现在依赖 `anchor_obs` 的硬匹配来构造 step group；如果硬匹配把大量可比较状态拆散，那么后续可以考虑用更软的“状态既视感”补充 group。但在接入训练前，需要先离线验证哪些信号可靠。

## 采集方式

### 运行入口

复现实验脚本：

```bash
EXPS/analysis/021_soft_gigpo_trace/run_collect_v3_ckpt_train_subset.sh
```

实际成功采集的 run tag：

```bash
RUN_TAG=20260610_021_prelim_rerun \
EXPS/analysis/021_soft_gigpo_trace/run_collect_v3_ckpt_train_subset.sh
```

输出目录：

```text
EXPS/analysis/021_soft_gigpo_trace/v3_ckpt250_train_subset_20260610_021_prelim_rerun/
```

### checkpoint 与 dataloader 处理

使用的模型 checkpoint 是已训练 StepPPO-v3 value-head 实验的：

```text
checkpoints/verl_agent_webshop/step_ppo_v3_value_head_detach_false_qwen2.5_1.5b_4gpu_paper_align_full_e250/global_step_250
```

第一次直接 resume 时没有产生 trace，原因是 checkpoint 内的 `data.pt` 会恢复 dataloader 状态，导致从 `global_step_250` 恢复后 dataloader 已在末尾。为避免这个问题，采集脚本默认 `SKIP_DATALOADER_STATE=1`，会在 `/tmp/ziyuhuan/` 下创建临时 checkpoint 目录，只软链接原 checkpoint 的 `actor/` 和 `critic/`，不带 `data.pt`。这样模型权重来自 `global_step_250`，但 train dataloader 从子集开头重新取一个 batch。

脚本还会从 `global_step_250` 自动推导：

```text
trainer.total_training_steps = 251
```

因此训练循环只进入一个 train batch。

### 训练/采样配置

本次采集保持 StepPPO-v3 value-head 训练脚本的基础配置，但关闭实际更新与周期性副作用：

- `trainer.val_before_train=False`：不做开头 validation。
- `trainer.test_freq=-1`：不做训练中 validation。
- `trainer.save_freq=-1`：不保存新 checkpoint。
- `trainer.critic_warmup=999999`：阻止 actor update；本次只做 rollout、old log prob、ref log prob、advantage 和 value diagnostic 计算。
- `trainer.logger='[console]'`：只写 console log。
- `trainer.value_diagnostics_dir=<trace_dir>/value_diagnostics`：打开 step-level value diagnostic。
- `trainer.value_diagnostics_interval=1`：本步必 dump。
- `trainer.value_diagnostics_max_rows=0`：dump 全部 step rows。
- `trainer.value_diagnostics_include_text=True`：额外 dump decoded prompt/response 文本。
- `trainer.rollout_data_dir=<trace_dir>/rollout_generations`：额外 dump decoded rollout generations。

WebShop 子集规模来自现有脚本：

- `data.train_batch_size=16`
- `env.rollout.n=8`
- `env.max_steps=15`

因此理论上是 16 个 train tasks，每个 task 采 8 条 trajectory；实际每条 trajectory 会在 done 后停止贡献 active step。

## 原始采集文件

### value diagnostics

主文件：

```text
EXPS/analysis/021_soft_gigpo_trace/v3_ckpt250_train_subset_20260610_021_prelim_rerun/value_diagnostics/step_000251.jsonl
```

每一行对应一个 active environment step。已采到 896 行，覆盖 16 个任务、128 条 trajectory，平均每条 trajectory 7.00 个 active step。

逐行字段如下：

| 字段 | 含义 |
| --- | --- |
| `global_step` | trainer global step，本次为 251。 |
| `epoch` | 当前 epoch，本次为 0。 |
| `row_idx` | dump 文件中的行号。 |
| `traj_uid` | 单条 trajectory 的唯一 id。 |
| `step_idx` | trajectory 内 step index。 |
| `step_sample_uid` | step sample 唯一 id，格式近似 `traj_uid:step_idx:row_idx`。 |
| `is_action_valid` | 环境返回的 action 是否有效。 |
| `raw_reward` | 当前 step 的即时环境 reward。 |
| `episode_reward` | 该 trajectory 的最终累计 reward。 |
| `episode_length` | 该 trajectory 的最终长度。 |
| `value_pred_before_update` | actor-side value head 在 update 前对当前 step state 的预测。 |
| `value_target_gae_return` | value head 使用的 step-level GAE return target。 |
| `value_error_before_update` | `value_pred_before_update - value_target_gae_return`。 |
| `value_step_reward` | value target 构造中使用的 shaped/immediate step reward。 |
| `gigpo_step_reward` | GiGPO 使用的 step discounted return。 |
| `policy_advantage_mean` | response token 上 policy advantage 的 mask mean。 |
| `policy_return_mean` | response token 上 policy return 的 mask mean。 |
| `token_score_sum` | token-level score 求和。 |
| `token_reward_sum` | token-level reward 求和。 |
| `prompt_length` | 当前 step prompt token 数。 |
| `response_length` | 当前 step response token 数。 |
| `prompt_text` | decoded 当前 pre-action prompt，包含任务、历史、当前 observation 和 admissible actions。 |
| `response_text` | decoded model response，通常包含 `<think>` 和 `<action>`。 |

汇总文件：

```text
EXPS/analysis/021_soft_gigpo_trace/v3_ckpt250_train_subset_20260610_021_prelim_rerun/value_diagnostics/summary.jsonl
```

它记录本次 value diagnostic 的全局统计，包括 value pred/target 均值方差、RMSE、MAE、correlation、value position、是否 detach backbone、valid action ratio 等。

### rollout generations

辅助文件：

```text
EXPS/analysis/021_soft_gigpo_trace/v3_ckpt250_train_subset_20260610_021_prelim_rerun/rollout_generations/251.jsonl
```

该文件主要用于人工阅读模型行为，每行包含 decoded `input`、`output`、`score`、`step`。相比 value diagnostics，它缺少 value/advantage 等训练内信号，但更适合快速检查模型在某个 observation 下输出了什么 action。

## 后处理分析产物

分析目录：

```text
EXPS/analysis/021_soft_gigpo_trace/v3_ckpt250_train_subset_20260610_021_prelim_rerun/analysis/
```

主要产物：

| 文件 | 内容 |
| --- | --- |
| `README_021_prelim.md` | 本次实验的人类可读摘要。 |
| `candidate_signal_summary.json` | 机器可读的核心统计。 |
| `step_trace_rows.csv` | 从 JSONL 展平后的逐 step 表。 |
| `trajectory_summary.csv` | 按 `traj_uid` 汇总 episode reward、length、valid rate。 |
| `task_summary.csv` | 按任务汇总 8 条 rollout 的表现。 |
| `summary_by_step_idx.csv` | 按 step index 汇总的 reward/value/valid/length 指标。 |
| `summary_by_stage.csv` | 按粗粒度 stage 汇总。 |
| `summary_by_action_cmd.csv` | 按 action 类型汇总。 |
| `group_summary_by_task_exact_obs.csv` | 用 `task + current_observation` 近似 hard anchor group 的统计。 |
| `group_summary_by_task_step.csv` | 用 `task + step_idx` 构造 group 的统计。 |
| `group_summary_by_task_stage.csv` | 用 `task + stage` 构造 group 的统计。 |
| `task_step_obs_pair_sample.csv` | `task + step_idx` 内 observation pair 文本相似度样例。 |

后处理从 `prompt_text` 中解析了：

- `task_text`
- `current_observation`
- 当前 step number

从 `response_text` 中解析了：

- `<action>...</action>` 原文
- `action_cmd`，例如 `search`、`click`
- `action_arg`

另外按 observation 内容打了一个粗粒度 `stage`，例如：

- `search_home`
- `search_results`
- `product_detail`
- `product_or_option`
- `terminal_or_done`
- `other`

这些派生字段只用于离线分析，不参与训练。

## 当前主要结论

### 1. hard anchor 确实会产生大量碎片组

用 `task + exact current_observation` 近似当前 hard anchor group，共得到 160 个 group：

- cluster mean size = 5.60
- singleton group 比例 = 57.5%
- max group size = 71

这说明 hard match 的问题是真实存在的：很多 step 状态无法形成有效组内比较，后续 soft matching 最应该优先补这些 singleton 或小 group。

### 2. 纯文本软匹配增益很有限

在 `task + step_idx` 内计算 observation pair 的 SequenceMatcher 文本相似度：

- mean similarity = 0.821
- p90 = 1.000
- similarity >= 0.95 的比例 = 72.6%
- exact match 的比例 = 72.6%

`>=0.95` 与 exact match 基本一样，说明简单字符串相似度几乎只是在复现硬匹配，没有明显扩展可匹配样本。若要做“模型既视感”，不应依赖 raw text similarity。

### 3. step index 是强先验，但不能单独作为 group key

`task + step_idx` 得到 123 个 group：

- cluster mean size = 7.28
- singleton group 比例 = 1.6%
- max group size = 11

这个粒度显著扩大 group，但会混入不同页面状态，例如 search result、product detail、terminal 等。它适合做候选过滤条件，但不能单独作为 advantage group。

### 4. stage 适合作 prefilter，不适合直接做 group

`task + stage` 得到 48 个 group：

- cluster mean size = 18.67
- singleton group 比例 = 0%
- max group size = 71

stage 能有效扩大候选池，但过粗，会把价值差异较大的状态混在一起。因此 stage 更像 action-mask/state-family prefilter，而不是最终 group key。

### 5. value head 当前是有用的 diagnostic 信号

本次 value head 的统计：

- `value_pred_before_update` 与 `value_target_gae_return` Pearson correlation = 0.881
- RMSE = 2.019
- MAE = 1.007

value prediction 与 step-level return target 的相关性较强，说明它可以用于 confidence、ranking 或 sanity check。但不建议把 value prediction 单独作为 matching key，因为这会退化成 value-only matching，容易把“值接近但状态/动作机会不同”的样本混在一起。

### 6. response clipping 是重要噪声信号

本次 row-level response clip ratio = 46.3%。这意味着相当多 response 到达 `max_response_length=512`，这些样本可能包含格式拖长、思考过长或 action 输出不稳定的问题。后续做 soft neighbor 审计时，`response_length==512` 应作为低置信或过滤信号，而不是状态相似信号。

## 当前未采到但后续需要补的字段

这次采集已经够做第一轮离线观察，但还不足以直接验证 021 中推荐的 policy-state soft matching。主要缺口如下：

1. `uid`
   - rollout 内部有 `uid`，每 8 条同 task trajectory 共享一个 `uid`；但当前 `value_diagnostics` 没有 dump。
   - 没有 `uid` 就只能用 `task_text` 反推同任务 group，不能精确复现 GiGPO 的 group 边界。

2. 原始 `anchor_obs`
   - GiGPO 真正 hard group 用的是 `anchor_obs`。
   - 当前分析用从 `prompt_text` 解析出的 `current_observation` 近似，不保证与内部 `anchor_obs` 完全一致。

3. action mask / admissible action set
   - 021 的理论更推荐先按 action opportunity 过滤。
   - 当前 admissible actions 只在 prompt 文本里，尚未结构化 dump。

4. pre-action hidden/state embedding
   - 如果要做“模型既视感”，需要 frozen actor 在当前 pre-action state 上的 hidden 表示，而不是字符串相似度。

5. action distribution / entropy / top-k action logprob
   - 021 推荐用 frozen actor action distribution 距离衡量策略状态等价性。
   - 当前只有 response token logprob 参与训练流程，没有离线 dump 成每个 state 的候选动作分布。

## 建议下一步

短期先做离线审计，不接入训练：

1. 在 `value_diagnostics` 中补 `uid` 和原始 `anchor_obs`，重新采同等规模 trace。
2. 结构化解析 admissible action set，得到 action-mask/stage prefilter。
3. 对每个 step state dump frozen actor 的 pre-action hidden 或 action distribution summary。
4. 在同 `uid` 内做 leave-one-out top-k soft neighbor：先按 stage/action-mask 过滤，再按 hidden 或 action-distribution 距离选邻居。
5. 比较 hard group、`task+step_idx`、`stage` prefilter、soft top-k neighbor 的 target 方差、episode reward 一致性、valid action ratio 和 singleton 覆盖率。

只有当 soft neighbor 在这些离线指标上比 hard group 更稳，并且不会明显混入不同 action opportunity 的状态，才考虑进入训练 advantage 路径。
