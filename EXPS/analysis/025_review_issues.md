# 025 Review 问题说明：StepPPO-v3 脚本与 SSCA EXP 文档

日期：2026-06-29  
分支：`0611_trace_colllect`  
当前 HEAD：`802f838 docs: add SSCA retry experiment summary`  
Review 范围：

- 本地未提交脚本：`EXPS/run_webshop_step_ppo_v3_value_head_4gpu_paper_align_simple.sh`
- 最新 EXP 文档：`EXPS/analysis/026_ssca_retry_experiment_summary.md`
- 对应 TeX/PDF：`EXPS/tex/024_ssca_retry_experiment_summary.tex`、`EXPS/024_ssca_retry_experiment_summary.pdf`
- 当前包内 SSCA 代码：`recipe/SSCA/`

## 1. 总结

这次 review 发现两个高优先级问题：

1. 最新 SSCA EXP 文档描述的是一个更完整的 `SSCA + VIMPO + summary quality` 版本，但当前 packaged copy 里的 `recipe/SSCA/` 代码并不包含这些实现。
2. 本地 StepPPO-v3 wrapper 被改成了 `detach_value_backbone=False` 主入口，这和当前 StepPPO-v3 规则中的 safe default 不一致。

此外还有两个工程一致性问题：`EXPS/` 与 `exps/` 大小写路径并存，以及 SSCA 文档编号在 Markdown 与 TeX/PDF 之间不一致。

## 2. High：SSCA EXP 文档与当前代码不一致

### 2.1 文档声称的实现

`EXPS/analysis/026_ssca_retry_experiment_summary.md` 描述了如下能力：

- 固定五行 summary prompt：
  - `Task:`
  - `Known:`
  - `Attempt1:`
  - `Plan:`
  - `Rule:`
- summary 输入使用 `compact_obs_trace`，不是完整轨迹。
- 有 summary quality parser，统计 `five_label_rate`、`ordered_five_line_rate`、`plan_ok_rate` 等指标。
- summary row 在训练时获得 `summary_quality_reward_coef * score` 的小 bonus。
- 有 VIMPO-style auxiliary value loss：
  - `recipe/SSCA/vimpo_trainer.py`
  - `recipe/SSCA/vimpo_core.py`
- 有 full 运行脚本：
  - `scripts/run_ssca_retry_grpo_alfworld_1p5b_full.sh`
  - `scripts/run_ssca_retry_grpo_alfworld_1p5b_full_3seeds.sh`
- 记录了 `ssca_traj1_success_rate`、`ssca_traj2_success_rate`、`ssca_retry_improved_success_rate`、`reward_delta_mean`、`summary_quality_reward_corr` 等指标。

### 2.2 当前代码实际情况

当前包内 `recipe/SSCA/` 只有：

- `recipe/SSCA/__init__.py`
- `recipe/SSCA/main_ssca.py`
- `recipe/SSCA/rollout_loop.py`

并没有：

- `recipe/SSCA/vimpo_trainer.py`
- `recipe/SSCA/vimpo_core.py`
- `scripts/run_ssca_retry_grpo_alfworld_1p5b_full.sh`
- `scripts/run_ssca_retry_grpo_alfworld_1p5b_full_3seeds.sh`

代码里的 summary prompt 也不是文档描述的五行格式。当前 `SUMMARY_INSTRUCTION` 在 `recipe/SSCA/rollout_loop.py:14`，要求的是四段 policy-improvement note：

- State belief
- Failure / success causes
- Policy improvement
- Retry plan

当前 `_build_summary_obs` 在 `recipe/SSCA/rollout_loop.py:189`，拼接的是：

- `[Original task prompt]`
- `[First trajectory]`
- `[Outcome feedback]`

其中 `[First trajectory]` 来自 `"\n".join(history)`，再按 `summary_max_history_chars` 截断；这不是文档里的 `compact_obs_trace`。

当前 summary row 的 reward 在 `recipe/SSCA/rollout_loop.py:387` 被设为 0：

```python
summary_batch.non_tensor_batch["rewards"] = np.zeros(batch_size, dtype=object)
```

没有看到 summary quality parser，也没有看到 `summary_quality_reward_coef * score` 的 bonus 接入。

当前 `vanilla_multi_turn_loop` 最终只返回：

- `success_rate`
- `ssca_retry_prompt_match_rate`

没有文档中列出的 traj1/traj2 paired 指标、retry improved/degraded 指标、reward delta 指标和 summary quality 指标。

### 2.3 影响

如果读者只看这份 packaged copy，会以为 SSCA+VIMPO full 代码已经在仓库中，但实际无法按文档复现。

这会影响三件事：

1. 复现实验入口不可用。
2. 文档中的指标口径无法从当前代码验证。
3. “VIMPO-style value loss 已接入但没学到”的结论无法由当前包内代码支撑，只能理解为来自原始工作区或未打包代码。

### 2.4 建议

二选一处理：

1. 如果完整 SSCA+VIMPO 代码存在于原始工作区：把对应代码、脚本和必要轻量日志补进当前分支，或在文档开头明确说明“本文基于未包含在 packaged copy 中的原始工作区实现”。
2. 如果当前 packaged copy 就是要 review 的唯一代码：把 `026` 文档降级为当前实现说明，删除或标注尚未落地的 VIMPO、summary quality、full 脚本和指标描述。

## 3. High：StepPPO-v3 主 wrapper 被改成 detach-false

### 3.1 当前本地修改

本地未提交脚本 `EXPS/run_webshop_step_ppo_v3_value_head_4gpu_paper_align_simple.sh` 把主入口从 detach-true 改成了 detach-false：

- base script 从 `run_webshop_gigpo_v1_value_aux_detach_true_...` 改为 `run_webshop_gigpo_v1_value_aux_detach_false_...`
- `DEFAULT_EXPERIMENT` 改为 `step_ppo_v3_value_head_detach_false_...`
- `actor_rollout_ref.actor.value_head.detach_value_backbone=False`
- log label 改为 `step_ppo_v3_value_head_detach_false`

### 3.2 和当前 StepPPO-v3 规则冲突

当前 StepPPO-v3 规则要求：

- v3 是 GiGPO 上的 value-head-only 实验。
- GiGPO policy training 应保持不变。
- `detach_value_backbone=true` 是 main v3 value-head run 的 safe default。
- 目标是检查 value head 能否稳定学习，同时尽量不影响 GiGPO。

但 `detach_value_backbone=False` 会让 value loss 梯度回流 actor backbone。实现上，`verl/workers/actor/value_head.py:86` 只有在 `detach_value_backbone=True` 时才会对 hidden states 做 detach；否则 value head 的 loss 会通过 hidden states 更新 backbone。

同时，`verl/workers/actor/step_ppo_actor.py:554` 附近会把 value loss 加进 actor policy loss：

```python
policy_loss = policy_loss + value_loss_coef * value_loss
```

因此 detach-false 不再是“只观察 value head 学习是否稳定”的主实验，而是会改变 actor backbone 优化路径的 ablation。

### 3.3 影响

这会混淆 StepPPO-v3 的主要问题：

- 如果 policy 指标变差，无法区分是 value head 本身的问题，还是 value loss 更新 backbone 导致 GiGPO policy 被扰动。
- 如果 value metric 变好，也不能说明 detach-true safe default 下 value head 能稳定学习。
- 实验名虽然写了 `detach_false`，但文件名仍是主 v3 wrapper，容易被误用为 main run。

### 3.4 建议

建议恢复主 wrapper 为 detach-true：

- `BASE_SCRIPT` 指向 detach-true base。
- `DEFAULT_EXPERIMENT` 使用 `step_ppo_v3_value_head_detach_true_...`。
- `actor_rollout_ref.actor.value_head.detach_value_backbone=True`。
- 保留 value diagnostics 配置可以，但应确保实验名和诊断目录也写 `detach_true`。

如果确实要继续 detach-false 实验，建议新增一个明确的 ablation 脚本，例如：

```text
EXPS/run_webshop_step_ppo_v3_value_head_detach_false_4gpu_paper_align_simple.sh
```

这样 main v3 入口和 ablation 入口不会混在一起。

## 4. Medium：`EXPS/` 与 `exps/` 路径大小写并存

当前 Git 中同时跟踪了 `EXPS/...` 和 `exps/...` 下的脚本。当前 macOS 工作区里 `EXPS` 和 `exps` 指向同一个 inode，因此本地看起来像同一个目录；但 Linux 训练机通常是大小写敏感文件系统，两套路径会表现为两个不同目录。

这会带来几个风险：

- 本地修改显示在 `EXPS/...`，但远端训练脚本可能实际调用 `exps/...`。
- 在 macOS 上不容易察觉大小写路径差异。
- 提交、pull、checkout 时容易出现大小写路径污染。

建议后续统一一个 canonical 路径。考虑已有 EXP 文档都在 `EXPS/`，建议把实验文档继续放 `EXPS/`；脚本路径是否统一到 `EXPS/` 或 `exps/` 需要结合远端训练机上的实际入口决定。

在统一前，新增脚本时应避免同时改两份大小写路径，避免产生不可预期的跨平台差异。

## 5. Low：EXP 文档编号不一致

最新 SSCA summary 文档存在编号不一致：

- Markdown：`EXPS/analysis/026_ssca_retry_experiment_summary.md`
- TeX：`EXPS/tex/024_ssca_retry_experiment_summary.tex`
- PDF：`EXPS/024_ssca_retry_experiment_summary.pdf`
- TeX 标题：`024：SSCA Retry Full 实验总结`

这会影响后续引用和索引。当前新增本 review 文档使用 `025`，但原 SSCA summary 文档仍然存在 `024/026` 混用。

建议选择一种编号策略：

1. 如果 SSCA Retry Full Summary 是第 024 篇：把 Markdown 从 `026_...` 改为 `024_...`。
2. 如果它应该是第 026 篇：把 TeX/PDF 和标题从 `024` 改为 `026`。
3. 如果本 review 文档要作为第 025 篇插入：保留本文件为 `025`，并在后续整理时统一 024/026 的交叉引用。

## 6. 建议处理顺序

1. 先决定 StepPPO-v3 主 wrapper 是否恢复 detach-true。这个会直接影响后续训练入口。
2. 再决定 SSCA 文档是“补代码”还是“改文档”。当前文档与代码差距较大，不建议直接作为当前仓库可复现说明。
3. 统一 `EXPS/` 和 `exps/` 路径策略，避免后续脚本继续分叉。
4. 最后统一 SSCA 文档编号，并在 README 或文档索引中明确 024/025/026 的顺序。

## 7. 当前未处理事项

本 review 文档只记录问题，没有修改任何运行脚本，也没有重新编译 PDF。

如果下一步要修，推荐最小变更是：

1. 恢复 `EXPS/run_webshop_step_ppo_v3_value_head_4gpu_paper_align_simple.sh` 的 detach-true 默认。
2. 另存 detach-false ablation 脚本。
3. 在 `026_ssca_retry_experiment_summary.md` 开头加一段“当前 packaged copy 缺少完整 SSCA+VIMPO 实现”的说明，或补齐对应代码后再保留现有结论。
