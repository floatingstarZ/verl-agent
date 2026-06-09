# 013：当前项目进展与 Value 收敛分析

日期：2026-06-09

本文记录当前 `verl-agent` 项目的实验进展，并重点分析 StepPPO / GiGPO value-aux 方向中 value head 的收敛情况。分析基于当前仓库内的脚本、文档和 console log，不引入外部结果。

## 1. 当前项目进展

当前项目主线已经从单纯复现 GiGPO baseline，推进到 StepPPO 及 shared value head 的稳定性诊断阶段。

### 1.1 已完成或已有有效日志的实验

| 实验 | 日志 | 状态 | 关键结论 |
|---|---|---:|---|
| GiGPO seed 2026 | `logs/webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_20260530_114442.log` | 完整 250 step | 强基线，最终 task/success/text = 0.903/0.812/7.963 |
| GiGPO seed 2077 | `logs/webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_20260531_004613.log` | 完整 250 step | 最终 task/success/text = 0.883/0.754/7.341 |
| GiGPO seed 2501 | `logs/webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_20260531_123134.log` | 完整 250 step | 最终 task/success/text = 0.869/0.691/6.952 |
| StepPPO-v1 step_norm seed 2026 | `logs/webshop_step_ppo_qwen25_15b_4gpu_paper_align_simple_20260604_102927.log` | 跑到 step 187 / val 185 | 明显弱于 GiGPO，value loss 较大，v1 不稳定性已记录于 `EXPS/009_step_ppo_instability_analysis.md` |
| StepPPO-v2 seed 2026 | `logs/webshop_step_ppo_qwen25_15b_4gpu_paper_align_simple_20260605_031621.log` | 完整 250 step | 相比 v1 后期 value 有所缓和，但 valid action ratio 低，validation 仍未追上 GiGPO |
| GiGPO-v1 value-aux detach=false seed 2026 | `logs/webshop_gigpo_v1_value_aux_detach_false_qwen25_15b_4gpu_paper_align_simple_20260608_115519.log` | 跑到 step 204 / val 200 | policy advantage 保持 GiGPO，但 auxiliary value loss 仍带来明显退化和局部崩盘 |

三个 GiGPO baseline 已形成较稳定参照：step 250 的平均 task score 约 0.885，平均 success rate 约 0.752，平均 text score 约 7.419。StepPPO-v1/v2 和 GiGPO-value-aux 目前都没有达到该水平。

### 1.2 当前算法与脚本状态

- StepPPO-v1 使用 `normalize_episode_advantage=True`、`normalize_step_advantage=True`、`step_advantage_w=1.0`、`value_loss_coef=0.1`，且 `detach_value_backbone=False`。
- StepPPO-v2 实际脚本为 `exps/run_webshop_step_ppo_v2_4gpu_paper_align_simple.sh`，核心配置是 `step_advantage_w=0.5`、`episode_mode=mean_norm`、`final_advantage_mode=direct`、`normalize_episode_advantage=False`、`normalize_step_advantage=True`、`normalize_final_advantage=False`、`value_loss_coef=0.03`、`detach_value_backbone=False`、`ppo_micro_batch_size_per_gpu=8`。
- GiGPO-v1 value-aux 由 `verl/trainer/ppo/gigpo_v1_value_aux_trainer.py` 接入：policy advantage 仍是原 GiGPO，额外把 StepPPO-style `step_returns` 塞给 shared value head 做 auxiliary regression。
- `exps/run_webshop_gigpo_value_aux_ablation_grpo_seed2026.sh` 已规划 `detach_false -> detach_true -> GRPO` 的 batch ablation，但当前有效 full/long 日志主要是 `detach_false`；没有看到正式完整 4GPU GRPO baseline 日志。
- `EXPS/tex/011_step_ppo_v2_improvements.tex` 中早期叙述曾提到 detach value backbone；但以当前脚本、日志和 `EXPS/tex/012_step_ppo_v2_paper_method.tex` 为准，当前 v2 是 `detach_value_backbone=False`。

## 2. Validation 进展概览

| 实验 | val last step | best task | last task/success/text | step 200 task/success/text |
|---|---:|---:|---:|---:|
| GiGPO seed 2026 | 250 | 0.905 | 0.903 / 0.812 / 7.963 | 0.858 / 0.719 / 7.286 |
| GiGPO seed 2077 | 250 | 0.912 | 0.883 / 0.754 / 7.341 | 0.841 / 0.719 / 6.314 |
| GiGPO seed 2501 | 250 | 0.897 | 0.869 / 0.691 / 6.952 | 0.812 / 0.668 / 6.362 |
| StepPPO-v1 seed 2026 | 185 | 0.766 | 0.703 / 0.492 / 3.866 | NA |
| StepPPO-v2 seed 2026 | 250 | 0.806 | 0.662 / 0.547 / 3.485 | 0.562 / 0.469 / 3.000 |
| GiGPO-v1 value-aux detach=false seed 2026 | 200 | 0.802 | 0.545 / 0.398 / 2.950 | 0.545 / 0.398 / 2.950 |

结论：当前 value-head 相关实验还没有证明收益。StepPPO-v2 的 best task score 到过 0.806，但最终回落到 0.662；GiGPO-value-aux best task score 到过 0.802，但在 step 180--185 出现严重 validation collapse，step 200 仍明显低于原 GiGPO。

## 3. Value 收敛情况

### 3.1 总体判断

当前 value head 还不能认为已经稳定收敛。原因有三点：

1. `actor/value_loss` 没有形成单调下降或低方差平台；多个实验中后期仍在 3--5 左右波动。
2. `actor/value_pred_mean` 和 `actor/value_return_mean` 的均值差距已经不大，但这只说明均值校准有所改善，不代表 per-sample value regression 收敛；value loss 仍然较高，说明 target 方差或 outlier 仍很强。
3. value loss 与 policy/format 指标存在同步异常：StepPPO-v2 的 value loss 后期下降，但 valid action ratio 长期偏低；GiGPO-value-aux 保持较高 valid action ratio，却在 step 178--185 出现 response clipping 和 validation 崩盘。

### 3.2 分段 value 指标

| 实验 | step 段 | value_loss | abs(pred-return mean) | value_clipfrac | valid_action_ratio |
|---|---:|---:|---:|---:|---:|
| StepPPO-v1 | 1--50 | 1.816 | 0.226 | 0.054 | 0.671 |
| StepPPO-v1 | 51--100 | 3.568 | 0.291 | 0.104 | 0.780 |
| StepPPO-v1 | 101--150 | 4.948 | 0.376 | 0.103 | 0.872 |
| StepPPO-v1 | 151--187 | 4.276 | 0.290 | 0.110 | 0.918 |
| StepPPO-v2 | 1--50 | 2.988 | 0.416 | 0.033 | 0.313 |
| StepPPO-v2 | 51--100 | 3.918 | 0.397 | 0.117 | 0.156 |
| StepPPO-v2 | 101--150 | 3.773 | 0.315 | 0.100 | 0.337 |
| StepPPO-v2 | 151--200 | 4.254 | 0.375 | 0.138 | 0.277 |
| StepPPO-v2 | 201--250 | 2.797 | 0.283 | 0.086 | 0.328 |
| GiGPO-value-aux detach=false | 1--50 | 5.098 | 0.484 | 0.078 | 0.917 |
| GiGPO-value-aux detach=false | 51--100 | 6.246 | 0.465 | 0.161 | 0.921 |
| GiGPO-value-aux detach=false | 101--150 | 4.831 | 0.444 | 0.129 | 0.929 |
| GiGPO-value-aux detach=false | 151--200 | 4.573 | 0.399 | 0.126 | 0.877 |
| GiGPO-value-aux detach=false | 201--204 | 4.426 | 0.507 | 0.106 | 0.883 |

### 3.3 Last-10 value 状态

| 实验 | last-10 step | value_loss | abs(pred-return mean) | value_clipfrac | value loss 实际项 | pg_loss | valid_action_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| StepPPO-v1 | 178--187 | 3.552 | 0.243 | 0.098 | 0.355 | -0.005 | 0.935 |
| StepPPO-v2 | 241--250 | 3.011 | 0.380 | 0.093 | 0.090 | -0.085 | 0.281 |
| GiGPO-value-aux detach=false | 195--204 | 4.207 | 0.336 | 0.089 | 0.126 | -0.254 | 0.908 |

这里的 `value loss 实际项` 是 `actor/value_loss * actor/value_loss_coef` 的近似均值。v1 因为 `value_loss_coef=0.1`，后期 value 项约 0.355，数值上远大于 policy loss 均值；v2 和 value-aux 降到 `0.03` 后，value 项降到 0.09--0.13，但仍可能通过 shared backbone 影响 policy representation。

## 4. 关键诊断

### 4.1 StepPPO-v1：value loss 对 backbone 干扰偏强

v1 的 value loss 从 early stage 的 1.816 上升到 101--150 的 4.948，后期仍在 4 左右。由于 `value_loss_coef=0.1` 且 `detach_value_backbone=False`，value regression 会直接更新 actor backbone。结合 v1 validation 明显弱于 GiGPO，可以认为 v1 的 value 学习既没有稳定收敛，也可能干扰了 policy 的 action-format 能力。

### 4.2 StepPPO-v2：value loss 后期有所缓和，但 policy 行为仍异常

v2 把 value 系数降到 0.03，并对齐 GiGPO 的 episode advantage 形式。value_loss 在 201--250 段降到 2.797，比 151--200 段的 4.254 好；但 valid action ratio 长期只有 0.15--0.33，远低于 GiGPO 和 value-aux 的 0.9 左右。因此 v2 的主要瓶颈不只是 value 是否拟合，而是 final advantage / shared-backbone update 后造成了 policy action format 退化。

### 4.3 GiGPO-value-aux：policy advantage 不变仍会退化

GiGPO-value-aux 的设计是保持 GiGPO policy advantage 不变，仅新增 value auxiliary target。按理说它应当接近 GiGPO baseline；但实际 step 200 的 task/success/text 只有 0.545/0.398/2.950，而原 GiGPO seed 2026 step 200 是 0.858/0.719/7.286。

这说明退化可以仅由 shared value head 的辅助损失引入，不一定来自 StepPPO 的 mixed advantage。尤其在 step 178--185 附近，GiGPO-value-aux 出现明显异常：

| step | value_loss | value_clipfrac | valid_action_ratio | response_clip_ratio | validation 附近表现 |
|---:|---:|---:|---:|---:|---|
| 178 | 8.230 | 0.253 | 0.700 | 0.307 | 即将崩盘 |
| 179 | 11.031 | 0.345 | 0.653 | 0.352 | 即将崩盘 |
| 180 | 6.752 | 0.125 | 0.746 | 0.252 | val task/success/text = 0.153/0.074/0.560 |
| 185 | 6.169 | 0.202 | 0.778 | 0.216 | val task/success/text = 0.155/0.102/0.798 |

该现象提示 value target 的高方差或 clipped value update 的大幅波动，会通过 shared backbone 影响生成分布；即使 valid action ratio 很快恢复，validation 也可能出现短时严重退化。

## 5. 当前结论

1. GiGPO baseline 已经稳定，仍是当前最强对照。
2. StepPPO-v1 的 value head 没有收敛到可靠 critic，且 value loss coefficient 偏大，干扰风险最高。
3. StepPPO-v2 降低了 value loss 的有效权重，后期 value_loss 有改善，但 action-format 学习明显退化，最终效果仍弱于 GiGPO。
4. GiGPO-value-aux detach=false 证明：只加 auxiliary value loss，也足以显著破坏 GiGPO；因此当前首要问题是 shared value head / target / backbone gradient 的稳定性，而不是只调 StepPPO advantage mix。
5. 仅看 `value_pred_mean` 与 `value_return_mean` 的均值接近会过于乐观；需要看 RMSE、explained variance、target 分布和 per-step / per-validity 分层误差。

## 6. 建议下一步

### 6.1 必跑 ablation

1. 补跑 `GiGPO-v1 value-aux detach=true` 的 full run。若 detach=true 明显接近原 GiGPO，则可以确认主要问题来自 value loss 回传 backbone。
2. 补齐正式 `GRPO seed=2026` baseline，以便区分 non-step baseline、GiGPO step-aware baseline、StepPPO/value-aux 的差异。
3. 对 StepPPO-v2 增加 `detach_value_backbone=True` 或更低 `value_loss_coef=0.01` 的 ablation；这不一定作为最终方法，但可以定位干扰来源。

### 6.2 必加 value diagnostics

建议在训练日志中新增以下 value 诊断：

- `actor/value_rmse`、`actor/value_mae`、`actor/value_explained_variance`。
- `actor/value_pred_std`、`actor/value_return_std`、`actor/value_return_p05/p50/p95`。
- 按 `step_idx` 分桶的 value loss / return mean / return std。
- 按 `is_action_valid` 分层的 value loss，确认 invalid action penalty 是否造成 target outlier。
- value loss 与 `response_length/clip_ratio`、`episode/valid_action_ratio` 的 rolling correlation。

### 6.3 可能的稳定化方向

- 对 value target 做 batch 或 group-level normalization，只把 normalized return 用于 value loss；policy advantage 是否归一化另行控制。
- 降低或 warmup `value_loss_coef`，例如从 0 到 0.01 / 0.03 线性 warmup，避免早期 value target 噪声破坏 backbone。
- 比较 `detach_value_backbone=True` 与 `False`，确认最终方法是否真的需要 value loss 更新 LM backbone。
- 对 GAE target 做 clipping 或 reward normalization，尤其关注 WebShop sparse/invalid penalty 混合后的 return outlier。
- 在可视化中叠加 value_loss、response_clip_ratio、valid_action_ratio 和 validation task score，重点复盘 GiGPO-value-aux step 170--190 的崩盘窗口。

## 7. 本文使用的解析口径

- 训练行：包含 `actor/value_loss` 的 `step:<global_step> - ...` console metric 行。
- validation 行：包含 `val/success_rate`、`val/webshop_task_score (not success_rate)` 或 `val/text/test_score` 的 metric 行。
- `abs(pred-return mean)`：`abs(actor/value_pred_mean - actor/value_return_mean)` 的分段均值，只反映均值校准，不等价于 per-sample RMSE。
- `value loss 实际项`：`actor/value_loss * actor/value_loss_coef`，用于粗略观察 value loss 在总 actor loss 中的数值量级。
