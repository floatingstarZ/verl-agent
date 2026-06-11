# 021 Soft-GiGPO 前期 Trace 实验摘要

## 运行设置

- ckpt：`global_step_250` 的 StepPPO-v3 value-head 版本；用临时软链接去掉 `data.pt`，避免恢复到 dataloader 末尾。
- 数据：WebShop train 子集，`train_batch_size=16`，`env.rollout.n=8`，理论上 16 个任务 × 8 条轨迹。
- 更新：`trainer.critic_warmup=999999`，所以这次只采样、算logprob/ref/adv/value diagnostic，不更新 actor。
- 输出：`value_diagnostics/step_000251.jsonl` 与 `rollout_generations/251.jsonl`。

## 样本概况

- step rows：896；任务数：16；轨迹数：128；平均每轨迹 7.00 step。
- trajectory 平均 episode reward：6.719；reward≥10 比例：67.2%。
- row-level valid action ratio：94.2%；response clip ratio：46.3%。
- value head：pred/target corr 对 `value_target_gae_return` Pearson=0.881，RMSE=2.019，MAE=1.007。

## Group 粒度对比

- `task + exact current_observation`：160 组，cluster mean size=5.60，singleton=57.5%，max=71。这是当前 hard anchor 的近似代理。
- `task + step_idx`：123 组，cluster mean size=7.28，max=11。这个粒度覆盖更大，但会混入不同页面状态。
- `task + stage`：48 组，cluster mean size=18.67，max=71。这个粒度过粗，适合做 prefilter，不适合直接做 advantage group。
- 在 `task + step_idx` 内，current observation pair similarity：mean=0.821，p90=1.000，≥0.95=72.6%，≥0.90=73.6%，exact=72.6%。

## 初步判断

- `exact current_observation`/`anchor_obs` 仍是最干净的 hard group 信号，但会产生一批 singleton；021 的 soft matching 主要应补这些 singleton/小组。
- `step_idx` 是强先验但不能单独使用；同一步会有 search result、product detail、terminal 等不同 state。
- `stage` 可作为 action-mask/state-family prefilter；直接按 stage 聚合太粗，会把价值差异大的状态混在一起。
- `value_pred_before_update` 与 GAE target 当前相关性高，适合做 confidence/ranking diagnostic；但不建议单独当 matching key，避免 value-only matching。
- `response_length==512` 的比例较高，应作为噪声/低置信过滤信号，而不是状态相似信号。

## 下一步建议补采字段

- 在 `value_diagnostics` 里补 `uid` 与原始 `anchor_obs`，这样可精确复现 GiGPO hard group，而不是从 prompt 反解。
- 补 actor 侧 pre-action hidden/state embedding、action distribution/logprob entropy、可行动作集合/action mask。
- 补 leave-one-out top-k soft neighbor 表：同 `uid` 内先按 `stage/action-mask` 过滤，再比较 frozen actor distribution 或 hidden similarity。
- 暂时只做离线审计：比较 soft neighbor 的 target/return 方差、valid ratio、episode reward 一致性，不接入训练。

## 产物

- `step_trace_rows.csv`
- `trajectory_summary.csv`
- `task_summary.csv`
- `summary_by_step_idx.csv`
- `summary_by_stage.csv`
- `summary_by_action_cmd.csv`
- `group_summary_by_task_step.csv`
- `group_summary_by_task_exact_obs.csv`
- `group_summary_by_task_stage.csv`
- `task_step_obs_pair_sample.csv`
- `candidate_signal_summary.json`
