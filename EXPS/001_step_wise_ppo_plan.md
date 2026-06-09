# Step-wise PPO 初步实验规划

## 0. 范围说明

- 目标：在当前 `verl-agent` 的多步 agent 环境中设计一个 step-wise PPO 实验，让 PPO 的 actor-critic 更新从“每个 step 都吃同一个 episode return”推进到“每个 step 使用该 step 的折扣回报/step-level credit”。
- 术语：本规划明确基于当前仓库中的 `GRPO` 与 `GiGPO` 具体实现，不涉及其他未定义算法。
- 当前只写实验规划，不直接改 trainer/actor/critic 逻辑。

## 1. 当前 PPO 实现理解

### 1.1 入口与训练流

- 训练入口是 `verl/trainer/main_ppo.py`，agent 版本强制 `actor_rollout_ref.rollout.n == 1`，分组采样通过 `env.rollout.n` 完成。
- `RayPPOTrainer.fit()` 每轮从 dataloader 取 prompt，经 `TrajectoryCollector.multi_turn_loop()` 生成环境交互轨迹，再计算 log prob、ref log prob、critic value、reward、advantage，最后更新 critic 和 actor。
- 多步交互数据不是拼成一个长上下文，而是在每个环境 step 重新构造当前 observation 的单步输入；`gather_rollout_data()` 会把一个 episode 拆成多条 step sample。

### 1.2 PPO 的 advantage 与 critic

- `AdvantageEstimator.GAE` 会打开 critic：`RayPPOTrainer.__init__()` 中仅 `gae` 分支设置 `self.use_critic = True`。
- PPO advantage 在 `compute_gae_advantage_return()` 中计算：输入是 `token_level_rewards`、`values`、`response_mask`，按 response token 维度反向做 GAE，之后 whiten advantage。
- actor loss 在 `DataParallelPPOActor.update_policy()` 中执行，核心是 `compute_policy_loss()`：使用 `exp(log_prob - old_log_prob)` 的 PPO clipped surrogate，并可叠加 entropy 和 KL loss。
- critic loss 在 `DataParallelPPOCritic.update_critic()` 中执行，核心是 clipped value loss，监督信号来自 `returns`。

### 1.3 当前 PPO 在 agent 环境里的问题

- `EpisodeRewardManager` 会把 `episode_rewards` 写到每个 step sample 的 response 最后一个 token 上。
- 因为 `gather_rollout_data()` 会保留 episode 内所有 active step，当前 PPO 会让同一条 trajectory 的每个 step sample 都拿同一个 full episode reward。
- 这更像“episode-level PPO on each step input”，没有利用环境每一步返回的 `rewards` 和跨 step 的折扣结构，也没有让 critic 明确拟合 step-level return。

## 2. 当前 GRPO 实现理解

### 2.1 分组机制

- `env.rollout.n` 控制同一个初始状态的重复 rollout 数量；`TrajectoryCollector.vanilla_multi_turn_loop()` 为每组样本赋同一个 `uid`，每条 trajectory 另有 `traj_uid`。
- GRPO 的 group 是 episode-level group：同一 `uid` 下的多条 trajectory 相互比较。

### 2.2 Advantage 计算

- `compute_advantage(..., adv_estimator=grpo)` 进入 `compute_grpo_outcome_advantage()`。
- 该函数把 `token_level_rewards.sum(-1)` 作为每条 step sample 的 score，再按 `uid` 做均值/标准差归一，得到相对 advantage。
- 当前代码额外用 `traj_uid` 去重，避免把同一 trajectory 的多个 step 在 episode-level group 中重复计入，具体行为受 `compute_mean_std_cross_steps` 控制。

### 2.3 更新方式

- GRPO 不用 critic，`self.use_critic = False`。
- actor 仍然复用 PPO clipped objective，只是 advantage 换成 group-relative reward。
- WebShop GRPO 脚本设置 `algorithm.adv_estimator=grpo`、`env.rollout.n=8`、`actor_rollout_ref.actor.use_kl_loss=True`。

## 3. 当前 GiGPO 实现理解

### 3.1 核心思想

- GiGPO 是 critic-free，但比 GRPO 多了 step-level credit assignment。
- 它同时计算 episode-level relative advantage 和 step-level relative advantage，然后相加：`episode_adv + step_advantage_w * step_adv`。

### 3.2 Step return 与 step group

- `compute_step_discounted_returns()` 使用 `batch.non_tensor_batch['rewards']` 和 `traj_uid`，按 trajectory 反向计算每个 step 的折扣回报：`G_t = r_t + gamma * G_{t+1}`。
- `build_step_group()` 在同一 episode group `uid` 内按 `anchor_obs` 聚类；默认是完全相同的 anchor observation，也支持相似度聚类。
- `step_norm_reward()` 在 step group 内对 step discounted return 做均值或均值/标准差归一，形成 step-level relative advantage。

### 3.3 与 GRPO/PPO 的差异

- 相比 GRPO，GiGPO 用 `anchor_obs` 把相同/相似状态下的 action 放进同一个 step group，解决 long-horizon credit 粗糙的问题。
- 相比 PPO，GiGPO 没有 critic；它的 step signal 来自组内相对比较，而不是 value baseline。
- WebShop GiGPO 脚本设置 `algorithm.adv_estimator=gigpo`、`algorithm.gamma=0.95`、`env.rollout.n=8`、`algorithm.gigpo.step_advantage_w=1.0`。

## 4. Step-wise PPO 严格定义

### 4.1 Agentic step 定义

- 一个 episode 是环境轨迹：`s_0, a_0, r_0, s_1, a_1, r_1, ...`。
- 一个 step 是一次 agent 决策：agent 接收当前 context/observation，输出一个完整 action，例如 tool call 或文本动作。
- 一个 action 由多个 token 组成，但 RL 语义上先视为一个整体 action。
- Step-wise PPO 的 value、return、TD-error、GAE 都沿 environment step 维度计算，不沿 response token 维度计算。

### 4.2 严格 state-value 版本

- 第一版采用严格 PPO state-value 定义：critic 估计 `V(s_t)`，不让 baseline 依赖已采样 action。
- 当前 critic 底层仍会为每个 token 输出 scalar value；step-wise PPO 只抽取一个边界 value 作为该 step 的 `V_t`。
- 默认边界为 action 生成前的最后一个 context token：`value_position=pre_action_last_context_token`。
- 不建议默认取第一个生成 token 的 value，因为该位置通常已经条件化到第一个 action token；如果取它，更接近 action-conditioned baseline，会作为 ablation 记录。

### 4.3 Action-conditioned 备选想法

- 备选方案：critic 看完整 `context + action`，取最后一个 response token 或加权 action-token pooling，得到 `V(s_t, a_t)` 或近似 action-value。
- 这个方案可能利用完整 action 信息，工程上也容易复用现有 response value 切片。
- 风险是 baseline 依赖 sampled action，严格 policy-gradient/PPO 理论上会引入 bias，因此不作为第一版。
- 后续可以作为 `value_position=last_response_token` 或 `value_pooling=weighted_response_tokens` 的 ablation。

### 4.4 Step 维度 return 与 GAE

- 对每条 trajectory 按 `traj_uid` 和 step 顺序计算 `G_t = r_t + gamma * G_{t+1}`。
- 严格版 TD-error：`delta_t = r_t + gamma * V_{t+1} * (1 - done_t) - V_t`。
- 严格版 step GAE：`A_step_t = delta_t + gamma * lam * (1 - done_t) * A_step_{t+1}`。
- critic return target 为 `R_step_t = A_step_t + V_t`，或在 `lam=1` 时直接使用 discounted return `G_t`。
- Actor 更新时先把 step-level scalar advantage 映射回该 action 的 response tokens，用于复用现有 PPO actor loss；后续可改成 action-level ratio。

### 4.5 与 GRPO/GiGPO 的关系

- GRPO 提供 episode-level relative advantage：同一 `uid` 下比较完整 trajectory 的 episode return。
- GiGPO 在 GRPO episode advantage 之上加入 step-level group advantage。
- 本实验希望保留 GRPO 的 episode advantage 作为主信号，再用 strict step-wise PPO 的 critic advantage 做修正。

## 5. Advantage 混合与实现路线

### 5.1 混合 advantage 定义

- `A_episode`：复用 GRPO 的 episode-level relative advantage，按 `uid` 分组、用 `traj_uid` 去重，得到每条 trajectory 的 episode advantage，再广播到该 trajectory 的所有 step。
- `A_step`：按 environment step 维度由 state-value critic 计算出来的 GAE/TD advantage。
- 初始采用加权平均，而不是直接相加：`A_final = (1 - alpha) * A_episode + alpha * A_step`。
- 也可用 GiGPO 风格权重参数表示：`A_final = (A_episode + w_step * A_step) / (1 + w_step)`，其中 `alpha = w_step / (1 + w_step)`。
- 为避免 scale 不一致，`A_episode` 和 `A_step` 都应在有效 step 上做 batch/group 归一或 whiten 后再混合。

### 5.2 新增算法分支

- 在 `AdvantageEstimator` 中新增 `STEP_PPO = "step_ppo"` 或 `STEP_GRPO_PPO = "step_grpo_ppo"`。
- 在 `RayPPOTrainer.__init__()` 中让该分支启用 critic。
- 在 rollout 后保留 trajectory step 顺序信息，用于按 `traj_uid` 计算 step-level return/GAE。
- 不能直接复用当前 `compute_gae_advantage_return()`，因为它在 response token 维度做 GAE。

### 5.3 Value 抽取方案

- 默认严格方案：抽取 action 生成前边界 value，作为 `V(s_t)`。
- 当前 critic forward 需要扩展一个 step-level value 抽取函数，而不是只返回 response-token values。
- 第一个生成 token value 是否可用：可以作为 ablation，但默认不采用，因为它已经条件化到 action token。
- action-conditioned 方案记录为未来扩展，不进入第一版主实验。

### 5.4 Actor loss 映射

- 第一版为了少改 actor，把 `A_final_t` broadcast 到该 step action 的所有有效 response tokens，复用当前 token-level clipped loss。
- 这仍可能有长度 bias；后续更严格方案应使用 action-level log-ratio：`ratio_t = exp(sum(logp_new - logp_old))` 或长度归一 log-ratio。
- 若继续用 token-level loss，建议优先使用 `seq-mean-token-mean` 或显式 sequence/action-level 聚合做 ablation。

### 5.5 需要新增的配置

```yaml
step_ppo:
  value_position: pre_action_last_context_token
  action_conditioned_ablation: false
  step_gamma: ${algorithm.gamma}
  step_lam: ${algorithm.lam}
  advantage_mix_mode: weighted_average
  step_advantage_w: 1.0
  normalize_episode_advantage: true
  normalize_step_advantage: true
  actor_ratio_mode: token_broadcast
```

说明：

- `value_position` 控制严格 state-value 边界抽取；默认不取 response token。
- `step_advantage_w` 对应 GiGPO 风格的 step 修正强度，`w=1` 等价于 episode/step 各占 50%。
- `actor_ratio_mode=token_broadcast` 表示第一版把 step advantage 广播到 action tokens，后续可切到 action-level ratio。

## 6. 实验矩阵

### 6.1 主任务

- 先用 WebShop，因为当前已有 PPO/GRPO/GiGPO WebShop 脚本和日志经验。
- 复用现有 4GPU H100 wrapper 风格，保持 `train_data_size=16`、`env.rollout.n=8` 时一轮生成 128 条 trajectory，对齐 GRPO/GiGPO 的采样预算。
- 如果 critic 显存压力太大，降级到 2GPU 或把 `critic.ppo_micro_batch_size_per_gpu` 下调。

### 6.2 对照组

- PPO baseline：`algorithm.adv_estimator=gae`，`data.train_batch_size=128`，无 env group。
- GRPO baseline：`algorithm.adv_estimator=grpo`，`data.train_batch_size=16`，`env.rollout.n=8`。
- GiGPO baseline：`algorithm.adv_estimator=gigpo`，`data.train_batch_size=16`，`env.rollout.n=8`，`gamma=0.95`。
- Step-wise PPO：`algorithm.adv_estimator=step_ppo`，`data.train_batch_size=16`，`env.rollout.n=8`，`gamma=0.95`，启用 critic。

### 6.3 Ablation

- `step_gamma`: 0.95 vs 1.0。
- `step_advantage_w`: 0.25 vs 1.0 vs 3.0，控制 step 修正强度。
- `value_position`: `pre_action_last_context_token` vs `first_response_token` vs `last_response_token`。
- `actor_ratio_mode`: token broadcast vs action-level length-normalized ratio。
- sampling：`env.rollout.n=1` vs `env.rollout.n=8`，区分 PPO 是否需要 group env。
- KL：`use_kl_loss=True, kl_loss_coef=0.01` vs `0.005`。

## 7. 指标与日志

### 7.1 训练核心指标

- validation success rate / average reward。
- episode length、valid action ratio、tool_callings。
- `actor/pg_loss`、`actor/pg_clipfrac`、`actor/ppo_kl`、`actor/grad_norm`。
- `critic/vf_loss`、`critic/vpred_mean`、`critic/vf_clipfrac`、`critic/grad_norm`。

### 7.2 新增建议指标

- `step_ppo/episode_adv_mean`、`step_ppo/step_adv_mean`、`step_ppo/final_adv_mean`。
- `step_ppo/step_return_mean`、`step_ppo/step_return_std`。
- `step_ppo/state_value_mean`、`step_ppo/td_error_mean`。
- `step_ppo/return_minus_value_mean/std`。
- 按 step index 分桶的 return/value/advantage 均值，检查 early-step credit 是否稳定。

## 8. 预期风险

- critic target 分布改变后，`critic.optim.lr=1e-5` 可能偏大；如果 `vf_loss` 爆炸，先降到 `5e-6` 或 `1e-6`。
- `A_episode` 与 `A_step` 的尺度不同会导致一个信号吞掉另一个，混合前必须做归一化并记录各自分布。
- 取生成前边界 value 需要扩展 critic value 抽取；如果误取 response token value，就会退化为 action-conditioned 版本。
- 第一版 token broadcast 复用 actor loss 仍可能有长度 bias，需要尽快对比 action-level ratio。
- 当前 `balance_batch` 会 reorder batch，必须确保 step order、`step_rewards`、`traj_uid`、`uid` 等字段同步维护。

## 9. 里程碑

1. 静态实现：新增 step-level state value 抽取、step 维度 GAE、GRPO episode advantage 混合。
2. Smoke test：WebShop 小步数、小 batch，确认 rollout、step order、critic update、actor update 都能闭环。
3. 单 seed 对齐实验：与现有 GiGPO 4GPU WebShop 配置对齐，跑 50 epoch 看曲线是否合理。
4. 三 seed 复现实验：复用 `exps/run_webshop_4gpu_paper_align_simple_3seeds.sh` 的 seed 结构。
5. Ablation：比较 `step_advantage_w`、`value_position`、actor ratio、critic lr、`env.rollout.n`。

## 10. 初始命令草案

下面是未来实现 `step_ppo` 后的命令形态，不是当前可直接运行命令：

```bash
bash EXPS/run_webshop_step_ppo_4gpu.sh \
  algorithm.adv_estimator=step_ppo \
  algorithm.gamma=0.95 \
  algorithm.step_ppo.value_position=pre_action_last_context_token \
  algorithm.step_ppo.step_advantage_w=1.0 \
  algorithm.step_ppo.advantage_mix_mode=weighted_average \
  algorithm.step_ppo.actor_ratio_mode=token_broadcast \
  data.train_batch_size=16 \
  env.rollout.n=8 \
  critic.optim.lr=5e-6 \
  trainer.experiment_name=step_ppo_qwen2.5_1.5b_4gpu_webshop_e250
```

## 11. 当前代码参考位置

- PPO/GRPO/GiGPO advantage 分发：`verl/trainer/ppo/ray_trainer.py`
- PPO core loss 与 GAE：`verl/trainer/ppo/core_algos.py`
- Actor PPO clipped loss 调用：`verl/workers/actor/dp_actor.py`
- Critic value loss 调用：`verl/workers/critic/dp_critic.py`
- Step return 与 GiGPO step group：`gigpo/core_gigpo.py`
- 多步环境 rollout 与 trajectory 展平：`agent_system/multi_turn_rollout/rollout_loop.py`
- Episode reward 写入 token：`agent_system/reward_manager/episode.py`
- 当前 WebShop 脚本：`examples/ppo_trainer/run_webshop.sh`、`examples/grpo_trainer/run_webshop.sh`、`examples/gigpo_trainer/run_webshop.sh`
