# 005 Step-wise PPO 代码实现说明

## 结论

当前 StepPPO 是 shared-backbone actor-critic 变体。它不是纯 GRPO，也不是纯 GiGPO。policy loss 仍然是 PPO clipped surrogate，但 advantage 由 GRPO-style episode signal 和 learned step-wise GAE signal 混合得到。

critic 不是独立模型，而是 actor backbone 上额外挂的 value head：

```text
A_final(step) = (A_episode + w * A_step_GAE) / (1 + w)
```

## Baseline 建议

- non-step baseline：GRPO。
- step-aware critic-free baseline：GiGPO。
- learned step-aware method：StepPPO。

因为 GiGPO 和 StepPPO 都使用 step 信息，所以 GiGPO 是判断 learned critic 是否有价值的更强 baseline。GRPO 用于衡量 step information 本身的贡献。

## 代码入口

主要文件包括：

- `verl/trainer/main_step_ppo.py`：StepPPO 训练入口。
- `verl/trainer/ppo/step_ppo_trainer.py`：StepPPO trainer 与 estimator 检查。
- `verl/trainer/ppo/step_ppo_algos.py`：step GAE、episode advantage、final mixing。
- `verl/workers/actor/value_head.py`：TRL-style value head 挂载。
- `verl/workers/actor/step_ppo_actor.py`：old value collection、actor update、value loss。
- `verl/workers/step_ppo_fsdp_workers.py`：FSDP worker 集成。
- `verl/workers/sharding_manager/step_ppo_filters.py`：rollout weight sync 前过滤 value-head 参数。

## Runtime Path

一次 StepPPO iteration 的顺序是：

1. 采集 multi-turn trajectories。
2. 将 episode flatten 成 environment-step samples。
3. 重新计算 old token log probabilities。
4. 用 shared actor value head 计算 old scalar state values。
5. 计算 reference log probabilities。
6. 计算 immediate step rewards 和 episode scores。
7. 计算 episode-level group advantage 与 step-level GAE advantage。
8. 将 scalar final advantage 广播到 response token。
9. 联合更新 actor 和 value head。

## Rollout 字段

实现依赖三个关键字段：

- `uid`：初始任务 group，用于 GRPO-style episode normalization。
- `traj_uid`：trajectory id，用于恢复 episode 内 step 序列。
- `step_idx`：环境 step index，用于按时间顺序计算 GAE。

如果缺少 `step_idx`，batch rearrangement 可能静默破坏 temporal ordering。

## Value Position

默认 value position 是 `pre_action_last_context_token`，也就是 action 生成前最后一个 context token。这个位置对应严格的 `V(s_t)`。`first_response_token` 和 `last_response_token` 更接近 action-conditioned value，只适合作为 ablation。

## Advantage 与 Return

`advantages` tensor 保持 token shape，但语义上是 step scalar：

```tex
A_{t,\ell}=A_t^{final} \cdot m_{t,\ell}
```

`step_returns` 保持 scalar，用于监督 value head。

## Actor Update

当 batch 中存在 `step_returns` 时，`StepWisePPOActor.update_policy` 会同时计算 current log probabilities 和 current values。value loss 使用 old values 与 scalar returns 做 clipped regression。policy loss 和 value loss 共享同一个 optimizer step。

## 语义检查

当前实现满足 StepPPO 的核心语义：

- value、return、TD error 和 GAE 都在 step 维度计算。
- 每个 action 只取一个 scalar value。
- scalar advantage 广播到 token 只是为了复用已有 token-level PPO loss。

## 当前训练快照

已有 StepPPO-v1 WebShop 日志到 step 187。相对 GiGPO，它表现为 validation reward 更低、valid action ratio 更低、response clipping 更高、gradient norm 更大。详细分析见文档 `009` 和 `010`。
