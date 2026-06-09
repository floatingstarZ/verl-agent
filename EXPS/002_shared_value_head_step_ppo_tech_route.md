# Shared-Backbone Step-wise PPO 技术路线

## 0. 目标

实现一个严格 step-wise PPO 变体：不再维护独立 critic 模型，而是在 actor CausalLM 上挂一个额外 value MLP head，用同一个 actor backbone 同时做 policy 和 step-level state value 回归。

核心目标：

- 不创建 `Role.Critic` worker，不加载第二套 critic backbone。
- actor 模型保留 LM head，同时新增 `value_head`。
- critic value 定义为严格 `V(s_t)`，默认取 action 生成前的最后一个 context token hidden state。
- return、TD-error、GAE 都沿 environment step 维度计算。
- final advantage 是 GRPO episode advantage 的 step-wise critic 修正，使用加权平均。

## 1. TRL ValueHead 参考点

TRL 的 `AutoModelForCausalLMWithValueHead` 思路可以抽象为：

- 用一个 wrapper 包住 causal LM。
- forward 时强制 `output_hidden_states=True`。
- 从最后一层 hidden states 经过 value head 得到每个 token 的 scalar value。
- value head 本质是 dropout 后接 `hidden_size -> 1` 的 regression head。

本项目不直接照搬 wrapper 形式，原因是当前 FSDP、rollout weight sync、checkpoint 都默认 actor 是 HF CausalLM。更稳的做法是只借鉴 TRL 的 value-head 思路：在 actor module 上直接挂 `value_head`，不改变 actor module 的主 forward 接口。

## 2. 确定的算法定义

### 2.1 Trajectory 维度

一个 episode 是：

```text
s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T, a_T, r_T
```

其中一个 environment step 对应一次 agent action。一个 action 可以由多个 response token 组成，但 RL 计算把它视为一个整体 action。

### 2.2 State value 定义

默认严格定义：

```text
V_t = V(s_t)
```

实现上：

```text
V_t = value_head(hidden_state[prompt_boundary_t])
```

其中 `prompt_boundary_t` 是当前 step action 生成前的最后一个 context token 位置，对应当前张量里的：

```text
boundary_idx = sequence_length - response_length - 1
```

不默认取第一个生成 token，也不默认取最后一个 response token，因为这些位置已经条件化到 sampled action，属于 action-conditioned value。该想法保留为 ablation。

### 2.3 Step GAE

对每条 `traj_uid`，按 `step_idx` 排序后计算：

```text
delta_t = r_t + gamma * V_{t+1} * (1 - done_t) - V_t
A_step_t = delta_t + gamma * lambda * (1 - done_t) * A_step_{t+1}
R_step_t = A_step_t + V_t
```

`done_t = 1` 当该 trajectory 没有下一步。

### 2.4 Episode advantage 修正

保留 GRPO/GiGPO 的 episode-level group signal：

```text
A_episode = normalize_by_uid(episode_return)
```

再用 step-level PPO advantage 修正：

```text
A_final = (A_episode + w_step * A_step) / (1 + w_step)
```

其中：

- `A_episode` 先按 `uid` 分组、按 `traj_uid` 去重，得到每条 trajectory 的 episode relative advantage，再广播到该 trajectory 的所有 step。
- `A_step` 由 shared value head 的 step-level GAE 得到。
- 混合前对 `A_episode` 与 `A_step` 分别做有效 step 上的 normalize/whiten，避免尺度不一致。

第一版 actor loss 把 `A_final_t` broadcast 到 action 的有效 response tokens，复用现有 PPO clipped loss。后续再实现 action-level ratio。

## 3. 模型结构

### 3.1 新增 head

新增模块建议：

```text
verl/workers/actor/value_head.py
```

推荐第一版使用 2 层 MLP：

```python
class StepValueHead(nn.Module):
    def __init__(self, hidden_size, intermediate_size=None, dropout=0.0):
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc1 = nn.Linear(hidden_size, intermediate_size or hidden_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(intermediate_size or hidden_size, 1)

    def forward(self, hidden_states):
        x = self.dropout(hidden_states)
        x = x.to(self.fc1.weight.dtype)
        return self.fc2(self.act(self.fc1(x))).squeeze(-1)
```

同时保留 `head_type=linear` 作为 TRL-compatible ablation：

```text
hidden_size -> 1
```

### 3.2 挂载位置

在 `verl/workers/fsdp_workers.py` 的 `_build_model_optimizer()` 中，actor module 从 `AutoModelForCausalLM.from_pretrained()` 创建后、FSDP 包装前执行：

```python
if role == "actor" and config.actor.value_head.enable:
    actor_module.value_head = StepValueHead(...)
```

这样：

- optimizer 自动包含 `value_head` 参数。
- checkpoint 自动保存 `value_head`。
- actor 主 forward 仍然是 causal LM forward。
- rollout engine 不需要知道 `value_head`。

### 3.3 Rollout 权重同步过滤

因为 vLLM/sglang 只接受 LM 权重，必须在 sharding manager 同步 actor state dict 到 rollout engine 前过滤 value head 参数。

需要修改：

```text
verl/workers/sharding_manager/fsdp_vllm.py
verl/workers/sharding_manager/fsdp_sglang.py
```

过滤规则：

```python
params = {k: v for k, v in params.items() if not k.startswith("value_head.") and ".value_head." not in k}
```

LoRA 路径也要确认不会把 `value_head` 放进 LoRA params。第一版目标先支持 non-LoRA FSDP；LoRA 作为后续兼容项。

## 4. 数据与 rollout 字段

### 4.1 新增 step index

当前 rollout 已有：

- `uid`: episode group id。
- `traj_uid`: trajectory id。
- `rewards`: environment immediate reward。
- `active_masks`: 当前 step 是否 active。
- `anchor_obs`: GiGPO step grouping 用。

需要在 `agent_system/multi_turn_rollout/rollout_loop.py` 中新增：

```python
batch.non_tensor_batch["step_idx"] = np.full(batch_size, _step, dtype=np.int32)
```

原因：后续 batch 可能被 `adjust_batch()` 或 `balance_batch()` reorder，不能依赖 DataProto 当前顺序恢复 trajectory step 顺序。

### 4.2 Immediate reward 张量

在 trainer 中构造：

```python
batch.batch["step_immediate_rewards"] = torch.tensor(batch.non_tensor_batch["rewards"], ...)
```

invalid action penalty 应该施加在 immediate step reward 上，再用于 step GAE。不要先算 discounted return 再扣 penalty，否则 penalty 不会向前传播到 earlier steps。

第一版建议：

- episode advantage 继续使用 `token_level_rewards` 中的 episode reward。
- step advantage 使用 `step_immediate_rewards`。
- `algorithm.use_kl_in_reward=False`，KL 仍放在 actor loss 中，避免把 token-level KL 聚合到 step reward 的额外复杂性。

## 5. Forward 与 value 抽取

### 5.1 Actor forward 扩展

在 `verl/workers/actor/dp_actor.py` 中扩展 `_forward_micro_batch()`：

```python
entropy, log_probs, state_values = self._forward_micro_batch(
    micro_batch=data,
    temperature=temperature,
    calculate_entropy=calculate_entropy,
    calculate_values=self.config.value_head.enable,
)
```

当 `calculate_values=True` 时：

- forward 参数加 `output_hidden_states=True`。
- 取 `output.hidden_states[-1]`。
- 用 `value_head` 计算 scalar values。
- 默认取 `boundary_idx = -response_length - 1`。

remove-padding 路径需要先对 unpadded hidden states 跑 value head，再 pad 回 `[batch, seqlen]` 后取 boundary value，和当前 logits pad 回来的处理保持一致。

### 5.2 old state values

`ActorRolloutRefWorker.compute_log_prob()` 当前返回：

```python
old_log_probs
entropys
```

step-wise PPO 需要扩展为：

```python
old_log_probs
entropys
state_values
```

这里的 `state_values` 是 rollout 后、actor update 前的 old value prediction，用于：

- step GAE。
- clipped value loss 的 old value anchor。

## 6. Trainer 逻辑

### 6.1 AdvantageEstimator

在 `verl/trainer/ppo/ray_trainer.py` 中新增：

```python
STEP_PPO = "step_ppo"
```

初始化逻辑：

```python
if adv_estimator == STEP_PPO:
    self.use_critic = False
    self.use_shared_value_head = True
```

不要创建或调用 external critic worker。

### 6.2 训练 loop 顺序

step-wise PPO 每轮训练顺序：

1. `TrajectoryCollector.multi_turn_loop()` 采集 step-level samples。
2. actor worker 计算 `old_log_probs`、`entropys`、`state_values`。
3. ref policy 计算 `ref_log_prob`，用于 KL loss。
4. reward manager 写 episode reward 到 `token_level_scores`。
5. 构造 `step_immediate_rewards` 并施加 invalid action penalty。
6. 计算 `A_episode`。
7. 按 `traj_uid + step_idx` 计算 `A_step` 与 `R_step`。
8. 混合得到 `A_final`。
9. 将 `A_final` broadcast 到 response tokens，得到 actor 使用的 `advantages`。
10. actor update 内同时计算 policy loss 与 value loss。

### 6.3 compute_advantage 新函数

建议在 `verl/trainer/ppo/core_algos.py` 新增：

```python
def compute_grpo_episode_advantage_scalar(...):
    """Return one scalar episode advantage for each step sample."""


def compute_step_gae_advantage_return(...):
    """Compute step-level GAE by traj_uid and step_idx."""


def mix_episode_step_advantage(...):
    """Normalize and weighted-average episode and step advantages."""
```

输出字段：

```python
batch.batch["step_advantages"]      # [bs]
batch.batch["step_returns"]         # [bs]
batch.batch["episode_advantages"]   # [bs]
batch.batch["final_step_advantages"]# [bs]
batch.batch["advantages"]           # [bs, response_length], broadcast for actor loss
batch.batch["returns"]              # optional [bs] for value loss; use step_returns
batch.batch["state_values"]         # [bs], old value prediction
```

## 7. Actor update 与 value loss

### 7.1 update_policy 输入

在 `DataParallelPPOActor.update_policy()` 中为 step-wise PPO 额外选择：

```python
state_values
step_returns
final_step_advantages
```

第一版仍然保留 token-broadcast `advantages` 给 policy loss。

### 7.2 current value prediction

actor update forward 重新计算：

```python
log_prob
entropy
current_state_values
```

### 7.3 scalar clipped value loss

新增 scalar value loss：

```python
vpred_clipped = old_values + clamp(current_values - old_values, -cliprange_value, cliprange_value)
vf_loss_1 = (current_values - step_returns) ** 2
vf_loss_2 = (vpred_clipped - step_returns) ** 2
value_loss = mean(max(vf_loss_1, vf_loss_2))
```

总 loss：

```text
loss = policy_loss
     + value_loss_coef * value_loss
     + kl_loss_coef * kl_loss
     - entropy_coeff * entropy_loss
```

第一版确定让 value loss 回传 shared backbone 和 value head，使用较小的：

```text
value_loss_coef = 0.1
```

如果训练不稳定，再加 ablation：`detach_value_backbone=true`，只训练 value head。

## 8. 配置项

在 `verl/trainer/config/ppo_trainer.yaml` 中建议新增：

```yaml
algorithm:
  adv_estimator: step_ppo
  step_ppo:
    step_gamma: ${algorithm.gamma}
    step_lam: ${algorithm.lam}
    advantage_mix_mode: weighted_average
    step_advantage_w: 1.0
    normalize_episode_advantage: true
    normalize_step_advantage: true
    actor_ratio_mode: token_broadcast

actor_rollout_ref:
  actor:
    loss_agg_mode: seq-mean-token-mean
    value_head:
      enable: false
      head_type: mlp
      intermediate_size: null
      dropout: 0.0
      value_position: pre_action_last_context_token
      value_loss_coef: 0.1
      cliprange_value: 0.5
      detach_value_backbone: false
```

说明：

- `value_head.enable=true` 只应在 `algorithm.adv_estimator=step_ppo` 时打开。
- `loss_agg_mode=seq-mean-token-mean` 是第一版推荐，减少长 action token 数造成的权重偏差。
- `actor_ratio_mode=token_broadcast` 是兼容当前 actor loss 的第一版实现。

## 9. 需要修改的文件

第一批必改：

```text
verl/trainer/config/ppo_trainer.yaml
verl/trainer/ppo/ray_trainer.py
verl/trainer/ppo/core_algos.py
verl/workers/fsdp_workers.py
verl/workers/actor/dp_actor.py
verl/workers/actor/value_head.py
verl/workers/sharding_manager/fsdp_vllm.py
verl/workers/sharding_manager/fsdp_sglang.py
agent_system/multi_turn_rollout/rollout_loop.py
```

第一版不支持 Megatron shared value head。Megatron 路径后续单独实现。

## 10. 验证计划

### 10.1 单元测试

新增或扩展：

```text
tests/trainer/ppo/test_step_ppo_algos.py
```

覆盖：

- 按 `traj_uid + step_idx` 恢复 step 顺序。
- step GAE 在 done 边界正确截断。
- GRPO episode advantage 按 `uid` 分组且按 `traj_uid` 去重。
- `A_final = (A_episode + w * A_step) / (1 + w)` 数值正确。
- token broadcast 后 mask 外为 0。

### 10.2 Smoke test

最小 WebShop smoke：

```bash
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=step_ppo \
  actor_rollout_ref.actor.value_head.enable=true \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  data.train_batch_size=2 \
  env.rollout.n=2 \
  env.max_steps=3 \
  trainer.total_epochs=1 \
  trainer.test_freq=-1 \
  trainer.logger=['console']
```

目标：能完成 rollout、old value 计算、step GAE、actor update、checkpoint save。

### 10.3 主实验

与现有 GiGPO WebShop 对齐：

```bash
bash EXPS/run_webshop_step_ppo_4gpu.sh \
  algorithm.adv_estimator=step_ppo \
  algorithm.gamma=0.95 \
  algorithm.lam=1.0 \
  algorithm.step_ppo.step_advantage_w=1.0 \
  actor_rollout_ref.actor.value_head.enable=true \
  actor_rollout_ref.actor.value_head.value_position=pre_action_last_context_token \
  actor_rollout_ref.actor.value_head.value_loss_coef=0.1 \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  data.train_batch_size=16 \
  env.rollout.n=8
```

## 11. 主要风险与处理

- vLLM/sglang 同步失败：优先检查 `value_head.*` 是否被过滤。
- value loss 干扰 policy：先降 `value_loss_coef`，再试 `detach_value_backbone=true`。
- advantage scale 不匹配：记录 `episode_adv/std`、`step_adv/std`、`final_adv/std`，必要时强制 whiten。
- token broadcast 长度偏差：优先使用 `seq-mean-token-mean`，后续实现 action-level ratio。
- batch reorder 破坏 step 顺序：必须使用 `step_idx`，不能依赖 batch 当前顺序。

## 12. 后续扩展

- action-conditioned value：`first_response_token`、`last_response_token`、response weighted pooling。
- action-level PPO ratio：按 action 汇总 log-prob ratio，而不是 token broadcast。
- LoRA shared value head：确认 value head 与 LoRA adapter 的保存、同步、加载行为。
- Megatron shared value head。
