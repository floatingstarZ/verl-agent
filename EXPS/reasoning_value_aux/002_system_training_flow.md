# 002 Reasoning-Value Auxiliary Branch System Training Flow

Date: 2026-07-02

## 1. 文档目标

本文从系统执行层面说明当前 `reasoning_value_aux` 实验在一次 ALFWorld GRPO 训练 step 中如何运行，重点覆盖：

1. Ray / worker / rollout engine 如何接入。
2. multi-turn environment rollout 如何产生 action trajectory。
3. reasoning-value side branch 如何构造 prompt、生成或跳过生成、写入 batch。
4. reward、old logprob、reference logprob、advantage 如何计算。
5. actor update 时 policy loss 和 auxiliary value loss 如何合并。
6. 当前 `prompt_only` 优化为什么能提高训练效率，以及它改变了什么、没有改变什么。

核心结论：当前实现没有改变 GRPO 的 action policy 训练口径。`reasoning_value_aux` 只是给 actor 额外加一个共享 value head 的 side regression loss；该 loss 的输入来自 value prompt 分支，目标来自同一条环境 trajectory 的 reward-to-go。

## 2. 总体训练链路

一次训练 step 的主链路在 `RayPPOTrainer.fit()` 中执行，关键顺序如下：

```text
train dataloader batch
  -> gen_batch
  -> traj_collector.multi_turn_loop(...)
  -> reward computation
  -> actor old_log_prob recomputation
  -> optional reference logprob
  -> advantage computation
  -> optional trace / diagnostics dump
  -> actor update
  -> optional validation
  -> optional checkpoint
  -> metrics logging
```

对应代码位置：

- `verl/trainer/main_ppo.py`：初始化 Ray、环境、worker、tokenizer、trajectory collector。
- `verl/trainer/ppo/ray_trainer.py`：训练主循环、reward、logprob、advantage、update、validation。
- `agent_system/multi_turn_rollout/rollout_loop.py`：原始 multi-turn environment rollout。
- `agent_system/multi_turn_rollout/reasoning_value_rollout.py`：带 reasoning-value side branch 的 rollout collector。
- `verl/workers/step_ppo_fsdp_workers.py`：value-head actor worker。
- `verl/workers/actor/step_ppo_actor.py`：actor logprob、policy loss、reasoning-value loss。

## 3. Worker 与模型接入

### 3.1 为什么使用 StepWise worker

当配置中存在：

```text
actor_rollout_ref.actor.value_head.enable=True
actor_rollout_ref.actor.reasoning_value_aux.enable=True
```

`verl/trainer/main_ppo.py` 会选择 `ReasoningValueTrajectoryCollector`，并要求 actor 使用带 value head 的 worker。

系统层面的变化是：

```text
Base ActorRolloutRefWorker
  -> StepPPO ActorRolloutRefWorker
  -> attach_step_value_head(model)
  -> replace actor with StepWisePPOActor
```

具体效果：

- rollout engine 仍然负责 action generation。
- actor FSDP model 额外挂一个 shared scalar value head。
- value head 参数不会同步到 vLLM / SGLang rollout engine；rollout 只需要 policy weights。
- actor update 时可以在任意指定 token hidden state 上读 scalar value。

### 3.2 三类前向计算要区分

同一个 actor policy 在训练中会经历三类不同用途的前向：

```text
1. rollout generate_sequences
   用于采样 action，驱动环境。

2. compute_log_prob
   用当前 actor 重新计算 rollout action 的 old_log_probs。

3. update_policy
   用当前训练态 actor 计算新的 log_prob，并反传 policy loss 和 auxiliary value loss。
```

这三类计算使用同一个 actor 参数，但执行目的不同。`reasoning_value_aux` 只影响第 1 类 rollout 数据结构和第 3 类 actor update loss；它不改变 GRPO advantage 的定义。

## 4. ALFWorld Action Rollout

### 4.1 Actor 每一步看到什么

ALFWorld actor 在每个 step 不是延续聊天历史，而是环境重新构造一条完整 user prompt。该 prompt 通常包含：

```text
Your task is to: {task_description}
Prior observations and actions: {recent_history}
Current observation: {current_observation}
Admissible actions: {actions}
Now output <think>...</think><action>...</action>
```

系统把该文本套进 tokenizer chat template 后传给 rollout engine。

### 4.2 原始 rollout 循环

不启用 `reasoning_value_aux` 时，每个 environment step 的逻辑是：

```text
obs_t
  -> preprocess_batch(obs_t)
  -> actor_rollout_wg.generate_sequences(action_prompt_t)
  -> decode response_t
  -> extract / execute action_t
  -> env.step(action_t)
  -> receive obs_{t+1}, reward_t, done_t, info_t
  -> append one row to trajectory buffer
```

每个 row 保存 action policy 训练需要的字段：

```text
prompts
responses
input_ids
attention_mask
position_ids
rollout_log_probs, if returned by rollout backend
uid
traj_uid
rewards
active_masks
is_action_valid
episode_rewards
episode_lengths
```

其中：

- `uid` 用于 GRPO 同一 task / group 内比较。
- `traj_uid` 用于标识完整 episode trajectory。
- `active_masks` 表示该 env step 是否真实有效，已经 done 的环境后续 row 不参与 loss。

## 5. Reasoning-Value Branch 的 Rollout 改造

### 5.1 分支位置

启用 `reasoning_value_aux` 后，每个 environment step 先构造 normal action batch，然后额外挂 value branch：

```text
batch = preprocess_batch(obs_t)                  # action prompt
batch = attach_reasoning_value_branch(batch)     # side value prompt / tensors
batch_output = generate_sequences(action batch)  # action branch
batch = batch union batch_output
env.step(action)
```

重要点：value branch 不进入环境，不改变 action branch 的动作，也不改变 trajectory 本身。

### 5.2 两种 generation mode

当前实现支持两种 value branch 模式。

#### generate 模式

用于 case visualization / qualitative review：

```text
value_prompt_t
  -> actor_rollout_wg.generate_sequences(value_prompt_t)
  -> value_response_t = <think>...</think> + integer
  -> locate token after </think>
  -> value head reads that hidden state during update
```

该模式会真实生成 value reasoning 文本，所以可用于 HTML / JSON 可视化分析，但训练速度慢。

#### prompt_only 模式

用于 full training 默认配置：

```text
value_prompt_t ending with "Score:"
  -> no value text generation
  -> value head reads last prompt token hidden state
```

该模式跳过第二次 `generate_sequences`，只保留 value prompt 的 forward supervision。因为 value branch 不影响环境 action，训练时没有必要每步生成可读的 `<think>` 文本。

当前 full 脚本默认：

```text
REASON_VALUE_GENERATION_MODE=prompt_only
REASON_VALUE_MAX_PROMPT_LENGTH=768
```

case-viz 脚本默认：

```text
REASON_VALUE_GENERATION_MODE=generate
REASON_VALUE_MAX_PROMPT_LENGTH=3072
```

这样训练效率和可解释性 review 分离：训练走快路径，分析时再走文本生成路径。

### 5.3 prompt_only 的物理意义

`prompt_only` 下 value head 不再读取 `</think>` 后 token，而是读取 prompt 末尾 `Score:` 的 hidden state。这个 hidden state 已经编码了：

```text
task
verified task-relevant facts
recent task-relevant trajectory
filtered current observation
most relevant available actions
value scoring instruction
```

因此它表示“模型读完 value 判断问题后、准备输出分数前”的状态。它牺牲了显式 generated reasoning 的监督，但大幅减少 rollout 开销。

### 5.4 value branch 字段

训练 batch 中新增字段：

```text
reason_value_prompts
reason_value_responses
reason_value_input_ids
reason_value_attention_mask
reason_value_position_ids
reason_value_indices
reason_value_loss_mask
reason_value_targets
reason_value_think_close_found, generate mode only
reason_value_response_valid_len, generate mode only
```

字段含义：

- `reason_value_input_ids`：value branch 的完整输入序列。`generate` 模式下是 prompt + response；`prompt_only` 模式下只有 prompt。
- `reason_value_indices`：value head 读取 hidden state 的 token index。
- `reason_value_loss_mask`：该 row 是否参与 auxiliary value loss。
- `reason_value_targets`：episode 结束后反向计算的 reward-to-go target。
- `reason_value_prompts` / `reason_value_responses`：主要服务可视化与 debug。

## 6. Reasoning-Value Target 计算

episode 收集完成后，collector 对每条 trajectory 反向计算 target：

```text
running_value = 0
for t from T to 0:
    scaled_reward_t = reward_t / target_scale
    running_value = scaled_reward_t + gamma * running_value
    target_t = clip(running_value, target_min, target_max)
```

默认配置：

```text
target_gamma = 0.97
target_scale = 10.0
target_min = 0.0
target_max = 1.0
clip_target = True
```

在 ALFWorld 中，成功 reward 通常是 `10.0`。因此：

```text
成功终点附近 target 接近 1.0
失败轨迹 target 接近 0.0
较早成功前状态根据 gamma 得到折扣 value
```

这个 target 是 value regression 的监督信号，不参与 GRPO advantage 计算。

## 7. Reward 阶段

rollout 完成后，trainer 进入 reward 阶段：

```text
batch = gen_batch_output
reward_tensor, reward_extra_infos = compute_reward(batch, reward_fn)
batch.batch["token_level_scores"] = reward_tensor
```

当前 ALFWorld 使用 rule-based / environment reward manager。reward 与 action branch 绑定：

- 环境只执行 action branch 的 `<action>`。
- value branch 不产生 action，也不直接产生 reward。
- 若启用 invalid action penalty，会在 token-level reward 上扣分，并更新 `episode/valid_action_ratio`。

若 `algorithm.use_kl_in_reward=True`，系统会把 reference KL penalty 加进 token-level reward；当前 full 脚本中该项为：

```text
algorithm.use_kl_in_reward=False
```

因此当前训练中：

```text
token_level_rewards = token_level_scores
```

## 8. Old Logprob 阶段

### 8.1 为什么要重新算 old_log_probs

rollout engine 负责采样 action，但 PPO / GRPO 更新需要行为策略下的 log probability：

```text
old_log_probs = log pi_old(action_tokens | prompt)
```

在 hybrid engine / vLLM 场景下，rollout 返回的 logprob 不一定作为训练权威值使用，所以 trainer 会调用 actor worker 重新计算：

```text
old_log_prob = actor_rollout_wg.compute_log_prob(batch)
batch = batch union old_log_prob
```

### 8.2 只针对 action branch

`compute_log_prob` 使用的是 action branch 的字段：

```text
input_ids
responses
attention_mask
position_ids
```

输出：

```text
old_log_probs: [batch, response_length]
entropys: [batch, response_length]
```

当前配置：

```text
actor_rollout_ref.actor.value_head.compute_state_values=False
```

含义：

- old logprob 阶段不计算旧版 action-prompt value。
- old logprob 阶段不训练 reasoning-value branch。
- value branch 的 regression 只在 actor update 阶段计算。

### 8.3 entropy metric

trainer 会用 `entropys` 和 `response_mask` 计算：

```text
actor/entropy_loss
```

随后移除 `entropys`，只把 `old_log_probs` 合并回 batch。

## 9. Reference Logprob 阶段

如果配置启用 reference policy，trainer 会计算：

```text
ref_log_prob = log pi_ref(action_tokens | prompt)
```

它只服务 action branch 的 KL loss 或 KL reward penalty。

当前 reasoning-value branch 没有 reference logprob：

```text
reason_value_input_ids 不参与 ref policy KL
reason_value_targets 不参与 PPO ratio
reason_value_loss 是单独 regression loss
```

## 10. Advantage 阶段

trainer 在 reward 和 old logprob 后调用：

```text
compute_advantage(batch, adv_estimator=algorithm.adv_estimator, ...)
```

当前 full 脚本使用：

```text
algorithm.adv_estimator=grpo
```

GRPO advantage 仍然由 action branch 的 reward / group 信息决定：

```text
token_level_rewards
response_mask
uid / index grouping
traj_uid, if needed by estimator
  -> advantages
  -> returns
```

reasoning-value branch 不进入 advantage 计算。这一点很关键：

```text
policy gradient signal = 原 GRPO action reward signal
auxiliary value signal = reason_value_targets regression signal
```

两者只在 actor update 阶段通过 loss 相加耦合。

## 11. Actor Update 阶段

actor update 在 `StepWisePPOActor.update_policy()` 中执行。每个 minibatch / microbatch 的逻辑是：

```text
1. forward action branch
   input_ids + responses
   -> current log_prob
   -> optional entropy

2. policy loss
   old_log_probs, current log_prob, advantages, response_mask
   -> PPO / GRPO clipped policy loss

3. optional reasoning-value loss
   reason_value_input_ids, reason_value_indices
   -> value head scalar prediction
   -> regression to reason_value_targets

4. total loss backward
   policy_loss + value_loss_coef * reason_value_loss
```

### 11.1 Policy loss

默认 policy loss 使用 PPO clipped objective：

```text
ratio = exp(log_prob - old_log_prob)
loss1 = -advantage * ratio
loss2 = -advantage * clamp(ratio, 1 - clip_low, 1 + clip_high)
pg_loss = aggregate(max(loss1, loss2), response_mask)
```

这部分完全来自 action branch。

### 11.2 Reasoning-value loss

当 batch 中有 `reason_value_targets` 时，actor update 会额外执行：

```text
reason_values = value_head(hidden_state[reason_value_indices])
reason_targets = batch["reason_value_targets"]
reason_mask = batch["reason_value_loss_mask"]
```

默认 loss：

```text
reason_value_loss = 0.5 * mean_masked((reason_values - reason_targets)^2)
total_actor_loss += reason_value_loss_coef * reason_value_loss
```

如果配置 `loss_type=huber`，则使用 Huber loss。

### 11.3 梯度流

当前 full 脚本默认：

```text
REASON_VALUE_DETACH_BACKBONE=False
```

含义取决于 value head attach 实现：

- 若 detach 为 false，value loss 可以通过 value branch hidden state 回传到 actor backbone。
- 若 detach 为 true，value loss 只训练 value head，不更新 backbone。

如果目标是“只检查 value head 能否学稳且不干扰 policy”，可把它设为 true；如果目标是让 value reasoning 辅助 representation 学习，则设为 false 更符合当前实验意图。

### 11.4 Metrics

reasoning-value loss 会记录：

```text
actor/reason_value/loss
actor/reason_value/rmse
actor/reason_value/mae
actor/reason_value/explained_variance
actor/reason_value/corr
actor/reason_value/pred_mean
actor/reason_value/pred_std
actor/reason_value/target_mean
actor/reason_value/target_std
actor/reason_value/loss_mask_ratio
actor/reason_value/loss_coef
```

`generate` 模式还会记录：

```text
actor/reason_value/think_close_found_ratio
actor/reason_value/response_valid_len_mean
```

`prompt_only` 模式没有真实 response，因此不记录这两个文本生成质量指标。

## 12. Validation 与 Case Visualization

### 12.1 Full training validation

full 脚本中：

```text
reasoning_value_aux.train_only=True
```

因此 validation 默认不运行 value side branch。validation 只评估 action policy 的 ALFWorld 成功率和文本 score，避免额外开销。

### 12.2 Case-viz

case-viz 脚本会设置：

```text
reasoning_value_aux.train_only=False
REASON_VALUE_GENERATION_MODE=generate
```

这样 validation / rollout 时会生成 value text，并 dump：

```text
action prompt
action response
action
value prompt
value response
value scalar prediction
value target
reward
valid flag
```

case-viz 还会额外调用 `compute_reasoning_values`，对已经 dump 的 value branch 重新计算 scalar prediction，用于 HTML 曲线展示。

## 13. 当前效率优化

### 13.1 动态 prompt padding

旧实现对 value prompt 固定 padding 到：

```text
max_prompt_length = 3072
```

但当前 refined value prompt 的真实长度大约是：

```text
mean ~= 517 tokens
p95 ~= 583 tokens
max ~= 613 tokens
```

因此固定 padding 会让 value branch 大量空算。当前实现改为 batch 内动态 padding：

```text
local_max_prompt_length = min(max(actual prompt lengths in batch), configured max_prompt_length)
```

full 默认 `REASON_VALUE_MAX_PROMPT_LENGTH=768`，保留安全余量。

### 13.2 prompt_only 跳过 value 文本生成

旧训练路径：

```text
每个 env step:
  generate value response
  generate action response
```

新 full 默认路径：

```text
每个 env step:
  build value prompt tensors only
  generate action response
```

value branch 仍会在 actor update 阶段 forward 一次，用于 value loss；但 rollout 阶段不再做额外解码。由于 value response 不影响环境 trajectory，这个优化不改变 action 数据分布。

### 13.3 哪些计算仍然存在

`prompt_only` 不等于零成本。它仍然需要：

```text
CPU-side value prompt construction and tokenization
actor update 中 value branch forward
value head regression backward
```

但它消除了 rollout 阶段最贵的 value text autoregressive generation，并显著减少 padding token。

## 14. 训练数据张量流总结

### 14.1 Rollout 后 batch

```text
action branch:
  prompts
  responses
  input_ids
  attention_mask
  position_ids
  rewards
  active_masks
  uid
  traj_uid
  is_action_valid

reasoning-value branch:
  reason_value_input_ids
  reason_value_attention_mask
  reason_value_position_ids
  reason_value_indices
  reason_value_loss_mask
  reason_value_targets
```

### 14.2 Old logprob 后 batch

新增：

```text
old_log_probs
response_mask
```

### 14.3 Advantage 后 batch

新增：

```text
advantages
returns
```

### 14.4 Actor update 使用字段

policy loss 使用：

```text
responses
input_ids
attention_mask
position_ids
old_log_probs
advantages
response_mask / loss_mask
```

reasoning-value loss 使用：

```text
reason_value_input_ids
reason_value_attention_mask
reason_value_position_ids
reason_value_indices
reason_value_targets
reason_value_loss_mask
```

## 15. 当前实验口径

当前 full 脚本代表的实验口径是：

```text
Base algorithm: ALFWorld GRPO
Action rollout: unchanged
Action reward: unchanged except invalid-action penalty
GRPO advantage: unchanged
Policy loss: unchanged PPO/GRPO clipped loss
Auxiliary task: actor-side reasoning-value regression
Training value mode: prompt_only
Visualization value mode: generate
```

因此，分析实验结果时应分开看两类指标：

```text
Policy behavior:
  val/success_rate
  episode/valid_action_ratio
  response_length/mean
  actor/pg_loss
  actor/ppo_kl
  actor/pg_clipfrac

Auxiliary value learning:
  actor/reason_value/loss
  actor/reason_value/rmse
  actor/reason_value/mae
  actor/reason_value/explained_variance
  actor/reason_value/corr
  actor/reason_value/pred_mean / target_mean
```

如果 policy 指标退化但 value 指标变好，说明 auxiliary loss 可能干扰 actor backbone；可以尝试：

```text
REASON_VALUE_DETACH_BACKBONE=True
REASON_VALUE_LOSS_COEF lower, e.g. 0.01 or 0.02
REASON_VALUE_GENERATION_MODE=prompt_only unchanged
```

如果 value 指标不学习，则优先检查：

```text
reason_value_targets 是否有方差
reason_value_loss_mask_ratio 是否接近 1
pred_mean / pred_std 是否塌缩
explained_variance / corr 是否持续接近 0 或负值
```

## 16. 推荐运行配置

full training 默认建议：

```bash
REASON_VALUE_GENERATION_MODE=prompt_only \
REASON_VALUE_PROMPT_STYLE=alfworld_actor_aligned_refined \
REASON_VALUE_MAX_PROMPT_LENGTH=768 \
/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/EXPS/reasoning_value_aux/scripts/run_reasoning_value_grpo_alfworld_1p5b_full.sh
```

case visualization 建议：

```bash
REASON_VALUE_GENERATION_MODE=generate \
REASON_VALUE_PROMPT_STYLE=alfworld_actor_aligned_refined \
/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/EXPS/reasoning_value_aux/scripts/run_reason_value_case_viz_10val.sh
```

这两个命令使用同一套 state/prompt 设计，只是训练时跳过文本生成，分析时保留文本生成。
