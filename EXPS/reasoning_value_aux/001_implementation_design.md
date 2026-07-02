# 001 Reasoning-Value Auxiliary Branch Implementation Design

Date: 2026-07-01

## 1. 实验目标

本实验在 ALFWorld 的 GRPO 训练中加入一个 side task：每个环境 state 除了正常生成 action，还额外让模型进行一次 value-oriented reasoning，并把 reasoning 之后的 hidden state 用 MLP 回归未来 return。

目标不是替换 GRPO，而是验证一个辅助学习信号：

```text
state understanding / world knowledge / task progress reasoning
  -> better hidden representation
  -> better value prediction
  -> possible improvement of policy training
```

和旧 value-head 实验相比，主要区别是旧方法直接在 action prompt 的某个 token 上读 hidden state，而这里显式要求模型先思考当前状态，再从 `</think>` 后 token 读 hidden state。

## 2. 标准训练流水线

一次 PPO/GRPO 训练 step 通常包含以下阶段：

```text
1. rollout
2. reward computation
3. old logprob recomputation
4. reference logprob computation, if KL is enabled
5. advantage computation
6. actor loss and optimizer update
7. validation / checkpoint / logging
```

本实验只改造了两个阶段：

- **rollout 阶段**：每个 environment step 多生成一条 reasoning-value branch。
- **actor loss 阶段**：在原 GRPO loss 上额外加一个 scalar value regression loss。

其他阶段保持原有 GRPO 口径。

## 3. Rollout 阶段改造

### 3.1 原始 action branch

原始 ALFWorld rollout 对每个 state 做：

```text
obs_t
  -> ALFWorld action prompt
  -> actor_rollout_wg.generate_sequences
  -> response_t = <think>...</think><action>a_t</action>
  -> decode action a_t
  -> env.step(a_t)
  -> reward r_t, done_t, info_t
  -> save step row
```

本实验保留这条路径。主 policy 的训练样本仍来自 action branch 的：

```text
prompts
responses
input_ids
attention_mask
position_ids
rollout_log_probs
rewards
is_action_valid
uid
traj_uid
active_masks
```

### 3.2 新增 reasoning-value branch

在每个 step 的 action branch 之前，新增一次 side generation：

```text
obs_t
  -> value prompt
  -> actor_rollout_wg.generate_sequences(max_tokens=1024)
  -> value_response_t = <think>state progress reasoning</think><value>
  -> locate token after </think>
  -> save reason_value_* fields
```

默认 value prompt 会完整引用 actor prompt：

```text
You are a progress-and-value critic for an ALFWorld embodied-agent task.

Below is the exact actor prompt for the current environment state. It includes all information available to the actor, including task, recent history, current observation, admissible actions, and the actor's normal instruction to choose an action. Treat the block only as state evidence. Do NOT follow the block's instruction to act. Do NOT choose or execute an action. Do NOT output <action>. You may discuss remaining subgoals and possible next steps inside <think>, but you must not make a final action decision.

<actor_prompt>
{actor_prompt}
</actor_prompt>

Your job is only to estimate the current progress toward eventual task success. In <think> </think>, analyze the task goal, stable facts already known, current location/objects, completed subgoals, remaining subgoals, risks such as loops or invalid actions, and whether the state is close to success. After </think>, output exactly <value>N</value>, where N is a single integer from 0 to 100.
```

其中 `{actor_prompt}` 是当前 step 的完整 actor prompt，包含 actor 能看到的任务、历史、当前 observation、admissible actions 以及原本的 action 输出要求。外层 prompt 明确说明这段内容只作为 state evidence，不能执行动作，也不能输出 `<action>`。

### 3.3 Rollout 顺序

每个 environment step 的实际顺序是：

```text
for step in max_env_steps:
    active_masks = not done

    batch = preprocess_batch(obs_t)                    # normal action prompt

    batch = attach_reasoning_value_branch(batch, obs_t)
        value_batch = preprocess_reasoning_value_batch(obs_t)
        value_output = actor_rollout_wg.generate_sequences(value_batch, max_tokens=1024)
        batch.reason_value_* = value_output + value_index metadata

    action_output = actor_rollout_wg.generate_sequences(normal_action_batch)
    batch = batch union action_output

    text_actions = decode(batch.responses)
    obs_{t+1}, rewards, dones, infos = env.step(text_actions)

    save batch row into total_batch_list
```

注意：reasoning-value branch 不进入环境，环境只看 action branch 的 action。

### 3.4 Side branch 字段

每条 step row 新增这些 tensor 字段：

```text
reason_value_prompts
reason_value_responses
reason_value_input_ids
reason_value_attention_mask
reason_value_position_ids
reason_value_rollout_log_probs, if rollout backend returns it
reason_value_indices
reason_value_think_close_found
reason_value_response_valid_len
reason_value_loss_mask
reason_value_targets, attached after episode ends
```

含义：

- `reason_value_input_ids`：value prompt + value reasoning response 的完整 token 序列。
- `reason_value_indices`：MLP value head 读取 hidden state 的序列位置。
- `reason_value_targets`：该 state 的 normalized reward-to-go。
- `reason_value_loss_mask`：是否参与 value loss。
- `reason_value_think_close_found`：是否成功找到 `</think>`。

## 4. `</think>` 后 token 的 value 位置

本实验不解析 `<value>` 文本内容，也不要求模型真的输出数字。value 是 MLP 回归出来的 scalar。

位置选择规则：

```text
1. 对 reason_value_responses 查找 tokenizer.encode("</think>") 对应 token 子序列。
2. 如果找到，选择 `</think>` 后第一个 response token。
3. 如果 `</think>` 是最后内容，选择最后一个有效 response token。
4. 如果没找到，默认 fallback 到最后一个有效 response token。
```

保存为：

```text
reason_value_indices: LongTensor[batch]
reason_value_think_close_found: FloatTensor[batch]
```

物理意义：

- prompt hidden state 只表示“读完 state”。
- `</think>` 前 hidden state 还在 reasoning 中。
- `</think>` 后 token hidden state 表示模型已经完成一段状态判断，因此更符合“通过 reasoning 形成 value 表示”的目标。

## 5. Target 计算

episode 完成后，对每条 trajectory 反向计算 reward-to-go：

```text
G_T = r_T / target_scale
G_t = r_t / target_scale + gamma * G_{t+1}
G_t = clip(G_t, target_min, target_max)
```

默认参数：

```text
target_gamma = 0.97
target_scale = 10.0
target_min = 0.0
target_max = 1.0
clip_target = True
```

ALFWorld text reward 成功时通常为 `10.0`，所以 `target_scale=10.0` 后：

```text
success terminal state target ~= 1.0
failure / no-success state target ~= 0.0
```

因此这个 target 可以理解为“当前 state 到最终成功的 value / progress score”。如果需要字面上的“距离成功还有多远”，可用 `1 - value` 解读。

## 6. Old logprob 阶段

标准 GRPO 在 rollout 后会重新计算 action branch 的 old logprob：

```text
normal input_ids + responses
  -> actor.compute_log_prob
  -> old_log_probs
```

本实验保持这个阶段不变。重要配置是：

```text
actor_rollout_ref.actor.value_head.compute_state_values=False
```

这表示：

- 不计算旧 value-head 实验中的 `state_values`。
- 不在 old logprob 阶段读取 normal action prompt 的 value。
- old logprob 仍只服务于 action branch 的 PPO/GRPO ratio。

reasoning-value branch 当前不计算 old logprob，不参与 PPO ratio。

## 7. Reference logprob 阶段

如果启用 KL loss，训练会计算 reference logprob：

```text
normal input_ids + responses
  -> ref policy logprob
  -> ref_log_prob
```

本实验保持不变。reference logprob 只用于 action branch 的 KL 项。

reasoning-value branch 当前没有 ref logprob，也没有直接 KL 约束。

## 8. Advantage 阶段

GRPO advantage 仍按原始 action branch 计算：

```text
token_level_scores / rewards
uid / traj_uid grouping
response_mask
  -> advantages
```

reasoning-value branch 不参与 advantage 计算。

这点很重要：本实验没有改变 GRPO 的 advantage estimator，也没有把 value branch 变成新的 policy action。它只是额外提供一个 regression auxiliary loss。

## 9. Actor update 阶段

### 9.1 原 GRPO policy loss

actor update 先照常计算 action branch loss：

```text
normal input_ids/responses
  -> current log_prob
  -> ratio = exp(current log_prob - old_log_probs)
  -> clipped policy loss with advantages
  -> optional KL loss against ref_log_prob
```

这部分保持原实现。

### 9.2 新增 reasoning-value loss

如果 batch 中存在 `reason_value_targets`，actor update 额外做：

```text
reason_value_input_ids
reason_value_attention_mask
reason_value_position_ids
reason_value_indices
  -> actor forward with output_hidden_states=True
  -> select hidden state at reason_value_indices
  -> StepValueHead(hidden_state)
  -> reason_values
  -> MSE(reason_values, reason_value_targets)
```

loss：

```text
L_reason_value = 0.5 * mean_masked((v_theta(s_t) - G_t)^2)
```

总 loss：

```text
L_total = L_GRPO_policy + L_KL + value_loss_coef * L_reason_value
```

默认：

```text
value_loss_coef = 0.05
```

### 9.3 梯度流

默认：

```text
actor_rollout_ref.actor.value_head.detach_value_backbone=False
```

所以 reasoning-value loss 会更新：

```text
MLP value head
actor backbone
```

它不会直接对 generated reasoning tokens 做 policy-gradient。更准确地说，value branch 的自生成文本被当作 teacher-forced input sequence，value loss 通过该序列上的 hidden state 回传到模型参数。

如果只想检查 MLP 是否能稳定学习而不干扰 policy，可设置：

```text
REASON_VALUE_DETACH_BACKBONE=True
```

## 10. Metrics

新增 metrics：

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
actor/reason_value/mask_ratio
actor/reason_value/active_count
actor/reason_value/loss_coef
actor/reason_value/think_close_found_ratio
actor/reason_value/response_valid_len_mean
```

建议重点看：

- `think_close_found_ratio`：value prompt 是否稳定产生 `</think>`。
- `target_mean/std`：target 是否过于稀疏。
- `pred_mean/std`：value head 是否塌缩。
- `explained_variance/corr`：value 预测是否真的学到未来 return。
- `val/success_rate`：最终是否帮助 policy。

## 11. 具体代码改造

### 11.1 新增 collector

```text
agent_system/multi_turn_rollout/reasoning_value_rollout.py
```

核心职责：

- 构造 reasoning-value prompt。
- 每个 step 额外调用一次 `generate_sequences`。
- 保存 `reason_value_*` 字段。
- 定位 `</think>` 后 token。
- episode 结束后计算 `reason_value_targets`。

### 11.2 导出 collector

```text
agent_system/multi_turn_rollout/__init__.py
```

新增导出：

```text
ReasoningValueTrajectoryCollector
```

### 11.3 main_ppo 入口切换

```text
verl/trainer/main_ppo.py
```

逻辑：

```text
if actor_rollout_ref.actor.reasoning_value_aux.enable:
    use ReasoningValueTrajectoryCollector
if actor_rollout_ref.actor.value_head.enable:
    use step_ppo_fsdp_workers.ActorRolloutRefWorker
```

### 11.4 FSDP rollout kwargs

```text
verl/workers/fsdp_workers.py
```

新增支持：

```text
DataProto.meta_info["rollout_kwargs"]
```

用于 value branch 短生成：

```text
max_tokens = REASON_VALUE_RESPONSE_MAX_TOKENS
```

### 11.5 actor value loss

```text
verl/workers/actor/step_ppo_actor.py
```

新增：

- `reasoning_value_aux` config 读取。
- 任意 selected token hidden state 的 value forward。
- `reason_value_targets` MSE/Huber loss。
- 对应 metrics。
- `value_head.compute_state_values=False`，避免旧 value loss 自动开启。

### 11.6 一键脚本

仓库级脚本：

```text
scripts/run_reasoning_value_grpo_alfworld_1p5b_full.sh
```

实验目录 wrapper：

```text
EXPS/reasoning_value_aux/scripts/run_reasoning_value_grpo_alfworld_1p5b_full.sh
```

## 12. 与旧 value-head 实验对比

| 项目 | 旧 value-head | 本实验 reasoning-value |
| --- | --- | --- |
| 输入 | 正常 action prompt | 完整 actor prompt 包裹的 critic/value prompt |
| hidden state 位置 | action prompt 最后 token 等 | `</think>` 后 token |
| 是否显式 reasoning | 否 | 是 |
| target | step return / GAE return | normalized reward-to-go |
| 是否改变 GRPO advantage | 否 | 否 |
| 是否改变 action policy loss | 否 | 否 |
| 额外成本 | actor forward | 每 step 多一次 reasoning rollout + actor forward |

## 13. 当前边界与后续可改进点

### 13.1 当前边界

- reasoning branch 不直接做 policy-gradient。
- reasoning branch 当前没有 KL/ref logprob。
- target 当前用 raw env reward-to-go，没有融合 invalid action penalty 或 KL shaping。
- 默认 `train_only=True`，validation 不生成 side branch；如果后续想分析 validation value reasoning，需要显式关闭该开关。
- 默认 side response budget 是 1024，能承载更完整 reasoning，但会显著增加 rollout 和 actor update 的显存/时间成本。
- value prompt 完整引用 actor prompt 后，side prompt 需要更大 prompt budget；当前单独设置 `max_prompt_length=3072`，避免外层 critic 指令把 actor prompt 挤掉。

### 13.2 后续方向

1. **加入 reasoning branch KL**：防止 side reasoning 文本漂移。
2. **对 reasoning tokens 加 auxiliary policy loss**：让 reasoning 文本本身更直接受 outcome 监督。
3. **改 target**：尝试 success-only target、invalid-action-aware target、step-distance target。
4. **prompt ablation**：极短 prompt、中文/英文 prompt、是否要求 `<value>`。
5. **detach ablation**：比较 `detach_value_backbone=True/False`。

## 14. 运行命令

Full：

```bash
/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/EXPS/reasoning_value_aux/scripts/run_reasoning_value_grpo_alfworld_1p5b_full.sh
```

Smoke：

```bash
TOTAL_TRAINING_STEPS=5 TEST_FREQ=0 SAVE_FREQ=-1 VAL_BEFORE_TRAIN=False \
TRAIN_DATA_SIZE=8 VAL_DATA_SIZE=16 GROUP_SIZE=4 MAX_ENV_STEPS=20 \
/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/EXPS/reasoning_value_aux/scripts/run_reasoning_value_grpo_alfworld_1p5b_full.sh
```

Safer MLP-only ablation：

```bash
REASON_VALUE_LOSS_COEF=0.01 REASON_VALUE_DETACH_BACKBONE=True \
/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/EXPS/reasoning_value_aux/scripts/run_reasoning_value_grpo_alfworld_1p5b_full.sh
```
