# ALFWorld Reasoning-Value Auxiliary Branch

本目录是一个独立实验，用于验证：在 ALFWorld GRPO 训练中，是否可以通过一个 side reasoning-value task，让模型在每个 state 先进行显式状态判断，再用 reasoning 后的 hidden state 回归未来成功 value，从而辅助 actor 学习世界状态、任务进度和剩余距离。

## 实验假设

旧 value-head 实验直接读取 action prompt 的某个 hidden state，例如最后一个 context token，再通过 MLP 回归 value。这个做法不够自然：模型没有被要求显式思考当前状态，也没有把 value 预测和 reasoning 过程绑定起来。

本实验改为：

```text
当前 ALFWorld state
  -> value prompt
  -> model generates <think>progress reasoning</think><value>
  -> take hidden state at the token after </think>
  -> MLP predicts scalar value
```

如果这个 auxiliary task 有效，应当观察到：

- `actor/reason_value/loss`、`rmse`、`mae` 下降。
- `actor/reason_value/explained_variance` 或 `corr` 提高。
- policy 主指标不明显退化：`val/success_rate`、`episode/valid_action_ratio`、KL、clipfrac 等保持稳定。
- 更理想情况下，validation success 相比纯 GRPO 提升。

## 当前状态

- 第一版代码已接入 GRPO 训练路径。
- 主 GRPO policy loss、advantage、reward 计算保持不变。
- 每个 training environment step 额外生成一条 value reasoning branch；默认 validation 不生成该 side branch，避免额外验证开销。
- actor update 时额外加 scalar value regression loss。
- 已完成静态编译、dry-run 和轻量逻辑单测；尚未启动 full 训练。

## 主要文件

- `001_implementation_design.md`：详细实现设计、训练流水线和字段说明。
- `scripts/run_reasoning_value_grpo_alfworld_1p5b_full.sh`：本实验目录内的一键启动 wrapper。
- `outputs/`：预留给后续 smoke/full 结果、metric 摘要和可视化输出。

核心代码入口：

```text
agent_system/multi_turn_rollout/reasoning_value_rollout.py
verl/workers/actor/step_ppo_actor.py
verl/trainer/main_ppo.py
verl/workers/fsdp_workers.py
scripts/run_reasoning_value_grpo_alfworld_1p5b_full.sh
```

## 一键运行

推荐从仓库任意位置直接运行：

```bash
/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/EXPS/reasoning_value_aux/scripts/run_reasoning_value_grpo_alfworld_1p5b_full.sh
```

该 wrapper 会调用仓库级脚本：

```text
/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/scripts/run_reasoning_value_grpo_alfworld_1p5b_full.sh
```

## 建议 smoke 命令

```bash
TOTAL_TRAINING_STEPS=5 TEST_FREQ=0 SAVE_FREQ=-1 VAL_BEFORE_TRAIN=False \
TRAIN_DATA_SIZE=8 VAL_DATA_SIZE=16 GROUP_SIZE=4 MAX_ENV_STEPS=20 \
/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/EXPS/reasoning_value_aux/scripts/run_reasoning_value_grpo_alfworld_1p5b_full.sh
```

## 关键配置

```text
actor_rollout_ref.actor.value_head.enable=True
actor_rollout_ref.actor.value_head.compute_state_values=False
actor_rollout_ref.actor.value_head.detach_value_backbone=False
actor_rollout_ref.actor.reasoning_value_aux.enable=True
actor_rollout_ref.actor.reasoning_value_aux.value_loss_coef=0.05
actor_rollout_ref.actor.reasoning_value_aux.target_gamma=0.97
actor_rollout_ref.actor.reasoning_value_aux.target_scale=10.0
actor_rollout_ref.actor.reasoning_value_aux.response_max_tokens=1024
actor_rollout_ref.actor.reasoning_value_aux.max_prompt_length=3072
actor_rollout_ref.actor.reasoning_value_aux.train_only=True
actor_rollout_ref.actor.reasoning_value_aux.strip_action_instruction=False
```

`compute_state_values=False` 表示关闭旧的 action-prompt value loss，只使用 reasoning-value branch。`strip_action_instruction=False` 表示 value prompt 引用完整 actor prompt；外层指令会明确禁止执行动作。

## 后续记录约定

- smoke 日志摘要写入 `outputs/smoke_<date>.md`。
- full run 指标分析写入 `outputs/full_<seed>_<date>.md`。
- 若生成图表或 HTML，可放在 `outputs/figures/` 或 `outputs/html/`。
