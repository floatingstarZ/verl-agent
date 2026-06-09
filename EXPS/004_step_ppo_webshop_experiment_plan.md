# 004 Step-wise PPO WebShop 实验计划

## 背景

本文档记录 StepPPO 在 WebShop 上的第一版可执行实验计划。目标是尽量复用已经跑通的 WebShop GiGPO 路径，同时把 StepPPO 隔离在新的训练入口和脚本中，避免直接改动原始 PPO/GRPO/GiGPO 脚本。

StepPPO 相关入口包括：

- `verl.trainer.main_step_ppo`
- `verl/trainer/config/step_ppo_trainer.yaml`
- `verl/workers/step_ppo_fsdp_workers.py`
- `verl/workers/actor/step_ppo_actor.py`

## 已恢复的 GiGPO 基线

已有三个完整的 GiGPO 4GPU WebShop 日志：

| Seed | 日志 | 状态 |
|---:|---|---|
| 2026 | `logs/webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_20260530_114442.log` | 完整 250 steps |
| 2077 | `logs/webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_20260531_004613.log` | 完整 250 steps |
| 2501 | `logs/webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_20260531_123134.log` | 完整 250 steps |

这些日志用于比较 validation task score、success rate、text score、valid action ratio、response length、timing 和资源消耗。

## 运行环境要求

实验从仓库根目录启动，并使用准备好的 conda/Python 环境。脚本会设置 Hugging Face、Triton、vLLM 和 Ray cache。WebShop 环境侧 CPU 压力较大，因此脚本中会限制线程数和 JVM 资源。

## StepPPO 与 GiGPO 的差异

StepPPO 的核心差异是：

- 使用 `verl.trainer.main_step_ppo`。
- 启用 `actor_rollout_ref.actor.value_head`。
- 在 old-log-probability 阶段额外记录 old state values。
- 使用 immediate step rewards 计算 step-wise GAE。
- 在 actor update 中加入 clipped scalar value loss。

GiGPO 没有 learned critic，它通过 step group 构造 critic-free relative step advantage。因此 GiGPO 是 StepPPO 最重要的 step-aware baseline。

## 已准备脚本

- `exps/run_webshop_step_ppo_2gpu_smoke.sh`：2GPU smoke test。
- `exps/run_webshop_step_ppo_4gpu_paper_align_simple.sh`：4GPU StepPPO base script。
- `exps/run_webshop_step_ppo_v1_step_norm_4gpu_paper_align_simple.sh`：v1 step-normalized 版本。
- `exps/run_webshop_step_ppo_v0_base_4gpu_paper_align_simple.sh`：v0 unnormalized step-advantage 版本。
- `exps/run_webshop_grpo_4gpu_paper_align_simple.sh`：GRPO baseline。

## 首轮运行顺序

首轮只使用 seed `2026`：

```text
StepPPO-v1 2026 -> StepPPO-v0 2026 -> GRPO 2026
```

对应 wrapper：

```bash
bash exps/run_webshop_step_ppo_v1_v0_grpo_seed2026.sh
```

## 已知风险

StepPPO 会比 GiGPO 多 value prediction 和 value loss 开销。如果 value loss 干扰 actor backbone，还会导致 response 更长、invalid action 更多，从而进一步拖慢 rollout generation。validation 本身也很贵，因此比较 wall-clock 时必须保持 `trainer.test_freq` 一致。
