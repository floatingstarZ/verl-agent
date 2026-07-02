# 027 ALFWorld PPO Baseline Construction

日期：2026-06-29

## 1. 背景理解

当前目录里的实验主线可以分成两层：

- `EXPS/tex/` 与 `EXPS/run_webshop_*.sh` 主要记录 WebShop 上的 StepPPO / GiGPO / GRPO 对照实验，当前 StepPPO-v3 约定是 value-head-only，不改变 GiGPO policy objective。
- ALFWorld 方向的近期实验主要在 `scripts/run_grpo_alfworld_1p5b_full.sh`、`scripts/run_graphgpo_alfworld_1p5b_full.sh`、`scripts/run_crf_ntf_alfworld_1p5b_full.sh`、`scripts/run_ssca_retry_grpo_alfworld_1p5b_full.sh` 与 `EXPS/analysis/02*.md` 中，核心是在 GRPO/GiGPO/GraphGPO/SSCA/CRF-NTF 上比较 critic-free 或结构化 credit assignment。

因此 PPO baseline 的目标不是新 StepPPO 版本，而是补一个传统 critic-based baseline：

```text
ALFWorld + main_ppo + algorithm.adv_estimator=gae + separate critic
```

## 2. Baseline 口径

新增 PPO baseline 使用现有 `verl.trainer.main_ppo` 与 `agent_system` ALFWorld 环境路径：

- 环境：`env.env_name=alfworld/AlfredTWEnv`
- actor：`Qwen/Qwen2.5-1.5B-Instruct`
- advantage：`algorithm.adv_estimator=gae`
- critic：单独 FSDP critic，`critic.model.path` 与 actor 相同
- reward：沿用 `EpisodeRewardManager`，每条有效 action row 使用 episode reward，并保留 invalid action penalty
- KL：`actor_rollout_ref.actor.use_kl_loss=True`，`algorithm.use_kl_in_reward=False`

为了和 ALFWorld GRPO/GiGPO 的 `16 tasks x 8 rollouts` 采样量对齐，PPO 默认直接跑：

```text
TRAIN_DATA_SIZE=128
GROUP_SIZE=1
```

也就是说 PPO 不做同 prompt group sampling；每个 batch 是 128 个独立 ALFWorld env/task。

## 3. 新增入口

主脚本：

```bash
bash scripts/run_ppo_alfworld_1p5b_full.sh
```

三种子脚本：

```bash
bash scripts/run_ppo_alfworld_1p5b_full_3seeds.sh
```

EXPS 兼容入口：

```bash
bash EXPS/run_alfworld_ppo_4gpu_paper_align_simple.sh
```

Dry run 检查命令：

```bash
DRY_RUN=1 bash scripts/run_ppo_alfworld_1p5b_full.sh
```

## 4. 默认资源配置

默认配置面向当前 H100 4 卡 full run：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS_PER_NODE=4
RAY_NUM_CPUS=64
ENV_WORKER_CPUS=0.1
TENSOR_MODEL_PARALLEL_SIZE=2
TRAIN_DATA_SIZE=128
VAL_DATA_SIZE=128
TOTAL_EPOCHS=150
TEST_FREQ=5
SAVE_FREQ=-1
```

如果要先做小 smoke：

```bash
TRAIN_DATA_SIZE=16 \
VAL_DATA_SIZE=16 \
PPO_MINI_BATCH_SIZE=64 \
ACTOR_MICRO_BATCH_SIZE=4 \
CRITIC_MICRO_BATCH_SIZE=4 \
TOTAL_TRAINING_STEPS=1 \
TEST_FREQ=1 \
SAVE_FREQ=-1 \
bash scripts/run_ppo_alfworld_1p5b_full.sh
```

## 5. 需要注意的实现语义

当前 `main_ppo` 的 multi-turn rollout 会把每个 ALFWorld action step 作为一条训练 row。`EpisodeRewardManager` 把整条 episode reward 写到该 row 的 response 末 token，因此这个 PPO baseline 是“critic-based sequence/action-row PPO”，而不是跨环境步显式 bootstrapping 的 step-level actor-critic。

这与现有 GRPO/GiGPO 路径可直接比较，因为它们也使用同一个 `agent_system.multi_turn_rollout` 数据契约和同一个 ALFWorld reward/valid-action 处理逻辑；差异主要是 PPO 使用 separate critic + GAE，而 GRPO/GiGPO 使用 critic-free group advantage。

## 6. 建议对照

建议先跑一组 seed 2026：

```bash
SEED=2026 bash scripts/run_ppo_alfworld_1p5b_full.sh
```

再和已有 full run 对齐比较：

```text
logs/grpo_qwen25_15b_alfworld_full_seed2026_*.log
logs/graphgpo_qwen25_15b_alfworld_full_seed2026_*.log
logs/ssca_retry_grpo_qwen25_15b_alfworld_full_seed2026_*.log
```

核心指标：

- `val/success_rate`
- `episode/valid_action_ratio`
- `episode/reward/mean`
- `critic/vf_loss`
- `critic/vpred_mean`
- `actor/pg_loss`
- `actor/kl_loss`
- `response_length/clip_ratio`

## 7. 5-step Smoke 验证

2026-06-29 已完成一次 5-step smoke run：

```text
logs/ppo_alfworld_5step_smoke_20260629_085932.log
```

运行使用的关键覆盖项：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
SEED=2026 \
TRAIN_DATA_SIZE=4 \
VAL_DATA_SIZE=4 \
PPO_MINI_BATCH_SIZE=4 \
ACTOR_MICRO_BATCH_SIZE=1 \
CRITIC_MICRO_BATCH_SIZE=1 \
LOG_PROB_MICRO_BATCH_SIZE=1 \
REF_LOG_PROB_MICRO_BATCH_SIZE=1 \
TOTAL_TRAINING_STEPS=5 \
TOTAL_EPOCHS=5 \
TEST_FREQ=5 \
VAL_BEFORE_TRAIN=False \
SAVE_FREQ=-1 \
MAX_ENV_STEPS=5 \
RAY_NUM_CPUS=16 \
DATALOADER_NUM_WORKERS=0 \
ENV_WORKER_CPUS=0.1 \
CHECK_FLASH_ATTN=1 \
bash scripts/run_ppo_alfworld_1p5b_full.sh
```

验证结果：

- `step:1` 到 `step:5` 均完成，包含 rollout、reward、old log-prob、ref log-prob、critic values、GAE、critic update、actor update。
- `step:5` 触发 validation，输出 `Final validation metrics`。
- 最终运行正常退出，没有 `RayTaskError`、`RuntimeError`、`ValidationError`、`Too Many Requests` 或 DataLoader worker killed traceback。

本次 smoke 前修复了两个运行阻塞点：

- Hydra struct 模式下 `attn_implementation` 必须用 `+actor_rollout_ref.model.attn_implementation=...` 和 `+critic.model.attn_implementation=...` 追加。
- `verl/workers/critic/dp_critic.py` 中 3D `position_ids_rmpad` 赋值在当前 HEAD 中断行导致 syntax error，已合并成一行合法赋值。

脚本现在默认优先使用本地 Hugging Face snapshot：

```text
$HF_HOME/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/<refs/main>
```

这样可以避免 vLLM 初始化时访问 Hugging Face Hub 触发 429 rate limit。若需要强制使用远端 repo id，可以显式设置：

```bash
MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct bash scripts/run_ppo_alfworld_1p5b_full.sh
```

## 8. Full Run Step-5 检查

2026-06-29 启动一次 seed 2026 的 full PPO run：

```text
logs/ppo_qwen25_15b_alfworld_full_seed2026_20260629_092215.log
```

关键配置保持 full baseline：`TRAIN_DATA_SIZE=128`、`VAL_DATA_SIZE=128`、`MAX_ENV_STEPS=50`、`TOTAL_TRAINING_STEPS=150`、`TEST_FREQ=5`、`DATALOADER_NUM_WORKERS=0`。

截至 `step:5`，训练和验证均未崩溃，且第 5 step 触发 validation：

- `val/success_rate: 0.125`
- `val/text/test_score: 0.624`
- `episode/success_rate: 0.117`
- `episode/reward/mean: 1.172`
- `episode/valid_action_ratio: 0.846`
- `critic/vf_loss: 6.688`
- `critic/vf_explained_var: 0.020`
- `response_length/clip_ratio: 0.001`
- `timing_s/step: 317.805`，其中 validation 用时 `111.621s`

前 5 个 training step 的 train success rate 约在 `0.094-0.117` 区间波动；5 step 只能说明 full 配置已越过 rollout/update/validation 的早期稳定性检查，是否真正收敛还需要继续观察后续 validation 点（例如 step 50/100/150）。
