# 026 Rethinking Groups in Critic-Free RLVR: C-RF + NTF Reproduction

## 1. Paper Target

Paper: `arXiv:2606.17250`, **Rethinking Groups in Critic-Free RLVR**.

The implementation target here is the paper's agentic-task setting on ALFWorld:

- group-free critic-free RL, one rollout per prompt (`G=1`);
- batch-level positive/negative split instead of prompt-level GRPO groups;
- Contrastive-REINFORCE (`C-RF`) objective for sparse binary success feedback;
- Negative Token Filtering (`NTF`): on negative trajectories, keep only the lowest-probability response tokens for the negative update;
- no reward KL penalty and no actor KL loss;
- no invalid-action shaping for the paper-aligned run.

## 2. Core Algorithm

### 2.1 C-RF Advantage Labels

Important repo detail: `agent_system/multi_turn_rollout/rollout_loop.py` stores one DataProto row per generated environment action, not one row per full episode. Rows from the same episode share `traj_uid` and `episode_rewards`.

For paper alignment, the C-RF implementation first groups rows by `traj_uid` and computes one outcome score per trajectory:

```text
score_traj = mean_row(sum_t token_level_rewards[row, t])
```

For ALFWorld, success is represented in this repo as a positive terminal reward, usually `10`; failures are `0`. Since C-RF only needs the sign/label, the script sets:

```text
positive_threshold = 0.0
positive trajectory if score_traj > 0.0
negative trajectory otherwise
```

Every action row in the same trajectory receives the same sign label:

```text
A[row, t] = +1 for rows from positive trajectories
A[row, t] = -1 for rows from negative trajectories
```

masked by valid response tokens. This is deliberately not GRPO normalization; it is a batch-level group-free split.

### 2.2 C-RF Loss

For current policy `pi_theta` and rollout behavior policy `pi_old`, define the token ratio:

```text
rho_t = exp(log_prob_t - old_log_prob_t)
```

For positive trajectories:

```text
l_pos(o) = (1 / |o|) * sum_t min(rho_t, 1 + eps_high)
```

For negative trajectories, NTF constructs a kept-token set `K(o)` from the lowest-probability response tokens. Because actor updates are micro-batched, the implementation precomputes `K(o)` once per full trajectory from `old_log_probs` after rollout log-prob recomputation. With `ppo_epochs=1`, this is effectively the same policy as the pre-update current policy, and it avoids micro-batch-dependent token selection.

```text
l_neg(o) = (1 / |o|) * sum_{t in K(o)} max(rho_t, 1 - eps_low)
```

Important implementation detail: the denominator remains the full valid trajectory response-token length `|o|`, not the number of kept tokens. This matches the paper's description that masked tokens contribute zero but still count in the length normalization.

The mini-batch objective is:

```text
L = -0.5 * (mean_pos(l_pos) - mean_neg(l_neg))
```

For one-sided mini-batches, the present side is used directly:

```text
only positives: L = -mean_pos(l_pos)
only negatives: L =  mean_neg(l_neg)
```

### 2.3 NTF Keep Ratio

The implementation uses `ntf_keep_ratio`, not `mask_ratio`, to avoid the notation ambiguity in the paper text/tables.

Default:

```text
ntf_keep_ratio = 0.1
```

This means: for negative rows, keep the lowest-probability 10% response tokens and mask the other 90% from the negative loss.

## 3. Code Mapping

### 3.1 Advantage Estimator

Implemented in:

- `verl/trainer/ppo/core_algos.py`
  - `compute_contrastive_reinforce_outcome_advantage(...)`

Trainer integration:

- `verl/trainer/ppo/ray_trainer.py`
  - added `AdvantageEstimator.CONTRASTIVE_REINFORCE = "contrastive_reinforce"`;
  - added no-critic handling for this estimator;
  - dispatches to C-RF advantage computation;
  - passes `traj_uid` so C-RF labels/weights are trajectory-level;
  - attaches `contrastive_rf_scores`, `contrastive_rf_labels`, `contrastive_rf_traj_token_weight`, and `contrastive_rf_ntf_mask` for loss/metrics/trace.

### 3.2 Actor Loss

Implemented in:

- `verl/trainer/ppo/core_algos.py`
  - `compute_c_rf_ntf_policy_loss(...)`

Actor integration:

- `verl/workers/actor/dp_actor.py`
- `verl/workers/actor/step_ppo_actor.py`

New loss mode:

```yaml
actor_rollout_ref.actor.policy_loss.loss_mode: c_rf_ntf
actor_rollout_ref.actor.policy_loss.ntf_keep_ratio: 0.1
actor_rollout_ref.actor.policy_loss.ntf_min_keep_tokens: 1
```

The StepPPO actor was updated only for compatibility. The reproduction script uses the normal PPO actor path, not the StepPPO value-head path.

Trajectory weighting details:

- `contrastive_rf_traj_token_weight` sums to `1.0` over all valid action tokens in each trajectory;
- positive and negative objectives are normalized separately by their trajectory-token weight mass;
- long failed episodes therefore do not dominate just because they produce more action rows;
- `contrastive_rf_ntf_mask` is selected over the full negative trajectory, not independently per action row.

### 3.3 Scheduler

The paper table lists an exponential scheduler for ALFWorld/WebShop. The repo previously only supported `constant` and `cosine` actor schedules, so this reproduction adds:

- `verl/utils/torch_functional.py`
  - `get_exponential_schedule_with_warmup(...)`
- `verl/workers/fsdp_workers.py`
  - actor/critic branch for `warmup_style=exponential`

Default script values:

```text
lr = 1e-6
lr_warmup_steps = 10
warmup_style = exponential
min_lr_ratio = 0.1
weight_decay = 0.1
```

The paper does not specify the final exponential decay ratio, so `LR_MIN_RATIO=0.1` is a configurable local choice.

## 4. Paper Alignment

The full ALFWorld script defaults are chosen to follow the paper where the repo supports it:

| Item | Paper | Script |
| --- | --- | --- |
| Backbone | Qwen2.5-1.5B-Instruct | `Qwen/Qwen2.5-1.5B-Instruct` |
| ALFWorld train subset | 1024 tasks | `TRAIN_DATA_SIZE=1024` |
| Rollouts per prompt | `G=1` | `GROUP_SIZE=1` / `env.rollout.n=1` |
| Rollout batch size | 64 prompts | `TRAIN_BATCH_SIZE=64` |
| PPO mini-batch size | 64 | `PPO_MINI_BATCH_SIZE=64` |
| LR | `1e-6` | `LR=1e-6` |
| Warmup | 10 steps | `LR_WARMUP_STEPS=10` |
| Weight decay | 0.1 | `WEIGHT_DECAY=0.1` |
| Clip low | 0.2 | `clip_ratio_low=0.2` |
| Clip high | 10 | `clip_ratio_high=10.0` |
| NTF | keep lowest-prob 10% negative tokens | `NTF_KEEP_RATIO=0.1` |
| KL in reward | disabled | `algorithm.use_kl_in_reward=False` |
| KL loss | disabled | `actor.use_kl_loss=False` |
| Max ALFWorld steps | 50 | `MAX_ENV_STEPS=50` |
| Train sampling | temp 1.0, top-p 1.0 | `temperature=1.0`, `top_p=1.0` |
| Eval sampling | temp 1.0, top-p 0.7, n=3 | default validation overrides |

Known deviations / repo-specific choices:

- The repo's ALFWorld success reward is positive terminal reward, commonly `10`, not literally `1`; C-RF uses only the positive/negative label, so this does not change the C-RF loss sign.
- The paper does not expose total training steps for ALFWorld. The script defaults to `TOTAL_TRAINING_STEPS=150` to stay comparable with the local GRPO/GiGPO/GraphGPO full scripts; set `TOTAL_TRAINING_STEPS=1000` or another value for a longer paper-style run.
- The script defaults to `MAX_RESPONSE_LENGTH=4096` for paper alignment. The smoke test used `MAX_RESPONSE_LENGTH=512` for speed.
- `LR_MIN_RATIO=0.1` is configurable because the paper table says exponential scheduler but does not state the final decay ratio.
- NTF token ranking uses recomputed behavior/old log-probabilities at rollout time rather than recomputing a full-trajectory ranking inside the actor micro-batch. With `ppo_epochs=1`, this avoids a micro-batch artifact while staying very close to the pre-update current policy.

## 5. One-Command Scripts

### 5.1 Single Seed Full Run

Absolute path:

```bash
/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/scripts/run_crf_ntf_alfworld_1p5b_full.sh
```

Command:

```bash
bash /prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/scripts/run_crf_ntf_alfworld_1p5b_full.sh
```

Useful overrides:

```bash
SEED=2026 TOTAL_TRAINING_STEPS=1000 ENABLE_WANDB=1 \
bash /prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/scripts/run_crf_ntf_alfworld_1p5b_full.sh
```

### 5.2 Three Seeds Serial Run

Absolute path:

```bash
/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/scripts/run_crf_ntf_alfworld_1p5b_full_3seeds.sh
```

Command:

```bash
bash /prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/scripts/run_crf_ntf_alfworld_1p5b_full_3seeds.sh
```

Default seeds:

```text
2026, 2077, 2501
```

Custom seeds:

```bash
SEEDS="2026 2077 2501" \
bash /prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/scripts/run_crf_ntf_alfworld_1p5b_full_3seeds.sh
```

## 6. Metrics Added

Actor-side NTF metrics:

```text
actor/c_rf_ntf/positive_rate
actor/c_rf_ntf/positive_rows
actor/c_rf_ntf/negative_rows
actor/c_rf_ntf/positive_objective_mean
actor/c_rf_ntf/negative_objective_mean
actor/c_rf_ntf/ntf_keep_ratio
actor/c_rf_ntf/ntf_kept_token_frac
actor/c_rf_ntf/ntf_kept_logprob_mean
actor/c_rf_ntf/ntf_masked_logprob_mean
```

Batch/trajectory metrics:

```text
contrastive_rf/positive_rate
contrastive_rf/negative_rate
contrastive_rf/positive_traj_rate
contrastive_rf/negative_traj_rate
contrastive_rf/unique_traj_count
contrastive_rf/score_mean
contrastive_rf/score_max
contrastive_rf/score_min
contrastive_rf/traj_score_mean
contrastive_rf/traj_score_max
contrastive_rf/traj_score_min
contrastive_rf/traj_token_weight_total
contrastive_rf/row_weight_mass_mean
contrastive_rf/row_weight_mass_max
contrastive_rf/ntf_kept_token_frac
```

These metrics make it possible to check:

- whether the batch contains both positive and negative rows;
- whether the negative kept-token fraction is close to `NTF_KEEP_RATIO`;
- whether kept tokens really have lower log-probability than masked tokens;
- whether success-rate changes correlate with the NTF update.

## 7. Smoke Test

A 1-step smoke was run with a small setup:

```bash
CLEAN_RAY=1 \
EXPERIMENT_NAME=crf_ntf_smoke_20260629_020422 \
TRAIN_DATA_SIZE=8 \
TRAIN_BATCH_SIZE=8 \
VAL_DATA_SIZE=8 \
TOTAL_TRAINING_STEPS=1 \
TOTAL_EPOCHS=1 \
TEST_FREQ=-1 \
VAL_BEFORE_TRAIN=False \
MAX_RESPONSE_LENGTH=512 \
PPO_MINI_BATCH_SIZE=8 \
PPO_MICRO_BATCH_SIZE_PER_GPU=2 \
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2 \
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2 \
GPU_MEMORY_UTILIZATION=0.5 \
RAY_NUM_CPUS=60 \
ENV_WORKER_CPUS=0.1 \
timeout 900s scripts/run_crf_ntf_alfworld_1p5b_full.sh
```

Log:

```text
logs/crf_ntf_traj_smoke_20260629_021800.log
```

Key smoke metrics at step 1:

```text
actor/c_rf_ntf/trajectory_weighted: 1.000
actor/c_rf_ntf/precomputed_ntf_mask: 1.000
actor/c_rf_ntf/ntf_keep_ratio: 0.100
actor/c_rf_ntf/ntf_kept_token_frac: 0.091
actor/c_rf_ntf/ntf_kept_logprob_mean: -4.699
actor/c_rf_ntf/ntf_masked_logprob_mean: -0.567
contrastive_rf/positive_traj_rate: 0.125
contrastive_rf/negative_traj_rate: 0.875
contrastive_rf/unique_traj_count: 8.000
contrastive_rf/traj_token_weight_total: 8.000
contrastive_rf/ntf_kept_token_frac: 0.100
actor/pg_loss: -0.043
actor/grad_norm: 6.885
episode/success_rate: 0.125
```

The smoke completed one training step and emitted the trajectory-aware C-RF/NTF metrics. At process shutdown, Ray cleanup printed a DataLoader worker `Killed` message after final metrics; the shell command exited `0`, so this is treated as shutdown noise rather than a training-step failure.

## 8. Next Checks for a Full Reproduction

For a real full run, check the following early metrics:

1. `contrastive_rf/positive_rate` should not be exactly `0` for too long; if it is, the positive signal is too sparse for C-RF to learn.
2. `actor/c_rf_ntf/ntf_kept_token_frac` should stay close to `0.1` on negative rows.
3. `actor/c_rf_ntf/ntf_kept_logprob_mean` should be lower than `actor/c_rf_ntf/ntf_masked_logprob_mean`; otherwise the low-probability token selection is wrong.
4. `actor/grad_norm` and `response_length/mean` should be watched for collapse, matching the paper's NTF sensitivity discussion.
5. Validation should use `n=3`, `temperature=1.0`, `top_p=0.7` for paper-style ALFWorld reporting.
