# SSCA Retry + VIMPO-style Value Loss Implementation Notes

Date: 2026-06-26

This document explains the current SSCA two-trajectory retry implementation and the VIMPO-style value loss added on top of the GRPO training path. It is intended as an implementation review and experiment guide.

## Status

Current implementation status:

- The SSCA rollout path runs `traj1 -> summary -> traj2`.
- The original GRPO policy advantage path is kept.
- The earlier value-head MSE prototype has been replaced as the default by a critic-free VIMPO-style terminal value loss.
- The value head is disabled by default for the SSCA VIMPO full script.
- Smoke tests have run through actor update and validation on ALFWorld 1.5B.
- The full one-command script is ready to launch.

Main script:

```text
/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/scripts/run_ssca_retry_grpo_alfworld_1p5b_full.sh
```

Main implementation files:

```text
recipe/SSCA/rollout_loop.py
recipe/SSCA/vimpo_trainer.py
recipe/SSCA/vimpo_core.py
recipe/SSCA/main_ssca.py
verl/workers/actor/dp_actor.py
verl/workers/actor/step_ppo_actor.py
verl/trainer/ppo/metric_utils.py
verl/trainer/ppo/ray_trainer.py
scripts/run_ssca_retry_grpo_alfworld_base.sh
scripts/run_ssca_retry_grpo_alfworld_1p5b_full.sh
```

## VIMPO Principle

The VIMPO paper is `VIMPO: Value-Implicit Policy Optimization for LLMs`, arXiv `2606.20008`, with official code at `backprop07/VIMPO`.

The key point is that VIMPO does not train a separate critic. It derives a policy-implied value relation from KL-regularized optimality.

For an autoregressive state/action pair `(s_t, a_t)`, the central identity is:

```text
beta * log(pi*(a_t | s_t) / pi_ref(a_t | s_t))
  = r(s_t, a_t) + gamma * V*(s_{t+1}) - V*(s_t)
    + beta * KL(pi*(. | s_t) || pi_ref(. | s_t))
```

Rearranged:

```text
B_t = beta * log(pi*(a_t | s_t) / pi_ref(a_t | s_t))
      - beta * KL*(s_t)
      - r(s_t, a_t)

B_t = gamma * V_t+1 - V_t
```

Solving the recurrence forward gives a value implied by the policy log-ratio instead of a learned value network.

For final-reward-only RLVR with `gamma=1`, VIMPO's operational value loss is:

```text
L_V = 1/2 * [ sum_t (rho_t - sg[kappa_t]) - (R_final - group_mean(R_final)) ]^2

rho_t   = beta * log(pi_theta(a_t | s_t) / pi_ref(a_t | s_t))
kappa_t = beta * KL(pi_theta(. | s_t) || pi_ref(. | s_t))
```

The stop-gradient on `kappa_t` matters. It makes the KL term a centering baseline in the value loss, rather than giving the model a cheap way to reduce the loss by directly manipulating the full-distribution KL term.

## What We Adapted

The original VIMPO target is centered by `group_mean(R_final)`. In SSCA retry, the user-specified baseline is the paired first attempt:

```text
pi_ref behavior = prompt-only first attempt, traj1
pi_theta behavior = retry branch, summary + traj2
```

Therefore, the SSCA target is:

```text
branch_delta = reward(traj2) - reward(traj1)
branch_target = clip((branch_delta + margin) / reward_scale, -target_clip, target_clip)
```

Default values:

```text
margin = 0.1
reward_scale = 10.0
target_clip = 2.0
```

This is a paired-retry adaptation of VIMPO. It preserves the VIMPO core idea:

```text
cumulative policy-reference log-ratio should match an outcome target
```

but replaces the group-mean baseline with the paired first trajectory reward.

## Rollout Contract

For each ALFWorld task, SSCA generates:

```text
P = original ALFWorld task prompt / initial observation

traj1:
  state:  P + current ALFWorld observation/history
  action: ALFWorld command
  reward: r1 from environment

summary:
  state:  compact traj1 trace + outcome feedback + original task anchor
  action: generated retry memory, treated as a policy action
  downstream reward: r2 from traj2
  optional shaping: summary quality bonus during training only

traj2:
  state:  P + generated summary + current retry observation/history
  action: ALFWorld command
  reward: r2 from environment
```

Context rule:

```text
traj2 sees: original task prompt + generated summary
traj2 does not see: raw traj1 actions, raw traj1 observations, or raw invalid outputs
```

This keeps the summary as the information bottleneck. If `traj2` improves, the improvement must come through the generated summary memory.

## Summary Prompt Contract

The current prompt is the v3 prompt chosen from the prompt-loop review.

Target output format:

```text
Task: <target object and destination/receptacle>
Known: <stable scene facts discovered in attempt 1>
Attempt1: <why attempt 1 succeeded or failed>
Plan: <next concrete ALFWorld command-style plan>
Rule: <one safety rule to avoid invalid/repeated actions>
```

Constraints:

- Output only the five-line retry memory.
- Do not output markdown, XML, wrappers, or free-form analysis.
- Anchor to the original ALFWorld task.
- Use ALFWorld-like plan verbs: `go to`, `open`, `take`, `put`, `examine`, `inventory`, `look`.
- Stop after the `Rule` line.

The summary remains a sampled model action. It is not deterministically rewritten before being injected into `traj2`.

## Reward Handling

For one paired task:

```text
r1 = raw environment episode reward of traj1
r2 = raw environment episode reward of traj2
q  = summary quality score in [0, 1]
b  = summary quality coefficient, default 0.2 for training and 0 for validation
```

Policy rewards used by GRPO:

```text
traj1 rows:    episode_rewards = r1
summary row:   episode_rewards = r2 + b * q
traj2 rows:    episode_rewards = r2
```

Raw environment rewards are preserved separately:

```text
ssca_env_episode_rewards = r1 for traj1 rows
ssca_env_episode_rewards = r2 for summary/traj2 rows
```

The VIMPO-style value target uses raw environment reward only:

```text
branch_target = clip((r2 - r1 + margin) / reward_scale, -target_clip, target_clip)
```

It intentionally ignores the summary-quality bonus. This keeps the VIMPO value loss focused on actual retry improvement, while the GRPO reward path can still gently encourage parseable summaries.

## Target Distribution Across Rows

VIMPO's original terminal loss is trajectory-level:

```text
sum over all response tokens in the trajectory
```

In ALFWorld, a retry branch is stored as multiple generated rows:

```text
summary row + traj2 action row 1 + traj2 action row 2 + ...
```

The actor update operates row-wise/microbatch-wise, so the first safe implementation approximates the branch-level terminal sum by distributing the branch target across active retry rows.

Default:

```text
target_distribution = split_by_retry_rows
row_target = branch_target / number_of_active_retry_rows
```

This prevents long trajectories from receiving the full target repeatedly on every action row.

For debugging, both target levels are logged:

```text
episode/ssca_metric/vimpo/terminal_target_mean           # row target
episode/ssca_metric/vimpo/branch_terminal_target_mean    # original branch target
```

If we later want a stronger summary-specific objective, a possible extension is:

```text
summary row gets a separate summary-target term
traj2 rows share the branch target
```

The current version does not do this; it uses the split target for both summary and traj2 rows.

## Actor-side VIMPO Loss

The default actor-side loss is implemented in `recipe/SSCA/vimpo_core.py` and called by the actor update.

For each active summary/traj2 row:

```text
rho_t = beta * (log_prob_t - ref_log_prob_t)
kappa_t = beta * KL_estimator(log_prob_t, ref_log_prob_t)
term_t = rho_t - sg[kappa_t]
terminal_value_row = sum_t term_t over response tokens
```

The row-level loss is:

```text
L_row = 1/2 * (terminal_value_row - row_target)^2
```

By default it uses Huber form for numerical robustness:

```text
value_loss_type = huber
huber_delta = 1.0
```

The actor loss becomes:

```text
L_total = L_GRPO_policy + L_actor_KL + entropy_term + value_loss_coef * mean_active(L_row)
```

Default VIMPO actor settings:

```text
beta = 0.01
value_loss_coef = 0.05
kl_estimator = low_var_kl
detach_kl = True
value_loss_type = huber
huber_delta = 1.0
```

The implementation keeps the existing GRPO actor branch instead of replacing it with the VIMPO PPO-advantage branch. This was chosen because the request was to modify GRPO into the two-trajectory SSCA method, not to replace the actor objective entirely.

## Relation to Original VIMPO

### Aligned with VIMPO

- Uses a critic-free value loss.
- No learned critic and no default value head.
- Uses cumulative policy-reference log-ratio as the policy-implied terminal value.
- Uses stop-gradient KL in the value loss.
- External reward enters the value loss through an outcome target.
- PPO/GRPO actor loss is separate from the value loss.

### Adapted for SSCA

- Baseline is paired `traj1` reward rather than group mean reward.
- Target is retry improvement `r2 - r1 + margin`.
- Retry branch is multi-row, so the branch target is split across retry rows.
- `pi_ref` is operationally the paired prompt-only behavior baseline for the target, while the token-level log-ratio still uses the frozen reference model on the same generated state/action because tokenwise `pi_ref(a|s)` must be defined on the same state/action.

### Not an exact reproduction

This is not exact paper VIMPO in two ways:

1. The actor advantage remains GRPO rather than VIMPO's detached policy-implied PPO advantage.
2. The branch-level terminal loss is approximated row-wise because the current actor update backprops by microbatch rows.

These are deliberate engineering choices for a first stable ALFWorld implementation.

## Why Token-level `pi_ref` Is Still the Frozen Reference

The user-level statement was:

```text
traj2 policy pi_theta should be better than traj1 policy pi_ref
```

For reward targeting, we implement exactly that:

```text
target = reward(traj2) - reward(traj1)
```

For the VIMPO token log-ratio, however, `pi_ref(a_t | s_t)` must be evaluated at the same state and same sampled action as `pi_theta(a_t | s_t)`. `traj1` and `traj2` generally have different states and different actions, so there is no well-defined tokenwise log-probability of `traj2`'s action under the raw `traj1` trajectory.

Therefore:

- paired `traj1` is the outcome baseline;
- frozen reference model is the tokenwise log-ratio reference;
- the combination gives a well-defined and trainable SSCA/VIMPO objective.

## Code Flow

### `recipe/SSCA/rollout_loop.py`

Responsibilities:

- executes `traj1 -> summary -> traj2`;
- stores row metadata;
- applies summary quality bonus to the summary row during training;
- preserves raw environment rewards;
- logs retry, validity, and summary metrics.

Important row metadata:

```text
ssca_phase:
  "traj1", "summary", or "traj2"

traj_uid:
  unique id for the generated trajectory branch

ssca_linked_first_traj_uid:
  for summary/traj2 rows, points back to paired traj1

ssca_summary_is_policy_action:
  True only for summary rows

ssca_visible_first_trajectory:
  True only for summary rows

ssca_context_contract:
  describes what information the row is allowed to see

ssca_env_episode_rewards:
  raw r1 or r2 before summary-quality reward shaping

ssca_reward_bonus:
  nonzero only for summary rows during training
```

### `recipe/SSCA/vimpo_trainer.py`

Responsibilities:

- wraps the standard `RayPPOTrainer`;
- keeps original `compute_advantage(..., adv_estimator=grpo)` unchanged;
- attaches VIMPO target tensors after GRPO advantages are computed;
- logs target construction metrics under `ssca_metric/vimpo/...`.

Added tensors:

```text
ssca_vimpo_terminal_target:
  row-level target after splitting branch target across retry rows

ssca_vimpo_raw_terminal_target:
  unclipped row-level target

ssca_vimpo_branch_terminal_target:
  branch-level clipped target before row split

ssca_vimpo_raw_branch_terminal_target:
  branch-level raw target before clipping and row split

ssca_vimpo_reward_delta:
  raw r2 - r1

ssca_vimpo_loss_mask:
  1 for summary/traj2 rows, 0 for traj1 rows

ssca_vimpo_is_summary:
  1 for summary rows

ssca_vimpo_is_traj2:
  1 for traj2 action rows
```

### `recipe/SSCA/vimpo_core.py`

Responsibilities:

- checks actor-side SSCA VIMPO config;
- appends required batch keys to actor update selection;
- computes the critic-free VIMPO-style value loss;
- reports actor-side diagnostics.

Core computation:

```text
sampled_kl = low_var_kl(log_prob, ref_log_prob)
terminal_value = sum_t beta * (log_prob_t - ref_log_prob_t - stopgrad(sampled_kl_t))
loss = value_loss_coef * mean_active(huber(terminal_value - row_target))
```

### `verl/workers/actor/dp_actor.py`

Responsibilities:

- includes `ref_log_prob`, `ssca_vimpo_terminal_target`, and `ssca_vimpo_loss_mask` in actor update data when SSCA VIMPO is enabled;
- adds the SSCA VIMPO loss to `policy_loss`;
- logs actor-side VIMPO metrics.

### `verl/workers/actor/step_ppo_actor.py`

The same SSCA VIMPO hooks are present here for compatibility if the value-head worker is enabled manually. The full SSCA VIMPO script disables value head by default.

### `recipe/SSCA/main_ssca.py`

Responsibilities:

- selects `SSCAVIMPORayTrainer` when `algorithm.ssca_vimpo.enable=True`;
- uses the normal FSDP worker when value head is disabled;
- uses the StepPPO worker only if a value head is explicitly enabled.

### `verl/trainer/ppo/metric_utils.py`

Responsibilities:

- aggregates generic `ssca_metric/...` keys;
- fixes robust conversion for numpy/object scalar episode metrics;
- logs episode-level VIMPO target metrics.

### `verl/trainer/ppo/ray_trainer.py`

Responsibilities:

- validation metric aggregation includes `success_rate` and `ssca_metric/...` fields.

## Script Defaults

Main defaults in `run_ssca_retry_grpo_alfworld_1p5b_full.sh`:

```bash
SSCA_VIMPO_ENABLE=True
SSCA_VIMPO_REWARD_SCALE=10.0
SSCA_VIMPO_MARGIN=0.1
SSCA_VIMPO_TARGET_CLIP=2.0
SSCA_VIMPO_TARGET_DISTRIBUTION=split_by_retry_rows
SSCA_VIMPO_BETA=0.01
SSCA_VIMPO_VALUE_LOSS_COEF=0.05
SSCA_VIMPO_KL_ESTIMATOR=low_var_kl
SSCA_VIMPO_DETACH_KL=True
SSCA_VIMPO_VALUE_LOSS_TYPE=huber
SSCA_VIMPO_HUBER_DELTA=1.0
SSCA_VALUE_HEAD_ENABLE=False
SSCA_VALUE_LOSS_COEF=0.0
SSCA_VALUE_DETACH_BACKBONE=True
```

GRPO baseline alignment kept:

```bash
algorithm.adv_estimator=grpo
algorithm.use_kl_in_reward=False
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.01
actor_rollout_ref.actor.kl_loss_type=low_var_kl
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.model.use_remove_padding=True
actor_rollout_ref.model.enable_gradient_checkpointing=True
actor_rollout_ref.rollout.tensor_model_parallel_size=2
actor_rollout_ref.rollout.name=vllm
```

## Metrics

### Retry outcome metrics

```text
episode/ssca_metric/traj1/reward_mean
episode/ssca_metric/traj2/reward_mean
episode/ssca_metric/retry/reward_delta_mean
episode/ssca_metric/retry/reward_delta_std
episode/ssca_metric/retry/reward_improved_rate
episode/ssca_metric/retry/reward_degraded_rate
episode/ssca_traj1_success_rate
episode/ssca_traj2_success_rate
episode/ssca_retry_improved_success_rate
episode/ssca_retry_degraded_success_rate
```

Desired trend:

- `traj2` reward should exceed `traj1` reward.
- improved rate should exceed degraded rate.
- validation `traj2_success_rate` should not regress versus `traj1_success_rate`.

### Valid-action metrics

```text
episode/ssca_metric/traj1/valid_action_ratio
episode/ssca_metric/traj2/valid_action_ratio
episode/ssca_metric/traj1/invalid_count_mean
episode/ssca_metric/traj2/invalid_count_mean
```

Desired trend:

- retry memory should not induce invalid repeated commands;
- `traj2` valid action ratio should approach or exceed `traj1`.

### Summary quality metrics

```text
episode/ssca_metric/summary/quality_mean
episode/ssca_metric/summary/quality_std
episode/ssca_metric/summary/label_count_mean
episode/ssca_metric/summary/five_label_rate
episode/ssca_metric/summary/ordered_five_line_rate
episode/ssca_metric/summary/plan_ok_rate
episode/ssca_metric/summary/word_count_mean
episode/ssca_metric/summary/bonus_mean
episode/ssca_metric/summary/bonus_nonzero_rate
episode/ssca_metric/summary/quality_reward_corr
```

Desired trend:

- summary quality should not collapse;
- word count should remain bounded;
- format rates should improve with training if summary bonus is useful.

### VIMPO target construction metrics

```text
episode/ssca_metric/vimpo/terminal_target_mean
episode/ssca_metric/vimpo/terminal_target_std
episode/ssca_metric/vimpo/raw_terminal_target_mean
episode/ssca_metric/vimpo/raw_terminal_target_std
episode/ssca_metric/vimpo/branch_terminal_target_mean
episode/ssca_metric/vimpo/branch_terminal_target_std
episode/ssca_metric/vimpo/raw_branch_terminal_target_mean
episode/ssca_metric/vimpo/raw_branch_terminal_target_std
episode/ssca_metric/vimpo/reward_delta_mean
episode/ssca_metric/vimpo/reward_delta_std
episode/ssca_metric/vimpo/reward_delta_positive_rate
episode/ssca_metric/vimpo/reward_delta_negative_rate
episode/ssca_metric/vimpo/target_positive_rate
episode/ssca_metric/vimpo/value_loss_mask_rate
episode/ssca_metric/vimpo/summary_mask_rate
episode/ssca_metric/vimpo/traj2_mask_rate
episode/ssca_metric/vimpo/missing_ref_count
episode/ssca_metric/vimpo/target_clip_count
episode/ssca_metric/vimpo/reward_scale
episode/ssca_metric/vimpo/margin
episode/ssca_metric/vimpo/split_by_retry_rows
```

Must-have checks:

- `missing_ref_count = 0`
- `value_loss_mask_rate > 0`
- `summary_mask_rate > 0`
- `traj2_mask_rate > 0`
- `split_by_retry_rows = 1`

### Actor-side VIMPO loss metrics

```text
actor/ssca_vimpo_value_loss
actor/ssca_vimpo_value_loss_raw
actor/ssca_vimpo_value_loss_x1e4
actor/ssca_vimpo_value_loss_raw_x1e4
actor/ssca_vimpo_beta
actor/ssca_vimpo_value_loss_coef
actor/ssca_vimpo_active_row_ratio
actor/ssca_vimpo_active_row_count
actor/ssca_vimpo_active_token_count
actor/ssca_vimpo_terminal_value_mean_active
actor/ssca_vimpo_terminal_value_std_active
actor/ssca_vimpo_terminal_target_mean_active
actor/ssca_vimpo_terminal_target_std_active
actor/ssca_vimpo_terminal_target_abs_mean_active
actor/ssca_vimpo_residual_mean_active
actor/ssca_vimpo_residual_std_active
actor/ssca_vimpo_rmse_active
actor/ssca_vimpo_mae_active
actor/ssca_vimpo_value_target_corr_active
actor/ssca_vimpo_sign_acc_active
actor/ssca_vimpo_target_positive_rate_active
actor/ssca_vimpo_value_positive_rate_active
actor/ssca_vimpo_logratio_token_mean_active
actor/ssca_vimpo_kl_token_mean_active
actor/ssca_vimpo_token_term_sum_mean_active
```

The `x1e4` metrics are present because early losses are small and the console logger rounds to three decimals.

## Smoke Results

### Smoke 1

Run:

```text
ssca_vimpo_smoke1_20260626_225801
```

Result:

```text
failed before training
AssertionError: real_train_batch_size (2) must be divisible by total n_gpus (4)
```

Fix:

```text
use TRAIN_BATCH_SIZE=4 for 4 GPUs
```

### Smoke 2

Run:

```text
ssca_vimpo_smoke2_seed2026_ssca_vimpo_smoke2_20260626_231037
```

Result:

- initialized Ray/FSDP/vLLM;
- ran initial validation;
- attached SSCA VIMPO tensors;
- failed in `metric_utils` because an object/float episode metric was assumed to support `.item()`.

Fix:

```text
metric_utils now converts episode arrays through robust numpy float helpers
```

### Smoke 3

Run:

```text
ssca_vimpo_smoke3_seed2026_ssca_vimpo_smoke3_20260626_231532
```

Configuration:

```text
TOTAL_EPOCHS=2
VAL_BEFORE_TRAIN=False
TEST_FREQ=-1
MAX_ENV_STEPS=10
```

Result:

- completed 2 training steps;
- actor update ran;
- VIMPO tensors were present in the training batch;
- VIMPO actor metrics appeared;
- rollout generations were dumped.

Key metrics at step 2:

```text
actor/ssca_vimpo_active_row_ratio: 0.545
actor/ssca_vimpo_rmse: 0.013
actor/ssca_vimpo_mae: 0.011
episode/ssca_metric/vimpo/missing_ref_count: 0.000
episode/ssca_metric/vimpo/value_loss_mask_rate: 0.528
episode/ssca_metric/vimpo/summary_mask_rate: 0.051
episode/ssca_metric/vimpo/traj2_mask_rate: 0.477
```

### Smoke 4

Run:

```text
ssca_vimpo_smoke4_val_seed2026_ssca_vimpo_smoke4_val_20260626_231957
```

Configuration:

```text
TOTAL_EPOCHS=1
VAL_BEFORE_TRAIN=True
TEST_FREQ=1
MAX_ENV_STEPS=8
```

Result:

- completed training step;
- ran validation before and after training;
- actor VIMPO diagnostics appeared;
- validation SSCA metrics appeared.

Key metrics:

```text
actor/ssca_vimpo_value_loss_x1e4: 0.016
actor/ssca_vimpo_value_loss_raw_x1e4: 0.318
actor/ssca_vimpo_terminal_target_mean_active: 0.010
actor/ssca_vimpo_rmse_active: 0.011
episode/ssca_metric/vimpo/missing_ref_count: 0.000
val/ssca_metric/traj1/valid_action_ratio: 1.000
val/ssca_metric/traj2/valid_action_ratio: 1.000
```

### Smoke 5

Run:

```text
ssca_vimpo_smoke5_split_seed2026_ssca_vimpo_smoke5_split_20260626_232505
```

Configuration:

```text
TOTAL_EPOCHS=1
VAL_BEFORE_TRAIN=False
TEST_FREQ=-1
MAX_ENV_STEPS=6
SSCA_VIMPO_TARGET_DISTRIBUTION=split_by_retry_rows
```

Result:

- completed training step after target splitting change;
- confirmed row target and branch target are both logged;
- confirmed split mode is active.

Key metrics:

```text
actor/ssca_vimpo_value_loss_x1e4: 0.003
actor/ssca_vimpo_value_loss_raw_x1e4: 0.055
actor/ssca_vimpo_terminal_target_mean_active: 0.001
actor/ssca_vimpo_terminal_target_abs_mean_active: 0.001
episode/ssca_metric/vimpo/terminal_target_mean: 0.001
episode/ssca_metric/vimpo/branch_terminal_target_mean: 0.010
episode/ssca_metric/vimpo/missing_ref_count: 0.000
episode/ssca_metric/vimpo/split_by_retry_rows: 1.000
```

The post-run message below appears during Ray/DataLoader teardown after successful completion and did not change the process exit code:

```text
RuntimeError: DataLoader worker (...) is killed by signal: Killed.
```

For smoke runs, the shell process exited with code `0` after completed training. This appears to be a shutdown/cleanup artifact, not a training failure.

## Full Run Command

Launch one full SSCA VIMPO run:

```bash
bash /prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent/scripts/run_ssca_retry_grpo_alfworld_1p5b_full.sh
```

The script writes:

```text
logs/<experiment_name>_<timestamp>.log
EXPS/analysis/024_ssca_retry_full/<experiment_name>/rollout_generations/*.jsonl
```

Default full settings:

```text
TRAIN_DATA_SIZE=16
VAL_DATA_SIZE=128
TRAIN_BATCH_SIZE=16
GROUP_SIZE=8
MAX_ENV_STEPS=50
TOTAL_EPOCHS=150
TEST_FREQ=5
SAVE_FREQ=-1
RAY_NUM_CPUS=60
ENV_WORKER_CPUS=0.1
```

## Full Run Monitoring Checklist

At startup:

```text
actor_rollout_ref.actor.ssca_vimpo.enable=True
actor_rollout_ref.actor.value_head.enable=False
algorithm.ssca_vimpo.target_distribution=split_by_retry_rows
```

At first training step:

```text
actor/ssca_vimpo_value_loss_x1e4 is present
actor/ssca_vimpo_active_row_ratio > 0
episode/ssca_metric/vimpo/missing_ref_count = 0
episode/ssca_metric/vimpo/split_by_retry_rows = 1
episode/ssca_metric/vimpo/value_loss_mask_rate > 0
```

During training:

```text
episode/ssca_metric/retry/reward_delta_mean
episode/ssca_metric/retry/reward_improved_rate
episode/ssca_metric/traj2/valid_action_ratio
actor/ssca_vimpo_rmse_active
actor/ssca_vimpo_sign_acc_active
actor/ssca_vimpo_logratio_token_mean_active
val/ssca_metric/retry/reward_delta_mean
val/ssca_metric/traj2/valid_action_ratio
```

Potential red flags:

```text
missing_ref_count > 0
active_row_ratio = 0
summary_mask_rate = 0
traj2 valid action ratio collapses
actor/grad_norm becomes non-finite
actor/ssca_vimpo_value_loss_x1e4 explodes
```

## Tuning Options

If VIMPO signal is too weak:

```bash
SSCA_VIMPO_VALUE_LOSS_COEF=0.1
SSCA_VIMPO_BETA=0.02
```

If VIMPO signal is too strong or destabilizes policy:

```bash
SSCA_VIMPO_VALUE_LOSS_COEF=0.01
SSCA_VIMPO_BETA=0.005
```

If reward deltas are frequently clipped:

```bash
SSCA_VIMPO_TARGET_CLIP=4.0
```

If row-splitting makes summary supervision too weak:

```bash
SSCA_VIMPO_TARGET_DISTRIBUTION=per_row_full
```

This last option is less faithful to trajectory-level VIMPO and can overweight long retry branches, so it is not the default.

## Remaining Limitations

1. The current actor branch remains GRPO, not VIMPO's policy-implied PPO advantage.
2. The branch terminal value is approximated by row-level losses, not one exact loss over all tokens in `summary + traj2`.
3. The token-level log-ratio reference is the frozen reference model, while paired `traj1` is used as the reward baseline.
4. Early smoke rewards are mostly zero, so the target is dominated by the small positive margin. Longer full runs are needed to see nonzero reward deltas.
5. The summary format is still imperfect; prompt-quality metrics should be watched closely.

## Review Conclusion

The previous value-head prototype was not sufficiently aligned with VIMPO. The current implementation is better aligned because the default training signal is now critic-free and uses the VIMPO terminal log-ratio value loss.

The implementation is ready for a full ALFWorld run, with the caveat that this is an SSCA adaptation of VIMPO rather than an exact reproduction of the paper algorithm.
