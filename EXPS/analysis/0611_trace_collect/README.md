# 0611 Full RL Trace Collection

This folder contains a resume-and-run trace collection script for Soft-GiGPO state matching analysis.

The script runs several real RL training steps from a checkpoint and enables `trainer.rl_trace_dir`. For every traced global step it writes:

- `rl_trace/step_<global_step>/batch.pkl`: full `DataProto` after rollout, reward, actor old logprob, reference logprob, value prediction, GiGPO advantage, and value target construction, before actor update.
- `rl_trace/step_<global_step>/metadata.json`: tensor keys, shapes, dtypes, non-tensor keys, meta info, and trace settings.
- `rl_trace/step_<global_step>/rows.jsonl`: per-row readable view with ids, non-tensor rollout metadata, decoded prompt/response, scalar values/returns/rewards, token ids, token logprobs/probs, rewards, advantages, and masks.
- `value_diagnostics/step_<global_step>.jsonl`: value-head target/prediction/error rows from the same traced batch.
- `rollout_generations/<global_step>.jsonl`: compact prompt/response/score generation dump from the same rollout.

`batch.pkl` is the source of truth for offline analysis. `rows.jsonl` is meant for quick inspection and can be truncated with `trainer.rl_trace_token_array_max_items`; the binary `DataProto` still keeps full tensors.

## Usage

Minimal 2-step run from the default checkpoint:

```bash
RUN_TAG=20260611_smoke TRACE_STEPS=2 \
EXPS/analysis/0611_trace_collect/run_collect_full_rl_trace.sh
```

Low-cost smoke run with shorter WebShop horizon and truncated JSON arrays:

```bash
RUN_TAG=20260611_2step_smoke TRACE_STEPS=2 MAX_ENV_STEPS=3 \
GPU_MEMORY_UTILIZATION=0.3 \
EXPS/analysis/0611_trace_collect/run_collect_full_rl_trace.sh \
trainer.rl_trace_token_array_max_items=32
```

Set `DO_UPDATE=0` to collect traces without actor updates. The default `DO_UPDATE=1` simulates the later RL process: each traced batch is dumped immediately before the normal actor update, then training continues to the next rollout. Saving and validation are disabled.

## Collected Fields

The current hook preserves these important tensor fields when present:

- Token-level rollout/actor/reference probabilities: `rollout_log_probs`, `old_log_probs`, `ref_log_prob`, plus JSON convenience arrays `rollout_probs`, `old_probs`, and `ref_prob`.
- Value and target fields: `state_values`, `step_returns`, `value_step_rewards`, `step_rewards`.
- Reward/advantage fields: `token_level_scores`, `token_level_rewards`, `advantages`, `returns`.
- Token/mask fields: `prompts`, `responses`, `input_ids`, `attention_mask`, `position_ids`, `response_mask`.
- Optional actor entropy: `actor_entropys`, enabled by `trainer.rl_trace_keep_entropy=True`.
- Rollout metadata: `uid`, `traj_uid`, `anchor_obs`, `raw_prompt`, `is_action_valid`, `rewards`, `active_masks`, episode metrics, `step_idx`, and `step_sample_uid`.

## Validation

Validated on H100 with the default `global_step_250` StepPPO-v3 value-head checkpoint.

- `20260611_1step_smoke_fixed`: completed one resumed training step, dumped `step_000251`, then ran the actor update.
- `20260611_2step_smoke_fixed`: completed two resumed training steps with updates, dumped `step_000251` and `step_000252`.
- Each smoke step produced `batch.pkl`, `metadata.json`, `rows.jsonl`, value diagnostics, rollout generations, and console metrics including `old_log_prob`, `ref`, `adv`, `dump_rl_trace`, and `update_actor` timings.
- A representative traced batch has 384 rollout rows with 512 response tokens; full logprob/reference/value/reward/advantage tensors are present in `batch.pkl`.

Example quick check:

```bash
python - <<'PY'
from verl import DataProto
trace = 'EXPS/analysis/0611_trace_collect/full_rl_trace_20260611_2step_smoke_fixed/rl_trace/step_000252/batch.pkl'
batch = DataProto.load_from_disk(trace)
print(list(batch.batch.keys()))
for key in ['rollout_log_probs', 'old_log_probs', 'ref_log_prob', 'state_values', 'step_returns', 'token_level_rewards', 'advantages', 'returns']:
    value = batch.batch[key]
    print(key, tuple(value.shape), value.dtype)
print(list(batch.non_tensor_batch.keys()))
PY
```
