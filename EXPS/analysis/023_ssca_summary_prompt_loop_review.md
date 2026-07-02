# SSCA Summary Prompt Loop Review

Date: 2026-06-26

This review ran three prompt -> rollout -> trajectory review iterations on the same small ALFWorld SSCA smoke setup.

## Runs

| Version | Run directory | Prompt change | Summary strict-clean pass | Traj1 valid ratio | Traj2 valid ratio | Invalid delta |
|---|---|---|---:|---:|---:|---:|
| v1 | `EXPS/analysis/022_ssca_retry_smoke/ssca_retry_grpo_qwen25_15b_alfworld_smoke_seed2026_ssca_promptloop_v1_20260626_145148` | 5-line RETRY_MEMORY, compact trace | 0/9 | 0.810 | 0.756 | +3 |
| v2 | `EXPS/analysis/022_ssca_retry_smoke/ssca_retry_grpo_qwen25_15b_alfworld_smoke_seed2026_ssca_promptloop_v2_20260626_145601` | harder fill-template, raw invalid response removed | 2/9 | 0.829 | 0.761 | +4 |
| v3 | `EXPS/analysis/022_ssca_retry_smoke/ssca_retry_grpo_qwen25_15b_alfworld_smoke_seed2026_ssca_promptloop_v3_20260626_145956` | task-target anchoring and ALFWorld command-style plan verbs | 3/12 | 0.805 | 0.884 | -3 |

Strict-clean means: correct five labels, no markdown/wrapper, <=110 words, no guessing/system-talk, no multi-action connector in the plan, and no non-command plan verb.

## Observations

- v1 summaries were short but often collapsed to only a `Retry`/`first` line, so the requested schema did not hold.
- v1 also induced invalid or mixed plans such as `inspect countertop 1 or drawer 1`, and traj2 valid ratio dropped below traj1.
- v2 improved label following, but still produced `check/inspect/explore`-style plans and occasional hallucinated or generic rule text.
- Removing raw invalid model responses from compact trace reduced direct priming from malformed Chinese / bad XML outputs.
- v3 had the best trajectory effect: traj2 valid ratio improved to 0.884 versus traj1 0.805, despite imperfect summary formatting.

## Kept Version

The code keeps v3 because the current priority is summary impact on traj2 behavior, not only summary prettiness.

Main retained design:

- Compact trace only, with no raw malformed response excerpt by default.
- Five-line safe retry memory schema: `Task`, `Known`, `Attempt1`, `Plan`, `Rule`.
- The target object and receptacle are explicitly anchored to the task.
- `Plan` is constrained to ALFWorld-style verbs: `go to`, `open`, `take`, `put`, `examine`, `inventory`, `look`.
- Summary trace limits are reduced: max steps 6, obs/feedback 240 chars, initial prompt 1200 chars.

## Remaining Risk

v3 still has imperfect schema adherence. Several summaries copied trace fragments or became too long. The next non-prompt-only improvement should be a lightweight summary quality gate or deterministic postprocessor before injecting memory into traj2.
