#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent"
SEED=2026
V1_SCRIPT="$ROOT_DIR/exps/run_webshop_step_ppo_v1_step_norm_4gpu_paper_align_simple.sh"
V0_SCRIPT="$ROOT_DIR/exps/run_webshop_step_ppo_v0_base_4gpu_paper_align_simple.sh"
GRPO_SCRIPT="$ROOT_DIR/exps/run_webshop_grpo_4gpu_paper_align_simple.sh"

V1_EXPERIMENT="step_ppo_v1_step_norm_qwen2.5_1.5b_4gpu_paper_align_full_e250_seed${SEED}"
V0_EXPERIMENT="step_ppo_v0_base_qwen2.5_1.5b_4gpu_paper_align_full_e250_seed${SEED}"
GRPO_EXPERIMENT="grpo_qwen2.5_1.5b_4gpu_paper_align_full_e250_seed${SEED}"

for script in "$V1_SCRIPT" "$V0_SCRIPT" "$GRPO_SCRIPT"; do
  if [ ! -x "$script" ]; then
    echo "[FATAL] Missing executable script: $script" >&2
    exit 1
  fi
done

cd "$ROOT_DIR"
echo "[ENV] StepPPO version/GRPO batch target: H100 (sm_90), seed=${SEED}"
echo "[ORDER] v1_step_norm:${SEED} v0_base:${SEED} grpo:${SEED}"

run_one() {
  local label="$1"
  local script="$2"
  local exp_name="$3"
  shift 3

  echo "============================================================"
  echo "[RUN] method=${label} seed=${SEED} experiment=${exp_name}"
  echo "============================================================"
  bash "$script" \
    env.seed="${SEED}" \
    trainer.experiment_name="${exp_name}" \
    "$@"
}

run_one "StepPPO-v1_step_norm" "$V1_SCRIPT" "$V1_EXPERIMENT" "$@"
run_one "StepPPO-v0_base" "$V0_SCRIPT" "$V0_EXPERIMENT" "$@"
run_one "GRPO" "$GRPO_SCRIPT" "$GRPO_EXPERIMENT" "$@"

echo "[DONE] Finished v1_step_norm, v0_base, and GRPO for seed=${SEED}"
