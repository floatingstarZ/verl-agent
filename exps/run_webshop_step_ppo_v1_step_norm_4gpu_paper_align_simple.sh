#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent"
BASE_SCRIPT="$ROOT_DIR/exps/run_webshop_step_ppo_4gpu_paper_align_simple.sh"
DEFAULT_EXPERIMENT="step_ppo_v1_step_norm_qwen2.5_1.5b_4gpu_paper_align_full_e250"

if [ ! -x "$BASE_SCRIPT" ]; then
  echo "[FATAL] Missing executable script: $BASE_SCRIPT" >&2
  exit 1
fi

cd "$ROOT_DIR"
exec bash "$BASE_SCRIPT" \
  algorithm.step_ppo.normalize_step_advantage=True \
  trainer.experiment_name="$DEFAULT_EXPERIMENT" \
  "$@"
