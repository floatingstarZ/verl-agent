#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent"
BASE_SCRIPT="$ROOT_DIR/exps/run_webshop_gigpo_v1_value_aux_detach_false_4gpu_paper_align_simple.sh"
DEFAULT_EXPERIMENT="gigpo_v1_value_aux_detach_true_qwen2.5_1.5b_4gpu_paper_align_full_e250"

cd "$ROOT_DIR"
export GIGPO_V1_VALUE_AUX_LOG_LABEL="detach_true"
exec bash "$BASE_SCRIPT" \
  actor_rollout_ref.actor.value_head.detach_value_backbone=True \
  trainer.experiment_name="$DEFAULT_EXPERIMENT" \
  "$@"
