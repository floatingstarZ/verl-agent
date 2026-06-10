#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent"
BASE_SCRIPT="$ROOT_DIR/exps/run_webshop_gigpo_v1_value_aux_detach_false_4gpu_paper_align_simple.sh"
DEFAULT_EXPERIMENT="step_ppo_v3_value_head_detach_false_qwen2.5_1.5b_4gpu_paper_align_full_e250"
DEFAULT_VALUE_DIAGNOSTICS_DIR="logs/value_diagnostics/$DEFAULT_EXPERIMENT"

if [ ! -x "$BASE_SCRIPT" ]; then
  echo "[FATAL] Missing executable script: $BASE_SCRIPT" >&2
  exit 1
fi

cd "$ROOT_DIR"
export GIGPO_V1_VALUE_AUX_LOG_LABEL="step_ppo_v3_value_head_detach_false"
echo "[INFO] StepPPO-v3 scope: value head only; GiGPO policy objective unchanged."
echo "[INFO] detach_value_backbone=False: value loss can update the selected-state backbone path."

exec bash "$BASE_SCRIPT" \
  algorithm.adv_estimator=gigpo \
  algorithm.gigpo.step_advantage_w=1.0 \
  algorithm.gigpo.mode=mean_norm \
  algorithm.gigpo_v1_value_aux.value_target=gae_returns \
  algorithm.gigpo_v1_value_aux.step_gamma=0.95 \
  algorithm.gigpo_v1_value_aux.step_lam=0.90 \
  actor_rollout_ref.actor.value_head.enable=True \
  actor_rollout_ref.actor.value_head.detach_value_backbone=False \
  actor_rollout_ref.actor.value_head.value_loss_coef=0.03 \
  actor_rollout_ref.actor.value_head.cliprange_value=0.5 \
  trainer.experiment_name="$DEFAULT_EXPERIMENT" \
  'trainer.logger=[console]' \
  trainer.value_diagnostics_dir="$DEFAULT_VALUE_DIAGNOSTICS_DIR" \
  trainer.value_diagnostics_interval=1 \
  trainer.value_diagnostics_max_rows=512 \
  trainer.value_diagnostics_include_text=False \
  "$@"
