#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent"
BASE_SCRIPT="$ROOT_DIR/exps/run_webshop_gigpo_v1_value_aux_detach_true_4gpu_paper_align_simple.sh"
DEFAULT_EXPERIMENT="step_ppo_v3_value_head_detach_true_qwen2.5_1.5b_4gpu_paper_align_full_e250"

if [ ! -x "$BASE_SCRIPT" ]; then
  echo "[FATAL] Missing executable script: $BASE_SCRIPT" >&2
  exit 1
fi

cd "$ROOT_DIR"
echo "[INFO] StepPPO-v3 scope: value head only; GiGPO policy training unchanged."
echo "[INFO] Detached value head learns step GAE returns without updating the actor backbone."

exec bash "$BASE_SCRIPT" \
  algorithm.adv_estimator=gigpo \
  algorithm.gigpo.step_advantage_w=1.0 \
  algorithm.gigpo.mode=mean_norm \
  algorithm.gigpo_v1_value_aux.value_target=gae_returns \
  algorithm.gigpo_v1_value_aux.step_gamma=0.95 \
  algorithm.gigpo_v1_value_aux.step_lam=0.90 \
  actor_rollout_ref.actor.value_head.enable=True \
  actor_rollout_ref.actor.value_head.detach_value_backbone=True \
  actor_rollout_ref.actor.value_head.value_loss_coef=0.03 \
  actor_rollout_ref.actor.value_head.cliprange_value=0.5 \
  trainer.experiment_name="$DEFAULT_EXPERIMENT" \
  "$@"
