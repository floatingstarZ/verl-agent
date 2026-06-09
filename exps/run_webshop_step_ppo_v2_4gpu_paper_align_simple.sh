#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent"
BASE_SCRIPT="$ROOT_DIR/exps/run_webshop_step_ppo_4gpu_paper_align_simple.sh"
DEFAULT_EXPERIMENT="step_ppo_v2_qwen2.5_1.5b_4gpu_paper_align_full_e250"

if [ ! -x "$BASE_SCRIPT" ]; then
  echo "[FATAL] Missing executable script: $BASE_SCRIPT" >&2
  exit 1
fi

cd "$ROOT_DIR"
exec bash "$BASE_SCRIPT" \
  algorithm.step_ppo.step_advantage_w=0.5 \
  algorithm.step_ppo.episode_mode=mean_norm \
  algorithm.step_ppo.final_advantage_mode=direct \
  algorithm.step_ppo.normalize_episode_advantage=False \
  algorithm.step_ppo.normalize_step_advantage=True \
  algorithm.step_ppo.normalize_final_advantage=False \
  actor_rollout_ref.actor.value_head.detach_value_backbone=False \
  actor_rollout_ref.actor.value_head.value_loss_coef=0.03 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  trainer.experiment_name="$DEFAULT_EXPERIMENT" \
  "$@"
