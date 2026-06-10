#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
VALUE_DIAGNOSTICS_DIR="${VALUE_DIAGNOSTICS_DIR:-logs/value_diagnostics/step_ppo_v3_value_head_detach_false_qwen2.5_1.5b_4gpu_paper_align_full_e250_resume_${RUN_TAG}}"

cd "$ROOT_DIR"

echo "[INFO] Resume StepPPO-v3 value-head-only run."
echo "[INFO] Experiment resumes from the latest checkpoint automatically."
echo "[INFO] Value diagnostics: $VALUE_DIAGNOSTICS_DIR"

exec bash exps/run_webshop_step_ppo_v3_value_head_4gpu_paper_align_simple.sh \
  trainer.val_before_train=False \
  trainer.save_freq=25 \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.max_critic_ckpt_to_keep=2 \
  trainer.value_diagnostics_dir="$VALUE_DIAGNOSTICS_DIR" \
  "$@"
