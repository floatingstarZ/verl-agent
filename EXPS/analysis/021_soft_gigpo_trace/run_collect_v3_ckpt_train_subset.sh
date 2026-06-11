#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
CKPT="${CKPT:-checkpoints/verl_agent_webshop/step_ppo_v3_value_head_detach_false_qwen2.5_1.5b_4gpu_paper_align_full_e250/global_step_250}"
TRACE_DIR="${TRACE_DIR:-EXPS/analysis/021_soft_gigpo_trace/v3_ckpt250_train_subset_${RUN_TAG}}"
LOG_FILE="logs/021_soft_gigpo_trace/collect_v3_ckpt250_train_subset_${RUN_TAG}.log"
SKIP_DATALOADER_STATE="${SKIP_DATALOADER_STATE:-1}"

mkdir -p "$TRACE_DIR" "$(dirname "$LOG_FILE")" /tmp/ziyuhuan

step_name="$(basename "$CKPT")"
if [[ ! "$step_name" =~ ^global_step_([0-9]+)$ ]]; then
  echo "[ERROR] CKPT must end with global_step_<N>: $CKPT" >&2
  exit 1
fi
global_step="${BASH_REMATCH[1]}"
total_training_steps="${TOTAL_TRAINING_STEPS:-$((global_step + 1))}"
resume_ckpt="$CKPT"
tmp_resume_dir=""

if [[ "$SKIP_DATALOADER_STATE" == "1" ]]; then
  tmp_resume_dir="$(mktemp -d /tmp/ziyuhuan/soft_gigpo_trace_ckpt.XXXXXX)"
  resume_ckpt="$tmp_resume_dir/$step_name"
  mkdir -p "$resume_ckpt"
  ln -s "$(realpath "$CKPT/actor")" "$resume_ckpt/actor"
  if [[ -d "$CKPT/critic" ]]; then
    ln -s "$(realpath "$CKPT/critic")" "$resume_ckpt/critic"
  fi
  trap 'rm -rf "$tmp_resume_dir"' EXIT
fi

echo "[INFO] source_ckpt=$CKPT"
echo "[INFO] resume_ckpt=$resume_ckpt"
echo "[INFO] skip_dataloader_state=$SKIP_DATALOADER_STATE"
echo "[INFO] trace_dir=$TRACE_DIR"
echo "[INFO] log=$LOG_FILE"
echo "[INFO] total_training_steps=$total_training_steps"
echo "[INFO] Running one train batch from a trained v3 checkpoint; actor update is disabled by critic_warmup."

bash exps/run_webshop_step_ppo_v3_value_head_4gpu_paper_align_simple.sh \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$resume_ckpt" \
  trainer.experiment_name=soft_gigpo_trace_v3_ckpt250_train_subset \
  trainer.val_before_train=False \
  trainer.total_training_steps="$total_training_steps" \
  trainer.total_epochs=1 \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.critic_warmup=999999 \
  trainer.max_actor_ckpt_to_keep=0 \
  trainer.max_critic_ckpt_to_keep=0 \
  trainer.value_diagnostics_dir="$TRACE_DIR/value_diagnostics" \
  trainer.value_diagnostics_interval=1 \
  trainer.value_diagnostics_max_rows=0 \
  trainer.value_diagnostics_include_text=True \
  trainer.rollout_data_dir="$TRACE_DIR/rollout_generations" \
  trainer.logger='[console]' \
  "$@" \
  2>&1 | tee "$LOG_FILE"
