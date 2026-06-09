#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent"
SEED="${SEED:-2026}"
SMOKE_STEPS="${SMOKE_STEPS:-}"

DETACH_FALSE_SCRIPT="$ROOT_DIR/exps/run_webshop_gigpo_v1_value_aux_detach_false_4gpu_paper_align_simple.sh"
DETACH_TRUE_SCRIPT="$ROOT_DIR/exps/run_webshop_gigpo_v1_value_aux_detach_true_4gpu_paper_align_simple.sh"
GRPO_SCRIPT="$ROOT_DIR/exps/run_webshop_grpo_4gpu_paper_align_simple.sh"

for script in "$DETACH_FALSE_SCRIPT" "$DETACH_TRUE_SCRIPT" "$GRPO_SCRIPT"; do
  if [ ! -x "$script" ]; then
    echo "[FATAL] Missing executable script: $script" >&2
    exit 1
  fi
done

cd "$ROOT_DIR"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

name_with_mode() {
  local base="$1"
  if [ -n "$SMOKE_STEPS" ]; then
    printf '%s_seed%s_smoke_s%s' "$base" "$SEED" "$SMOKE_STEPS"
  else
    printf '%s_seed%s' "$base" "$SEED"
  fi
}

DETACH_FALSE_EXPERIMENT="$(name_with_mode 'gigpo_v1_value_aux_detach_false_qwen2.5_1.5b_4gpu_paper_align_full_e250')"
DETACH_TRUE_EXPERIMENT="$(name_with_mode 'gigpo_v1_value_aux_detach_true_qwen2.5_1.5b_4gpu_paper_align_full_e250')"
GRPO_EXPERIMENT="$(name_with_mode 'grpo_qwen2.5_1.5b_4gpu_paper_align_full_e250')"

COMMON_OVERRIDES=(
  "env.seed=${SEED}"
)

SMOKE_OVERRIDES=()
if [ -n "$SMOKE_STEPS" ]; then
  SMOKE_OVERRIDES=(
    "trainer.total_training_steps=${SMOKE_STEPS}"
    "trainer.total_epochs=1"
    "trainer.val_before_train=False"
    "trainer.test_freq=-1"
    "trainer.save_freq=-1"
    "trainer.logger=['console']"
    "data.train_batch_size=4"
    "data.val_batch_size=8"
    "actor_rollout_ref.actor.ppo_mini_batch_size=4"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
    "env.rollout.n=4"
    "ray_init.num_cpus=32"
  )
fi

echo "[ENV] GiGPO value-aux ablation + GRPO batch: seed=${SEED} smoke_steps=${SMOKE_STEPS:-none}"
echo "[ORDER] value_aux_detach_false -> value_aux_detach_true -> grpo"

run_one() {
  local label="$1"
  local script="$2"
  local exp_name="$3"
  shift 3

  echo "============================================================"
  echo "[RUN] method=${label} seed=${SEED} experiment=${exp_name}"
  echo "============================================================"
  bash "$script" \
    "trainer.experiment_name=${exp_name}" \
    "${COMMON_OVERRIDES[@]}" \
    "${SMOKE_OVERRIDES[@]}" \
    "$@"
}

run_one "GiGPO-v1-value-aux-detach-false" "$DETACH_FALSE_SCRIPT" "$DETACH_FALSE_EXPERIMENT" "$@"
run_one "GiGPO-v1-value-aux-detach-true" "$DETACH_TRUE_SCRIPT" "$DETACH_TRUE_EXPERIMENT" "$@"
run_one "GRPO" "$GRPO_SCRIPT" "$GRPO_EXPERIMENT" "$@"

echo "[DONE] Finished value-aux detach_false, value-aux detach_true, and GRPO for seed=${SEED}"
