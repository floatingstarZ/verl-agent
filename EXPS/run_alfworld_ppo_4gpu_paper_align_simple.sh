#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent"
BASE_SCRIPT="$ROOT_DIR/scripts/run_ppo_alfworld_1p5b_full.sh"

if [[ ! -x "$BASE_SCRIPT" ]]; then
  echo "[FATAL] Missing executable base script: $BASE_SCRIPT" >&2
  exit 1
fi

cd "$ROOT_DIR"

export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-128}"
export VAL_DATA_SIZE="${VAL_DATA_SIZE:-128}"
export GROUP_SIZE="${GROUP_SIZE:-1}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-150}"
export TEST_FREQ="${TEST_FREQ:-5}"
export SAVE_FREQ="${SAVE_FREQ:--1}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
export ENV_WORKER_CPUS="${ENV_WORKER_CPUS:-0.1}"

echo "[INFO] ALFWorld PPO baseline: GAE + separate critic."
echo "[INFO] Sample budget aligns with GRPO/GiGPO 16x8 by using 128 independent PPO envs."

exec bash "$BASE_SCRIPT" "$@"
