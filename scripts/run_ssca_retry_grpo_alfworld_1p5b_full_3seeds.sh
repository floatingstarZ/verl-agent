#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_SCRIPT="$ROOT_DIR/scripts/run_ssca_retry_grpo_alfworld_1p5b_full.sh"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
SEED_LIST="${SEED_LIST:-2026 2077 2501}"
BASE_EXPERIMENT="${BASE_EXPERIMENT:-ssca_retry_grpo_qwen25_15b_alfworld_full}"

if [[ ! -x "$BASE_SCRIPT" ]]; then
  echo "[FATAL] Missing executable base script: $BASE_SCRIPT" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

echo "[INFO] ROOT_DIR=$ROOT_DIR"
echo "[INFO] BASE_SCRIPT=$BASE_SCRIPT"
echo "[INFO] SEED_LIST=$SEED_LIST"
echo "[INFO] BASE_EXPERIMENT=$BASE_EXPERIMENT"
echo "[INFO] LOG_DIR=$LOG_DIR"

for seed in $SEED_LIST; do
  run_tag="${RUN_TAG_PREFIX:-full}_seed${seed}_$(date +%Y%m%d_%H%M%S)"
  experiment_name="${BASE_EXPERIMENT}_seed${seed}"
  log_file="$LOG_DIR/${experiment_name}_$(date +%Y%m%d_%H%M%S).log"

  echo "============================================================"
  echo "[RUN] seed=${seed} experiment=${experiment_name}"
  echo "[LOG] ${log_file}"
  echo "============================================================"

  RUN_TAG="$run_tag" \
  SEED="$seed" \
  BASE_EXPERIMENT="$BASE_EXPERIMENT" \
  EXPERIMENT_NAME="$experiment_name" \
  LOG_FILE="$log_file" \
  LOG_DIR="$LOG_DIR" \
  bash "$BASE_SCRIPT" "$@"

  echo "[INFO] ===== Finished seed $seed: $experiment_name ====="
done

echo "[INFO] All SSCA retry GRPO ALFWorld seeds finished: $SEED_LIST"
