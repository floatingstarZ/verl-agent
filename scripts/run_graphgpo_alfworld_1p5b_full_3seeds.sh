#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_SCRIPT="$ROOT_DIR/scripts/run_graphgpo_alfworld_1p5b_full.sh"
BASE_EXPERIMENT="${BASE_EXPERIMENT:-graphgpo_qwen25_15b_alfworld_full}"
SEED_LIST="${SEED_LIST:-2026 2077 2501}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"

if [[ ! -x "$BASE_SCRIPT" ]]; then
  echo "[FATAL] Missing executable script: $BASE_SCRIPT" >&2
  exit 1
fi

cd "$ROOT_DIR"
mkdir -p "$LOG_DIR"

echo "[ENV] Batch target: GraphGPO ALFWorld Qwen2.5-1.5B, seeds=${SEED_LIST}"
echo "[ENV] BASE_SCRIPT=$BASE_SCRIPT"
echo "[ENV] LOG_DIR=$LOG_DIR"

for seed in $SEED_LIST; do
  exp_name="${BASE_EXPERIMENT}_seed${seed}"
  log_file="$LOG_DIR/${exp_name}_$(date +%Y%m%d_%H%M%S).log"

  echo "============================================================"
  echo "[RUN] seed=${seed} experiment=${exp_name}"
  echo "[LOG] ${log_file}"
  echo "============================================================"

  SEED="$seed" \
  EXPERIMENT_NAME="$exp_name" \
  LOG_FILE="$log_file" \
  bash "$BASE_SCRIPT" "$@"

done

echo "[DONE] All seed runs finished: ${SEED_LIST}"
