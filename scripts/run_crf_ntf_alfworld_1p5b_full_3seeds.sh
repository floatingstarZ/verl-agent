#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEEDS=(${SEEDS:-2026 2077 2501})
BASE_EXPERIMENT_PREFIX="${BASE_EXPERIMENT_PREFIX:-crf_ntf_qwen25_15b_alfworld_full}"

for seed in "${SEEDS[@]}"; do
  echo "[INFO] Starting C-RF NTF ALFWorld seed=$seed"
  SEED="$seed" \
  EXPERIMENT_NAME="${BASE_EXPERIMENT_PREFIX}_seed${seed}" \
  bash "$ROOT_DIR/scripts/run_crf_ntf_alfworld_1p5b_full.sh" "$@"
  echo "[INFO] Finished C-RF NTF ALFWorld seed=$seed"
done
