#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_SCRIPT="$ROOT_DIR/scripts/run_webshop_istar_4gpu_gigpo_align_full.sh"
BASE_EXPERIMENT="istar_rloo_qwen25_15b_4gpu_gigpo_align_e250"
SEED_LIST="${SEED_LIST:-2026 2077 2501}"

if [[ ! -x "$BASE_SCRIPT" ]]; then
  echo "[FATAL] Missing executable script: $BASE_SCRIPT" >&2
  exit 1
fi

cd "$ROOT_DIR"
mkdir -p logs

cleanup_ray() {
  if [[ -x /opt/conda/bin/ray ]]; then
    /opt/conda/bin/ray stop --force >/dev/null 2>&1 || true
  elif command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}

trap cleanup_ray EXIT INT TERM

echo "[ENV] Batch target: WebShop iStar 4GPU, seeds=${SEED_LIST}"

for seed in $SEED_LIST; do
  exp_name="${BASE_EXPERIMENT}_seed${seed}"
  log_file="logs/webshop_istar_qwen25_15b_4gpu_gigpo_align_e250_seed${seed}_$(date +%Y%m%d_%H%M%S).log"

  echo "============================================================"
  echo "[RUN] seed=${seed} experiment=${exp_name}"
  echo "[LOG] ${log_file}"
  echo "============================================================"

  cleanup_ray
  SEED="$seed" EXPERIMENT_NAME="$exp_name" bash "$BASE_SCRIPT" "$@" 2>&1 | tee "$log_file"
  cleanup_ray

done

echo "[DONE] All seed runs finished: ${SEED_LIST}"
