#!/usr/bin/env bash
set -euo pipefail

# Batch runner target environment: H100
ROOT_DIR="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent"
BASE_SCRIPT="$ROOT_DIR/exps/run_webshop_4gpu_paper_align_simple.sh"
BASE_EXPERIMENT="gigpo_qwen2.5_1.5b_4gpu_paper_align_full_e250"
SEEDS=(2026 2077 2501)

if [ ! -x "$BASE_SCRIPT" ]; then
  echo "[FATAL] Missing executable script: $BASE_SCRIPT" >&2
  exit 1
fi

cd "$ROOT_DIR"
echo "[ENV] Batch target: H100 (sm_90)"

for seed in "${SEEDS[@]}"; do
  exp_name="${BASE_EXPERIMENT}_seed${seed}"
  echo "============================================================"
  echo "[RUN] seed=${seed} experiment=${exp_name}"
  echo "============================================================"
  bash "$BASE_SCRIPT" \
    env.seed="${seed}" \
    trainer.experiment_name="${exp_name}"
done

echo "[DONE] All seed runs finished: ${SEEDS[*]}"
