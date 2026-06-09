#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent"
NEW_BATCH_SCRIPT="$ROOT_DIR/exps/run_webshop_step_ppo_v1_v0_grpo_seed2026.sh"

if [ ! -x "$NEW_BATCH_SCRIPT" ]; then
  echo "[FATAL] Missing executable script: $NEW_BATCH_SCRIPT" >&2
  exit 1
fi

echo "[WARN] This legacy batch entry now delegates to the seed-2026 v1/v0/GRPO plan."
exec bash "$NEW_BATCH_SCRIPT" "$@"
