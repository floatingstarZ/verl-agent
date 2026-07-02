#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PLATFORM_HOME="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan"
export HOME="${RUN_HOME:-$PLATFORM_HOME}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export PYTHONPATH="$HOME/.local/lib/python3.11/site-packages:$ROOT_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL="${MODEL:-azure::gpt-5.4-mini}"
NUM_TASKS="${NUM_TASKS:-10}"
MAX_STEPS="${MAX_STEPS:-50}"
MAX_WORKERS="${MAX_WORKERS:-2}"
SEED="${SEED:-2026}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
ENV_WORKER_CPUS="${ENV_WORKER_CPUS:-0.2}"
SUMMARY_CONTEXT_MODE="${SUMMARY_CONTEXT_MODE:-adaptive}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/EXPS/alfworld_effective_summary/outputs/traj_refine_train${NUM_TASKS}_${MODEL//[:\/]/_}_seed${SEED}_${RUN_TAG}}"

mkdir -p "$OUT_DIR"

echo "[INFO] ROOT_DIR=$ROOT_DIR"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] MODEL=$MODEL NUM_TASKS=$NUM_TASKS MAX_STEPS=$MAX_STEPS MAX_WORKERS=$MAX_WORKERS SEED=$SEED SUMMARY_CONTEXT_MODE=$SUMMARY_CONTEXT_MODE"
echo "[INFO] ALFWORLD_DATA=$ALFWORLD_DATA"

python3 EXPS/alfworld_effective_summary/scripts/run_traj_refine_eval.py \
  --num-tasks "$NUM_TASKS" \
  --seed "$SEED" \
  --max-steps "$MAX_STEPS" \
  --model "$MODEL" \
  --max-workers "$MAX_WORKERS" \
  --ray-num-cpus "$RAY_NUM_CPUS" \
  --worker-cpus "$ENV_WORKER_CPUS" \
  --summary-context-mode "$SUMMARY_CONTEXT_MODE" \
  --out-dir "$OUT_DIR"
