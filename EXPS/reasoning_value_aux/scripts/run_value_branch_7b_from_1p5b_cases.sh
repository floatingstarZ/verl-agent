#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PLATFORM_HOME="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan"
export HOME="${RUN_HOME:-$PLATFORM_HOME}"
export PYTHONUSERBASE="${PYTHONUSERBASE:-$HOME/.local}"
export PATH="${PYTHON_BIN_DIR:-/opt/conda/bin}:$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

INPUT_JSON="${INPUT_JSON:-$ROOT_DIR/EXPS/reasoning_value_aux/outputs/value_case_viz/reason_value_case_viz_10val_fixed_20260702_055406/reason_value_cases.json}"
MODEL_PATH="${MODEL_PATH:-$HF_HOME/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-value_branch_qwen25_7b_from_1p5b_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/EXPS/reasoning_value_aux/outputs/value_case_viz/$EXPERIMENT_NAME}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${EXPERIMENT_NAME}.log}"
TP_SIZE="${TP_SIZE:-4}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.72}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LIMIT_CASES="${LIMIT_CASES:-0}"
LIMIT_STEPS="${LIMIT_STEPS:-0}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [[ ! -f "$INPUT_JSON" ]]; then
  echo "[FATAL] Missing INPUT_JSON=$INPUT_JSON" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "[FATAL] Missing Qwen2.5-7B model config at MODEL_PATH=$MODEL_PATH" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/recipe/GraphGPO/setup_flash_attn_env.sh"
graphgpo_setup_flash_attn_sm90

echo "[INFO] INPUT_JSON=$INPUT_JSON"
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] LOG_FILE=$LOG_FILE"
echo "[INFO] TP_SIZE=$TP_SIZE MAX_TOKENS=$MAX_TOKENS LIMIT_CASES=$LIMIT_CASES LIMIT_STEPS=$LIMIT_STEPS"

python3 EXPS/reasoning_value_aux/scripts/rerun_value_branch_from_cases.py \
  --input-json "$INPUT_JSON" \
  --output-dir "$OUTPUT_DIR" \
  --model-path "$MODEL_PATH" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-tokens "$MAX_TOKENS" \
  --batch-size "$BATCH_SIZE" \
  --limit-cases "$LIMIT_CASES" \
  --limit-steps "$LIMIT_STEPS" \
  "$@" 2>&1 | tee "$LOG_FILE"
