#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PLATFORM_HOME="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan"
export HOME="${RUN_HOME:-$PLATFORM_HOME}"
export PYTHONUSERBASE="${PYTHONUSERBASE:-$HOME/.local}"
export PATH="${PYTHON_BIN_DIR:-/opt/conda/bin}:$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.11/site-packages:$ROOT_DIR:${PYTHONPATH:-}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$XDG_CACHE_HOME/alfworld}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

BACKEND="${BACKEND:-hf}"
MODEL="${MODEL:-azure::gpt-5.4-mini}"
MODEL_PATH="${MODEL_PATH:-}"
MERGED_MODEL_PATH="${MERGED_MODEL_PATH:-}"
SPLITS="${SPLITS:-train,test}"
TARGET_SUCCESS="${TARGET_SUCCESS:-16}"
TARGET_FAILURE="${TARGET_FAILURE:-16}"
ENV_NUM="${ENV_NUM:-8}"
MAX_STEPS="${MAX_STEPS:-50}"
MAX_ATTEMPT_BATCHES="${MAX_ATTEMPT_BATCHES:-80}"
SEED="${SEED:-2026}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
ENV_WORKER_CPUS="${ENV_WORKER_CPUS:-0.2}"
MAX_WORKERS="${MAX_WORKERS:-2}"
HF_BATCH_SIZE="${HF_BATCH_SIZE:-4}"
ACTION_MAX_TOKENS="${ACTION_MAX_TOKENS:-160}"
TEMPERATURE="${TEMPERATURE:-0.4}"
TOP_P="${TOP_P:-0.95}"
DTYPE="${DTYPE:-bf16}"
EVAL_DATASET="${EVAL_DATASET:-eval_in_distribution}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/EXPS/alfworld_effective_summary/outputs/prev_traj_${BACKEND}_succ${TARGET_SUCCESS}_fail${TARGET_FAILURE}_seed${SEED}_${RUN_TAG}}"

mkdir -p "$OUT_DIR"

if [[ ! -d "$ALFWORLD_DATA/json_2.1.1" ]]; then
  echo "[FATAL] Missing ALFWorld data under $ALFWORLD_DATA/json_2.1.1" >&2
  exit 1
fi

if command -v ray >/dev/null 2>&1; then
  ray stop --force >/dev/null 2>&1 || true
fi


if [[ "$BACKEND" == "hf" && -n "$MODEL_PATH" ]]; then
  if [[ -d "$MODEL_PATH/actor" && -f "$MODEL_PATH/actor/config.json" ]]; then
    MODEL_PATH="$MODEL_PATH/actor"
  fi
  if compgen -G "$MODEL_PATH/model_world_size_*_rank_0.pt" >/dev/null; then
    if [[ -z "$MERGED_MODEL_PATH" ]]; then
      safe_name="$(echo "$MODEL_PATH" | sed 's#^/##; s#[/:]#_#g')"
      MERGED_MODEL_PATH="$ROOT_DIR/EXPS/alfworld_effective_summary/outputs/merged_models/${safe_name}_hf"
    fi
    if [[ ! -f "$MERGED_MODEL_PATH/config.json" || ! -f "$MERGED_MODEL_PATH/model.safetensors" ]]; then
      mkdir -p "$(dirname "$MERGED_MODEL_PATH")"
      echo "[INFO] Merging verl FSDP actor checkpoint to HF: $MODEL_PATH -> $MERGED_MODEL_PATH"
      python3 scripts/model_merger.py merge --backend fsdp --local_dir "$MODEL_PATH" --target_dir "$MERGED_MODEL_PATH"
    else
      echo "[INFO] Reusing merged HF model: $MERGED_MODEL_PATH"
    fi
    MODEL_PATH="$MERGED_MODEL_PATH"
  fi
fi

ARGS=(
  --backend "$BACKEND"
  --model "$MODEL"
  --splits "$SPLITS"
  --target-success "$TARGET_SUCCESS"
  --target-failure "$TARGET_FAILURE"
  --env-num "$ENV_NUM"
  --max-steps "$MAX_STEPS"
  --max-attempt-batches "$MAX_ATTEMPT_BATCHES"
  --seed "$SEED"
  --ray-num-cpus "$RAY_NUM_CPUS"
  --worker-cpus "$ENV_WORKER_CPUS"
  --max-workers "$MAX_WORKERS"
  --hf-batch-size "$HF_BATCH_SIZE"
  --action-max-tokens "$ACTION_MAX_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --dtype "$DTYPE"
  --eval-dataset "$EVAL_DATASET"
  --out-dir "$OUT_DIR"
)
if [[ -n "$MODEL_PATH" ]]; then
  ARGS+=(--model-path "$MODEL_PATH")
fi

printf '[INFO] OUT_DIR=%s\n' "$OUT_DIR"
printf '[INFO] BACKEND=%s MODEL=%s MODEL_PATH=%s MERGED_MODEL_PATH=%s\n' "$BACKEND" "$MODEL" "${MODEL_PATH:-<auto/base>}" "${MERGED_MODEL_PATH:-<auto>}"
printf '[INFO] SPLITS=%s TARGET_SUCCESS=%s TARGET_FAILURE=%s ENV_NUM=%s MAX_STEPS=%s\n' "$SPLITS" "$TARGET_SUCCESS" "$TARGET_FAILURE" "$ENV_NUM" "$MAX_STEPS"

python3 EXPS/alfworld_effective_summary/scripts/collect_prev_trajectories.py "${ARGS[@]}"
