#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PLATFORM_HOME="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan"
export HOME="${RUN_HOME:-$PLATFORM_HOME}"
export PYTHONUSERBASE="${PYTHONUSERBASE:-$HOME/.local}"
export PATH="${PYTHON_BIN_DIR:-/opt/conda/bin}:$PYTHONUSERBASE/bin:$PATH"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$XDG_CACHE_HOME/torch}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$XDG_CACHE_HOME/alfworld}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}"
export WANDB_DIR="${WANDB_DIR:-$HOME/wandb}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

ulimit -n 65536 || true
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MALLOC_CONF="${MALLOC_CONF:-background_thread:false,narenas:1,dirty_decay_ms:0,muzzy_decay_ms:0}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-150}"
TEST_FREQ="${TEST_FREQ:-5}"
SAVE_FREQ="${SAVE_FREQ:--1}"
SEED="${SEED:-0}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
ENV_WORKER_CPUS="${ENV_WORKER_CPUS:-0.1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-graphgpo_qwen25_15b_alfworld_full_seed${SEED}}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$HOME/checkpoints}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
ENGINE="${ENGINE:-vllm}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log}"

if [[ "${ENABLE_WANDB:-0}" == "1" ]]; then
  LOGGER_OVERRIDE="trainer.logger=['console','wandb']"
  export WANDB_MODE="${WANDB_MODE:-online}"
else
  LOGGER_OVERRIDE="trainer.logger=['console']"
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi

mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" \
  "$ALFWORLD_DATA" "$MPLCONFIGDIR" "$WANDB_DIR" "$CHECKPOINTS_DIR" "$LOG_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[FATAL] python3 not found. Set PYTHON_BIN_DIR, e.g. PYTHON_BIN_DIR=/opt/conda/bin" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[FATAL] nvidia-smi not found; this full run requires CUDA GPUs." >&2
  exit 1
fi

visible_gpu_count="$N_GPUS_PER_NODE"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  visible_gpu_count="$(awk -F',' '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
fi
if (( visible_gpu_count < N_GPUS_PER_NODE )); then
  echo "[FATAL] Need at least $N_GPUS_PER_NODE visible GPUs, got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

if [[ ! -d "$ALFWORLD_DATA/json_2.1.1" ]]; then
  echo "[FATAL] Missing ALFWorld data under $ALFWORLD_DATA/json_2.1.1" >&2
  echo "[FATAL] Run: alfworld-download --data-dir '$ALFWORLD_DATA'" >&2
  exit 1
fi

cleanup_ray() {
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}
if [[ "${CLEAN_RAY:-1}" == "1" ]]; then
  cleanup_ray
  trap cleanup_ray EXIT INT TERM
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/recipe/GraphGPO/setup_flash_attn_env.sh"
graphgpo_setup_flash_attn_sm90

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  python3 - <<'PY'
import sys
import flash_attn, flash_attn_2_cuda
print(f"[INFO] python={sys.executable}")
print(f"[INFO] flash_attn={flash_attn.__file__}")
print(f"[INFO] flash_attn_2_cuda={flash_attn_2_cuda.__file__}")
PY
else
  echo "[INFO] DRY_RUN=1; skipping flash-attn Python import check"
fi

echo "[INFO] ROOT_DIR=$ROOT_DIR"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] N_GPUS_PER_NODE=$N_GPUS_PER_NODE"
echo "[INFO] ALFWORLD_DATA=$ALFWORLD_DATA"
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[INFO] TOTAL_EPOCHS=$TOTAL_EPOCHS TEST_FREQ=$TEST_FREQ SAVE_FREQ=$SAVE_FREQ"
echo "[INFO] RAY_NUM_CPUS=$RAY_NUM_CPUS ENV_WORKER_CPUS=$ENV_WORKER_CPUS"
echo "[INFO] LOG_FILE=$LOG_FILE"
echo "[INFO] LOGGER=$LOGGER_OVERRIDE WANDB_MODE=$WANDB_MODE"

CMD=(bash recipe/GraphGPO/run_qwen2.5_1.5b_alfworld_train.sh "$ENGINE")
OVERRIDES=(
  "actor_rollout_ref.model.path=$MODEL_PATH"
  "actor_rollout_ref.model.attn_implementation=flash_attention_2"
  "trainer.experiment_name=$EXPERIMENT_NAME"
  "trainer.default_local_dir=$CHECKPOINTS_DIR/$EXPERIMENT_NAME"
  "trainer.n_gpus_per_node=$N_GPUS_PER_NODE"
  "trainer.total_epochs=$TOTAL_EPOCHS"
  "trainer.test_freq=$TEST_FREQ"
  "trainer.save_freq=$SAVE_FREQ"
  "trainer.val_before_train=True"
  "$LOGGER_OVERRIDE"
  "env.seed=$SEED"
  "ray_init.num_cpus=$RAY_NUM_CPUS"
  "env.resources_per_worker.num_cpus=$ENV_WORKER_CPUS"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[DRY_RUN] '; printf '%q ' "${CMD[@]}" "${OVERRIDES[@]}" "$@"; printf '\n'
  exit 0
fi

"${CMD[@]}" "${OVERRIDES[@]}" "$@" 2>&1 | tee "$LOG_FILE"
