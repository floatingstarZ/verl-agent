#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent"
PYTHON_BIN="/opt/conda/bin/python"
EXPECTED_GPUS=2

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[FATAL] Missing python: $PYTHON_BIN" >&2
  exit 1
fi

cd "$ROOT_DIR"
mkdir -p logs

export PATH="/opt/conda/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHON_BIN="$PYTHON_BIN"
# Keep the smoke run isolated to two GPUs unless the caller explicitly sets a device list.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

cleanup_ray() {
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup_ray EXIT INT TERM

GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
if echo "$GPU_NAMES" | head -n "$EXPECTED_GPUS" | rg -q 'H100'; then
  export FLASH_ATTN_SM90_SITE="${FLASH_ATTN_SM90_SITE:-/tmp/flash_attn_sm80_sm90_site}"
  FLASH_ATTN_SM90_EXT="$(find "$FLASH_ATTN_SM90_SITE" -maxdepth 1 -name 'flash_attn_2_cuda*.so' 2>/dev/null | head -1 || true)"
  if [ -z "$FLASH_ATTN_SM90_EXT" ]; then
    echo "[FATAL] Missing sm_90 flash-attn override under $FLASH_ATTN_SM90_SITE" >&2
    exit 1
  fi
  if command -v cuobjdump >/dev/null 2>&1 && ! cuobjdump --list-elf "$FLASH_ATTN_SM90_EXT" | awk '/sm_90/{found=1} END{exit !found}'; then
    echo "[FATAL] $FLASH_ATTN_SM90_EXT does not contain sm_90 kernels" >&2
    exit 1
  fi
  export PYTHONPATH="$FLASH_ATTN_SM90_SITE:${PYTHONPATH:-}"
fi

export TRITON_CACHE_DIR="/tmp/${USER:-$(id -un)}/.triton-cache"
export XDG_CACHE_HOME="/tmp/${USER:-$(id -un)}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export VLLM_CONFIG_ROOT="/tmp/${USER:-$(id -un)}/.vllm"
export FLASHINFER_WORKSPACE_BASE="/tmp/${USER:-$(id -un)}"
mkdir -p "$TRITON_CACHE_DIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" "$VLLM_CONFIG_ROOT" "$FLASHINFER_WORKSPACE_BASE"

VISIBLE_GPU_COUNT="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')"
if [ "${VISIBLE_GPU_COUNT:-0}" -lt "$EXPECTED_GPUS" ]; then
  echo "[FATAL] Need >=${EXPECTED_GPUS} visible GPUs in CUDA_VISIBLE_DEVICES, got ${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 1
fi

echo "[ENV] Step-PPO smoke target: GPUs=$CUDA_VISIBLE_DEVICES"
ray stop --force || true

ulimit -n 65536 || true
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export MALLOC_CONF=background_thread:false,narenas:1,dirty_decay_ms:0,muzzy_decay_ms:0

bash <(
  awk 'NR==3{print "shift || true"} {print}' examples/step_ppo_trainer/run_webshop.sh \
  | sed \
      -e 's/export VLLM_ATTENTION_BACKEND=XFORMERS/export VLLM_ATTENTION_BACKEND=FLASH_ATTN/' \
      -e 's/actor_rollout_ref.rollout.enforce_eager=False/actor_rollout_ref.rollout.enforce_eager=True/'
) vllm \
  data.train_batch_size=4 \
  data.val_batch_size=8 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  env.rollout.n=4 \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  trainer.experiment_name=step_ppo_qwen2.5_1.5b_2gpu_smoke \
  trainer.val_before_train=False \
  trainer.total_epochs=1 \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  ray_init.num_cpus=32 \
  env.resources_per_worker.num_cpus=0.1 \
  "$@" \
  2>&1 | tee "logs/webshop_step_ppo_qwen25_15b_2gpu_smoke_$(date +%Y%m%d_%H%M%S).log"
