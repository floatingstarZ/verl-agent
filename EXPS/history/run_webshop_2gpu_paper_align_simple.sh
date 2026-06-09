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
export PYTHON_BIN="/opt/conda/bin/python"
export FLASH_ATTN_SM90_SITE="${FLASH_ATTN_SM90_SITE:-/tmp/flash_attn_sm80_sm90_site}"
FLASH_ATTN_SM90_EXT="$(find "$FLASH_ATTN_SM90_SITE" -maxdepth 1 -name 'flash_attn_2_cuda*.so' 2>/dev/null | head -1 || true)"
if [ -z "$FLASH_ATTN_SM90_EXT" ]; then
  echo "[FATAL] Missing sm_90 flash-attn override under $FLASH_ATTN_SM90_SITE" >&2
  echo "[FATAL] Build/copy flash_attn_2_cuda*.so there before running this H100 script." >&2
  exit 1
fi
if command -v cuobjdump >/dev/null 2>&1 && ! cuobjdump --list-elf "$FLASH_ATTN_SM90_EXT" | awk '/sm_90/{found=1} END{exit !found}'; then
  echo "[FATAL] $FLASH_ATTN_SM90_EXT does not contain sm_90 kernels" >&2
  exit 1
fi
export PYTHONPATH="$FLASH_ATTN_SM90_SITE:${PYTHONPATH:-}"
export TRITON_CACHE_DIR="/tmp/${USER:-$(id -un)}/.triton-cache"
export XDG_CACHE_HOME="/tmp/${USER:-$(id -un)}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export VLLM_CONFIG_ROOT="/tmp/${USER:-$(id -un)}/.vllm"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" "$VLLM_CONFIG_ROOT"

VISIBLE_GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
if [ "${VISIBLE_GPU_COUNT:-0}" -lt "$EXPECTED_GPUS" ]; then
  echo "[FATAL] Need >=${EXPECTED_GPUS} visible GPUs, got ${VISIBLE_GPU_COUNT:-0}" >&2
  exit 1
fi

ray stop --force || true

ulimit -n 65536 || true
export WANDB_MODE=offline
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
  awk 'NR==3{print "shift || true"} {print}' examples/gigpo_trainer/run_webshop.sh \
  | sed \
      -e 's|^python3 -m |/opt/conda/bin/python -m |' \
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
  trainer.experiment_name=gigpo_qwen2.5_1.5b_2gpu_paper_align_simple \
  trainer.val_before_train=False \
  trainer.total_epochs=1 \
  trainer.test_freq=-1 \
  ray_init.num_cpus=32 \
  env.resources_per_worker.num_cpus=0.1 \
  "$@" \
  2>&1 | tee "logs/webshop_gigpo_qwen25_15b_2gpu_paper_align_simple_$(date +%Y%m%d_%H%M%S).log"
