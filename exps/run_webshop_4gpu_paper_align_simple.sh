#!/usr/bin/env bash
set -euo pipefail

# Runtime environment target: H100 (sm_90)
ROOT_DIR="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent"
PYTHON_BIN="/opt/conda/bin/python"
EXPECTED_GPUS=4

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[FATAL] Missing python: $PYTHON_BIN" >&2
  exit 1
fi

cd "$ROOT_DIR"
mkdir -p logs

export PATH="/opt/conda/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHON_BIN="/opt/conda/bin/python"

cleanup_ray() {
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup_ray EXIT INT TERM

# H100 flash-attn override (sm_90).
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

export TRITON_CACHE_DIR="/tmp/${USER:-$(id -un)}/.triton-cache"
export XDG_CACHE_HOME="/tmp/${USER:-$(id -un)}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
export VLLM_CONFIG_ROOT="/tmp/${USER:-$(id -un)}/.vllm"
mkdir -p "$TRITON_CACHE_DIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" "$VLLM_CONFIG_ROOT"

VISIBLE_GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
if [ "${VISIBLE_GPU_COUNT:-0}" -lt "$EXPECTED_GPUS" ]; then
  echo "[FATAL] Need >=${EXPECTED_GPUS} visible GPUs, got ${VISIBLE_GPU_COUNT:-0}" >&2
  exit 1
fi
if ! nvidia-smi --query-gpu=name --format=csv,noheader | head -n "$EXPECTED_GPUS" | rg -q 'H100'; then
  echo "[FATAL] This script is for H100 runtime; non-H100 GPU detected." >&2
  nvidia-smi --query-gpu=name --format=csv,noheader >&2 || true
  exit 1
fi
echo "[ENV] Runtime target confirmed: H100 (sm_90), GPUs=$VISIBLE_GPU_COUNT"

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
  awk 'NR==3{print "shift || true"} {print}' examples/gigpo_trainer/run_webshop.sh \
  | sed \
      -e 's|^python3 -m |/opt/conda/bin/python -m |' \
      -e 's/export VLLM_ATTENTION_BACKEND=XFORMERS/export VLLM_ATTENTION_BACKEND=FLASH_ATTN/' \
      -e 's/actor_rollout_ref.rollout.enforce_eager=False/actor_rollout_ref.rollout.enforce_eager=True/'
) vllm \
  data.train_batch_size=16 \
  data.val_batch_size=128 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  env.rollout.n=8 \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  trainer.experiment_name=gigpo_qwen2.5_1.5b_4gpu_paper_align_full_e250 \
  trainer.val_before_train=True \
  trainer.total_epochs=250 \
  trainer.test_freq=5 \
  trainer.save_freq=50 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  ray_init.num_cpus=64 \
  env.resources_per_worker.num_cpus=0.1 \
  "$@" \
  2>&1 | tee "logs/webshop_gigpo_qwen25_15b_4gpu_paper_align_simple_$(date +%Y%m%d_%H%M%S).log"
