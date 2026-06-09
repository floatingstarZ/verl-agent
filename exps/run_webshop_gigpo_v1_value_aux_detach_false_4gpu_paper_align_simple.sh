#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent"
BASE_SCRIPT="$ROOT_DIR/examples/gigpo_v1_value_aux_trainer/run_webshop.sh"
DEFAULT_EXPERIMENT="gigpo_v1_value_aux_detach_false_qwen2.5_1.5b_4gpu_paper_align_full_e250"
LOG_LABEL="${GIGPO_V1_VALUE_AUX_LOG_LABEL:-detach_false}"

cd "$ROOT_DIR"

export PATH="/opt/conda/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
export VERL_AGENT_CACHE_ROOT="${VERL_AGENT_CACHE_ROOT:-/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/.cache}"
export VERL_AGENT_DATA_ROOT="${VERL_AGENT_DATA_ROOT:-/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/data/verl-agent}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER:-$(id -un)}/.triton-cache}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-/tmp/${USER:-$(id -un)}/.vllm}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/${USER:-$(id -un)}}"
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

cleanup_ray() {
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup_ray EXIT INT TERM

GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
if echo "$GPU_NAMES" | head -n 4 | rg -q 'H100'; then
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

cleanup_ray
mkdir -p logs

bash "$BASE_SCRIPT" vllm \
  actor_rollout_ref.rollout.enforce_eager=True \
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
  actor_rollout_ref.actor.value_head.detach_value_backbone=False \
  actor_rollout_ref.actor.value_head.value_loss_coef=0.03 \
  trainer.experiment_name="$DEFAULT_EXPERIMENT" \
  trainer.val_before_train=True \
  trainer.total_epochs=250 \
  trainer.test_freq=5 \
  trainer.save_freq=50 \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  ray_init.num_cpus=64 \
  env.resources_per_worker.num_cpus=0.1 \
  "$@" \
  2>&1 | tee "logs/webshop_gigpo_v1_value_aux_${LOG_LABEL}_qwen25_15b_4gpu_paper_align_simple_$(date +%Y%m%d_%H%M%S).log"
