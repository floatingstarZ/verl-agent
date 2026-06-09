#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent"
PYTHON_BIN="/opt/conda/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[FATAL] Missing python: $PYTHON_BIN" >&2
  exit 1
fi

cd "$ROOT_DIR"
mkdir -p logs

export PATH="/opt/conda/bin:$PATH"
export PYTHONNOUSERSITE=1

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
      -e '/examples\.data_preprocess\.prepare/,+3d' \
      -e 's|^python3 -m |/opt/conda/bin/python -m |' \
      -e 's/export VLLM_ATTENTION_BACKEND=XFORMERS/export VLLM_ATTENTION_BACKEND=FLASH_ATTN/' \
      -e 's/actor_rollout_ref.rollout.enforce_eager=False/actor_rollout_ref.rollout.enforce_eager=True/'
) vllm \
  data.train_batch_size=2 \
  data.val_batch_size=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  env.rollout.n=2 \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  trainer.experiment_name=gigpo_qwen2.5_1.5b_1gpu_paper_align_simple \
  trainer.val_before_train=False \
  trainer.total_epochs=1 \
  trainer.test_freq=-1 \
  ray_init.num_cpus=16 \
  env.resources_per_worker.num_cpus=0.1 \
  "$@" \
  2>&1 | tee "logs/webshop_gigpo_qwen25_15b_1gpu_paper_align_simple_$(date +%Y%m%d_%H%M%S).log"
