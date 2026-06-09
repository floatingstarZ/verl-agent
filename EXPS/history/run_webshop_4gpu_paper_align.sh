#!/usr/bin/env bash
set -euo pipefail

source /prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/miniconda3/etc/profile.d/conda.sh
conda activate verl-agent
cd /prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent

ray stop --force || true
mkdir -p logs

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
      -e 's/export VLLM_ATTENTION_BACKEND=XFORMERS/export VLLM_ATTENTION_BACKEND=FLASH_ATTN/' \
      -e 's/actor_rollout_ref.rollout.enforce_eager=False/actor_rollout_ref.rollout.enforce_eager=True/'
) vllm \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.experiment_name=gigpo_qwen2.5_1.5b_4gpu_paper_align \
  ray_init.num_cpus=32 \
  env.resources_per_worker.num_cpus=0.1 \
  2>&1 | tee logs/webshop_gigpo_qwen25_15b_4gpu_paper_align_$(date +%Y%m%d_%H%M%S).log
