#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

N_GPUS_PER_NODE=1 \
TENSOR_MODEL_PARALLEL_SIZE=1 \
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gigpo_qwen2.5_1.5b_1gpu_paper_align_opt}" \
bash exps/run_webshop_4gpu_paper_align_opt.sh "$@"
