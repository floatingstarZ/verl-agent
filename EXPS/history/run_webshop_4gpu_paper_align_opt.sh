#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/opt/conda/bin/python"
WRAPPER_PATH="/tmp/verl-agent-run/run_webshop_paper_align_opt_wrapper.sh"
LOCAL_PY_OVERLAY="$ROOT_DIR/.deps/opt_py311_overlay"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "[FATAL] Missing expected python: $PYTHON_BIN" >&2
    exit 1
fi

cd "$ROOT_DIR"
mkdir -p logs /tmp/verl-agent-run

# Hard-pin runtime to /opt/conda to avoid accidentally using a user/local conda base.
export CONDA_PYTHON="$PYTHON_BIN"
export PYTHONNOUSERSITE=1
export PATH="/opt/conda/bin:$PATH"
if [ -d "$LOCAL_PY_OVERLAY" ]; then
    export PYTHONPATH="$LOCAL_PY_OVERLAY:${PYTHONPATH:-}"
fi

# Validate the critical stack before starting a long training job.
"$PYTHON_BIN" - <<'PY'
import sys

mods = ["torch", "ray", "vllm", "flash_attn", "setuptools"]
print("[env] python", sys.version.split()[0], "exe", sys.executable)
for name in mods:
    m = __import__(name)
    print(f"[env] {name}", getattr(m, "__version__", "n/a"))
PY

ray stop --force || true

ulimit -n 65536 || true
export WANDB_MODE="${WANDB_MODE:-offline}"
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

awk 'NR==3{print "shift || true"} {print}' examples/gigpo_trainer/run_webshop.sh \
    | sed \
        -e '/examples\.data_preprocess\.prepare/,+3d' \
        -e 's|^python3 -m |/opt/conda/bin/python -m |' \
        -e 's/export VLLM_ATTENTION_BACKEND=XFORMERS/export VLLM_ATTENTION_BACKEND=FLASH_ATTN/' \
        -e 's/actor_rollout_ref.rollout.enforce_eager=False/actor_rollout_ref.rollout.enforce_eager=True/' \
    > "$WRAPPER_PATH"
chmod +x "$WRAPPER_PATH"

ENGINE="${ENGINE:-vllm}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
NNODES="${NNODES:-1}"
RAY_INIT_CPUS="${RAY_INIT_CPUS:-32}"
ENV_WORKER_CPUS="${ENV_WORKER_CPUS:-0.1}"
if [ "${TENSOR_MODEL_PARALLEL_SIZE:-}" = "" ]; then
    if [ "$N_GPUS_PER_NODE" -eq 1 ]; then
        TENSOR_MODEL_PARALLEL_SIZE=1
    else
        TENSOR_MODEL_PARALLEL_SIZE=2
    fi
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-gigpo_qwen2.5_1.5b_${N_GPUS_PER_NODE}gpu_paper_align_opt}"

VISIBLE_GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
if [ "$VISIBLE_GPU_COUNT" -lt "$N_GPUS_PER_NODE" ]; then
    echo "[WARN] visible GPUs=$VISIBLE_GPU_COUNT but trainer.n_gpus_per_node=$N_GPUS_PER_NODE" >&2
fi
if [ "$TENSOR_MODEL_PARALLEL_SIZE" -gt "$N_GPUS_PER_NODE" ]; then
    echo "[WARN] tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE > n_gpus_per_node=$N_GPUS_PER_NODE" >&2
fi

LOG_FILE="logs/webshop_gigpo_qwen25_15b_${N_GPUS_PER_NODE}gpu_paper_align_opt_$(date +%Y%m%d_%H%M%S).log"

bash "$WRAPPER_PATH" "$ENGINE" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$TENSOR_MODEL_PARALLEL_SIZE" \
    ray_init.num_cpus="$RAY_INIT_CPUS" \
    env.resources_per_worker.num_cpus="$ENV_WORKER_CPUS" \
    "$@" \
    2>&1 | tee "$LOG_FILE"
