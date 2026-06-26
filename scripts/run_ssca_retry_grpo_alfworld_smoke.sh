#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PLATFORM_HOME="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan"
export HOME="${RUN_HOME:-$PLATFORM_HOME}"
export PYTHONUSERBASE="${PYTHONUSERBASE:-$HOME/.local}"
export PATH="${PYTHON_BIN_DIR:-/opt/conda/bin}:$PYTHONUSERBASE/bin:$PATH"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
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

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-2026}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
ENV_WORKER_CPUS="${ENV_WORKER_CPUS:-0.1}"
TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-4}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GROUP_SIZE="${GROUP_SIZE:-2}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TEST_FREQ="${TEST_FREQ:--1}"
SAVE_FREQ="${SAVE_FREQ:--1}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
ENGINE="${ENGINE:-vllm}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.4}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ssca_retry_grpo_qwen25_15b_alfworld_smoke_seed${SEED}_${RUN_TAG}}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$HOME/checkpoints}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${EXPERIMENT_NAME}.log}"
TRACE_DIR="${TRACE_DIR:-$ROOT_DIR/EXPS/analysis/022_ssca_retry_smoke/${EXPERIMENT_NAME}}"
DATA_ROOT="${DATA_ROOT:-$HOME/data/verl-agent-ssca-retry-${RUN_TAG}}"

if [[ "${ENABLE_WANDB:-0}" == "1" ]]; then
  LOGGER_OVERRIDE="trainer.logger=['console','wandb']"
  export WANDB_MODE="${WANDB_MODE:-online}"
else
  LOGGER_OVERRIDE="trainer.logger=['console']"
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi

mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" \
  "$ALFWORLD_DATA" "$MPLCONFIGDIR" "$WANDB_DIR" "$CHECKPOINTS_DIR" "$LOG_DIR" "$TRACE_DIR" "$DATA_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[FATAL] python3 not found. Set PYTHON_BIN_DIR, e.g. PYTHON_BIN_DIR=/opt/conda/bin" >&2
  exit 1
fi
if [[ "${DRY_RUN:-0}" != "1" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[FATAL] nvidia-smi not found; this run requires CUDA GPUs." >&2
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

if [[ "${DRY_RUN:-0}" != "1" && ! -d "$ALFWORLD_DATA/json_2.1.1" ]]; then
  echo "[FATAL] Missing ALFWorld data under $ALFWORLD_DATA/json_2.1.1" >&2
  echo "[FATAL] Run: alfworld-download --data-dir '$ALFWORLD_DATA'" >&2
  exit 1
fi

cleanup_ray() {
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}
if [[ "${CLEAN_RAY:-1}" == "1" && "${DRY_RUN:-0}" != "1" ]]; then
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
echo "[INFO] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES N_GPUS_PER_NODE=$N_GPUS_PER_NODE"
echo "[INFO] ALFWORLD_DATA=$ALFWORLD_DATA"
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE TRAIN_DATA_SIZE=$TRAIN_DATA_SIZE VAL_DATA_SIZE=$VAL_DATA_SIZE GROUP_SIZE=$GROUP_SIZE"
echo "[INFO] MAX_ENV_STEPS=$MAX_ENV_STEPS TOTAL_EPOCHS=$TOTAL_EPOCHS"
echo "[INFO] RAY_NUM_CPUS=$RAY_NUM_CPUS ENV_WORKER_CPUS=$ENV_WORKER_CPUS"
echo "[INFO] DATA_ROOT=$DATA_ROOT"
echo "[INFO] TRACE_DIR=$TRACE_DIR"
echo "[INFO] LOG_FILE=$LOG_FILE"
echo "[INFO] LOGGER=$LOGGER_OVERRIDE WANDB_MODE=$WANDB_MODE"

PREP_CMD=(python3 examples/data_preprocess/prepare.py --mode text --local_dir "$DATA_ROOT" --train_data_size "$TRAIN_DATA_SIZE" --val_data_size "$VAL_DATA_SIZE")
CMD=(python3 -m recipe.SSCA.main_ssca)
OVERRIDES=(
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=False"
  "data.train_files=$DATA_ROOT/text/train.parquet"
  "data.val_files=$DATA_ROOT/text/test.parquet"
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  "data.val_batch_size=$VAL_DATA_SIZE"
  "data.max_prompt_length=4096"
  "data.max_response_length=512"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "data.return_raw_chat=True"
  "actor_rollout_ref.model.path=$MODEL_PATH"
  "+actor_rollout_ref.model.attn_implementation=flash_attention_2"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.actor.ppo_mini_batch_size=32"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8"
  "actor_rollout_ref.actor.use_kl_loss=True"
  "actor_rollout_ref.actor.kl_loss_coef=0.01"
  "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=2"
  "actor_rollout_ref.rollout.name=$ENGINE"
  "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
  "actor_rollout_ref.rollout.enable_chunked_prefill=False"
  "actor_rollout_ref.rollout.enforce_eager=True"
  "actor_rollout_ref.rollout.free_cache_engine=False"
  "actor_rollout_ref.rollout.val_kwargs.temperature=0.4"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "actor_rollout_ref.actor.use_invalid_action_penalty=True"
  "actor_rollout_ref.actor.invalid_action_penalty_coef=0.1"
  "env.env_name=alfworld/AlfredTWEnv"
  "env.seed=$SEED"
  "env.max_steps=$MAX_ENV_STEPS"
  "env.rollout.n=$GROUP_SIZE"
  "env.resources_per_worker.num_cpus=$ENV_WORKER_CPUS"
  "+ssca.retry_same_seed=${SSCA_RETRY_SAME_SEED:-True}"
  "+ssca.failure_reward_threshold=${SSCA_FAILURE_REWARD_THRESHOLD:-0.0}"
  "+ssca.summary_max_history_chars=${SSCA_SUMMARY_MAX_HISTORY_CHARS:-12000}"
  "+ssca.summary_max_prompt_chars=${SSCA_SUMMARY_MAX_PROMPT_CHARS:-4000}"
  "+ssca.summary_max_chars=${SSCA_SUMMARY_MAX_CHARS:-2000}"
  "trainer.critic_warmup=0"
  "$LOGGER_OVERRIDE"
  "trainer.project_name=verl_agent_alfworld"
  "trainer.experiment_name=$EXPERIMENT_NAME"
  "trainer.n_gpus_per_node=$N_GPUS_PER_NODE"
  "trainer.nnodes=1"
  "trainer.save_freq=$SAVE_FREQ"
  "trainer.test_freq=$TEST_FREQ"
  "trainer.total_epochs=$TOTAL_EPOCHS"
  "trainer.val_before_train=False"
  "trainer.default_local_dir=$CHECKPOINTS_DIR/$EXPERIMENT_NAME"
  "trainer.rollout_data_dir=$TRACE_DIR/rollout_generations"
  "+trainer.rl_trace_dir=$TRACE_DIR/rl_trace"
  "+trainer.rl_trace_interval=1"
  "+trainer.rl_trace_include_text=True"
  "+trainer.rl_trace_include_token_arrays=True"
  "+trainer.rl_trace_include_prob_arrays=False"
  "+trainer.rl_trace_include_full_prompt_tokens=False"
  "+trainer.rl_trace_save_dataproto=True"
  "ray_init.num_cpus=$RAY_NUM_CPUS"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[DRY_RUN_PREP] '; printf '%q ' "${PREP_CMD[@]}"; printf '\n'
  printf '[DRY_RUN_TRAIN] '; printf '%q ' "${CMD[@]}" "${OVERRIDES[@]}" "$@"; printf '\n'
  exit 0
fi

{
  "${PREP_CMD[@]}"
  "${CMD[@]}" "${OVERRIDES[@]}" "$@"
} 2>&1 | tee "$LOG_FILE"
