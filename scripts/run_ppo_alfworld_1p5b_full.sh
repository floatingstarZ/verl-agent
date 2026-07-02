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

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-150}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
TEST_FREQ="${TEST_FREQ:-5}"
SAVE_FREQ="${SAVE_FREQ:--1}"
SEED="${SEED:-0}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
ENV_WORKER_CPUS="${ENV_WORKER_CPUS:-0.1}"
TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-128}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-128}"
GROUP_SIZE="${GROUP_SIZE:-1}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-50}"
GAE_GAMMA="${GAE_GAMMA:-1.0}"
GAE_LAM="${GAE_LAM:-1.0}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
ACTOR_MICRO_BATCH_SIZE="${ACTOR_MICRO_BATCH_SIZE:-16}"
CRITIC_MICRO_BATCH_SIZE="${CRITIC_MICRO_BATCH_SIZE:-16}"
LOG_PROB_MICRO_BATCH_SIZE="${LOG_PROB_MICRO_BATCH_SIZE:-32}"
REF_LOG_PROB_MICRO_BATCH_SIZE="${REF_LOG_PROB_MICRO_BATCH_SIZE:-32}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-2}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-ppo_qwen25_15b_alfworld_full_seed${SEED}}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$HOME/checkpoints}"
MODEL_REPO_ID="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_CACHE_REF="$HF_HOME/hub/models--Qwen--Qwen2.5-1.5B-Instruct/refs/main"
CACHED_MODEL_PATH=""
if [[ -f "$MODEL_CACHE_REF" ]]; then
  MODEL_CACHE_REVISION="$(cat "$MODEL_CACHE_REF")"
  MODEL_CACHE_SNAPSHOT="$HF_HOME/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/$MODEL_CACHE_REVISION"
  if [[ -f "$MODEL_CACHE_SNAPSHOT/config.json" ]]; then
    CACHED_MODEL_PATH="$MODEL_CACHE_SNAPSHOT"
  fi
fi
MODEL_PATH="${MODEL_PATH:-${CACHED_MODEL_PATH:-$MODEL_REPO_ID}}"
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

if [[ "${DRY_RUN:-0}" != "1" ]]; then
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

if [[ -f "$ROOT_DIR/recipe/GraphGPO/setup_flash_attn_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/recipe/GraphGPO/setup_flash_attn_env.sh"
  graphgpo_setup_flash_attn_sm90
elif [[ -d "${FLASH_ATTN_SM90_SITE:-}" ]]; then
  export PYTHONPATH="${FLASH_ATTN_SM90_SITE}:$PYTHONPATH"
fi

if [[ "${DRY_RUN:-0}" != "1" && "${CHECK_FLASH_ATTN:-1}" == "1" ]]; then
  python3 - <<'PY'
import sys
import flash_attn, flash_attn_2_cuda
print(f"[INFO] python={sys.executable}")
print(f"[INFO] flash_attn={flash_attn.__file__}")
print(f"[INFO] flash_attn_2_cuda={flash_attn_2_cuda.__file__}")
PY
else
  echo "[INFO] DRY_RUN=1 or CHECK_FLASH_ATTN=0; skipping flash-attn Python import check"
fi

echo "[INFO] ROOT_DIR=$ROOT_DIR"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] N_GPUS_PER_NODE=$N_GPUS_PER_NODE"
echo "[INFO] ALFWORLD_DATA=$ALFWORLD_DATA"
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[INFO] TOTAL_EPOCHS=$TOTAL_EPOCHS TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS TEST_FREQ=$TEST_FREQ SAVE_FREQ=$SAVE_FREQ"
echo "[INFO] RAY_NUM_CPUS=$RAY_NUM_CPUS ENV_WORKER_CPUS=$ENV_WORKER_CPUS DATALOADER_NUM_WORKERS=$DATALOADER_NUM_WORKERS"
echo "[INFO] TRAIN_DATA_SIZE=$TRAIN_DATA_SIZE VAL_DATA_SIZE=$VAL_DATA_SIZE GROUP_SIZE=$GROUP_SIZE"
echo "[INFO] PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE ACTOR_MICRO_BATCH_SIZE=$ACTOR_MICRO_BATCH_SIZE CRITIC_MICRO_BATCH_SIZE=$CRITIC_MICRO_BATCH_SIZE"
echo "[INFO] GAE_GAMMA=$GAE_GAMMA GAE_LAM=$GAE_LAM"
echo "[INFO] LOG_FILE=$LOG_FILE"
echo "[INFO] LOGGER=$LOGGER_OVERRIDE WANDB_MODE=$WANDB_MODE"

PREP_CMD=(python3 examples/data_preprocess/prepare.py --mode text --train_data_size "$TRAIN_DATA_SIZE" --val_data_size "$VAL_DATA_SIZE")
CMD=(python3 -m verl.trainer.main_ppo)
OVERRIDES=(
  "algorithm.adv_estimator=gae"
  "algorithm.gamma=$GAE_GAMMA"
  "algorithm.lam=$GAE_LAM"
  "algorithm.use_kl_in_reward=False"
  "data.train_files=$HOME/data/verl-agent/text/train.parquet"
  "data.val_files=$HOME/data/verl-agent/text/test.parquet"
  "data.train_batch_size=$TRAIN_DATA_SIZE"
  "data.val_batch_size=$VAL_DATA_SIZE"
  "data.max_prompt_length=2048"
  "data.max_response_length=512"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "data.return_raw_chat=True"
  "+data.dataloader_num_workers=$DATALOADER_NUM_WORKERS"
  "actor_rollout_ref.model.path=$MODEL_PATH"
  "+actor_rollout_ref.model.attn_implementation=flash_attention_2"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ACTOR_MICRO_BATCH_SIZE"
  "actor_rollout_ref.actor.use_kl_loss=True"
  "actor_rollout_ref.actor.kl_loss_coef=0.01"
  "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE"
  "actor_rollout_ref.rollout.name=$ENGINE"
  "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION"
  "actor_rollout_ref.rollout.enable_chunked_prefill=False"
  "actor_rollout_ref.rollout.enforce_eager=False"
  "actor_rollout_ref.rollout.free_cache_engine=False"
  "actor_rollout_ref.rollout.val_kwargs.temperature=0.4"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_SIZE"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "actor_rollout_ref.actor.use_invalid_action_penalty=True"
  "actor_rollout_ref.actor.invalid_action_penalty_coef=0.1"
  "critic.optim.lr=1e-5"
  "critic.model.path=$MODEL_PATH"
  "+critic.model.attn_implementation=flash_attention_2"
  "critic.model.use_remove_padding=True"
  "critic.model.enable_gradient_checkpointing=True"
  "critic.ppo_micro_batch_size_per_gpu=$CRITIC_MICRO_BATCH_SIZE"
  "critic.model.fsdp_config.param_offload=False"
  "critic.model.fsdp_config.optimizer_offload=False"
  "env.env_name=alfworld/AlfredTWEnv"
  "env.seed=$SEED"
  "env.max_steps=$MAX_ENV_STEPS"
  "env.rollout.n=$GROUP_SIZE"
  "env.resources_per_worker.num_cpus=$ENV_WORKER_CPUS"
  "trainer.critic_warmup=0"
  "$LOGGER_OVERRIDE"
  "trainer.project_name=verl_agent_alfworld"
  "trainer.experiment_name=$EXPERIMENT_NAME"
  "trainer.n_gpus_per_node=$N_GPUS_PER_NODE"
  "trainer.nnodes=1"
  "trainer.save_freq=$SAVE_FREQ"
  "trainer.test_freq=$TEST_FREQ"
  "trainer.total_epochs=$TOTAL_EPOCHS"
  "trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
  "trainer.default_local_dir=$CHECKPOINTS_DIR/$EXPERIMENT_NAME"
  "trainer.val_before_train=${VAL_BEFORE_TRAIN:-True}"
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
