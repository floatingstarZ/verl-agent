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
SEED="${SEED:-2026}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
ENV_WORKER_CPUS="${ENV_WORKER_CPUS:-0.1}"
TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-1024}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-128}"
GROUP_SIZE="${GROUP_SIZE:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-9999}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"
TEST_FREQ="${TEST_FREQ:-5}"
SAVE_FREQ="${SAVE_FREQ:--1}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-50}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-16}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}"
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}"
NTF_KEEP_RATIO="${NTF_KEEP_RATIO:-0.1}"
NTF_MIN_KEEP_TOKENS="${NTF_MIN_KEEP_TOKENS:-1}"
POSITIVE_THRESHOLD="${POSITIVE_THRESHOLD:-0.0}"
LR="${LR:-1e-6}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.6}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
ENGINE="${ENGINE:-vllm}"
DATA_ROOT="${DATA_ROOT:-$HOME/data/verl-agent}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$HOME/checkpoints}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-crf_ntf_qwen25_15b_alfworld_full_seed${SEED}}"
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

cleanup_ray() {
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}
if [[ "${CLEAN_RAY:-1}" == "1" ]]; then
  cleanup_ray
  trap cleanup_ray EXIT INT TERM
fi

# shellcheck disable=SC1091
source "$ROOT_DIR/recipe/GraphGPO/setup_flash_attn_env.sh"
graphgpo_setup_flash_attn_sm90

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  python3 - <<'FLASHCHECK'
import sys
import flash_attn, flash_attn_2_cuda
print(f"[INFO] python={sys.executable}")
print(f"[INFO] flash_attn={flash_attn.__file__}")
print(f"[INFO] flash_attn_2_cuda={flash_attn_2_cuda.__file__}")
FLASHCHECK
else
  echo "[INFO] DRY_RUN=1; skipping flash-attn Python import check"
fi

echo "[INFO] ROOT_DIR=$ROOT_DIR"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[INFO] TRAIN_DATA_SIZE=$TRAIN_DATA_SIZE TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE GROUP_SIZE=$GROUP_SIZE"
echo "[INFO] TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS TOTAL_EPOCHS=$TOTAL_EPOCHS TEST_FREQ=$TEST_FREQ"
echo "[INFO] C-RF NTF keep_ratio=$NTF_KEEP_RATIO positive_threshold=$POSITIVE_THRESHOLD"
echo "[INFO] LOG_FILE=$LOG_FILE"

PREP_CMD=(python3 examples/data_preprocess/prepare.py --mode text --local_dir "$DATA_ROOT" --train_data_size "$TRAIN_DATA_SIZE" --val_data_size "$VAL_DATA_SIZE")
CMD=(python3 -m verl.trainer.main_ppo)
OVERRIDES=(
  "algorithm.adv_estimator=contrastive_reinforce"
  "algorithm.use_kl_in_reward=False"
  "algorithm.contrastive_rf.positive_threshold=$POSITIVE_THRESHOLD"
  "data.train_files=$DATA_ROOT/text/train.parquet"
  "data.val_files=$DATA_ROOT/text/test.parquet"
  "data.train_batch_size=$TRAIN_BATCH_SIZE"
  "data.val_batch_size=$VAL_DATA_SIZE"
  "data.max_prompt_length=$MAX_PROMPT_LENGTH"
  "data.max_response_length=$MAX_RESPONSE_LENGTH"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "data.return_raw_chat=True"
  "actor_rollout_ref.model.path=$MODEL_PATH"
  "+actor_rollout_ref.model.attn_implementation=flash_attention_2"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.optim.lr=$LR"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS"
  "actor_rollout_ref.actor.optim.warmup_style=exponential"
  "actor_rollout_ref.actor.optim.min_lr_ratio=$LR_MIN_RATIO"
  "actor_rollout_ref.actor.optim.weight_decay=$WEIGHT_DECAY"
  "actor_rollout_ref.actor.grad_clip=1.0"
  "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU"
  "actor_rollout_ref.actor.clip_ratio_low=0.2"
  "actor_rollout_ref.actor.clip_ratio_high=10.0"
  "actor_rollout_ref.actor.entropy_coeff=0.0"
  "actor_rollout_ref.actor.policy_loss.loss_mode=c_rf_ntf"
  "actor_rollout_ref.actor.policy_loss.ntf_keep_ratio=$NTF_KEEP_RATIO"
  "actor_rollout_ref.actor.policy_loss.ntf_min_keep_tokens=$NTF_MIN_KEEP_TOKENS"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.kl_loss_coef=0.0"
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=2"
  "actor_rollout_ref.rollout.name=$ENGINE"
  "actor_rollout_ref.rollout.temperature=1.0"
  "actor_rollout_ref.rollout.top_p=1.0"
  "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
  "actor_rollout_ref.rollout.enable_chunked_prefill=False"
  "actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER:-False}"
  "actor_rollout_ref.rollout.free_cache_engine=False"
  "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE:-1.0}"
  "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-0.7}"
  "actor_rollout_ref.rollout.val_kwargs.n=${VAL_SAMPLES_PER_PROMPT:-1}"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "actor_rollout_ref.actor.use_invalid_action_penalty=False"
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
  "trainer.val_before_train=${VAL_BEFORE_TRAIN:-True}"
  "trainer.default_local_dir=$CHECKPOINTS_DIR/$EXPERIMENT_NAME"
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
