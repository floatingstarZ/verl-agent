#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
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
export WANDB_MODE="${WANDB_MODE:-offline}"
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
SEED="${SEED:-0}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-64}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
ENV_WORKER_CPUS="${ENV_WORKER_CPUS:-0.1}"
TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-16}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-10}"
GROUP_SIZE="${GROUP_SIZE:-8}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-50}"
ENGINE="${ENGINE:-vllm}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$HOME/checkpoints}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$CHECKPOINTS_DIR/reason_value_grpo_qwen25_15b_alfworld_full_seed0/global_step_150}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-reason_value_case_viz_10val_$(date +%Y%m%d_%H%M%S)}"
CASE_VIZ_OUT_DIR="${CASE_VIZ_OUT_DIR:-$ROOT_DIR/EXPS/reasoning_value_aux/outputs/value_case_viz/$EXPERIMENT_NAME}"
CASE_VIZ_MAX_CASES="${CASE_VIZ_MAX_CASES:-10}"
CASE_DATA_DIR="${CASE_DATA_DIR:-$CASE_VIZ_OUT_DIR/data}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${EXPERIMENT_NAME}.log}"

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

REASON_VALUE_LOSS_COEF="${REASON_VALUE_LOSS_COEF:-0.05}"
REASON_VALUE_TARGET_GAMMA="${REASON_VALUE_TARGET_GAMMA:-0.97}"
REASON_VALUE_TARGET_SCALE="${REASON_VALUE_TARGET_SCALE:-10.0}"
REASON_VALUE_RESPONSE_MAX_TOKENS="${REASON_VALUE_RESPONSE_MAX_TOKENS:-1024}"
REASON_VALUE_MAX_PROMPT_LENGTH="${REASON_VALUE_MAX_PROMPT_LENGTH:-3072}"
REASON_VALUE_REQUIRE_THINK_CLOSE="${REASON_VALUE_REQUIRE_THINK_CLOSE:-False}"
REASON_VALUE_DETACH_BACKBONE="${REASON_VALUE_DETACH_BACKBONE:-False}"
REASON_VALUE_PROMPT_STYLE="${REASON_VALUE_PROMPT_STYLE:-alfworld_actor_aligned_refined}"
REASON_VALUE_GENERATION_MODE="${REASON_VALUE_GENERATION_MODE:-generate}"

mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" \
  "$ALFWORLD_DATA" "$MPLCONFIGDIR" "$WANDB_DIR" "$CHECKPOINTS_DIR" "$LOG_DIR" "$CASE_VIZ_OUT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[FATAL] python3 not found. Set PYTHON_BIN_DIR, e.g. PYTHON_BIN_DIR=/opt/conda/bin" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[FATAL] nvidia-smi not found; this run requires CUDA GPUs." >&2
  exit 1
fi
if [[ ! -d "$CHECKPOINT_PATH" ]]; then
  echo "[FATAL] Missing checkpoint: $CHECKPOINT_PATH" >&2
  exit 1
fi
if [[ ! -d "$ALFWORLD_DATA/json_2.1.1" ]]; then
  echo "[FATAL] Missing ALFWorld data under $ALFWORLD_DATA/json_2.1.1" >&2
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

if [[ "${CHECK_FLASH_ATTN:-1}" == "1" ]]; then
  python3 - <<'PY'
import sys
import flash_attn, flash_attn_2_cuda
print(f"[INFO] python={sys.executable}")
print(f"[INFO] flash_attn={flash_attn.__file__}")
print(f"[INFO] flash_attn_2_cuda={flash_attn_2_cuda.__file__}")
PY
fi

echo "[INFO] ROOT_DIR=$ROOT_DIR"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "[INFO] CASE_VIZ_OUT_DIR=$CASE_VIZ_OUT_DIR"
echo "[INFO] LOG_FILE=$LOG_FILE"
echo "[INFO] VAL_DATA_SIZE=$VAL_DATA_SIZE CASE_VIZ_MAX_CASES=$CASE_VIZ_MAX_CASES"
echo "[INFO] REASON_VALUE_GENERATION_MODE=$REASON_VALUE_GENERATION_MODE REASON_VALUE_RESPONSE_MAX_TOKENS=$REASON_VALUE_RESPONSE_MAX_TOKENS MAX_PROMPT_LENGTH=$REASON_VALUE_MAX_PROMPT_LENGTH"

PREP_CMD=(python3 examples/data_preprocess/prepare.py --mode text --local_dir "$CASE_DATA_DIR" --train_data_size "$TRAIN_DATA_SIZE" --val_data_size "$VAL_DATA_SIZE")
CMD=(python3 -m verl.trainer.main_ppo)
OVERRIDES=(
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=False"
  "data.train_files=$CASE_DATA_DIR/text/train.parquet"
  "data.val_files=$CASE_DATA_DIR/text/test.parquet"
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
  "actor_rollout_ref.actor.ppo_mini_batch_size=256"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32"
  "actor_rollout_ref.actor.use_kl_loss=True"
  "actor_rollout_ref.actor.kl_loss_coef=0.01"
  "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=2"
  "actor_rollout_ref.rollout.name=$ENGINE"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.6"
  "actor_rollout_ref.rollout.enable_chunked_prefill=False"
  "actor_rollout_ref.rollout.enforce_eager=False"
  "actor_rollout_ref.rollout.free_cache_engine=False"
  "actor_rollout_ref.rollout.val_kwargs.temperature=0.4"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "actor_rollout_ref.actor.use_invalid_action_penalty=True"
  "actor_rollout_ref.actor.invalid_action_penalty_coef=0.1"
  "+actor_rollout_ref.actor.value_head.enable=True"
  "+actor_rollout_ref.actor.value_head.compute_state_values=False"
  "+actor_rollout_ref.actor.value_head.head_type=mlp"
  "+actor_rollout_ref.actor.value_head.detach_value_backbone=$REASON_VALUE_DETACH_BACKBONE"
  "+actor_rollout_ref.actor.reasoning_value_aux.enable=True"
  "+actor_rollout_ref.actor.reasoning_value_aux.train_only=False"
  "+actor_rollout_ref.actor.reasoning_value_aux.value_loss_coef=$REASON_VALUE_LOSS_COEF"
  "+actor_rollout_ref.actor.reasoning_value_aux.loss_type=mse"
  "+actor_rollout_ref.actor.reasoning_value_aux.target_gamma=$REASON_VALUE_TARGET_GAMMA"
  "+actor_rollout_ref.actor.reasoning_value_aux.target_scale=$REASON_VALUE_TARGET_SCALE"
  "+actor_rollout_ref.actor.reasoning_value_aux.clip_target=True"
  "+actor_rollout_ref.actor.reasoning_value_aux.target_min=0.0"
  "+actor_rollout_ref.actor.reasoning_value_aux.target_max=1.0"
  "+actor_rollout_ref.actor.reasoning_value_aux.response_max_tokens=$REASON_VALUE_RESPONSE_MAX_TOKENS"
  "+actor_rollout_ref.actor.reasoning_value_aux.max_prompt_length=$REASON_VALUE_MAX_PROMPT_LENGTH"
  "+actor_rollout_ref.actor.reasoning_value_aux.require_think_close=$REASON_VALUE_REQUIRE_THINK_CLOSE"
  "+actor_rollout_ref.actor.reasoning_value_aux.fallback_position=last_response_token"
  "+actor_rollout_ref.actor.reasoning_value_aux.strip_action_instruction=False"
  "+actor_rollout_ref.actor.reasoning_value_aux.prompt_style=$REASON_VALUE_PROMPT_STYLE"
  "+actor_rollout_ref.actor.reasoning_value_aux.generation_mode=$REASON_VALUE_GENERATION_MODE"
  "env.env_name=alfworld/AlfredTWEnv"
  "env.seed=$SEED"
  "env.max_steps=$MAX_ENV_STEPS"
  "env.rollout.n=$GROUP_SIZE"
  "env.resources_per_worker.num_cpus=$ENV_WORKER_CPUS"
  "trainer.critic_warmup=0"
  "trainer.logger=['console']"
  "trainer.project_name=verl_agent_alfworld_case_viz"
  "trainer.experiment_name=$EXPERIMENT_NAME"
  "trainer.n_gpus_per_node=$N_GPUS_PER_NODE"
  "trainer.nnodes=1"
  "trainer.save_freq=999999"
  "trainer.test_freq=1"
  "trainer.total_epochs=1"
  "trainer.total_training_steps=null"
  "trainer.default_local_dir=$CHECKPOINTS_DIR/reason_value_grpo_qwen25_15b_alfworld_full_seed0"
  "trainer.resume_mode=resume_path"
  "trainer.resume_from_path=$CHECKPOINT_PATH"
  "trainer.del_local_ckpt_after_load=False"
  "trainer.val_before_train=True"
  "trainer.val_only=True"
  "+trainer.reason_value_case_dump_dir=$CASE_VIZ_OUT_DIR"
  "+trainer.reason_value_case_dump_max_cases=$CASE_VIZ_MAX_CASES"
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
  echo "[INFO] WROTE_HTML=$CASE_VIZ_OUT_DIR/reason_value_cases.html"
  echo "[INFO] WROTE_JSON=$CASE_VIZ_OUT_DIR/reason_value_cases.json"
} 2>&1 | tee "$LOG_FILE"
