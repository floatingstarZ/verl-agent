#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Source shared runtime env vars, but use /opt/conda Python for WebShop
# because the successful verl-agent run depends on numpy<2 + faiss ABI there.
# shellcheck disable=SC1091
source "$ROOT_DIR/scripts/activate_qwen_character_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
FLASH_ATTN_SM90_SITE="${FLASH_ATTN_SM90_SITE:-/tmp/flash_attn_sm80_sm90_site}"
FLASH_ATTN_SM90_EXT="$(find "$FLASH_ATTN_SM90_SITE" -maxdepth 1 -name 'flash_attn_2_cuda*.so' 2>/dev/null | head -1 || true)"
if [[ -z "$FLASH_ATTN_SM90_EXT" ]]; then
  echo "[FATAL] Missing sm_90 flash-attn override under $FLASH_ATTN_SM90_SITE" >&2
  exit 1
fi
export PYTHONPATH="$FLASH_ATTN_SM90_SITE:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/repro_data_gigpo_align}"
SEED="${SEED:-2026}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-istar_rloo_qwen25_15b_4gpu_gigpo_align_e250_seed${SEED}}"

TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
VAL_PREP_SIZE="${VAL_PREP_SIZE:-256}"
GROUP_SIZE="${GROUP_SIZE:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-250}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[FATAL] PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[FATAL] MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -e agent_system/environments/env_package/webshop/webshop/data/items_shuffle_1000.json ]]; then
  echo "[FATAL] Missing WebShop data. Expected agent_system/environments/env_package/webshop/webshop/data/items_shuffle_1000.json" >&2
  exit 1
fi

mkdir -p "$DATA_ROOT" logs

echo "[INFO] ROOT_DIR=$ROOT_DIR"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] DATA_ROOT=$DATA_ROOT"
echo "[INFO] SEED=$SEED"
echo "[INFO] EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "[INFO] TOTAL_EPOCHS=$TOTAL_EPOCHS"
echo "[INFO] PYTHON_BIN=$PYTHON_BIN"
echo "[INFO] FLASH_ATTN_SM90_SITE=$FLASH_ATTN_SM90_SITE"
echo "[INFO] FLASH_ATTN_SM90_EXT=$FLASH_ATTN_SM90_EXT"
"$PYTHON_BIN" - <<'PY'
import sys, numpy
print(f"[INFO] python_executable={sys.executable}")
print(f"[INFO] numpy={numpy.__version__}")
PY

"$PYTHON_BIN" examples/data_preprocess/prepare.py \
  --mode text \
  --local_dir "$DATA_ROOT" \
  --train_data_size "$TRAIN_DATA_SIZE" \
  --val_data_size "$VAL_PREP_SIZE"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=istar_rloo \
  data.train_files="$DATA_ROOT/text/train.parquet" \
  data.val_files="$DATA_ROOT/text/test.parquet" \
  data.train_batch_size="$TRAIN_DATA_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length=4096 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  reward_model.enable=True \
  reward_model.model.ref_path=null \
  reward_model.model.path="$MODEL_PATH" \
  reward_model.micro_batch_size_per_gpu=8 \
  reward_model.model.update=after \
  reward_model.model.loss_type=eto \
  reward_model.model.beta_train=0.05 \
  reward_model.model.optim.lr=1e-6 \
  reward_model.model.optim.grad_clip=10.0 \
  reward_model.model.input_tokenizer=null \
  reward_model.mini_batch_size=64 \
  reward_model.num_rollout="$GROUP_SIZE" \
  reward_model.step_granularity=step \
  algorithm.use_kl_in_reward=False \
  algorithm.gigpo.step_advantage_w=1.0 \
  env.env_name=Webshop \
  env.seed="$SEED" \
  env.max_steps=15 \
  env.rollout.n="$GROUP_SIZE" \
  trainer.critic_warmup=0 \
  trainer.logger=[console] \
  trainer.project_name=istar_vs_gigpo_webshop \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=5 \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.val_before_train=True \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.max_critic_ckpt_to_keep=1 \
  +trainer.max_rm_ckpt_to_keep=1 \
  "$@"
