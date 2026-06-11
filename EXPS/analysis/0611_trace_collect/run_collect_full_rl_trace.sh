#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/verl-agent"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
CKPT="${CKPT:-checkpoints/verl_agent_webshop/step_ppo_v3_value_head_detach_false_qwen2.5_1.5b_4gpu_paper_align_full_e250/global_step_250}"
TRACE_STEPS="${TRACE_STEPS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
TRAIN_DATA_SIZE="${TRAIN_DATA_SIZE:-$((TRAIN_BATCH_SIZE * TRACE_STEPS))}"
VAL_DATA_SIZE="${VAL_DATA_SIZE:-16}"
GROUP_SIZE="${GROUP_SIZE:-8}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-15}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.3}"
DO_UPDATE="${DO_UPDATE:-1}"
SKIP_DATALOADER_STATE="${SKIP_DATALOADER_STATE:-1}"
TRACE_DIR="${TRACE_DIR:-EXPS/analysis/0611_trace_collect/full_rl_trace_${RUN_TAG}}"
LOG_FILE="logs/0611_trace_collect/full_rl_trace_${RUN_TAG}.log"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
USER_NAME="${USER:-$(id -un)}"
CACHE_ROOT="${VERL_AGENT_CACHE_ROOT:-/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/.cache}"
DATA_ROOT="${VERL_AGENT_DATA_ROOT:-/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/data/verl-agent-trace-${RUN_TAG}}"

step_name="$(basename "$CKPT")"
if [[ ! "$step_name" =~ ^global_step_([0-9]+)$ ]]; then
  echo "[ERROR] CKPT must end with global_step_<N>: $CKPT" >&2
  exit 1
fi
GLOBAL_STEP="${BASH_REMATCH[1]}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-$((GLOBAL_STEP + TRACE_STEPS))}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-$TRACE_STEPS}"
CRITIC_WARMUP="${CRITIC_WARMUP:-0}"
if [[ "$DO_UPDATE" == "0" ]]; then
  CRITIC_WARMUP=999999
fi

mkdir -p "$TRACE_DIR" "$(dirname "$LOG_FILE")" "$DATA_ROOT" /tmp/${USER_NAME}

resume_ckpt="$CKPT"
tmp_resume_dir=""
if [[ "$SKIP_DATALOADER_STATE" == "1" ]]; then
  tmp_resume_dir="$(mktemp -d /tmp/${USER_NAME}/full_rl_trace_ckpt.XXXXXX)"
  resume_ckpt="$tmp_resume_dir/$step_name"
  mkdir -p "$resume_ckpt"
  ln -s "$(realpath "$CKPT/actor")" "$resume_ckpt/actor"
  if [[ -d "$CKPT/critic" ]]; then
    ln -s "$(realpath "$CKPT/critic")" "$resume_ckpt/critic"
  fi
  trap 'rm -rf "$tmp_resume_dir"; ray stop --force >/dev/null 2>&1 || true' EXIT INT TERM
else
  trap 'ray stop --force >/dev/null 2>&1 || true' EXIT INT TERM
fi

export PATH="/opt/conda/bin:$PATH"
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$CACHE_ROOT"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER_NAME}/.triton-cache}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT/torch}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-/tmp/${USER_NAME}/.vllm}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/${USER_NAME}}"
export TOKENIZERS_PARALLELISM=false
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -XX:ActiveProcessorCount=4 -XX:ParallelGCThreads=2 -XX:ConcGCThreads=1"
export VLLM_ATTENTION_BACKEND=XFORMERS

GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
if echo "$GPU_NAMES" | head -n 4 | rg -q 'H100'; then
  export FLASH_ATTN_SM90_SITE="${FLASH_ATTN_SM90_SITE:-/tmp/flash_attn_sm80_sm90_site}"
  FLASH_ATTN_SM90_EXT="$(find "$FLASH_ATTN_SM90_SITE" -maxdepth 1 -name 'flash_attn_2_cuda*.so' 2>/dev/null | head -1 || true)"
  if [[ -z "$FLASH_ATTN_SM90_EXT" ]]; then
    echo "[FATAL] Missing sm_90 flash-attn override under $FLASH_ATTN_SM90_SITE" >&2
    exit 1
  fi
  if command -v cuobjdump >/dev/null 2>&1 && ! cuobjdump --list-elf "$FLASH_ATTN_SM90_EXT" | awk '/sm_90/{found=1} END{exit !found}'; then
    echo "[FATAL] $FLASH_ATTN_SM90_EXT does not contain sm_90 kernels" >&2
    exit 1
  fi
  export PYTHONPATH="$FLASH_ATTN_SM90_SITE:${PYTHONPATH:-}"
  echo "[INFO] Using H100 flash-attn override: $FLASH_ATTN_SM90_EXT"
fi

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRITON_CACHE_DIR" "$TORCH_HOME" "$VLLM_CONFIG_ROOT" "$FLASHINFER_WORKSPACE_BASE"

hard_nproc_limit="$(ulimit -Hu)"
if [[ "$hard_nproc_limit" == "unlimited" ]]; then
  ulimit -Su unlimited || true
elif [[ "$hard_nproc_limit" -gt 65536 ]] 2>/dev/null; then
  ulimit -Su "$hard_nproc_limit" || true
fi

ray stop --force >/dev/null 2>&1 || true

echo "[INFO] source_ckpt=$CKPT"
echo "[INFO] resume_ckpt=$resume_ckpt"
echo "[INFO] trace_dir=$TRACE_DIR"
echo "[INFO] log=$LOG_FILE"
echo "[INFO] trace_steps=$TRACE_STEPS total_training_steps=$TOTAL_TRAINING_STEPS total_epochs=$TOTAL_EPOCHS"
echo "[INFO] train_data_size=$TRAIN_DATA_SIZE train_batch_size=$TRAIN_BATCH_SIZE group_size=$GROUP_SIZE max_env_steps=$MAX_ENV_STEPS"
echo "[INFO] gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
echo "[INFO] do_update=$DO_UPDATE critic_warmup=$CRITIC_WARMUP skip_dataloader_state=$SKIP_DATALOADER_STATE"

{
  set -x
  PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" examples/data_preprocess/prepare.py \
    --mode text \
    --local_dir "$DATA_ROOT" \
    --train_data_size "$TRAIN_DATA_SIZE" \
    --val_data_size "$((VAL_DATA_SIZE * 2))"

  PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" -m verl.trainer.main_gigpo_v1_value_aux \
    data.train_files="$DATA_ROOT/text/train.parquet" \
    data.val_files="$DATA_ROOT/text/test.parquet" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_DATA_SIZE" \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.value_head.enable=True \
    actor_rollout_ref.actor.value_head.value_loss_coef=0.03 \
    actor_rollout_ref.actor.value_head.cliprange_value=0.5 \
    actor_rollout_ref.actor.value_head.detach_value_backbone=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.adv_estimator=gigpo \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_norm \
    algorithm.gigpo_v1_value_aux.value_target=gae_returns \
    algorithm.gigpo_v1_value_aux.step_gamma=0.95 \
    algorithm.gigpo_v1_value_aux.step_lam=0.90 \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps="$MAX_ENV_STEPS" \
    env.rollout.n="$GROUP_SIZE" \
    env.resources_per_worker.num_cpus=0.1 \
    ray_init.num_cpus=64 \
    trainer.project_name=verl_agent_webshop \
    trainer.experiment_name="full_rl_trace_${RUN_TAG}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.critic_warmup="$CRITIC_WARMUP" \
    trainer.max_actor_ckpt_to_keep=0 \
    trainer.max_critic_ckpt_to_keep=0 \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$resume_ckpt" \
    trainer.logger='[console]' \
    trainer.rl_trace_dir="$TRACE_DIR/rl_trace" \
    trainer.rl_trace_interval=1 \
    trainer.rl_trace_max_rows=0 \
    trainer.rl_trace_save_dataproto=True \
    trainer.rl_trace_include_text=True \
    trainer.rl_trace_include_token_arrays=True \
    trainer.rl_trace_include_prob_arrays=True \
    trainer.rl_trace_include_full_prompt_tokens=False \
    trainer.rl_trace_token_array_max_items=0 \
    trainer.rl_trace_keep_entropy=True \
    trainer.value_diagnostics_dir="$TRACE_DIR/value_diagnostics" \
    trainer.value_diagnostics_interval=1 \
    trainer.value_diagnostics_max_rows=0 \
    trainer.value_diagnostics_include_text=True \
    trainer.rollout_data_dir="$TRACE_DIR/rollout_generations" \
    "$@"
} 2>&1 | tee "$LOG_FILE"
