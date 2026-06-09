set -x
ENGINE=${1:-vllm}
PYTHON_BIN="${PYTHON_BIN:-python3}"

user_name="${USER:-$(id -un)}"
cache_root="${VERL_AGENT_CACHE_ROOT:-/tmp/${user_name}/.cache}"
data_root="${VERL_AGENT_DATA_ROOT:-/tmp/${user_name}/data/verl-agent}"

export XDG_CACHE_HOME="$cache_root"
export HF_HOME="${HF_HOME:-$cache_root/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${user_name}/.triton-cache}"
export TORCH_HOME="${TORCH_HOME:-$cache_root/torch}"
export VLLM_CONFIG_ROOT="${VLLM_CONFIG_ROOT:-/tmp/${user_name}/.vllm}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp/${user_name}}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRITON_CACHE_DIR" "$TORCH_HOME" "$VLLM_CONFIG_ROOT"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"
mkdir -p "$data_root"

# Avoid RLIMIT_NPROC exhaustion when many Ray workers spawn JVM subprocesses.
hard_nproc_limit="$(ulimit -Hu)"
if [ "$hard_nproc_limit" = "unlimited" ]; then
    ulimit -Su unlimited || true
elif [ "$hard_nproc_limit" -gt 65536 ] 2>/dev/null; then
    ulimit -Su "$hard_nproc_limit" || true
fi

# Keep JVM footprint small for WebShop env workers (pyjnius).
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -XX:ActiveProcessorCount=4 -XX:ParallelGCThreads=2 -XX:ConcGCThreads=1"
export VLLM_ATTENTION_BACKEND=XFORMERS

num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=16
val_data_size=128
group_size=8
mode="mean_norm" # "mean_norm" or "mean_std_norm"

# We only use data preparation to indicate the modality and the data size.
PYTHONPATH="$(pwd):${PYTHONPATH:-}" "$PYTHON_BIN" examples/data_preprocess/prepare.py \
    --mode 'text' \
    --local_dir "$data_root" \
    --train_data_size $train_data_size \
    --val_data_size $((val_data_size * 2)) # evaluate 2 × val_data_size tasks during each iteration

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$data_root/text/train.parquet \
    data.val_files=$data_root/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
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
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name='gigpo_qwen2.5_1.5b' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@
