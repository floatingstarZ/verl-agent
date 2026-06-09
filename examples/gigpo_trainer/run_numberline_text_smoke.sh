set -x
ENGINE=${1:-vllm}
if [[ "$ENGINE" == "vllm" || "$ENGINE" == "hf" ]]; then
    shift || true
else
    ENGINE=vllm
fi
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export WANDB_MODE=${WANDB_MODE:-offline}

MODEL_PATH=${MODEL_PATH:-/prj/corp/crd/morpheus/lasvegas/china-scratch/ziyuhuan/Projects/models/Qwen3-4B}
num_cpus_per_env_worker=0.1
train_data_size=2
val_data_size=2
group_size=2
mode="mean_norm"

python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size ${train_data_size} \
    --val_data_size ${val_data_size}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=${train_data_size} \
    data.val_batch_size=${val_data_size} \
    data.max_prompt_length=1024 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.attn_implementation=eager \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=gym_cards/NumberLine-v0 \
    +env.gym_cards.text_only=True \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=${group_size} \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_numberline_text' \
    trainer.experiment_name='gigpo_qwen3_4b_smoke' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    ray_init.num_cpus=16 \
    "$@"
