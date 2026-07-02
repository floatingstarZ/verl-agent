# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FSDP worker variants for the step-wise PPO experiment."""

from contextlib import nullcontext

import torch

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_torch_device
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.workers.actor.step_ppo_actor import StepWisePPOActor
from verl.workers.actor.value_head import attach_step_value_head
from verl.workers.fsdp_workers import ActorRolloutRefWorker as BaseActorRolloutRefWorker
from verl.workers.fsdp_workers import logger


class ActorRolloutRefWorker(BaseActorRolloutRefWorker):
    """Actor/rollout/ref worker that enables shared-backbone step-wise PPO.

    The default worker is left untouched. This subclass attaches the value head
    before FSDP wrapping, swaps in ``StepWisePPOActor`` for actor updates, and
    filters value-head parameters when syncing actor weights to rollout engines.
    """

    def _step_ppo_enabled(self) -> bool:
        if not self._is_actor:
            return False
        return bool(self.config.actor.get("value_head", {}).get("enable", False))

    def _build_model_optimizer(self, *args, **kwargs):
        role = kwargs.get("role", None)
        if role is None and len(args) >= 10:
            role = args[9]
        if role != "actor" or not self._step_ppo_enabled():
            return super()._build_model_optimizer(*args, **kwargs)

        from transformers import AutoModelForCausalLM, AutoModelForVision2Seq

        causal_attr = AutoModelForCausalLM.__dict__["from_pretrained"]
        vision_attr = AutoModelForVision2Seq.__dict__["from_pretrained"]
        causal_from_pretrained = AutoModelForCausalLM.from_pretrained
        vision_from_pretrained = AutoModelForVision2Seq.from_pretrained
        value_head_config = self.config.actor.value_head

        def attach_after_load(loader):
            def wrapped(*loader_args, **loader_kwargs):
                model = loader(*loader_args, **loader_kwargs)
                model_config = loader_kwargs.get("config", getattr(model, "config", None))
                return attach_step_value_head(model, model_config, value_head_config)

            return wrapped

        AutoModelForCausalLM.from_pretrained = staticmethod(attach_after_load(causal_from_pretrained))
        AutoModelForVision2Seq.from_pretrained = staticmethod(attach_after_load(vision_from_pretrained))
        try:
            return super()._build_model_optimizer(*args, **kwargs)
        finally:
            AutoModelForCausalLM.from_pretrained = causal_attr
            AutoModelForVision2Seq.from_pretrained = vision_attr

    def _build_rollout(self, trust_remote_code=False):
        if not self._step_ppo_enabled():
            return super()._build_rollout(trust_remote_code=trust_remote_code)

        rollout_name = self.config.rollout.name
        if rollout_name == "vllm":
            import verl.workers.sharding_manager.fsdp_vllm as manager_module
            from verl.workers.sharding_manager.step_ppo_filters import get_step_ppo_vllm_sharding_manager_cls

            old_cls = manager_module.FSDPVLLMShardingManager
            manager_module.FSDPVLLMShardingManager = get_step_ppo_vllm_sharding_manager_cls()
        elif rollout_name in ["sglang", "sglang_async"]:
            import verl.workers.sharding_manager.fsdp_sglang as manager_module
            from verl.workers.sharding_manager.step_ppo_filters import get_step_ppo_sglang_sharding_manager_cls

            old_cls = manager_module.FSDPSGLangShardingManager
            manager_module.FSDPSGLangShardingManager = get_step_ppo_sglang_sharding_manager_cls()
        else:
            return super()._build_rollout(trust_remote_code=trust_remote_code)

        try:
            return super()._build_rollout(trust_remote_code=trust_remote_code)
        finally:
            if rollout_name == "vllm":
                manager_module.FSDPVLLMShardingManager = old_cls
            else:
                manager_module.FSDPSGLangShardingManager = old_cls

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._is_actor and self._step_ppo_enabled():
            self.actor = StepWisePPOActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_reasoning_values(self, data: DataProto):
        """Compute value-head predictions for reasoning-value visualization/eval."""
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        data = data.to(get_torch_device().current_device())
        data.meta_info["micro_batch_size"] = self.config.actor.ppo_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.actor.ppo_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.actor.use_dynamic_bsz

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            values = self.actor.compute_reasoning_values(data=data)
            output = DataProto.from_dict(tensors={"reason_value_preds": values})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_reasoning_values", logger=logger)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        data = data.to(get_torch_device().current_device())
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            with adapter_ctx:
                actor_output = self.actor.compute_log_prob(data=data, calculate_entropy=True)
            if len(actor_output) == 2:
                log_probs, entropys = actor_output
                state_values = None
            else:
                log_probs, entropys, state_values = actor_output

            tensors = {"old_log_probs": log_probs, "entropys": entropys}
            if state_values is not None:
                tensors["state_values"] = state_values
            output = DataProto.from_dict(
                tensors=tensors,
                meta_info={"temperature": self.config.rollout.temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during compute_log_prob", logger=logger)

        return output
