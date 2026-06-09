# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Filtered rollout sharding managers for actor models with value heads."""

from collections import OrderedDict
from contextlib import contextmanager


def filter_value_head_state_dict(state_dict):
    return OrderedDict(
        (key, value)
        for key, value in state_dict.items()
        if not key.startswith("value_head.") and ".value_head." not in key
    )


@contextmanager
def patched_filtered_state_dict(module):
    original_state_dict = module.state_dict

    def state_dict_without_value_head(*args, **kwargs):
        return filter_value_head_state_dict(original_state_dict(*args, **kwargs))

    module.state_dict = state_dict_without_value_head
    try:
        yield
    finally:
        module.state_dict = original_state_dict


def get_step_ppo_vllm_sharding_manager_cls():
    from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

    class StepPPOFSDPVLLMShardingManager(FSDPVLLMShardingManager):
        def __enter__(self):
            with patched_filtered_state_dict(self.module):
                return super().__enter__()

    return StepPPOFSDPVLLMShardingManager


def get_step_ppo_sglang_sharding_manager_cls():
    from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

    class StepPPOFSDPSGLangShardingManager(FSDPSGLangShardingManager):
        def __enter__(self):
            with patched_filtered_state_dict(self.module):
                return super().__enter__()

    return StepPPOFSDPSGLangShardingManager
