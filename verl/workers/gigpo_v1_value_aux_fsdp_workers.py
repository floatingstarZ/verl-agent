# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FSDP worker entry for GiGPO-v1 value-aux experiments.

The implementation intentionally reuses the shared value-head worker used by
StepPPO. The algorithm-side difference is handled in the GiGPO-v1 value-aux
trainer: GiGPO advantages are left untouched and ``step_returns`` is attached as
an auxiliary value target.
"""

from verl.workers.step_ppo_fsdp_workers import ActorRolloutRefWorker

__all__ = ["ActorRolloutRefWorker"]
