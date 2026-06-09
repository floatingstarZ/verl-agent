# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Trainer shim for the step-wise PPO experiment.

This module keeps the default ``ray_trainer.py`` unchanged. It reuses the
standard trainer loop and monkey-patches only the two extension points needed
for this experiment while ``fit`` is running:

1. Prepare step indices/rewards before batch adjustment duplicates/reorders data.
2. Replace advantage computation with step-wise PPO when adv_estimator=step_ppo.
"""

from omegaconf import open_dict

from verl.trainer.ppo import ray_trainer as ray_trainer_module
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer
from verl.trainer.ppo import step_ppo_algos


class StepWisePPORayTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        config = kwargs.get("config", args[0] if args else None)
        if config is None:
            raise ValueError("StepWisePPORayTrainer requires a config")

        original_adv_estimator = config.algorithm.adv_estimator
        value_head_enabled = bool(config.actor_rollout_ref.actor.get("value_head", {}).get("enable", False))
        if not value_head_enabled:
            raise ValueError("step-wise PPO requires actor_rollout_ref.actor.value_head.enable=True")

        # Let the base trainer pass existing validation without changing it.
        with open_dict(config):
            config.algorithm.adv_estimator = AdvantageEstimator.GRPO
        try:
            super().__init__(*args, **kwargs)
        finally:
            with open_dict(config):
                config.algorithm.adv_estimator = original_adv_estimator

        self.use_critic = False

    def fit(self):
        original_compute_advantage = ray_trainer_module.compute_advantage
        original_adjust_batch = ray_trainer_module.adjust_batch

        def adjust_batch_with_step_fields(config, data, *args, **kwargs):
            if step_ppo_algos.is_step_ppo_estimator(self.config.algorithm.adv_estimator):
                step_ppo_algos.prepare_step_ppo_rollout_batch(data)
            return original_adjust_batch(config, data, *args, **kwargs)

        def compute_advantage_with_step_ppo(data, adv_estimator, *args, **kwargs):
            if step_ppo_algos.is_step_ppo_estimator(adv_estimator):
                return step_ppo_algos.compute_step_ppo_advantage_data(
                    data=data,
                    algorithm_config=self.config.algorithm,
                    multi_turn=kwargs.get("multi_turn", False),
                )
            return original_compute_advantage(data, adv_estimator, *args, **kwargs)

        ray_trainer_module.adjust_batch = adjust_batch_with_step_fields
        ray_trainer_module.compute_advantage = compute_advantage_with_step_ppo
        try:
            return super().fit()
        finally:
            ray_trainer_module.adjust_batch = original_adjust_batch
            ray_trainer_module.compute_advantage = original_compute_advantage
