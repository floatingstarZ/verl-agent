# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Trainer shim for GiGPO-v1 with an auxiliary shared value head.

This experiment keeps the original GiGPO policy advantage unchanged. The value
head is trained as an auxiliary critic on StepPPO-style step-level GAE returns.
"""

import numpy as np
import torch

from verl.trainer.ppo import ray_trainer as ray_trainer_module
from verl.trainer.ppo import step_ppo_algos
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, RayPPOTrainer


class GiGPOV1ValueAuxRayTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        config = kwargs.get("config", args[0] if args else None)
        if config is None:
            raise ValueError("GiGPOV1ValueAuxRayTrainer requires a config")

        value_head_enabled = bool(config.actor_rollout_ref.actor.get("value_head", {}).get("enable", False))
        if not value_head_enabled:
            raise ValueError("GiGPO-v1 value-aux requires actor_rollout_ref.actor.value_head.enable=True")
        if config.algorithm.adv_estimator != AdvantageEstimator.GiGPO:
            raise ValueError("GiGPO-v1 value-aux keeps the original GiGPO advantage; set algorithm.adv_estimator=gigpo")

        # The base trainer only needs to know that this is a no-critic method.
        super().__init__(*args, **kwargs)
        self.use_critic = False

    def _value_aux_config(self):
        return self.config.algorithm.get("gigpo_v1_value_aux", {})

    def _build_shaped_immediate_rewards(self, data) -> torch.Tensor:
        """Build immediate rewards for StepPPO-style GAE value targets.

        GiGPO's ``step_rewards`` is a discounted return used by its step relative
        advantage, so it must not be reused as an immediate reward for GAE.
        """
        if "state_values" not in data.batch:
            raise KeyError("GiGPO-v1 GAE value target requires state_values from the shared value head")
        if "rewards" not in data.non_tensor_batch:
            raise KeyError("GiGPO-v1 GAE value target requires raw per-step 'rewards' in non_tensor_batch")

        device = data.batch["state_values"].device
        rewards = torch.as_tensor(
            np.asarray(data.non_tensor_batch["rewards"], dtype=np.float32).reshape(-1),
            dtype=torch.float32,
            device=device,
        )

        if self.config.actor_rollout_ref.actor.get("use_invalid_action_penalty", True):
            if "is_action_valid" not in data.non_tensor_batch:
                raise KeyError("GiGPO-v1 GAE value target requires is_action_valid when invalid action penalty is enabled")
            action_valids = np.asarray(data.non_tensor_batch["is_action_valid"], dtype=np.float32).reshape(-1)
            action_invalids = torch.as_tensor(1.0 - action_valids, dtype=torch.float32, device=device)
            rewards = rewards - float(self.config.actor_rollout_ref.actor.invalid_action_penalty_coef) * action_invalids

        if self.config.algorithm.get("use_kl_in_reward", False):
            if "token_level_rewards" not in data.batch or "token_level_scores" not in data.batch:
                raise KeyError("KL-in-reward GAE value target requires token_level_rewards and token_level_scores")
            kl_reward_delta = (data.batch["token_level_rewards"] - data.batch["token_level_scores"]).float().sum(dim=-1)
            rewards = rewards + kl_reward_delta.to(device)

        return rewards

    def _attach_gae_value_targets(self, data):
        step_ppo_algos.ensure_step_indices(data)
        step_ppo_algos.ensure_step_sample_uids(data)
        shaped_step_rewards = self._build_shaped_immediate_rewards(data)

        aux_cfg = self._value_aux_config()
        _, step_returns = step_ppo_algos.compute_step_gae_advantage_return(
            step_rewards=shaped_step_rewards,
            values=data.batch["state_values"],
            traj_index=data.non_tensor_batch["traj_uid"],
            step_index=data.non_tensor_batch["step_idx"],
            gamma=aux_cfg.get("step_gamma", self.config.algorithm.get("gamma", 1.0)),
            lam=aux_cfg.get("step_lam", self.config.algorithm.get("lam", 1.0)),
            sample_index=data.non_tensor_batch.get("step_sample_uid", None),
        )
        # StepWisePPOActor only activates value loss when step_returns is present.
        data.batch["step_returns"] = step_returns.detach().clone()
        data.batch["value_step_rewards"] = shaped_step_rewards.detach().clone()
        return data

    def fit(self):
        original_compute_advantage = ray_trainer_module.compute_advantage
        original_adjust_batch = ray_trainer_module.adjust_batch

        def adjust_batch_with_value_aux_fields(config, data, *args, **kwargs):
            if config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                step_ppo_algos.ensure_step_indices(data)
                step_ppo_algos.ensure_step_sample_uids(data)
            return original_adjust_batch(config, data, *args, **kwargs)

        def compute_advantage_with_value_aux_target(data, adv_estimator, *args, **kwargs):
            data = original_compute_advantage(data, adv_estimator, *args, **kwargs)
            if adv_estimator == AdvantageEstimator.GiGPO:
                data = self._attach_gae_value_targets(data)
            return data

        ray_trainer_module.adjust_batch = adjust_batch_with_value_aux_fields
        ray_trainer_module.compute_advantage = compute_advantage_with_value_aux_target
        try:
            return super().fit()
        finally:
            ray_trainer_module.adjust_batch = original_adjust_batch
            ray_trainer_module.compute_advantage = original_compute_advantage
