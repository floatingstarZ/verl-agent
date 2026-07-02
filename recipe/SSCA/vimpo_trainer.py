"""SSCA VIMPO-style target attachment.

The trainer keeps the existing GRPO advantage path unchanged and adds tensors
for a critic-free VIMPO-style value loss.  The actor loss then encourages the
retry branch cumulative policy-reference log-ratio to match the paired reward
improvement over the first attempt.
"""

from __future__ import annotations

import numpy as np
import torch

from verl.trainer.ppo import ray_trainer as ray_trainer_module
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class SSCAVIMPORayTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        config = kwargs.get("config", args[0] if args else None)
        if config is None:
            raise ValueError("SSCAVIMPORayTrainer requires a config")
        super().__init__(*args, **kwargs)
        self.use_critic = False

    def _vimpo_config(self):
        return self.config.algorithm.get("ssca_vimpo", {})

    def _attach_pair_vimpo_targets(self, data):
        cfg = self._vimpo_config()
        reward_scale = float(cfg.get("reward_scale", 10.0) or 1.0)
        margin = float(cfg.get("margin", 0.0) or 0.0)
        target_clip = float(cfg.get("target_clip", 2.0) or 0.0)
        target_distribution = str(cfg.get("target_distribution", "split_by_retry_rows"))
        if target_distribution not in {"split_by_retry_rows", "per_row_full"}:
            raise ValueError(f"Unsupported SSCA VIMPO target_distribution={target_distribution!r}")

        phases = np.asarray(data.non_tensor_batch.get("ssca_phase", []), dtype=object)
        traj_uids = np.asarray(data.non_tensor_batch.get("traj_uid", []), dtype=object)
        linked_first = np.asarray(data.non_tensor_batch.get("ssca_linked_first_traj_uid", []), dtype=object)
        raw_rewards = data.non_tensor_batch.get("ssca_env_episode_rewards", data.non_tensor_batch.get("episode_rewards"))
        if raw_rewards is None or phases.size == 0:
            raise KeyError("SSCA VIMPO requires ssca_phase and ssca_env_episode_rewards metadata")
        rewards = np.asarray(raw_rewards, dtype=np.float32).reshape(-1)

        first_rewards: dict[object, float] = {}
        for idx, phase in enumerate(phases):
            if str(phase) == "traj1" and traj_uids[idx] not in first_rewards:
                first_rewards[traj_uids[idx]] = float(rewards[idx])

        targets = np.zeros_like(rewards, dtype=np.float32)
        raw_targets = np.zeros_like(rewards, dtype=np.float32)
        branch_targets = np.zeros_like(rewards, dtype=np.float32)
        raw_branch_targets = np.zeros_like(rewards, dtype=np.float32)
        masks = np.zeros_like(rewards, dtype=np.float32)
        deltas = np.zeros_like(rewards, dtype=np.float32)
        is_summary = np.zeros_like(rewards, dtype=np.float32)
        is_traj2 = np.zeros_like(rewards, dtype=np.float32)
        retry_row_counts: dict[object, int] = {}
        for idx, phase_obj in enumerate(phases):
            phase = str(phase_obj)
            if phase in {"summary", "traj2"} and linked_first[idx] in first_rewards:
                retry_row_counts[traj_uids[idx]] = retry_row_counts.get(traj_uids[idx], 0) + 1

        missing_ref = 0
        clipped = 0
        for idx, phase_obj in enumerate(phases):
            phase = str(phase_obj)
            if phase not in {"summary", "traj2"}:
                continue
            ref_uid = linked_first[idx]
            if ref_uid not in first_rewards:
                missing_ref += 1
                continue
            ref_reward = first_rewards[ref_uid]
            delta = float(rewards[idx]) - ref_reward
            raw_branch_target = (delta + margin) / reward_scale
            branch_target = raw_branch_target
            if target_clip > 0:
                clipped_target = float(np.clip(branch_target, -target_clip, target_clip))
                clipped += int(clipped_target != branch_target)
                branch_target = clipped_target
            row_divisor = max(1, retry_row_counts.get(traj_uids[idx], 1)) if target_distribution == "split_by_retry_rows" else 1
            raw_target = raw_branch_target / row_divisor
            target = branch_target / row_divisor
            deltas[idx] = delta
            raw_targets[idx] = raw_target
            targets[idx] = target
            raw_branch_targets[idx] = raw_branch_target
            branch_targets[idx] = branch_target
            masks[idx] = 1.0
            is_summary[idx] = float(phase == "summary")
            is_traj2[idx] = float(phase == "traj2")

        device = data.batch["responses"].device
        data.batch["ssca_vimpo_terminal_target"] = torch.as_tensor(targets, dtype=torch.float32, device=device)
        data.batch["ssca_vimpo_raw_terminal_target"] = torch.as_tensor(raw_targets, dtype=torch.float32, device=device)
        data.batch["ssca_vimpo_branch_terminal_target"] = torch.as_tensor(branch_targets, dtype=torch.float32, device=device)
        data.batch["ssca_vimpo_raw_branch_terminal_target"] = torch.as_tensor(raw_branch_targets, dtype=torch.float32, device=device)
        data.batch["ssca_vimpo_reward_delta"] = torch.as_tensor(deltas, dtype=torch.float32, device=device)
        data.batch["ssca_vimpo_loss_mask"] = torch.as_tensor(masks, dtype=torch.float32, device=device)
        data.batch["ssca_vimpo_is_summary"] = torch.as_tensor(is_summary, dtype=torch.float32, device=device)
        data.batch["ssca_vimpo_is_traj2"] = torch.as_tensor(is_traj2, dtype=torch.float32, device=device)

        active_targets = targets[masks > 0]
        active_raw_targets = raw_targets[masks > 0]
        active_branch_targets = branch_targets[masks > 0]
        active_raw_branch_targets = raw_branch_targets[masks > 0]
        active_deltas = deltas[masks > 0]
        metric_count = len(rewards)

        def repeated(value):
            return np.full(metric_count, float(value), dtype=np.float32)

        data.non_tensor_batch["ssca_metric/vimpo/terminal_target_mean"] = repeated(active_targets.mean() if active_targets.size else 0.0)
        data.non_tensor_batch["ssca_metric/vimpo/terminal_target_std"] = repeated(active_targets.std() if active_targets.size else 0.0)
        data.non_tensor_batch["ssca_metric/vimpo/raw_terminal_target_mean"] = repeated(active_raw_targets.mean() if active_raw_targets.size else 0.0)
        data.non_tensor_batch["ssca_metric/vimpo/raw_terminal_target_std"] = repeated(active_raw_targets.std() if active_raw_targets.size else 0.0)
        data.non_tensor_batch["ssca_metric/vimpo/branch_terminal_target_mean"] = repeated(active_branch_targets.mean() if active_branch_targets.size else 0.0)
        data.non_tensor_batch["ssca_metric/vimpo/branch_terminal_target_std"] = repeated(active_branch_targets.std() if active_branch_targets.size else 0.0)
        data.non_tensor_batch["ssca_metric/vimpo/raw_branch_terminal_target_mean"] = repeated(active_raw_branch_targets.mean() if active_raw_branch_targets.size else 0.0)
        data.non_tensor_batch["ssca_metric/vimpo/raw_branch_terminal_target_std"] = repeated(active_raw_branch_targets.std() if active_raw_branch_targets.size else 0.0)
        data.non_tensor_batch["ssca_metric/vimpo/reward_delta_mean"] = repeated(active_deltas.mean() if active_deltas.size else 0.0)
        data.non_tensor_batch["ssca_metric/vimpo/reward_delta_std"] = repeated(active_deltas.std() if active_deltas.size else 0.0)
        data.non_tensor_batch["ssca_metric/vimpo/reward_delta_positive_rate"] = repeated(float(np.mean(active_deltas > 0)) if active_deltas.size else 0.0)
        data.non_tensor_batch["ssca_metric/vimpo/reward_delta_negative_rate"] = repeated(float(np.mean(active_deltas < 0)) if active_deltas.size else 0.0)
        data.non_tensor_batch["ssca_metric/vimpo/target_positive_rate"] = repeated(float(np.mean(active_targets > 0)) if active_targets.size else 0.0)
        data.non_tensor_batch["ssca_metric/vimpo/value_loss_mask_rate"] = repeated(float(np.mean(masks > 0)) if masks.size else 0.0)
        data.non_tensor_batch["ssca_metric/vimpo/summary_mask_rate"] = repeated(float(np.mean(is_summary > 0)) if is_summary.size else 0.0)
        data.non_tensor_batch["ssca_metric/vimpo/traj2_mask_rate"] = repeated(float(np.mean(is_traj2 > 0)) if is_traj2.size else 0.0)
        data.non_tensor_batch["ssca_metric/vimpo/missing_ref_count"] = repeated(missing_ref)
        data.non_tensor_batch["ssca_metric/vimpo/target_clip_count"] = repeated(clipped)
        data.non_tensor_batch["ssca_metric/vimpo/reward_scale"] = repeated(reward_scale)
        data.non_tensor_batch["ssca_metric/vimpo/margin"] = repeated(margin)
        data.non_tensor_batch["ssca_metric/vimpo/split_by_retry_rows"] = repeated(float(target_distribution == "split_by_retry_rows"))
        return data

    def fit(self):
        original_compute_advantage = ray_trainer_module.compute_advantage

        def compute_advantage_with_ssca_vimpo(data, adv_estimator, *args, **kwargs):
            data = original_compute_advantage(data, adv_estimator, *args, **kwargs)
            return self._attach_pair_vimpo_targets(data)

        ray_trainer_module.compute_advantage = compute_advantage_with_ssca_vimpo
        try:
            return super().fit()
        finally:
            ray_trainer_module.compute_advantage = original_compute_advantage
