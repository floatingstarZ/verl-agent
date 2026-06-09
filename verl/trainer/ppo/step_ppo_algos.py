# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Step-wise PPO utilities for agentic multi-step rollouts.

The implementation is intentionally isolated from core PPO/GRPO code so the
experimental algorithm can evolve without changing the default trainer path.
"""

from collections import defaultdict

import numpy as np
import torch


def _whiten_1d(values: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    if values.numel() == 0:
        return values
    mean = values.mean()
    std = values.std(unbiased=False)
    return (values - mean) / (std + epsilon)


def compute_grpo_episode_advantage_scalar(
    episode_scores: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std: bool = True,
):
    """Compute one GRPO-style episode advantage scalar for each step sample."""
    scores = episode_scores.float().clone()
    sample_advantages = torch.zeros_like(scores)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()

    with torch.no_grad():
        for i in range(scores.shape[0]):
            pair = (index[i], traj_index[i])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            id2score[index[i]].append(scores[i])

        for idx, group_scores in id2score.items():
            stacked = torch.stack(group_scores).to(scores.device)
            if stacked.numel() <= 1:
                id2mean[idx] = stacked.mean() if stacked.numel() == 1 else scores.new_tensor(0.0)
                id2std[idx] = scores.new_tensor(1.0)
            else:
                id2mean[idx] = stacked.mean()
                id2std[idx] = stacked.std(unbiased=False)

        for i in range(scores.shape[0]):
            centered = scores[i] - id2mean[index[i]]
            if norm_adv_by_std:
                centered = centered / (id2std[index[i]] + epsilon)
            sample_advantages[i] = centered

    return sample_advantages


def compute_gigpo_episode_advantage_scalar(
    episode_scores: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    mode: str = "mean_norm",
):
    """Compute GiGPO-style row-level episode advantage scalars.

    This mirrors GiGPO's episode_norm_reward default behavior where episode
    score statistics are computed over all step rows in the same uid group.
    """
    scores = episode_scores.float().clone()
    sample_advantages = torch.zeros_like(scores)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    normalized_mode = str(mode).lower()

    if normalized_mode in {"gigpo_mean_norm", "mean_norm"}:
        remove_std = True
    elif normalized_mode in {"gigpo_mean_std_norm", "mean_std_norm"}:
        remove_std = False
    else:
        raise ValueError(f"Unsupported GiGPO episode advantage mode: {mode}")

    with torch.no_grad():
        for i in range(scores.shape[0]):
            id2score[index[i]].append(scores[i])

        for idx, group_scores in id2score.items():
            stacked = torch.stack(group_scores).to(scores.device)
            if stacked.numel() <= 1:
                id2mean[idx] = scores.new_tensor(0.0)
                id2std[idx] = scores.new_tensor(1.0)
            else:
                id2mean[idx] = stacked.mean()
                id2std[idx] = stacked.std(unbiased=True)

        for i in range(scores.shape[0]):
            centered = scores[i] - id2mean[index[i]]
            if not remove_std:
                centered = centered / (id2std[index[i]] + epsilon)
            sample_advantages[i] = centered

    return sample_advantages


def compute_episode_advantage_scalar(
    episode_scores: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    mode: str = "grpo_mean_std",
):
    """Dispatch episode advantage calculation for StepPPO variants."""
    normalized_mode = str(mode).lower()
    if normalized_mode in {"grpo", "grpo_mean_std", "grpo_std"}:
        return compute_grpo_episode_advantage_scalar(
            episode_scores=episode_scores,
            index=index,
            traj_index=traj_index,
            epsilon=epsilon,
            norm_adv_by_std=True,
        )
    if normalized_mode in {"grpo_mean", "grpo_no_std"}:
        return compute_grpo_episode_advantage_scalar(
            episode_scores=episode_scores,
            index=index,
            traj_index=traj_index,
            epsilon=epsilon,
            norm_adv_by_std=False,
        )
    if normalized_mode in {"gigpo_mean_norm", "mean_norm", "gigpo_mean_std_norm", "mean_std_norm"}:
        return compute_gigpo_episode_advantage_scalar(
            episode_scores=episode_scores,
            index=index,
            epsilon=epsilon,
            mode=normalized_mode,
        )
    raise ValueError(f"Unsupported StepPPO episode advantage mode: {mode}")


def compute_step_gae_advantage_return(
    step_rewards: torch.Tensor,
    values: torch.Tensor,
    traj_index: np.ndarray,
    step_index: np.ndarray,
    gamma: float = 1.0,
    lam: float = 1.0,
    sample_index: np.ndarray | None = None,
):
    """Compute GAE over environment steps, grouped by trajectory id.

    ``adjust_batch(mode=copy)`` can duplicate rows to satisfy distributed batch
    divisibility. ``sample_index`` identifies such duplicates so they do not
    become fake extra environment transitions.
    """
    rewards = step_rewards.float()
    old_values = values.float()
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    canonical_for = list(range(rewards.shape[0]))
    canonical_indices = list(range(rewards.shape[0]))
    if sample_index is not None:
        seen = {}
        canonical_indices = []
        for i, sample_uid in enumerate(sample_index):
            if sample_uid in seen:
                canonical_for[i] = seen[sample_uid]
            else:
                seen[sample_uid] = i
                canonical_indices.append(i)

    traj2indices = defaultdict(list)
    for i in canonical_indices:
        traj2indices[traj_index[i]].append(i)

    with torch.no_grad():
        for _, indices in traj2indices.items():
            ordered = sorted(indices, key=lambda idx: int(step_index[idx]))
            lastgaelam = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
            for pos in reversed(range(len(ordered))):
                idx = ordered[pos]
                if pos < len(ordered) - 1:
                    next_idx = ordered[pos + 1]
                    next_value = old_values[next_idx]
                    nonterminal = 1.0
                else:
                    next_value = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
                    nonterminal = 0.0
                delta = rewards[idx] + gamma * next_value * nonterminal - old_values[idx]
                lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
                advantages[idx] = lastgaelam
                returns[idx] = advantages[idx] + old_values[idx]

        if sample_index is not None:
            for i, canonical_idx in enumerate(canonical_for):
                if i != canonical_idx:
                    advantages[i] = advantages[canonical_idx]
                    returns[i] = returns[canonical_idx]

    return advantages, returns


def compute_step_ppo_advantage_return(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    state_values: torch.Tensor,
    step_rewards: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    step_index: np.ndarray,
    gamma: float = 1.0,
    lam: float = 1.0,
    step_advantage_w: float = 1.0,
    normalize_episode_advantage: bool = True,
    normalize_step_advantage: bool = True,
    normalize_final_advantage: bool = False,
    episode_scores: torch.Tensor | None = None,
    epsilon: float = 1e-6,
    sample_index: np.ndarray | None = None,
    episode_advantage_mode: str = "grpo_mean_std",
    final_advantage_mode: str = "average",
):
    """Compute token-broadcast advantages for shared-backbone step-wise PPO."""
    if episode_scores is None:
        episode_scores = token_level_rewards.sum(dim=-1)

    episode_advantages = compute_episode_advantage_scalar(
        episode_scores=episode_scores,
        index=index,
        traj_index=traj_index,
        epsilon=epsilon,
        mode=episode_advantage_mode,
    )
    step_advantages, step_returns = compute_step_gae_advantage_return(
        step_rewards=step_rewards,
        values=state_values,
        traj_index=traj_index,
        step_index=step_index,
        gamma=gamma,
        lam=lam,
        sample_index=sample_index,
    )

    if normalize_episode_advantage:
        episode_advantages = _whiten_1d(episode_advantages, epsilon=epsilon)
    if normalize_step_advantage:
        step_advantages = _whiten_1d(step_advantages, epsilon=epsilon)

    normalized_final_mode = str(final_advantage_mode).lower()
    if normalized_final_mode in {"average", "mean", "v1"}:
        final_step_advantages = (episode_advantages + step_advantage_w * step_advantages) / (1.0 + step_advantage_w)
    elif normalized_final_mode in {"direct", "sum", "gigpo"}:
        final_step_advantages = episode_advantages + step_advantage_w * step_advantages
    else:
        raise ValueError(f"Unsupported StepPPO final advantage mode: {final_advantage_mode}")

    if normalize_final_advantage:
        final_step_advantages = _whiten_1d(final_step_advantages, epsilon=epsilon)
    advantages = final_step_advantages.unsqueeze(-1) * response_mask
    returns = step_returns.unsqueeze(-1) * response_mask

    return {
        "advantages": advantages,
        "returns": returns,
        "episode_advantages": episode_advantages,
        "step_advantages": step_advantages,
        "step_returns": step_returns,
        "final_step_advantages": final_step_advantages,
    }



def is_step_ppo_estimator(adv_estimator) -> bool:
    return str(adv_estimator).lower().split(".")[-1] == "step_ppo"


def ensure_step_indices(data) -> None:
    """Attach step_idx if the rollout collector did not emit it.

    This must run before batch reordering. The current collector stores samples
    trajectory-by-trajectory, so occurrence count within each traj_uid recovers
    the environment step index.
    """
    if "step_idx" in data.non_tensor_batch:
        return
    if "traj_uid" not in data.non_tensor_batch:
        raise KeyError("step-wise PPO requires traj_uid in non_tensor_batch")

    traj_counts = defaultdict(int)
    step_idx = []
    for traj_uid in data.non_tensor_batch["traj_uid"]:
        step_idx.append(traj_counts[traj_uid])
        traj_counts[traj_uid] += 1
    data.non_tensor_batch["step_idx"] = np.asarray(step_idx, dtype=np.int32)


def ensure_step_rewards(data, reward_key: str = "rewards", tensor_key: str = "step_rewards") -> None:
    """Attach scalar per-action rewards as a tensor for step-wise GAE."""
    if tensor_key in data.batch:
        return
    if reward_key not in data.non_tensor_batch:
        raise KeyError(f"step-wise PPO requires {reward_key!r} in non_tensor_batch")

    rewards = np.asarray(data.non_tensor_batch[reward_key], dtype=np.float32).reshape(-1)
    device = data.batch["input_ids"].device if "input_ids" in data.batch else data.batch["responses"].device
    data.batch[tensor_key] = torch.as_tensor(rewards, dtype=torch.float32, device=device)


def ensure_step_sample_uids(data) -> None:
    """Attach stable row ids before adjust_batch may copy rows."""
    if "step_sample_uid" in data.non_tensor_batch:
        return
    ensure_step_indices(data)
    sample_uids = [
        f"{traj_uid}:{int(step_idx)}:{row_idx}"
        for row_idx, (traj_uid, step_idx) in enumerate(zip(data.non_tensor_batch["traj_uid"], data.non_tensor_batch["step_idx"]))
    ]
    data.non_tensor_batch["step_sample_uid"] = np.asarray(sample_uids, dtype=object)


def prepare_step_ppo_rollout_batch(data) -> None:
    """Attach step-wise fields before adjust/reorder duplicates the batch."""
    ensure_step_indices(data)
    ensure_step_sample_uids(data)
    ensure_step_rewards(data)


def compute_step_ppo_advantage_data(data, algorithm_config, multi_turn: bool = False, epsilon: float = 1e-6):
    """Compute and attach step-wise PPO advantages/returns to a DataProto."""
    if "response_mask" not in data.batch:
        responses = data.batch["responses"]
        response_length = responses.size(1)
        data.batch["response_mask"] = data.batch["attention_mask"][:, -response_length:]

    response_mask = data.batch["response_mask"]
    if multi_turn and "loss_mask" in data.batch:
        response_length = response_mask.size(1)
        response_mask = data.batch["loss_mask"][:, -response_length:]

    ensure_step_indices(data)
    ensure_step_rewards(data)
    if "state_values" not in data.batch:
        raise KeyError("step-wise PPO requires state_values from the shared actor value head")

    if "episode_rewards" in data.non_tensor_batch:
        episode_scores = torch.as_tensor(
            np.asarray(data.non_tensor_batch["episode_rewards"], dtype=np.float32).reshape(-1),
            dtype=torch.float32,
            device=data.batch["state_values"].device,
        )
    else:
        episode_scores = None

    step_cfg = algorithm_config.get("step_ppo", {})
    result = compute_step_ppo_advantage_return(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=response_mask,
        state_values=data.batch["state_values"],
        step_rewards=data.batch["step_rewards"],
        index=data.non_tensor_batch["uid"],
        traj_index=data.non_tensor_batch["traj_uid"],
        step_index=data.non_tensor_batch["step_idx"],
        gamma=step_cfg.get("step_gamma", algorithm_config.get("gamma", 1.0)),
        lam=step_cfg.get("step_lam", algorithm_config.get("lam", 1.0)),
        step_advantage_w=step_cfg.get("step_advantage_w", 1.0),
        normalize_episode_advantage=step_cfg.get("normalize_episode_advantage", True),
        normalize_step_advantage=step_cfg.get("normalize_step_advantage", True),
        normalize_final_advantage=step_cfg.get("normalize_final_advantage", False),
        episode_scores=episode_scores,
        epsilon=epsilon,
        sample_index=data.non_tensor_batch.get("step_sample_uid", None),
        episode_advantage_mode=step_cfg.get("episode_mode", step_cfg.get("episode_advantage_mode", "grpo_mean_std")),
        final_advantage_mode=step_cfg.get("final_advantage_mode", "average"),
    )
    for key, value in result.items():
        data.batch[key] = value
    return data
