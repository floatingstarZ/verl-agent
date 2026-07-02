"""SSCA VIMPO-style critic-free value loss helpers.

The original VIMPO value loss is critic-free: it makes the cumulative
policy-reference log-ratio along a sampled response match a centered outcome
reward.  For SSCA retry we use the paired first trajectory as the baseline, so
`target = reward(traj2) - reward(traj1) + margin` after scaling/clipping.
"""

from __future__ import annotations

import torch

from verl.trainer.ppo.core_algos import kl_penalty


def ssca_vimpo_actor_config(config):
    return config.get("ssca_vimpo", {}) if config is not None else {}


def ssca_vimpo_actor_enabled(config) -> bool:
    cfg = ssca_vimpo_actor_config(config)
    return bool(cfg.get("enable", False)) and float(cfg.get("value_loss_coef", 0.0) or 0.0) > 0.0


def ssca_vimpo_append_select_keys(config, select_keys: list[str]) -> None:
    if not ssca_vimpo_actor_enabled(config):
        return
    for key in ["ref_log_prob", "ssca_vimpo_terminal_target", "ssca_vimpo_loss_mask"]:
        if key not in select_keys:
            select_keys.append(key)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _active_stats(values: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if bool(active.any()):
        active_values = values[active]
        return active_values.mean(), active_values.std(unbiased=False) if active_values.numel() > 1 else torch.zeros((), device=values.device, dtype=values.dtype)
    zero = torch.zeros((), device=values.device, dtype=values.dtype)
    return zero, zero


def compute_ssca_vimpo_value_loss(
    log_prob: torch.Tensor,
    ref_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    terminal_target: torch.Tensor,
    loss_mask: torch.Tensor,
    config,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute a VIMPO-style terminal value loss for SSCA retry rows.

    This is the SSCA adaptation of the VIMPO operational loss:

        1/2 * [sum_t beta * (log pi - log pi_ref - sg[KL_t]) - target]^2

    where `target` is the paired retry improvement over `traj1`.  The loss is
    active only on summary/traj2 rows selected by `loss_mask`.
    """
    cfg = ssca_vimpo_actor_config(config)
    beta = float(cfg.get("beta", 0.01) or 0.0)
    value_loss_coef = float(cfg.get("value_loss_coef", 0.05) or 0.0)
    kl_estimator = str(cfg.get("kl_estimator", "low_var_kl"))
    detach_kl = bool(cfg.get("detach_kl", True))
    value_loss_type = str(cfg.get("value_loss_type", "huber"))
    huber_delta = float(cfg.get("huber_delta", 1.0) or 1.0)

    dtype = log_prob.dtype
    response_mask = response_mask.to(dtype=dtype)
    row_mask = loss_mask.to(device=log_prob.device, dtype=dtype).reshape(-1)
    active = row_mask > 0
    target = terminal_target.to(device=log_prob.device, dtype=dtype).reshape(-1)
    ref_log_prob = ref_log_prob.to(device=log_prob.device, dtype=dtype)

    sampled_kl = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=kl_estimator).to(dtype=dtype)
    if detach_kl:
        sampled_kl_for_loss = sampled_kl.detach()
    else:
        sampled_kl_for_loss = sampled_kl

    log_ratio = (log_prob - ref_log_prob) * response_mask
    token_term = beta * (log_prob - ref_log_prob - sampled_kl_for_loss) * response_mask
    terminal_value = token_term.sum(dim=-1)
    residual = terminal_value - target

    if value_loss_type == "huber":
        abs_residual = residual.abs()
        delta = torch.as_tensor(huber_delta, dtype=dtype, device=log_prob.device)
        row_loss = torch.where(abs_residual <= delta, 0.5 * residual.square(), delta * (abs_residual - 0.5 * delta))
    elif value_loss_type == "squared_error":
        row_loss = 0.5 * residual.square()
    else:
        raise ValueError(f"Unsupported SSCA VIMPO value_loss_type={value_loss_type!r}")

    raw_loss = (row_loss * row_mask).sum() / row_mask.sum().clamp_min(1.0)
    scaled_loss = raw_loss * value_loss_coef

    token_mask = response_mask * row_mask.unsqueeze(-1)
    token_count = token_mask.sum().clamp_min(1.0)
    has_active = bool(active.any())
    active_target = target[active] if has_active else target[:0]
    active_value = terminal_value[active] if has_active else terminal_value[:0]
    active_residual = residual[active] if has_active else residual[:0]
    target_mean, target_std = _active_stats(target, active)
    value_mean, value_std = _active_stats(terminal_value, active)
    residual_mean, residual_std = _active_stats(residual, active)

    if active_target.numel() > 1 and float(active_target.std(unbiased=False).detach().item()) > 1e-8 and float(active_value.std(unbiased=False).detach().item()) > 1e-8:
        target_centered = active_target - active_target.mean()
        value_centered = active_value - active_value.mean()
        corr = (target_centered * value_centered).mean() / (target_centered.std(unbiased=False) * value_centered.std(unbiased=False) + 1e-8)
    else:
        corr = torch.zeros((), dtype=dtype, device=log_prob.device)

    if active_target.numel() > 0:
        sign_acc = ((active_target >= 0) == (active_value >= 0)).to(dtype=dtype).mean()
        target_pos = (active_target > 0).to(dtype=dtype).mean()
        value_pos = (active_value > 0).to(dtype=dtype).mean()
        rmse = torch.sqrt(active_residual.square().mean().clamp_min(0.0))
        mae = active_residual.abs().mean()
    else:
        sign_acc = target_pos = value_pos = rmse = mae = torch.zeros((), dtype=dtype, device=log_prob.device)

    metrics = {
        "actor/ssca_vimpo_value_loss": scaled_loss.detach().item(),
        "actor/ssca_vimpo_value_loss_raw": raw_loss.detach().item(),
        "actor/ssca_vimpo_value_loss_x1e4": scaled_loss.detach().item() * 1.0e4,
        "actor/ssca_vimpo_value_loss_raw_x1e4": raw_loss.detach().item() * 1.0e4,
        "actor/ssca_vimpo_beta": beta,
        "actor/ssca_vimpo_value_loss_coef": value_loss_coef,
        "actor/ssca_vimpo_active_row_ratio": row_mask.mean().detach().item(),
        "actor/ssca_vimpo_active_row_count": row_mask.sum().detach().item(),
        "actor/ssca_vimpo_active_token_count": token_mask.sum().detach().item(),
    }
    if has_active:
        metrics.update(
            {
                "actor/ssca_vimpo_terminal_value_mean_active": value_mean.detach().item(),
                "actor/ssca_vimpo_terminal_value_std_active": value_std.detach().item(),
                "actor/ssca_vimpo_terminal_target_mean_active": target_mean.detach().item(),
                "actor/ssca_vimpo_terminal_target_std_active": target_std.detach().item(),
                "actor/ssca_vimpo_terminal_target_abs_mean_active": active_target.abs().mean().detach().item(),
                "actor/ssca_vimpo_residual_mean_active": residual_mean.detach().item(),
                "actor/ssca_vimpo_residual_std_active": residual_std.detach().item(),
                "actor/ssca_vimpo_rmse_active": rmse.detach().item(),
                "actor/ssca_vimpo_mae_active": mae.detach().item(),
                "actor/ssca_vimpo_value_target_corr_active": corr.detach().item(),
                "actor/ssca_vimpo_sign_acc_active": sign_acc.detach().item(),
                "actor/ssca_vimpo_target_positive_rate_active": target_pos.detach().item(),
                "actor/ssca_vimpo_value_positive_rate_active": value_pos.detach().item(),
                "actor/ssca_vimpo_logratio_token_mean_active": (log_ratio * token_mask).sum().detach().item() / float(token_count.detach().item()),
                "actor/ssca_vimpo_kl_token_mean_active": (sampled_kl.detach() * token_mask).sum().detach().item() / float(token_count.detach().item()),
                "actor/ssca_vimpo_token_term_sum_mean_active": value_mean.detach().item(),
            }
        )
    return scaled_loss, metrics
