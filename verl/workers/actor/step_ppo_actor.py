# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Data-parallel actor variant for shared-backbone step-wise PPO."""

import itertools
from typing import Optional

import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_policy_loss, compute_policy_loss_gspo, kl_penalty
from verl.utils.device import get_torch_device
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.utils.device import is_cuda_available, is_npu_available
from verl.workers.actor.dp_actor import DataParallelPPOActor

if is_cuda_available:
    try:
        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
    except ModuleNotFoundError:
        from einops import rearrange

        def index_first_axis(hidden_states, indices):
            return hidden_states[indices]

        def unpad_input(hidden_states, attention_mask):
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            batch, seqlen = attention_mask.shape
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            return index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices), indices, cu_seqlens, max_seqlen_in_batch

        def pad_input(hidden_states, indices, batch, seqlen):
            output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
            output[indices] = hidden_states
            return rearrange(output, "(b s) ... -> b s ...", b=batch, s=seqlen)
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input
else:
    from einops import rearrange

    def index_first_axis(hidden_states, indices):
        return hidden_states[indices]

    def unpad_input(hidden_states, attention_mask):
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        batch, seqlen = attention_mask.shape
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        return index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices), indices, cu_seqlens, max_seqlen_in_batch

    def pad_input(hidden_states, indices, batch, seqlen):
        output = hidden_states.new_zeros((batch * seqlen, *hidden_states.shape[1:]))
        output[indices] = hidden_states
        return rearrange(output, "(b s) ... -> b s ...", b=batch, s=seqlen)


class StepWisePPOActor(DataParallelPPOActor):
    """PPO actor with an action-level scalar value head.

    The policy loss remains token-level PPO. The value head predicts one scalar
    per environment action by selecting the hidden state at the pre-action
    boundary, then the scalar advantage/return is broadcast over generated
    action tokens for compatibility with the existing actor loss path.
    """

    def _value_head_config(self):
        return self.config.get("value_head", {})

    def _value_head_enabled(self) -> bool:
        cfg = self._value_head_config()
        return bool(cfg.get("enable", False))

    def _select_state_values(self, values: torch.Tensor, response_length: int) -> torch.Tensor:
        position = self._value_head_config().get("value_position", "pre_action_last_context_token")
        if position == "pre_action_last_context_token":
            value_index = -response_length - 1
        elif position == "first_response_token":
            value_index = -response_length
        elif position == "last_response_token":
            value_index = -1
        else:
            raise ValueError(f"Unsupported step value position: {position}")
        return values[:, value_index]

    def _split_step_output(self, output):
        if isinstance(output, tuple) and len(output) == 2:
            return output
        if hasattr(output, "step_values"):
            return output, output.step_values
        raise RuntimeError(
            "Step-wise PPO value head did not return step_values. "
            "The actor forward must run with return_step_values=True after attaching the shared value head."
        )

    def _forward_micro_batch_with_values(self, micro_batch, temperature, calculate_entropy=False):
        """Return token log-probs and one scalar value using a single actor forward."""
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    return_step_values=True,
                    **extra_args,
                )
                output, values_rmpad = self._split_step_output(output)
                if values_rmpad.dim() == 2 and values_rmpad.size(0) == 1:
                    values_rmpad = values_rmpad.squeeze(0)

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)
                    entropy_rmpad = output.entropy.squeeze(0)
                else:
                    logits_rmpad = output.logits.squeeze(0)
                    logits_rmpad.div_(temperature)
                    inplace_backward = not calculate_entropy
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)

                if self.use_ulysses_sp:
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    values_rmpad = gather_outpus_and_unpad(
                        values_rmpad,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )

                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                full_values = pad_input(
                    hidden_states=values_rmpad.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                ).squeeze(-1)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
                state_values = self._select_state_values(full_values, response_length)
                return entropy, log_probs, state_values

            extra_args = {}
            if self.use_fused_kernels:
                extra_args["temperature"] = temperature
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
                return_step_values=True,
                **extra_args,
            )
            output, values = self._split_step_output(output)

            if self.use_fused_kernels:
                log_probs = output.log_probs[:, -response_length - 1 : -1]
                entropy = output.entropy[:, -response_length - 1 : -1]
            else:
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1 : -1, :]
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                if calculate_entropy:
                    entropy = self.compute_entropy_from_logits(logits)

            state_values = self._select_state_values(values, response_length)
            return entropy, log_probs, state_values

    def _forward_value_micro_batch(self, micro_batch) -> torch.Tensor:
        """Return one scalar value per generated action."""
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    return_step_values=True,
                )
                output, values_rmpad = self._split_step_output(output)
                if values_rmpad.dim() == 2 and values_rmpad.size(0) == 1:
                    values_rmpad = values_rmpad.squeeze(0)

                if self.use_ulysses_sp:
                    values_rmpad = gather_outpus_and_unpad(
                        values_rmpad,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )

                full_values = pad_input(
                    hidden_states=values_rmpad.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                ).squeeze(-1)
                return self._select_state_values(full_values, response_length)

            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
                return_step_values=True,
            )
            output, values = self._split_step_output(output)
            return self._select_state_values(values, response_length)

    def compute_log_prob(self, data: DataProto, calculate_entropy=False):
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        state_values_lst = []
        calculate_values = self._value_head_enabled()
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                if calculate_values:
                    entropy, log_probs, state_values = self._forward_micro_batch_with_values(
                        micro_batch,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )
                    state_values_lst.append(state_values)
                else:
                    entropy, log_probs = self._forward_micro_batch(
                        micro_batch,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = torch.concat(entropy_lst, dim=0) if calculate_entropy else None
        state_values: Optional[torch.Tensor] = torch.concat(state_values_lst, dim=0) if calculate_values else None
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if entropys is not None:
                entropys = entropys[revert_indices]
            if state_values is not None:
                state_values = state_values[revert_indices]

        return log_probs, entropys, state_values

    def update_policy(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        multi_turn = data.meta_info.get("multi_turn", False)
        use_value_loss = self._value_head_enabled() and "step_returns" in data.batch

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if use_value_loss:
            select_keys.extend(["state_values", "step_returns"])
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for mini_batch in dataloader:
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_data in micro_batches:
                    if isinstance(micro_data, DataProto):
                        micro_data = {**micro_data.batch.to(get_torch_device().current_device()), **micro_data.non_tensor_batch}
                    else:
                        micro_data = micro_data.to(get_torch_device().current_device())

                    responses = micro_data["responses"]
                    response_length = responses.size(1)
                    attention_mask = micro_data["attention_mask"]
                    response_mask = micro_data["loss_mask"][:, -response_length:] if multi_turn else attention_mask[:, -response_length:]

                    old_log_prob = micro_data["old_log_probs"]
                    advantages = micro_data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = entropy_coeff != 0
                    if use_value_loss:
                        entropy, log_prob, current_values = self._forward_micro_batch_with_values(
                            micro_batch=micro_data,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                        )
                    else:
                        entropy, log_prob = self._forward_micro_batch(
                            micro_batch=micro_data,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                        )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = micro_data["ref_log_prob"]
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if use_value_loss:
                        current_values = current_values.float()
                        old_values = micro_data["state_values"].float()
                        step_returns = micro_data["step_returns"].float()
                        value_cfg = self._value_head_config()
                        cliprange_value = value_cfg.get("cliprange_value", 0.5)
                        value_loss_coef = value_cfg.get("value_loss_coef", 0.1)
                        values_clipped = old_values + torch.clamp(current_values - old_values, -cliprange_value, cliprange_value)
                        value_loss_unclipped = (current_values - step_returns).pow(2)
                        value_loss_clipped = (values_clipped - step_returns).pow(2)
                        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                        value_clipfrac = (value_loss_clipped > value_loss_unclipped).float().mean()
                        value_error = current_values - step_returns
                        value_mse = value_error.pow(2).mean()
                        value_rmse = torch.sqrt(value_mse.clamp_min(0.0))
                        value_mae = value_error.abs().mean()
                        value_return_var = torch.var(step_returns, unbiased=False)
                        value_residual_var = torch.var(step_returns - current_values, unbiased=False)
                        if value_return_var.detach().item() > 1e-8:
                            value_explained_variance = 1.0 - value_residual_var / (value_return_var + 1e-8)
                        else:
                            value_explained_variance = torch.zeros_like(value_return_var)
                        policy_loss = policy_loss + value_loss_coef * value_loss
                        append_to_dict(
                            metrics,
                            {
                                "actor/value_loss": value_loss.detach().item(),
                                "actor/value_clipfrac": value_clipfrac.detach().item(),
                                "actor/value_rmse": value_rmse.detach().item(),
                                "actor/value_mae": value_mae.detach().item(),
                                "actor/value_explained_variance": value_explained_variance.detach().item(),
                                "actor/value_pred_mean": current_values.detach().mean().item(),
                                "actor/value_pred_std": current_values.detach().std(unbiased=False).item(),
                                "actor/value_return_mean": step_returns.detach().mean().item(),
                                "actor/value_return_std": step_returns.detach().std(unbiased=False).item(),
                                "actor/value_loss_coef": value_loss_coef,
                            },
                        )

                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * (len(micro_data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    append_to_dict(
                        metrics,
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        },
                    )

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
        self.actor_optimizer.zero_grad()
        return metrics
