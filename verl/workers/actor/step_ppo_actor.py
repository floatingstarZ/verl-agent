# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Data-parallel actor variant for shared-backbone step-wise PPO."""

import itertools
from typing import Optional

import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_c_rf_ntf_policy_loss, compute_policy_loss, compute_policy_loss_gspo, kl_penalty
from verl.utils.device import get_torch_device
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from recipe.SSCA.vimpo_core import compute_ssca_vimpo_value_loss, ssca_vimpo_actor_enabled, ssca_vimpo_append_select_keys
from verl.utils.ulysses import gather_outpus_and_unpad, get_ulysses_sequence_parallel_rank, ulysses_pad, ulysses_pad_and_slice_inputs
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

    def _state_value_enabled(self) -> bool:
        cfg = self._value_head_config()
        return self._value_head_enabled() and bool(cfg.get("compute_state_values", True))

    def _reasoning_value_config(self):
        return self.config.get("reasoning_value_aux", {})

    def _reasoning_value_enabled(self) -> bool:
        cfg = self._reasoning_value_config()
        return self._value_head_enabled() and bool(cfg.get("enable", False))

    def _select_state_values(self, values: torch.Tensor, response_length: int) -> torch.Tensor:
        value_index = self._state_value_index(response_length)
        return values[:, value_index]

    def _state_value_index(self, response_length: int) -> int:
        position = self._value_head_config().get("value_position", "pre_action_last_context_token")
        if position == "pre_action_last_context_token":
            return -response_length - 1
        elif position == "first_response_token":
            return -response_length
        elif position == "last_response_token":
            return -1
        raise ValueError(f"Unsupported step value position: {position}")

    def _state_value_positions(self, batch_size: int, seqlen: int, response_length: int, device) -> torch.Tensor:
        value_index = self._state_value_index(response_length)
        if value_index < 0:
            value_index = seqlen + value_index
        return torch.full((batch_size,), value_index, dtype=torch.long, device=device)

    def _state_value_rmpad_indices(
        self,
        indices: torch.Tensor,
        batch_size: int,
        seqlen: int,
        response_length: int,
        device,
    ) -> torch.Tensor:
        value_positions = self._state_value_positions(batch_size, seqlen, response_length, device)
        flat_value_positions = torch.arange(batch_size, device=device, dtype=torch.long) * seqlen + value_positions
        reverse_indices = torch.empty(batch_size * seqlen, dtype=torch.long, device=device)
        reverse_indices[indices.to(device=device, dtype=torch.long)] = torch.arange(indices.numel(), dtype=torch.long, device=device)
        return reverse_indices[flat_value_positions]

    def _local_ulysses_value_indices(self, rmpad_indices: torch.Tensor, local_seq_len: int) -> torch.Tensor:
        if not self.use_ulysses_sp:
            return rmpad_indices
        rank = get_ulysses_sequence_parallel_rank()
        local_start = rank * local_seq_len
        local_end = local_start + local_seq_len
        local_mask = (rmpad_indices >= local_start) & (rmpad_indices < local_end)
        return rmpad_indices[local_mask] - local_start

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
                step_value_indices = self._state_value_rmpad_indices(indices, batch_size, seqlen, response_length, input_ids.device)

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
                local_step_value_indices = self._local_ulysses_value_indices(step_value_indices, input_ids_rmpad.size(1))

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
                    step_value_indices=local_step_value_indices,
                    step_value_output_shape=(1, input_ids_rmpad.size(1)),
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
            step_value_indices = self._state_value_positions(batch_size, seqlen, response_length, input_ids.device)
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
                return_step_values=True,
                step_value_indices=step_value_indices,
                step_value_output_shape=(batch_size, seqlen),
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
                step_value_indices = self._state_value_rmpad_indices(indices, batch_size, seqlen, response_length, input_ids.device)

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

                local_step_value_indices = self._local_ulysses_value_indices(step_value_indices, input_ids_rmpad.size(1))
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    return_step_values=True,
                    step_value_indices=local_step_value_indices,
                    step_value_output_shape=(1, input_ids_rmpad.size(1)),
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

            step_value_indices = self._state_value_positions(batch_size, seqlen, response_length, input_ids.device)
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
                return_step_values=True,
                step_value_indices=step_value_indices,
                step_value_output_shape=(batch_size, seqlen),
            )
            output, values = self._split_step_output(output)
            return self._select_state_values(values, response_length)

    def _forward_selected_value_micro_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        value_indices: torch.Tensor,
        multi_modal_inputs: Optional[dict] = None,
    ) -> torch.Tensor:
        """Return one scalar from selected sequence positions for each row."""
        multi_modal_inputs = multi_modal_inputs or {}
        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            batch_size, seqlen = input_ids.shape
            value_indices = value_indices.to(device=input_ids.device, dtype=torch.long).clamp_(0, seqlen - 1)
            if position_ids.dim() == 3:
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                flat_value_positions = torch.arange(batch_size, device=input_ids.device, dtype=torch.long) * seqlen + value_indices
                reverse_indices = torch.empty(batch_size * seqlen, dtype=torch.long, device=input_ids.device)
                reverse_indices[indices.to(device=input_ids.device, dtype=torch.long)] = torch.arange(indices.numel(), dtype=torch.long, device=input_ids.device)
                step_value_indices = reverse_indices[flat_value_positions]

                if self.use_ulysses_sp:
                    is_vlm_model = bool(multi_modal_inputs)
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
                local_step_value_indices = self._local_ulysses_value_indices(step_value_indices, input_ids_rmpad.size(1))
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    return_step_values=True,
                    step_value_indices=local_step_value_indices,
                    step_value_output_shape=(1, input_ids_rmpad.size(1)),
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
                batch_indices = torch.arange(batch_size, device=input_ids.device, dtype=torch.long)
                return full_values[batch_indices, value_indices]

            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
                return_step_values=True,
                step_value_indices=value_indices,
                step_value_output_shape=(batch_size, seqlen),
            )
            output, values = self._split_step_output(output)
            batch_indices = torch.arange(batch_size, device=input_ids.device, dtype=torch.long)
            return values[batch_indices, value_indices]

    def _forward_reasoning_value_micro_batch(self, micro_batch) -> torch.Tensor:
        """Return one scalar value for each reasoning-value side branch."""
        multi_modal_inputs = {}
        return self._forward_selected_value_micro_batch(
            input_ids=micro_batch["reason_value_input_ids"],
            attention_mask=micro_batch["reason_value_attention_mask"],
            position_ids=micro_batch["reason_value_position_ids"],
            value_indices=micro_batch["reason_value_indices"],
            multi_modal_inputs=multi_modal_inputs,
        )

    @staticmethod
    def _scalar_value_regression_metrics(
        prefix: str,
        current_values: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
        value_loss: torch.Tensor,
        loss_coef: float,
    ) -> dict:
        mask_sum = loss_mask.sum().clamp_min(1.0)
        error = current_values - targets
        mse = (error.pow(2) * loss_mask).sum() / mask_sum
        rmse = torch.sqrt(mse.clamp_min(0.0))
        mae = (error.abs() * loss_mask).sum() / mask_sum
        active_mask = loss_mask > 0
        active_values = current_values[active_mask]
        active_targets = targets[active_mask]
        if active_targets.numel() > 1:
            target_var = torch.var(active_targets, unbiased=False)
            residual_var = torch.var(active_targets - active_values, unbiased=False)
            value_var = torch.var(active_values, unbiased=False)
        else:
            target_var = torch.zeros((), device=current_values.device, dtype=current_values.dtype)
            residual_var = torch.zeros((), device=current_values.device, dtype=current_values.dtype)
            value_var = torch.zeros((), device=current_values.device, dtype=current_values.dtype)
        if target_var.detach().item() > 1e-8:
            explained_variance = 1.0 - residual_var / (target_var + 1e-8)
        else:
            explained_variance = torch.zeros_like(target_var)
        if active_targets.numel() > 1 and target_var.detach().item() > 1e-8 and value_var.detach().item() > 1e-8:
            centered_values = active_values - active_values.mean()
            centered_targets = active_targets - active_targets.mean()
            corr = (centered_values * centered_targets).mean() / torch.sqrt(value_var * target_var + 1e-8)
        else:
            corr = torch.zeros_like(target_var)
        return {
            f"{prefix}/loss": value_loss.detach().item(),
            f"{prefix}/rmse": rmse.detach().item(),
            f"{prefix}/mae": mae.detach().item(),
            f"{prefix}/explained_variance": explained_variance.detach().item(),
            f"{prefix}/corr": corr.detach().item(),
            f"{prefix}/pred_mean": active_values.detach().mean().item() if active_values.numel() else 0.0,
            f"{prefix}/pred_std": active_values.detach().std(unbiased=False).item() if active_values.numel() else 0.0,
            f"{prefix}/target_mean": active_targets.detach().mean().item() if active_targets.numel() else 0.0,
            f"{prefix}/target_std": active_targets.detach().std(unbiased=False).item() if active_targets.numel() else 0.0,
            f"{prefix}/mask_ratio": loss_mask.detach().mean().item(),
            f"{prefix}/active_count": loss_mask.detach().sum().item(),
            f"{prefix}/loss_coef": loss_coef,
        }

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
        calculate_values = self._state_value_enabled()
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

    def compute_reasoning_values(self, data: DataProto) -> torch.Tensor:
        """Compute scalar value-head predictions for reasoning-value side branches."""
        self.actor_module.eval()

        micro_batch_size = int(data.meta_info.get("micro_batch_size", self.config.ppo_micro_batch_size_per_gpu))
        use_dynamic_bsz = bool(data.meta_info.get("use_dynamic_bsz", self.config.get("use_dynamic_bsz", False)))
        select_keys = [
            "reason_value_input_ids",
            "reason_value_attention_mask",
            "reason_value_position_ids",
            "reason_value_indices",
        ]
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            max_token_len = int(data.meta_info.get("max_token_len", self.config.ppo_max_token_len_per_gpu)) * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)
            indices = None

        value_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch.to(get_torch_device().current_device()), **micro_batch.non_tensor_batch}
            else:
                micro_batch = micro_batch.to(get_torch_device().current_device())
            with torch.no_grad():
                value_lst.append(self._forward_reasoning_value_micro_batch(micro_batch).float())

        values = torch.concat(value_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == values.size(0), f"{len(indices)} vs. {values.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long, device=values.device)
            values = values[revert_indices]
        return values

    def update_policy(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        multi_turn = data.meta_info.get("multi_turn", False)
        use_value_loss = self._state_value_enabled() and "step_returns" in data.batch
        use_reasoning_value_loss = self._reasoning_value_enabled() and "reason_value_targets" in data.batch
        use_ssca_vimpo_loss = ssca_vimpo_actor_enabled(self.config)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if self.config.policy_loss.get("loss_mode", "vanilla") == "c_rf_ntf":
            contrastive_rf_extra_keys = ["contrastive_rf_traj_token_weight", "contrastive_rf_ntf_mask"]
            select_keys.extend([key for key in contrastive_rf_extra_keys if key in data.batch])
        if use_value_loss:
            select_keys.extend(["state_values", "step_returns"])
            if "value_loss_mask" in data.batch:
                select_keys.append("value_loss_mask")
        if use_reasoning_value_loss:
            select_keys.extend([
                "reason_value_input_ids",
                "reason_value_attention_mask",
                "reason_value_position_ids",
                "reason_value_indices",
                "reason_value_targets",
                "reason_value_loss_mask",
            ])
            if "reason_value_think_close_found" in data.batch:
                select_keys.append("reason_value_think_close_found")
            if "reason_value_response_valid_len" in data.batch:
                select_keys.append("reason_value_response_valid_len")
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        ssca_vimpo_append_select_keys(self.config, select_keys)
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
                    c_rf_ntf_metrics = None
                    if loss_mode == "vanilla":
                        policy_loss_fn = compute_policy_loss
                    elif loss_mode == "gspo":
                        policy_loss_fn = compute_policy_loss_gspo
                    elif loss_mode == "c_rf_ntf":
                        policy_loss_fn = compute_c_rf_ntf_policy_loss
                    else:
                        raise ValueError(f"Unsupported loss_mode: {loss_mode}")

                    policy_loss_output = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c,
                        loss_agg_mode=loss_agg_mode,
                        **(
                            {
                                "ntf_keep_ratio": self.config.policy_loss.get("ntf_keep_ratio", 0.1),
                                "ntf_min_keep_tokens": self.config.policy_loss.get("ntf_min_keep_tokens", 1),
                                "traj_token_weight": micro_data["contrastive_rf_traj_token_weight"] if "contrastive_rf_traj_token_weight" in micro_data else None,
                                "ntf_mask": micro_data["contrastive_rf_ntf_mask"] if "contrastive_rf_ntf_mask" in micro_data else None,
                            }
                            if loss_mode == "c_rf_ntf"
                            else {}
                        ),
                    )
                    if loss_mode == "c_rf_ntf":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, c_rf_ntf_metrics = policy_loss_output
                        append_to_dict(metrics, c_rf_ntf_metrics)
                    else:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_output

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
                        value_mask = micro_data.get("value_loss_mask", torch.ones_like(step_returns)).float()
                        value_mask_sum = value_mask.sum().clamp_min(1.0)
                        values_clipped = old_values + torch.clamp(current_values - old_values, -cliprange_value, cliprange_value)
                        value_loss_unclipped = (current_values - step_returns).pow(2)
                        value_loss_clipped = (values_clipped - step_returns).pow(2)
                        clipped_value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
                        value_loss = 0.5 * (clipped_value_loss * value_mask).sum() / value_mask_sum
                        value_clipfrac = ((value_loss_clipped > value_loss_unclipped).float() * value_mask).sum() / value_mask_sum
                        value_error = current_values - step_returns
                        value_mse = (value_error.pow(2) * value_mask).sum() / value_mask_sum
                        value_rmse = torch.sqrt(value_mse.clamp_min(0.0))
                        value_mae = (value_error.abs() * value_mask).sum() / value_mask_sum
                        active_mask = value_mask > 0
                        active_values = current_values[active_mask]
                        active_returns = step_returns[active_mask]
                        if active_returns.numel() > 1:
                            value_return_var = torch.var(active_returns, unbiased=False)
                            value_residual_var = torch.var(active_returns - active_values, unbiased=False)
                        else:
                            value_return_var = torch.zeros((), device=current_values.device, dtype=current_values.dtype)
                            value_residual_var = torch.zeros((), device=current_values.device, dtype=current_values.dtype)
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
                                "actor/value_pred_mean": active_values.detach().mean().item() if active_values.numel() else 0.0,
                                "actor/value_pred_std": active_values.detach().std(unbiased=False).item() if active_values.numel() else 0.0,
                                "actor/value_return_mean": active_returns.detach().mean().item() if active_returns.numel() else 0.0,
                                "actor/value_return_std": active_returns.detach().std(unbiased=False).item() if active_returns.numel() else 0.0,
                                "actor/value_loss_mask_ratio": value_mask.detach().mean().item(),
                                "actor/value_loss_coef": value_loss_coef,
                            },
                        )

                    if use_reasoning_value_loss:
                        reason_values = self._forward_reasoning_value_micro_batch(micro_data).float()
                        reason_targets = micro_data["reason_value_targets"].float()
                        reason_mask = micro_data["reason_value_loss_mask"].float()
                        reason_mask_sum = reason_mask.sum().clamp_min(1.0)
                        reason_cfg = self._reasoning_value_config()
                        reason_loss_coef = float(reason_cfg.get("value_loss_coef", 0.05))
                        reason_loss_type = reason_cfg.get("loss_type", "mse")
                        if reason_loss_type == "huber":
                            huber_delta = float(reason_cfg.get("huber_delta", 1.0))
                            reason_loss_unreduced = torch.nn.functional.huber_loss(
                                reason_values,
                                reason_targets,
                                reduction="none",
                                delta=huber_delta,
                            )
                            reason_value_loss = (reason_loss_unreduced * reason_mask).sum() / reason_mask_sum
                        elif reason_loss_type == "mse":
                            reason_value_loss = 0.5 * ((reason_values - reason_targets).pow(2) * reason_mask).sum() / reason_mask_sum
                        else:
                            raise ValueError(f"Unsupported reasoning_value_aux.loss_type: {reason_loss_type}")
                        policy_loss = policy_loss + reason_loss_coef * reason_value_loss
                        reason_metrics = self._scalar_value_regression_metrics(
                            prefix="actor/reason_value",
                            current_values=reason_values,
                            targets=reason_targets,
                            loss_mask=reason_mask,
                            value_loss=reason_value_loss,
                            loss_coef=reason_loss_coef,
                        )
                        if "reason_value_think_close_found" in micro_data:
                            reason_metrics["actor/reason_value/think_close_found_ratio"] = micro_data["reason_value_think_close_found"].float().mean().detach().item()
                        if "reason_value_response_valid_len" in micro_data:
                            reason_metrics["actor/reason_value/response_valid_len_mean"] = micro_data["reason_value_response_valid_len"].float().mean().detach().item()
                        append_to_dict(metrics, reason_metrics)

                    if use_ssca_vimpo_loss:
                        ssca_vimpo_loss, ssca_vimpo_metrics = compute_ssca_vimpo_value_loss(
                            log_prob=log_prob,
                            ref_log_prob=micro_data["ref_log_prob"],
                            response_mask=response_mask,
                            terminal_target=micro_data["ssca_vimpo_terminal_target"],
                            loss_mask=micro_data["ssca_vimpo_loss_mask"],
                            config=self.config,
                        )
                        policy_loss = policy_loss + ssca_vimpo_loss
                        append_to_dict(metrics, ssca_vimpo_metrics)

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
