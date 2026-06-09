# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Value-head utilities for shared-backbone actor-critic experiments."""

import torch
from torch import nn


class StepValueHead(nn.Module):
    """Small regression head that maps LM hidden states to scalar values."""

    def __init__(self, hidden_size: int, intermediate_size: int | None = None, dropout: float = 0.0, head_type: str = "mlp"):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.head_type = head_type

        if head_type == "linear":
            self.summary = nn.Linear(hidden_size, 1)
        elif head_type == "mlp":
            intermediate_size = intermediate_size or hidden_size
            self.summary = nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.GELU(),
                nn.Linear(intermediate_size, 1),
            )
        else:
            raise ValueError(f"Unsupported value head type: {head_type}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.dropout(hidden_states)
        first_weight = next(self.summary.parameters(), None)
        # FSDP with use_orig_params=False can hide child parameters behind a
        # flat parameter during forward. In that case, leave dtype handling to
        # autocast/FSDP instead of assuming child parameters are iterable.
        if first_weight is not None and output.dtype != first_weight.dtype:
            output = output.to(first_weight.dtype)
        return self.summary(output).squeeze(-1)


def infer_hidden_size(config) -> int:
    """Infer the hidden size from common HF model config layouts."""
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    if hasattr(config, "word_embed_proj_dim"):
        return config.word_embed_proj_dim
    if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
        return config.text_config.hidden_size
    if hasattr(config, "decoder") and hasattr(config.decoder, "hidden_size"):
        return config.decoder.hidden_size
    raise ValueError(f"Cannot infer hidden size from config type: {type(config)}")


def _cfg_get(config, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _patch_model_class_forward(model_cls) -> None:
    if getattr(model_cls, "_step_value_head_class_patched", False):
        return

    original_forward = model_cls.forward

    def forward_with_step_values(self, *args, **kwargs):
        return_step_values = bool(kwargs.pop("return_step_values", False))
        if return_step_values:
            if not hasattr(self, "value_head"):
                raise RuntimeError("return_step_values=True requires an attached step value head.")
            kwargs["output_hidden_states"] = True
            kwargs["return_dict"] = True

        output = original_forward(self, *args, **kwargs)
        if return_step_values:
            if not hasattr(output, "hidden_states") or output.hidden_states is None:
                raise RuntimeError("return_step_values=True requires model output_hidden_states.")
            hidden_states = output.hidden_states[-1]
            if bool(getattr(self, "_step_value_head_detach_backbone", False)):
                hidden_states = hidden_states.detach()
            step_values = self.value_head(hidden_states)
            output.step_values = step_values
            return output, step_values
        return output

    model_cls._step_ppo_original_forward = original_forward
    model_cls.forward = forward_with_step_values
    model_cls._step_value_head_class_patched = True


def attach_step_value_head(model: nn.Module, model_config, value_head_config) -> nn.Module:
    """Attach a TRL-style value head and compute values inside model.forward.

    Computing the head inside ``forward`` keeps the value-head parameters inside
    the FSDP forward/unshard window while preserving the base LM state_dict names.
    The patched forward accepts ``return_step_values=True`` and stores token-wise
    values on ``output.step_values``.
    """
    if getattr(model, "_step_value_head_attached", False):
        return model

    hidden_size = infer_hidden_size(model_config)
    model.value_head = StepValueHead(
        hidden_size=hidden_size,
        intermediate_size=_cfg_get(value_head_config, "intermediate_size", None),
        dropout=_cfg_get(value_head_config, "dropout", 0.0),
        head_type=_cfg_get(value_head_config, "head_type", "mlp"),
    )
    model._step_value_head_detach_backbone = bool(_cfg_get(value_head_config, "detach_value_backbone", False))
    _patch_model_class_forward(model.__class__)
    model._step_value_head_attached = True
    return model
