"""Chronos-2 pretrained time series encoder — the single time-series
encoder used by all three TSLM architectures (ChatTS, Flamingo, ITFormer).

Requires ``chronos-forecasting==2.2.2`` exactly (see pyproject.toml) — the
meta-device-buffer repair patch below reaches into private internals that
were renamed in later releases.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import numpy as np


def _repair_chronos2_load(model, model_id: str) -> None:
    """Undo two transformers-5.x ``from_pretrained`` bugs that leave
    Chronos-2 with random-init params + uninitialized RoPE buffers.

    1. Non-persistent buffers (``RoPE.inv_freq``, ``Chronos2Model.quantiles``)
       are computed in ``__init__`` but ``__init__`` runs under
       ``torch.device("meta")``, so the buffers materialize as
       uninitialized memory. Recompute them deterministically.
    2. ``from_pretrained`` reports 170/170 loaded but ``state_dict()`` values
       don't actually match the safetensors content — re-issue
       ``load_state_dict`` from the raw checkpoint to fix it.
    """
    from chronos.chronos2.layers import RoPE
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    for m in model.modules():
        if isinstance(m, RoPE):
            inv_freq = 1.0 / (
                m.base ** (torch.arange(0, m.dim, 2, dtype=torch.int64).float() / m.dim)
            )
            m.inv_freq.data.copy_(inv_freq.to(m.inv_freq.device, m.inv_freq.dtype))

    cfg = model.config.chronos_config
    quantiles = torch.tensor(cfg["quantiles"]) if isinstance(cfg, dict) else torch.tensor(cfg.quantiles)
    model.quantiles.data.copy_(quantiles.to(model.quantiles.device, model.quantiles.dtype))

    raw_state: dict[str, torch.Tensor] = {}
    for f in sorted(Path(snapshot_download(model_id)).glob("*.safetensors")):
        with safe_open(f, framework="pt") as g:
            for k in g.keys():
                raw_state[k] = g.get_tensor(k)
    device = next(model.parameters()).device
    raw_state = {k: v.to(device) for k, v in raw_state.items()}
    model.load_state_dict(raw_state, strict=False)


class Chronos2TSEncoder(nn.Module):
    """Pretrained Chronos-2 encoder usable as a drop-in TS encoder.

    Univariate (Flamingo, ChatTS): ``(B, L) -> (B, N, D)``
    Multivariate (ITFormer): ``(B, L, V) -> (B, V, N, D)``

    Chronos-2's ``GroupSelfAttention`` enables cross-variable information
    sharing when multiple channels share the same ``group_id``.
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        output_dim: int = 512,
        dropout: float = 0.0,
        freeze: bool = True,
        multi_variable: bool = False,
        max_patches: int | None = None,
        trained_patches: int | None = None,
        disable_context_length_limit: bool = False,
        **kwargs,
    ):
        try:
            from chronos import Chronos2Model
        except ImportError as exc:
            raise ImportError(
                "Chronos-2 encoder requires the chronos-forecasting package. "
                "Install it with: uv pip install 'chronos-forecasting==2.2.2'"
            ) from exc

        super().__init__()
        self.output_dim = output_dim
        self.dropout = dropout

        self.chronos_model = Chronos2Model.from_pretrained(model_id)
        
        if disable_context_length_limit:
            # Chronos-2's context length is hard-coded to 8192 in the config. 
            # To allow for longer inputs, we set the context length to infinity. 
            # Note that this may lead to increased memory usage and slower inference times for very long sequences.
            self.chronos_model.chronos_config.context_length = np.inf
        
        _repair_chronos2_load(self.chronos_model, model_id)
        self.output_dim = self.chronos_model.config.d_model
        self.multi_variable = multi_variable

        self.freeze = freeze
        if freeze:
            self.chronos_model.requires_grad_(False)

    def get_num_output_patches(self, input_length: int) -> int:
        """Return context patch count based on Chronos-2's internal patch size."""
        patch_size = self.chronos_model.config.chronos_config["input_patch_size"]
        return -(-input_length // patch_size)  # ceil division

    def _encode_single(
        self,
        x: torch.Tensor,
        group_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode univariate batch through Chronos-2.

        Args:
            x: ``(B, L)`` — univariate time series batch.
            group_ids: ``(B,)`` — optional group IDs for cross-variable
                attention via ``GroupSelfAttention``.

        Returns:
            ``(B, N, D)`` — context patch embeddings.
        """
        encoder_outputs, _loc_scale, _fut_mask, num_context_patches = (
            self.chronos_model.encode(
                context=x,
                group_ids=group_ids,
                num_output_patches=1,
            )
        )
        # last_hidden_state: (B, num_context_patches + [REG] + num_output_patches, D)
        # Keep only the context patches.
        hidden = encoder_outputs.last_hidden_state
        return hidden[:, :num_context_patches, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multi_variable:
            # x: (B, L, V) — multi-variable input (ITFormer path)
            B, L, V = x.shape
            # Stack all variables into batch dim: (B*V, L)
            x_flat = x.permute(0, 2, 1).reshape(B * V, L)
            # Variables from the same sample share a group →
            # enables Chronos-2's GroupSelfAttention across channels
            group_ids = torch.arange(B, device=x.device).repeat_interleave(V)
            encoded = self._encode_single(x_flat, group_ids=group_ids)  # (B*V, N, D)
            _, N, D = encoded.shape
            return encoded.reshape(B, V, N, D)
        else:
            # x: (B, L) — single-channel input (Flamingo/ChatTS path)
            return self._encode_single(x)
