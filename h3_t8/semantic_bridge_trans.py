"""Adapter for WushuBridge Transformer safetensors (text CONDITIONING only).

The module mirrors the trainer's inference architecture; it does not train or
convert weights. Model metadata and tensor shapes are checked before use.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransSpec:
    hidden: int
    layers: int
    heads: int
    max_tokens: int
    residual_skip: bool
    residual_scale: float
    application_contract: dict | None


def spec_from_metadata(metadata: dict) -> TransSpec:
    if metadata.get("arch") != "trans" or int(metadata.get("dim", "0")) != 5120:
        raise ValueError("Expected a 5120-dimensional WushuBridge trans model")
    hidden = int(metadata.get("hidden", "0"))
    layers = int(metadata.get("layers", "0"))
    heads = int(metadata.get("heads", "0"))
    max_tokens = int(metadata.get("max_tokens", "4096"))
    scale = float(metadata.get("residual_scale", "0.1"))
    if (hidden < 64 or hidden > 1024 or layers < 1 or layers > 4 or heads < 1
            or heads > 16 or hidden % heads or max_tokens < 1 or max_tokens > 16384
            or not math.isfinite(scale) or not 0 < scale <= 1):
        raise ValueError("Unsupported WushuBridge trans architecture")
    skip = metadata.get("residual_skip", "False").lower()
    if skip not in ("true", "false"):
        raise ValueError("Invalid WushuBridge residual_skip metadata")
    extra = json.loads(metadata.get("extra_json", "{}"))
    if not isinstance(extra, dict):
        raise ValueError("Invalid WushuBridge extra_json metadata")
    contract = extra.get("application_contract")
    if contract is not None and not isinstance(contract, dict):
        raise ValueError("Invalid WushuBridge application contract")
    return TransSpec(hidden, layers, heads, max_tokens, skip == "true", scale, contract)


def validate_contract(spec: TransSpec | dict | None, *, alpha: float, magnitude_match: str,
                      token_scope: str, chunk_tokens: int) -> None:
    contract = spec.application_contract if isinstance(spec, TransSpec) else spec
    if contract is None:
        return
    expected = {
        "schema": 1, "alpha_min": alpha, "alpha_max": alpha,
        "magnitude_match": magnitude_match,
        "token_span": "all" if token_scope == "all_tokens" else token_scope,
        "tail_ratio": 1.0, "chunk_tokens": chunk_tokens,
        "auto_alpha": False, "guard": False, "allow_dim_mismatch": False,
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("WushuBridge application contract mismatch: required "
                         f"alpha={contract.get('alpha_min')}, "
                         f"magnitude_match={contract.get('magnitude_match')}, "
                         f"token_scope=all_tokens, chunk_tokens={contract.get('chunk_tokens')}")


def _positions(length: int, hidden: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(torch.arange(0, hidden, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / hidden))
    values = torch.zeros(length, hidden, dtype=torch.float32)
    values[:, 0::2] = torch.sin(positions * divisor)
    values[:, 1::2] = torch.cos(positions * divisor)[:, :values[:, 1::2].shape[1]]
    return values.unsqueeze(0)


class TransBridge(nn.Module):
    def __init__(self, spec: TransSpec):
        super().__init__()
        self.spec = spec
        self.in_proj = nn.Linear(5120, spec.hidden)
        self.register_buffer("_pe", _positions(spec.max_tokens, spec.hidden), persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=spec.hidden, nhead=spec.heads, dim_feedforward=spec.hidden * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=spec.layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(spec.hidden)
        self.out_proj = nn.Linear(spec.hidden, 5120)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.in_proj(x)
        pe = self._pe
        if encoded.shape[1] > pe.shape[1]:
            pe = F.interpolate(pe.transpose(1, 2), size=encoded.shape[1],
                               mode="linear", align_corners=False).transpose(1, 2)
        encoded = encoded + pe[:, :encoded.shape[1]].to(device=x.device, dtype=x.dtype)
        correction = self.out_proj(self.norm(self.encoder(encoded)))
        return x + self.spec.residual_scale * correction if self.spec.residual_skip else correction


def build_trans_bridge(state: dict, metadata: dict) -> TransBridge:
    spec = spec_from_metadata(metadata)
    model = TransBridge(spec)
    expected = {"net." + key for key in model.state_dict()}
    if set(state) != expected:
        raise ValueError("WushuBridge trans tensor keys do not match the declared architecture")
    for key, value in state.items():
        if (tuple(value.shape) != tuple(model.state_dict()[key[4:]].shape)
                or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or not torch.isfinite(value).all().item()):
            raise ValueError(f"Invalid WushuBridge trans tensor: {key}")
    model.load_state_dict({key[4:]: value for key, value in state.items()}, strict=True)
    return model.eval()
