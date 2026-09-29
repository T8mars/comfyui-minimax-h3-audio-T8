"""Validate the trainer's variable-width MLP Semantic Bridge export."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrainerMLPSpec:
    hidden: int
    residual_skip: bool
    residual_scale: float
    application_contract: dict | None


def validate_trainer_mlp(state: dict, metadata: dict) -> TrainerMLPSpec:
    if metadata.get("arch") != "mlp" or int(metadata.get("dim", "0")) != 5120:
        raise ValueError("Expected a 5120-dimensional WushuBridge MLP model")
    hidden = int(metadata.get("hidden", "0"))
    if hidden < 1 or hidden > 4096:
        raise ValueError("Unsupported WushuBridge MLP width")
    shapes = {
        "net.fc1.weight": (hidden, 5120), "net.fc1.bias": (hidden,),
        "net.fc2.weight": (hidden, hidden), "net.fc2.bias": (hidden,),
        "net.fc3.weight": (5120, hidden), "net.fc3.bias": (5120,),
    }
    if set(state) != set(shapes):
        raise ValueError("WushuBridge MLP tensor keys do not match the declared architecture")
    for key, shape in shapes.items():
        value = state[key]
        if (tuple(value.shape) != shape
                or value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                or not torch.isfinite(value).all().item()):
            raise ValueError(f"Invalid WushuBridge MLP tensor: {key}")
    skip = metadata.get("residual_skip", "False").lower()
    scale = float(metadata.get("residual_scale", "0.1"))
    if skip not in ("true", "false") or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("Invalid WushuBridge MLP residual metadata")
    extra = json.loads(metadata.get("extra_json", "{}"))
    if not isinstance(extra, dict):
        raise ValueError("Invalid WushuBridge MLP extra_json metadata")
    contract = extra.get("application_contract")
    if contract is not None and not isinstance(contract, dict):
        raise ValueError("Invalid WushuBridge MLP application contract")
    return TrainerMLPSpec(hidden, skip == "true", scale, contract)
