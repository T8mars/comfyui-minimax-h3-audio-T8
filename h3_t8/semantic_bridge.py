"""Content-bound, opt-in H3 conditioning adapters. No import-time model/GPU work.

Independent implementation of the published 5120/512/512/5120 SiLU contract.
The adapters are not LoRAs. Ref/voice quality remains experimental.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import threading

import torch
from torch.nn import functional as F

SCHEMA = "t8_semantic_bridge_v1"
RECEIPT_KEY = "t8_semantic_bridge"
SHAPES = {
    "fc1.weight": (512, 5120), "fc1.bias": (512,),
    "fc2.weight": (512, 512), "fc2.bias": (512,),
    "fc3.weight": (5120, 512), "fc3.bias": (5120,),
}
KNOWN_MODELS = {
    "ac0dc8ac05f545ebdee12e2fcebe4515b049f9cfd9558eb4887a9bf3fd6d562e": {
        "name": "Semantic Bridge v1", "repo": "speach1sdef178/MiniMax-H3-Semantic-Bridge",
        "revision": "b9fe58ba6f428d990a59f20f09f719c8fbc67f7d",
    },
    "983380be6bf790544dbfa9be1bbe42e60ea841c7b6f7c5aac668de9380ab277a": {
        "name": "BUNNY ActionLogic v1", "repo": "JOKER141/BUNNY_H3_Conditioning_Bridge",
        "revision": "658bfbb0c49f6e8f79d727c7d261efa9e3853893",
    },
}
_CACHE = OrderedDict()  # CPU FP32 only; a live caller holds its own strong reference.
_CACHE_LOCK = threading.RLock()
_MAX_FILE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class _TransWeights:
    state: dict
    metadata: dict


@dataclass(frozen=True)
class _TrainerMLPWeights:
    state: dict
    spec: object


def _weight_metadata(data: bytes) -> dict:
    header_size = int.from_bytes(data[:8], "little")
    if header_size <= 0 or header_size > min(len(data) - 8, 1024 * 1024):
        raise ValueError("Invalid WushuBridge safetensors metadata header")
    metadata = json.loads(data[8:8 + header_size]).get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("Invalid WushuBridge safetensors metadata")
    return metadata


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


@dataclass(frozen=True)
class BridgeConfig:
    path: str
    sha256: str
    alpha: float = 0.10
    magnitude_match: str = "per_token"
    token_scope: str = "all_tokens"
    device: str = "auto"
    chunk_tokens: int = 256
    enabled: bool = True
    compute_profile: str = "fp32"

    def __post_init__(self):
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError("Bridge alpha must be finite and between 0 and 1")
        if self.magnitude_match not in ("per_token", "global", "none"):
            raise ValueError("Unknown Bridge magnitude_match")
        if self.token_scope not in ("all_tokens", "text_only_preserve_reference"):
            raise ValueError("Unknown Bridge token_scope")
        if self.device not in ("auto", "cpu", "cuda") or self.compute_profile != "fp32":
            raise ValueError("Unknown Bridge device/compute_profile")
        if not isinstance(self.chunk_tokens, int) or self.chunk_tokens < 0:
            raise ValueError("Bridge chunk_tokens must be zero (whole sequence) or positive")

    @property
    def active(self):
        return self.enabled and self.alpha != 0

    def identity(self):
        if not self.active:
            return None
        values = asdict(self)
        values.pop("path")  # Portable cache/receipt, no private absolute path.
        return {"schema": SCHEMA, **values}


def bridge_identity(config):
    if config is None:
        return None
    if not isinstance(config, BridgeConfig):
        raise TypeError("Expected T8 Semantic Bridge configuration")
    return config.identity()


def preflight_bridge(config):
    """Validate all active chain configs before any stage; never switch on continuation."""
    identity = bridge_identity(config)
    if identity is not None:
        _weights(config)
    return identity


def bridge_kwargs(config):
    return {"semantic_bridge": config} if bridge_identity(config) is not None else {}


def validate_weights(weights):
    if set(weights) != set(SHAPES):
        raise ValueError("Bridge requires exactly six fc1/fc2/fc3 weight/bias tensors")
    for key, shape in SHAPES.items():
        value = weights[key]
        if tuple(value.shape) != shape or value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"Invalid Bridge tensor {key}: {tuple(value.shape)} {value.dtype}")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Non-finite Bridge tensor: {key}")


def read_weights(path, expected_sha=None):
    from safetensors.torch import load

    path = Path(path)
    if path.suffix.lower() != ".safetensors" or path.stat().st_size > _MAX_FILE_BYTES:
        raise ValueError("Expected a small Semantic Bridge .safetensors file")
    # Decode exactly the bytes hashed, so path replacement cannot mix identities.
    with path.open("rb") as stream:
        data = stream.read(_MAX_FILE_BYTES + 1)
    if len(data) > _MAX_FILE_BYTES:
        raise ValueError("Bridge file exceeds supported size")
    sha = hashlib.sha256(data).hexdigest()
    if expected_sha is not None and sha != expected_sha:
        raise ValueError("Bridge model content changed; re-execute the configuration node")
    weights = load(data)
    if "net.in_proj.weight" in weights:
        from .semantic_bridge_trans import build_trans_bridge
        build_trans_bridge(weights, _weight_metadata(data))
    elif "net.fc1.weight" in weights:
        from .semantic_bridge_mlp import validate_trainer_mlp
        validate_trainer_mlp(weights, _weight_metadata(data))
    else:
        validate_weights(weights)
    return weights, sha


def _weights(config):
    # Always rehash, including same-size/same-mtime replacement. No CUDA cache.
    if file_sha(config.path) != config.sha256:
        raise ValueError("Bridge model content changed; re-execute the configuration node")
    with _CACHE_LOCK:
        cached = _CACHE.get(config.sha256)
        if cached is None:
            state, _ = read_weights(config.path, config.sha256)
            if "net.in_proj.weight" in state or "net.fc1.weight" in state:
                from .semantic_bridge_trans import spec_from_metadata
                from .semantic_bridge_mlp import validate_trainer_mlp
                with Path(config.path).open("rb") as stream:
                    data = stream.read(8 + 1024 * 1024)
                metadata = _weight_metadata(data)
                if file_sha(config.path) != config.sha256:
                    raise ValueError("Bridge model content changed; re-execute the configuration node")
                if "net.in_proj.weight" in state:
                    spec_from_metadata(metadata)
                    cached = _TransWeights({name: tensor.float() for name, tensor in state.items()}, metadata)
                else:
                    spec = validate_trainer_mlp(state, metadata)
                    cached = _TrainerMLPWeights(
                        {name[4:]: tensor.float() for name, tensor in state.items()}, spec)
            else:
                cached = {name: tensor.float() for name, tensor in state.items()}
            _CACHE[config.sha256] = cached
            while len(_CACHE) > 2:
                _CACHE.popitem(last=False)
        _CACHE.move_to_end(config.sha256)
        return cached


def _tensor_sha(tensor):
    data = tensor.detach().contiguous().cpu()
    sha = hashlib.sha256(str((tuple(data.shape), data.dtype)).encode())
    sha.update(data.view(torch.uint8).numpy().tobytes())
    return sha.hexdigest()


def _project(h, weights):
    with torch.autocast(device_type=h.device.type, enabled=False):
        normalized = h / (h.square().mean(dim=-1, keepdim=True) + 1e-6).sqrt()
        first = F.silu(F.linear(normalized, weights["fc1.weight"], weights["fc1.bias"]))
        second = F.silu(F.linear(first, weights["fc2.weight"], weights["fc2.bias"]))
        return F.linear(second, weights["fc3.weight"], weights["fc3.bias"])


def _predict_mlp(h, weights, trainer_spec=None):
    projected = _project(h, weights)
    if trainer_spec is not None and trainer_spec.residual_skip:
        return h + trainer_spec.residual_scale * projected
    return projected


def _apply_trans_bridge(conditioning, config, model, identity, cancel, encoding_source):
    """Apply the trainer's sequence-wide Transformer without flattening tokens."""
    outputs, receipts = [], []
    for native, metadata in conditioning:
        device = native.device if config.device == "auto" else torch.device(config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Bridge CUDA explicitly selected but CUDA is unavailable")
        model = model.to(device=device, dtype=torch.float32)
        h = native.to(device=device, dtype=torch.float32)
        pred = torch.empty_like(h)
        stride = config.chunk_tokens or h.shape[1]
        with torch.autocast(device_type=device.type, enabled=False):
            for start in range(0, h.shape[1], stride):
                cancel()
                segment = h[:, start:start + stride]
                normalized = segment / (segment.square().mean(dim=-1, keepdim=True) + 1e-6).sqrt()
                pred[:, start:start + stride] = model(normalized)
        if config.magnitude_match == "per_token":
            pred = pred * ((h.square().mean(-1, keepdim=True) + 1e-8).sqrt() /
                           (pred.square().mean(-1, keepdim=True) + 1e-8).sqrt())
        elif config.magnitude_match == "global":
            pred = pred * ((h.square().mean() + 1e-8).sqrt() /
                           (pred.square().mean() + 1e-8).sqrt())
        result = h + config.alpha * (pred - h)
        if config.token_scope == "text_only_preserve_reference":
            tags = metadata["minimax_token_tags"].to(device)
            result = torch.where((tags == 1)[None, :, None], result, h)
        output = result.to(device=native.device, dtype=native.dtype)
        if not torch.isfinite(output).all().item():
            raise ValueError("Non-finite Bridge output")
        receipt = {
            **identity, "encoding_source": encoding_source,
            "input_sha256": _tensor_sha(native), "output_sha256": _tensor_sha(output),
            "shape": list(native.shape), "dtype": str(native.dtype),
            "applied_tokens_per_batch": (int((metadata["minimax_token_tags"] == 1).sum().item())
                                         if config.token_scope != "all_tokens" else native.shape[1]),
            "reference_payload_present": bool(metadata.get("minimax_refs")),
        }
        receipt["receipt_sha256"] = digest(receipt)
        outputs.append([output, {**metadata, RECEIPT_KEY: receipt}])
        receipts.append(receipt)
    cancel()
    return outputs, {"enabled": True, "applied": True, "identity": identity, "items": receipts,
                     "quality": "experimental_not_human_qualified",
                     "warning": "Reference audio/singing can degrade; video quality requires actual A/B review."}


def _check_cancel():
    # Import at execution only; standalone converter/tests don't initialize CUDA.
    from comfy.model_management import throw_exception_if_processing_interrupted
    throw_exception_if_processing_interrupted()


@torch.inference_mode()
def apply_bridge(conditioning, config, *, encoding_source="external_unknown", cancel=None):
    identity = bridge_identity(config)
    if identity is None:
        return conditioning, {"enabled": False, "applied": False}
    cancel = cancel or _check_cancel
    if not isinstance(conditioning, (list, tuple)) or not conditioning:
        raise ValueError("Bridge requires non-empty CONDITIONING")
    # Validate every item before reading weights or producing output.
    for item in conditioning:
        if not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[1], dict):
            raise ValueError("Invalid CONDITIONING item")
        native, metadata = item
        if not isinstance(native, torch.Tensor) or native.ndim != 3 or native.shape[-1] != 5120:
            raise ValueError("Bridge expects raw H3 conditioning [B,T,5120], not projected embeds")
        if native.numel() == 0 or not native.is_floating_point() or not torch.isfinite(native).all().item():
            raise ValueError("Bridge input must contain non-empty finite floating point embeddings")
        if RECEIPT_KEY in metadata or metadata.get("sensenova_h3_distilled"):
            raise ValueError("Semantic Bridge already applied; use a fresh native encoding, do not stack bridges")
        if "minimax_prompt_relay_binding" in metadata or "t8_prompt_relay_binding_hash" in metadata:
            raise ValueError("Apply Bridge through the Prompt Relay optional input, before Relay binding")
        if config.token_scope == "text_only_preserve_reference":
            tags = metadata.get("minimax_token_tags")
            if not isinstance(tags, torch.Tensor) or tags.ndim != 1 or tags.numel() != native.shape[1]:
                raise ValueError("Text-only Bridge requires exact native minimax_token_tags")
            if not torch.all((tags == 0) | (tags == 1)).item():
                raise ValueError("Unknown native H3 token tags")
    cancel()
    state = _weights(config)
    if isinstance(state, _TransWeights):
        from .semantic_bridge_trans import build_trans_bridge, validate_contract
        model = build_trans_bridge(state.state, state.metadata)
        validate_contract(model.spec, alpha=config.alpha,
                          magnitude_match=config.magnitude_match,
                          token_scope=config.token_scope, chunk_tokens=config.chunk_tokens)
        return _apply_trans_bridge(conditioning, config, model, identity, cancel, encoding_source)
    trainer_spec = None
    if isinstance(state, _TrainerMLPWeights):
        from .semantic_bridge_trans import validate_contract
        trainer_spec = state.spec
        validate_contract(trainer_spec.application_contract, alpha=config.alpha,
                          magnitude_match=config.magnitude_match,
                          token_scope=config.token_scope, chunk_tokens=config.chunk_tokens)
        state = state.state
    outputs, receipts = [], []
    for native, metadata in conditioning:
        device = native.device if config.device == "auto" else torch.device(config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Bridge CUDA explicitly selected but CUDA is unavailable")
        # Scoped per-item weights, never globally retained on GPU.
        weights = {name: tensor.to(device=device) for name, tensor in state.items()}
        rows = native.reshape(-1, 5120)
        output = torch.empty_like(rows)
        global_scale = None
        if config.magnitude_match == "global":
            # Whole item statistics (all batches/tokens), not per-chunk matching.
            source_sum = torch.zeros((), dtype=torch.float64, device=device)
            target_sum = torch.zeros_like(source_sum)
            chunk_tokens = config.chunk_tokens or rows.shape[0]
            for start in range(0, rows.shape[0], chunk_tokens):
                cancel()
                h = rows[start:start + chunk_tokens].to(device=device, dtype=torch.float32)
                projected = _predict_mlp(h, weights, trainer_spec)
                source_sum += projected.double().square().sum()
                target_sum += h.double().square().sum()
            global_scale = ((target_sum / native.numel() + 1e-8) /
                            (source_sum / native.numel() + 1e-8)).sqrt().float()
        chunk_tokens = config.chunk_tokens or rows.shape[0]
        for start in range(0, rows.shape[0], chunk_tokens):
            cancel()
            h = rows[start:start + chunk_tokens].to(device=device, dtype=torch.float32)
            projected = _predict_mlp(h, weights, trainer_spec)
            if config.magnitude_match == "per_token":
                projected = projected * ((h.square().mean(-1, keepdim=True) + 1e-8).sqrt() /
                                         (projected.square().mean(-1, keepdim=True) + 1e-8).sqrt())
            elif global_scale is not None:
                projected = projected * global_scale
            result = h + config.alpha * (projected - h)
            if not torch.isfinite(result).all().item():
                raise ValueError("Non-finite Bridge output")
            output[start:start + chunk_tokens] = result.to(device=native.device, dtype=native.dtype)
        output = output.reshape_as(native)
        if config.token_scope == "text_only_preserve_reference":
            tags = metadata["minimax_token_tags"].to(native.device)
            output = torch.where((tags == 1)[None, :, None], output, native)
        if not torch.isfinite(output).all().item():
            raise ValueError("Bridge output overflows the original conditioning dtype")
        receipt = {
            **identity, "encoding_source": encoding_source,
            "input_sha256": _tensor_sha(native), "output_sha256": _tensor_sha(output),
            "shape": list(native.shape), "dtype": str(native.dtype),
            "applied_tokens_per_batch": (int((metadata["minimax_token_tags"] == 1).sum().item())
                                         if config.token_scope != "all_tokens" else native.shape[1]),
            "reference_payload_present": bool(metadata.get("minimax_refs")),
        }
        receipt["receipt_sha256"] = digest(receipt)
        outputs.append([output, {**metadata, RECEIPT_KEY: receipt}])
        receipts.append(receipt)
        del weights
    cancel()
    return outputs, {"enabled": True, "applied": True, "identity": identity, "items": receipts,
                     "quality": "experimental_not_human_qualified",
                     "warning": "Reference audio/singing can degrade; unchanged audio inputs do not prove generated audio quality."}
