"""Per-MODEL owner and audit for the model-free 64-token Veda experiment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import os
import time

import torch

from .h3_core_compat import plain_attention_backend, set_h3_attention_backend
from .patch_stack_policy import warn_patch_stack
from .veda_heuristic import (
    POOL_MODES, PRESETS, SINK_MODES, attend, parse_shapes, require_flex_runtime,
)


MODES = ("report_only", "apply_exp")


def target_grid(layout, sequence: int) -> tuple[int, tuple[int, int, int]]:
    if (layout is None or getattr(layout, "seq_len", None) != sequence
            or not hasattr(layout, "signature") or not hasattr(layout, "segments")):
        raise RuntimeError("Veda heuristic requires the native MiniMax H3 PackedLayout")
    segments = layout.segments
    if not segments or segments[-1][2] != "video":
        raise RuntimeError("Veda heuristic requires the native final target video segment")
    _text, frames, latent_h, latent_w, _audio = map(int, layout.signature)
    if frames < 1 or latent_h < 2 or latent_w < 2 or latent_h % 2 or latent_w % 2:
        raise RuntimeError("Veda heuristic received an invalid H3 latent grid")
    start, stop, _kind = segments[-1]
    grid = (frames, latent_h // 2, latent_w // 2)
    if start < 0 or stop != sequence or start + math.prod(grid) != sequence:
        raise RuntimeError("Veda heuristic target grid does not match the packed rows")
    return int(start), grid


@dataclass
class HeuristicRuntime:
    mode: str
    keep_ratio: float
    preset: str
    pool_mode: str
    min_tokens: int
    max_copy_mib: int
    previous_backend: str | None
    bypass_reason: str | None = None
    calls: int = 0
    sparse_calls: int = 0
    delegated_calls: int = 0
    delegate_reason: str | None = None
    grid: tuple[int, int, int] | None = None
    failure: str | None = None
    attention_wall_seconds: float = 0.0
    tile_cache: dict = field(default_factory=dict)
    start_percent: float = 0.0
    end_percent: float = 1.0
    sigma_start: float | None = None
    sigma_end: float | None = None
    sink_conditioning: str = "exact_kv_and_rows"
    window_delegated_calls: int = 0

    def snapshot(self) -> dict:
        return {
            "schema": "t8.veda.heuristic.audit.v1",
            "status": ("failed" if self.failure else
                       "bypassed_unverified_owner_preserved" if self.bypass_reason else
                       "observed_sparse_heuristic_experimental" if self.sparse_calls else
                       "observed_dense_delegate" if self.calls else "not_executed"),
            "algorithm": "model_free_tripool_tile64_not_official_trained_veda",
            "kernel": ("original_dense_delegate" if self.mode == "report_only" or self.bypass_reason
                       or (self.delegated_calls and not self.sparse_calls) else
                       "mixed_flex_and_dense_delegate" if self.delegated_calls else
                       "pytorch_flex_attention_not_official_fa4"),
            "mode": self.mode,
            "keep_ratio": self.keep_ratio,
            "head_tiling": self.preset,
            "pool_mode": self.pool_mode,
            "min_tokens": self.min_tokens,
            "max_copy_mib": self.max_copy_mib,
            "start_percent": self.start_percent,
            "end_percent": self.end_percent,
            "sigma_start": self.sigma_start,
            "sigma_end": self.sigma_end,
            "sink_conditioning": self.sink_conditioning,
            "window_delegated_calls": self.window_delegated_calls,
            "previous_backend": self.previous_backend,
            "bypass_reason": self.bypass_reason,
            "calls": self.calls,
            "sparse_calls": self.sparse_calls,
            "delegated_calls": self.delegated_calls,
            "delegate_reason": self.delegate_reason,
            "grid": self.grid,
            "attention_wall_seconds": round(self.attention_wall_seconds, 6),
            "failure": self.failure,
            "trained_predictor_used": False,
            "quality_accepted": False,
            "end_to_end_speedup_verified": False,
        }


def apply_heuristic(model, *, mode: str = "report_only", keep_percent: float = 5.0,
                    head_tiling: str = "dual_fast", custom_tiling: str = "",
                    pool_mode: str = "triplet", min_tokens: int = 4096,
                    max_copy_mib: int = 16, start_percent: float = 0.0,
                    end_percent: float = 1.0,
                    sink_conditioning: str = "exact_kv_and_rows"):
    if mode not in MODES or head_tiling not in (*PRESETS, "custom") or pool_mode not in POOL_MODES:
        raise ValueError("Unsupported Veda heuristic mode, tiling, or TripPool mode")
    if (sink_conditioning not in SINK_MODES
            or not math.isfinite(float(start_percent))
            or not math.isfinite(float(end_percent))
            or not 0.0 <= float(start_percent) < float(end_percent) <= 1.0):
        raise ValueError("Veda heuristic sigma window or conditioning sink is invalid")
    if (not math.isfinite(float(keep_percent)) or not 1.0 <= float(keep_percent) <= 100.0
            or not 0 <= int(min_tokens) <= 1_000_000
            or not 1 <= int(max_copy_mib) <= 256):
        raise ValueError("Veda heuristic resource or keep setting is invalid")
    shapes = parse_shapes(head_tiling, custom_tiling)
    sigma_start = sigma_end = None
    if float(start_percent) != 0.0 or float(end_percent) != 1.0:
        try:
            sampling = model.get_model_object("model_sampling")
        except (AttributeError, KeyError) as error:
            raise ValueError(
                "Veda heuristic sigma window requires model_sampling.percent_to_sigma"
            ) from error
        percent_to_sigma = getattr(sampling, "percent_to_sigma", None)
        if not callable(percent_to_sigma):
            raise ValueError("Veda heuristic sigma window requires model_sampling.percent_to_sigma")
        sigma_start = float(percent_to_sigma(float(start_percent)))
        sigma_end = float(percent_to_sigma(float(end_percent)))
        if (not math.isfinite(sigma_start) or not math.isfinite(sigma_end)
                or sigma_start < sigma_end):
            raise ValueError("Veda heuristic model_sampling returned invalid sigma bounds")
    core = model.get_model_object("diffusion_model")
    blocks = getattr(core, "blocks", None)
    if (blocks is None or len(blocks) != 50 or any(
            getattr(block.attn, "heads", None) != 56 or
            getattr(block.attn, "head_dim", None) != 128 for block in blocks)):
        raise ValueError("Veda heuristic requires the native 50-layer H3 56×128 DiT")
    options = model.model_options.get("transformer_options", {})
    patches = options.get("patches_replace", {})
    if not isinstance(patches, Mapping):
        raise TypeError("Veda heuristic MODEL patches_replace must be a mapping")
    previous = options.get("optimized_attention_override")
    if previous is not None and not callable(previous):
        raise TypeError("Veda heuristic existing attention owner must be callable")
    previous_backend = None if previous is None else plain_attention_backend(previous)
    bypass = []
    if any(bool(value) for value in patches.values()):
        bypass.append("existing_DiT_patch_owner")
    if previous is not None and previous_backend is None:
        bypass.append("unverified_attention_owner")
    bypass_reason = ",".join(bypass) or None
    if bypass_reason is not None:
        warn_patch_stack(
            f"Veda heuristic preserves {bypass_reason}; sparse coverage is unverified"
        )
    if mode == "apply_exp" and bypass_reason is None:
        from comfy.cli_args import enables_dynamic_vram

        if os.name == "nt" and enables_dynamic_vram():
            raise RuntimeError(
                "Veda heuristic Flex apply_exp requires --disable-dynamic-vram on this "
                "Windows/Triton stack; the original MODEL has not been changed"
            )
        require_flex_runtime(torch.device(model.load_device))
    cloned = model.clone()
    runtime = HeuristicRuntime(
        mode, float(keep_percent) / 100.0, head_tiling, pool_mode,
        int(min_tokens), int(max_copy_mib), previous_backend, bypass_reason,
        start_percent=float(start_percent), end_percent=float(end_percent),
        sigma_start=sigma_start, sigma_end=sigma_end,
        sink_conditioning=sink_conditioning,
    )
    if bypass_reason is not None:
        return cloned, runtime

    def delegate(q, k, v, heads, **kwargs):
        from comfy.ldm.modules import attention as core_attention

        kwargs["_inside_attn_wrapper"] = True
        if previous is not None:
            return previous(core_attention.optimized_attention, q, k, v, heads, **kwargs)
        return core_attention.optimized_attention(q, k, v, heads, **kwargs)

    def route(q, k, v, heads, mask=None, attn_precision=None,
              skip_reshape=False, skip_output_reshape=False,
              transformer_options=None, **kwargs):
        options = transformer_options or {}
        layout = options.get("minimax_h3_layout")
        layer_index = options.get("block_index")
        call_kwargs = dict(mask=mask, attn_precision=attn_precision,
                           skip_reshape=skip_reshape,
                           skip_output_reshape=skip_output_reshape,
                           transformer_options=transformer_options, **kwargs)
        if layout is None or layer_index is None:
            return delegate(q, k, v, heads, **call_kwargs)
        sequence = getattr(layout, "seq_len", None)
        if q.shape[-2] != sequence:
            if (isinstance(layer_index, int) and 0 <= layer_index < 50
                    and q.ndim == k.ndim == v.ndim == 4
                    and q.shape[:2] == k.shape[:2] == v.shape[:2] == (1, heads)
                    and k.shape[-2] == v.shape[-2] == sequence):
                runtime.calls += 1
                runtime.delegated_calls += 1
                runtime.delegate_reason = "partial_query_preserved"
            return delegate(q, k, v, heads, **call_kwargs)
        if (not isinstance(layer_index, int) or not 0 <= layer_index < 50
                or q.ndim != 4 or q.shape[0] != 1 or q.shape[1] != heads
                or q.shape != k.shape or q.shape != v.shape
                or heads != 56 or q.shape[-1] != 128
                or not skip_reshape or skip_output_reshape):
            raise RuntimeError("Veda heuristic received an unsupported H3 attention call")
        try:
            start, grid = target_grid(layout, q.shape[-2])
            runtime.grid = grid
            runtime.calls += 1
            if mode == "report_only" or mask is not None or q.shape[-2] < min_tokens:
                runtime.delegated_calls += 1
                runtime.delegate_reason = (
                    "report_only" if mode == "report_only" else
                    "mask_preserved" if mask is not None else "below_min_tokens")
                return delegate(q, k, v, heads, **call_kwargs)
            if sigma_start is not None:
                sigmas = options.get("sigmas")
                if sigmas is None:
                    raise RuntimeError("Veda heuristic sigma window requires live sampler sigmas")
                values = torch.as_tensor(sigmas).detach().flatten()
                if (not values.numel() or not bool(torch.isfinite(values).all())
                        or not bool((values == values[0]).all())):
                    raise RuntimeError("Veda heuristic requires one finite video sigma per call")
                sigma = float(values[0])
                if sigma > sigma_start or sigma < sigma_end:
                    runtime.delegated_calls += 1
                    runtime.window_delegated_calls += 1
                    runtime.delegate_reason = "outside_sigma_window"
                    return delegate(q, k, v, heads, **call_kwargs)
            started = time.monotonic()
            result = attend(
                q[0].transpose(0, 1), k[0].transpose(0, 1), v[0].transpose(0, 1),
                video_start=start, grid=grid, keep_ratio=runtime.keep_ratio,
                shapes=shapes, mode=pool_mode, max_copy_mib=max_copy_mib,
                tile_cache=runtime.tile_cache, sink_conditioning=sink_conditioning,
            )
            runtime.attention_wall_seconds += time.monotonic() - started
            runtime.sparse_calls += 1
            return result.reshape(1, q.shape[-2], heads * 128)
        except BaseException as error:
            runtime.failure = f"{type(error).__name__}: {error}"
            raise

    set_h3_attention_backend(cloned, route)
    return cloned, runtime
