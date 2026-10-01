"""Opt-in T2VA Veda attention adapter for the native MiniMax H3 Core."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import time

import torch

from .h3_core_compat import plain_attention_backend, set_h3_attention_backend
from .patch_stack_policy import warn_patch_stack
from .veda_flex import attend, require_flex_runtime
from .veda_vendor.veda import bundle as veda_bundle


MODEL_FORMAT = "miowtion-veda-predictor-v1"
MODES = ("report_only", "apply_exp")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


@dataclass(frozen=True)
class VedaBundleRef:
    path: Path
    sha256: str
    metadata: dict[str, str]

    def report(self) -> dict:
        import json

        plans = json.loads(self.metadata["plans"])
        return {
            "schema": "t8.veda.bundle.v1",
            "format": self.metadata["format"],
            "sha256": self.sha256,
            "file": self.path.name,
            "layers": int(self.metadata["num_layers"]),
            "heads": int(self.metadata["num_heads"]),
            "head_dim": int(self.metadata["head_dim"]),
            "trained_keep_ratio": float(self.metadata["keep_ratio"]),
            "plans": {name: plan["grid"] for name, plan in sorted(plans.items())},
            "status": "bundle_only_not_an_inference_result",
        }


def inspect_bundle(path: Path) -> VedaBundleRef:
    path = path.resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".safetensors":
        raise ValueError("Veda bundle must be an existing safetensors file")
    metadata = veda_bundle.read_metadata(str(path))
    if (metadata["format"] != MODEL_FORMAT or int(metadata["num_layers"]) != 50
            or int(metadata["num_heads"]) != 56 or int(metadata["head_dim"]) != 128):
        raise ValueError("Only the 50-layer, 56-head MiniMax H3 Veda bundle is supported")
    return VedaBundleRef(path, _sha256(path), metadata)


def exact_plan(plans, grid: tuple[int, int, int]):
    matches = [plan for plan in plans.plans.values() if tuple(plan.grid) == tuple(grid)]
    if len(matches) != 1:
        raise ValueError(
            f"Veda bundle has no unique plan for the exact live video grid {grid}; "
            "resize to a supported canvas and duration or remove Veda Apply"
        )
    return matches[0]


def _t2va_layout(layout, sequence: int):
    if (layout is None or getattr(layout, "seq_len", None) != sequence
            or not hasattr(layout, "signature") or not hasattr(layout, "segments")):
        raise RuntimeError("Veda requires the native MiniMax H3 PackedLayout")
    if [kind for _, _, kind in layout.segments] != ["text", "audio", "video"]:
        raise RuntimeError("This released Veda predictor is T2VA-only: keyframes/references are unsupported")
    text, frames, latent_h, latent_w, audio_t = map(int, layout.signature)
    if (text < 1 or frames < 1 or audio_t < 1 or latent_h % 2 or latent_w % 2):
        raise RuntimeError("Veda received an invalid native T2VA target grid")
    video_start, video_stop, _ = layout.segments[-1]
    grid = (frames, latent_h // 2, latent_w // 2)
    if video_stop != sequence or video_start + frames * grid[1] * grid[2] != sequence:
        raise RuntimeError("Veda video segment does not match the packed target grid")
    return int(video_start), grid


@dataclass
class VedaRuntime:
    bundle: VedaBundleRef
    mode: str
    keep_ratio: float
    max_copy_mib: int
    loaded: object
    previous_backend: str | None
    fused_tile_io: bool = False
    bypass_reason: str | None = None
    calls: int = 0
    sparse_calls: int = 0
    delegated_calls: int = 0
    delegate_reason: str | None = None
    grid: tuple[int, int, int] | None = None
    plan_name: str | None = None
    failure: str | None = None
    attention_wall_seconds: float = 0.0
    tile_cache: dict = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {
            "schema": "t8.veda.audit.v1",
            "status": ("failed" if self.failure else
                       "bypassed_unverified_owner_preserved" if self.bypass_reason else
                       "observed_sparse_experimental" if self.sparse_calls else
                       "observed_report_only" if self.calls and self.mode == "report_only" else
                       "observed_dense_delegate" if self.calls else "not_executed"),
            "mode": self.mode,
            "bundle_sha256": self.bundle.sha256,
            "trained_keep_ratio": self.loaded.keep_ratio,
            "requested_keep_ratio": self.keep_ratio,
            "kernel": ("original_dense_delegate" if self.mode == "report_only" or self.bypass_reason
                       or (self.delegated_calls and not self.sparse_calls) else
                       "mixed_flex_and_dense_delegate" if self.delegated_calls else
                       "pytorch_flex_attention_not_official_fa4"),
            "tile_io": ("not_run_owner_preserved" if self.bypass_reason else
                        "not_run_report_only" if self.mode == "report_only" else
                        "not_run_dense_delegate" if self.delegated_calls and not self.sparse_calls else
                        "pinned_triton_fused_exp" if self.fused_tile_io else "torch_reference"),
            "previous_backend": self.previous_backend,
            "bypass_reason": self.bypass_reason,
            "calls": self.calls,
            "sparse_calls": self.sparse_calls,
            "delegated_calls": self.delegated_calls,
            "delegate_reason": self.delegate_reason,
            "attention_wall_seconds": round(self.attention_wall_seconds, 6),
            "grid": self.grid,
            "plan": self.plan_name,
            "failure": self.failure,
            "quality_accepted": False,
            "end_to_end_speedup_verified": False,
        }


def apply_veda(model, reference: VedaBundleRef, *, mode: str = "report_only",
               keep_ratio: float | None = None, max_copy_mib: int = 64,
               fused_tile_io: bool = False):
    if type(reference) is not VedaBundleRef or mode not in MODES:
        raise ValueError("Use a Veda Bundle node and a supported apply mode")
    if type(fused_tile_io) is not bool:
        raise TypeError("Veda fused_tile_io must be a Boolean")
    if _sha256(reference.path) != reference.sha256:
        raise ValueError("Veda bundle changed since its Bundle node was evaluated")
    loaded = veda_bundle.load(str(reference.path), device="cpu")
    keep = loaded.keep_ratio if keep_ratio is None else float(keep_ratio)
    if not 0 < keep <= 1 or not 1 <= int(max_copy_mib) <= 256:
        raise ValueError("Veda keep_ratio or max_copy_mib is outside the safe range")
    core = model.get_model_object("diffusion_model")
    blocks = getattr(core, "blocks", None)
    if (blocks is None or len(blocks) != 50 or any(
            getattr(block.attn, "heads", None) != 56 or
            getattr(block.attn, "head_dim", None) != 128 for block in blocks)):
        raise ValueError("Veda requires the native 50-layer H3 56×128 DiT")
    options = model.model_options.get("transformer_options", {})
    patches = options.get("patches_replace", {})
    if not isinstance(patches, Mapping):
        raise TypeError("Veda MODEL patches_replace must be a mapping")
    previous = options.get("optimized_attention_override")
    if previous is not None and not callable(previous):
        raise TypeError("Veda existing attention owner must be callable")
    previous_backend = None if previous is None else plain_attention_backend(previous)
    bypass = []
    if any(bool(value) for value in patches.values()):
        bypass.append("existing_DiT_patch_owner")
    if previous is not None and previous_backend is None:
        bypass.append("unverified_attention_owner")
    bypass_reason = ",".join(bypass) or None
    if bypass_reason is not None:
        warn_patch_stack(
            f"Veda preserves and delegates to {bypass_reason}; sparse coverage is unverified"
        )
    if mode == "apply_exp" and bypass_reason is None:
        from comfy.cli_args import enables_dynamic_vram

        if os.name == "nt" and enables_dynamic_vram():
            raise RuntimeError(
                "Veda Flex apply_exp is not safe with ComfyUI DynamicVRAM on this "
                "Windows/Triton stack (isolated process crash observed). Start an "
                "isolated ComfyUI with --disable-dynamic-vram, or use report_only; "
                "the original MODEL has not been changed"
            )
        require_flex_runtime(torch.device(model.load_device))
    cloned = model.clone()
    runtime = VedaRuntime(reference, mode, keep, int(max_copy_mib), loaded,
                          previous_backend, fused_tile_io, bypass_reason)
    if bypass_reason is not None:
        # A foreign selector or DiT producer may not call the native H3
        # attention hook. Keep the chosen MODEL stack intact and report that
        # Veda did not run, rather than replacing or silently dropping it.
        return cloned, runtime

    def route(q, k, v, heads, mask=None, attn_precision=None,
              skip_reshape=False, skip_output_reshape=False,
              transformer_options=None, **kwargs):
        options = transformer_options or {}
        layout = options.get("minimax_h3_layout")
        layer_index = options.get("block_index")
        if layout is None or layer_index is None:
            # The text refiner is dense and does not have a packed video
            # layout. Only the 50 actual H3 DiT blocks are Veda candidates.
            return delegate(q, k, v, heads, mask=mask,
                            attn_precision=attn_precision,
                            skip_reshape=skip_reshape,
                            skip_output_reshape=skip_output_reshape,
                            transformer_options=transformer_options, **kwargs)
        sequence = getattr(layout, "seq_len", None)
        if q.shape[-2] != sequence:
            # Prompt Relay and other query-slicing owners pass a short Q with
            # full-length K/V and an additive bias. Veda's full-grid tile plan
            # cannot be applied to that call; preserve the selected delegate
            # and make the lack of sparse coverage visible in the audit.
            if (isinstance(layer_index, int) and 0 <= layer_index < 50
                    and q.ndim == k.ndim == v.ndim == 4
                    and q.shape[:2] == k.shape[:2] == v.shape[:2] == (1, heads)
                    and k.shape[-2] == v.shape[-2] == sequence):
                runtime.calls += 1
                runtime.delegated_calls += 1
                runtime.delegate_reason = "partial_query_preserved"
            return delegate(q, k, v, heads, mask=mask,
                            attn_precision=attn_precision,
                            skip_reshape=skip_reshape,
                            skip_output_reshape=skip_output_reshape,
                            transformer_options=transformer_options, **kwargs)
        if (not skip_reshape or skip_output_reshape or
                q.ndim != 4 or q.shape[0] != 1 or q.shape[1] != heads or
                q.shape != k.shape or q.shape != v.shape or
                heads != 56 or q.shape[-1] != 128 or
                not isinstance(layer_index, int) or not 0 <= layer_index < 50):
            raise RuntimeError("Veda received an unsupported H3 attention call")
        try:
            start, grid = _t2va_layout(layout, q.shape[-2])
            plan = exact_plan(loaded.plans, grid)
            runtime.grid, runtime.plan_name = grid, plan.geometry
            runtime.calls += 1
            if mode == "report_only" or mask is not None:
                runtime.delegated_calls += 1
                runtime.delegate_reason = "report_only" if mode == "report_only" else "mask_preserved"
                return delegate(q, k, v, heads, mask=mask,
                                attn_precision=attn_precision,
                                skip_reshape=skip_reshape,
                                skip_output_reshape=skip_output_reshape,
                                transformer_options=transformer_options, **kwargs)
            q_rows = q[0].transpose(0, 1)
            k_rows = k[0].transpose(0, 1)
            v_rows = v[0].transpose(0, 1)
            started = time.monotonic()
            output = attend(q_rows, k_rows, v_rows,
                            layer_index=layer_index, target_start=start,
                            target_grid=grid, plan=plan,
                            predictor=loaded.predictor, keep_ratio=keep,
                            max_copy_mib=int(max_copy_mib),
                            tile_cache=runtime.tile_cache,
                            fused_tile_io=fused_tile_io)
            runtime.attention_wall_seconds += time.monotonic() - started
            runtime.sparse_calls += 1
            return output.reshape(1, q.shape[-2], heads * 128)
        except BaseException as error:
            runtime.failure = f"{type(error).__name__}: {error}"
            raise

    def delegate(q, k, v, heads, **kwargs):
        from comfy.ldm.modules import attention as core_attention

        kwargs["_inside_attn_wrapper"] = True
        if previous is not None:
            return previous(core_attention.optimized_attention, q, k, v, heads, **kwargs)
        return core_attention.optimized_attention(q, k, v, heads, **kwargs)

    # The selected TensorContainers are consumed by Core before invoking
    # this callable. The output contract is exactly [1,S,H*D].
    set_h3_attention_backend(cloned, route)
    return cloned, runtime
