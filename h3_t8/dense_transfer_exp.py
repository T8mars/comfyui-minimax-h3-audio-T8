"""Opt-in H3 physical-grid learned transfer; no changes to legacy forwards.

The coordinate formula is derived from Core's native 2x2 patch positions.
This independently implemented operator does not import an upstream overlay.
"""
from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import math
from typing import ClassVar

import torch
import torch.nn.functional as F

import comfy.model_management as management
import comfy.nested_tensor

from . import learned_latent_upscale_advanced as learned
from .latent_upscale import _nested_parts

LATTICE = "h3_dense_patch_center_lattice_v2"
SCRATCH_BYTES = 64 << 20


def _spatial_size(value: int, name: str) -> int:
    if type(value) is not int or value < 2 or value % 2:
        raise ValueError(f"{name} must be a positive even latent dimension")
    return value


def dense_patch_grid(source_hw, target_hw, *, device):
    """Return target dense-cell centers expressed in source grid_sample units."""
    sh, sw = (_spatial_size(v, "source H/W") for v in source_hw)
    th, tw = (_spatial_size(v, "target H/W") for v in target_hw)
    source_area = math.sqrt(sh * sw)
    target_area = math.sqrt(th * tw)
    grid_axes = []
    for source_length, target_length in ((sw, tw), (sh, th)):
        # Core positions one 2x2 patch at 16*(1-n/sqrt(HW)) + p*64/sqrt(HW).
        # Dense centers straddle that patch center, rather than splitting phases.
        source_spacing = 32.0 / source_area
        source_origin = 16.0 * (1.0 - source_length / source_area) - source_spacing / 2.0
        target_spacing = 32.0 / target_area
        target_origin = 16.0 * (1.0 - target_length / target_area) - target_spacing / 2.0
        coordinates = torch.arange(target_length, device=device, dtype=torch.float32)
        coordinates = coordinates * target_spacing + target_origin
        source_indices = (coordinates - source_origin) / source_spacing
        grid_axes.append(source_indices * 2.0 / (source_length - 1) - 1.0)
    x, y = grid_axes
    xx = x.unsqueeze(0).expand(th, tw)
    yy = y.unsqueeze(1).expand(th, tw)
    return torch.stack((xx, yy), dim=-1).unsqueeze(0)


def resize_dense_field(value: torch.Tensor, target_h: int, target_w: int):
    """Spatial-only continuous transport, bounded scratch, unchanged B/C/T."""
    if not isinstance(value, torch.Tensor) or value.ndim != 5 or not value.is_floating_point():
        raise TypeError("Dense transport requires floating BxCxTxHxW features")
    if min(value.shape[:3]) < 1 or not torch.isfinite(value).all():
        raise ValueError("Dense transport requires nonempty finite features")
    sh, sw = value.shape[-2:]
    for item in (sh, sw, target_h, target_w):
        _spatial_size(item, "H/W")
    if (sh, sw) == (target_h, target_w):
        return value
    b, c, t = value.shape[:3]
    grid = dense_patch_grid((sh, sw), (target_h, target_w), device=value.device)
    # Process B/T as independent frames; encoder and decoder are never time-chunked.
    destination = value.new_empty((b, c, t, target_h, target_w))
    chunk_frames = max(1, SCRATCH_BYTES // (c * (sh * sw + target_h * target_w) * 4))
    for batch in range(b):
        for first in range(0, t, chunk_frames):
            last = min(t, first + chunk_frames)
            frames = value[batch, :, first:last].movedim(1, 0).float()
            resized = F.grid_sample(frames, grid.expand(last - first, -1, -1, -1),
                                    mode="bilinear", padding_mode="border", align_corners=True)
            destination[batch, :, first:last] = resized.movedim(0, 1).to(value.dtype)
    return destination


def _dense_network_forward(model, normalized, effective_scale, target_h, target_w):
    """Reuse T8 weights/layers, replacing only the encoder-to-decoder spatial map."""
    embedding = model.embed(normalized.new_tensor([[effective_scale - 1.0]]))
    embedding = embedding.expand(normalized.shape[0], -1)
    features = model._run_blocks(model.conv_in(normalized), embedding, model.in_blocks)
    features = resize_dense_field(features, target_h, target_w)
    features = model._run_blocks(features, embedding, model.out_blocks)
    return model.conv_out(F.silu(model.norm_out(features)))


@dataclass(frozen=True, slots=True)
class DenseTransferProvider:
    model_name: str
    precision: str = "fp16"
    release_policy: str = "offload_after"
    api_version: ClassVar[int] = 1
    kind: ClassVar[str] = "minimax_h3_learned_latent_upscaler"
    h3_patch_lattice_api: ClassVar[int] = 2

    def __post_init__(self):
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("Select a learned H3 upscaler checkpoint")
        learned._precision_dtype(self.precision)
        if self.release_policy not in learned.RELEASE_POLICIES:
            raise ValueError("Unknown upscaler release policy")

    @property
    def inference_device(self):
        return str(management.get_torch_device())

    def upscale_clean_video(self, video, *, target_h, target_w):
        # The ordinary API is deliberately NOT redirected to physical coordinates.
        return self._run(video, target_h, target_w, dense=False)

    def upscale_clean_video_h3_patch_lattice(self, video, *, target_h, target_w):
        return self._run(video, target_h, target_w, dense=True)

    def _run(self, video, target_h, target_w, *, dense):
        _spatial_size(target_h, "target H")
        _spatial_size(target_w, "target W")
        if (not isinstance(video, torch.Tensor) or video.ndim != 5
                or tuple(video.shape[:2]) != (1, 24) or not video.is_floating_point()):
            raise ValueError("Expected batch-1 floating native H3 Bx24xTxHxW video")
        _spatial_size(video.shape[-2], "source H")
        _spatial_size(video.shape[-1], "source W")
        if video.shape[2] < 1 or not torch.isfinite(video).all():
            raise ValueError("Expected finite nonempty video")
        if target_h < video.shape[-2] or target_w < video.shape[-1]:
            raise ValueError("Learned upscale cannot shrink either axis")
        if tuple(video.shape[-2:]) == (target_h, target_w):
            return video
        entry = None
        failed = True
        try:
            entry, _ = learned._load_cached_model(self.model_name, self.precision)
            dtype = learned._precision_dtype(self.precision)
            memory = 512 * video.shape[2] * target_h * target_w * torch.empty((), dtype=dtype).element_size() * 4
            management.load_models_gpu([entry.patcher], memory_required=memory + SCRATCH_BYTES,
                                       force_full_load=True)
            work = video.to(device=entry.patcher.load_device, dtype=dtype)
            mean = work.new_tensor(learned.LATENTS_MEAN).view(1, 24, 1, 1, 1)
            std = work.new_tensor(learned.LATENTS_STD).view(1, 24, 1, 1, 1)
            scale = ((target_h / video.shape[-2]) * (target_w / video.shape[-1])) ** 0.5
            with torch.inference_mode():
                normalized = (work - mean) / std
                if dense:
                    result = _dense_network_forward(entry.patcher.model, normalized, scale, target_h, target_w)
                else:
                    result = entry.patcher.model(normalized, scale, (video.shape[2], target_h, target_w))
                result = result * std + mean
            if result.shape != (1, 24, video.shape[2], target_h, target_w) or not torch.isfinite(result).all():
                raise RuntimeError("Learned transfer returned wrong shape or nonfinite output")
            result = result.to(device=video.device, dtype=video.dtype)
            failed = False
            return result
        finally:
            if entry is not None and (failed or self.release_policy != "keep_loaded"):
                management.unload_model_and_clones(entry.patcher)
            if entry is not None and (failed or self.release_policy == "clear_after"):
                learned._drop_cached_entry(entry)
            if failed or self.release_policy == "clear_after":
                gc.collect()
                management.soft_empty_cache()


def upscale_dense_av(*, av_latent, provider, scale_by=1.2):
    if not isinstance(av_latent, dict) or "samples" not in av_latent:
        raise ValueError("Expected native H3 AV LATENT")
    video, audio = _nested_parts(av_latent["samples"], "samples")
    if tuple(video.shape[:2]) != (1, 24) or tuple(audio.shape[:3]) != (1, 32, 2):
        raise ValueError("Expected batch-1 native joint H3 AV shapes")
    if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
        raise ValueError("AV input contains NaN or Inf")
    if getattr(provider, "h3_patch_lattice_api", None) != 2:
        raise ValueError("Physical transfer needs h3_patch_lattice_api=2, not phase-separated v1")
    operation = getattr(provider, "upscale_clean_video_h3_patch_lattice", None)
    if not callable(operation):
        raise TypeError("Provider lacks the explicit H3 dense transfer callable")
    geometry = learned.learned_upscale_geometry(video.shape[-1], video.shape[-2], "scale_by",
                                              scale_by, 1.0, 512, 288, "preserve_source", 1.05)
    height, width = int(geometry["output_latent_height"]), int(geometry["output_latent_width"])
    _spatial_size(video.shape[-2], "source H")
    _spatial_size(video.shape[-1], "source W")
    mask_status = "absent"
    mapped_mask = am = None
    if av_latent.get("noise_mask") is not None:
        vm, am = learned._nested_mask_parts(av_latent["noise_mask"])
        if vm.ndim != 5 or not vm.is_floating_point() or not torch.isfinite(vm).all():
            raise ValueError("Physical transfer expects a finite floating 5D video mask")
        if (vm.shape[0] not in (1, video.shape[0]) or vm.shape[1] not in (1, video.shape[1])
                or vm.shape[2] not in (1, video.shape[2]) or not torch.isfinite(am).all()
                or bool((vm < 0).any()) or bool((vm > 1).any())):
            raise ValueError("Invalid source-lattice video/audio noise mask")
        if vm.shape[-2] not in (1, video.shape[-2]) or vm.shape[-1] not in (1, video.shape[-1]):
            raise ValueError("Video mask is not on the source lattice")
        if tuple(vm.shape[-2:]) == (1, 1):
            mapped_mask = vm
            mask_status = "broadcast_preserved"
        else:
            expanded = vm.expand(*vm.shape[:-2], *video.shape[-2:])
            mapped_mask = resize_dense_field(expanded, height, width)
            mask_status = "same_dense_coordinate_transform"
    result = operation(video, target_h=height, target_w=width)
    if (not isinstance(result, torch.Tensor) or result.shape != (1, 24, video.shape[2], height, width)
            or result.dtype != video.dtype or result.device != video.device or not torch.isfinite(result).all()):
        raise RuntimeError("Provider returned invalid video shape/dtype/device/values")
    output = av_latent.copy()
    output["samples"] = comfy.nested_tensor.NestedTensor((result, audio))
    if mapped_mask is not None:
        output["noise_mask"] = comfy.nested_tensor.NestedTensor((mapped_mask, am))
    owned = type(provider) is DenseTransferProvider
    report = dict(schema_version=1, status="ok", spatial_lattice=LATTICE,
                  h3_patch_lattice_api=2, geometry=geometry,
                  transfer_position="encoder_to_decoder" if owned else "provider_declared_unverified",
                  complete_temporal_network=owned,
                  audio_object_preserved=True, audio_mask_object_preserved=True,
                  noise_mask=mask_status, extra_DiT_NFE=0,
                  provider_is_T8_owned=owned,
                  legacy_upscaler_changed=False, trained_quality_qualified=False,
                  full_Flow_workflow_qualified=False)
    return output, width * 16, height * 16, json.dumps(report, ensure_ascii=False, sort_keys=True)
