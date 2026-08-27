from __future__ import annotations

import gc
import hashlib
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management as model_management
import comfy.model_patcher
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
import folder_paths

from .latent_upscale import _nested_parts, _resize_mask_tensor
from .sampling import shift_sigma


PIXELS_PER_H3_LATENT = 16
PIXEL_ALIGNMENT = 32
H3_OFFICIAL_REFERENCE_PIXELS = 1920 * 1088
KNOWN_MODEL_SHA256 = "043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2"
SIZE_MODES = ("scale_by", "target_megapixels", "target_dimensions")
ASPECT_POLICIES = ("preserve_source", "honor_dimensions_exp")
PRECISIONS = ("fp16", "bf16", "fp32")
RELEASE_POLICIES = ("offload_after", "clear_after", "keep_loaded")
AUDIO_POLICIES = ("auto", "first_pass", "highres_template")
SECOND_PASS_AUDIO_SOURCES = ("legacy_policy", "first_pass", "highres_template")
UPSTREAM_WORKFLOW_COMMIT = "64fc9d4c7e2c03e8c61d6886182e3309365a1962"
UPSTREAM_REFINE_VIDEO_SIGMAS = {
    3: (0.9035, 0.6316, 0.3158, 0.0),
    4: (0.9035, 0.8000, 0.6316, 0.3158, 0.0),
    5: (0.9231, 0.8780, 0.8000, 0.6316, 0.3158, 0.0),
}


LATENTS_MEAN = (
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
)
LATENTS_STD = (
    1.2223774194717407,
    1.2767263650894165,
    1.6831774711608887,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.9653137922286987,
    1.0569885969161987,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049927,
    0.7197399735450745,
    0.6936293244361877,
    2.961095094680786,
    2.7694199085235596,
    3.0496184825897217,
    2.1088054180145264,
    3.276226282119751,
    3.1627357006073,
    2.2816812992095947,
    2.6127843856811523,
)


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels)


class _ResBlockEmb3D(nn.Module):
    def __init__(self, channels: int = 512, embed_dim: int = 64):
        super().__init__()
        self.in_layers = nn.Sequential(
            _group_norm(channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, channels * 2))
        self.out_norm = _group_norm(channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.in_layers(x)
        scale, shift = self.emb_layers(embedding).chunk(2, dim=1)
        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]
        hidden = self.out_norm(hidden) * (1.0 + scale) + shift
        return x + self.out_layers(hidden)


class _TemporalConv(nn.Module):
    def __init__(self, channels: int = 512):
        super().__init__()
        self.norm = _group_norm(channels)
        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(5, 1, 1),
            padding=(2, 0, 0),
            groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class MiniMaxH3LearnedResizer3D(nn.Module):
    """Network reconstructed from the published 3D checkpoint tensor contract."""

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv3d(24, 512, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
        self.in_blocks = self._make_blocks()
        self.out_blocks = self._make_blocks()
        self.norm_out = _group_norm(512)
        self.conv_out = nn.Conv3d(512, 24, 3, padding=1)
        self.device = torch.device("cpu")

    @staticmethod
    def _make_blocks() -> nn.ModuleList:
        blocks: list[nn.Module] = []
        for index in range(12):
            blocks.append(_ResBlockEmb3D())
            if index % 2 == 0:
                blocks.append(_TemporalConv())
        return nn.ModuleList(blocks)

    @staticmethod
    def _run_blocks(
        hidden: torch.Tensor,
        embedding: torch.Tensor,
        blocks: nn.ModuleList,
    ) -> torch.Tensor:
        for block in blocks:
            if isinstance(block, _ResBlockEmb3D):
                hidden = block(hidden, embedding)
            else:
                hidden = block(hidden)
        return hidden

    def forward(
        self,
        latent: torch.Tensor,
        effective_scale: float,
        target_size: tuple[int, int, int],
    ) -> torch.Tensor:
        if tuple(latent.shape[-3:]) == tuple(target_size):
            return latent
        scale_value = latent.new_tensor([[float(effective_scale) - 1.0]])
        embedding = self.embed(scale_value).expand(latent.shape[0], -1)
        hidden = self.conv_in(latent)
        hidden = self._run_blocks(hidden, embedding, self.in_blocks)
        hidden = F.interpolate(
            hidden,
            size=target_size,
            mode="trilinear",
            align_corners=False,
        )
        hidden = self._run_blocks(hidden, embedding, self.out_blocks)
        return self.conv_out(F.silu(self.norm_out(hidden)))


def _precision_dtype(precision: str) -> torch.dtype:
    mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    try:
        return mapping[precision]
    except KeyError as exc:
        raise ValueError(f"Unknown precision {precision!r}; expected one of {PRECISIONS}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_state_contract() -> dict[str, tuple[int, ...]]:
    with torch.device("meta"):
        model = MiniMaxH3LearnedResizer3D()
    return {key: tuple(value.shape) for key, value in model.state_dict().items()}


EXPECTED_STATE_CONTRACT = _expected_state_contract()


def validate_learned_resizer_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, Any]:
    actual = set(state_dict)
    expected = set(EXPECTED_STATE_CONTRACT)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    shape_mismatches = []
    for key in sorted(actual & expected):
        if tuple(state_dict[key].shape) != EXPECTED_STATE_CONTRACT[key]:
            shape_mismatches.append(
                {
                    "key": key,
                    "expected": EXPECTED_STATE_CONTRACT[key],
                    "actual": tuple(state_dict[key].shape),
                }
            )
    dtype_counts: dict[str, int] = {}
    for value in state_dict.values():
        dtype_counts[str(value.dtype)] = dtype_counts.get(str(value.dtype), 0) + 1
    if missing or unexpected or shape_mismatches or len(state_dict) != 322:
        raise ValueError(
            "Selected file is not the supported MiniMax H3 3D latent-upscaler checkpoint: "
            f"keys={len(state_dict)}, missing={missing[:4]}, unexpected={unexpected[:4]}, "
            f"shape_mismatches={shape_mismatches[:2]}"
        )
    if any(not torch.is_floating_point(value) for value in state_dict.values()):
        raise ValueError("The learned H3 latent-upscaler checkpoint must contain only floating tensors")
    return {
        "tensor_count": len(state_dict),
        "dtype_counts": dtype_counts,
        "channels": 24,
        "base_channels": 512,
        "residual_blocks_per_side": 12,
        "temporal_kernel": 5,
    }


@dataclass
class _CachedModel:
    patcher: comfy.model_patcher.ModelPatcher
    path: Path
    sha256: str
    precision: str
    contract: dict[str, Any]


_MODEL_CACHE: dict[tuple[str, int, int, str], _CachedModel] = {}
_MODEL_CACHE_LOCK = threading.RLock()


def _cache_key(path: Path, precision: str) -> tuple[str, int, int, str]:
    stat = path.stat()
    return str(path.resolve()).casefold(), int(stat.st_size), int(stat.st_mtime_ns), precision


def _drop_cached_entry(entry: _CachedModel) -> None:
    with _MODEL_CACHE_LOCK:
        for key, value in list(_MODEL_CACHE.items()):
            if value is entry:
                del _MODEL_CACHE[key]


def clear_learned_resizer_cache() -> int:
    with _MODEL_CACHE_LOCK:
        entries = list(_MODEL_CACHE.values())
        _MODEL_CACHE.clear()
    for entry in entries:
        model_management.unload_model_and_clones(entry.patcher)
    gc.collect()
    model_management.soft_empty_cache()
    return len(entries)


def _load_cached_model(model_name: str, precision: str) -> tuple[_CachedModel, bool]:
    model_path = Path(folder_paths.get_full_path_or_raise("latent_upscale_models", model_name))
    key = _cache_key(model_path, precision)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached, True

        file_hash = _sha256_file(model_path)
        state_dict = comfy.utils.load_torch_file(str(model_path), safe_load=True)
        try:
            contract = validate_learned_resizer_state_dict(state_dict)
            contract["state_contract_match"] = True
        except Exception as error:
            # User-selected model identity is diagnostic only.  The actual
            # architecture load below remains authoritative and may raise its
            # native PyTorch error when the file is incompatible.
            contract = {
                "state_contract_match": False,
                "state_contract_diagnostic": f"{type(error).__name__}: {error}",
            }
        contract["reference_file_match"] = file_hash.casefold() == KNOWN_MODEL_SHA256
        contract["model_identity_policy"] = "diagnostic_only_not_a_load_gate"
        with torch.device("meta"):
            network = MiniMaxH3LearnedResizer3D()
        network.load_state_dict(state_dict, strict=True, assign=True)
        del state_dict
        dtype = _precision_dtype(precision)
        if next(network.parameters()).dtype != dtype:
            network.to(device=torch.device("cpu"), dtype=dtype)
        network.eval()
        load_device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()
        patcher = comfy.model_patcher.CoreModelPatcher(
            network,
            load_device=load_device,
            offload_device=offload_device,
        )
        entry = _CachedModel(
            patcher=patcher,
            path=model_path,
            sha256=file_hash,
            precision=precision,
            contract=contract,
        )
        _MODEL_CACHE[key] = entry
        return entry, False


def _round_multiple(value: float, multiple: int = PIXEL_ALIGNMENT) -> int:
    return max(multiple, int(math.floor(value / multiple + 0.5)) * multiple)


def _best_aligned_size(
    ideal_width: float,
    ideal_height: float,
    source_aspect: float,
) -> tuple[int, int]:
    width_floor = max(PIXEL_ALIGNMENT, math.floor(ideal_width / PIXEL_ALIGNMENT) * PIXEL_ALIGNMENT)
    width_ceil = max(PIXEL_ALIGNMENT, math.ceil(ideal_width / PIXEL_ALIGNMENT) * PIXEL_ALIGNMENT)
    height_floor = max(PIXEL_ALIGNMENT, math.floor(ideal_height / PIXEL_ALIGNMENT) * PIXEL_ALIGNMENT)
    height_ceil = max(PIXEL_ALIGNMENT, math.ceil(ideal_height / PIXEL_ALIGNMENT) * PIXEL_ALIGNMENT)

    def score(size: tuple[int, int]) -> tuple[float, float]:
        width, height = size
        aspect_error = abs(math.log((width / height) / source_aspect))
        size_error = math.hypot(
            (width - ideal_width) / max(ideal_width, 1.0),
            (height - ideal_height) / max(ideal_height, 1.0),
        )
        return aspect_error, size_error

    return min(
        (
            (width, height)
            for width in {width_floor, width_ceil}
            for height in {height_floor, height_ceil}
        ),
        key=score,
    )


def learned_upscale_geometry(
    source_latent_width: int,
    source_latent_height: int,
    size_mode: str,
    scale_by: float,
    target_megapixels: float,
    target_width: int,
    target_height: int,
    aspect_policy: str,
    max_anisotropy: float,
) -> dict[str, Any]:
    if size_mode not in SIZE_MODES:
        raise ValueError(f"Unknown size_mode {size_mode!r}; expected one of {SIZE_MODES}")
    if aspect_policy not in ASPECT_POLICIES:
        raise ValueError(f"Unknown aspect_policy {aspect_policy!r}; expected one of {ASPECT_POLICIES}")
    source_width = int(source_latent_width) * PIXELS_PER_H3_LATENT
    source_height = int(source_latent_height) * PIXELS_PER_H3_LATENT
    source_aspect = source_width / source_height
    if size_mode == "scale_by":
        if not math.isfinite(scale_by) or not 1.0 <= float(scale_by) <= 4.0:
            raise ValueError("scale_by must be finite and within [1.0, 4.0]")
        ideal_width = source_width * float(scale_by)
        ideal_height = source_height * float(scale_by)
        output_width, output_height = _best_aligned_size(
            ideal_width, ideal_height, source_aspect
        )
    elif size_mode == "target_megapixels" or aspect_policy == "preserve_source":
        if size_mode == "target_megapixels":
            target_area = float(target_megapixels) * 1_000_000.0
            if not math.isfinite(target_area) or target_area <= 0:
                raise ValueError("target_megapixels must be positive and finite")
        else:
            target_area = float(target_width) * float(target_height)
        ideal_width = math.sqrt(target_area * source_aspect)
        ideal_height = ideal_width / source_aspect
        output_width, output_height = _best_aligned_size(
            ideal_width, ideal_height, source_aspect
        )
    else:
        ideal_width = float(target_width)
        ideal_height = float(target_height)
        output_width = _round_multiple(ideal_width)
        output_height = _round_multiple(ideal_height)

    if output_width < source_width or output_height < source_height:
        raise ValueError(
            "Learned latent resize only supports non-shrinking geometry: "
            f"source={source_width}x{source_height}, target={output_width}x{output_height}"
        )
    scale_x = output_width / source_width
    scale_y = output_height / source_height
    if max(scale_x, scale_y) > 4.0:
        raise ValueError("The learned model is limited to at most 4x on either spatial axis")
    anisotropy = max(scale_x, scale_y) / min(scale_x, scale_y)
    if not math.isfinite(max_anisotropy) or not 1.0 <= float(max_anisotropy) <= 2.0:
        raise ValueError("max_anisotropy must be finite and within [1.0, 2.0]")
    if anisotropy > float(max_anisotropy):
        raise ValueError(
            f"Requested anisotropic scale {scale_x:.5f}x{scale_y:.5f} has ratio "
            f"{anisotropy:.5f}, above max_anisotropy={float(max_anisotropy):.5f}"
        )
    output_pixels = output_width * output_height
    exceeds_official_reference_area = output_pixels > H3_OFFICIAL_REFERENCE_PIXELS
    return {
        "source_width": source_width,
        "source_height": source_height,
        "output_width": output_width,
        "output_height": output_height,
        "output_latent_width": output_width // PIXELS_PER_H3_LATENT,
        "output_latent_height": output_height // PIXELS_PER_H3_LATENT,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "effective_scale": math.sqrt(scale_x * scale_y),
        "anisotropy": anisotropy,
        "aspect_error_percent": abs((output_width / output_height) / source_aspect - 1.0)
        * 100.0,
        "size_mode": size_mode,
        "aspect_policy": aspect_policy,
        "output_pixels": output_pixels,
        "official_reference_pixels": H3_OFFICIAL_REFERENCE_PIXELS,
        "exceeds_official_reference_area": exceeds_official_reference_area,
        "memory_warning": (
            "Output exceeds the 1920x1088 official reference area. Execution is allowed; "
            "the user is responsible for VRAM, host-memory, runtime, and output validation."
            if exceeds_official_reference_area
            else None
        ),
    }


def _memory_snapshot(device: torch.device) -> dict[str, float | str | None]:
    snapshot: dict[str, float | str | None] = {"device": str(device)}
    try:
        snapshot["free_mib"] = float(model_management.get_free_memory(device)) / (1024**2)
    except Exception:
        snapshot["free_mib"] = None
    if device.type == "cuda" and torch.cuda.is_available():
        snapshot["allocated_mib"] = torch.cuda.memory_allocated(device) / (1024**2)
        snapshot["reserved_mib"] = torch.cuda.memory_reserved(device) / (1024**2)
    return snapshot


def learned_upscale_h3_av_latent(
    latent: dict,
    model_name: str,
    size_mode: str,
    scale_by: float,
    target_megapixels: float,
    target_width: int,
    target_height: int,
    aspect_policy: str,
    max_anisotropy: float,
    precision: str,
    release_policy: str,
) -> tuple[dict, int, int, str]:
    if release_policy not in RELEASE_POLICIES:
        raise ValueError(
            f"Unknown release_policy {release_policy!r}; expected one of {RELEASE_POLICIES}"
        )
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("Expected a MiniMax H3 LATENT dictionary containing samples")
    video, audio = _nested_parts(latent["samples"], "samples")
    if tuple(video.shape[:2]) != (1, 24) or tuple(audio.shape[:3]) != (1, 32, 2):
        raise ValueError(
            "Learned H3 latent upscale currently requires batch-1 native H3 AV shapes; "
            f"video={tuple(video.shape)}, audio={tuple(audio.shape)}"
        )
    if not torch.isfinite(video).all() or not torch.isfinite(audio).all():
        raise ValueError("Input H3 AV latent contains NaN or Inf")
    geometry = learned_upscale_geometry(
        int(video.shape[-1]),
        int(video.shape[-2]),
        size_mode,
        scale_by,
        target_megapixels,
        target_width,
        target_height,
        aspect_policy,
        max_anisotropy,
    )
    if tuple(video.shape[-2:]) == (
        geometry["output_latent_height"],
        geometry["output_latent_width"],
    ):
        report = {
            "schema_version": 1,
            "node": "MiniMaxH3LearnedLatentUpscaleT8Advanced",
            "status": "noop",
            "geometry": geometry,
            "model_loaded": False,
            "audio_preserved": True,
        }
        return (
            latent,
            int(geometry["output_width"]),
            int(geometry["output_height"]),
            json.dumps(report, ensure_ascii=False, sort_keys=True),
        )

    entry: _CachedModel | None = None
    cache_hit = False
    failed = True
    released = False
    cache_cleared = False
    device = model_management.get_torch_device()
    before = _memory_snapshot(device)
    try:
        entry, cache_hit = _load_cached_model(model_name, precision)
        target_h = int(geometry["output_latent_height"])
        target_w = int(geometry["output_latent_width"])
        dtype = _precision_dtype(precision)
        activation_estimate = (
            int(video.shape[0])
            * 512
            * int(video.shape[2])
            * target_h
            * target_w
            * torch.empty((), dtype=dtype).element_size()
            * 4
        )
        model_management.load_models_gpu(
            [entry.patcher],
            memory_required=activation_estimate,
            force_full_load=True,
        )
        compute_device = entry.patcher.load_device
        work = video.to(device=compute_device, dtype=dtype)
        mean = work.new_tensor(LATENTS_MEAN).view(1, 24, 1, 1, 1)
        std = work.new_tensor(LATENTS_STD).view(1, 24, 1, 1, 1)
        with torch.inference_mode():
            normalized = (work - mean) / std
            output_video = entry.patcher.model(
                normalized,
                float(geometry["effective_scale"]),
                (int(video.shape[2]), target_h, target_w),
            )
            output_video = output_video * std + mean
        if not torch.isfinite(output_video).all():
            raise RuntimeError("Learned H3 latent upscaler produced NaN or Inf")
        output_video = output_video.to(
            device=model_management.intermediate_device(), dtype=video.dtype
        )
        del work, normalized, mean, std
        output = latent.copy()
        output["samples"] = comfy.nested_tensor.NestedTensor((output_video, audio))
        mask_status = "absent"
        if latent.get("noise_mask") is not None:
            video_mask, audio_mask = _nested_parts(latent["noise_mask"], "noise_mask")
            resized_mask, mask_status = _resize_mask_tensor(
                video_mask,
                int(video.shape[-1]),
                int(video.shape[-2]),
                target_w,
                target_h,
            )
            output["noise_mask"] = comfy.nested_tensor.NestedTensor(
                (resized_mask, audio_mask)
            )
        failed = False
        if release_policy != "keep_loaded":
            model_management.unload_model_and_clones(entry.patcher)
            released = True
        if release_policy == "clear_after":
            _drop_cached_entry(entry)
            cache_cleared = True
            gc.collect()
            model_management.soft_empty_cache()
        after = _memory_snapshot(device)
        report = {
            "schema_version": 1,
            "node": "MiniMaxH3LearnedLatentUpscaleT8Advanced",
            "status": "ok",
            "model": {
                "name": model_name,
                "path": str(entry.path),
                "sha256": entry.sha256,
                "precision": precision,
                "cache_hit": cache_hit,
                "contract": entry.contract,
            },
            "geometry": geometry,
            "source_video_shape": list(video.shape),
            "output_video_shape": list(output_video.shape),
            "audio_shape": list(audio.shape),
            "audio_preserved": True,
            "noise_mask": mask_status,
            "release_policy": release_policy,
            "gpu_weights_released": released,
            "cpu_cache_cleared": cache_cleared,
            "memory_before": before,
            "memory_after_release": after,
        }
        return (
            output,
            int(geometry["output_width"]),
            int(geometry["output_height"]),
            json.dumps(report, ensure_ascii=False, sort_keys=True),
        )
    finally:
        if entry is not None and not released and (failed or release_policy != "keep_loaded"):
            model_management.unload_model_and_clones(entry.patcher)
        if entry is not None and not cache_cleared and (failed or release_policy == "clear_after"):
            _drop_cached_entry(entry)
        if failed or release_policy == "clear_after":
            gc.collect()
            model_management.soft_empty_cache()


def _clone_conditioning(conditioning):
    if not isinstance(conditioning, list):
        raise ValueError("positive must be a ComfyUI CONDITIONING list")
    return [[item[0], item[1].copy()] for item in conditioning]


def _validate_conditioning_spatial_contract(
    positive,
    target_h: int,
    target_w: int,
) -> dict[str, int]:
    keyframe_count = 0
    ref_count = 0
    for entry_index, item in enumerate(positive):
        if not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[1], dict):
            raise ValueError(f"positive[{entry_index}] is not a valid conditioning entry")
        metadata = item[1]
        for keyframe_index, keyframe in enumerate(metadata.get("minimax_keyframes") or []):
            keyframe_count += 1
            video_latent = keyframe.get("latent")
            if video_latent is not None:
                if not isinstance(video_latent, torch.Tensor) or video_latent.ndim != 5:
                    raise ValueError("MiniMax H3 keyframe latent must be rank-5")
                if tuple(video_latent.shape[-2:]) != (target_h, target_w):
                    raise ValueError(
                        "High-resolution conditioning was not rebuilt for the second pass: "
                        f"keyframe {keyframe_index} is {tuple(video_latent.shape[-2:])}, "
                        f"target is {(target_h, target_w)}. Run the current H3 Conditioning "
                        "node again at the high-resolution width/height."
                    )
        for ref_index, ref in enumerate(metadata.get("minimax_refs") or []):
            ref_count += 1
            kind = ref.get("kind")
            latent_value = ref.get("latent")
            if kind in {"image", "video", "video_audio"} and latent_value is not None:
                if not isinstance(latent_value, torch.Tensor) or latent_value.ndim != 5:
                    raise ValueError(f"Reference {ref_index} video latent must be rank-5")
                declared = (int(ref.get("latent_h", -1)), int(ref.get("latent_w", -1)))
                actual = tuple(int(value) for value in latent_value.shape[-2:])
                if declared != actual:
                    raise ValueError(
                        f"Reference {ref_index} metadata/latent mismatch: declared={declared}, "
                        f"actual={actual}"
                    )
    return {"keyframes": keyframe_count, "refs": ref_count}


def reconcile_two_pass_h3_latent(
    learned_latent: dict,
    highres_template: dict,
    positive,
    audio_policy: str,
    second_pass_audio_source: str = "legacy_policy",
    second_pass_audio_strength: float = 0.0,
) -> tuple[dict, Any, str]:
    if audio_policy not in AUDIO_POLICIES:
        raise ValueError(f"Unknown audio_policy {audio_policy!r}; expected one of {AUDIO_POLICIES}")
    if second_pass_audio_source not in SECOND_PASS_AUDIO_SOURCES:
        raise ValueError(
            f"Unknown second_pass_audio_source {second_pass_audio_source!r}; "
            f"expected one of {SECOND_PASS_AUDIO_SOURCES}"
        )
    second_pass_audio_strength = float(second_pass_audio_strength)
    if not math.isfinite(second_pass_audio_strength) or not 0.0 <= second_pass_audio_strength <= 1.0:
        raise ValueError("second_pass_audio_strength must be finite and between 0 and 1")
    learned_video, learned_audio = _nested_parts(learned_latent["samples"], "learned samples")
    template_video, template_audio = _nested_parts(
        highres_template["samples"], "high-resolution template samples"
    )
    if tuple(learned_video.shape) != tuple(template_video.shape):
        learned_pixels = (
            int(learned_video.shape[-1]) * PIXELS_PER_H3_LATENT,
            int(learned_video.shape[-2]) * PIXELS_PER_H3_LATENT,
        )
        template_pixels = (
            int(template_video.shape[-1]) * PIXELS_PER_H3_LATENT,
            int(template_video.shape[-2]) * PIXELS_PER_H3_LATENT,
        )
        raise ValueError(
            "Learned video latent and high-resolution Conditioning template must have identical "
            f"shape; learned={tuple(learned_video.shape)} ({learned_pixels[0]}x{learned_pixels[1]} px), "
            f"template={tuple(template_video.shape)} "
            f"({template_pixels[0]}x{template_pixels[1]} px). Connect this upscaler's width and "
            "height outputs directly to the high-resolution H3 Conditioning width and height "
            "inputs; do not maintain a second manual size."
        )
    if tuple(learned_audio.shape) != tuple(template_audio.shape):
        raise ValueError(
            "First-pass and high-resolution template audio shapes differ: "
            f"first={tuple(learned_audio.shape)}, template={tuple(template_audio.shape)}"
        )
    if not torch.isfinite(learned_video).all() or not torch.isfinite(learned_audio).all():
        raise ValueError("Learned first-pass latent contains NaN or Inf")
    positive_out = _clone_conditioning(positive)
    condition_counts = _validate_conditioning_spatial_contract(
        positive_out, int(learned_video.shape[-2]), int(learned_video.shape[-1])
    )
    template_mask = highres_template.get("noise_mask")
    locked_or_partial_audio = False
    template_video_mask = None
    template_audio_mask = None
    if template_mask is not None:
        template_video_mask, template_audio_mask = _nested_parts(
            template_mask, "high-resolution template noise_mask"
        )
        if tuple(template_video_mask.shape) != tuple(template_video.shape):
            raise ValueError("High-resolution template video noise_mask shape mismatch")
        if tuple(template_audio_mask.shape) != tuple(template_audio.shape):
            raise ValueError("High-resolution template audio noise_mask shape mismatch")
        locked_or_partial_audio = bool(torch.any(template_audio_mask < 1.0).item())
    if second_pass_audio_source == "legacy_policy":
        use_template_audio = audio_policy == "highres_template" or (
            audio_policy == "auto" and locked_or_partial_audio
        )
    else:
        use_template_audio = second_pass_audio_source == "highres_template"
    if use_template_audio:
        selected_audio = template_audio
        selected_audio_source = "highres_template"
    else:
        selected_audio = learned_audio
        selected_audio_source = "first_pass"
    output = highres_template.copy()
    output["samples"] = comfy.nested_tensor.NestedTensor((learned_video, selected_audio))
    if second_pass_audio_source == "legacy_policy" and template_mask is not None:
        output["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (template_video_mask, template_audio_mask)
        )
    elif second_pass_audio_source != "legacy_policy":
        if template_video_mask is None:
            template_video_mask = torch.ones_like(template_video)
            video_mask_source = "generated_all_ones"
        else:
            video_mask_source = "highres_template"
        explicit_audio_mask = torch.full_like(selected_audio, second_pass_audio_strength)
        output["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (template_video_mask, explicit_audio_mask)
        )
        template_audio_mask = explicit_audio_mask
    else:
        video_mask_source = "none"
    if second_pass_audio_source == "legacy_policy" and template_mask is not None:
        video_mask_source = "highres_template"
    audio_mask_min = None
    audio_mask_max = None
    if template_audio_mask is not None:
        audio_mask_min = float(torch.amin(template_audio_mask).item())
        audio_mask_max = float(torch.amax(template_audio_mask).item())
    report = {
        "schema_version": 2,
        "node": "MiniMaxH3TwoPassLatentReconcileT8Advanced",
        "status": "ok",
        "video_shape": list(learned_video.shape),
        "audio_shape": list(selected_audio.shape),
        "audio_policy": audio_policy,
        "audio_source": selected_audio_source,
        "second_pass_audio_source": second_pass_audio_source,
        "second_pass_audio_strength": (
            None
            if second_pass_audio_source == "legacy_policy"
            else second_pass_audio_strength
        ),
        "second_pass_audio_contract": (
            "legacy_template_mask"
            if second_pass_audio_source == "legacy_policy"
            else "explicit_source_and_strength"
        ),
        "second_pass_audio_locked": bool(
            template_audio_mask is not None and audio_mask_max == 0.0
        ),
        "second_pass_audio_mask_min": audio_mask_min,
        "second_pass_audio_mask_max": audio_mask_max,
        "second_pass_video_mask_source": video_mask_source,
        "template_has_mask": template_mask is not None,
        "template_audio_locked_or_partial": locked_or_partial_audio,
        "conditioning": condition_counts,
        "highres_template_metadata_authoritative": True,
    }
    return output, positive_out, json.dumps(report, ensure_ascii=False, sort_keys=True)


def audit_two_pass_h3_audio(
    second_pass_input: dict,
    second_pass_output: dict,
    expected_audio_strength: float,
    fail_on_locked_mismatch: bool,
    locked_atol: float,
) -> tuple[dict, str]:
    expected_audio_strength = float(expected_audio_strength)
    locked_atol = float(locked_atol)
    if not math.isfinite(expected_audio_strength) or not 0.0 <= expected_audio_strength <= 1.0:
        raise ValueError("expected_audio_strength must be finite and between 0 and 1")
    if not math.isfinite(locked_atol) or locked_atol < 0.0:
        raise ValueError("locked_atol must be finite and non-negative")

    input_video, input_audio = _nested_parts(
        second_pass_input["samples"], "second-pass input samples"
    )
    output_video, output_audio = _nested_parts(
        second_pass_output["samples"], "second-pass output samples"
    )
    if tuple(input_video.shape) != tuple(output_video.shape):
        raise ValueError(
            "Second-pass video shape changed unexpectedly: "
            f"input={tuple(input_video.shape)}, output={tuple(output_video.shape)}"
        )
    if tuple(input_audio.shape) != tuple(output_audio.shape):
        raise ValueError(
            "Second-pass audio shape changed unexpectedly: "
            f"input={tuple(input_audio.shape)}, output={tuple(output_audio.shape)}"
        )
    if not torch.isfinite(input_audio).all() or not torch.isfinite(output_audio).all():
        raise ValueError("Second-pass audio audit found NaN or Inf")

    input_mask = second_pass_input.get("noise_mask")
    if input_mask is None:
        raise ValueError(
            "Second-pass input has no noise_mask; audio preservation cannot be audited"
        )
    _video_mask, audio_mask = _nested_parts(input_mask, "second-pass input noise_mask")
    if tuple(audio_mask.shape) != tuple(input_audio.shape):
        raise ValueError("Second-pass audio noise_mask shape mismatch")
    audio_mask_min = float(torch.amin(audio_mask).item())
    audio_mask_max = float(torch.amax(audio_mask).item())
    if (
        abs(audio_mask_min - expected_audio_strength) > 1e-6
        or abs(audio_mask_max - expected_audio_strength) > 1e-6
    ):
        raise ValueError(
            "Second-pass audio mask does not match expected_audio_strength: "
            f"expected={expected_audio_strength}, min={audio_mask_min}, max={audio_mask_max}"
        )

    delta = output_audio.to(dtype=torch.float32) - input_audio.to(dtype=torch.float32)
    max_abs = float(torch.amax(torch.abs(delta)).item())
    rmse = float(torch.sqrt(torch.mean(delta.square())).item())
    exact_equal = bool(torch.equal(input_audio, output_audio))
    within_tolerance = max_abs <= locked_atol
    locked_mode = expected_audio_strength == 0.0
    if locked_mode and fail_on_locked_mismatch and not within_tolerance:
        raise ValueError(
            "Second-pass locked audio changed during sampling: "
            f"max_abs={max_abs}, rmse={rmse}, allowed_atol={locked_atol}. "
            "Do not decode or save this result."
        )

    verified_output = second_pass_output
    if locked_mode:
        verified_output = second_pass_output.copy()
        verified_output["samples"] = comfy.nested_tensor.NestedTensor(
            (output_video, input_audio)
        )

    report = {
        "schema_version": 1,
        "node": "MiniMaxH3TwoPassAudioAuditT8Advanced",
        "status": (
            "locked_audio_replaced_exact" if locked_mode and within_tolerance else "measured"
        ),
        "audio_shape": list(input_audio.shape),
        "expected_audio_strength": expected_audio_strength,
        "audio_mask_min": audio_mask_min,
        "audio_mask_max": audio_mask_max,
        "locked_mode": locked_mode,
        "fail_on_locked_mismatch": bool(fail_on_locked_mismatch),
        "locked_atol": locked_atol,
        "audio_exact_equal": exact_equal,
        "audio_within_tolerance": within_tolerance,
        "audio_max_abs": max_abs,
        "audio_rmse": rmse,
        "audio_relocked_exact": locked_mode,
        "sampled_latent_returned_unchanged": not locked_mode,
    }
    return verified_output, json.dumps(report, ensure_ascii=False, sort_keys=True)


def build_two_pass_sigma_plan(
    model,
    coarse_steps: int,
    refine_steps: int,
    restart_base_noise: float,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    coarse_steps = int(coarse_steps)
    refine_steps = int(refine_steps)
    restart_base_noise = float(restart_base_noise)
    if coarse_steps < 1 or refine_steps < 1:
        raise ValueError("coarse_steps and refine_steps must both be at least 1")
    if not 0.0 < restart_base_noise < 1.0:
        raise ValueError("restart_base_noise must be strictly between 0 and 1")
    model_sampling = model.get_model_object("model_sampling")
    shift_video = getattr(model_sampling, "shift", None)
    if shift_video is None:
        shift_video = getattr(model_sampling, "shift_video", None)
    if shift_video is None:
        raise ValueError(
            "The input MODEL does not expose a native-flow video shift. Connect the MODEL "
            "output of the current MiniMax H3 Dual-Clock Sampler setup."
        )
    shift_video = float(shift_video)
    transformer_options = model.model_options.get("transformer_options", {})
    shift_audio = transformer_options.get("minimax_h3_sigma_shift_audio")
    if shift_audio is None:
        shift_audio = getattr(model_sampling, "audio_shift", None)
    if shift_audio is None:
        raise ValueError(
            "The input MODEL does not expose the H3 audio shift. Connect the MODEL output "
            "of the current MiniMax H3 Dual-Clock Sampler setup."
        )
    shift_audio = float(shift_audio)
    coarse_q = torch.linspace(1.0, restart_base_noise, coarse_steps + 1, dtype=torch.float32)
    refine_q = torch.linspace(restart_base_noise, 0.0, refine_steps + 1, dtype=torch.float32)
    coarse_sigmas = shift_sigma(coarse_q, shift_video)
    refine_sigmas = shift_sigma(refine_q, shift_video)
    report = {
        "schema_version": 1,
        "node": "MiniMaxH3TwoPassSigmaPlanT8Advanced",
        "status": "ok",
        "schedule_domain": "base_flow_q",
        "coarse_steps": coarse_steps,
        "refine_steps": refine_steps,
        "total_nfe": coarse_steps + refine_steps,
        "restart_base_noise": restart_base_noise,
        "shift_video": shift_video,
        "shift_audio": shift_audio,
        "coarse_base_q": coarse_q.tolist(),
        "refine_base_q": refine_q.tolist(),
        "coarse_video_sigmas": coarse_sigmas.tolist(),
        "refine_video_sigmas": refine_sigmas.tolist(),
        "coarse_audio_sigmas": shift_sigma(coarse_q, shift_audio).tolist(),
        "refine_audio_sigmas": shift_sigma(refine_q, shift_audio).tolist(),
        "requirements": [
            "Use the first SamplerCustomAdvanced denoised_output, not its noisy output.",
            "Build Conditioning and dual-clock sampler setup separately for low and high resolutions.",
            "The H3 sampler derives the audio clock from the same base-flow q at runtime.",
        ],
    }
    return coarse_sigmas, refine_sigmas, json.dumps(report, ensure_ascii=False, sort_keys=True)


def _extract_h3_shifts(model) -> tuple[float, float]:
    model_sampling = model.get_model_object("model_sampling")
    shift_video = getattr(model_sampling, "shift", None)
    if shift_video is None:
        shift_video = getattr(model_sampling, "shift_video", None)
    if shift_video is None:
        raise ValueError(
            "The input MODEL does not expose a native-flow video shift. Connect the MODEL "
            "output of the current MiniMax H3 Dual-Clock Sampler setup."
        )
    transformer_options = model.model_options.get("transformer_options", {})
    shift_audio = transformer_options.get("minimax_h3_sigma_shift_audio")
    if shift_audio is None:
        shift_audio = getattr(model_sampling, "audio_shift", None)
    if shift_audio is None:
        raise ValueError(
            "The input MODEL does not expose the H3 audio shift. Connect the MODEL output "
            "of the current MiniMax H3 Dual-Clock Sampler setup."
        )
    shift_video = float(shift_video)
    shift_audio = float(shift_audio)
    if not math.isfinite(shift_video) or shift_video <= 0.0:
        raise ValueError("MiniMax H3 video shift must be finite and greater than zero")
    if not math.isfinite(shift_audio) or shift_audio <= 0.0:
        raise ValueError("MiniMax H3 audio shift must be finite and greater than zero")
    return shift_video, shift_audio


def _inverse_shift_sigma(values: torch.Tensor, shift: float) -> torch.Tensor:
    denominator = shift + values * (1.0 - shift)
    if not bool(torch.all(denominator > 0.0)):
        raise ValueError("Reference sigma shift produced a non-positive denominator")
    return values / denominator


def build_learned_two_pass_parity_plan(
    model,
    base_steps: int,
    coarse_steps: int,
    refine_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Reproduce the current published LBH two-pass schedule exactly.

    The upstream workflow uses a normal ``simple`` eight-step schedule, keeps its first
    four intervals for the low-resolution pass, then starts a fresh high-resolution
    pass from one of three manually published video-sigma sequences.  The I2V workflow
    applies those raw values at video shift 12 while the R2V workflow applies the same
    values at shift 6, so remapping them through a guessed reference shift is not parity.
    H3's active audio clock is still derived from each raw video sigma through base-flow
    time, exactly as the native joint AV sampler does.
    """
    base_steps = int(base_steps)
    coarse_steps = int(coarse_steps)
    refine_steps = int(refine_steps)
    if base_steps < 2 or base_steps > 1000:
        raise ValueError("base_steps must be between 2 and 1000")
    if coarse_steps < 1 or coarse_steps >= base_steps:
        raise ValueError("coarse_steps must satisfy 1 <= coarse_steps < base_steps")
    if refine_steps not in UPSTREAM_REFINE_VIDEO_SIGMAS:
        raise ValueError("refine_steps must be one of 3, 4, or 5 for upstream parity")

    shift_video, shift_audio = _extract_h3_shifts(model)
    model_sampling = model.get_model_object("model_sampling")
    full_sigmas = comfy.samplers.calculate_sigmas(
        model_sampling,
        "simple",
        base_steps,
    ).detach().to(device="cpu", dtype=torch.float32)
    if full_sigmas.numel() != base_steps + 1:
        raise RuntimeError(
            "ComfyUI simple scheduler returned an unexpected number of sigmas: "
            f"expected {base_steps + 1}, got {full_sigmas.numel()}"
        )
    if not bool(torch.all(full_sigmas[:-1] > full_sigmas[1:])) or float(full_sigmas[-1]) != 0.0:
        raise RuntimeError("ComfyUI simple scheduler did not return a strict H3 descent to zero")
    coarse_sigmas = full_sigmas[: coarse_steps + 1].clone()

    refine_sigmas = torch.tensor(
        UPSTREAM_REFINE_VIDEO_SIGMAS[refine_steps],
        dtype=torch.float32,
    )
    if not bool(torch.all(refine_sigmas[:-1] > refine_sigmas[1:])) or float(refine_sigmas[-1]) != 0.0:
        raise RuntimeError("Published upstream refine schedule is not a strict descent to zero")

    coarse_audio = shift_sigma(
        _inverse_shift_sigma(coarse_sigmas.to(dtype=torch.float64), shift_video),
        shift_audio,
    )
    refine_base_q = _inverse_shift_sigma(refine_sigmas.to(dtype=torch.float64), shift_video)
    refine_audio = shift_sigma(refine_base_q, shift_audio)
    report = {
        "schema_version": 2,
        "node": "MiniMaxH3LearnedTwoPassParityPlanT8Advanced",
        "status": "upstream_schedule_reproduced",
        "source": "LBH-123-AI Comfyui_Minimax_h3_latent_Upscaler workflow",
        "source_commit": UPSTREAM_WORKFLOW_COMMIT,
        "coarse_scheduler": "comfy.simple",
        "base_steps": base_steps,
        "coarse_steps": coarse_steps,
        "refine_steps": refine_steps,
        "total_nfe": coarse_steps + refine_steps,
        "shift_video": shift_video,
        "shift_audio": shift_audio,
        "published_refine_video_sigmas": refine_sigmas.tolist(),
        "refine_base_q_for_audio_clock": refine_base_q.tolist(),
        "coarse_video_sigmas": coarse_sigmas.tolist(),
        "coarse_audio_sigmas": coarse_audio.tolist(),
        "refine_video_sigmas": refine_sigmas.tolist(),
        "refine_audio_sigmas": refine_audio.tolist(),
        "requirements": [
            "Feed pass-1 denoised_output into the learned latent upscaler.",
            "Use fresh RandomNoise for pass 2 and decode SamplerCustomAdvanced output.",
            "Synchronize or rebuild visual conditioning for the enlarged latent.",
            "Apply Tail/Bias/STG/Restart only through the dedicated two-pass detail setup.",
        ],
        "previous_linear_q_plan_denied": True,
        "previous_reference_shift_remap_denied": True,
    }
    return coarse_sigmas, refine_sigmas, json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
    )
