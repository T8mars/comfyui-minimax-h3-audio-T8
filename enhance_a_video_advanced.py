from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import threading
from collections.abc import Mapping

import torch

import comfy.patcher_extension
from comfy.ldm.minimax.model import (
    FRAME_PER_TOKEN,
    Attention,
    MiniMaxH3Model,
    PackedLayout,
    patchify_video,
)
from comfy.ldm.modules import attention as attention_module
from comfy.model_base import MiniMaxH3 as MiniMaxH3BaseModel


# Clean-room H3 adapter derived from the equations in Enhance-A-Video / FETA.
# Paper: arXiv:2502.07508v3. Reference implementation (Apache-2.0):
# NUS-HPC-AI-Lab/Enhance-A-Video@16a7899e6f55f85ea19f1d3a415c6dc0c4096176.
EAV_RUNTIME_TYPE = "H3_T8_EAV_RUNTIME"
EAV_PATCH_VERSION = 1
EAV_RUNTIME_KEY = "t8_h3_eav_runtime"
EAV_WRAPPER_KEY = "t8_h3_eav_feta_v1"
EAV_MODES = ("disabled", "report_only", "apply_exp")
EAV_SAMPLING_PROFILES = ("stock20", "turbo8_alpha8")
EAV_ATTENTION_BACKENDS = ("native_optimized", "strict_sage_hnd")
EAV_SAGE_TASK_SCOPES = ("visual", "reference")
EAV_VISUAL_TASKS = ("T2VA", "I2VA", "FL2VA", "L2VA")
EAV_REFERENCE_TASKS = ("Ref2VA", "Hybrid")
EAV_ALL_TASKS = EAV_VISUAL_TASKS + EAV_REFERENCE_TASKS
EAV_TURBO8_BYPASS_HOOKS = 208

ATTENTION_FORWARD_SHA256S = {
    "4e8888f72ea5ccf68fb5ce5b1178ab0ddea66ca61137fcf01df2308ef27bf0be",
}
PACKED_LAYOUT_SHA256S = {
    "1124904e8835c6db068e61e304490d93784e6a8da6ca6b38afd93975611b3af4",
}
MODEL_FORWARD_SHA256S = {
    "14bdfccd6860f252005b8d43ab446aa9a938a13dc819061724b8f914218f5fd1",
}
PATCHIFY_VIDEO_SHA256S = {
    "b53a83b308cd69152a27f79a9e36f296f74f9f9a8ba8889319f8d32609cda645",
}


def _source_sha256(function) -> str:
    function = getattr(function, "__func__", function)
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _validate_sigma_schedule(sigmas: torch.Tensor, sampling_profile: str) -> dict:
    sampling_profile = str(sampling_profile)
    if sampling_profile not in EAV_SAMPLING_PROFILES:
        raise ValueError(f"Unknown H3 EAV sampling profile {sampling_profile!r}")
    nfe = 20 if sampling_profile == "stock20" else 8
    values = torch.as_tensor(sigmas).detach().float().cpu().flatten()
    if values.numel() != nfe + 1:
        raise ValueError(
            f"H3 EAV {sampling_profile} expects {nfe + 1} sigma entries including "
            "the terminal zero"
        )
    if not bool(torch.isfinite(values).all()):
        raise ValueError("H3 EAV received NaN/Inf sigmas")
    if not bool((values[:-1] > 0).all()) or abs(float(values[-1])) > 1e-7:
        raise ValueError(
            f"H3 EAV {sampling_profile} requires {nfe} positive sigmas followed by "
            "one exact zero"
        )
    if not bool((values[:-1] >= values[1:]).all()):
        raise ValueError("H3 EAV requires a monotonically non-increasing sigma schedule")
    digest = hashlib.sha256(values.contiguous().numpy().tobytes()).hexdigest()
    return {
        "profile": sampling_profile,
        "nfe": nfe,
        "entries": nfe + 1,
        "first_sigma": float(values[0]),
        "last_nonzero_sigma": float(values[-2]),
        "sigma_sha256": digest,
    }


def _validate_stock20_sigmas(sigmas: torch.Tensor) -> dict:
    """Backward-compatible internal helper retained for the original P1 tests."""
    return _validate_sigma_schedule(sigmas, "stock20")


def _pixel_frame_count(latent_frames: int) -> int:
    return sum(
        FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)]
        for index in range(int(latent_frames))
    )


def _classify_visual_task(keyframes, *, latent_frames: int, refs=None) -> str:
    keyframes = list(keyframes or ())
    if not keyframes:
        visual_task = "T2VA"
    else:
        final_frame = _pixel_frame_count(latent_frames) - 1
        positions = []
        for keyframe in keyframes:
            if not isinstance(keyframe, Mapping):
                raise RuntimeError("H3 EAV keyframe payload is not a mapping")
            if keyframe.get("latent") is None or keyframe.get("audio_latent") is not None:
                raise RuntimeError(
                    "H3 EAV currently accepts only stable visual first/last-frame guides"
                )
            try:
                positions.append(int(keyframe["resolved_frame_index"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("H3 EAV keyframe position is missing or invalid") from exc
        if positions == [0]:
            visual_task = "I2VA"
        elif positions == [final_frame]:
            visual_task = "L2VA"
        elif positions == [0, final_frame]:
            visual_task = "FL2VA"
        else:
            raise RuntimeError(
                "H3 EAV supports only stable first/last-frame keyframe layouts; "
                f"observed positions={positions}, expected final_frame={final_frame}"
            )
    if refs:
        return "Ref2VA" if visual_task == "T2VA" else "Hybrid"
    return visual_task


def _reference_segment_contract(refs) -> list[tuple[str, int]]:
    """Rebuild the pinned native PackedLayout reference segment sizes."""
    expected: list[tuple[str, int]] = []
    for index, block in enumerate(refs or ()):
        if not isinstance(block, Mapping):
            raise RuntimeError(f"H3 EAV reference block {index} is not a mapping")
        kind = str(block.get("kind", ""))
        if kind == "image":
            try:
                height = int(block["latent_h"])
                width = int(block["latent_w"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("H3 EAV image reference grid is missing or invalid") from exc
            if height <= 0 or width <= 0 or height % 2 or width % 2:
                raise RuntimeError("H3 EAV image reference grid must be positive and 2-aligned")
            expected.append(("ref_img", (height // 2) * (width // 2)))
        elif kind == "audio":
            try:
                audio_t = int(block["ref_audio_t"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("H3 EAV audio reference length is missing or invalid") from exc
            if audio_t < 0:
                raise RuntimeError("H3 EAV audio reference length cannot be negative")
            if audio_t:
                expected.append(("ref_audio", audio_t * 2))
        elif kind in {"video", "video_audio"}:
            try:
                latent_t = int(block["latent_t"])
                height = int(block["latent_h"])
                width = int(block["latent_w"])
                audio_t = int(block["ref_audio_t"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("H3 EAV video reference grid is missing or invalid") from exc
            if latent_t <= 0 or height <= 0 or width <= 0 or height % 2 or width % 2:
                raise RuntimeError(
                    "H3 EAV video reference grid must have positive time and 2-aligned space"
                )
            if audio_t < 0:
                raise RuntimeError("H3 EAV video reference audio length cannot be negative")
            if audio_t:
                expected.append(("ref_audio", audio_t * 2))
            expected.append(("ref_img", latent_t * (height // 2) * (width // 2)))
        else:
            raise RuntimeError(
                f"H3 EAV reference block {index} has unsupported kind {kind!r}"
            )
    return expected


def _turbo8_bypass_contract(model) -> dict:
    injections = getattr(model, "injections", {})
    nonempty = {key: value for key, value in injections.items() if bool(value)}
    if set(nonempty) != {"bypass_lora"} or len(nonempty["bypass_lora"]) != 1:
        raise RuntimeError(
            "H3 EAV turbo8_alpha8 requires exactly one bypass_lora injection from "
            "LoraLoaderBypassModelOnly"
        )
    injection = nonempty["bypass_lora"][0]
    inject = getattr(injection, "inject", None)
    closure = getattr(inject, "__closure__", None) or ()
    managers = [
        cell.cell_contents
        for cell in closure
        if type(cell.cell_contents).__name__ == "BypassInjectionManager"
    ]
    if len(managers) != 1:
        raise RuntimeError(
            "H3 EAV could not audit the current ComfyUI bypass-LoRA injection contract"
        )
    hooks = list(getattr(managers[0], "hooks", ()))
    multipliers = [float(getattr(hook, "multiplier", float("nan"))) for hook in hooks]
    if len(hooks) != EAV_TURBO8_BYPASS_HOOKS or not all(
        math.isfinite(value) and abs(value - 1.0) <= 1e-7 for value in multipliers
    ):
        raise RuntimeError(
            "H3 EAV turbo8_alpha8 requires the corrected 208-module Alpha8 bypass "
            "LoRA at strength 1.0"
        )
    return {
        "injection_key": "bypass_lora",
        "injection_count": 1,
        "hook_count": len(hooks),
        "strength_min": min(multipliers),
        "strength_max": max(multipliers),
    }


class EAVRuntime:
    """Execution-local telemetry shared by the MODEL patch and the audit node."""

    def __init__(self, config: Mapping):
        self.config = dict(config)
        self._lock = threading.Lock()
        self._run_index = 0
        self._consumed = True
        self._aborted = None
        self._forwards: list[dict] = []

    def begin_forward(self, *, sigma_video: float, progress_video: float, route: Mapping) -> int:
        with self._lock:
            if self._consumed:
                self._run_index += 1
                self._consumed = False
                self._aborted = None
                self._forwards = []
            forward = {
                "index": len(self._forwards),
                "sigma_video": float(sigma_video),
                "progress_video": float(progress_video),
                "active": bool(route["active"]),
                "task": str(route.get("task", "unknown")),
                "frames": int(route["frames"]),
                "spatial_tokens": int(route["spatial_tokens"]),
                "seq_len": int(route["seq_len"]),
                "audio_rows": int(route["audio_end"] - route["audio_start"]),
                "video_rows": int(route["video_end"] - route["video_start"]),
                "g_values": [],
                "cfi_values": [],
                "chunk_rows": [],
                "workspace_estimate_bytes": [],
                "strict_sage_call_count": 0,
                "strict_sage_failures": [],
            }
            self._forwards.append(forward)
            return int(forward["index"])

    def record(self, forward_index: int, *, g: float, cfi: float, chunk_rows: int, workspace: int):
        with self._lock:
            forward = self._forwards[int(forward_index)]
            forward["g_values"].append(float(g))
            forward["cfi_values"].append(float(cfi))
            forward["chunk_rows"].append(int(chunk_rows))
            forward["workspace_estimate_bytes"].append(int(workspace))

    def record_strict_sage_call(self, forward_index: int):
        with self._lock:
            forward = self._forwards[int(forward_index)]
            forward["strict_sage_call_count"] += 1

    def record_strict_sage_failure(self, forward_index: int, exc: BaseException):
        with self._lock:
            forward = self._forwards[int(forward_index)]
            forward["strict_sage_failures"].append(
                f"{type(exc).__name__}: {exc}"
            )

    def abort(self, exc: BaseException):
        with self._lock:
            self._aborted = f"{type(exc).__name__}: {exc}"
            self._consumed = True

    def snapshot(self, *, consume: bool) -> dict:
        with self._lock:
            all_g = [g for forward in self._forwards for g in forward["g_values"]]
            forwards = []
            for forward in self._forwards:
                g_values = list(forward["g_values"])
                cfi_values = list(forward["cfi_values"])
                workspaces = list(forward["workspace_estimate_bytes"])
                sage_failures = list(forward["strict_sage_failures"])
                forwards.append(
                    {
                        key: value
                        for key, value in forward.items()
                        if key
                        not in {
                            "g_values",
                            "cfi_values",
                            "chunk_rows",
                            "workspace_estimate_bytes",
                            "strict_sage_failures",
                        }
                    }
                    | {
                        "attention_count": len(g_values),
                        "g_min": min(g_values) if g_values else None,
                        "g_mean": (
                            sum(g_values) / len(g_values) if g_values else None
                        ),
                        "g_max": max(g_values) if g_values else None,
                        "cfi_mean": (
                            sum(cfi_values) / len(cfi_values) if cfi_values else None
                        ),
                        "chunk_rows": sorted(set(forward["chunk_rows"])),
                        "workspace_estimate_peak_mib": (
                            max(workspaces, default=0) / (1024 * 1024)
                        ),
                        "strict_sage_failure_count": len(sage_failures),
                        "strict_sage_failures": sage_failures,
                    }
                )
            active = [forward for forward in forwards if forward["active"]]
            report = {
                "schema": EAV_PATCH_VERSION,
                "run_index": self._run_index,
                "config": dict(self.config),
                "aborted": self._aborted,
                "model_forward_count": len(forwards),
                "active_forward_count": len(active),
                "attention_measurement_count": len(all_g),
                "attention_calls_per_active_forward": [
                    int(forward["attention_count"]) for forward in active
                ],
                "strict_sage_call_count": sum(
                    int(forward["strict_sage_call_count"]) for forward in forwards
                ),
                "strict_sage_calls_per_forward": [
                    int(forward["strict_sage_call_count"]) for forward in forwards
                ],
                "strict_sage_failure_count": sum(
                    int(forward["strict_sage_failure_count"]) for forward in forwards
                ),
                "strict_sage_fallback_count": 0,
                "g_min": min(all_g) if all_g else None,
                "g_mean": sum(all_g) / len(all_g) if all_g else None,
                "g_max": max(all_g) if all_g else None,
                "workspace_estimate_peak_mib": (
                    max(
                        (
                            value
                            for forward in self._forwards
                            for value in forward["workspace_estimate_bytes"]
                        ),
                        default=0,
                    )
                    / (1024 * 1024)
                ),
                "forwards": forwards,
            }
            if consume:
                self._consumed = True
            return report


def _assert_core_contract(model, *, sampling_profile: str) -> dict:
    if not hasattr(model, "clone") or not hasattr(model, "add_wrapper_with_key"):
        raise ValueError("H3 EAV requires a ComfyUI MODEL patcher")
    base = getattr(model, "model", None)
    if not isinstance(base, MiniMaxH3BaseModel):
        if type(getattr(base, "diffusion_model", None)).__name__ != "MiniMaxH3Model":
            raise ValueError("H3 EAV currently requires a native MiniMax H3 MODEL")

    hashes = {
        "attention_forward": _source_sha256(Attention.forward),
        "packed_layout": _source_sha256(PackedLayout.__init__),
        "model_forward": _source_sha256(MiniMaxH3Model._forward),
        "patchify_video": _source_sha256(patchify_video),
    }
    expected = {
        "attention_forward": ATTENTION_FORWARD_SHA256S,
        "packed_layout": PACKED_LAYOUT_SHA256S,
        "model_forward": MODEL_FORWARD_SHA256S,
        "patchify_video": PATCHIFY_VIDEO_SHA256S,
    }
    mismatches = [key for key, value in hashes.items() if value not in expected[key]]
    if mismatches:
        raise RuntimeError(
            "H3 EAV has not validated this ComfyUI H3 core contract: "
            + ", ".join(f"{key}={hashes[key]}" for key in mismatches)
        )

    transformer = getattr(model, "model_options", {}).get("transformer_options", {})
    if "optimized_attention_override" in transformer:
        raise RuntimeError("H3 EAV cannot stack with an existing attention override")
    replacements = transformer.get("patches_replace", {})
    if isinstance(replacements, Mapping) and any(bool(v) for v in replacements.values()):
        raise RuntimeError("H3 EAV cannot stack with BlockCache/STG/block replacements yet")
    wrappers = getattr(model, "wrappers", {})
    diffusion = wrappers.get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {})
    if any(bool(value) for value in diffusion.values()):
        raise RuntimeError("H3 EAV cannot stack with an existing diffusion wrapper yet")
    if bool(getattr(model, "patches", {})):
        raise RuntimeError("H3 EAV rejects ordinary weight patches and non-bypass LoRA")
    if sampling_profile == "stock20":
        if any(bool(value) for value in getattr(model, "injections", {}).values()):
            raise RuntimeError("H3 EAV stock20 rejects LoRA/weight injections")
        turbo_contract = None
    elif sampling_profile == "turbo8_alpha8":
        turbo_contract = _turbo8_bypass_contract(model)
    else:
        raise ValueError(f"Unknown H3 EAV sampling profile {sampling_profile!r}")
    object_patches = getattr(model, "object_patches", {})
    conflict_names = sorted(
        key
        for key in object_patches
        if key.startswith("diffusion_model.blocks.")
        or key in {"diffusion_model._forward", "diffusion_model.forward", "extra_conds"}
    )
    if conflict_names:
        raise RuntimeError(
            "H3 EAV cannot stack with existing H3 object patches: "
            + ", ".join(conflict_names)
        )
    return {"core_hashes": hashes, "turbo_contract": turbo_contract}


def _runtime_route(
    *,
    x,
    timestep,
    context,
    payload: Mapping,
    denoise_mask,
    audio_denoise_mask,
    start_progress: float,
    end_progress: float,
    allowed_tasks=EAV_VISUAL_TASKS,
    allow_reference_blocks: bool = False,
) -> dict:
    if denoise_mask is not None or audio_denoise_mask is not None:
        raise RuntimeError("H3 EAV rejects video/audio denoise masks")
    refs = list(payload.get("refs") or ())
    if refs and not allow_reference_blocks:
        raise RuntimeError("H3 EAV currently rejects Ref2VA/Hybrid reference blocks")
    try:
        video, audio = x[0], x[1]
    except (IndexError, TypeError) as exc:
        raise RuntimeError("H3 EAV requires the native two-stream AV latent") from exc
    if video.ndim != 5 or audio.ndim < 3 or int(video.shape[0]) != 1:
        raise RuntimeError("H3 EAV requires batch-1 native H3 video/audio latents")
    layout = payload.get("layout")
    if layout is None or not hasattr(layout, "segments"):
        raise RuntimeError("H3 EAV requires the native H3 PackedLayout payload")

    segments = list(layout.segments)
    video_segments = [segment for segment in segments if segment[2] == "video"]
    audio_segments = [segment for segment in segments if segment[2] == "audio"]
    if len(video_segments) != 1 or len(audio_segments) != 1:
        raise RuntimeError("H3 EAV could not uniquely isolate target audio/video rows")
    audio_start, audio_end, _ = audio_segments[0]
    video_start, video_end, _ = video_segments[0]
    if audio_end != video_start or video_end != int(layout.seq_len):
        raise RuntimeError("H3 EAV requires target audio/video as the final packed segments")

    frames = int(video.shape[2])
    keyframes = list(payload.get("keyframes") or ())
    task = _classify_visual_task(keyframes, latent_frames=frames, refs=refs)
    if task not in set(allowed_tasks):
        raise RuntimeError(f"H3 EAV task {task} is outside the enabled task scope")
    video_rows = int(video_end - video_start)
    if frames < 2 or video_rows % frames:
        raise RuntimeError("H3 EAV target-video rows do not form a valid temporal grid")
    spatial_tokens = video_rows // frames
    patch_h, patch_w = 2, 2
    expected_spatial = math.ceil(int(video.shape[3]) / patch_h) * math.ceil(
        int(video.shape[4]) / patch_w
    )
    if spatial_tokens != expected_spatial:
        raise RuntimeError("H3 EAV target-video token order/grid contract did not match H3")
    visual_task = _classify_visual_task(keyframes, latent_frames=frames)
    expected_cond_count = 0 if visual_task == "T2VA" else (2 if visual_task == "FL2VA" else 1)
    expected_segments = [("text", int(context.shape[1]))]
    expected_segments.extend(("cond", spatial_tokens) for _ in range(expected_cond_count))
    expected_segments.extend(_reference_segment_contract(refs))
    expected_segments.extend(
        [
            ("audio", int(audio_end - audio_start)),
            ("video", int(video_end - video_start)),
        ]
    )
    actual_segments = [
        (str(kind), int(end - start)) for start, end, kind in segments
    ]
    if actual_segments != expected_segments:
        raise RuntimeError(
            "H3 EAV PackedLayout segment order/sizes differ from the pinned native contract: "
            f"expected={expected_segments}, observed={actual_segments}"
        )
    if tuple(layout.signature) != (
        int(context.shape[1]),
        frames,
        int(video.shape[3]),
        int(video.shape[4]),
        int(audio.shape[-1]),
    ):
        raise RuntimeError("H3 EAV runtime latent dimensions differ from PackedLayout")

    sigma_video = float((timestep.flatten()[0] / 1000.0).detach().cpu())
    progress_video = 1.0 - sigma_video
    return {
        "seq_len": int(layout.seq_len),
        "task": task,
        "audio_start": int(audio_start),
        "audio_end": int(audio_end),
        "video_start": int(video_start),
        "video_end": int(video_end),
        "frames": frames,
        "spatial_tokens": spatial_tokens,
        "sigma_video": sigma_video,
        "progress_video": progress_video,
        "active": float(start_progress) <= progress_video <= float(end_progress),
        "reference_block_count": len(refs),
    }


def exact_chunked_cfi(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    frames: int,
    spatial_tokens: int,
    max_workspace_mib: int,
) -> tuple[torch.Tensor, int, int]:
    """Compute the paper CFI exactly, chunked over spatial positions and heads."""
    if q.ndim != 4 or k.ndim != 4 or q.shape != k.shape:
        raise RuntimeError("H3 EAV received an unsupported Q/K tensor layout")
    batch, heads, rows, head_dim = q.shape
    if batch != 1 or rows != int(frames) * int(spatial_tokens):
        raise RuntimeError("H3 EAV Q/K rows do not match the target-video grid")
    if frames < 2 or heads < 1 or head_dim < 1:
        raise RuntimeError("H3 EAV received an invalid temporal attention grid")

    # Live score storage includes the original logits, its FP32 copy, and softmax.
    bytes_per_spatial = int(heads) * int(frames) * int(frames) * 12
    budget = int(max_workspace_mib) * 1024 * 1024
    chunk_rows = max(1, min(int(spatial_tokens), budget // max(bytes_per_spatial, 1)))
    estimated_workspace = chunk_rows * bytes_per_spatial
    scale = float(head_dim) ** -0.5

    q_grid = q[0].reshape(heads, frames, spatial_tokens, head_dim).permute(2, 0, 1, 3)
    k_grid = k[0].reshape(heads, frames, spatial_tokens, head_dim).permute(2, 0, 1, 3)
    trace = torch.zeros((), device=q.device, dtype=torch.float64)
    for start in range(0, spatial_tokens, chunk_rows):
        end = min(start + chunk_rows, spatial_tokens)
        query = q_grid[start:end] * scale
        key = k_grid[start:end]
        logits = torch.matmul(query, key.transpose(-2, -1)).to(torch.float32)
        probabilities = torch.softmax(logits, dim=-1)
        trace = trace + torch.diagonal(probabilities, dim1=-2, dim2=-1).sum(
            dtype=torch.float64
        )
        del query, key, logits, probabilities

    matrix_count = int(spatial_tokens) * int(heads)
    numerator = float(matrix_count * frames) - trace
    denominator = float(matrix_count * frames * (frames - 1))
    cfi = numerator / denominator
    return cfi.to(dtype=torch.float32), int(chunk_rows), int(estimated_workspace)


def _strict_sage_contract() -> dict:
    available = bool(getattr(attention_module, "SAGE_ATTENTION_IS_AVAILABLE", False))
    kernel = getattr(attention_module, "sageattn", None)
    if not available or not callable(kernel):
        raise RuntimeError(
            "H3 EAV + Strict Sage requires a working sageattention installation"
        )
    try:
        signature = inspect.signature(kernel)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "H3 EAV + Strict Sage could not audit the sageattn call signature"
        ) from exc
    required = {"q", "k", "v", "tensor_layout", "is_causal", "sm_scale"}
    if not required.issubset(signature.parameters):
        raise RuntimeError(
            "H3 EAV + Strict Sage found an unsupported sageattn call signature: "
            f"{signature}"
        )
    try:
        version = importlib.metadata.version("sageattention")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {
        "backend": "sageattention.sageattn",
        "package_version": version,
        "signature": str(signature),
        "tensor_layout": "HND",
        "is_causal": False,
        "smooth_k": False,
        "silent_fallback": False,
        "scope": "native MiniMax H3 main DiT packed attention only",
    }


def _strict_sage_attention(
    q,
    k,
    v,
    heads,
    *,
    mask,
    skip_reshape,
    skip_output_reshape,
    **kwargs,
):
    if mask is not None:
        raise RuntimeError("H3 EAV + Strict Sage does not accept attention masks")
    if not skip_reshape or skip_output_reshape:
        raise RuntimeError(
            "H3 EAV + Strict Sage requires the native H3 HND input/output contract"
        )
    if kwargs.get("enable_gqa", False):
        raise RuntimeError("H3 EAV + Strict Sage does not accept GQA")
    if kwargs.get("low_precision_attention", True) is False:
        raise RuntimeError(
            "H3 EAV + Strict Sage cannot honor low_precision_attention=False"
        )
    if not all(isinstance(value, torch.Tensor) for value in (q, k, v)):
        raise RuntimeError("H3 EAV + Strict Sage requires tensor Q/K/V inputs")
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise RuntimeError("H3 EAV + Strict Sage requires equal 4D Q/K/V tensors")
    batch, observed_heads, _rows, dim_head = q.shape
    if batch != 1 or observed_heads != int(heads) or dim_head < 1:
        raise RuntimeError("H3 EAV + Strict Sage received an invalid H3 head layout")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise RuntimeError("H3 EAV + Strict Sage requires Q/K/V on one CUDA device")
    if q.dtype not in {torch.float16, torch.bfloat16} or not (
        k.dtype == q.dtype == v.dtype
    ):
        raise RuntimeError(
            "H3 EAV + Strict Sage requires matching FP16 or BF16 Q/K/V tensors"
        )

    kernel = getattr(attention_module, "sageattn", None)
    if not bool(
        getattr(attention_module, "SAGE_ATTENTION_IS_AVAILABLE", False)
    ) or not callable(kernel):
        raise RuntimeError("H3 EAV + Strict Sage kernel became unavailable at runtime")
    try:
        output = kernel(
            q,
            k,
            v,
            tensor_layout="HND",
            is_causal=False,
            sm_scale=kwargs.get("scale"),
            smooth_k=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "H3 EAV + Strict Sage kernel failed; no PyTorch-attention fallback was used"
        ) from exc
    if not isinstance(output, torch.Tensor) or output.shape != q.shape:
        raise RuntimeError(
            "H3 EAV + Strict Sage kernel returned an unexpected tensor shape"
        )
    return output.transpose(1, 2).reshape(batch, -1, int(heads) * dim_head)


def _delegate_eav_attention(
    q,
    k,
    v,
    heads,
    *,
    route,
    mask,
    attn_precision,
    skip_reshape,
    skip_output_reshape,
    transformer_options,
    delegate_kwargs,
):
    backend = str(route.get("attention_backend", "native_optimized"))
    if backend == "native_optimized":
        return attention_module.optimized_attention(
            q,
            k,
            v,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            transformer_options=transformer_options,
            **delegate_kwargs,
        )
    if backend != "strict_sage_hnd":
        raise RuntimeError(f"Unknown H3 EAV attention backend {backend!r}")
    try:
        output = _strict_sage_attention(
            q,
            k,
            v,
            heads,
            mask=mask,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **delegate_kwargs,
        )
    except Exception as exc:
        route["runtime"].record_strict_sage_failure(
            int(route["forward_index"]), exc
        )
        raise
    route["runtime"].record_strict_sage_call(int(route["forward_index"]))
    return output


def route_eav_attention(
    q,
    k,
    v,
    heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    skip_output_reshape=False,
    transformer_options=None,
    **kwargs,
):
    transformer_options = transformer_options or {}
    route = transformer_options.get(EAV_RUNTIME_KEY)
    delegate_kwargs = dict(kwargs)
    delegate_kwargs["_inside_attn_wrapper"] = True
    if route is None or q.shape[-2] != int(route["seq_len"]):
        return attention_module.optimized_attention(
            q,
            k,
            v,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            transformer_options=transformer_options,
            **delegate_kwargs,
        )
    if mask is not None:
        raise RuntimeError("H3 EAV cannot stack with a pre-existing attention mask")
    if not skip_reshape or skip_output_reshape:
        raise RuntimeError("H3 EAV received an unsupported native H3 attention call")
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[1] != heads:
        raise RuntimeError("H3 EAV requires batch-1 packed attention")

    output = _delegate_eav_attention(
        q,
        k,
        v,
        heads,
        route=route,
        mask=None,
        attn_precision=attn_precision,
        skip_reshape=True,
        skip_output_reshape=False,
        transformer_options=transformer_options,
        delegate_kwargs=delegate_kwargs,
    )
    if not route["active"]:
        return output

    video_start = int(route["video_start"])
    video_end = int(route["video_end"])
    cfi, chunk_rows, workspace = exact_chunked_cfi(
        q[:, :, video_start:video_end],
        k[:, :, video_start:video_end],
        frames=int(route["frames"]),
        spatial_tokens=int(route["spatial_tokens"]),
        max_workspace_mib=int(route["max_workspace_mib"]),
    )
    g = torch.clamp_min((float(route["frames"]) + float(route["tau"])) * cfi, 1.0)
    g_value = float(g.detach().cpu())
    cfi_value = float(cfi.detach().cpu())
    if not math.isfinite(g_value) or g_value < 1.0:
        raise RuntimeError("H3 EAV produced a non-finite/invalid enhancement factor")
    if g_value > float(route["g_hard_limit"]):
        raise RuntimeError(
            f"H3 EAV g={g_value:.6f} exceeded the configured hard limit "
            f"{float(route['g_hard_limit']):.6f}; output was refused rather than clamped"
        )
    route["runtime"].record(
        int(route["forward_index"]),
        g=g_value,
        cfi=cfi_value,
        chunk_rows=chunk_rows,
        workspace=workspace,
    )
    if route["mode"] == "report_only":
        return output
    if route["mode"] != "apply_exp":
        raise RuntimeError(f"Unknown H3 EAV runtime mode {route['mode']!r}")

    # optimized_attention returns a fresh result owned by this routing call.  Do
    # not clone the full packed AV output merely to touch the target-video rows:
    # at 0.7MP that transient copy is hundreds of MiB and can force DynamicVRAM
    # paging.  The in-place slice multiply is mathematically identical, leaves
    # text/condition/audio rows untouched, and preserves the existing backend as
    # the authoritative attention implementation.
    output[:, video_start:video_end].mul_(
        g.to(device=output.device, dtype=output.dtype)
    )
    return output


def build_eav_model(
    model,
    sigmas: torch.Tensor,
    *,
    mode: str,
    tau: float,
    start_video_progress: float,
    end_video_progress: float,
    max_workspace_mib: int,
    g_hard_limit: float,
    sampling_profile: str = "stock20",
    allowed_tasks=EAV_VISUAL_TASKS,
    allow_reference_blocks: bool = False,
    composer_profile: str = "isolated_visual",
    attention_backend: str = "native_optimized",
):
    mode = str(mode)
    if mode not in EAV_MODES:
        raise ValueError(f"Unknown H3 EAV mode {mode!r}")
    if not 0.0 <= float(start_video_progress) < float(end_video_progress) <= 1.0:
        raise ValueError("H3 EAV progress window must satisfy 0 <= start < end <= 1")
    if not -32.0 <= float(tau) <= 32.0:
        raise ValueError("H3 EAV tau must be between -32 and 32")
    if not 4 <= int(max_workspace_mib) <= 512:
        raise ValueError("H3 EAV max_workspace_mib must be between 4 and 512")
    if not 1.0 <= float(g_hard_limit) <= 3.0:
        raise ValueError("H3 EAV g_hard_limit must be between 1 and 3")

    sampling_profile = str(sampling_profile)
    attention_backend = str(attention_backend)
    if attention_backend not in EAV_ATTENTION_BACKENDS:
        raise ValueError(f"Unknown H3 EAV attention backend {attention_backend!r}")
    allowed_tasks = tuple(str(task) for task in allowed_tasks)
    if not allowed_tasks or not set(allowed_tasks).issubset(set(EAV_ALL_TASKS)):
        raise ValueError(f"H3 EAV received an invalid task scope: {allowed_tasks!r}")
    if allow_reference_blocks and not set(allowed_tasks).issubset(set(EAV_REFERENCE_TASKS)):
        raise ValueError(
            "H3 EAV reference composer can enable only Ref2VA/Hybrid task scopes"
        )
    if not allow_reference_blocks and any(
        task in set(EAV_REFERENCE_TASKS) for task in allowed_tasks
    ):
        raise ValueError("H3 EAV reference tasks require the explicit reference composer")
    sigma_contract = _validate_sigma_schedule(sigmas, sampling_profile)
    config = {
        "schema": EAV_PATCH_VERSION,
        "mode": mode,
        "paper": "Enhance-A-Video, arXiv:2502.07508v3",
        "reference_commit": "16a7899e6f55f85ea19f1d3a415c6dc0c4096176",
        "adapter_scope": "target_video_only_full3d_h3_exp",
        "composer_profile": str(composer_profile),
        "task_scope": list(allowed_tasks),
        "allow_reference_blocks": bool(allow_reference_blocks),
        "sampling_profile": sampling_profile,
        "attention_backend": attention_backend,
        "attention_backend_scope": "native MiniMax H3 main DiT packed attention only",
        "tau": float(tau),
        "start_video_progress": float(start_video_progress),
        "end_video_progress": float(end_video_progress),
        "max_workspace_mib": int(max_workspace_mib),
        "g_hard_limit": float(g_hard_limit),
        "direct_scaled_rows": ["target_video"],
        "direct_audio_scaling": False,
        "output_scaling": "in_place_target_video_slice_no_full_packed_clone",
        "sigma_contract": sigma_contract,
        "scientific_boundary": (
            "H3 is a joint packed AV Transformer. This adapter computes temporal CFI from "
            "target-video Q/K and directly scales only target-video attention output rows, "
            "but later layers may still change audio indirectly. H3 quality is not assumed."
        ),
    }
    runtime = EAVRuntime(config)
    if mode == "disabled":
        config["core_hashes"] = None
        config["notes"] = [
            "disabled returns the original MODEL without installing any wrapper or attention hook",
            "tau=0 is not an off switch in the paper equation; use disabled for exact bypass",
        ]
        return model, runtime, _json(config)

    contracts = _assert_core_contract(model, sampling_profile=sampling_profile)
    config["core_hashes"] = contracts["core_hashes"]
    config["turbo_contract"] = contracts["turbo_contract"]
    config["attention_backend_contract"] = (
        _strict_sage_contract() if attention_backend == "strict_sage_hnd" else None
    )
    config["notes"] = [
        "report_only computes CFI/g but leaves the attention output unchanged",
        "apply_exp follows the paper residual gain pattern through an H3 full-3D adapter",
        (
            "Ref2VA/Hybrid native reference segments are explicitly audited while only target-video "
            "rows are measured/scaled; denoise masks remain rejected"
            if allow_reference_blocks
            else "T2VA/I2VA/FL2VA/L2VA are isolated mechanically; references and masks remain rejected"
        ),
        (
            "the explicit Strict Sage composer owns the audited Sage backend; external Sage "
            "object patches remain rejected"
            if attention_backend == "strict_sage_hnd"
            else "Prompt Relay, BlockCache, Sage object patches and STG remain rejected"
        ),
        "Turbo8 accepts only the corrected 208-module Alpha8 bypass LoRA at strength 1.0",
        "the runtime audit after sampling is authoritative for observed g and call counts",
    ]
    patched = model.clone()

    def _diffusion_wrapper(
        executor,
        x,
        timestep,
        context,
        transformer_options=None,
        **kwargs,
    ):
        transformer_options = transformer_options if transformer_options is not None else {}
        if len(executor.wrappers) != 1:
            raise RuntimeError("H3 EAV detected another diffusion wrapper added after binding")
        installed = transformer_options.get("optimized_attention_override")
        if getattr(installed, "_t8_h3_eav_patch_version", None) != EAV_PATCH_VERSION:
            raise RuntimeError("H3 EAV attention override was replaced after binding")
        replacements = transformer_options.get("patches_replace", {})
        if isinstance(replacements, Mapping) and any(bool(v) for v in replacements.values()):
            raise RuntimeError("H3 EAV detected a runtime block replacement and refused it")
        payload = kwargs.get("minimax_payload")
        if not isinstance(payload, Mapping):
            raise RuntimeError("H3 EAV could not find the native H3 minimax_payload")
        if EAV_RUNTIME_KEY in transformer_options:
            raise RuntimeError("Nested H3 EAV runtime state was refused")
        try:
            route = _runtime_route(
                x=x,
                timestep=timestep,
                context=context,
                payload=payload,
                denoise_mask=kwargs.get("denoise_mask"),
                audio_denoise_mask=kwargs.get("audio_denoise_mask"),
                start_progress=float(start_video_progress),
                end_progress=float(end_video_progress),
                allowed_tasks=allowed_tasks,
                allow_reference_blocks=bool(allow_reference_blocks),
            )
            forward_index = runtime.begin_forward(
                sigma_video=route["sigma_video"],
                progress_video=route["progress_video"],
                route=route,
            )
            route.update(
                {
                    "mode": mode,
                    "tau": float(tau),
                    "max_workspace_mib": int(max_workspace_mib),
                    "g_hard_limit": float(g_hard_limit),
                    "runtime": runtime,
                    "forward_index": forward_index,
                    "attention_backend": attention_backend,
                }
            )
            transformer_options[EAV_RUNTIME_KEY] = route
            return executor(x, timestep, context, transformer_options, **kwargs)
        except BaseException as exc:
            runtime.abort(exc)
            raise
        finally:
            transformer_options.pop(EAV_RUNTIME_KEY, None)

    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        EAV_WRAPPER_KEY,
        _diffusion_wrapper,
    )
    patched.set_model_optimized_attention(route_eav_attention)
    installed = patched.model_options["transformer_options"]["optimized_attention_override"]
    installed._t8_h3_eav_patch_version = EAV_PATCH_VERSION
    if hasattr(patched, "set_attachments"):
        patched.set_attachments(EAV_WRAPPER_KEY, dict(config))
    return patched, runtime, _json(config)


def finalize_eav_runtime(av_latent, runtime: EAVRuntime):
    if not isinstance(runtime, EAVRuntime):
        raise TypeError("H3 EAV Audit requires the runtime token from the matching EAV node")
    report = runtime.snapshot(consume=True)
    mode = report["config"]["mode"]
    if report["aborted"]:
        raise RuntimeError("H3 EAV sampling aborted: " + report["aborted"])
    if mode == "disabled":
        report["status"] = "disabled_identity"
    else:
        expected_nfe = int(report["config"]["sigma_contract"]["nfe"])
        if report["model_forward_count"] != expected_nfe:
            raise RuntimeError(
                f"H3 EAV {report['config']['sampling_profile']} audit expected "
                f"{expected_nfe} model forwards, observed "
                f"{report['model_forward_count']}"
            )
        active_counts = report["attention_calls_per_active_forward"]
        if not active_counts or any(count != 50 for count in active_counts):
            raise RuntimeError(
                "H3 EAV expected exactly 50 main DiT attention measurements per active "
                f"forward, observed {active_counts}"
            )
        if report["config"].get("attention_backend") == "strict_sage_hnd":
            expected_backend_calls = [50] * expected_nfe
            observed_backend_calls = report["strict_sage_calls_per_forward"]
            if observed_backend_calls != expected_backend_calls:
                raise RuntimeError(
                    "H3 EAV + Strict Sage expected exactly 50 successful Sage calls per "
                    f"model forward, observed {observed_backend_calls}"
                )
            if report["strict_sage_failure_count"] or report[
                "strict_sage_fallback_count"
            ]:
                raise RuntimeError(
                    "H3 EAV + Strict Sage detected a kernel failure or backend fallback"
                )
        report["status"] = "report_only_verified" if mode == "report_only" else "apply_exp_verified"
    report["quality_claim"] = (
        "mechanically audited only; visual motion/detail and joint-AV audio quality require "
        "the controlled baseline/apply A/B review"
    )
    return av_latent, _json(report)
