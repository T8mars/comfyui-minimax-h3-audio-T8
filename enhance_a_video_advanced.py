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

from .detail_sampling_advanced import (
    _parse_h3_blocks,
    apply_h3_spatiotemporal_guidance,
)


# Clean-room H3 adapter derived from the equations in Enhance-A-Video / FETA.
# Paper: arXiv:2502.07508v3. Reference implementation (Apache-2.0):
# NUS-HPC-AI-Lab/Enhance-A-Video@16a7899e6f55f85ea19f1d3a415c6dc0c4096176.
EAV_RUNTIME_TYPE = "H3_T8_EAV_RUNTIME"
EAV_PATCH_VERSION = 1
EAV_RUNTIME_KEY = "t8_h3_eav_runtime"
EAV_WRAPPER_KEY = "t8_h3_eav_feta_v1"
EAV_PROMPT_RELAY_WRAPPER_KEY = "t8_h3_eav_prompt_relay_v1"
EAV_PROMPT_RELAY_PATCH_VERSION = 1
EAV_BLOCK_CACHE_WRAPPER_KEY = "t8_h3_eav_block_cache_v1"
EAV_BLOCK_CACHE_PATCH_VERSION = 1
EAV_STG_WRAPPER_KEY = "t8_h3_eav_stg_v1"
EAV_STG_PATCH_VERSION = 1
EAV_STG_BRANCH_KEY = "t8_h3_eav_stg_branch_v1"
EAV_LONG_VIDEO_WRAPPER_KEY = "t8_h3_eav_long_video_v1"
EAV_LONG_VIDEO_PATCH_VERSION = 1
EAV_PROMPT_RELAY_LONG_VIDEO_WRAPPER_KEY = "t8_h3_eav_prompt_relay_long_video_v1"
EAV_PROMPT_RELAY_LONG_VIDEO_PATCH_VERSION = 1
BLOCK_CACHE_KEY = "minimax_h3_block_cache_t8"
BLOCK_CACHE_WRAPPER_KEY = "minimax_h3_block_cache_t8"
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
BLOCK_CACHE_OUTER_WRAPPER_SHA256S = {
    "6a150ac20157ae73ba9b6669b9af4350f818c9234c327af7812728c7d792f4cd",
}
BLOCK_CACHE_DIFFUSION_WRAPPER_SHA256S = {
    "408e4d515653bee5b9c3bd06165f9247e7b3a2cf5c5fce9a321dba9b25ca7226",
}
BLOCK_CACHE_CLASS_SHA256S = {
    "85f5bf37cc5c8828b28e15e6ba4ec4a3c31608c08458dd3d3445ac372403b7a5",
}
BLOCK_CACHE_PATCH_CALL_SHA256S = {
    "58f154e9de31d6fd8c38c8454f91fe04b27b62f64028641f1b846726bf4e6cac",
}
BLOCK_CACHE_CONFIG_CLASS_SHA256S = {
    "eb1661386fc2e5c489da2caff6856f747964a305114af858d9b470599e894388",
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
    bypass_values = list(nonempty.get("bypass_lora", ()) or ())
    injection = bypass_values[0] if bypass_values else None
    inject = getattr(injection, "inject", None)
    closure = getattr(inject, "__closure__", None) or ()
    managers = [
        cell.cell_contents
        for cell in closure
        if type(cell.cell_contents).__name__ == "BypassInjectionManager"
    ]
    hooks = list(getattr(managers[0], "hooks", ())) if managers else []
    multipliers = [float(getattr(hook, "multiplier", float("nan"))) for hook in hooks]
    strength_match = bool(multipliers) and all(
        math.isfinite(value) and abs(value - 1.0) <= 1e-7 for value in multipliers
    )
    return {
        "injection_key": "bypass_lora" if bypass_values else None,
        "injection_count": len(bypass_values),
        "other_injection_keys": sorted(key for key in nonempty if key != "bypass_lora"),
        "manager_count": len(managers),
        "hook_count": len(hooks),
        "strength_min": min(multipliers) if multipliers else None,
        "strength_max": max(multipliers) if multipliers else None,
        "reference_hook_count_match": len(hooks) == EAV_TURBO8_BYPASS_HOOKS,
        "reference_strength_match": strength_match,
        "model_identity_policy": "diagnostic_only_not_a_load_gate",
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

    def begin_forward(
        self,
        *,
        sigma_video: float,
        progress_video: float,
        route: Mapping,
        branch: str = "main",
        skipped_blocks=(),
    ) -> int:
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
                "branch": str(branch),
                "skipped_blocks": [int(value) for value in skipped_blocks],
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
                "block_cache_decision": None,
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

    def record_block_cache_decision(self, forward_index: int, decision: str):
        decision = str(decision)
        if decision not in {"full", "hit"}:
            raise ValueError(f"Unknown H3 BlockCache decision {decision!r}")
        with self._lock:
            forward = self._forwards[int(forward_index)]
            if forward["block_cache_decision"] is not None:
                raise RuntimeError("H3 EAV BlockCache decision was recorded twice")
            forward["block_cache_decision"] = decision

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


def _assert_no_sampler_guidance_hooks(model, *, owner: str) -> None:
    options = getattr(model, "model_options", {})
    conflict_keys = (
        "sampler_cfg_function",
        "sampler_pre_cfg_function",
        "sampler_post_cfg_function",
        "sampler_calc_cond_batch_function",
        "model_function_wrapper",
    )
    conflicts = [key for key in conflict_keys if bool(options.get(key))]
    if conflicts:
        raise RuntimeError(
            f"{owner} refuses existing sampler/model guidance hooks: "
            + ", ".join(conflicts)
        )


def _assert_core_contract(
    model,
    *,
    sampling_profile: str,
    allowed_live_extra_conds_patch_versions: tuple[int, ...] = (),
) -> dict:
    if not hasattr(model, "clone") or not hasattr(model, "add_wrapper_with_key"):
        raise ValueError("H3 EAV requires a ComfyUI MODEL patcher")
    _assert_no_sampler_guidance_hooks(model, owner="H3 EAV")
    base = getattr(model, "model", None)
    native_h3_model_observed = isinstance(base, MiniMaxH3BaseModel) or (
        type(getattr(base, "diffusion_model", None)).__name__ == "MiniMaxH3Model"
    )

    hashes = {
        "attention_forward": _source_sha256(Attention.forward),
        "packed_layout": _source_sha256(PackedLayout.__init__),
        "model_forward": _source_sha256(MiniMaxH3Model._forward),
        "patchify_video": _source_sha256(patchify_video),
    }
    from .sla_attention_advanced import _core_semantic_contract

    semantic_contract = _core_semantic_contract()

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
    if sampling_profile == "stock20":
        turbo_contract = {
            "model_patches_observed": len(getattr(model, "patches", {}) or {}),
            "injection_keys_observed": sorted(
                key
                for key, value in (getattr(model, "injections", {}) or {}).items()
                if bool(value)
            ),
            "model_identity_policy": "diagnostic_only_not_a_load_gate",
        }
    elif sampling_profile == "turbo8_alpha8":
        turbo_contract = _turbo8_bypass_contract(model)
    else:
        raise ValueError(f"Unknown H3 EAV sampling profile {sampling_profile!r}")
    object_patches = getattr(model, "object_patches", {})
    conflict_names = []
    for key, value in object_patches.items():
        if key == "extra_conds" and allowed_live_extra_conds_patch_versions:
            function = getattr(value, "__func__", value)
            version = getattr(function, "_t8_long_video_patch_version", None)
            if version in set(allowed_live_extra_conds_patch_versions):
                continue
        if key.startswith("diffusion_model.blocks.") or key in {
            "diffusion_model._forward",
            "diffusion_model.forward",
            "extra_conds",
        }:
            conflict_names.append(key)
    conflict_names.sort()
    if conflict_names:
        raise RuntimeError(
            "H3 EAV cannot stack with existing H3 object patches: "
            + ", ".join(conflict_names)
        )
    expected = {
        "attention_forward": ATTENTION_FORWARD_SHA256S,
        "packed_layout": PACKED_LAYOUT_SHA256S,
        "model_forward": MODEL_FORWARD_SHA256S,
        "patchify_video": PATCHIFY_VIDEO_SHA256S,
    }
    core_contract = {
        "source_hashes": hashes,
        "source_hash_policy": "diagnostic_only_not_a_compatibility_gate",
        "reference_source_match": {
            key: value in expected[key] for key, value in hashes.items()
        },
        "semantic_contract": semantic_contract,
    }
    return {
        "native_h3_model_observed": native_h3_model_observed,
        "model_identity_policy": "diagnostic_only_not_a_load_gate",
        "core_hashes": hashes,
        "core_contract": core_contract,
        "turbo_contract": turbo_contract,
    }


def _assert_block_cache_contract(model) -> dict:
    """Authenticate the separately installed T8 BlockCache without importing it.

    The composer calls the already-attached BlockCache diffusion wrapper instead of
    copying its cache/finalization implementation. Compatibility is admitted by the
    executable wrapper/config/replacement structure below; source hashes are retained
    only as diagnostics so comment/refactor-only changes do not break old workflows.
    """
    if not hasattr(model, "get_wrappers") or not hasattr(
        model, "remove_wrappers_with_key"
    ):
        raise ValueError("H3 EAV + BlockCache requires a current ComfyUI MODEL patcher")

    transformer = getattr(model, "model_options", {}).get("transformer_options", {})
    prototype = transformer.get(BLOCK_CACHE_KEY)
    if prototype is None:
        raise RuntimeError(
            "H3 EAV + BlockCache requires the MODEL output of MiniMaxH3BlockCacheT8"
        )
    total_blocks = int(getattr(prototype, "total_blocks", -1))
    if total_blocks != 50:
        raise RuntimeError(
            "H3 EAV + BlockCache requires the native 50-block H3 cache contract"
        )

    config = getattr(prototype, "config", None)
    required_config = (
        "residual_diff_threshold",
        "start_percent",
        "end_percent",
        "max_consecutive_hits",
        "cache_device",
        "metric_stride",
        "verbose",
    )
    if config is None or any(not hasattr(config, key) for key in required_config):
        raise RuntimeError("H3 EAV + BlockCache cache configuration is incomplete")
    config_report = {key: getattr(config, key) for key in required_config}
    threshold = float(config_report["residual_diff_threshold"])
    start_percent = float(config_report["start_percent"])
    end_percent = float(config_report["end_percent"])
    max_hits = int(config_report["max_consecutive_hits"])
    metric_stride = int(config_report["metric_stride"])
    cache_device = str(config_report["cache_device"])
    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError("H3 EAV + BlockCache residual threshold is outside 0..1")
    if not 0.0 <= start_percent < end_percent <= 1.0:
        raise RuntimeError("H3 EAV + BlockCache sampling window is invalid")
    if not 1 <= max_hits <= 10 or not 1 <= metric_stride <= 32:
        raise RuntimeError("H3 EAV + BlockCache cache limits are invalid")
    if cache_device != "cpu":
        raise RuntimeError(
            "H3 EAV + BlockCache first contract requires cache_device=cpu; GPU cache "
            "adds unaudited VRAM pressure"
        )

    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    outer_type = comfy.patcher_extension.WrappersMP.OUTER_SAMPLE
    diffusion_wrappers = list(model.get_wrappers(wrapper_type, BLOCK_CACHE_WRAPPER_KEY))
    outer_wrappers = list(model.get_wrappers(outer_type, BLOCK_CACHE_WRAPPER_KEY))
    if len(diffusion_wrappers) != 1 or len(outer_wrappers) != 1:
        raise RuntimeError(
            "H3 EAV + BlockCache requires exactly one cache diffusion wrapper and "
            "one execution-scoped outer wrapper"
        )
    wrappers = getattr(model, "wrappers", {})
    wrapper_inventory = {
        kind: {
            str(key): len(values)
            for key, values in keyed.items()
            if bool(values)
        }
        for kind, keyed in wrappers.items()
        if isinstance(keyed, Mapping)
        and any(bool(values) for values in keyed.values())
    }
    expected_inventory = {
        outer_type: {BLOCK_CACHE_WRAPPER_KEY: 1},
        wrapper_type: {BLOCK_CACHE_WRAPPER_KEY: 1},
    }
    if wrapper_inventory != expected_inventory:
        raise RuntimeError(
            "H3 EAV + BlockCache refuses additional model/sample wrappers: "
            f"observed={wrapper_inventory}"
        )

    replacements = transformer.get("patches_replace", {})
    if not isinstance(replacements, Mapping) or set(replacements) != {"dit"}:
        raise RuntimeError("H3 EAV + BlockCache replacement scope is not exact")
    dit_replacements = replacements.get("dit")
    expected_keys = {("double_block", 0), ("double_block", 49)}
    if not isinstance(dit_replacements, Mapping) or set(dit_replacements) != expected_keys:
        raise RuntimeError(
            "H3 EAV + BlockCache requires only the boundary block 0/49 replacements"
        )
    for key, patch in dit_replacements.items():
        if int(getattr(patch, "block_index", -1)) != int(key[1]):
            raise RuntimeError("H3 EAV + BlockCache boundary patch identity is invalid")

    hashes = {
        "outer_wrapper": _source_sha256(outer_wrappers[0]),
        "diffusion_wrapper": _source_sha256(diffusion_wrappers[0]),
        "cache_class": _source_sha256(type(prototype)),
        "patch_call": _source_sha256(type(next(iter(dit_replacements.values()))).__call__),
        "config_class": _source_sha256(type(config)),
    }
    expected_hashes = {
        "outer_wrapper": BLOCK_CACHE_OUTER_WRAPPER_SHA256S,
        "diffusion_wrapper": BLOCK_CACHE_DIFFUSION_WRAPPER_SHA256S,
        "cache_class": BLOCK_CACHE_CLASS_SHA256S,
        "patch_call": BLOCK_CACHE_PATCH_CALL_SHA256S,
        "config_class": BLOCK_CACHE_CONFIG_CLASS_SHA256S,
    }
    return {
        "prototype": prototype,
        "diffusion_wrapper": diffusion_wrappers[0],
        "replacements": dict(dit_replacements),
        "report": {
            "patch_version": EAV_BLOCK_CACHE_PATCH_VERSION,
            "source_hashes": hashes,
            "source_hash_policy": "diagnostic_only_not_a_compatibility_gate",
            "reference_source_match": {
                key: value in expected_hashes[key] for key, value in hashes.items()
            },
            "total_blocks": total_blocks,
            "boundary_blocks": [0, 49],
            "cache_device": cache_device,
            "config": {
                "residual_diff_threshold": threshold,
                "start_percent": start_percent,
                "end_percent": end_percent,
                "max_consecutive_hits": max_hits,
                "metric_stride": metric_stride,
                "verbose": bool(config_report["verbose"]),
            },
            "composition_order": (
                "EAV_on_each_executed_block; BlockCache_may_reuse_cached_blocks_1_to_49"
            ),
            "adds_model_forwards": False,
        },
    }


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
    long_video_contract: Mapping | None = None,
) -> dict:
    if denoise_mask is not None or audio_denoise_mask is not None:
        raise RuntimeError("H3 EAV rejects video/audio denoise masks")
    refs = list(payload.get("refs") or ())
    if refs and not allow_reference_blocks and long_video_contract is None:
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
    if long_video_contract is None:
        task = _classify_visual_task(keyframes, latent_frames=frames, refs=refs)
        expected_cond_count = None
    else:
        from .long_video import (
            CONTEXT_FRAME_STEPS,
            LONG_VIDEO_PATCH_VERSION,
            MOTION_FRAME_INDEX,
            step_offsets,
        )

        if int(payload.get("t8_long_video_patch_version", -1)) != LONG_VIDEO_PATCH_VERSION:
            raise RuntimeError("H3 EAV + Long Video requires the repaired runtime payload")
        segment_index = int(long_video_contract["segment_index"])
        context_frames = int(long_video_contract["context_frames"])
        motion_keyframes = [
            item for item in keyframes if MOTION_FRAME_INDEX in item
        ]
        if segment_index == 0:
            if context_frames != 0 or motion_keyframes:
                raise RuntimeError(
                    "H3 EAV + Long Video segment 0 cannot contain motion context"
                )
            task = "LongVideoSegment0"
        else:
            expected_steps = int(CONTEXT_FRAME_STEPS[context_frames])
            observed_offsets = [
                int(item[MOTION_FRAME_INDEX]) for item in motion_keyframes
            ]
            expected_offsets = [int(value) for value in step_offsets(expected_steps)]
            if observed_offsets != expected_offsets:
                raise RuntimeError(
                    "H3 EAV + Long Video motion-context offsets changed: "
                    f"observed={observed_offsets}, expected={expected_offsets}"
                )
            task = "LongVideoMotion"
        expected_cond_count = len(keyframes)
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
    if expected_cond_count is None:
        visual_task = _classify_visual_task(keyframes, latent_frames=frames)
        expected_cond_count = (
            0 if visual_task == "T2VA" else (2 if visual_task == "FL2VA" else 1)
        )
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
        "sm120_high_token_guard_rows": 50_000,
    }


def _strict_sage_architecture_guard(rows: int, compute_capability) -> None:
    if not isinstance(compute_capability, (tuple, list)) or len(compute_capability) < 2:
        raise RuntimeError(
            "H3 EAV + Strict Sage could not verify the CUDA compute capability"
        )
    major, minor = int(compute_capability[0]), int(compute_capability[1])
    if major >= 12 and int(rows) >= 50_000:
        raise RuntimeError(
            "H3 EAV + Strict Sage blocks compute capability "
            f"{major}.{minor} at {int(rows)} packed rows because this high-token kernel "
            "profile has a reported pure-noise output failure; use stock attention"
        )


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
    try:
        compute_capability = torch.cuda.get_device_capability(q.device)
    except Exception as exc:
        raise RuntimeError(
            "H3 EAV + Strict Sage could not inspect the CUDA architecture"
        ) from exc
    _strict_sage_architecture_guard(int(q.shape[2]), compute_capability)

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
    return _apply_eav_output_gain(q, k, output, route)


def _apply_eav_output_gain(q, k, output, route):
    """Measure FETA once for a main H3 block and scale target-video rows only."""
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


def route_eav_prompt_relay_attention(
    q,
    k,
    v,
    heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    skip_output_reshape=False,
    transformer_options=None,
    *,
    query_chunk_rows: int,
    **kwargs,
):
    """Run authenticated Prompt Relay attention, then apply target-video FETA gain."""
    from .prompt_relay_advanced import (
        PROMPT_RELAY_RUNTIME_KEY,
        route_prompt_relay_attention,
    )

    transformer_options = transformer_options or {}
    relay_route = transformer_options.get(PROMPT_RELAY_RUNTIME_KEY)
    eav_route = transformer_options.get(EAV_RUNTIME_KEY)
    relay_active = relay_route is not None and q.shape[-2] == int(relay_route["seq_len"])
    eav_active = eav_route is not None and q.shape[-2] == int(eav_route["seq_len"])
    if relay_active != eav_active:
        raise RuntimeError(
            "H3 EAV + Prompt Relay runtime routes disagree on the active packed sequence"
        )
    output = route_prompt_relay_attention(
        q,
        k,
        v,
        heads,
        mask=mask,
        attn_precision=attn_precision,
        skip_reshape=skip_reshape,
        skip_output_reshape=skip_output_reshape,
        transformer_options=transformer_options,
        query_chunk_rows=int(query_chunk_rows),
        **kwargs,
    )
    if not eav_active:
        return output
    return _apply_eav_output_gain(q, k, output, eav_route)


def build_eav_prompt_relay_model(
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
):
    """Replace one exact Relay patch with a single Relay→FETA composer owner."""
    from .prompt_relay_advanced import (
        PROMPT_RELAY_PATCH_VERSION,
        PROMPT_RELAY_PAYLOAD_KEY,
        PROMPT_RELAY_RUNTIME_KEY,
        PROMPT_RELAY_WRAPPER_KEY,
        _runtime_route as _prompt_relay_runtime_route,
        prompt_relay_model_contract,
    )

    relay = prompt_relay_model_contract(model)
    binding = dict(relay["binding"])
    task_lookup = {
        "t2va": "T2VA",
        "i2va": "I2VA",
        "fl2va": "FL2VA",
        "l2va": "L2VA",
        "ref2va": "Ref2VA",
        "hybrid": "Hybrid",
    }
    task_key = str(binding.get("task", "")).lower()
    if task_key not in task_lookup:
        raise RuntimeError(
            f"H3 EAV + Prompt Relay received unsupported bound task {task_key!r}"
        )
    task = task_lookup[task_key]
    sampling_profile = str(sampling_profile)
    reference_task = task in set(EAV_REFERENCE_TASKS)
    if reference_task and sampling_profile != "stock20":
        raise ValueError("H3 EAV + Prompt Relay reference tasks currently require stock20")
    if sampling_profile == "turbo8_alpha8" and task != "T2VA":
        raise ValueError(
            "H3 EAV + Prompt Relay turbo8_alpha8 is currently limited to audited T2VA"
        )

    clean = model.clone()
    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    clean.remove_wrappers_with_key(wrapper_type, PROMPT_RELAY_WRAPPER_KEY)
    clean.model_options["transformer_options"].pop("optimized_attention_override", None)
    if hasattr(clean, "remove_attachments"):
        clean.remove_attachments(PROMPT_RELAY_WRAPPER_KEY)

    eav_model, runtime, _report = build_eav_model(
        clean,
        sigmas,
        mode=mode,
        tau=tau,
        start_video_progress=start_video_progress,
        end_video_progress=end_video_progress,
        max_workspace_mib=max_workspace_mib,
        g_hard_limit=g_hard_limit,
        sampling_profile=sampling_profile,
        allowed_tasks=(task,),
        allow_reference_blocks=reference_task,
        composer_profile=f"prompt_relay_{task_key}_{binding['query_route']}_v1",
    )
    relay_summary = {
        "patch_version": PROMPT_RELAY_PATCH_VERSION,
        "plan_hash": str(binding["plan_hash"]),
        "binding_hash": str(binding["binding_hash"]),
        "layout_contract_hash": str(binding["layout_contract"]["contract_hash"]),
        "query_route": str(binding["query_route"]),
        "query_chunk_rows": int(relay["query_chunk_rows"]),
        "event_count": len(binding["events"]),
        "task": task,
        "composition_order": "prompt_relay_attention_then_target_video_feta_gain",
        "adds_model_forwards": False,
    }
    runtime.config["prompt_relay_contract"] = relay_summary
    runtime.config["notes"] = [
        "one combined wrapper owns both authenticated Prompt Relay and H3 FETA",
        "Relay routes local text attention first; FETA then scales only target-video output rows",
        "the composer adds no model forwards; the runtime audit still requires the exact schedule NFE and 50 H3 main blocks per active forward",
        "joint_av_exp may directly bias target-audio queries, while FETA never directly scales audio rows; full audio review remains required",
        "BlockCache, STG, Long Video, Sage and unknown wrappers remain rejected",
    ]
    if str(mode) == "disabled":
        return model, runtime, _json(runtime.config)

    eav_model.remove_wrappers_with_key(wrapper_type, EAV_WRAPPER_KEY)
    eav_model.model_options["transformer_options"].pop(
        "optimized_attention_override", None
    )
    if hasattr(eav_model, "remove_attachments"):
        eav_model.remove_attachments(EAV_WRAPPER_KEY)
    expected_hash = str(binding["binding_hash"])

    def _combined_wrapper(
        executor,
        x,
        timestep,
        context,
        transformer_options=None,
        **kwargs,
    ):
        transformer_options = transformer_options if transformer_options is not None else {}
        if len(executor.wrappers) != 1:
            raise RuntimeError(
                "H3 EAV + Prompt Relay detected another diffusion wrapper after binding"
            )
        installed = transformer_options.get("optimized_attention_override")
        if (
            getattr(installed, "_t8_h3_eav_prompt_relay_patch_version", None)
            != EAV_PROMPT_RELAY_PATCH_VERSION
            or getattr(installed, "_t8_prompt_relay_binding_hash", None)
            != expected_hash
        ):
            raise RuntimeError("H3 EAV + Prompt Relay attention owner was replaced")
        replacements = transformer_options.get("patches_replace", {})
        if isinstance(replacements, Mapping) and any(
            bool(value) for value in replacements.values()
        ):
            raise RuntimeError("H3 EAV + Prompt Relay refuses runtime block replacements")
        supplied_hash = kwargs.pop(PROMPT_RELAY_PAYLOAD_KEY, None)
        if supplied_hash != expected_hash:
            raise RuntimeError(
                "H3 EAV + Prompt Relay MODEL and CONDITIONING binding hashes differ"
            )
        payload = kwargs.get("minimax_payload")
        if not isinstance(payload, Mapping):
            raise RuntimeError("H3 EAV + Prompt Relay could not find minimax_payload")
        if EAV_RUNTIME_KEY in transformer_options or PROMPT_RELAY_RUNTIME_KEY in transformer_options:
            raise RuntimeError("Nested H3 EAV + Prompt Relay runtime state was refused")
        try:
            relay_route = _prompt_relay_runtime_route(
                payload.get("layout"), binding, x[0].device
            )
            eav_route = _runtime_route(
                x=x,
                timestep=timestep,
                context=context,
                payload=payload,
                denoise_mask=kwargs.get("denoise_mask"),
                audio_denoise_mask=kwargs.get("audio_denoise_mask"),
                start_progress=float(start_video_progress),
                end_progress=float(end_video_progress),
                allowed_tasks=(task,),
                allow_reference_blocks=reference_task,
            )
            if str(eav_route["task"]).lower() != task_key:
                raise RuntimeError(
                    "H3 EAV runtime task does not match the authenticated Prompt Relay binding"
                )
            forward_index = runtime.begin_forward(
                sigma_video=eav_route["sigma_video"],
                progress_video=eav_route["progress_video"],
                route=eav_route,
            )
            eav_route.update(
                {
                    "mode": str(mode),
                    "tau": float(tau),
                    "max_workspace_mib": int(max_workspace_mib),
                    "g_hard_limit": float(g_hard_limit),
                    "runtime": runtime,
                    "forward_index": forward_index,
                    "attention_backend": "native_optimized",
                }
            )
            transformer_options[PROMPT_RELAY_RUNTIME_KEY] = relay_route
            transformer_options[EAV_RUNTIME_KEY] = eav_route
            return executor(x, timestep, context, transformer_options, **kwargs)
        except BaseException as exc:
            runtime.abort(exc)
            raise
        finally:
            transformer_options.pop(PROMPT_RELAY_RUNTIME_KEY, None)
            transformer_options.pop(EAV_RUNTIME_KEY, None)

    def _combined_attention(*args, **kwargs):
        return route_eav_prompt_relay_attention(
            *args,
            query_chunk_rows=int(relay["query_chunk_rows"]),
            **kwargs,
        )

    eav_model.add_wrapper_with_key(
        wrapper_type,
        EAV_PROMPT_RELAY_WRAPPER_KEY,
        _combined_wrapper,
    )
    eav_model.set_model_optimized_attention(_combined_attention)
    installed = eav_model.model_options["transformer_options"][
        "optimized_attention_override"
    ]
    installed._t8_h3_eav_patch_version = EAV_PATCH_VERSION
    installed._t8_prompt_relay_binding_hash = expected_hash
    installed._t8_h3_eav_prompt_relay_patch_version = EAV_PROMPT_RELAY_PATCH_VERSION
    if hasattr(eav_model, "set_attachments"):
        eav_model.set_attachments(
            EAV_PROMPT_RELAY_WRAPPER_KEY,
            {
                "patch_version": EAV_PROMPT_RELAY_PATCH_VERSION,
                "eav_config": dict(runtime.config),
                "prompt_relay": relay_summary,
            },
        )
    return eav_model, runtime, _json(runtime.config)


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
    stg_contract: Mapping | None = None,
    long_video_contract: Mapping | None = None,
    allowed_live_extra_conds_patch_versions: tuple[int, ...] = (),
    wrapper_key: str = EAV_WRAPPER_KEY,
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
    valid_tasks = (
        {"LongVideoSegment0", "LongVideoMotion"}
        if long_video_contract is not None
        else set(EAV_ALL_TASKS)
    )
    if not allowed_tasks or not set(allowed_tasks).issubset(valid_tasks):
        raise ValueError(f"H3 EAV received an invalid task scope: {allowed_tasks!r}")
    if (
        long_video_contract is None
        and allow_reference_blocks
        and not set(allowed_tasks).issubset(set(EAV_REFERENCE_TASKS))
    ):
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
        "stg_contract": dict(stg_contract) if stg_contract is not None else None,
        "long_video_contract": (
            dict(long_video_contract) if long_video_contract is not None else None
        ),
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

    contracts = _assert_core_contract(
        model,
        sampling_profile=sampling_profile,
        allowed_live_extra_conds_patch_versions=tuple(
            int(value) for value in allowed_live_extra_conds_patch_versions
        ),
    )
    config["core_hashes"] = contracts["core_hashes"]
    config["core_contract"] = contracts["core_contract"]
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
        "Turbo8 LoRA hook count and strength are diagnostic only; user-selected model stacks are not rejected",
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
        branch = "main"
        skipped_blocks = []
        if stg_contract is None:
            if EAV_STG_BRANCH_KEY in transformer_options:
                raise RuntimeError("H3 EAV found an unauthenticated STG branch marker")
            if isinstance(replacements, Mapping) and any(
                bool(v) for v in replacements.values()
            ):
                raise RuntimeError("H3 EAV detected a runtime block replacement and refused it")
        else:
            marker = transformer_options.get(EAV_STG_BRANCH_KEY)
            if marker is None:
                if isinstance(replacements, Mapping) and any(
                    bool(v) for v in replacements.values()
                ):
                    raise RuntimeError(
                        "H3 EAV + STG main branch received a block replacement"
                    )
            else:
                expected_marker = stg_contract["weak_branch_marker"]
                if not isinstance(marker, Mapping) or dict(marker) != dict(
                    expected_marker
                ):
                    raise RuntimeError("H3 EAV + STG weak branch marker was invalid")
                skipped_blocks = [
                    int(value) for value in stg_contract["double_blocks"]
                ]
                dit = (
                    replacements.get("dit", {})
                    if isinstance(replacements, Mapping)
                    else {}
                )
                expected_keys = {
                    ("double_block", int(value)) for value in skipped_blocks
                }
                if set(replacements) != {"dit"} or set(dit) != expected_keys:
                    raise RuntimeError(
                        "H3 EAV + STG weak branch block replacements changed"
                    )
                for patch in dit.values():
                    if (
                        getattr(patch, "_t8_h3_stg_patch_version", None)
                        != EAV_STG_PATCH_VERSION
                        or getattr(patch, "_t8_h3_stg_binding_hash", None)
                        != stg_contract["binding_hash"]
                    ):
                        raise RuntimeError(
                            "H3 EAV + STG weak branch skip patch was not authenticated"
                        )
                branch = "stg_weak"
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
                long_video_contract=long_video_contract,
            )
            forward_index = runtime.begin_forward(
                sigma_video=route["sigma_video"],
                progress_video=route["progress_video"],
                route=route,
                branch=branch,
                skipped_blocks=skipped_blocks,
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
        str(wrapper_key),
        _diffusion_wrapper,
    )
    patched.set_model_optimized_attention(route_eav_attention)
    installed = patched.model_options["transformer_options"]["optimized_attention_override"]
    installed._t8_h3_eav_patch_version = EAV_PATCH_VERSION
    if hasattr(patched, "set_attachments"):
        patched.set_attachments(str(wrapper_key), dict(config))
    return patched, runtime, _json(config)


def build_eav_stg_model(
    model,
    sigmas: torch.Tensor,
    *,
    mode: str,
    tau: float,
    start_video_progress: float,
    end_video_progress: float,
    max_workspace_mib: int,
    g_hard_limit: float,
    stg_scale: float,
    stg_double_blocks: str,
    stg_start_progress: float,
    stg_end_progress: float,
    shift_video: float,
    rescale: float,
):
    """Compose FETA with the project's H3 skip-block STG as one audited owner."""
    _assert_no_sampler_guidance_hooks(model, owner="H3 EAV + STG")
    contracts = _assert_core_contract(model, sampling_profile="stock20")
    blocks = _parse_h3_blocks(stg_double_blocks)
    scale = float(stg_scale)
    start = float(stg_start_progress)
    end = float(stg_end_progress)
    shift = float(shift_video)
    if not math.isfinite(scale) or not 0.0 <= scale <= 5.0:
        raise ValueError("H3 EAV + STG scale must be finite and between 0 and 5")
    if not 0.0 <= start < end <= 1.0:
        raise ValueError("H3 EAV + STG progress must satisfy 0 <= start < end <= 1")
    if not math.isfinite(shift) or shift <= 0.0:
        raise ValueError("H3 EAV + STG shift_video must be finite and positive")
    if not math.isfinite(float(rescale)) or float(rescale) != 0.0:
        raise ValueError("H3 EAV + STG requires rescale=0")

    schedule = torch.as_tensor(sigmas).detach().float().cpu().flatten()
    if schedule.numel() < 2:
        raise ValueError("H3 EAV + STG requires a non-empty sigma schedule")
    call_sigmas = schedule[:-1]
    denominator = shift + call_sigmas * (1.0 - shift)
    if not bool((denominator > 0).all()):
        raise ValueError("H3 EAV + STG shift produced an invalid progress denominator")
    progress = 1.0 - call_sigmas / denominator
    weak_mask = (progress >= start) & (progress <= end)
    if scale <= 0.0:
        weak_mask = torch.zeros_like(weak_mask, dtype=torch.bool)
    weak_flags = weak_mask.tolist()
    expected_branches = []
    for weak in weak_flags:
        expected_branches.append("main")
        if bool(weak):
            expected_branches.append("stg_weak")
    contract = {
        "schema": EAV_STG_PATCH_VERSION,
        "applied": bool(scale > 0.0),
        "scale": scale,
        "double_blocks": blocks,
        "start_progress": start,
        "end_progress": end,
        "shift_video": shift,
        "rescale": float(rescale),
        "base_nfe": int(call_sigmas.numel()),
        "expected_weak_forwards": int(sum(bool(value) for value in weak_flags)),
        "expected_total_forwards": len(expected_branches),
        "expected_branches": expected_branches,
        "feta_on_main_and_weak": True,
        "weak_feta_measurements_when_active": 50 - len(blocks),
    }
    binding_payload = json.dumps(
        contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    contract["binding_hash"] = hashlib.sha256(binding_payload).hexdigest()
    marker = {
        "schema": EAV_STG_PATCH_VERSION,
        "binding_hash": contract["binding_hash"],
        "branch": "stg_weak",
        "skipped_double_blocks": blocks,
    }
    contract["weak_branch_marker"] = marker

    patched, runtime, _report_json = build_eav_model(
        model,
        sigmas,
        mode=mode,
        tau=tau,
        start_video_progress=start_video_progress,
        end_video_progress=end_video_progress,
        max_workspace_mib=max_workspace_mib,
        g_hard_limit=g_hard_limit,
        sampling_profile="stock20",
        allowed_tasks=EAV_VISUAL_TASKS,
        allow_reference_blocks=False,
        composer_profile="stg_visual_stock20_v1",
        attention_backend="native_optimized",
        stg_contract=contract,
        wrapper_key=EAV_STG_WRAPPER_KEY,
    )
    runtime.config["core_hashes"] = contracts["core_hashes"]
    runtime.config["core_contract"] = contracts["core_contract"]
    runtime.config["turbo_contract"] = None
    runtime.config["stg_contract"] = dict(contract)
    runtime.config["notes"] = [
        "one composer owns EAV routing plus the single H3 STG post-CFG hook",
        "EAV runs on both the main conditional forward and the STG weak forward so the guidance difference is not confounded by enabling FETA on only one branch",
        f"main forwards execute 50 measured blocks; active STG weak forwards skip {blocks} and therefore execute {50 - len(blocks)} measured blocks",
        "mode=disabled disables only EAV and keeps STG as the explicit comparison baseline",
        "STG adds one shared audio-video Transformer forward in each configured active step and can change sound as well as picture",
        "quality, audio non-inferiority, performance and 16GiB safety are not assumed",
    ]
    stg_model, stg_report_json = apply_h3_spatiotemporal_guidance(
        patched,
        scale=scale,
        double_blocks=",".join(str(value) for value in blocks),
        start_progress=start,
        end_progress=end,
        shift_video=shift,
        rescale=float(rescale),
        weak_branch_marker=(EAV_STG_BRANCH_KEY, marker),
    )
    runtime.config["stg_node_report"] = json.loads(stg_report_json)
    if stg_model is not model and hasattr(stg_model, "set_attachments"):
        stg_model.set_attachments(
            EAV_STG_WRAPPER_KEY,
            {
                "patch_version": EAV_STG_PATCH_VERSION,
                "eav_config": dict(runtime.config),
                "stg_contract": dict(contract),
            },
        )
    return stg_model, runtime, _json(runtime.config)


def _assert_long_video_contract(model, *, segment_index: int, context_frames: int) -> dict:
    from .long_video import CONTEXT_FRAME_STEPS, LONG_VIDEO_PATCH_VERSION

    segment_index = int(segment_index)
    context_frames = int(context_frames)
    if segment_index < 0:
        raise ValueError("H3 EAV + Long Video segment_index cannot be negative")
    if segment_index == 0:
        if context_frames != 0:
            raise ValueError("H3 EAV + Long Video segment 0 requires context_frames=0")
    elif context_frames not in CONTEXT_FRAME_STEPS:
        raise ValueError(
            "H3 EAV + Long Video continuation context_frames must be 5, 22, or 39"
        )
    object_patches = getattr(model, "object_patches", {})
    patched_extra_conds = object_patches.get("extra_conds")
    if patched_extra_conds is None:
        raise RuntimeError(
            "H3 EAV + Long Video requires the MODEL output of Long Video Conditioning"
        )
    function = getattr(patched_extra_conds, "__func__", patched_extra_conds)
    version = getattr(function, "_t8_long_video_patch_version", None)
    original = getattr(function, "_t8_long_video_original_extra_conds", None)
    if version != LONG_VIDEO_PATCH_VERSION or not callable(original):
        raise RuntimeError("H3 EAV + Long Video extra_conds patch is not authentic")
    contract = {
        "schema": EAV_LONG_VIDEO_PATCH_VERSION,
        "long_video_patch_version": int(version),
        "segment_index": segment_index,
        "context_frames": context_frames,
        "expected_motion_latent_steps": (
            0 if segment_index == 0 else int(CONTEXT_FRAME_STEPS[context_frames])
        ),
        "extra_conds_patch_sha256": _source_sha256(function),
        "resume_scope": "execution_local_eav_runtime_per_segment",
    }
    binding = json.dumps(
        contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    contract["binding_hash"] = hashlib.sha256(binding).hexdigest()
    return contract


def build_eav_long_video_model(
    model,
    sigmas: torch.Tensor,
    *,
    segment_index: int,
    context_frames: int,
    mode: str,
    tau: float,
    start_video_progress: float,
    end_video_progress: float,
    max_workspace_mib: int,
    g_hard_limit: float,
):
    """Compose scoped Long Video layout repair with one per-segment FETA owner."""
    from .long_video import LONG_VIDEO_PATCH_VERSION

    contract = _assert_long_video_contract(
        model,
        segment_index=segment_index,
        context_frames=context_frames,
    )
    core = _assert_core_contract(
        model,
        sampling_profile="stock20",
        allowed_live_extra_conds_patch_versions=(LONG_VIDEO_PATCH_VERSION,),
    )
    patched, runtime, _report_json = build_eav_model(
        model,
        sigmas,
        mode=mode,
        tau=tau,
        start_video_progress=start_video_progress,
        end_video_progress=end_video_progress,
        max_workspace_mib=max_workspace_mib,
        g_hard_limit=g_hard_limit,
        sampling_profile="stock20",
        allowed_tasks=("LongVideoSegment0", "LongVideoMotion"),
        allow_reference_blocks=True,
        composer_profile="long_video_segment_stock20_v1",
        attention_backend="native_optimized",
        long_video_contract=contract,
        allowed_live_extra_conds_patch_versions=(LONG_VIDEO_PATCH_VERSION,),
        wrapper_key=EAV_LONG_VIDEO_WRAPPER_KEY,
    )
    runtime.config["core_hashes"] = core["core_hashes"]
    runtime.config["core_contract"] = core["core_contract"]
    runtime.config["long_video_contract"] = dict(contract)
    runtime.config["notes"] = [
        "Long Video Conditioning remains the only extra_conds/layout owner; EAV owns only its per-segment diffusion and attention route",
        "segment_index and context_frames force a fresh execution-local EAV runtime, so resume never reuses a consumed audit token from an earlier segment",
        "segment 0 requires no motion context; continuation segments require the exact 5/22/39-frame motion-keyframe offsets from the immediately preceding accepted context",
        "each Stock20 segment is audited independently for 20 model forwards and 50 active block measurements per forward",
        "quality, seam continuity, audio non-inferiority, performance and 16GiB safety are not assumed",
    ]
    if hasattr(patched, "set_attachments") and patched is not model:
        patched.set_attachments(
            EAV_LONG_VIDEO_WRAPPER_KEY,
            {
                "patch_version": EAV_LONG_VIDEO_PATCH_VERSION,
                "eav_config": dict(runtime.config),
                "long_video_contract": dict(contract),
            },
        )
    return patched, runtime, _json(runtime.config)


def build_eav_prompt_relay_long_video_model(
    model,
    sigmas: torch.Tensor,
    *,
    segment_index: int,
    context_frames: int,
    mode: str,
    tau: float,
    start_video_progress: float,
    end_video_progress: float,
    max_workspace_mib: int,
    g_hard_limit: float,
):
    """Compose Prompt Relay, scoped Long Video and FETA under one attention owner.

    Long Video remains the only ``extra_conds``/packed-layout owner. Prompt Relay
    supplies the segment-projected text route, then FETA measures and scales only
    target-video output rows. A fresh EAV runtime is created for every segment so
    an interrupted run can never reuse an already-consumed audit token.
    """
    from .long_video import LONG_VIDEO_PATCH_VERSION
    from .prompt_relay_advanced import (
        PROMPT_RELAY_PATCH_VERSION,
        PROMPT_RELAY_PAYLOAD_KEY,
        PROMPT_RELAY_RUNTIME_KEY,
        PROMPT_RELAY_WRAPPER_KEY,
        _runtime_route as _prompt_relay_runtime_route,
        prompt_relay_model_contract,
    )
    from .prompt_relay_long_video_advanced import (
        PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY,
        PROMPT_RELAY_LONG_VIDEO_PROJECTION_SCHEMA,
    )

    relay = prompt_relay_model_contract(model)
    binding = dict(relay["binding"])
    long_video_contract = _assert_long_video_contract(
        model,
        segment_index=segment_index,
        context_frames=context_frames,
    )
    projection = (
        model.get_attachment(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY)
        if hasattr(model, "get_attachment")
        else None
    )
    if not isinstance(projection, Mapping):
        raise RuntimeError(
            "H3 EAV + Prompt Relay + Long Video requires the applied projected "
            "Long Video Prompt Relay MODEL"
        )
    expected_projection = {
        "schema": PROMPT_RELAY_LONG_VIDEO_PROJECTION_SCHEMA,
        "binding_hash": str(binding["binding_hash"]),
        "segment_index": int(segment_index),
    }
    mismatches = [
        key
        for key, value in expected_projection.items()
        if projection.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "H3 EAV + Prompt Relay + Long Video projection binding changed: "
            + ", ".join(mismatches)
        )

    clean = model.clone()
    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    clean.remove_wrappers_with_key(wrapper_type, PROMPT_RELAY_WRAPPER_KEY)
    clean.model_options["transformer_options"].pop("optimized_attention_override", None)
    if hasattr(clean, "remove_attachments"):
        clean.remove_attachments(PROMPT_RELAY_WRAPPER_KEY)

    eav_model, runtime, _report = build_eav_model(
        clean,
        sigmas,
        mode=mode,
        tau=tau,
        start_video_progress=start_video_progress,
        end_video_progress=end_video_progress,
        max_workspace_mib=max_workspace_mib,
        g_hard_limit=g_hard_limit,
        sampling_profile="stock20",
        allowed_tasks=("LongVideoSegment0", "LongVideoMotion"),
        allow_reference_blocks=True,
        composer_profile="prompt_relay_long_video_segment_stock20_v1",
        attention_backend="native_optimized",
        long_video_contract=long_video_contract,
        allowed_live_extra_conds_patch_versions=(LONG_VIDEO_PATCH_VERSION,),
        wrapper_key=EAV_PROMPT_RELAY_LONG_VIDEO_WRAPPER_KEY,
    )
    relay_summary = {
        "patch_version": PROMPT_RELAY_PATCH_VERSION,
        "global_plan_hash": str(projection.get("global_plan_hash", "")),
        "projected_plan_hash": str(projection.get("projected_plan_hash", "")),
        "binding_hash": str(binding["binding_hash"]),
        "layout_contract_hash": str(binding["layout_contract"]["contract_hash"]),
        "query_route": str(binding["query_route"]),
        "query_chunk_rows": int(relay["query_chunk_rows"]),
        "event_count": len(binding["events"]),
        "segment_index": int(segment_index),
        "composition_order": (
            "long_video_layout_then_prompt_relay_attention_then_target_video_feta_gain"
        ),
        "adds_model_forwards": False,
    }
    runtime.config["prompt_relay_contract"] = relay_summary
    runtime.config["long_video_contract"] = dict(long_video_contract)
    runtime.config["notes"] = [
        "Long Video is the sole extra_conds and packed-layout owner",
        "one combined wrapper applies projected Prompt Relay first and target-video FETA second",
        "the composer adds no model forwards and creates one fresh EAV audit runtime per segment",
        "Stock20 is the first admitted combined sampling contract",
        "audio is never directly scaled by FETA, but joint-AV listening remains required",
    ]
    if str(mode) == "disabled":
        return model, runtime, _json(runtime.config)

    eav_model.remove_wrappers_with_key(
        wrapper_type, EAV_PROMPT_RELAY_LONG_VIDEO_WRAPPER_KEY
    )
    eav_model.model_options["transformer_options"].pop(
        "optimized_attention_override", None
    )
    if hasattr(eav_model, "remove_attachments"):
        eav_model.remove_attachments(EAV_PROMPT_RELAY_LONG_VIDEO_WRAPPER_KEY)
    expected_hash = str(binding["binding_hash"])
    expected_task = (
        "LongVideoSegment0" if int(segment_index) == 0 else "LongVideoMotion"
    )

    def _combined_wrapper(
        executor,
        x,
        timestep,
        context,
        transformer_options=None,
        **kwargs,
    ):
        transformer_options = transformer_options if transformer_options is not None else {}
        if len(executor.wrappers) != 1:
            raise RuntimeError(
                "H3 EAV + Prompt Relay + Long Video detected another diffusion "
                "wrapper after binding"
            )
        installed = transformer_options.get("optimized_attention_override")
        if (
            getattr(installed, "_t8_h3_eav_prompt_relay_long_video_patch_version", None)
            != EAV_PROMPT_RELAY_LONG_VIDEO_PATCH_VERSION
            or getattr(installed, "_t8_prompt_relay_binding_hash", None)
            != expected_hash
        ):
            raise RuntimeError(
                "H3 EAV + Prompt Relay + Long Video attention owner was replaced"
            )
        replacements = transformer_options.get("patches_replace", {})
        if isinstance(replacements, Mapping) and any(
            bool(value) for value in replacements.values()
        ):
            raise RuntimeError(
                "H3 EAV + Prompt Relay + Long Video refuses runtime block replacements"
            )
        supplied_hash = kwargs.pop(PROMPT_RELAY_PAYLOAD_KEY, None)
        if supplied_hash != expected_hash:
            raise RuntimeError(
                "H3 EAV + Prompt Relay + Long Video MODEL and CONDITIONING hashes differ"
            )
        payload = kwargs.get("minimax_payload")
        if not isinstance(payload, Mapping):
            raise RuntimeError(
                "H3 EAV + Prompt Relay + Long Video could not find minimax_payload"
            )
        if (
            EAV_RUNTIME_KEY in transformer_options
            or PROMPT_RELAY_RUNTIME_KEY in transformer_options
        ):
            raise RuntimeError(
                "Nested H3 EAV + Prompt Relay + Long Video runtime state was refused"
            )
        try:
            relay_route = _prompt_relay_runtime_route(
                payload.get("layout"), binding, x[0].device
            )
            eav_route = _runtime_route(
                x=x,
                timestep=timestep,
                context=context,
                payload=payload,
                denoise_mask=kwargs.get("denoise_mask"),
                audio_denoise_mask=kwargs.get("audio_denoise_mask"),
                start_progress=float(start_video_progress),
                end_progress=float(end_video_progress),
                allowed_tasks=("LongVideoSegment0", "LongVideoMotion"),
                allow_reference_blocks=True,
                long_video_contract=long_video_contract,
            )
            if str(eav_route["task"]) != expected_task:
                raise RuntimeError(
                    "H3 EAV + Prompt Relay + Long Video segment task drifted"
                )
            forward_index = runtime.begin_forward(
                sigma_video=eav_route["sigma_video"],
                progress_video=eav_route["progress_video"],
                route=eav_route,
            )
            eav_route.update(
                {
                    "mode": str(mode),
                    "tau": float(tau),
                    "max_workspace_mib": int(max_workspace_mib),
                    "g_hard_limit": float(g_hard_limit),
                    "runtime": runtime,
                    "forward_index": forward_index,
                    "attention_backend": "native_optimized",
                }
            )
            transformer_options[PROMPT_RELAY_RUNTIME_KEY] = relay_route
            transformer_options[EAV_RUNTIME_KEY] = eav_route
            return executor(x, timestep, context, transformer_options, **kwargs)
        except BaseException as exc:
            runtime.abort(exc)
            raise
        finally:
            transformer_options.pop(PROMPT_RELAY_RUNTIME_KEY, None)
            transformer_options.pop(EAV_RUNTIME_KEY, None)

    def _combined_attention(*args, **kwargs):
        return route_eav_prompt_relay_attention(
            *args,
            query_chunk_rows=int(relay["query_chunk_rows"]),
            **kwargs,
        )

    eav_model.add_wrapper_with_key(
        wrapper_type,
        EAV_PROMPT_RELAY_LONG_VIDEO_WRAPPER_KEY,
        _combined_wrapper,
    )
    eav_model.set_model_optimized_attention(_combined_attention)
    installed = eav_model.model_options["transformer_options"][
        "optimized_attention_override"
    ]
    installed._t8_h3_eav_patch_version = EAV_PATCH_VERSION
    installed._t8_prompt_relay_binding_hash = expected_hash
    installed._t8_h3_eav_prompt_relay_long_video_patch_version = (
        EAV_PROMPT_RELAY_LONG_VIDEO_PATCH_VERSION
    )
    if hasattr(eav_model, "set_attachments"):
        eav_model.set_attachments(
            EAV_PROMPT_RELAY_LONG_VIDEO_WRAPPER_KEY,
            {
                "patch_version": EAV_PROMPT_RELAY_LONG_VIDEO_PATCH_VERSION,
                "eav_config": dict(runtime.config),
                "prompt_relay": relay_summary,
                "long_video_contract": dict(long_video_contract),
            },
        )
    return eav_model, runtime, _json(runtime.config)


def build_eav_block_cache_model(
    model,
    sigmas: torch.Tensor,
    *,
    mode: str,
    tau: float,
    start_video_progress: float,
    end_video_progress: float,
    max_workspace_mib: int,
    g_hard_limit: float,
):
    """Compose one authenticated CPU BlockCache owner with target-video FETA.

    The cache's own outer-sample wrapper remains authoritative for creating and
    releasing execution-local cache state. Its diffusion wrapper is called from
    the combined owner so cache-hit H3 finalization is not duplicated here.
    """
    cache_contract = _assert_block_cache_contract(model)
    clean = model.clone()
    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    clean.remove_wrappers_with_key(wrapper_type, BLOCK_CACHE_WRAPPER_KEY)

    transformer = clean.model_options.get("transformer_options", {}).copy()
    transformer.pop(BLOCK_CACHE_KEY, None)
    replacements = transformer.get("patches_replace", {})
    replacements = dict(replacements) if isinstance(replacements, Mapping) else {}
    replacements.pop("dit", None)
    if replacements:
        transformer["patches_replace"] = replacements
    else:
        transformer.pop("patches_replace", None)
    clean.model_options["transformer_options"] = transformer

    patched, runtime, _report_json = build_eav_model(
        clean,
        sigmas,
        mode=mode,
        tau=tau,
        start_video_progress=start_video_progress,
        end_video_progress=end_video_progress,
        max_workspace_mib=max_workspace_mib,
        g_hard_limit=g_hard_limit,
        sampling_profile="stock20",
        allowed_tasks=EAV_VISUAL_TASKS,
        allow_reference_blocks=False,
        composer_profile="block_cache_visual_stock20_v1",
        attention_backend="native_optimized",
    )
    runtime.config["block_cache_contract"] = dict(cache_contract["report"])
    runtime.config["notes"] = [
        "the installed T8 CPU BlockCache outer wrapper still owns execution-scoped cache allocation and release",
        "one combined diffusion wrapper runs EAV on every block that actually executes and delegates cache-hit finalization to the authenticated BlockCache wrapper",
        "a cache miss must record 50 active FETA measurements; a cache hit must record only block 0, because blocks 1-49 are reused rather than executed",
        "the first combined contract is native Stock20 visual tasks only; reference tasks, Turbo8, GPU cache, Prompt Relay, Sage, STG and Long Video remain rejected",
        "joint H3 layers can still change audio indirectly; cache speed, visual quality, audio non-inferiority and 16GiB safety are not assumed",
    ]
    if str(mode) == "disabled":
        return model, runtime, _json(runtime.config)

    patched.remove_wrappers_with_key(wrapper_type, EAV_WRAPPER_KEY)
    patched_transformer = patched.model_options.get("transformer_options", {}).copy()
    patched_transformer[BLOCK_CACHE_KEY] = cache_contract["prototype"]
    patched.model_options["transformer_options"] = patched_transformer
    for (block_name, block_index), replacement in cache_contract[
        "replacements"
    ].items():
        patched.set_model_patch_replace(
            replacement,
            "dit",
            block_name,
            int(block_index),
        )

    installed = patched.model_options["transformer_options"].get(
        "optimized_attention_override"
    )
    if getattr(installed, "_t8_h3_eav_patch_version", None) != EAV_PATCH_VERSION:
        raise RuntimeError("H3 EAV + BlockCache lost the EAV attention owner while composing")
    installed._t8_h3_eav_block_cache_patch_version = EAV_BLOCK_CACHE_PATCH_VERSION
    block_cache_diffusion_wrapper = cache_contract["diffusion_wrapper"]
    prototype = cache_contract["prototype"]

    def _combined_wrapper(
        executor,
        x,
        timestep,
        context,
        transformer_options=None,
        **kwargs,
    ):
        transformer_options = transformer_options if transformer_options is not None else {}
        if len(executor.wrappers) != 1:
            raise RuntimeError(
                "H3 EAV + BlockCache detected another diffusion wrapper after binding"
            )
        active_attention = transformer_options.get("optimized_attention_override")
        if (
            getattr(active_attention, "_t8_h3_eav_patch_version", None)
            != EAV_PATCH_VERSION
            or getattr(
                active_attention, "_t8_h3_eav_block_cache_patch_version", None
            )
            != EAV_BLOCK_CACHE_PATCH_VERSION
        ):
            raise RuntimeError("H3 EAV + BlockCache attention owner was replaced")
        runtime_cache = transformer_options.get(BLOCK_CACHE_KEY)
        if runtime_cache is None or runtime_cache is prototype:
            raise RuntimeError(
                "H3 EAV + BlockCache requires its execution-scoped outer sample wrapper"
            )
        runtime_config = getattr(runtime_cache, "config", None)
        if (
            int(getattr(runtime_cache, "total_blocks", -1)) != 50
            or runtime_config is None
            or str(getattr(runtime_config, "cache_device", "")) != "cpu"
        ):
            raise RuntimeError("H3 EAV + BlockCache runtime cache contract drifted")
        replacements = transformer_options.get("patches_replace", {})
        dit_replacements = (
            replacements.get("dit", {}) if isinstance(replacements, Mapping) else {}
        )
        if set(dit_replacements) != {("double_block", 0), ("double_block", 49)}:
            raise RuntimeError("H3 EAV + BlockCache runtime boundary patches changed")

        payload = kwargs.get("minimax_payload")
        if not isinstance(payload, Mapping):
            raise RuntimeError("H3 EAV + BlockCache could not find minimax_payload")
        if EAV_RUNTIME_KEY in transformer_options:
            raise RuntimeError("Nested H3 EAV + BlockCache runtime state was refused")
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
                allowed_tasks=EAV_VISUAL_TASKS,
                allow_reference_blocks=False,
            )
            forward_index = runtime.begin_forward(
                sigma_video=route["sigma_video"],
                progress_video=route["progress_video"],
                route=route,
            )
            route.update(
                {
                    "mode": str(mode),
                    "tau": float(tau),
                    "max_workspace_mib": int(max_workspace_mib),
                    "g_hard_limit": float(g_hard_limit),
                    "runtime": runtime,
                    "forward_index": forward_index,
                    "attention_backend": "native_optimized",
                }
            )
            transformer_options[EAV_RUNTIME_KEY] = route
            before = (
                int(getattr(runtime_cache, "total_forwards", -1)),
                int(getattr(runtime_cache, "full_forwards", -1)),
                int(getattr(runtime_cache, "cache_hits", -1)),
            )
            result = block_cache_diffusion_wrapper(
                executor,
                x,
                timestep,
                context,
                transformer_options,
                **kwargs,
            )
            after = (
                int(getattr(runtime_cache, "total_forwards", -1)),
                int(getattr(runtime_cache, "full_forwards", -1)),
                int(getattr(runtime_cache, "cache_hits", -1)),
            )
            delta = tuple(end - start for start, end in zip(before, after, strict=True))
            if delta == (1, 1, 0):
                decision = "full"
            elif delta == (1, 0, 1):
                decision = "hit"
            else:
                raise RuntimeError(
                    "H3 EAV + BlockCache observed an unauditable cache transition: "
                    f"before={before}, after={after}"
                )
            runtime.record_block_cache_decision(forward_index, decision)
            return result
        except BaseException as exc:
            runtime.abort(exc)
            raise
        finally:
            transformer_options.pop(EAV_RUNTIME_KEY, None)

    patched.add_wrapper_with_key(
        wrapper_type,
        EAV_BLOCK_CACHE_WRAPPER_KEY,
        _combined_wrapper,
    )
    if hasattr(patched, "remove_attachments"):
        patched.remove_attachments(EAV_WRAPPER_KEY)
    if hasattr(patched, "set_attachments"):
        patched.set_attachments(
            EAV_BLOCK_CACHE_WRAPPER_KEY,
            {
                "patch_version": EAV_BLOCK_CACHE_PATCH_VERSION,
                "eav_config": dict(runtime.config),
                "block_cache": dict(cache_contract["report"]),
            },
        )
    return patched, runtime, _json(runtime.config)


def finalize_eav_runtime(av_latent, runtime: EAVRuntime):
    if not isinstance(runtime, EAVRuntime):
        raise TypeError("H3 EAV Audit requires the runtime token from the matching EAV node")
    report = runtime.snapshot(consume=True)
    mode = report["config"]["mode"]
    if report["aborted"]:
        raise RuntimeError("H3 EAV sampling aborted: " + report["aborted"])
    stg_contract = report["config"].get("stg_contract")
    if mode == "disabled":
        if stg_contract is not None and bool(stg_contract.get("applied")):
            report["status"] = "eav_disabled_stg_active"
        elif report["config"].get("long_video_contract") is not None:
            report["status"] = "eav_disabled_long_video_passthrough"
        else:
            report["status"] = "disabled_identity"
    else:
        expected_nfe = int(report["config"]["sigma_contract"]["nfe"])
        expected_total_forwards = expected_nfe
        if stg_contract is not None:
            expected_total_forwards = int(stg_contract["expected_total_forwards"])
        if report["model_forward_count"] != expected_total_forwards:
            raise RuntimeError(
                f"H3 EAV {report['config']['sampling_profile']} audit expected "
                f"{expected_total_forwards} model forwards, observed "
                f"{report['model_forward_count']}"
            )
        block_cache_contract = report["config"].get("block_cache_contract")
        if stg_contract is not None:
            forwards = report["forwards"]
            observed_branches = [str(forward.get("branch", "main")) for forward in forwards]
            expected_branches = [str(value) for value in stg_contract["expected_branches"]]
            if observed_branches != expected_branches:
                raise RuntimeError(
                    "H3 EAV + STG branch sequence disagrees with the sigma contract: "
                    f"observed={observed_branches}, expected={expected_branches}"
                )
            skipped_blocks = tuple(int(value) for value in stg_contract["double_blocks"])
            weak_measurements = 50 - len(skipped_blocks)
            for forward in forwards:
                branch = str(forward.get("branch", "main"))
                expected_skipped = skipped_blocks if branch == "stg_weak" else ()
                observed_skipped = tuple(int(value) for value in forward.get("skipped_blocks", ()))
                if observed_skipped != expected_skipped:
                    raise RuntimeError(
                        "H3 EAV + STG skipped-block audit failed: "
                        f"branch={branch}, observed={observed_skipped}, expected={expected_skipped}"
                    )
                expected_count = 0
                if bool(forward["active"]):
                    expected_count = weak_measurements if branch == "stg_weak" else 50
                if int(forward["attention_count"]) != expected_count:
                    raise RuntimeError(
                        "H3 EAV + STG attention measurements disagree with the branch: "
                        f"branch={branch}, active={forward['active']}, "
                        f"observed={forward['attention_count']}, expected={expected_count}"
                    )
            observed_weak = observed_branches.count("stg_weak")
            expected_weak = int(stg_contract["expected_weak_forwards"])
            if observed_weak != expected_weak:
                raise RuntimeError(
                    "H3 EAV + STG weak-forward count drifted: "
                    f"observed={observed_weak}, expected={expected_weak}"
                )
            report["stg"] = {
                "base_nfe": expected_nfe,
                "weak_forwards": observed_weak,
                "total_joint_av_forwards": len(forwards),
                "skipped_double_blocks": list(skipped_blocks),
                "active_main_measurements": 50,
                "active_weak_measurements": weak_measurements,
                "eav_applied_to_main_and_weak": True,
            }
        elif block_cache_contract is None:
            active_counts = report["attention_calls_per_active_forward"]
            if not active_counts or any(count != 50 for count in active_counts):
                raise RuntimeError(
                    "H3 EAV expected exactly 50 main DiT attention measurements per active "
                    f"forward, observed {active_counts}"
                )
        else:
            forwards = report["forwards"]
            decisions = [forward.get("block_cache_decision") for forward in forwards]
            if not decisions or any(value not in {"full", "hit"} for value in decisions):
                raise RuntimeError(
                    "H3 EAV + BlockCache requires one audited full/hit decision per forward; "
                    f"observed {decisions}"
                )
            if decisions[0] != "full":
                raise RuntimeError("H3 EAV + BlockCache first forward must warm with a full pass")
            consecutive_hits = 0
            max_observed_hits = 0
            for forward, decision in zip(forwards, decisions, strict=True):
                expected_count = 0
                if bool(forward["active"]):
                    expected_count = 1 if decision == "hit" else 50
                if int(forward["attention_count"]) != expected_count:
                    raise RuntimeError(
                        "H3 EAV + BlockCache attention measurements disagree with the "
                        f"cache decision: decision={decision}, active={forward['active']}, "
                        f"observed={forward['attention_count']}, expected={expected_count}"
                    )
                consecutive_hits = consecutive_hits + 1 if decision == "hit" else 0
                max_observed_hits = max(max_observed_hits, consecutive_hits)
            configured_max_hits = int(
                block_cache_contract["config"]["max_consecutive_hits"]
            )
            if max_observed_hits > configured_max_hits:
                raise RuntimeError(
                    "H3 EAV + BlockCache exceeded its consecutive-hit contract: "
                    f"observed={max_observed_hits}, configured={configured_max_hits}"
                )
            hit_count = decisions.count("hit")
            full_count = decisions.count("full")
            report["block_cache"] = {
                "model_forwards": len(decisions),
                "full_forwards": full_count,
                "cache_hits": hit_count,
                "hit_rate": hit_count / len(decisions),
                "max_consecutive_hits_observed": max_observed_hits,
                "active_measurements_expected_from_decisions": sum(
                    0
                    if not bool(forward["active"])
                    else (1 if decision == "hit" else 50)
                    for forward, decision in zip(forwards, decisions, strict=True)
                ),
            }
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
        long_video_contract = report["config"].get("long_video_contract")
        if long_video_contract is not None:
            expected_task = (
                "LongVideoSegment0"
                if int(long_video_contract["segment_index"]) == 0
                else "LongVideoMotion"
            )
            observed_tasks = [str(forward.get("task")) for forward in report["forwards"]]
            if observed_tasks != [expected_task] * expected_nfe:
                raise RuntimeError(
                    "H3 EAV + Long Video segment task drifted: "
                    f"observed={observed_tasks}, expected={expected_task}"
                )
            report["long_video"] = {
                "segment_index": int(long_video_contract["segment_index"]),
                "context_frames": int(long_video_contract["context_frames"]),
                "binding_hash": str(long_video_contract["binding_hash"]),
                "model_forwards": expected_nfe,
                "execution_local_runtime_consumed": True,
            }
        if stg_contract is not None:
            report["status"] = (
                "report_only_stg_verified"
                if mode == "report_only"
                else "apply_exp_stg_verified"
            )
        elif long_video_contract is not None:
            report["status"] = (
                "report_only_long_video_segment_verified"
                if mode == "report_only"
                else "apply_exp_long_video_segment_verified"
            )
        elif block_cache_contract is None:
            report["status"] = (
                "report_only_verified" if mode == "report_only" else "apply_exp_verified"
            )
        else:
            report["status"] = (
                "report_only_block_cache_verified"
                if mode == "report_only"
                else "apply_exp_block_cache_verified"
            )
    report["quality_claim"] = (
        "mechanically audited only; visual motion/detail and joint-AV audio quality require "
        "the controlled baseline/apply A/B review"
    )
    return av_latent, _json(report)
