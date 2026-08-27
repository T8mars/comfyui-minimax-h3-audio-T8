from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

from .vram_policy import runtime_snapshot


ENVIRONMENT_AUDIT_SCHEMA = "t8.minimax_h3.environment_audit.v1"
REFERENCE_PIXEL_AREA = 1920 * 1088
VRAM_CAUTION_PIXEL_AREA = 1344 * 768

KNOWN_CORE_COMMITS = {
    "h3_flow_av_audio_scale": "bdcb886a4705a03cf40f4a7226de9fc7c059fc90",
    "video_vae_generic_chunked_io": "2a68ce33b4c9ea6ee4283e618a74560cefb32694",
    "attention_peak_clone": "62b3c94bd45154f6486c7abf1b9efcacee96ea69",
    "tiled_decode_nested_tensor_fix": "6233790c6dff26bf35113d46d6d3367b7041b1d8",
    "audio_vae_full_offload_fix": "2340099d93305bfdf4eaa29e9f8d32ec92d3035f",
}

KJ_SAGE_REQUIRED_CORE_SYMBOLS = (
    "per_thread_int8_triton",
    "per_warp_int8_cuda",
    "per_block_int8_triton",
    "per_channel_fp8",
    "get_cuda_arch_versions",
    "attn_false",
)


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
    )


def _issue(target: list[dict[str, Any]], code: str, message: str, **evidence: Any) -> None:
    item: dict[str, Any] = {"code": code, "message": message}
    if evidence:
        item["evidence"] = evidence
    target.append(item)


def _capability(state: str, evidence: str, **details: Any) -> dict[str, Any]:
    if state not in {"supported", "unsupported", "unknown"}:
        raise ValueError(f"invalid capability state: {state!r}")
    result: dict[str, Any] = {"state": state, "evidence": evidence}
    if details:
        result["details"] = details
    return result


def _safe_source(value: Any) -> str | None:
    try:
        return inspect.getsource(value)
    except (OSError, TypeError):
        return None


def _discover_comfy_root() -> Path | None:
    try:
        comfy = importlib.import_module("comfy")
        package_file = getattr(comfy, "__file__", None)
        if package_file:
            location = Path(package_file).resolve()
        else:
            package_paths = list(getattr(comfy, "__path__", []))
            if not package_paths:
                return None
            package_dir = Path(package_paths[0]).resolve()
            return package_dir.parent
    except (AttributeError, ImportError, OSError, TypeError):
        return None
    return location.parent.parent


def _git_capture(root: Path, *arguments: str) -> tuple[int | None, str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None, ""
    return int(result.returncode), result.stdout.strip()


def _git_snapshot(root: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "root": None if root is None else str(root),
        "head": None,
        "available": False,
        "known_commit_ancestry": {},
    }
    if root is None:
        result["inspection_error"] = "ComfyUI root could not be discovered"
        return result
    code, head = _git_capture(root, "rev-parse", "HEAD")
    if code != 0 or not head:
        result["inspection_error"] = "ComfyUI git metadata is unavailable"
        return result
    result["available"] = True
    result["head"] = head
    for name, commit in KNOWN_CORE_COMMITS.items():
        ancestry_code, _ = _git_capture(root, "merge-base", "--is-ancestor", commit, "HEAD")
        result["known_commit_ancestry"][name] = (
            None if ancestry_code is None or ancestry_code not in {0, 1} else ancestry_code == 0
        )
    return result


def _module_sources() -> dict[str, str | None]:
    targets = {
        "model": "comfy.ldm.minimax.model",
        "video_vae": "comfy.ldm.minimax.vae",
        "audio_vae": "comfy.ldm.minimax.audio_vae",
        "native_nodes": "comfy_extras.nodes_minimax_h3",
        "model_base": "comfy.model_base",
    }
    result: dict[str, str | None] = {}
    for key, module_name in targets.items():
        try:
            module = importlib.import_module(module_name)
        except Exception:
            result[key] = None
            continue
        result[key] = _safe_source(module)
    return result


def _source_capability(
    source: str | None,
    predicate: bool,
    supported_evidence: str,
    unsupported_evidence: str,
) -> dict[str, Any]:
    if source is None:
        return _capability("unknown", "runtime source inspection failed")
    return _capability(
        "supported" if predicate else "unsupported",
        supported_evidence if predicate else unsupported_evidence,
    )


def _constant_property_value(owner: type, name: str) -> float | None:
    descriptor = inspect.getattr_static(owner, name, None)
    if not isinstance(descriptor, property) or descriptor.fget is None:
        return None
    try:
        value = float(descriptor.fget(None))
    except (AttributeError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def collect_h3_audio_scale_snapshot() -> dict[str, Any]:
    """Inspect the pre/post FLOW_AV contract without loading a model or CUDA kernel."""

    result: dict[str, Any] = {
        "inspected": True,
        "core_protocol": "unknown",
        "core_requires_model_sampling_audio_scale": None,
        "native_model_sampling_av_available": False,
        "native_audio_scale_12_over_3": None,
        "stable_custom_audio_scale": None,
        "multirate_custom_audio_scale": None,
        "custom_sampling_compatible": None,
        "errors": [],
    }
    try:
        model_base = importlib.import_module("comfy.model_base")
        minimax_base = getattr(model_base, "MiniMaxH3")
        audio_scale_method = getattr(minimax_base, "audio_scale", None)
        audio_scale_source = _safe_source(audio_scale_method)
        requires_audio_scale = bool(
            callable(audio_scale_method)
            and audio_scale_source
            and "model_sampling.audio_scale" in audio_scale_source
        )
        result["core_requires_model_sampling_audio_scale"] = requires_audio_scale
        result["core_protocol"] = "flow_av" if requires_audio_scale else "legacy_dual_clock"
    except Exception as error:
        result["errors"].append(
            {
                "stage": "core_model_base",
                "type": type(error).__name__,
                "message": str(error),
            }
        )

    try:
        model_sampling = importlib.import_module("comfy.model_sampling")
        native_cls = getattr(model_sampling, "ModelSamplingAV", None)
        result["native_model_sampling_av_available"] = native_cls is not None
        if native_cls is not None:
            config = type(
                "H3AudioScaleAuditConfig",
                (),
                {"sampling_settings": {"shift": 1.0, "multiplier": 1000}},
            )()
            native = native_cls(config)
            native.set_parameters(shift=12.0, audio_shift=3.0)
            result["native_audio_scale_12_over_3"] = float(native.audio_scale)
    except Exception as error:
        result["errors"].append(
            {
                "stage": "native_model_sampling_av",
                "type": type(error).__name__,
                "message": str(error),
            }
        )

    try:
        from .sampling import MiniMaxH3FlowSampling
        from .sampling_multirate_exp import MiniMaxH3MultiRateFlowSamplingEXP

        result["stable_custom_audio_scale"] = _constant_property_value(
            MiniMaxH3FlowSampling, "audio_scale"
        )
        result["multirate_custom_audio_scale"] = _constant_property_value(
            MiniMaxH3MultiRateFlowSamplingEXP, "audio_scale"
        )
    except Exception as error:
        result["errors"].append(
            {
                "stage": "t8_custom_sampling",
                "type": type(error).__name__,
                "message": str(error),
            }
        )

    core_requires = result["core_requires_model_sampling_audio_scale"]
    neutral_custom_scales = (
        result["stable_custom_audio_scale"] == 1.0
        and result["multirate_custom_audio_scale"] == 1.0
    )
    if core_requires is None:
        result["custom_sampling_compatible"] = None
    elif core_requires:
        result["custom_sampling_compatible"] = neutral_custom_scales
    else:
        # Legacy H3 never reads audio_scale, while the same neutral properties are
        # harmless and keep one plugin build usable on both core generations.
        result["custom_sampling_compatible"] = neutral_custom_scales
    return result


def _capabilities(git: Mapping[str, Any], sources: Mapping[str, str | None]) -> dict[str, Any]:
    model_source = sources.get("model")
    vae_source = sources.get("video_vae")
    native_source = sources.get("native_nodes")
    model_base_source = sources.get("model_base")
    ancestry = git.get("known_commit_ancestry", {})
    audio_scale_snapshot = collect_h3_audio_scale_snapshot()

    try:
        model_sampling = importlib.import_module("comfy.model_sampling")
        native_av = hasattr(model_sampling, "ModelSamplingAV")
    except Exception:
        native_av = None

    try:
        vae_module = importlib.import_module("comfy.ldm.minimax.vae")
        video_vae_cls = getattr(vae_module, "MiniMaxH3VideoVAE")
        create_token_ids = getattr(vae_module, "create_token_ids")
        generic_chunked_attr = bool(getattr(video_vae_cls, "comfy_has_chunked_io", False))
        token_id_source = _safe_source(create_token_ids)
    except Exception:
        generic_chunked_attr = None
        token_id_source = None

    chunked_commit = ancestry.get("video_vae_generic_chunked_io")
    generic_chunked = generic_chunked_attr is True or chunked_commit is True
    if generic_chunked_attr is None and chunked_commit is None:
        generic_chunked_cap = _capability("unknown", "class attribute and git ancestry are unavailable")
    else:
        generic_chunked_cap = _capability(
            "supported" if generic_chunked else "unsupported",
            (
                "MiniMaxH3VideoVAE exposes comfy_has_chunked_io or the known merge is an ancestor"
                if generic_chunked
                else "no comfy_has_chunked_io marker and the known merge is not an ancestor"
            ),
            class_marker=generic_chunked_attr,
            known_commit_ancestor=chunked_commit,
        )

    tiled_nested = ancestry.get("tiled_decode_nested_tensor_fix")
    if tiled_nested is None:
        tiled_nested_cap = _capability("unknown", "git ancestry is unavailable")
    else:
        tiled_nested_cap = _capability(
            "supported" if tiled_nested else "unsupported",
            "known tiled decode NestedTensor fix ancestry check",
            known_commit_ancestor=tiled_nested,
        )

    audio_offload = ancestry.get("audio_vae_full_offload_fix")
    if audio_offload is None:
        audio_offload_cap = _capability("unknown", "git ancestry is unavailable")
    else:
        audio_offload_cap = _capability(
            "supported" if audio_offload else "unsupported",
            "known H3 audio VAE full-offload fix ancestry check",
            known_commit_ancestor=audio_offload,
        )

    peak_clone_source = model_source is not None and "v = v.clone()" in model_source
    peak_clone_commit = ancestry.get("attention_peak_clone")
    if model_source is None and peak_clone_commit is None:
        peak_clone_cap = _capability("unknown", "source and git ancestry are unavailable")
    else:
        peak_clone = peak_clone_source or peak_clone_commit is True
        peak_clone_cap = _capability(
            "supported" if peak_clone else "unsupported",
            "source marker or known H3 attention peak-memory commit ancestry",
            source_marker=peak_clone_source,
            known_commit_ancestor=peak_clone_commit,
        )

    arbitrary_guides = bool(
        native_source
        and "MiniMaxH3AddGuide" in native_source
        and model_source
        and "only first/last keyframe anchors are supported" not in model_source
    )
    per_token_masks = bool(
        model_base_source
        and model_source
        and (
            "minimax_video_noise_mask" in model_base_source
            or "minimax_audio_noise_mask" in model_base_source
        )
        and "noise_mask" in model_source
    )
    attention_hooks = bool(
        model_source
        and any(
            marker in model_source
            for marker in (
                'patches_replace.get("attention"',
                "patches_replace.get('attention'",
                "minimax_h3_attention_patch",
            )
        )
    )
    tiled_global_coordinates = bool(
        token_id_source
        and "full_dims" in token_id_source
        and "offset" in token_id_source
    )

    if native_av is None:
        native_av_cap = _capability("unknown", "ModelSamplingAV import failed")
    else:
        native_av_cap = _capability(
            "supported" if native_av else "unsupported",
            "runtime ModelSamplingAV symbol inspection",
        )

    audio_scale_compatible = audio_scale_snapshot.get("custom_sampling_compatible")
    if audio_scale_compatible is None:
        audio_scale_cap = _capability(
            "unknown",
            "MiniMax H3 audio-scale contract inspection was inconclusive",
            **audio_scale_snapshot,
            known_flow_av_commit_ancestor=ancestry.get("h3_flow_av_audio_scale"),
        )
    else:
        audio_scale_cap = _capability(
            "supported" if audio_scale_compatible else "unsupported",
            (
                "stable and multirate custom samplers expose neutral audio_scale=1.0 for the current FLOW_AV core"
                if audio_scale_compatible
                and audio_scale_snapshot.get("core_requires_model_sampling_audio_scale")
                else "neutral custom audio-scale properties preserve the legacy dual-clock core contract"
                if audio_scale_compatible
                else "a custom H3 sampler does not satisfy the active core audio-scale contract"
            ),
            **audio_scale_snapshot,
            known_flow_av_commit_ancestor=ancestry.get("h3_flow_av_audio_scale"),
        )

    return {
        "native_model_sampling_av": native_av_cap,
        "t8_custom_sampling_audio_scale": audio_scale_cap,
        "diffusion_model_wrapper": _source_capability(
            model_source,
            bool(model_source and "WrappersMP.DIFFUSION_MODEL" in model_source),
            "MiniMaxH3Model.forward uses the ComfyUI diffusion-model wrapper executor",
            "MiniMaxH3Model.forward has no diffusion-model wrapper marker",
        ),
        "dit_double_block_replace": _source_capability(
            model_source,
            bool(model_source and 'blocks_replace.get' not in model_source and '"double_block"' in model_source),
            "MiniMaxH3Model._forward exposes dit/double_block replacements",
            "dit/double_block replacement marker is absent",
        ),
        "video_vae_internal_temporal_chunking": _source_capability(
            vae_source,
            bool(
                vae_source
                and "def encode_temporal" in vae_source
                and "def decode_temporal" in vae_source
            ),
            "MiniMaxH3VideoVAE implements its own temporal encode/decode chunk loop",
            "internal temporal chunk-loop methods are absent",
        ),
        "video_vae_generic_chunked_io": generic_chunked_cap,
        "tiled_decode_nested_tensor_fix": tiled_nested_cap,
        "tiled_decode_global_coordinates": _source_capability(
            token_id_source,
            tiled_global_coordinates,
            "token-coordinate construction accepts full dimensions and tile offsets",
            "token-coordinate construction has no full-dimension/tile-offset contract",
        ),
        "audio_vae_full_offload_fix": audio_offload_cap,
        "attention_peak_clone_fix": peak_clone_cap,
        "native_arbitrary_guides": _source_capability(
            native_source if model_source is not None else None,
            arbitrary_guides,
            "native AddGuide and arbitrary PackedLayout positions are both present",
            "native AddGuide/arbitrary PackedLayout support is incomplete",
        ),
        "per_token_h3_latent_masks": _source_capability(
            model_source if model_base_source is not None else None,
            per_token_masks,
            "H3-specific video/audio token-mask payload is present",
            "H3-specific per-token video/audio mask payload is absent",
        ),
        "h3_attention_patch_hooks": _source_capability(
            model_source,
            attention_hooks,
            "H3 attention-specific replacement hook marker is present",
            "only whole-block replacement is visible; no H3 attention-specific hook marker",
        ),
    }


def _callable_owner(value: Any) -> str:
    target = value
    if hasattr(value, "__call__") and not inspect.isfunction(value) and not inspect.ismethod(value):
        target = type(value)
    module = getattr(target, "__module__", type(value).__module__)
    name = getattr(target, "__qualname__", getattr(target, "__name__", type(value).__qualname__))
    return f"{module}.{name}"


def _model_patch_snapshot(model: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "connected": model is not None,
        "patch_replace_groups": {},
        "loaded_related_modules": [],
        "packed_layout_owner": None,
        "packed_layout_expected_owner": True,
    }
    try:
        minimax_model = importlib.import_module("comfy.ldm.minimax.model")
        packed_layout = getattr(minimax_model, "PackedLayout")
        owner = f"{packed_layout.__init__.__module__}.{packed_layout.__init__.__qualname__}"
        result["packed_layout_owner"] = owner
        result["packed_layout_expected_owner"] = owner.startswith(
            "comfy.ldm.minimax.model.PackedLayout."
        )
    except Exception as error:
        result["packed_layout_owner"] = f"inspection failed: {type(error).__name__}: {error}"
        result["packed_layout_expected_owner"] = None

    related_markers = (
        "minimax",
        "spectrum",
        "stepcache",
        "blockcache",
        "block_cache",
        "sageattention",
        "motion_context",
    )
    result["loaded_related_modules"] = sorted(
        name for name in sys.modules if any(marker in name.lower() for marker in related_markers)
    )
    if model is None:
        return result
    try:
        options = getattr(model, "model_options", {})
        transformer = options.get("transformer_options", {}) if isinstance(options, Mapping) else {}
        patches_replace = transformer.get("patches_replace", {}) if isinstance(transformer, Mapping) else {}
        if isinstance(patches_replace, Mapping):
            for group, raw_entries in patches_replace.items():
                owners: list[str] = []
                entries = raw_entries.values() if isinstance(raw_entries, Mapping) else []
                for value in entries:
                    owners.append(_callable_owner(value))
                result["patch_replace_groups"][str(group)] = {
                    "entry_count": len(owners),
                    "owners": sorted(set(owners)),
                }
        wrappers = transformer.get("wrappers", {}) if isinstance(transformer, Mapping) else {}
        if isinstance(wrappers, Mapping):
            result["wrapper_groups"] = sorted(str(key) for key in wrappers)
        attachments = getattr(model, "attachments", {})
        if isinstance(attachments, Mapping):
            result["attachment_keys"] = sorted(str(key) for key in attachments)
    except Exception as error:
        result["inspection_error"] = f"{type(error).__name__}: {error}"
    return result


def _conditioning_snapshot(positive: Any) -> dict[str, Any]:
    result = {"connected": positive is not None, "metadata_keys": [], "reference_kinds": []}
    if positive is None:
        return result
    keys: set[str] = set()
    kinds: list[str] = []
    try:
        entries = positive if isinstance(positive, Sequence) else []
        for entry in entries:
            if not isinstance(entry, Sequence) or len(entry) < 2 or not isinstance(entry[1], Mapping):
                continue
            metadata = entry[1]
            keys.update(str(key) for key in metadata)
            refs = metadata.get("minimax_refs", [])
            if isinstance(refs, Sequence):
                for ref in refs:
                    if isinstance(ref, Mapping):
                        kinds.append(str(ref.get("kind", "unknown")))
        result["metadata_keys"] = sorted(keys)
        result["reference_kinds"] = kinds
    except Exception as error:
        result["inspection_error"] = f"{type(error).__name__}: {error}"
    return result


def _loaded_model_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": False,
        "count": 0,
        "currently_used_count": 0,
        "total_model_mib": 0.0,
        "total_loaded_mib": 0.0,
        "models": [],
    }
    try:
        model_management = importlib.import_module("comfy.model_management")
        loaded = list(getattr(model_management, "current_loaded_models", []) or [])
        result["available"] = True
        for entry in loaded:
            patcher = getattr(entry, "model", None)
            if patcher is None:
                continue
            total = int(patcher.model_size())
            resident = int(patcher.loaded_size())
            base = getattr(patcher, "model", None)
            item = {
                "class": type(base).__name__,
                "device": str(getattr(entry, "device", getattr(patcher, "load_device", "unknown"))),
                "currently_used": bool(getattr(entry, "currently_used", False)),
                "dynamic": bool(patcher.is_dynamic()),
                "model_mib": total / (1024**2),
                "loaded_mib": resident / (1024**2),
                "offloaded_mib": max(0, total - resident) / (1024**2),
            }
            result["models"].append(item)
        result["count"] = len(result["models"])
        result["currently_used_count"] = sum(
            1 for item in result["models"] if item["currently_used"]
        )
        result["total_model_mib"] = sum(item["model_mib"] for item in result["models"])
        result["total_loaded_mib"] = sum(item["loaded_mib"] for item in result["models"])
    except Exception as error:
        result["inspection_error"] = f"{type(error).__name__}: {error}"
    return result


def collect_sageattention_snapshot() -> dict[str, Any]:
    """Inspect the exact SageAttention core contract used by current KJNodes.

    KJNodes intentionally catches every import/probe exception while initializing its
    MiniMax H3 Sage patch, which collapses incompatible wheels, missing symbols and
    CUDA-architecture probe failures into the same ``_cuda_archs is None`` message.
    This read-only probe preserves the actual failure stage without importing KJNodes,
    loading an H3 model or running a CUDA attention kernel.
    """

    result: dict[str, Any] = {
        "inspected": True,
        "package_imported": False,
        "core_imported": False,
        "package_version": None,
        "package_file": None,
        "core_file": None,
        "required_symbols": {},
        "missing_symbols": list(KJ_SAGE_REQUIRED_CORE_SYMBOLS),
        "architectures": [],
        "kj_contract_ready": False,
        "errors": [],
    }
    try:
        metadata = importlib.import_module("importlib.metadata")
        result["package_version"] = metadata.version("sageattention")
    except Exception:
        # Several source/wheel builds do not publish distribution metadata. Import
        # and symbol evidence below is authoritative for this compatibility check.
        pass

    try:
        package = importlib.import_module("sageattention")
        result["package_imported"] = True
        result["package_file"] = str(getattr(package, "__file__", None))
    except Exception as error:
        result["errors"].append(
            {
                "stage": "package_import",
                "type": type(error).__name__,
                "message": str(error),
            }
        )
        return result

    try:
        core = importlib.import_module("sageattention.core")
        result["core_imported"] = True
        result["core_file"] = str(getattr(core, "__file__", None))
    except Exception as error:
        result["errors"].append(
            {
                "stage": "core_import",
                "type": type(error).__name__,
                "message": str(error),
            }
        )
        return result

    symbols = {
        name: callable(getattr(core, name, None))
        for name in KJ_SAGE_REQUIRED_CORE_SYMBOLS
    }
    missing = [name for name, available in symbols.items() if not available]
    result["required_symbols"] = symbols
    result["missing_symbols"] = missing
    if missing:
        return result

    try:
        raw_architectures = core.get_cuda_arch_versions()
        if isinstance(raw_architectures, str):
            raw_architectures = [raw_architectures]
        result["architectures"] = sorted(
            {
                str(item).strip().lower()
                for item in (raw_architectures or [])
                if str(item).strip()
            }
        )
    except Exception as error:
        result["errors"].append(
            {
                "stage": "cuda_architecture_probe",
                "type": type(error).__name__,
                "message": str(error),
            }
        )
        return result

    result["kj_contract_ready"] = bool(result["architectures"])
    return result


def collect_environment_snapshot(model: Any = None, positive: Any = None) -> dict[str, Any]:
    root = _discover_comfy_root()
    git = _git_snapshot(root)
    sources = _module_sources()
    return {
        "git": git,
        "capabilities": _capabilities(git, sources),
        "runtime": runtime_snapshot(),
        "model_patch_stack": _model_patch_snapshot(model),
        "conditioning": _conditioning_snapshot(positive),
        "loaded_models": _loaded_model_snapshot(),
    }


def _state(capabilities: Mapping[str, Any], key: str) -> str:
    value = capabilities.get(key, {})
    return str(value.get("state", "unknown")) if isinstance(value, Mapping) else "unknown"


def _estimated_rows(width: int, height: int, length: int, reference_count: int) -> dict[str, int]:
    latent_t = max(1, ((length - 5) // 17) * 5 + 2) if length >= 5 else 1
    frame_rows = math.ceil(width / 32) * math.ceil(height / 32)
    video_rows = frame_rows * latent_t
    audio_steps = max(1, math.ceil(length / 24.0 * 40.0))
    return {
        "latent_video_t": latent_t,
        "frame_rows": frame_rows,
        "target_video_rows": video_rows,
        "target_audio_rows": audio_steps * 2,
        "estimated_single_image_reference_rows": frame_rows * max(0, reference_count),
        "estimated_target_plus_single_image_rows": video_rows + audio_steps * 2 + frame_rows * max(0, reference_count),
    }


def audit_h3_environment(
    workload_profile: str,
    width: int,
    height: int,
    length: int,
    model_family: str,
    model_precision: str,
    attention_backend: str,
    cache_backend: str,
    decode_mode: str,
    dynamic_vram_mode: str,
    reference_media_count: int,
    middle_keyframe_count: int,
    minimum_current_headroom_mib: float,
    model: Any = None,
    positive: Any = None,
    *,
    snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if width <= 0 or height <= 0 or length <= 0:
        raise ValueError("width, height and length must be positive")
    if reference_media_count < 0 or middle_keyframe_count < 0:
        raise ValueError("reference and middle-keyframe counts must be non-negative")
    if not math.isfinite(float(minimum_current_headroom_mib)) or minimum_current_headroom_mib < 0:
        raise ValueError("minimum_current_headroom_mib must be finite and non-negative")

    current = dict(snapshot) if snapshot is not None else collect_environment_snapshot(model, positive)
    sage_runtime = current.get("sageattention_runtime")
    if attention_backend == "sage_attention" and not isinstance(sage_runtime, Mapping):
        if snapshot is None:
            sage_runtime = collect_sageattention_snapshot()
        else:
            # Supplied snapshots are deterministic evidence fixtures or externally
            # captured states. Never mix them with this machine's live Sage install.
            sage_runtime = {
                "inspected": False,
                "kj_contract_ready": None,
                "reason": "sageattention_runtime was absent from the supplied snapshot",
            }
        current["sageattention_runtime"] = sage_runtime
    capabilities = current.get("capabilities", {})
    runtime = current.get("runtime", {})
    patch_stack = current.get("model_patch_stack", {})
    conditioning = current.get("conditioning", {})
    loaded_models = current.get("loaded_models", {})
    pixels = int(width) * int(height)
    rows = _estimated_rows(width, height, length, reference_media_count + middle_keyframe_count)

    hard: list[dict[str, Any]] = []
    high_risk: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []

    if pixels > REFERENCE_PIXEL_AREA:
        _issue(
            warnings,
            "canvas_area_above_reference_area",
            "Canvas area exceeds the 1920x1088 reference area; this is warning-only and does not block execution.",
            pixels=pixels,
            reference_area=REFERENCE_PIXEL_AREA,
        )
    audio_scale_state = _state(capabilities, "t8_custom_sampling_audio_scale")
    if audio_scale_state == "unsupported":
        _issue(
            hard,
            "h3_custom_sampling_audio_scale_incompatible",
            "The active ComfyUI MiniMax H3 core reads model_sampling.audio_scale, but the T8 custom sampling contract is incomplete.",
            capability=capabilities.get("t8_custom_sampling_audio_scale"),
        )
    elif audio_scale_state == "unknown":
        _issue(
            unknown,
            "h3_custom_sampling_audio_scale_unknown",
            "The MiniMax H3 custom-sampler audio-scale compatibility contract could not be inspected.",
        )
    if width % 32 or height % 32:
        _issue(
            hard,
            "canvas_not_multiple_of_32",
            "MiniMax H3 canvas dimensions must be multiples of 32.",
            width=width,
            height=height,
        )
    if length < 5 or (length - 5) % 17:
        _issue(
            hard,
            "frame_count_off_h3_grid",
            "Frame count must satisfy the MiniMax H3 17n+5 grid.",
            length=length,
        )
    if length < 124 or length > 362:
        _issue(
            warnings,
            "outside_approximate_training_window",
            "The requested frame count is outside the approximate 124-362 training window.",
            length=length,
        )
    if pixels > VRAM_CAUTION_PIXEL_AREA:
        _issue(
            warnings,
            "high_canvas_activation_pressure",
            "Canvas exceeds the 1344x768 caution area; no memory-safe inference is made.",
            pixels=pixels,
            caution_area=VRAM_CAUTION_PIXEL_AREA,
        )

    if patch_stack.get("packed_layout_expected_owner") is False:
        _issue(
            hard,
            "global_packed_layout_patch_detected",
            "PackedLayout.__init__ is owned outside the native ComfyUI MiniMax module.",
            owner=patch_stack.get("packed_layout_owner"),
        )
    elif patch_stack.get("packed_layout_expected_owner") is None:
        _issue(
            unknown,
            "packed_layout_owner_unknown",
            "PackedLayout ownership could not be verified.",
        )

    internal_spatial_tiling_expected = width > 256 or height > 256
    tiled_coordinate_state = _state(capabilities, "tiled_decode_global_coordinates")
    if internal_spatial_tiling_expected and tiled_coordinate_state == "unsupported":
        _issue(
            high_risk,
            "h3_spatial_tiling_global_coordinates_missing",
            "Current H3 regular and explicit tiled decode both use internal spatial tiles above 256 pixels, but global tile coordinates are absent.",
            capability_state=tiled_coordinate_state,
            width=width,
            height=height,
            internal_tile_size_pixels=256,
            requested_decode_mode=decode_mode,
        )
    elif internal_spatial_tiling_expected and tiled_coordinate_state != "supported":
        _issue(
            unknown,
            "h3_spatial_tiling_coordinate_contract_unknown",
            "The H3 internal spatial-tile coordinate contract could not be verified for this canvas.",
            capability_state=tiled_coordinate_state,
            width=width,
            height=height,
            internal_tile_size_pixels=256,
            requested_decode_mode=decode_mode,
        )
    if decode_mode == "tiled" and _state(capabilities, "tiled_decode_nested_tensor_fix") != "supported":
        _issue(
            warnings,
            "tiled_decode_nested_tensor_fix_missing",
            "The known tiled-decode NestedTensor compatibility fix is not verified in this core.",
            capability_state=_state(capabilities, "tiled_decode_nested_tensor_fix"),
        )

    high_row_count = rows["estimated_target_plus_single_image_rows"] >= 50_000
    resolved_dynamic = dynamic_vram_mode
    if dynamic_vram_mode == "auto_detect":
        dynamic_flag = runtime.get("comfy", {}).get("dynamic_vram_enabled")
        resolved_dynamic = "enabled" if dynamic_flag is True else "disabled" if dynamic_flag is False else "unknown"
    if (
        model_family == "ref2va"
        and model_precision == "fp8"
        and attention_backend == "sage_attention"
        and resolved_dynamic == "enabled"
        and high_row_count
    ):
        _issue(
            high_risk,
            "fp8_ref2va_sage_dynamic_high_token_risk",
            "FP8 Ref2VA + SageAttention + DynamicVRAM at high token counts matches a known crash-risk profile; this is not certified safe.",
            estimated_rows=rows["estimated_target_plus_single_image_rows"],
        )

    gpu_info = runtime.get("gpu", {}) if isinstance(runtime, Mapping) else {}
    compute_capability = (
        gpu_info.get("compute_capability") if isinstance(gpu_info, Mapping) else None
    )
    capability_major = None
    capability_minor = None
    if (
        isinstance(compute_capability, Sequence)
        and len(compute_capability) >= 2
        and not isinstance(compute_capability, (str, bytes))
    ):
        try:
            capability_major = int(compute_capability[0])
            capability_minor = int(compute_capability[1])
        except (TypeError, ValueError):
            capability_major = None
            capability_minor = None
    if attention_backend == "sage_attention" and isinstance(sage_runtime, Mapping):
        if sage_runtime.get("inspected") is False:
            _issue(
                unknown,
                "sageattention_runtime_not_inspected",
                "The supplied environment snapshot does not include the SageAttention core contract.",
            )
        elif sage_runtime.get("package_imported") is not True or sage_runtime.get("core_imported") is not True:
            _issue(
                hard,
                "sageattention_core_import_failed",
                "SageAttention or sageattention.core failed to import in the active ComfyUI Python environment.",
                errors=sage_runtime.get("errors", []),
                package_version=sage_runtime.get("package_version"),
            )
        elif sage_runtime.get("missing_symbols"):
            _issue(
                hard,
                "sageattention_kj_contract_symbols_missing",
                "The installed SageAttention core does not expose every symbol required by the current KJNodes MiniMax H3 Sage patch.",
                missing_symbols=list(sage_runtime.get("missing_symbols", [])),
                package_version=sage_runtime.get("package_version"),
            )
        elif not sage_runtime.get("architectures"):
            _issue(
                hard,
                "sageattention_cuda_architecture_undetected",
                "SageAttention imported, but its CUDA architecture probe returned no usable architecture.",
                errors=sage_runtime.get("errors", []),
                package_version=sage_runtime.get("package_version"),
            )
        elif capability_major is not None and capability_minor is not None:
            expected_architecture = f"sm{capability_major}{capability_minor}"
            compiled_architectures = {
                str(item).lower() for item in sage_runtime.get("architectures", [])
            }
            if expected_architecture not in compiled_architectures:
                _issue(
                    hard,
                    "sageattention_cuda_architecture_mismatch",
                    "The SageAttention wheel did not report a kernel architecture matching the active GPU.",
                    expected=expected_architecture,
                    reported=sorted(compiled_architectures),
                    gpu_name=gpu_info.get("name"),
                )
    if attention_backend == "sage_attention" and high_row_count:
        if capability_major is not None and capability_major >= 12:
            _issue(
                high_risk,
                "sage_sm120_high_token_output_corruption_risk",
                "SageAttention on compute capability 12.x at high H3 token counts has a reported pure-noise output failure; use stock attention until that exact kernel path is validated.",
                compute_capability=list(compute_capability),
                estimated_rows=rows["estimated_target_plus_single_image_rows"],
                gpu_name=gpu_info.get("name"),
            )
        elif capability_major is None:
            _issue(
                warnings,
                "sage_gpu_architecture_unknown_at_high_token_count",
                "GPU compute capability could not be verified for a high-token SageAttention request; import success alone does not establish output correctness.",
                estimated_rows=rows["estimated_target_plus_single_image_rows"],
            )

    if cache_backend in {"step_cache", "spectrum"}:
        _issue(
            warnings,
            "external_cache_not_calibrated",
            "This cache backend is outside the T8 Block Cache validation matrix; H3 audio and patch ordering require separate verification.",
            cache_backend=cache_backend,
        )
    if cache_backend == "other_custom":
        _issue(
            unknown,
            "custom_cache_owner_unknown",
            "The selected custom cache backend has no known T8 compatibility contract.",
        )
    if attention_backend == "other_custom":
        _issue(
            unknown,
            "custom_attention_owner_unknown",
            "The selected attention backend has no known T8 compatibility contract.",
        )

    if middle_keyframe_count > 0 and _state(capabilities, "native_arbitrary_guides") != "supported":
        if workload_profile == "multikeyframe":
            _issue(
                warnings,
                "native_interior_guides_unavailable",
                "This core cannot place interior guides natively; only the T8 scoped MultiKeyframe Advanced route is eligible.",
                middle_keyframe_count=middle_keyframe_count,
            )
        else:
            _issue(
                high_risk,
                "interior_guides_without_scoped_route",
                "Interior guides were requested outside the T8 MultiKeyframe Advanced workload route on a core without native support.",
                middle_keyframe_count=middle_keyframe_count,
            )

    if workload_profile in {"speech", "ref2va", "hybrid", "long_video"}:
        audio_offload_state = _state(capabilities, "audio_vae_full_offload_fix")
        if audio_offload_state == "unsupported":
            _issue(
                high_risk,
                "audio_vae_full_offload_fix_missing",
                "The core predates the known H3 audio VAE full-offload fix; memory behavior is high risk.",
            )
        elif audio_offload_state == "unknown":
            _issue(
                unknown,
                "audio_vae_full_offload_state_unknown",
                "The H3 audio VAE full-offload fix could not be verified.",
            )

    if high_row_count and _state(capabilities, "video_vae_generic_chunked_io") != "supported":
        _issue(
            warnings,
            "generic_video_vae_chunked_io_missing",
            "The later generic H3 VAE chunked-I/O path is not verified; internal VAE chunking may still exist but is not equivalent.",
            estimated_rows=rows["estimated_target_plus_single_image_rows"],
        )

    if model_family == "auto_unknown":
        _issue(unknown, "model_family_unknown", "Model family was not explicitly identified.")
    if model_precision == "auto_unknown":
        _issue(unknown, "model_precision_unknown", "Model precision/quantization was not explicitly identified.")
    if attention_backend == "auto_detect" and not patch_stack.get("connected"):
        _issue(unknown, "attention_backend_unknown", "Connect MODEL or select the attention backend explicitly.")
    if cache_backend == "auto_detect" and not patch_stack.get("connected"):
        _issue(unknown, "cache_backend_unknown", "Connect MODEL or select the cache backend explicitly.")
    if resolved_dynamic == "unknown":
        _issue(unknown, "dynamic_vram_state_unknown", "DynamicVRAM state could not be detected.")

    free_mib = runtime.get("gpu", {}).get("whole_device_free_mib")
    if free_mib is None:
        _issue(unknown, "current_vram_headroom_unknown", "Whole-device CUDA headroom is unavailable.")
    elif float(free_mib) < float(minimum_current_headroom_mib):
        _issue(
            high_risk,
            "current_vram_headroom_below_gate",
            "Current whole-device free VRAM is below the configured audit gate.",
            current_mib=float(free_mib),
            minimum_mib=float(minimum_current_headroom_mib),
        )

    host = runtime.get("host", {}) if isinstance(runtime, Mapping) else {}
    ram_available_gib = host.get("ram_available_gib") if isinstance(host, Mapping) else None
    commit_headroom_gib = host.get("commit_headroom_gib") if isinstance(host, Mapping) else None
    staged_mib = loaded_models.get("total_model_mib") if isinstance(loaded_models, Mapping) else None
    resident_mib = loaded_models.get("total_loaded_mib") if isinstance(loaded_models, Mapping) else None
    loaded_count = loaded_models.get("count") if isinstance(loaded_models, Mapping) else None
    if commit_headroom_gib is not None and float(commit_headroom_gib) < 16.0:
        _issue(
            high_risk,
            "host_commit_headroom_low",
            "Host commit headroom is below the conservative 16GiB resource-thrashing gate.",
            commit_headroom_gib=float(commit_headroom_gib),
        )
    elif commit_headroom_gib is None:
        _issue(unknown, "host_commit_headroom_unknown", "Host commit headroom is unavailable.")
    if ram_available_gib is not None and float(ram_available_gib) < 8.0:
        _issue(
            high_risk,
            "host_ram_available_low",
            "Available physical RAM is below 8GiB; model/VAE page churn is high risk.",
            ram_available_gib=float(ram_available_gib),
        )
    elif ram_available_gib is None:
        _issue(unknown, "host_ram_available_unknown", "Available physical RAM is unavailable.")
    if staged_mib is not None and commit_headroom_gib is not None:
        staged_gib = float(staged_mib) / 1024.0
        if staged_gib > 0 and float(commit_headroom_gib) < max(16.0, staged_gib * 0.25):
            _issue(
                high_risk,
                "loaded_model_commit_thrashing_risk",
                "Loaded model footprint is large relative to remaining commit headroom; repeated DiT/CLIP/VAE switches may page-thrash or fail.",
                loaded_model_count=loaded_count,
                staged_model_gib=staged_gib,
                resident_model_gib=None if resident_mib is None else float(resident_mib) / 1024.0,
                commit_headroom_gib=float(commit_headroom_gib),
            )
    if isinstance(loaded_models, Mapping) and loaded_models.get("available") is False:
        _issue(unknown, "loaded_model_state_unknown", "ComfyUI loaded-model state is unavailable.")

    comfy_runtime = runtime.get("comfy", {}) if isinstance(runtime, Mapping) else {}
    if isinstance(comfy_runtime, Mapping) and comfy_runtime.get("fast_disk_enabled") is True:
        _issue(
            warnings,
            "disk_backed_dynamic_loading_enabled",
            "ComfyUI fast-disk mode is enabled; capture per-run read-byte/page-fault deltas before calling the workload usable.",
        )
    gpu_health = runtime.get("gpu_health", {}) if isinstance(runtime, Mapping) else {}
    if isinstance(gpu_health, Mapping) and gpu_health.get("available") is True:
        temperature = gpu_health.get("temperature_c")
        if gpu_health.get("thermal_throttling") is True:
            _issue(
                high_risk,
                "gpu_thermal_throttling_observed",
                "NVML reports active thermal clock throttling.",
                temperature_c=temperature,
                throttle_reasons_raw=gpu_health.get("throttle_reasons_raw"),
            )
        elif isinstance(temperature, (int, float)) and float(temperature) >= 84.0:
            _issue(
                warnings,
                "gpu_temperature_high",
                "GPU temperature is at or above the conservative 84C audit warning.",
                temperature_c=float(temperature),
            )

    if hard:
        status = "blocked"
    elif high_risk:
        status = "high_risk"
    elif unknown:
        status = "unknown"
    elif warnings:
        status = "pass_with_warnings"
    else:
        status = "pass"

    process_runtime = runtime.get("process", {}) if isinstance(runtime, Mapping) else {}
    resource_fit = (
        "unsafe_current_state"
        if hard or high_risk
        else "fits_current_snapshot_thrashing_unmeasured"
        if isinstance(process_runtime, Mapping) and process_runtime.get("available") is True
        else "unknown_process_io"
    )
    return {
        "schema": ENVIRONMENT_AUDIT_SCHEMA,
        "status": status,
        "no_known_blocker": not hard and not high_risk,
        "memory_safe_claim": False,
        "quality_safe_claim": False,
        "resource_fit_classification": resource_fit,
        "requested": {
            "workload_profile": workload_profile,
            "width": int(width),
            "height": int(height),
            "length": int(length),
            "pixels": pixels,
            "model_family": model_family,
            "model_precision": model_precision,
            "attention_backend": attention_backend,
            "cache_backend": cache_backend,
            "decode_mode": decode_mode,
            "h3_internal_spatial_tiling_expected": internal_spatial_tiling_expected,
            "dynamic_vram_mode": dynamic_vram_mode,
            "resolved_dynamic_vram_mode": resolved_dynamic,
            "reference_media_count": int(reference_media_count),
            "middle_keyframe_count": int(middle_keyframe_count),
            "minimum_current_headroom_mib": float(minimum_current_headroom_mib),
        },
        "estimated_packed_rows": rows,
        "environment": current,
        "observed_conditioning_reference_kinds": conditioning.get("reference_kinds", []),
        "issues": {
            "hard": hard,
            "high_risk": high_risk,
            "warnings": warnings,
            "unknown": unknown,
        },
        "scientific_boundaries": [
            "A current-state audit is not a denoising peak-memory predictor.",
            "Internal video-VAE temporal chunking is not equivalent to the later generic chunked-I/O interface.",
            "Current H3 regular decode also enters internal spatial tiling above 256 pixels; selecting regular is not a full-frame coordinate control.",
            "Unknown capability or patch ownership is never treated as compatible.",
            "Mechanical compatibility does not prove visual, audio, identity, or cache quality.",
            "A loaded-model snapshot is read-only and indicates present pressure, not future page-fault or peak-memory cost.",
            "Process read bytes and page faults are cumulative counters; classify thrashing only from before/after workload deltas.",
        ],
    }


def blocking_summary(report: Mapping[str, Any]) -> str:
    issues = report.get("issues", {})
    values: list[Mapping[str, Any]] = []
    if isinstance(issues, Mapping):
        for key in ("hard", "high_risk"):
            group = issues.get(key, [])
            if isinstance(group, Sequence):
                values.extend(item for item in group if isinstance(item, Mapping))
    return "; ".join(
        f"{item.get('code', 'unknown')}: {item.get('message', '')}" for item in values
    ) or "no known blocker"
