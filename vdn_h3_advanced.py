"""Comfy-native OpenVDN MiniMax H3 hybrid-attention integration.

The mathematical design is adapted from OpenVDN/vdn-minimax-h3 at commit
``b8cb28fbfca0266d1c7742a9f25ab8b58191de97`` (Apache-2.0).  The integration is
rewritten around ComfyUI's fused-QKV MiniMax H3 model, PackedLayout and ModelPatcher
lifecycle; it does not import or patch Diffusers and it never downloads at runtime.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import folder_paths
import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

import comfy.ldm.minimax.model as minimax_model
import comfy.lora
import comfy.lora_convert
import comfy.model_management
import comfy.ops
import comfy.patcher_extension
import comfy.utils
from comfy.model_patcher import ModelPatcher

from .h3_lora_compat_advanced import (
    build_minimax_h3_lora_key_map,
    convert_fastvideo_h3_adapter,
)
from .sampling import setup_dual_clock_sampling


HF_REPOSITORY = "OpenVDN/vdn-minimax-h3"
HF_REVISION = "18be6bcc4ee72585eee322ba28b5ccac2cf85ef0"
SOURCE_REVISION = "b8cb28fbfca0266d1c7742a9f25ab8b58191de97"
BASE_REVISION = "939557dc319dd91227e30195a763f272ba7f8765"
SOURCE_LICENSE = "Apache-2.0"
WEIGHT_LICENSE = "MiniMax H3 Community License Agreement"
WEIGHT_TERRITORY_NOTICE = (
    "Applicable Territory excludes the European Union, United Kingdom, "
    "Republic of Korea, and United States of America"
)

SCHEMA = "t8.minimax_h3.openvdn.v2"
ATTACHMENT_KEY = "t8_minimax_h3_openvdn_contract_v2"
ADDITIONAL_MODEL_KEY = "t8_minimax_h3_openvdn_branch_v1"
LAYOUT_KEY = "t8_minimax_h3_openvdn_layout_v2"
WRAPPER_KEY = "t8_minimax_h3_openvdn_layout_wrapper_v2"

CONDITION_SEGMENT_KINDS = frozenset(
    {"cond", "cond_audio", "ref_img", "ref_audio"}
)

EXPECTED_ASSETS = {
    "linear_branch/model.safetensors": {
        "bytes": 4_279_428_112,
        "sha256": "dec6981c7874f5b3bc92d1a02e256b673a3b3499dc1a124714bb3b19da602855",
        "tensors": 800,
    },
    "adapters/default/adapter_model.safetensors": {
        "bytes": 334_026_912,
        "sha256": "58558fef506f88bb41649242de9b9b3a365da806b51b2e96afbbe1625222058a",
        "tensors": 416,
    },
    "adapters/turbo/adapter_model.safetensors": {
        "bytes": 851_452_696,
        "sha256": "24fc93c82fe84dc45d0627f4e72c637bc387d282ba18f60ed3b7f8c81089392c",
        "tensors": 726,
    },
}

STAGES = {
    "stage_dmd_8nfe": {
        "directory": "stage-dmd-step-250",
        "steps": 8,
        "adapters": ("default", "turbo"),
    },
    "stage_b_50nfe": {
        "directory": "stage-b-step-2000",
        "steps": 50,
        "adapters": ("default",),
    },
}

BRANCH_TENSOR_SUFFIXES = (
    "linear_attention.short_conv.k_sp.weight",
    "linear_attention.short_conv.k_tm.weight",
    "linear_attention.short_conv.v_sp.weight",
    "linear_attention.short_conv.v_tm.weight",
    "linear_attention.alpha.A_log",
    "linear_attention.alpha.dt_bias",
    "linear_attention.alpha.down.weight",
    "linear_attention.alpha.up.weight",
    "linear_attention.beta_proj.weight",
    "linear_attention.output_gate.down.weight",
    "linear_attention.output_gate.up.weight",
    "linear_attention.output_gate.up.bias",
    "linear_attention.norm.weight",
    "softmax_gate.up.weight",
    "softmax_gate.up.bias",
    "to_out_linear.weight",
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def available_vdn_roots() -> list[str]:
    results: set[str] = set()
    for base in folder_paths.get_folder_paths("diffusion_models"):
        root = Path(base)
        if not root.is_dir():
            continue
        for spec in root.glob("**/stage-dmd-step-250/model_spec.json"):
            results.add(spec.parent.parent.relative_to(root).as_posix())
    return sorted(results) or ["OpenVDN/vdn-minimax-h3"]


def resolve_vdn_root(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute() and candidate.is_dir():
        return candidate.resolve()
    for base in folder_paths.get_folder_paths("diffusion_models"):
        joined = Path(base) / candidate
        if joined.is_dir():
            return joined.resolve()
    searched = [
        str(Path(base) / candidate)
        for base in folder_paths.get_folder_paths("diffusion_models")
    ]
    raise FileNotFoundError(
        f"OpenVDN root {str(value)!r} was not found under diffusion_models: {searched}"
    )


@functools.lru_cache(maxsize=32)
def _sha256_cached(path_text: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    stat = path.stat()
    return _sha256_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _stage_assets(root: Path, stage: str) -> dict[str, Path]:
    if stage not in STAGES:
        raise ValueError(f"unknown OpenVDN stage: {stage}")
    stage_dir = root / STAGES[stage]["directory"]
    dmd_dir = root / STAGES["stage_dmd_8nfe"]["directory"]
    # Stage B and DMD publish byte-identical branch/default files.  The local package
    # deliberately stores only one physical copy and reuses it for the 50-NFE route.
    branch = stage_dir / "linear_branch" / "model.safetensors"
    default = stage_dir / "adapters" / "default" / "adapter_model.safetensors"
    if not branch.is_file():
        branch = dmd_dir / "linear_branch" / "model.safetensors"
    if not default.is_file():
        default = dmd_dir / "adapters" / "default" / "adapter_model.safetensors"
    return {
        "stage_dir": stage_dir,
        "model_spec": stage_dir / "model_spec.json",
        "metadata": stage_dir / "metadata.json",
        "linear_branch": branch,
        "default_adapter": default,
        "turbo_adapter": dmd_dir / "adapters" / "turbo" / "adapter_model.safetensors",
    }


def _safetensor_manifest(path: Path) -> dict[str, Any]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        dtypes = sorted({str(handle.get_slice(key).get_dtype()) for key in keys})
    return {"tensor_count": len(keys), "dtypes": dtypes}


def _asset_report(
    root: Path, stage: str, verify_hashes: bool
) -> tuple[dict[str, Any], list[str]]:
    assets = _stage_assets(root, stage)
    errors: list[str] = []
    report: dict[str, Any] = {}
    selected = [
        "linear_branch/model.safetensors",
        "adapters/default/adapter_model.safetensors",
    ]
    if stage == "stage_dmd_8nfe":
        selected.append("adapters/turbo/adapter_model.safetensors")
    paths = {
        "linear_branch/model.safetensors": assets["linear_branch"],
        "adapters/default/adapter_model.safetensors": assets["default_adapter"],
        "adapters/turbo/adapter_model.safetensors": assets["turbo_adapter"],
    }
    for rel in selected:
        path = paths[rel]
        expected = EXPECTED_ASSETS[rel]
        item: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if not path.is_file():
            errors.append(f"missing {rel}: {path}")
            report[rel] = item
            continue
        item["bytes"] = path.stat().st_size
        if item["bytes"] != expected["bytes"]:
            errors.append(f"{rel} size {item['bytes']} != pinned {expected['bytes']}")
        try:
            item.update(_safetensor_manifest(path))
        except Exception as exc:  # noqa: BLE001 - audit must report corrupt headers
            errors.append(
                f"{rel} safetensors header failed: {type(exc).__name__}: {exc}"
            )
        if item.get("tensor_count") != expected["tensors"]:
            errors.append(
                f"{rel} tensors {item.get('tensor_count')} != pinned {expected['tensors']}"
            )
        if verify_hashes:
            item["sha256"] = file_sha256(path)
            item["sha256_match"] = item["sha256"] == expected["sha256"]
            if not item["sha256_match"]:
                errors.append(f"{rel} SHA-256 does not match pinned OpenVDN revision")
        report[rel] = item
    for name in ("model_spec", "metadata"):
        path = assets[name]
        if not path.is_file():
            errors.append(f"missing {name}: {path}")
    return report, errors


def _attention_conflicts(model) -> list[str]:
    conflicts: list[str] = []
    options = getattr(model, "model_options", {}).get("transformer_options", {})
    replacements = options.get("patches_replace", {}).get("dit", {})
    if replacements:
        conflicts.append("existing DiT block replacement")
    if "optimized_attention_override" in options:
        conflicts.append("optimized_attention_override")
    patches = options.get("patches", {})
    if patches.get("attn1_patch") or patches.get("attn1_output_patch"):
        conflicts.append("attention hook patch")
    if getattr(model, "patches", {}):
        conflicts.append("pre-existing MODEL weight patches/LoRA")
    if getattr(model, "get_attachment", lambda _key: None)(ATTACHMENT_KEY) is not None:
        conflicts.append("OpenVDN already attached")
    return conflicts


def _model_structure(model) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        diffusion = model.get_model_object("diffusion_model")
    except (AttributeError, KeyError) as exc:
        return {}, [f"MODEL does not expose diffusion_model: {exc}"]
    blocks = list(getattr(diffusion, "blocks", ()))
    first = getattr(blocks[0], "attn", None) if blocks else None
    report = {
        "class": diffusion.__class__.__name__,
        "blocks": len(blocks),
        "hidden_size": int(getattr(diffusion, "hidden_size", 0)),
        "heads": int(getattr(first, "heads", 0)),
        "head_dim": int(getattr(first, "head_dim", 0)),
        "video_latent_channels": int(getattr(diffusion, "latents_dim", 0)),
        "audio_latent_channels": int(getattr(diffusion, "audio_latents_dim", 0)),
        "patch_size": list(getattr(diffusion, "patch_size", ())),
        "adaln_curve_basis": bool(getattr(diffusion, "use_adaln_curves", False)),
        "adaln_input_dim": int(
            getattr(
                getattr(getattr(blocks[0], "adaln_proj", None), "linear", None),
                "in_features",
                0,
            )
        )
        if blocks
        else 0,
    }
    expected = {
        "class": "MiniMaxH3Model",
        "blocks": 50,
        "hidden_size": 5376,
        "heads": 56,
        "head_dim": 128,
        "video_latent_channels": 24,
        "audio_latent_channels": 32,
        "patch_size": [1, 2, 2],
    }
    for key, value in expected.items():
        if report.get(key) != value:
            errors.append(f"MODEL {key}={report.get(key)!r}, expected {value!r}")
    return report, errors


def audit_vdn_runtime(
    model,
    vdn_root: str | Path,
    stage: str,
    verify_hashes: bool = True,
    allow_structural_base: bool = False,
) -> tuple[bool, dict[str, Any]]:
    root = resolve_vdn_root(vdn_root)
    assets, errors = _asset_report(root, stage, bool(verify_hashes))
    structure, structure_errors = _model_structure(model)
    errors.extend(structure_errors)
    if stage == "stage_dmd_8nfe" and (
        structure.get("adaln_curve_basis")
        or structure.get("adaln_input_dim") != 2688
    ):
        errors.append(
            "OpenVDN DMD turbo owns 51 full-width AdaLN LoRA targets and requires "
            "a native H3 base with adaln_input_dim=2688; compressed adaln_t_table "
            "curve-basis/pruned checkpoints are incompatible and would silently skip "
            "those patches"
        )
    conflicts = _attention_conflicts(model)
    errors.extend(f"composition conflict: {item}" for item in conflicts)

    try:
        spec = json.loads(
            _stage_assets(root, stage)["model_spec"].read_text(encoding="utf-8")
        )
    except Exception as exc:  # noqa: BLE001
        spec = {}
        errors.append(f"model_spec parse failed: {type(exc).__name__}: {exc}")
    base = spec.get("base", {}) if isinstance(spec, dict) else {}
    base_revision_matches = base.get("revision") == BASE_REVISION
    if not base_revision_matches:
        errors.append(
            "OpenVDN model_spec base revision differs from the pinned release"
        )
    provenance = getattr(model, "get_attachment", lambda _key: None)(
        "t8_minimax_h3_base_provenance"
    )
    exact_base = (
        isinstance(provenance, dict) and provenance.get("revision") == BASE_REVISION
    )
    warnings: list[str] = []
    if not exact_base:
        message = (
            "selected Comfy H3 base is structurally compatible but its exact upstream "
            f"revision {BASE_REVISION} is not proven"
        )
        if allow_structural_base:
            warnings.append(message)
        else:
            errors.append(
                message
                + "; enable allow_structural_base only after accepting the reported "
                "provenance boundary"
            )

    license_paths = {
        "notice": root / "NOTICE",
        "source_license": root / "LICENSE",
        "weight_license": (
            root / "licenses" / "MiniMax-H3-Community-License-Agreement.txt"
        ),
    }
    missing_license_files = [
        name for name, path in license_paths.items() if not path.is_file()
    ]
    if missing_license_files:
        warnings.append(
            "local OpenVDN license documents are incomplete: "
            + ", ".join(missing_license_files)
            + "; review the model repository terms before use"
        )

    cuda = torch.cuda.is_available()
    capability = torch.cuda.get_device_capability(0) if cuda else None
    backend = "sdpa_grouped"
    report = {
        "schema": SCHEMA,
        "status": "ready" if not errors else "blocked",
        "root": str(root),
        "stage": stage,
        "hf_repository": HF_REPOSITORY,
        "hf_revision": HF_REVISION,
        "source_revision": SOURCE_REVISION,
        "base_revision": BASE_REVISION,
        "assets": assets,
        "model": structure,
        "base_provenance_exact": exact_base,
        "allow_structural_base": bool(allow_structural_base),
        "runtime": {
            "torch": torch.__version__,
            "cuda_available": cuda,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if cuda else "cpu",
            "compute_capability": list(capability) if capability else None,
            "resolved_backend": backend,
            "global_dependency_upgrade": False,
        },
        "scope": "all_native_h3_packed_layouts",
        "supported_tasks": [
            "T2VA",
            "I2VA",
            "FL2VA",
            "L2VA",
            "Ref2VA",
            "Hybrid",
        ],
        "upstream_declared_scope": "T2VA",
        "t8_extension_validation": "real multimodal validation required per route",
        "license": {
            "source": SOURCE_LICENSE,
            "weights": WEIGHT_LICENSE,
            "applicable_territory_notice": WEIGHT_TERRITORY_NOTICE,
            "local_files": {
                name: {"path": str(path), "present": path.is_file()}
                for name, path in license_paths.items()
            },
            "user_must_review_terms": True,
        },
        "warnings": warnings,
        "errors": errors,
    }
    return not errors, report


@dataclass(frozen=True)
class VDNSequenceLayout:
    seq_len: int
    video_start: int
    num_frames: int
    tokens_per_frame: int
    frame_height: int
    frame_width: int
    text_start: int
    text_len: int
    condition_kinds: tuple[str, ...] = ()

    @property
    def video_end(self) -> int:
        return self.video_start + self.num_frames * self.tokens_per_frame

    def global_index(self, device) -> torch.Tensor:
        index = torch.arange(self.seq_len, device=device)
        return torch.cat((index[: self.video_start], index[self.video_end :]))


def layout_from_packed(layout) -> VDNSequenceLayout:
    segments = list(getattr(layout, "segments", ()))
    kinds = [str(segment[2]) for segment in segments]
    middle_kinds = kinds[1:-2]
    if (
        len(kinds) < 3
        or kinds[0] != "text"
        or kinds[-2:] != ["audio", "video"]
        or any(kind not in CONDITION_SEGMENT_KINDS for kind in middle_kinds)
    ):
        raise RuntimeError(
            "OpenVDN T8 requires native H3 [text | optional cond/ref rows | "
            "audio | video] PackedLayout; received "
            + ",".join(kinds)
        )
    signature = tuple(getattr(layout, "signature", ()))
    if len(signature) != 5:
        raise RuntimeError(
            "OpenVDN requires the native MiniMax H3 PackedLayout signature"
        )
    text_len, latent_t, latent_h, latent_w, _audio_t = map(int, signature)
    text_start, text_end, _ = segments[0]
    for index, (start, stop, _kind) in enumerate(segments):
        if stop <= start:
            raise RuntimeError("OpenVDN PackedLayout contains an empty segment")
        if index and start != segments[index - 1][1]:
            raise RuntimeError("OpenVDN PackedLayout segments are not contiguous")
    video_start, video_end, _ = segments[-1]
    frame_height, frame_width = latent_h // 2, latent_w // 2
    tokens_per_frame = frame_height * frame_width
    if text_start != 0 or text_end - text_start != text_len:
        raise RuntimeError(
            "OpenVDN H3 text rows are not the expected contiguous prefix"
        )
    if video_end - video_start != latent_t * tokens_per_frame:
        raise RuntimeError("OpenVDN H3 video rows do not match PackedLayout geometry")
    if int(getattr(layout, "seq_len", -1)) != video_end:
        raise RuntimeError("OpenVDN H3 target video must be the final packed segment")
    return VDNSequenceLayout(
        seq_len=video_end,
        video_start=video_start,
        num_frames=latent_t,
        tokens_per_frame=tokens_per_frame,
        frame_height=frame_height,
        frame_width=frame_width,
        text_start=text_start,
        text_len=text_len,
        condition_kinds=tuple(middle_kinds),
    )


def window_bounds(
    num_frames: int, radius: int = 1, chunk: int = 5
) -> list[tuple[int, int]]:
    if chunk <= 0:
        return [(frame - radius, frame + radius) for frame in range(num_frames)]
    return [
        (
            ((frame // chunk) - radius) * chunk,
            ((frame // chunk) + radius + 1) * chunk - 1,
        )
        for frame in range(num_frames)
    ]


def _sdpa_rows(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float
) -> torch.Tensor:
    out = F.scaled_dot_product_attention(
        q.permute(1, 0, 2).unsqueeze(0),
        k.permute(1, 0, 2).unsqueeze(0),
        v.permute(1, 0, 2).unsqueeze(0),
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    return out.squeeze(0).permute(1, 0, 2)


def window_softmax_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: VDNSequenceLayout,
    bounds: list[tuple[int, int]],
    scale: float,
) -> torch.Tensor:
    """Exact c1/r1/both-anchor mask as grouped native SDPA calls.

    Consecutive frames in one VAE chunk share the same key set, so they are evaluated
    together.  This is the Ada/Windows-safe backend: no FlexAttention, FA4, Triton or
    patched Diffusers dependency is required.
    """

    vs, ve = layout.video_start, layout.video_end
    frames, per_frame = layout.num_frames, layout.tokens_per_frame
    global_idx = layout.global_index(query.device)
    out = torch.empty_like(query)
    # anchor rows and all global rows remain full attention.
    dense_idx = torch.cat(
        (
            global_idx,
            torch.arange(vs, vs + per_frame, device=query.device),
            torch.arange(ve - per_frame, ve, device=query.device),
        )
    ).unique(sorted=True)
    out[dense_idx] = _sdpa_rows(query[dense_idx], key, value, scale)

    frame_key = key[vs:ve].view(frames, per_frame, key.shape[1], key.shape[2])
    frame_value = value[vs:ve].view(frames, per_frame, value.shape[1], value.shape[2])
    frame = 1
    while frame < frames - 1:
        group = [frame]
        while group[-1] + 1 < frames - 1 and bounds[group[-1] + 1] == bounds[frame]:
            group.append(group[-1] + 1)
        lo, hi = bounds[frame]
        selected_frames = set(range(max(lo, 0), min(hi, frames - 1) + 1))
        selected_frames.update((0, frames - 1))
        frame_ids = sorted(selected_frames)
        key_rows = torch.cat(
            (
                key[global_idx],
                frame_key[frame_ids].reshape(-1, key.shape[1], key.shape[2]),
            )
        )
        value_rows = torch.cat(
            (
                value[global_idx],
                frame_value[frame_ids].reshape(-1, value.shape[1], value.shape[2]),
            )
        )
        qa = vs + group[0] * per_frame
        qb = vs + (group[-1] + 1) * per_frame
        out[qa:qb] = _sdpa_rows(query[qa:qb], key_rows, value_rows, scale)
        frame = group[-1] + 1
    return out


class OutputGate(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        bottleneck: int | None = None,
        *,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        out_features = num_heads * (head_dim or 1)
        self.down = (
            None
            if bottleneck is None
            else comfy.ops.manual_cast.Linear(
                hidden_size, bottleneck, bias=False, device=device, dtype=dtype
            )
        )
        self.up = comfy.ops.manual_cast.Linear(
            bottleneck or hidden_size,
            out_features,
            bias=True,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.up(x if self.down is None else self.down(x)))
        return gate.view(-1, self.num_heads, self.head_dim or 1)


class BranchRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, *, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        moment = (
            torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32).pow(
                2
            )
            / x.shape[-1]
        )
        weight = comfy.model_management.cast_to(
            self.weight, device=x.device, dtype=x.dtype
        )
        return x * torch.rsqrt(moment + self.eps).to(x.dtype) * weight


class FrameKDAAlpha(nn.Module):
    def __init__(
        self, hidden: int, heads: int, head_dim: int, *, device=None, dtype=None
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.down = comfy.ops.manual_cast.Linear(
            hidden, head_dim, bias=False, device=device, dtype=dtype
        )
        self.up = comfy.ops.manual_cast.Linear(
            head_dim, heads * head_dim, bias=False, device=device, dtype=dtype
        )
        self.A_log = nn.Parameter(torch.empty(heads, device=device, dtype=dtype))
        self.dt_bias = nn.Parameter(
            torch.empty(heads * head_dim, device=device, dtype=dtype)
        )

    def forward(self, frame_mean: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=frame_mean.device.type, enabled=False):
            delta = self.down(frame_mean.float())
            delta = self.up(delta)
            dt_bias = comfy.model_management.cast_to(
                self.dt_bias, device=delta.device, dtype=torch.float32
            )
            a_log = comfy.model_management.cast_to(
                self.A_log, device=delta.device, dtype=torch.float32
            )
            delta = delta + dt_bias
            scale = torch.exp(a_log)[:, None]
            return torch.exp(
                -scale * F.softplus(delta.view(-1, self.heads, self.head_dim))
            )


class LinearAttentionSepConv(nn.Module):
    KERNEL = 5

    def __init__(self, channels: int, *, device=None, dtype=None):
        super().__init__()
        for name in ("k", "v"):
            setattr(
                self,
                f"{name}_sp",
                comfy.ops.manual_cast.Conv2d(
                    channels,
                    channels,
                    self.KERNEL,
                    padding=2,
                    groups=channels,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
            )
            setattr(
                self,
                f"{name}_tm",
                comfy.ops.manual_cast.Conv1d(
                    channels,
                    channels,
                    self.KERNEL,
                    padding=2,
                    groups=channels,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
            )

    def apply(
        self,
        name: str,
        tokens: torch.Tensor,
        frames: int,
        frame_size: tuple[int, int],
    ) -> torch.Tensor:
        if name == "q":
            return tokens
        heads, head_dim = tokens.shape[-2:]
        height, width = frame_size
        channels = heads * head_dim
        volume = tokens.reshape(frames, height, width, channels).permute(0, 3, 1, 2)
        spatial = getattr(self, f"{name}_sp")
        volume = spatial(volume)
        x = volume.permute(0, 2, 3, 1).reshape(frames, height * width, channels)
        weights = comfy.model_management.cast_to(
            getattr(self, f"{name}_tm").weight,
            device=x.device,
            dtype=x.dtype,
        ).squeeze(1)
        padded = F.pad(x, (0, 0, 0, 0, 2, 2))
        result = None
        for tap in range(self.KERNEL):
            part = padded[tap : tap + frames] * weights[:, tap].view(1, 1, -1)
            result = part if result is None else result + part
        assert result is not None
        return result.reshape(-1, heads, head_dim)


class BidirectionalLinearBranch(nn.Module):
    TEXT_STATE_SCALE = 0.5

    def __init__(
        self, hidden: int, heads: int, head_dim: int, *, device=None, dtype=None
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.short_conv = LinearAttentionSepConv(
            heads * head_dim, device=device, dtype=dtype
        )
        self.alpha = FrameKDAAlpha(hidden, heads, head_dim, device=device, dtype=dtype)
        self.beta_proj = comfy.ops.manual_cast.Linear(
            hidden, heads, bias=False, device=device, dtype=dtype
        )
        self.output_gate = OutputGate(
            hidden,
            heads,
            head_dim,
            bottleneck=head_dim,
            device=device,
            dtype=dtype,
        )
        self.norm = BranchRMSNorm(head_dim, device=device, dtype=dtype)

    @staticmethod
    def _activate(value: torch.Tensor, normalize: bool) -> torch.Tensor:
        result = F.silu(value)
        return (
            F.normalize(result, dim=-1, eps=1e-6).to(result.dtype)
            if normalize
            else result
        )

    def _features(
        self,
        qkv_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        frames: int,
        frame_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = []
        for name, value in zip(("q", "k", "v"), qkv_raw):
            value = self.short_conv.apply(name, value, frames, frame_size)
            output.append(self._activate(value, name != "v"))
        return tuple(output)  # type: ignore[return-value]

    def _text_state(
        self,
        text_x: torch.Tensor,
        text_qkv_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        length = text_x.shape[0]
        key = (
            self._activate(text_qkv_raw[1], True)
            .view(1, length, self.heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        value = (
            self._activate(text_qkv_raw[2], False)
            .view(1, length, self.heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        beta = (
            torch.sigmoid(self.beta_proj(text_x))
            .view(1, length, self.heads)
            .permute(0, 2, 1)
        )
        a, b = _frame_statistics(key, value, beta)
        _transition, injection = _vdn_factor(
            torch.ones(1, self.heads, self.head_dim, device=a.device), a, b
        )
        return self.TEXT_STATE_SCALE * injection[0]

    def forward(
        self,
        video_x: torch.Tensor,
        layout: VDNSequenceLayout,
        bounds: list[tuple[int, int]],
        video_qkv_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        text_x: torch.Tensor,
        text_qkv_raw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        frames, tokens = layout.num_frames, layout.tokens_per_frame
        if frames <= 2:
            return video_x.new_zeros(frames * tokens, self.heads * self.head_dim)
        # anchor_frames="both": remove the two exact-softmax anchor frames from the
        # recurrence and rebase the c1 bounds onto the interior sequence.
        inner = slice(tokens, (frames - 1) * tokens)
        xv = video_x[inner]
        qkv = tuple(value[inner] for value in video_qkv_raw)
        inner_frames = frames - 2
        inner_bounds = [(lo - 1, hi - 1) for lo, hi in bounds[1:-1]]
        query, key, value = self._features(
            qkv, inner_frames, (layout.frame_height, layout.frame_width)
        )
        shape = (inner_frames, tokens, self.heads, self.head_dim)
        query_by_frame = query.view(shape)
        key_by_frame = key.view(shape).permute(0, 2, 1, 3)
        value_by_frame = value.view(shape).permute(0, 2, 1, 3)
        beta = (
            torch.sigmoid(self.beta_proj(xv))
            .view(inner_frames, tokens, self.heads)
            .permute(0, 2, 1)
        )
        a, b = _frame_statistics(key_by_frame, value_by_frame, beta)
        alpha = self.alpha(
            xv.view(inner_frames, tokens, -1).mean(dim=1, dtype=torch.float32)
        )
        text_state = self._text_state(text_x, text_qkv_raw)
        prefix, suffix = _run_scans(alpha, a, b, text_state)
        state = _gather_linear_state(prefix, suffix, alpha, inner_bounds, text_state)
        state = state.to(xv.dtype)
        readout = torch.einsum("fhvk,fshk->fshv", state, query_by_frame)
        readout = self.norm(
            readout.reshape(inner_frames * tokens, self.heads, self.head_dim)
        )
        readout = (readout * self.output_gate(xv)).reshape(
            inner_frames * tokens, self.heads * self.head_dim
        )
        output = readout.new_zeros(frames * tokens, readout.shape[-1])
        output[inner] = readout
        return output


def _frame_statistics(
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type=key.device.type, enabled=False):
        key_bf = key.contiguous()
        key_fp = key_bf.float()
        scaled = (key_fp * beta.unsqueeze(-1).float()).contiguous()
        a = torch.matmul(scaled.transpose(-1, -2), key_fp)
        a = 0.5 * (a + a.transpose(-1, -2))
        vb = (value * beta.unsqueeze(-1).to(value.dtype)).contiguous()
        b = torch.matmul(vb.transpose(-1, -2), key_bf).float()
        return a, b


def _vdn_factor(
    alpha: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    a32 = a.float()
    eye = torch.eye(a32.shape[-1], device=a32.device, dtype=torch.float32).expand_as(
        a32
    )
    chol = torch.linalg.cholesky(a32 + eye)
    inverse_l = torch.linalg.solve_triangular(chol, eye, upper=False, left=True)
    inverse = inverse_l.transpose(-1, -2) @ inverse_l
    transition = alpha.unsqueeze(-1) * inverse
    injection = b.float() @ inverse
    return transition, injection


def _run_scans(
    alpha: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    text_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type=a.device.type, enabled=False):
        transitions, injections = _vdn_factor(alpha, a, b)
        frames = transitions.shape[0]
        start = text_state.to(injections.dtype)
        prefix = torch.empty(
            (frames, *start.shape), device=a.device, dtype=injections.dtype
        )
        suffix = torch.empty_like(prefix)
        state = start
        for frame in range(frames):
            torch.baddbmm(
                injections[frame], state, transitions[frame], out=prefix[frame]
            )
            state = prefix[frame]
        state = start
        for frame in range(frames - 1, -1, -1):
            torch.baddbmm(
                injections[frame], state, transitions[frame], out=suffix[frame]
            )
            state = suffix[frame]
        return prefix, suffix


def _gather_linear_state(
    prefix: torch.Tensor,
    suffix: torch.Tensor,
    alpha: torch.Tensor,
    bounds: list[tuple[int, int]],
    text_state: torch.Tensor,
) -> torch.Tensor:
    frames = prefix.shape[0]
    device = prefix.device
    before_raw = torch.tensor([lo for lo, _ in bounds], device=device) - 1
    after_raw = torch.tensor([hi for _, hi in bounds], device=device) + 1
    has_before = before_raw >= 0
    has_after = after_raw < frames
    before = prefix[before_raw.clamp(min=0)]
    after = suffix[after_raw.clamp(max=frames - 1)]
    text = text_state.to(before.dtype)
    before = torch.where(has_before.view(-1, 1, 1, 1), before, text)
    after = torch.where(has_after.view(-1, 1, 1, 1), after, text)
    log_alpha = torch.log(alpha.clamp_min(1e-12))
    prefix_log = torch.cat((torch.zeros_like(log_alpha[:1]), log_alpha.cumsum(0)))
    frame_ids = torch.arange(frames, device=device)
    bridge_before = (before_raw + 1).clamp(min=0)
    bridge_after = after_raw.clamp(max=frames)
    before_scale = torch.exp(prefix_log[frame_ids + 1] - prefix_log[bridge_before])
    after_scale = torch.exp(prefix_log[bridge_after] - prefix_log[frame_ids])
    return before * before_scale.unsqueeze(2) + after * after_scale.unsqueeze(2)


class VDNBlockBranch(nn.Module):
    def __init__(self, *, device="meta", dtype=torch.bfloat16):
        super().__init__()
        self.linear_attention = BidirectionalLinearBranch(
            5376, 56, 128, device=device, dtype=dtype
        )
        self.softmax_gate = OutputGate(5376, 56, device=device, dtype=dtype)
        self.to_out_linear = comfy.ops.manual_cast.Linear(
            56 * 128, 5376, bias=False, device=device, dtype=dtype
        )


class VDNBranchModel(nn.Module):
    def __init__(self, blocks: int = 50, *, device="meta", dtype=torch.bfloat16):
        super().__init__()
        self.blocks = nn.ModuleList(
            [VDNBlockBranch(device=device, dtype=dtype) for _ in range(blocks)]
        )


def _branch_key(source: str) -> str:
    prefix = "transformer_blocks."
    marker = ".attn."
    if not source.startswith(prefix) or marker not in source:
        raise ValueError(f"unsupported OpenVDN branch key: {source}")
    block, rest = source[len(prefix) :].split(marker, 1)
    return f"blocks.{int(block)}.{rest}"


def load_branch_model(
    path: Path, load_device, offload_device
) -> tuple[ModelPatcher, dict[str, Any]]:
    branch = VDNBranchModel(device="meta")
    # AIMDO deliberately leaves Linear.weight/bias registered as None until a
    # checkpoint is assigned.  Consequently state_dict() is incomplete in a live
    # low-VRAM server even though the module geometry is valid.  Build the immutable
    # v2 manifest from the published 50 x 16 architecture instead of introspecting
    # whether this process has AIMDO enabled.
    expected = {
        f"blocks.{block}.{suffix}"
        for block in range(50)
        for suffix in BRANCH_TENSOR_SUFFIXES
    }
    with safe_open(path, framework="pt", device="cpu") as handle:
        source_keys = list(handle.keys())
        mapped = {_branch_key(key): key for key in source_keys}
        missing = sorted(expected - set(mapped))
        unexpected = sorted(set(mapped) - expected)
        if missing or unexpected:
            raise ValueError(
                "OpenVDN branch manifest mismatch: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        for target, source in mapped.items():
            comfy.utils.set_attr_param(branch, target, handle.get_tensor(source))
    actual = set(branch.state_dict())
    if actual != expected:
        raise RuntimeError(
            "OpenVDN branch assignment did not materialize the exact manifest: "
            f"missing={sorted(expected - actual)[:8]}, "
            f"unexpected={sorted(actual - expected)[:8]}"
        )
    if any(parameter.is_meta for parameter in branch.parameters()):
        raise RuntimeError("OpenVDN branch still contains meta parameters after load")
    branch.eval().requires_grad_(False)
    patcher = ModelPatcher(
        branch,
        load_device=load_device,
        offload_device=offload_device,
        size=path.stat().st_size,
        weight_inplace_update=False,
    )
    return patcher, {
        "tensor_count": len(source_keys),
        "missing": 0,
        "unexpected": 0,
        "bytes": path.stat().st_size,
        "path": str(path),
    }


def _normalize_adapter_state(
    state: dict[str, torch.Tensor], adapter_name: str
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        normalized = key.replace(".attn.orig.", ".attn.")
        normalized = normalized.replace(
            f".lora_A.{adapter_name}.weight", ".lora_A.weight"
        ).replace(f".lora_B.{adapter_name}.weight", ".lora_B.weight")
        if normalized == key and f".{adapter_name}." in key:
            raise ValueError(f"unsupported named OpenVDN adapter key: {key}")
        if normalized in result:
            raise ValueError(f"duplicate normalized OpenVDN adapter key: {normalized}")
        result[normalized] = value
    return result


def _load_adapter_patches(model, path: Path, adapter_name: str):
    raw = comfy.utils.load_torch_file(str(path), safe_load=True)
    normalized = _normalize_adapter_state(raw, adapter_name)
    converted, conversion = convert_fastvideo_h3_adapter(normalized)
    key_map, _aliases = build_minimax_h3_lora_key_map(model.model)
    shape_report = _validate_adapter_target_shapes(
        converted, key_map, model.model.state_dict()
    )
    converted = comfy.lora_convert.convert_lora(converted)
    patches = comfy.lora.load_lora(converted, key_map, log_missing=False)
    return raw, converted, patches, conversion, shape_report


def _validate_adapter_target_shapes(
    converted: dict[str, torch.Tensor],
    key_map: dict[str, str],
    target_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Fail before sampling when a converted LoRA cannot fit the selected base.

    Comfy registers a patch by key even when its dense delta cannot be reshaped onto
    that parameter.  Curve-basis MiniMax H3 checkpoints expose 8-column AdaLN
    weights, while OpenVDN's DMD turbo adapter was trained against the original
    2688-column time embedding.  Without this check Comfy logs one error per AdaLN
    target per denoising step and continues with those 51 patches omitted.
    """

    checked: list[dict[str, Any]] = []
    for a_key in sorted(converted):
        suffix = ".lora_A.weight"
        if not a_key.endswith(suffix):
            continue
        module = a_key[: -len(suffix)]
        b_key = f"{module}.lora_B.weight"
        if b_key not in converted:
            raise ValueError(f"OpenVDN converted LoRA has no B tensor: {a_key}")
        target_key = key_map.get(module)
        if target_key is None or target_key not in target_state:
            raise ValueError(f"OpenVDN converted LoRA target is unavailable: {module}")
        a = converted[a_key]
        b = converted[b_key]
        if a.ndim != 2 or b.ndim != 2 or int(b.shape[1]) != int(a.shape[0]):
            raise ValueError(
                f"OpenVDN converted LoRA factors are invalid for {module}: "
                f"A={tuple(a.shape)}, B={tuple(b.shape)}"
            )
        target_shape = tuple(int(value) for value in target_state[target_key].shape)
        delta_shape = (int(b.shape[0]), int(a.shape[1]))
        if len(target_shape) != 2 or delta_shape != target_shape:
            raise ValueError(
                f"OpenVDN LoRA shape mismatch for {target_key}: "
                f"delta={delta_shape}, selected base={target_shape}. "
                "DMD requires a full-width 2688-column H3 AdaLN base; do not use "
                "an adaln_t_table curve-basis/pruned checkpoint."
            )
        checked.append(
            {
                "module": module,
                "target": target_key,
                "shape": list(target_shape),
            }
        )
    return {
        "checked_targets": len(checked),
        "adaln_targets": sum("adaln_proj.linear" in item["module"] for item in checked),
        "all_shapes_exact": True,
    }


def _qkv(
    attention,
    x: torch.Tensor,
    rope_freqs,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    sequence = x.shape[0]
    q, k, v = attention.qkv_proj(x).split(attention.heads * attention.head_dim, dim=-1)
    q_raw = q.view(sequence, attention.heads, attention.head_dim).clone()
    k_raw = k.view(sequence, attention.heads, attention.head_dim).clone()
    v_raw = v.view(sequence, attention.heads, attention.head_dim)
    if rope_freqs is not None:
        q = q.view(1, sequence, attention.heads, attention.head_dim)
        k = k.view(1, sequence, attention.heads, attention.head_dim)
        qw = comfy.model_management.cast_to(attention.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(attention.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = minimax_model.comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=attention.q_norm.eps, rot_dim=rot
            )
        else:
            minimax_model.comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=attention.q_norm.eps, rot_dim=rot
            )
        q, k = q[0], k[0]
    else:
        q = attention.q_norm(q.view(sequence, attention.heads, attention.head_dim))
        k = attention.k_norm(k.view(sequence, attention.heads, attention.head_dim))
    return q, k, v_raw.clone(), (q_raw, k_raw, v_raw)


def _vdn_attention(
    block,
    branch: VDNBlockBranch,
    x: torch.Tensor,
    rope_freqs,
    layout: VDNSequenceLayout,
) -> torch.Tensor:
    query, key, value, qkv_raw = _qkv(block.attn, x, rope_freqs)
    bounds = window_bounds(layout.num_frames, radius=1, chunk=5)
    full_cover = all(lo <= 0 and hi >= layout.num_frames - 1 for lo, hi in bounds)
    softmax = window_softmax_sdpa(
        query,
        key,
        value,
        layout,
        bounds,
        block.attn.head_dim**-0.5,
    )
    softmax = softmax * branch.softmax_gate(x)
    output = block.attn.out_proj(softmax.reshape(x.shape[0], -1).type_as(x))
    del query, key, value, softmax

    # The published implementation switches the linear branch off when the c1
    # window already covers the complete clip.  Nothing is outside the window in
    # that case, so adding even the text-seeded recurrence would double-count it.
    if full_cover:
        return output

    vs, ve = layout.video_start, layout.video_end
    ts, te = layout.text_start, layout.text_start + layout.text_len
    linear = branch.linear_attention(
        x[vs:ve],
        layout,
        bounds,
        tuple(value[vs:ve] for value in qkv_raw),
        x[ts:te],
        tuple(value[ts:te] for value in qkv_raw),
    )
    output[vs:ve].add_(branch.to_out_linear(linear.type_as(x)))
    return output


def _vdn_block(block, branch: VDNBlockBranch, args, original):
    layout = args["transformer_options"].get(LAYOUT_KEY)
    if not isinstance(layout, VDNSequenceLayout):
        raise RuntimeError("OpenVDN execution has no validated native H3 PackedLayout")
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(
        args["t_emb"]
    )
    hidden = minimax_model._mod_scale_shift(
        block.norm1(args["img"]), shift_msa, scale_msa, args["mod_segments"]
    )
    image = minimax_model._mod_gate(
        args["img"],
        gate_msa,
        _vdn_attention(block, branch, hidden, args["rope_freqs"], layout),
        args["mod_segments"],
    )
    hidden = minimax_model._mod_scale_shift(
        block.norm2(image), shift_mlp, scale_mlp, args["mod_segments"]
    )
    return {
        "img": minimax_model._mod_gate(
            image, gate_mlp, block.mlp(hidden), args["mod_segments"]
        )
    }


def _layout_wrapper(executor, *args, **kwargs):
    options = kwargs.get("transformer_options")
    if not isinstance(options, dict) and len(args) >= 4 and isinstance(args[3], dict):
        options = args[3]
    payload = kwargs.get("minimax_payload")
    if not isinstance(payload, dict):
        raise RuntimeError("OpenVDN requires native MiniMax H3 minimax_payload")
    packed = payload.get("layout")
    if packed is None:
        raise RuntimeError("OpenVDN requires Conditioning to provide a PackedLayout")
    if not isinstance(options, dict):
        raise RuntimeError("OpenVDN could not access transformer_options")
    options[LAYOUT_KEY] = layout_from_packed(packed)
    return executor(*args, **kwargs)


def compose_vdn_model(
    model,
    vdn_root: str | Path,
    stage: str,
    *,
    verify_hashes: bool = True,
    allow_structural_base: bool = False,
):
    ready, audit = audit_vdn_runtime(
        model,
        vdn_root,
        stage,
        verify_hashes=verify_hashes,
        allow_structural_base=allow_structural_base,
    )
    if not ready:
        raise RuntimeError(
            "OpenVDN audit blocked composition:\n" + "\n".join(audit["errors"])
        )
    root = Path(audit["root"])
    assets = _stage_assets(root, stage)
    diffusion = model.get_model_object("diffusion_model")
    blocks = list(diffusion.blocks)

    branch_patcher, branch_report = load_branch_model(
        assets["linear_branch"], model.load_device, model.offload_device
    )
    patched = model.clone()
    adapter_reports: list[dict[str, Any]] = []
    for adapter_name in STAGES[stage]["adapters"]:
        path = assets[f"{adapter_name}_adapter"]
        raw, converted, patches, conversion, shape_report = _load_adapter_patches(
            patched, path, adapter_name
        )
        applied = set(patched.add_patches(patches, 1.0))
        if applied != set(patches):
            missing = sorted(set(patches) - applied)
            raise RuntimeError(
                f"OpenVDN {adapter_name} adapter did not fully apply: {missing[:12]}"
            )
        adapter_reports.append(
            {
                "name": adapter_name,
                "path": str(path),
                "input_tensors": len(raw),
                "converted_tensors": len(converted),
                "patch_targets": len(patches),
                "applied_targets": len(applied),
                "conversion": conversion,
                "shape_validation": shape_report,
                "strength": 1.0,
            }
        )

    for index, block in enumerate(blocks):
        branch = branch_patcher.model.blocks[index]

        def hook(args, original, _block=block, _branch=branch):
            return _vdn_block(_block, _branch, args, original["original_block"])

        patched.set_model_patch_replace(hook, "dit", "double_block", index)
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        WRAPPER_KEY,
        _layout_wrapper,
    )
    patched.set_additional_models(ADDITIONAL_MODEL_KEY, [branch_patcher])
    receipt = {
        "schema": SCHEMA,
        "status": "configured",
        "stage": stage,
        "steps": STAGES[stage]["steps"],
        "video_shift": 12.0,
        "audio_shift": 3.0,
        "backend": "sdpa_grouped",
        "task_scope": "all_native_h3_packed_layouts",
        "supported_tasks": [
            "T2VA",
            "I2VA",
            "FL2VA",
            "L2VA",
            "Ref2VA",
            "Hybrid",
        ],
        "upstream_declared_scope": "T2VA",
        "multimodal_support_owner": "T8 real-validation extension",
        "hf_revision": HF_REVISION,
        "source_revision": SOURCE_REVISION,
        "base_revision": BASE_REVISION,
        "base_provenance_exact": audit["base_provenance_exact"],
        "allow_structural_base": bool(allow_structural_base),
        "branch": branch_report,
        "adapters": adapter_reports,
        "main_block_count": len(blocks),
        "additional_model_lifecycle": True,
        "conflicts": "SLA/VSA/FastH3/attention overrides/pre-existing LoRA rejected",
        "runtime_downloads": False,
        "license": audit["license"],
    }
    patched.set_attachments(ATTACHMENT_KEY, receipt)
    return patched, _json(receipt)


def setup_vdn_execution(model, av_latent):
    receipt = getattr(model, "get_attachment", lambda _key: None)(ATTACHMENT_KEY)
    if not isinstance(receipt, dict) or receipt.get("status") != "configured":
        raise RuntimeError("OpenVDN Execution Plan requires the configured VDN MODEL")
    steps = int(receipt["steps"])
    planned, sampler, sigmas = setup_dual_clock_sampling(
        model,
        av_latent,
        steps,
        12.0,
        3.0,
        "euler",
        "native_flow",
    )
    report = dict(receipt)
    report.update(
        {
            "status": "execution_planned",
            "sampler": "euler",
            "scheduler": "native_flow",
            "nfe": steps,
            "sigma_count": int(sigmas.numel()),
        }
    )
    return planned, sampler, sigmas, _json(report)
