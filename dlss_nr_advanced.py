from __future__ import annotations

import hashlib
from fractions import Fraction
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import queue
import re
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Mapping
import uuid
import zipfile

import torch
import torch.nn.functional as torch_functional


RUNTIME_MANIFEST_SCHEMA = "t8.dlss_nr.runtime_manifest.v1"
RUNTIME_MANIFEST_NAME = "t8-runtime-manifest.json"
V12_NATIVE_SOURCE_AUDIT_COMMIT = "e1946117699c4e6dcd531f5e042401d04268320e"
V13_NATIVE_SOURCE_AUDIT_COMMIT = "55a4ceb588a419b9b56497aa0b563d0c9e2b6c77"
NATIVE_SOURCE_AUDIT_COMMIT = V13_NATIVE_SOURCE_AUDIT_COMMIT
WRAPPER_SOURCE_AUDIT_COMMIT = "4d329a864e99267734fffa2ee4b7ddeafc005c4a"
V12_RELEASE_TAG_COMMIT = "1f49c3429e4a2f4dd62e28f09f5f21decb7bb38f"
V13_RELEASE_TAG_COMMIT = V13_NATIVE_SOURCE_AUDIT_COMMIT
MINIMUM_DRIVER_VERSION = (616, 56)
MINIMUM_FREE_VRAM_MIB = 512.0
PROBE_TIMEOUT_SECONDS = 60.0
PROBE_MEMORY_TOLERANCE_MIB = 512.0
PROBE_MODES = ("static_only", "feature_probe_1_frame")
RUNTIME_HANDLE_SCHEMA = "t8.dlss_nr.runtime_handle.v1"
PROCESSING_MODES = ("nr_only", "sr_only", "sr_nr")
SUPPORTED_SCALES = (1.0, 1.5, 2.0, 3.0)
MOTION_ENGINES = ("auto", "nvof", "lk")
SR_PRESETS = ("default", "E", "F", "J", "K", "L", "M")
QUALITY_PROFILE_NAMES = (
    "standard",
    "max_detail",
    "portrait",
    "night",
    "light",
    "custom",
)
MEDIA_TIMEOUT_SECONDS = 1800.0

# These named profiles reproduce the public ComfyUI-DLSS-NR wrapper's effective
# values.  The previous T8-only "conservative" profile combined the mildest style,
# half model intensity and half final composition, which hid the feature's effect.
QUALITY_PROFILES: dict[str, dict[str, Any]] = {
    "standard": {
        "nr_preset": 0,
        "nr_style": 0,
        "nr_intensity": 1.5,
        "nr_local_structure": 1.0,
        "nr_local_tone": 1.0,
        "nr_skin": -1.0,
        "nr_global_tone": -1.0,
        "nr_detail": 1.0,
        "nr_color": 1.0,
        "nr_ui_correction": False,
        "nr_auto_mask": False,
    },
    "max_detail": {
        "nr_preset": 0,
        "nr_style": 0,
        "nr_intensity": 2.0,
        "nr_local_structure": 1.0,
        "nr_local_tone": 1.0,
        "nr_skin": -1.0,
        "nr_global_tone": -1.0,
        "nr_detail": 1.0,
        "nr_color": 1.0,
        "nr_ui_correction": False,
        "nr_auto_mask": False,
    },
    "portrait": {
        "nr_preset": 0,
        "nr_style": 1,
        "nr_intensity": 1.2,
        "nr_local_structure": 1.0,
        "nr_local_tone": 1.0,
        "nr_skin": 1.0,
        "nr_global_tone": -1.0,
        "nr_detail": 0.9,
        "nr_color": 0.8,
        "nr_ui_correction": False,
        "nr_auto_mask": False,
    },
    "night": {
        "nr_preset": 0,
        "nr_style": 2,
        "nr_intensity": 1.8,
        "nr_local_structure": 1.0,
        "nr_local_tone": 1.0,
        "nr_skin": -1.0,
        "nr_global_tone": -1.0,
        "nr_detail": 1.0,
        "nr_color": 1.0,
        "nr_ui_correction": False,
        "nr_auto_mask": False,
    },
    "light": {
        "nr_preset": 0,
        "nr_style": 0,
        "nr_intensity": 1.0,
        "nr_local_structure": 1.0,
        "nr_local_tone": 1.0,
        "nr_skin": -1.0,
        "nr_global_tone": -1.0,
        "nr_detail": 0.8,
        "nr_color": 1.0,
        "nr_ui_correction": False,
        "nr_auto_mask": False,
    },
}

# Only audited full official release archives are accepted. The much smaller
# "light" archives are intentionally absent because they do not contain the
# governed DLL set.
APPROVED_RELEASES: dict[str, dict[str, Any]] = {
    "1.2": {
        "runtime_version": "1.2",
        "release_tag": "v1.2",
        "release_tag_commit": V12_RELEASE_TAG_COMMIT,
        "source_audit_commit": V12_NATIVE_SOURCE_AUDIT_COMMIT,
        "archives": {
            "video2dlssnr_release.zip": {
                "bytes": 247_409_690,
                "sha256": "df9de64b92b62ad381e7b83a995c6b84d5bbb6bf4db59e7d46cb75c6e3f5feb3",
            },
            "video2dlssnr-comfyui.zip": {
                "bytes": 247_400_780,
                "sha256": "26f30045a89bb9f957411303d8f4beb1a5aecf572b189455f7a94b60cdd20ce5",
            },
        },
    },
    "1.3": {
        "runtime_version": "1.3",
        "release_tag": "v1.3",
        "release_tag_commit": V13_RELEASE_TAG_COMMIT,
        "source_audit_commit": V13_NATIVE_SOURCE_AUDIT_COMMIT,
        "archives": {
            "video2dlssnr_release.zip": {
                "bytes": 247_420_277,
                "sha256": "1cab80a30927421c7fb36f42eb5f11d50fef2f2dbc8c1236424a0ccab2eff0bd",
            },
        },
    },
}

REQUIRED_RUNTIME_FILES = {
    "executable": "video2dlssnr.exe",
    "dlss_sr": "nvngx_dlss.dll",
    "dlss_nr": "nvngx_dlssnr.dll",
    "forwarder": "nvngx.dll_dlssnr.dll",
}

ProbeRunner = Callable[..., Mapping[str, Any]]


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        indent=indent,
        allow_nan=False,
    )


def _float_arg(value: float) -> str:
    return format(float(value), ".8g")


def resolve_quality_profile(
    quality_profile: str,
    manual_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if quality_profile not in QUALITY_PROFILE_NAMES:
        raise ValueError(
            f"quality_profile must be one of the supported values {QUALITY_PROFILE_NAMES}"
        )
    if quality_profile == "custom":
        profile = dict(manual_parameters or {})
        expected = set(QUALITY_PROFILES["standard"])
        if set(profile) != expected:
            raise ValueError(
                f"custom DLSS-NR parameters must be exactly {sorted(expected)}"
            )
    else:
        profile = dict(QUALITY_PROFILES[quality_profile])

    integer_ranges = {"nr_preset": (0, 3), "nr_style": (0, 2)}
    float_ranges = {
        "nr_intensity": (0.0, 2.0),
        "nr_local_structure": (0.0, 2.0),
        "nr_local_tone": (0.0, 2.0),
        "nr_skin": (-1.0, 2.0),
        "nr_global_tone": (-1.0, 2.0),
        "nr_detail": (0.0, 1.0),
        "nr_color": (0.0, 1.0),
    }
    for name, (minimum, maximum) in integer_ranges.items():
        value = profile.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= maximum
        ):
            raise ValueError(f"{name} must be an integer within {minimum}..{maximum}")
    for name, (minimum, maximum) in float_ranges.items():
        try:
            value = float(profile.get(name))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be finite within {minimum}..{maximum}"
            ) from exc
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be finite within {minimum}..{maximum}")
        profile[name] = value
    for name in ("nr_ui_correction", "nr_auto_mask"):
        if not isinstance(profile.get(name), bool):
            raise ValueError(f"{name} must be a boolean")
    return profile


def effective_quality_profile(
    mode: str,
    quality_profile: str,
    manual_parameters: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return both requested and effective NR parameters.

    The audited upstream executable still evaluates NR for ``sr_only`` and
    composites it with detail set to zero. Keeping both dictionaries makes
    that distinction explicit in reports instead of claiming that the named
    profile's detail value was used for the output.
    """

    requested = resolve_quality_profile(quality_profile, manual_parameters)
    effective = dict(requested)
    if mode == "sr_only":
        effective["nr_detail"] = 0.0
    return requested, effective


def validate_sr_preset(value: str) -> str:
    normalized = str(value)
    if normalized.lower() == "default":
        return "default"
    normalized = normalized.upper()
    if normalized not in SR_PRESETS:
        raise ValueError(f"sr_preset must be one of the supported values {SR_PRESETS}")
    return normalized


def target_dimensions(width: int, height: int, scale: float) -> tuple[int, int]:
    width = int(width)
    height = int(height)
    scale = float(scale)
    if width < 1 or height < 1:
        raise ValueError("input dimensions must be positive")
    if scale not in SUPPORTED_SCALES:
        raise ValueError(
            f"scale must be one of the supported values {SUPPORTED_SCALES}"
        )
    return (
        max(1, int(math.floor(width * scale + 0.5))),
        max(1, int(math.floor(height * scale + 0.5))),
    )


def validate_processing_contract(mode: str, scale: float) -> None:
    if mode not in PROCESSING_MODES:
        raise ValueError(f"unsupported DLSS-NR processing mode: {mode!r}")
    if float(scale) not in SUPPORTED_SCALES:
        raise ValueError(
            f"scale must be one of the supported values {SUPPORTED_SCALES}"
        )
    if mode == "nr_only" and float(scale) != 1.0:
        raise ValueError("NR only requires 1.0x so no hidden upscale path can run")
    if mode in {"sr_only", "sr_nr"} and float(scale) <= 1.0:
        raise ValueError(f"{mode} requires an explicit 1.5x, 2x or 3x upscale")


def _validate_frames(frames: torch.Tensor) -> tuple[int, int, int, int]:
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise ValueError("IMAGE must be a BHWC torch.Tensor")
    batch, height, width, channels = map(int, frames.shape)
    if batch < 1 or height < 1 or width < 1 or channels < 3:
        raise ValueError("IMAGE must contain at least one positive-size RGB frame")
    if not bool(torch.isfinite(frames).all()):
        raise ValueError("IMAGE contains NaN or Inf")
    if bool((frames < 0.0).any()) or bool((frames > 1.0).any()):
        raise ValueError("SDR IMAGE values must stay within 0..1")
    return batch, height, width, channels


def resize_extra_channels(
    extras: torch.Tensor, target_height: int, target_width: int
) -> torch.Tensor:
    if extras.ndim != 4:
        raise ValueError("extra channels must use BHWC layout")
    if int(extras.shape[-1]) == 0:
        return extras.new_empty(
            (int(extras.shape[0]), int(target_height), int(target_width), 0)
        )
    resized = torch_functional.interpolate(
        extras.detach().float().movedim(-1, 1),
        size=(int(target_height), int(target_width)),
        mode="bilinear",
        align_corners=False,
    ).movedim(1, -1)
    return resized.to(device=extras.device, dtype=extras.dtype).contiguous()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_zip_member(bundle: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    digest = hashlib.sha256()
    with bundle.open(info, "r") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_relative_posix(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.anchor or ":" in path.parts[0]:
        raise ValueError(f"{label} must be a non-empty relative POSIX path")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(
            f"{label} must be a non-empty relative POSIX path without traversal"
        )
    return path


def resolve_runtime_member(root: str | os.PathLike[str], value: Any) -> Path:
    root_path = Path(root)
    relative = _validate_relative_posix(value, label="runtime member")
    if root_path.is_symlink():
        raise ValueError("runtime root must not be a symbolic link")
    candidate = root_path.joinpath(*relative.parts)
    current = root_path
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(
                f"runtime member must not traverse a symbolic link: {value!r}"
            )
    resolved_root = root_path.resolve()
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"runtime member escaped its version directory: {value!r}"
        ) from exc
    return candidate


def available_runtime_versions(models_dir: str | os.PathLike[str]) -> list[str]:
    root = Path(models_dir) / "DLSS-NR"
    installed = []
    if root.is_dir() and not root.is_symlink():
        installed = [
            item.name
            for item in root.iterdir()
            if item.is_dir()
            and not item.is_symlink()
            and item.name in APPROVED_RELEASES
        ]
    return sorted(
        set(APPROVED_RELEASES) | set(installed),
        key=lambda value: tuple(map(int, value.split("."))),
    )


def runtime_root(models_dir: str | os.PathLike[str], runtime_version: str) -> Path:
    if runtime_version not in APPROVED_RELEASES:
        raise ValueError(
            f"DLSS-NR runtime version {runtime_version!r} is not allowlisted"
        )
    return Path(models_dir) / "DLSS-NR" / runtime_version


def _parse_driver_version(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, str):
        return None
    parts = re.findall(r"\d+", value)
    if len(parts) < 2:
        return None
    return tuple(int(item) for item in parts[:4])


def _pci_identity(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(
        r"(?:(?P<domain>[0-9a-fA-F]+):)?(?P<bus>[0-9a-fA-F]{2}):"
        r"(?P<device>[0-9a-fA-F]{2})\.(?P<function>[0-7])",
        value.strip(),
    )
    if match is None:
        return None
    return tuple(
        int(match.group(name) or "0", 16)
        for name in ("domain", "bus", "device", "function")
    )


def _gpu_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _run_nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,memory.free,pci.bus_id,uuid",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=15,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "command": command, "error": str(exc), "gpus": []}
    gpus = []
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) != 7:
                continue
            try:
                gpus.append(
                    {
                        "index": int(fields[0]),
                        "name": fields[1],
                        "driver_version": fields[2],
                        "memory_total_mib": float(fields[3]),
                        "memory_free_mib": float(fields[4]),
                        "pci_bus_id": fields[5],
                        "uuid": fields[6],
                    }
                )
            except ValueError:
                continue
    return {
        "available": completed.returncode == 0 and bool(gpus),
        "command": command,
        "returncode": completed.returncode,
        "stderr": completed.stderr[-4096:],
        "gpus": gpus,
    }


def _torch_cuda_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on host installation
        return {"available": False, "error": str(exc), "devices": []}
    try:
        if not torch.cuda.is_available():
            return {"available": False, "devices": []}
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            pci_bus = getattr(props, "pci_bus_id", None)
            if not isinstance(pci_bus, str):
                domain = getattr(props, "pci_domain_id", None)
                bus = pci_bus
                device = getattr(props, "pci_device_id", None)
                if all(isinstance(item, int) for item in (domain, bus, device)):
                    pci_bus = f"{domain:04x}:{bus:02x}:{device:02x}.0"
                else:
                    pci_bus = None
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "memory_total_mib": float(props.total_memory) / (1024.0 * 1024.0),
                    "pci_bus_id": pci_bus,
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
        return {"available": True, "devices": devices}
    except Exception as exc:  # pragma: no cover - depends on CUDA runtime state
        return {"available": False, "error": str(exc), "devices": []}


def collect_host_info() -> dict[str, Any]:
    return {
        "platform": platform.system(),
        "windows_release": platform.release(),
        "nvidia_smi": _run_nvidia_smi(),
        "torch_cuda": _torch_cuda_info(),
    }


def _mapped_devices(
    host_info: Mapping[str, Any],
    dxgi_adapter_index: int,
    cuda_device_index: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    smi = host_info.get("nvidia_smi")
    torch_cuda = host_info.get("torch_cuda")
    smi_gpus = list(smi.get("gpus", [])) if isinstance(smi, Mapping) else []
    cuda_gpus = (
        list(torch_cuda.get("devices", [])) if isinstance(torch_cuda, Mapping) else []
    )
    selected_cuda = next(
        (
            item
            for item in cuda_gpus
            if int(item.get("index", -1)) == int(cuda_device_index)
        ),
        None,
    )
    selected_smi = None
    if selected_cuda is None:
        errors.append(f"Torch CUDA device index {cuda_device_index} was not found")
    else:
        cuda_pci = _pci_identity(selected_cuda.get("pci_bus_id"))
        if cuda_pci is not None:
            candidates = [
                item
                for item in smi_gpus
                if _pci_identity(item.get("pci_bus_id")) == cuda_pci
            ]
        else:
            candidates = [
                item
                for item in smi_gpus
                if _gpu_name(item.get("name")) == _gpu_name(selected_cuda.get("name"))
                and abs(
                    float(item.get("memory_total_mib", -1.0))
                    - float(selected_cuda.get("memory_total_mib", -2.0))
                )
                <= 256.0
            ]
        if len(candidates) == 1:
            selected_smi = candidates[0]
        else:
            errors.append(
                "selected Torch CUDA device cannot be mapped unambiguously to one NVIDIA-SMI GPU"
            )
    if not 0 <= int(dxgi_adapter_index) <= 31:
        errors.append("DXGI adapter index must stay within 0..31")
    return selected_smi, selected_cuda, errors


def _exact_keys(mapping: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{label} must be an object")
    actual = set(mapping)
    if actual != expected:
        raise ValueError(
            f"{label} keys must be exactly {sorted(expected)}; got {sorted(actual)}"
        )
    return mapping


def _verify_runtime_files(
    root: Path,
    manifest: Mapping[str, Any],
    approved: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    archive_spec = _exact_keys(
        manifest.get("release_archive"),
        {"path", "bytes", "sha256"},
        label="release_archive",
    )
    archive_relative = _validate_relative_posix(
        archive_spec.get("path"), label="release_archive.path"
    )
    if len(archive_relative.parts) != 1:
        raise ValueError(
            "release_archive.path must name a file in the runtime version root"
        )
    allowlisted_archive = approved["archives"].get(archive_relative.as_posix())
    if allowlisted_archive is None:
        raise ValueError("release archive is not on the audited full-archive allowlist")
    for field in ("bytes", "sha256"):
        if archive_spec.get(field) != allowlisted_archive[field]:
            raise ValueError(
                f"release_archive.{field} does not match the built-in allowlist"
            )
    archive_path = resolve_runtime_member(root, archive_relative.as_posix())
    if not archive_path.is_file():
        raise ValueError(f"release archive is missing: {archive_path}")
    actual_archive = {
        "path": archive_relative.as_posix(),
        "bytes": archive_path.stat().st_size,
        "sha256": _sha256_file(archive_path),
    }
    if actual_archive["bytes"] != allowlisted_archive["bytes"]:
        raise ValueError(
            "release archive byte size does not match the built-in allowlist"
        )
    if actual_archive["sha256"] != allowlisted_archive["sha256"]:
        raise ValueError(
            "release archive SHA-256 does not match the built-in allowlist"
        )

    files = _exact_keys(
        manifest.get("files"), set(REQUIRED_RUNTIME_FILES), label="files"
    )
    runtime_files: dict[str, Any] = {}
    with zipfile.ZipFile(archive_path, "r") as bundle:
        infos = bundle.infolist()
        info_by_name: dict[str, zipfile.ZipInfo] = {}
        for info in infos:
            # The official v1.3 Windows archive stores member separators as
            # backslashes. Canonicalize only for lookup after validating that
            # the resulting POSIX name is still relative and traversal-free.
            canonical_name = info.filename.replace("\\", "/")
            _validate_relative_posix(canonical_name, label="release archive member")
            if canonical_name in info_by_name:
                raise ValueError(
                    "release archive contains duplicate canonical member names"
                )
            info_by_name[canonical_name] = info
        seen_installed: set[str] = set()
        seen_archive: set[str] = set()
        for logical, required_basename in REQUIRED_RUNTIME_FILES.items():
            spec = _exact_keys(
                files.get(logical), {"path", "archive_path"}, label=f"files.{logical}"
            )
            installed_relative = _validate_relative_posix(
                spec.get("path"), label=f"files.{logical}.path"
            ).as_posix()
            archive_member = _validate_relative_posix(
                spec.get("archive_path"), label=f"files.{logical}.archive_path"
            ).as_posix()
            if PurePosixPath(installed_relative).name != required_basename:
                raise ValueError(
                    f"files.{logical}.path must end with {required_basename}"
                )
            if PurePosixPath(archive_member).name != required_basename:
                raise ValueError(
                    f"files.{logical}.archive_path must end with {required_basename}"
                )
            if installed_relative in seen_installed or archive_member in seen_archive:
                raise ValueError("runtime manifest file paths must be unique")
            seen_installed.add(installed_relative)
            seen_archive.add(archive_member)
            info = info_by_name.get(archive_member)
            if info is None or info.is_dir():
                raise ValueError(
                    f"required release archive member is missing: {archive_member}"
                )
            mode = (info.external_attr >> 16) & 0xFFFF
            if mode and stat.S_ISLNK(mode):
                raise ValueError(
                    f"release archive member must not be a symbolic link: {archive_member}"
                )
            installed_path = resolve_runtime_member(root, installed_relative)
            if not installed_path.is_file():
                raise ValueError(
                    f"required installed runtime file is missing: {installed_relative}"
                )
            archive_hash = _sha256_zip_member(bundle, info)
            installed_hash = _sha256_file(installed_path)
            installed_size = installed_path.stat().st_size
            matches = (
                installed_size == info.file_size and installed_hash == archive_hash
            )
            runtime_files[logical] = {
                "path": installed_relative,
                "archive_path": archive_member,
                "bytes": installed_size,
                "sha256": installed_hash,
                "archive_bytes": info.file_size,
                "archive_sha256": archive_hash,
                "matches_archive": matches,
            }
            if not matches:
                raise ValueError(
                    f"installed runtime file does not match the verified release archive: {installed_relative}"
                )
    _runtime_binary_directory(root, runtime_files)
    return actual_archive, runtime_files


def _runtime_binary_directory(root: Path, runtime_files: Mapping[str, Any]) -> Path:
    """Resolve the one directory from which the audited executable loads every DLL."""

    directories: set[Path] = set()
    for logical in REQUIRED_RUNTIME_FILES:
        spec = runtime_files.get(logical)
        if not isinstance(spec, Mapping) or not isinstance(spec.get("path"), str):
            raise ValueError(f"runtime file {logical} is malformed")
        directories.add(resolve_runtime_member(root, spec["path"]).resolve().parent)
    if len(directories) != 1:
        raise ValueError(
            "the executable and all three DLSS runtime DLLs must be installed in one directory"
        )
    return next(iter(directories))


def _load_and_verify_manifest(
    root: Path, runtime_version: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    approved = APPROVED_RELEASES.get(runtime_version)
    if approved is None:
        raise ValueError(
            f"DLSS-NR runtime version {runtime_version!r} is not allowlisted"
        )
    if root.name != runtime_version:
        raise ValueError(
            "runtime directory name must equal the selected allowlisted version"
        )
    manifest_path = resolve_runtime_member(root, RUNTIME_MANIFEST_NAME)
    if not manifest_path.is_file():
        raise ValueError(f"runtime manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"runtime manifest could not be read as strict UTF-8 JSON: {exc}"
        ) from exc
    manifest = dict(
        _exact_keys(
            manifest,
            {
                "schema",
                "runtime_version",
                "release_tag",
                "release_tag_commit",
                "source_audit_commit",
                "release_archive",
                "files",
            },
            label="runtime manifest",
        )
    )
    expected_identity = {
        "schema": RUNTIME_MANIFEST_SCHEMA,
        "runtime_version": approved["runtime_version"],
        "release_tag": approved["release_tag"],
        "release_tag_commit": approved["release_tag_commit"],
        "source_audit_commit": approved["source_audit_commit"],
    }
    for field, expected in expected_identity.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"runtime manifest {field} does not match the audited release"
            )
    archive, runtime_files = _verify_runtime_files(root, manifest, approved)
    return manifest, archive, runtime_files


def _default_probe_runner(
    command: list[str], *, timeout_seconds: float
) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(Path(command[0]).parent),
            capture_output=True,
            text=True,
            errors="replace",
            check=False,
            timeout=timeout_seconds,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout[-65536:],
            "stderr": completed.stderr[-65536:],
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "stdout": str(exc.stdout or "")[-65536:],
            "stderr": str(exc.stderr or "")[-65536:],
            "timed_out": True,
        }
    except OSError as exc:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "timed_out": False,
        }


def build_feature_probe_command(
    runtime_files: Mapping[str, Any],
    root: Path,
    dxgi_adapter_index: int,
) -> list[str]:
    executable = resolve_runtime_member(
        root, runtime_files["executable"]["path"]
    ).resolve()
    dll_dir = _runtime_binary_directory(root, runtime_files)
    return [
        str(executable),
        "--probe-nr",
        "--dll-dir",
        str(dll_dir),
        "--adapter",
        str(int(dxgi_adapter_index)),
        "--nr-in",
        "64x64",
        "--nr-out",
        "64x64",
        "--nr-preset",
        "0",
    ]


def _probe_gpu(text: str) -> dict[str, Any] | None:
    matches = re.findall(r"gpu:\s*(.+?)\s*\((\d+)\s*MB\)", text, flags=re.IGNORECASE)
    if len(matches) != 1:
        return None
    name, memory = matches[0]
    return {"name": name.strip(), "memory_total_mib": float(memory)}


def audit_dlss_nr_runtime(
    root: str | os.PathLike[str],
    runtime_version: str,
    *,
    accept_external_runtime_license: bool,
    probe_mode: str,
    dxgi_adapter_index: int,
    cuda_device_index: int,
    host_info: Mapping[str, Any] | None = None,
    probe_runner: ProbeRunner | None = None,
) -> tuple[bool, dict[str, Any]]:
    if probe_mode not in PROBE_MODES:
        raise ValueError(f"unsupported DLSS-NR probe mode: {probe_mode!r}")
    root_path = Path(root)
    host = dict(host_info) if host_info is not None else collect_host_info()
    approved_release = APPROVED_RELEASES.get(runtime_version)
    report: dict[str, Any] = {
        "schema": "t8.dlss_nr.runtime_audit.v1",
        "status": "BLOCKED",
        "ready": False,
        "runtime_root": str(root_path.resolve(strict=False)),
        "runtime_version": runtime_version,
        "probe_mode": probe_mode,
        "license_acceptance": bool(accept_external_runtime_license),
        "upstream": {
            "native_repository": "https://github.com/DaniilSokolyuk/video2dlssnr",
            "native_source_audit_commit": (
                approved_release["source_audit_commit"]
                if approved_release is not None
                else None
            ),
            "native_source_license_at_audit_commit": "NO_ROOT_LICENSE",
            "wrapper_repository": "https://github.com/piscesbody/ComfyUI-DLSS-NR",
            "wrapper_source_audit_commit": WRAPPER_SOURCE_AUDIT_COMMIT,
            "wrapper_source_license": "MIT",
        },
        "host": host,
        "runtime_manifest": None,
        "release_archive": None,
        "runtime_files": {},
        "device_mapping": None,
        "static_validation": {"passed": False},
        "feature_probe": None,
        "minimum_driver_version": ".".join(map(str, MINIMUM_DRIVER_VERSION)),
        "minimum_free_vram_mib": MINIMUM_FREE_VRAM_MIB,
        "probe_memory_tolerance_mib": PROBE_MEMORY_TOLERANCE_MIB,
        "errors": [],
        "warnings": [],
        "mutations": [],
        "downloads": [],
    }
    errors: list[str] = report["errors"]
    if not accept_external_runtime_license:
        errors.append(
            "external runtime license acceptance is required before auditing proprietary files"
        )
    if str(host.get("platform")) != "Windows":
        errors.append("DLSS-NR v1 is Windows-only")

    selected_smi, selected_cuda, mapping_errors = _mapped_devices(
        host, dxgi_adapter_index, cuda_device_index
    )
    errors.extend(mapping_errors)
    if selected_smi is not None and selected_cuda is not None:
        report["device_mapping"] = {
            "dxgi_adapter_index": int(dxgi_adapter_index),
            "cuda_device_index": int(cuda_device_index),
            "nvidia": selected_smi,
            "torch_cuda": selected_cuda,
        }
        driver = _parse_driver_version(selected_smi.get("driver_version"))
        if driver is None or driver < MINIMUM_DRIVER_VERSION:
            errors.append(
                "NVIDIA driver must be 616.56 or newer for the audited DLSS-NR runtime"
            )
        free_mib = selected_smi.get("memory_free_mib")
        try:
            free_value = float(free_mib)
        except (TypeError, ValueError):
            errors.append("selected NVIDIA adapter free VRAM could not be measured")
        else:
            if free_value < MINIMUM_FREE_VRAM_MIB:
                errors.append(
                    "selected NVIDIA adapter must have at least 512 MiB free VRAM"
                )

    if runtime_version not in APPROVED_RELEASES:
        errors.append(f"DLSS-NR runtime version {runtime_version!r} is not allowlisted")
    elif accept_external_runtime_license:
        try:
            manifest, archive, runtime_files = _load_and_verify_manifest(
                root_path, runtime_version
            )
            report["runtime_manifest"] = manifest
            report["release_archive"] = archive
            report["runtime_files"] = runtime_files
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            errors.append(str(exc))

    report["static_validation"] = {"passed": not errors}
    if errors:
        return False, report
    if probe_mode == "static_only":
        report["status"] = "STATIC_PASS_REAL_PROBE_REQUIRED"
        return False, report

    command = build_feature_probe_command(
        report["runtime_files"], root_path, dxgi_adapter_index
    )
    runner = probe_runner or _default_probe_runner
    result = dict(runner(command, timeout_seconds=PROBE_TIMEOUT_SECONDS))
    stdout = str(result.get("stdout", ""))
    stderr = str(result.get("stderr", ""))
    combined = stdout + "\n" + stderr
    parsed_gpu = _probe_gpu(combined)
    feature_probe = {
        "command": command,
        "timeout_seconds": PROBE_TIMEOUT_SECONDS,
        "returncode": result.get("returncode"),
        "timed_out": bool(result.get("timed_out", False)),
        "gpu": parsed_gpu,
        "route_f_success_marker": "Neural Rendering RAN via route F" in combined,
        "stdout_tail": stdout[-8192:],
        "stderr_tail": stderr[-8192:],
    }
    report["feature_probe"] = feature_probe
    if feature_probe["timed_out"]:
        errors.append("DLSS-NR feature probe timed out")
    elif result.get("returncode") != 0:
        errors.append(
            f"DLSS-NR feature probe failed with exit code {result.get('returncode')}"
        )
    if not feature_probe["route_f_success_marker"]:
        errors.append(
            "DLSS-NR feature probe did not execute the required forwarder route F"
        )
    if parsed_gpu is None:
        errors.append("DLSS-NR feature probe did not report exactly one DXGI adapter")
    else:
        expected_name = _gpu_name(selected_smi.get("name"))
        actual_name = _gpu_name(parsed_gpu.get("name"))
        memory_delta = abs(
            float(selected_smi.get("memory_total_mib", -1.0))
            - float(parsed_gpu.get("memory_total_mib", -2.0))
        )
        if expected_name != actual_name or memory_delta > PROBE_MEMORY_TOLERANCE_MIB:
            errors.append(
                "feature probe DXGI adapter does not match the selected NVIDIA/Torch device"
            )
        smi = host.get("nvidia_smi")
        smi_gpus = list(smi.get("gpus", [])) if isinstance(smi, Mapping) else []
        identity_candidates = [
            item
            for item in smi_gpus
            if _gpu_name(item.get("name")) == actual_name
            and abs(
                float(item.get("memory_total_mib", -1.0))
                - float(parsed_gpu.get("memory_total_mib", -2.0))
            )
            <= PROBE_MEMORY_TOLERANCE_MIB
        ]
        feature_probe["dxgi_identity_candidate_count"] = len(identity_candidates)
        feature_probe["dxgi_identity_unique"] = len(identity_candidates) == 1
        if len(identity_candidates) != 1:
            errors.append(
                "feature probe cannot bind the DXGI adapter uniquely because multiple NVIDIA GPUs "
                "share the reported name and memory capacity"
            )
    if errors:
        return False, report
    report["status"] = "READY"
    report["ready"] = True
    return True, report


def runtime_handle_from_report(report: Mapping[str, Any]) -> dict[str, Any] | None:
    if report.get("status") != "READY" or not bool(report.get("ready")):
        return None
    mapping = report.get("device_mapping")
    files = report.get("runtime_files")
    if not isinstance(mapping, Mapping) or not isinstance(files, Mapping):
        raise ValueError(
            "ready DLSS-NR audit report is missing its device or runtime files"
        )
    fingerprint_payload = {
        "runtime_root": report.get("runtime_root"),
        "runtime_version": report.get("runtime_version"),
        "release_archive": report.get("release_archive"),
        "runtime_files": files,
        "dxgi_adapter_index": mapping.get("dxgi_adapter_index"),
        "cuda_device_index": mapping.get("cuda_device_index"),
    }
    fingerprint = hashlib.sha256(
        canonical_json(fingerprint_payload).encode("utf-8")
    ).hexdigest()
    return {
        "schema": RUNTIME_HANDLE_SCHEMA,
        "runtime_root": str(report["runtime_root"]),
        "runtime_version": str(report["runtime_version"]),
        "dxgi_adapter_index": int(mapping["dxgi_adapter_index"]),
        "cuda_device_index": int(mapping["cuda_device_index"]),
        "runtime_files": {key: dict(value) for key, value in files.items()},
        "audit_fingerprint": fingerprint,
    }


def _validated_runtime_handle(
    runtime: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    runtime = dict(
        _exact_keys(
            runtime,
            {
                "schema",
                "runtime_root",
                "runtime_version",
                "dxgi_adapter_index",
                "cuda_device_index",
                "runtime_files",
                "audit_fingerprint",
            },
            label="DLSS-NR runtime handle",
        )
    )
    if runtime["schema"] != RUNTIME_HANDLE_SCHEMA:
        raise ValueError("unsupported DLSS-NR runtime handle schema")
    root = Path(os.fspath(runtime["runtime_root"]))
    if not root.is_absolute():
        raise ValueError("DLSS-NR runtime handle root must be absolute")
    files = _exact_keys(
        runtime.get("runtime_files"),
        set(REQUIRED_RUNTIME_FILES),
        label="runtime handle files",
    )
    normalized_files: dict[str, Any] = {}
    for logical, basename in REQUIRED_RUNTIME_FILES.items():
        spec = files[logical]
        if not isinstance(spec, Mapping) or not isinstance(spec.get("path"), str):
            raise ValueError(f"runtime handle file {logical} is malformed")
        path = resolve_runtime_member(root, spec["path"])
        if path.name != basename:
            raise ValueError(f"runtime handle file {logical} must end with {basename}")
        if not path.is_file():
            raise ValueError(f"runtime handle file is missing: {path}")
        normalized_files[logical] = dict(spec)
    _runtime_binary_directory(root, normalized_files)
    runtime["runtime_files"] = normalized_files
    return root, runtime


def revalidate_runtime_handle(
    runtime: Mapping[str, Any], host_info: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    root, handle = _validated_runtime_handle(runtime)
    host = dict(host_info) if host_info is not None else collect_host_info()
    errors: list[str] = []
    if str(host.get("platform")) != "Windows":
        errors.append("DLSS-NR v1 is Windows-only")
    selected_smi, selected_cuda, mapping_errors = _mapped_devices(
        host,
        int(handle["dxgi_adapter_index"]),
        int(handle["cuda_device_index"]),
    )
    errors.extend(mapping_errors)
    if selected_smi is not None:
        driver = _parse_driver_version(selected_smi.get("driver_version"))
        if driver is None or driver < MINIMUM_DRIVER_VERSION:
            errors.append("NVIDIA driver must be 616.56 or newer")
        try:
            free_mib = float(selected_smi.get("memory_free_mib"))
        except (TypeError, ValueError):
            errors.append("selected NVIDIA adapter free VRAM could not be measured")
        else:
            if free_mib < MINIMUM_FREE_VRAM_MIB:
                errors.append(
                    "selected NVIDIA adapter must have at least 512 MiB free VRAM"
                )
        smi = host.get("nvidia_smi")
        smi_gpus = list(smi.get("gpus", [])) if isinstance(smi, Mapping) else []
        same_signature = [
            item
            for item in smi_gpus
            if _gpu_name(item.get("name")) == _gpu_name(selected_smi.get("name"))
            and abs(
                float(item.get("memory_total_mib", -1.0))
                - float(selected_smi.get("memory_total_mib", -2.0))
            )
            <= PROBE_MEMORY_TOLERANCE_MIB
        ]
        if len(same_signature) != 1:
            errors.append(
                "the audited DXGI adapter cannot be revalidated uniquely because multiple NVIDIA "
                "GPUs share its name and memory capacity"
            )
    try:
        _manifest, archive, runtime_files = _load_and_verify_manifest(
            root, str(handle["runtime_version"])
        )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        errors.append(str(exc))
        archive = None
        runtime_files = None
    if errors:
        raise RuntimeError(
            "DLSS-NR execution revalidation failed: " + "; ".join(errors)
        )
    fingerprint_payload = {
        "runtime_root": str(root.resolve()),
        "runtime_version": str(handle["runtime_version"]),
        "release_archive": archive,
        "runtime_files": runtime_files,
        "dxgi_adapter_index": int(handle["dxgi_adapter_index"]),
        "cuda_device_index": int(handle["cuda_device_index"]),
    }
    fingerprint = hashlib.sha256(
        canonical_json(fingerprint_payload).encode("utf-8")
    ).hexdigest()
    if fingerprint != handle["audit_fingerprint"]:
        raise RuntimeError("DLSS-NR runtime changed after its successful feature audit")
    return {
        "status": "READY_REVALIDATED",
        "runtime_root": str(root.resolve()),
        "device_mapping": {
            "nvidia": selected_smi,
            "torch_cuda": selected_cuda,
        },
        "minimum_free_vram_mib": MINIMUM_FREE_VRAM_MIB,
        "audit_fingerprint": fingerprint,
    }


def _profile_arguments(mode: str, profile: Mapping[str, Any]) -> list[str]:
    detail = 0.0 if mode == "sr_only" else float(profile["nr_detail"])
    arguments = [
        "--nr-preset",
        str(int(profile["nr_preset"])),
        "--nr-style",
        str(int(profile["nr_style"])),
        "--nr-intensity",
        _float_arg(profile["nr_intensity"]),
        "--nr-local-structure",
        _float_arg(profile["nr_local_structure"]),
        "--nr-local-tone",
        _float_arg(profile["nr_local_tone"]),
        "--nr-skin",
        _float_arg(profile["nr_skin"]),
        "--nr-global-tone",
        _float_arg(profile["nr_global_tone"]),
        "--nr-detail",
        _float_arg(detail),
        "--nr-color",
        _float_arg(profile["nr_color"]),
        "--nr-ui-correction",
        "1" if profile["nr_ui_correction"] else "0",
    ]
    if profile["nr_auto_mask"]:
        arguments.append("--nr-auto-mask")
    return arguments


def build_image_command(
    runtime: Mapping[str, Any],
    *,
    source_path: Path,
    output_directory: Path,
    mode: str,
    scale: float,
    quality_profile: str = "standard",
    sr_preset: str = "default",
    manual_parameters: Mapping[str, Any] | None = None,
) -> list[str]:
    validate_processing_contract(mode, scale)
    root, runtime = _validated_runtime_handle(runtime)
    runtime_version = str(runtime["runtime_version"])
    sr_preset = validate_sr_preset(sr_preset)
    if runtime_version == "1.2" and mode != "nr_only":
        raise RuntimeError(
            "video2dlssnr v1.2 uses the wrong DLSS SR quality mode for scaled output; "
            "select the audited v1.3 runtime"
        )
    if runtime_version == "1.2" and sr_preset != "default":
        raise RuntimeError("video2dlssnr v1.2 does not support DLSS SR model selection")
    _requested_profile, profile = effective_quality_profile(
        mode, quality_profile, manual_parameters
    )
    executable = resolve_runtime_member(
        root, runtime["runtime_files"]["executable"]["path"]
    ).resolve()
    dll_dir = _runtime_binary_directory(root, runtime["runtime_files"])
    command = [
        str(executable),
        "--nr-run",
        "--in",
        str(source_path.resolve()),
        "--out",
        str(output_directory.resolve()),
        "--dll-dir",
        str(dll_dir),
        "--adapter",
        str(int(runtime["dxgi_adapter_index"])),
        "--nr-scale",
        _float_arg(scale),
    ]
    if runtime_version == "1.3":
        command.extend(("--nr-sr-preset", sr_preset))
    command.extend(_profile_arguments(mode, profile))
    return command


def build_raw_video_command(
    runtime: Mapping[str, Any],
    *,
    width: int,
    height: int,
    frame_count: int,
    mode: str,
    scale: float,
    motion_engine: str,
    quality_profile: str = "standard",
    sr_preset: str = "default",
    manual_parameters: Mapping[str, Any] | None = None,
) -> list[str]:
    validate_processing_contract(mode, scale)
    if int(width) < 1 or int(height) < 1 or int(frame_count) < 1:
        raise ValueError("raw video dimensions and frame count must be positive")
    if motion_engine not in MOTION_ENGINES:
        raise ValueError(f"unsupported DLSS-NR motion engine: {motion_engine!r}")
    root, runtime = _validated_runtime_handle(runtime)
    runtime_version = str(runtime["runtime_version"])
    sr_preset = validate_sr_preset(sr_preset)
    if runtime_version == "1.2" and mode != "nr_only":
        raise RuntimeError(
            "video2dlssnr v1.2 uses the wrong DLSS SR quality mode for scaled output; "
            "select the audited v1.3 runtime"
        )
    if runtime_version == "1.2" and sr_preset != "default":
        raise RuntimeError("video2dlssnr v1.2 does not support DLSS SR model selection")
    _requested_profile, profile = effective_quality_profile(
        mode, quality_profile, manual_parameters
    )
    executable = resolve_runtime_member(
        root, runtime["runtime_files"]["executable"]["path"]
    ).resolve()
    dll_dir = _runtime_binary_directory(root, runtime["runtime_files"])
    command = [
        str(executable),
        "--nr-video",
        "--dll-dir",
        str(dll_dir),
        "--adapter",
        str(int(runtime["dxgi_adapter_index"])),
        "--nr-in",
        f"{int(width)}x{int(height)}",
        "--nr-scale",
        _float_arg(scale),
        "--nr-motion",
        "1",
        "--nr-motion-engine",
        motion_engine,
    ]
    if runtime_version == "1.3":
        command.extend(("--nr-sr-preset", sr_preset))
    command.extend(_profile_arguments(mode, profile))
    return command


def _default_media_runner(
    command: list[str],
    *,
    cwd: str,
    timeout_seconds: float,
    interrupt_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        shell=False,
        creationflags=flags,
    )
    check = interrupt_check or _default_interrupt_check
    started = time.monotonic()
    try:
        while True:
            check()
            remaining = float(timeout_seconds) - (time.monotonic() - started)
            if remaining <= 0:
                _terminate_process(process)
                return {
                    "returncode": None,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": True,
                }
            try:
                stdout, stderr = process.communicate(timeout=min(0.1, remaining))
                return {
                    "returncode": process.returncode,
                    "stdout": stdout[-65536:],
                    "stderr": stderr[-65536:],
                    "timed_out": False,
                }
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        _terminate_process(process)
        raise
    finally:
        if process.poll() is None:
            _terminate_process(process)


def _sr_only_execution_report(mode: str) -> dict[str, Any]:
    return {
        "nr_evaluation_required_by_upstream": mode == "sr_only",
        "output_composite_nr_detail": 0.0 if mode == "sr_only" else None,
    }


def _rgba8_numpy(frame: torch.Tensor):
    rgb = frame[..., :3].detach().float().cpu()
    rgb8 = rgb.mul(255.0).round().to(torch.uint8).contiguous()
    alpha = torch.full(
        (*rgb8.shape[:-1], 1), 255, dtype=torch.uint8, device=rgb8.device
    )
    return torch.cat((rgb8, alpha), dim=-1).numpy()


def process_image_batch(
    runtime: Mapping[str, Any],
    images: torch.Tensor,
    *,
    mode: str,
    scale: float,
    quality_profile: str = "standard",
    sr_preset: str = "default",
    manual_parameters: Mapping[str, Any] | None = None,
    runner: Callable[..., Mapping[str, Any]] | None = None,
    interrupt_check: Callable[[], None] | None = None,
    timeout_seconds: float = MEDIA_TIMEOUT_SECONDS,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    batch, height, width, channels = _validate_frames(images)
    validate_processing_contract(mode, scale)
    target_width, target_height = target_dimensions(width, height, scale)
    root, runtime_handle = _validated_runtime_handle(runtime)
    requested_profile, effective_profile = effective_quality_profile(
        mode, quality_profile, manual_parameters
    )
    sr_preset = validate_sr_preset(sr_preset)
    check = interrupt_check or _default_interrupt_check
    candidates = []
    process_reports = []
    started = time.perf_counter()
    from PIL import Image

    with tempfile.TemporaryDirectory(prefix="t8-dlss-nr-image-") as temporary:
        temporary_path = Path(temporary)
        for frame_index in range(batch):
            check()
            input_path = temporary_path / f"frame_{frame_index:06d}.png"
            Image.fromarray(_rgba8_numpy(images[frame_index]), mode="RGBA").save(
                input_path
            )
            command = build_image_command(
                runtime,
                source_path=input_path,
                output_directory=temporary_path,
                mode=mode,
                scale=scale,
                quality_profile=quality_profile,
                sr_preset=sr_preset,
                manual_parameters=manual_parameters,
            )
            if runner is None:
                raw_result = _default_media_runner(
                    command,
                    cwd=str(root),
                    timeout_seconds=float(timeout_seconds),
                    interrupt_check=check,
                )
            else:
                raw_result = runner(
                    command,
                    cwd=str(root),
                    timeout_seconds=float(timeout_seconds),
                )
            result = dict(raw_result)
            check()
            if bool(result.get("timed_out", False)):
                raise RuntimeError(f"DLSS-NR image frame {frame_index} timed out")
            if result.get("returncode") != 0:
                raise RuntimeError(
                    f"DLSS-NR image frame {frame_index} failed with exit code "
                    f"{result.get('returncode')}: {str(result.get('stderr', ''))[-2000:]}"
                )
            diagnostics = (
                str(result.get("stdout", "")) + "\n" + str(result.get("stderr", ""))
            )
            if "finishing the rest bilinear" in diagnostics.lower():
                raise RuntimeError(
                    "DLSS SR was refused and upstream attempted a bilinear fallback; candidate rejected"
                )
            # The audited CLI appends ``_nr.png`` to the complete input
            # filename, including its original extension.
            output_path = temporary_path / f"{input_path.name}_nr.png"
            if not output_path.is_file():
                raise RuntimeError(
                    f"DLSS-NR image frame {frame_index} did not produce expected output "
                    f"{output_path.name!r}: {diagnostics[-2000:]}"
                )
            with Image.open(output_path) as opened:
                if opened.size != (target_width, target_height):
                    raise RuntimeError(
                        "DLSS-NR image output dimensions do not match the exact requested scale"
                    )
                rgba = torch.from_numpy(
                    __import__("numpy").array(opened.convert("RGBA"), copy=True)
                )
            candidates.append(rgba[..., :3].to(dtype=torch.float32).div_(255.0))
            process_reports.append(
                {
                    "frame_index": frame_index,
                    "returncode": result.get("returncode"),
                    "stdout_tail": str(result.get("stdout", ""))[-2048:],
                    "stderr_tail": str(result.get("stderr", ""))[-2048:],
                }
            )
    candidate = torch.stack(candidates, dim=0).to(device=images.device)
    if channels > 3:
        extras = resize_extra_channels(images[..., 3:], target_height, target_width)
        candidate = torch.cat((candidate.to(dtype=images.dtype), extras), dim=-1)
    else:
        candidate = candidate.to(dtype=images.dtype)
    if tuple(candidate.shape) != (batch, target_height, target_width, channels):
        raise RuntimeError("DLSS-NR image batch output shape is inconsistent")
    report = {
        "schema": "t8.dlss_nr.image.v1",
        "status": "EXP_CANDIDATE_GENERATED",
        "mode": mode,
        "scale": float(scale),
        "runtime_version": str(runtime_handle["runtime_version"]),
        "quality_profile": quality_profile,
        "requested_nr_parameters": requested_profile,
        "nr_parameters": effective_profile,
        "sr_only_execution": _sr_only_execution_report(mode),
        "sr_preset": sr_preset,
        "input_shape": list(images.shape),
        "output_shape": list(candidate.shape),
        "frame_order_exact": True,
        "rgb_bridge": "rounded_uint8_rgba_png",
        "extra_channels": "bilinear_resize_source_then_reattach",
        "no_bilinear_rgb_fallback": True,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "processes": process_reports,
    }
    return candidate, images, report


def _default_interrupt_check() -> None:
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except (ImportError, AttributeError):
        return
    throw_exception_if_processing_interrupted()


def _terminate_process(process) -> None:
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(process, stream_name, None)
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
    if process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                pass


def _read_exact(stream, byte_count: int) -> bytes:
    output = bytearray()
    while len(output) < byte_count:
        chunk = stream.read(byte_count - len(output))
        if not chunk:
            break
        output.extend(chunk)
    return bytes(output)


def _resolve_motion_backend(diagnostics: str, requested: str) -> str:
    if requested not in MOTION_ENGINES:
        raise ValueError(f"unsupported DLSS-NR motion engine: {requested!r}")
    matches = re.findall(
        r"^\s*optical flow:\s*(.+?)\s*$",
        diagnostics,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    resolved = []
    for line in matches:
        normalized = line.casefold()
        if "nvidia nvofa" in normalized and "hardware" in normalized:
            resolved.append("nvof")
        elif "lucas-kanade" in normalized and "gpu compute" in normalized:
            resolved.append("lk")
        else:
            resolved.append("unknown")
    if len(resolved) != 1 or resolved[0] == "unknown":
        raise RuntimeError(
            "DLSS-NR video helper did not report exactly one recognized optical-flow backend"
        )
    actual = resolved[0]
    if requested != "auto" and actual != requested:
        raise RuntimeError(
            f"DLSS-NR requested motion backend {requested!r} but actually used {actual!r}"
        )
    return actual


def _stream_raw_frames(
    command: list[str],
    raw_inputs,
    *,
    frame_count: int,
    output_frame_bytes: int,
    requested_motion_engine: str,
    process_factory=None,
    interrupt_check: Callable[[], None] | None = None,
    timeout_seconds: float = MEDIA_TIMEOUT_SECONDS,
    output_consumer: Callable[[int, bytes], None] | None = None,
) -> tuple[list[bytes], dict[str, Any]]:
    factory = process_factory or subprocess.Popen
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    process = factory(
        command,
        cwd=str(Path(command[0]).parent),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        creationflags=flags,
    )
    if process.stdin is None or process.stdout is None or process.stderr is None:
        _terminate_process(process)
        raise RuntimeError("DLSS-NR helper did not expose all required raw pipes")
    stop = threading.Event()
    errors: queue.Queue[BaseException] = queue.Queue()
    outputs: queue.Queue[tuple[int, bytes]] = queue.Queue(maxsize=4)
    stderr_chunks: list[bytes] = []
    stderr_prefix = bytearray()

    def put_output(item: tuple[int, bytes]) -> None:
        while not stop.is_set():
            try:
                outputs.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def writer() -> None:
        try:
            written = 0
            for written, raw in enumerate(raw_inputs, start=1):
                if stop.is_set():
                    break
                process.stdin.write(raw)
            if written != frame_count and not stop.is_set():
                raise RuntimeError(
                    f"DLSS-NR input iterator produced {written} frames, expected {frame_count}"
                )
            process.stdin.close()
        except BaseException as exc:
            errors.put(exc)
            stop.set()

    def reader() -> None:
        try:
            for frame_index in range(frame_count):
                data = _read_exact(process.stdout, output_frame_bytes)
                if len(data) != output_frame_bytes:
                    raise RuntimeError(
                        f"DLSS-NR raw output is incomplete at frame {frame_index}: "
                        f"{len(data)} of {output_frame_bytes} bytes"
                    )
                put_output((frame_index, data))
            trailing = process.stdout.read(1)
            if trailing:
                raise RuntimeError(
                    "DLSS-NR raw output contains unexpected trailing bytes"
                )
        except BaseException as exc:
            errors.put(exc)
            stop.set()

    def stderr_reader() -> None:
        try:
            while True:
                chunk = process.stderr.read(4096)
                if not chunk:
                    return
                if len(stderr_prefix) < 16384:
                    stderr_prefix.extend(chunk[: 16384 - len(stderr_prefix)])
                stderr_chunks.append(chunk)
                if sum(map(len, stderr_chunks)) > 65536:
                    del stderr_chunks[0]
        except (OSError, ValueError):
            return

    threads = [
        threading.Thread(target=writer, name="t8-dlss-nr-writer", daemon=True),
        threading.Thread(target=reader, name="t8-dlss-nr-reader", daemon=True),
        threading.Thread(target=stderr_reader, name="t8-dlss-nr-stderr", daemon=True),
    ]
    for thread in threads:
        thread.start()
    collected: dict[int, bytes] = {}
    consumed_count = 0
    started = time.monotonic()
    check = interrupt_check or _default_interrupt_check
    try:
        while consumed_count < frame_count:
            check()
            if time.monotonic() - started > float(timeout_seconds):
                raise RuntimeError("DLSS-NR raw video process timed out")
            if not errors.empty():
                raise errors.get()
            try:
                frame_index, data = outputs.get(timeout=0.1)
                if frame_index != consumed_count:
                    raise RuntimeError("DLSS-NR raw output frame order changed")
                if output_consumer is None:
                    collected[frame_index] = data
                else:
                    output_consumer(frame_index, data)
                consumed_count += 1
            except queue.Empty:
                if process.poll() is not None and not threads[1].is_alive():
                    break
        if not errors.empty():
            raise errors.get()
        remaining = max(0.1, float(timeout_seconds) - (time.monotonic() - started))
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "DLSS-NR raw video process timed out while exiting"
            ) from exc
        for thread in threads:
            thread.join(timeout=2)
        if not errors.empty():
            raise errors.get()
        if returncode != 0:
            stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"DLSS-NR raw video helper failed with exit code {returncode}: {stderr[-4000:]}"
            )
        stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        if "finishing the rest bilinear" in stderr.lower():
            raise RuntimeError(
                "DLSS SR was refused and upstream attempted a bilinear fallback; candidate rejected"
            )
        motion_diagnostics = bytes(stderr_prefix).decode("utf-8", errors="replace")
        resolved_motion_engine = _resolve_motion_backend(
            motion_diagnostics, requested_motion_engine
        )
        if consumed_count != frame_count:
            raise RuntimeError(
                f"DLSS-NR raw video returned {consumed_count} frames, expected {frame_count}"
            )
        raw_output = (
            [collected[index] for index in range(frame_count)]
            if output_consumer is None
            else []
        )
        return raw_output, {
            "returncode": returncode,
            "motion_engine_requested": requested_motion_engine,
            "motion_engine_resolved": resolved_motion_engine,
            "stderr_tail": stderr[-8192:],
        }
    finally:
        stop.set()
        _terminate_process(process)
        for thread in threads:
            thread.join(timeout=2)


def process_video_frame_batch(
    runtime: Mapping[str, Any],
    frames: torch.Tensor,
    *,
    fps: float,
    audio: Any = None,
    mode: str,
    scale: float,
    motion_engine: str = "auto",
    quality_profile: str = "standard",
    sr_preset: str = "default",
    manual_parameters: Mapping[str, Any] | None = None,
    process_factory=None,
    interrupt_check: Callable[[], None] | None = None,
    timeout_seconds: float = MEDIA_TIMEOUT_SECONDS,
) -> tuple[torch.Tensor, torch.Tensor, Any, dict[str, Any]]:
    frame_count, height, width, channels = _validate_frames(frames)
    if not math.isfinite(float(fps)) or not 0.0 < float(fps) <= 240.0:
        raise ValueError("fps must be finite and within 0..240")
    validate_processing_contract(mode, scale)
    target_width, target_height = target_dimensions(width, height, scale)
    _root, runtime_handle = _validated_runtime_handle(runtime)
    requested_profile, effective_profile = effective_quality_profile(
        mode, quality_profile, manual_parameters
    )
    sr_preset = validate_sr_preset(sr_preset)
    command = build_raw_video_command(
        runtime,
        width=width,
        height=height,
        frame_count=frame_count,
        mode=mode,
        scale=scale,
        motion_engine=motion_engine,
        quality_profile=quality_profile,
        sr_preset=sr_preset,
        manual_parameters=manual_parameters,
    )

    def inputs():
        for frame in frames:
            yield _rgba8_numpy(frame).tobytes()

    started = time.perf_counter()
    raw_outputs, process_report = _stream_raw_frames(
        command,
        inputs(),
        frame_count=frame_count,
        output_frame_bytes=target_width * target_height * 4,
        requested_motion_engine=motion_engine,
        process_factory=process_factory,
        interrupt_check=interrupt_check,
        timeout_seconds=timeout_seconds,
    )
    rgb = []
    for raw in raw_outputs:
        rgba = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(
            target_height, target_width, 4
        )
        rgb.append(rgba[..., :3].to(dtype=torch.float32).div_(255.0))
    candidate = torch.stack(rgb, dim=0).to(device=frames.device, dtype=frames.dtype)
    if channels > 3:
        candidate = torch.cat(
            (
                candidate,
                resize_extra_channels(frames[..., 3:], target_height, target_width),
            ),
            dim=-1,
        )
    if tuple(candidate.shape) != (frame_count, target_height, target_width, channels):
        raise RuntimeError("DLSS-NR raw video output shape is inconsistent")
    report = {
        "schema": "t8.dlss_nr.video_frames.v1",
        "status": "EXP_CANDIDATE_GENERATED",
        "mode": mode,
        "scale": float(scale),
        "runtime_version": str(runtime_handle["runtime_version"]),
        "quality_profile": quality_profile,
        "requested_nr_parameters": requested_profile,
        "nr_parameters": effective_profile,
        "sr_only_execution": _sr_only_execution_report(mode),
        "sr_preset": sr_preset,
        "motion_engine": process_report["motion_engine_resolved"],
        "motion_engine_requested": motion_engine,
        "motion_engine_resolved": process_report["motion_engine_resolved"],
        "fps": float(fps),
        "input_frame_count": frame_count,
        "output_frame_count": int(candidate.shape[0]),
        "input_shape": list(frames.shape),
        "output_shape": list(candidate.shape),
        "frame_order_exact": True,
        "single_persistent_process": True,
        "rgb_bridge": "rounded_uint8_raw_rgba",
        "extra_channels": "bilinear_resize_source_then_reattach",
        "audio_object_identity_preserved": True,
        "no_frame_interpolation": True,
        "no_bilinear_rgb_fallback": True,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "process": process_report,
    }
    return candidate, frames, audio, report


def validate_cfr_pts(
    pts: list[int], time_base: Fraction, rate: Fraction
) -> dict[str, Any]:
    if not pts or any(value is None for value in pts):
        raise ValueError("source VIDEO must provide a PTS for every frame")
    if len(pts) > 1:
        deltas = [later - earlier for earlier, later in zip(pts, pts[1:])]
        if any(delta <= 0 for delta in deltas) or len(set(deltas)) != 1:
            raise ValueError("source VIDEO is VFR or has non-uniform frame timestamps")
        actual_period = Fraction(deltas[0]) * Fraction(time_base)
        expected_period = Fraction(1, 1) / Fraction(rate)
        if actual_period != expected_period:
            raise ValueError(
                "source VIDEO timestamps do not match its declared CFR rate"
            )
        delta = deltas[0]
    else:
        delta = None
    return {
        "frame_count": len(pts),
        "first_pts": pts[0],
        "last_pts": pts[-1],
        "constant_pts_delta": delta,
        "time_base": str(time_base),
        "rate": str(rate),
        "cfr": True,
    }


def _timestamp_seconds(value: Any, time_base: Any) -> str | None:
    if value is None:
        return None
    if time_base is None:
        raise RuntimeError("encoded media timestamp has no time base")
    return str(Fraction(int(value), 1) * Fraction(time_base))


def _audio_packet_digests(path: Path) -> list[dict[str, Any]]:
    import av

    with av.open(str(path), mode="r") as container:
        stream_count = len(container.streams.audio)
    reports = []
    for position in range(stream_count):
        digest = hashlib.sha256()
        timeline_digest = hashlib.sha256()
        packet_count = 0
        payload_bytes = 0
        first_packet_pts_seconds = None
        last_packet_end_seconds = None
        with av.open(str(path), mode="r") as container:
            stream = container.streams.audio[position]
            codec_name = str(stream.codec.name)
            stream_time_base = Fraction(stream.time_base)
            stream_start_time_seconds = _timestamp_seconds(
                stream.start_time, stream_time_base
            )
            stream_duration_seconds = _timestamp_seconds(
                stream.duration, stream_time_base
            )
            for packet in container.demux(stream):
                if packet.dts is None:
                    continue
                payload = bytes(packet)
                digest.update(len(payload).to_bytes(8, "little"))
                digest.update(payload)
                packet_time_base = packet.time_base or stream_time_base
                pts_seconds = _timestamp_seconds(packet.pts, packet_time_base)
                dts_seconds = _timestamp_seconds(packet.dts, packet_time_base)
                duration_seconds = _timestamp_seconds(packet.duration, packet_time_base)
                timeline_row = {
                    "packet_index": packet_count,
                    "pts_seconds": pts_seconds,
                    "dts_seconds": dts_seconds,
                    "duration_seconds": duration_seconds,
                }
                timeline_digest.update(canonical_json(timeline_row).encode("utf-8"))
                timeline_digest.update(b"\n")
                if first_packet_pts_seconds is None:
                    first_packet_pts_seconds = pts_seconds
                if packet.pts is not None and packet.duration is not None:
                    last_packet_end_seconds = _timestamp_seconds(
                        int(packet.pts) + int(packet.duration), packet_time_base
                    )
                packet_count += 1
                payload_bytes += len(payload)
        reports.append(
            {
                "stream_position": position,
                "codec": codec_name,
                "packet_count": packet_count,
                "payload_bytes": payload_bytes,
                "payload_sha256": digest.hexdigest(),
                "packet_timeline_sha256": timeline_digest.hexdigest(),
                "first_packet_pts_seconds": first_packet_pts_seconds,
                "last_packet_end_seconds": last_packet_end_seconds,
                "stream_time_base": str(stream_time_base),
                "stream_start_time_seconds": stream_start_time_seconds,
                "stream_duration_seconds": stream_duration_seconds,
            }
        )
    return reports


def _audio_pcm_digests(path: Path) -> list[dict[str, Any]]:
    import av

    with av.open(str(path), mode="r") as container:
        stream_count = len(container.streams.audio)
    reports = []
    for position in range(stream_count):
        digest = hashlib.sha256()
        timeline_digest = hashlib.sha256()
        frame_count = 0
        sample_count = 0
        first_frame_pts_seconds = None
        last_frame_end_seconds = None
        with av.open(str(path), mode="r") as container:
            stream = container.streams.audio[position]
            for frame in container.decode(stream):
                if (
                    frame.pts is None
                    or frame.time_base is None
                    or not frame.sample_rate
                ):
                    raise RuntimeError(
                        "decoded audio frame has no complete PTS/time-base/sample-rate timeline"
                    )
                array = frame.to_ndarray()
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(bytes(str(tuple(array.shape)), "ascii"))
                digest.update(array.tobytes())
                pts_seconds = _timestamp_seconds(frame.pts, frame.time_base)
                duration = Fraction(int(frame.samples), int(frame.sample_rate))
                end_seconds = str(Fraction(pts_seconds) + duration)
                timeline_row = {
                    "frame_index": frame_count,
                    "pts_seconds": pts_seconds,
                    "duration_seconds": str(duration),
                    "sample_rate": int(frame.sample_rate),
                    "samples": int(frame.samples),
                }
                timeline_digest.update(canonical_json(timeline_row).encode("utf-8"))
                timeline_digest.update(b"\n")
                if first_frame_pts_seconds is None:
                    first_frame_pts_seconds = pts_seconds
                last_frame_end_seconds = end_seconds
                frame_count += 1
                sample_count += int(frame.samples)
        reports.append(
            {
                "stream_position": position,
                "decoded_frame_count": frame_count,
                "decoded_sample_count": sample_count,
                "pcm_sha256": digest.hexdigest(),
                "decoded_timeline_sha256": timeline_digest.hexdigest(),
                "first_frame_pts_seconds": first_frame_pts_seconds,
                "last_frame_end_seconds": last_frame_end_seconds,
            }
        )
    return reports


def _identity_fields(
    reports: list[dict[str, Any]], fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    return [{field: report.get(field) for field in fields} for report in reports]


def _validate_audio_identity(
    source_packets: list[dict[str, Any]],
    output_packets: list[dict[str, Any]],
    source_pcm: list[dict[str, Any]],
    output_pcm: list[dict[str, Any]],
) -> dict[str, bool]:
    packet_payload_fields = (
        "stream_position",
        "codec",
        "packet_count",
        "payload_bytes",
        "payload_sha256",
    )
    packet_timeline_fields = (
        "stream_position",
        "packet_count",
        "packet_timeline_sha256",
        "first_packet_pts_seconds",
        "last_packet_end_seconds",
    )
    pcm_content_fields = (
        "stream_position",
        "decoded_frame_count",
        "decoded_sample_count",
        "pcm_sha256",
    )
    pcm_timeline_fields = (
        "stream_position",
        "decoded_frame_count",
        "decoded_sample_count",
        "decoded_timeline_sha256",
        "first_frame_pts_seconds",
        "last_frame_end_seconds",
    )
    if _identity_fields(source_packets, packet_payload_fields) != _identity_fields(
        output_packets, packet_payload_fields
    ):
        raise RuntimeError(
            "DLSS-NR output audio packet payloads differ from the source"
        )
    if _identity_fields(source_packets, packet_timeline_fields) != _identity_fields(
        output_packets, packet_timeline_fields
    ):
        raise RuntimeError(
            "DLSS-NR output audio packet timestamps differ from the source"
        )
    if _identity_fields(source_pcm, pcm_content_fields) != _identity_fields(
        output_pcm, pcm_content_fields
    ):
        raise RuntimeError("DLSS-NR output decoded audio PCM differs from the source")
    if _identity_fields(source_pcm, pcm_timeline_fields) != _identity_fields(
        output_pcm, pcm_timeline_fields
    ):
        raise RuntimeError(
            "DLSS-NR output decoded audio timestamps differ from the source"
        )
    return {
        "audio_packet_payload_exact": True,
        "audio_packet_timeline_exact": True,
        "audio_decoded_pcm_exact": True,
        "audio_decoded_timeline_exact": True,
    }


def _file_source_contract(source_video) -> tuple[Path, dict[str, Any]]:
    from .skin_finish_p1 import _file_video_source_path, _validate_sdr_video_stream

    source_path = _file_video_source_path(source_video)
    frame_count = int(source_video.get_frame_count())
    width, height = map(int, source_video.get_dimensions())
    bit_depth = int(source_video.get_bit_depth())
    if frame_count < 1 or width < 2 or height < 2:
        raise ValueError("source VIDEO has invalid frame count or geometry")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("FFprobe is required for the strict DLSS-NR file contract")
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-of",
            "json",
            str(source_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=30,
        shell=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if probe.returncode:
        raise ValueError(
            "FFprobe could not validate the source VIDEO: " + probe.stderr[-2000:]
        )
    try:
        probe_streams = json.loads(probe.stdout)["streams"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("FFprobe returned malformed source VIDEO metadata") from exc
    if len(probe_streams) != 1:
        raise ValueError(
            "FFprobe did not find exactly one selected source video stream"
        )
    probe_stream = probe_streams[0]
    for field in ("crop_top", "crop_bottom", "crop_left", "crop_right"):
        try:
            crop_value = int(probe_stream.get(field, 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"source VIDEO has malformed {field} metadata") from exc
        if crop_value != 0:
            raise ValueError(
                "cropped VIDEO metadata is outside the first DLSS-NR file contract"
            )
    rotate_tag = str(probe_stream.get("tags", {}).get("rotate", "0"))
    if rotate_tag not in {"", "0", "0.0"}:
        raise ValueError("rotated VIDEO is outside the first DLSS-NR file contract")
    for side_data in probe_stream.get("side_data_list", []) or []:
        try:
            rotation = float(side_data.get("rotation", 0) or 0)
        except (TypeError, ValueError):
            rotation = 1.0
        if rotation != 0.0:
            raise ValueError("rotated VIDEO is outside the first DLSS-NR file contract")
    if str(probe_stream.get("sample_aspect_ratio", "1:1")) not in {"1:1", "N/A"}:
        raise ValueError(
            "non-square sample aspect ratio is outside the DLSS-NR file contract"
        )
    if probe_stream.get("r_frame_rate") != probe_stream.get("avg_frame_rate"):
        raise ValueError(
            "source VIDEO declares unequal real/average rates and is not strict CFR"
        )

    import av

    with av.open(str(source_path), mode="r") as container:
        if len(container.streams.video) != 1:
            raise ValueError("source VIDEO must contain exactly one video stream")
        stream = container.streams.video[0]
        if (int(stream.width), int(stream.height)) != (width, height):
            raise ValueError("ComfyUI VIDEO geometry differs from its encoded stream")
        if getattr(stream, "rotation", 0):
            raise ValueError("rotated VIDEO is outside the first DLSS-NR file contract")
        sar = getattr(stream, "sample_aspect_ratio", None)
        if sar is not None and Fraction(sar) != Fraction(1, 1):
            raise ValueError(
                "non-square sample aspect ratio is outside the DLSS-NR file contract"
            )
        sdr = _validate_sdr_video_stream(stream, reported_bit_depth=bit_depth)
        if stream.average_rate is None or Fraction(stream.average_rate) <= 0:
            raise ValueError("source VIDEO has no finite declared CFR rate")
        rate = Fraction(stream.average_rate)
        stream_time_base = Fraction(stream.time_base)
        decoded_pts = []
        for frame in container.decode(stream):
            if frame.width != width or frame.height != height:
                raise ValueError("source VIDEO changes geometry during strict decode")
            if frame.pts is None:
                raise ValueError("source VIDEO must provide a PTS for every frame")
            decoded_pts.append(int(frame.pts))
    if len(decoded_pts) != frame_count:
        raise ValueError(
            f"source VIDEO strictly decoded {len(decoded_pts)} frames, expected {frame_count}"
        )
    cfr = validate_cfr_pts(decoded_pts, stream_time_base, rate)
    reported_rate = None
    if hasattr(source_video, "get_frame_rate"):
        reported_rate = float(source_video.get_frame_rate())
        if not math.isclose(reported_rate, float(rate), rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError("ComfyUI VIDEO frame rate differs from its encoded stream")
    packet_digests = _audio_packet_digests(source_path)
    for audio in packet_digests:
        if audio["codec"] not in {"aac", "mp3", "alac", "ac3", "eac3"}:
            raise ValueError(
                f"audio codec {audio['codec']!r} is not approved for MP4 packet-copy"
            )
    pcm_digests = _audio_pcm_digests(source_path)
    return source_path, {
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "rate": rate,
        "time_base": stream_time_base,
        "pts": decoded_pts,
        "reported_rate": reported_rate,
        "sdr": sdr,
        "cfr": cfr,
        "audio_packets": packet_digests,
        "audio_pcm": pcm_digests,
        "ffprobe": {
            "pixel_format": probe_stream.get("pix_fmt"),
            "real_frame_rate": probe_stream.get("r_frame_rate"),
            "average_frame_rate": probe_stream.get("avg_frame_rate"),
            "frame_count": probe_stream.get("nb_frames"),
            "sample_aspect_ratio": probe_stream.get("sample_aspect_ratio"),
            "crop": [
                int(probe_stream.get(field, 0) or 0)
                for field in ("crop_top", "crop_bottom", "crop_left", "crop_right")
            ],
            "rotation": 0,
        },
    }


def _packet_copy_video_and_audio(video_only: Path, source: Path, target: Path) -> None:
    import av

    with (
        av.open(str(video_only), mode="r") as video_container,
        av.open(str(source), mode="r") as audio_container,
        av.open(
            str(target),
            mode="w",
            format="mp4",
            options={"movflags": "use_metadata_tags+faststart"},
        ) as output,
    ):
        if len(video_container.streams.video) != 1:
            raise RuntimeError(
                "DLSS-NR video-only temporary has no unique video stream"
            )
        output_video = output.add_stream_from_template(
            video_container.streams.video[0], opaque=True
        )
        audio_map = {
            stream: output.add_stream_from_template(stream, opaque=True)
            for stream in audio_container.streams.audio
        }
        for key, value in audio_container.metadata.items():
            output.metadata[str(key)] = str(value)
        for packet in video_container.demux(video_container.streams.video[0]):
            if packet.dts is None:
                continue
            packet.stream = output_video
            output.mux(packet)
        for packet in audio_container.demux(*audio_map):
            if packet.stream not in audio_map or packet.dts is None:
                continue
            packet.stream = audio_map[packet.stream]
            output.mux(packet)


def _validate_final_file(
    path: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
    rate: Fraction,
    source_audio_packets: list[dict[str, Any]],
    source_audio_pcm: list[dict[str, Any]],
) -> dict[str, Any]:
    from .skin_finish_p1 import _strict_validate_encoded_video

    import av

    pts: list[int] = []
    audio_decoded_frames = 0
    with av.open(str(path), mode="r") as container:
        if len(container.streams.video) != 1:
            raise RuntimeError("DLSS-NR output must contain exactly one video stream")
        stream = container.streams.video[0]
        if (int(stream.width), int(stream.height)) != (width, height):
            raise RuntimeError(
                "DLSS-NR output geometry does not match the exact target"
            )
        if stream.average_rate is None or Fraction(stream.average_rate) != rate:
            raise RuntimeError(
                "DLSS-NR output frame rate differs from the source CFR rate"
            )
        time_base = Fraction(stream.time_base)
        for packet in container.demux():
            for frame in packet.decode():
                if packet.stream.type == "video":
                    if (
                        frame.width != width
                        or frame.height != height
                        or frame.pts is None
                    ):
                        raise RuntimeError(
                            "DLSS-NR output has invalid video frame geometry or PTS"
                        )
                    pts.append(int(frame.pts))
                elif packet.stream.type == "audio":
                    audio_decoded_frames += 1
    if len(pts) != frame_count:
        raise RuntimeError(
            f"DLSS-NR output decoded {len(pts)} frames, expected {frame_count}"
        )
    cfr = validate_cfr_pts(pts, time_base, rate)
    output_packets = _audio_packet_digests(path)
    output_pcm = _audio_pcm_digests(path)
    audio_identity = _validate_audio_identity(
        source_audio_packets,
        output_packets,
        source_audio_pcm,
        output_pcm,
    )
    _strict_validate_encoded_video(path)
    return {
        "decoded_video_frames": len(pts),
        "joint_audio_frames": audio_decoded_frames,
        "cfr": cfr,
        "audio_packets": output_packets,
        "audio_pcm": output_pcm,
        **audio_identity,
        "strict_ffmpeg_decode": True,
    }


def process_video_file(
    runtime: Mapping[str, Any],
    source_video,
    *,
    output_path: str | os.PathLike[str],
    mode: str,
    scale: float,
    motion_engine: str = "auto",
    quality_profile: str = "standard",
    sr_preset: str = "default",
    manual_parameters: Mapping[str, Any] | None = None,
    crf: float = 18.0,
    process_factory=None,
    interrupt_check: Callable[[], None] | None = None,
    timeout_seconds: float = MEDIA_TIMEOUT_SECONDS,
) -> tuple[Path, Any, dict[str, Any]]:
    validate_processing_contract(mode, scale)
    _root, runtime_handle = _validated_runtime_handle(runtime)
    requested_profile, effective_profile = effective_quality_profile(
        mode, quality_profile, manual_parameters
    )
    sr_preset = validate_sr_preset(sr_preset)
    source_path, source_contract = _file_source_contract(source_video)
    frame_count = int(source_contract["frame_count"])
    width = int(source_contract["width"])
    height = int(source_contract["height"])
    rate = Fraction(source_contract["rate"])
    target_width, target_height = target_dimensions(width, height, scale)
    if target_width % 2 or target_height % 2:
        raise ValueError(
            "DLSS-NR MP4 output dimensions must be even; choose another source or scale"
        )
    if not 0.0 <= float(crf) <= 51.0:
        raise ValueError("crf must stay within 0..51")
    output = Path(output_path).resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("first-version DLSS-NR file output must use .mp4")
    if output == source_path:
        raise ValueError("DLSS-NR output must not overwrite the source VIDEO")
    output.parent.mkdir(parents=True, exist_ok=True)
    video_only = output.with_name(f"{output.stem}.video-only-{uuid.uuid4().hex}.mp4")
    combined = output.with_name(f"{output.stem}.partial-{uuid.uuid4().hex}.mp4")
    metadata: queue.Queue[tuple[int, Fraction]] = queue.Queue()
    command = build_raw_video_command(
        runtime,
        width=width,
        height=height,
        frame_count=frame_count,
        mode=mode,
        scale=scale,
        motion_engine=motion_engine,
        quality_profile=quality_profile,
        sr_preset=sr_preset,
        manual_parameters=manual_parameters,
    )
    started = time.perf_counter()
    encoded_count = 0
    copied_color = None
    try:
        import av
        import numpy as np

        def raw_inputs():
            with av.open(str(source_path), mode="r") as container:
                stream = container.streams.video[0]
                for source_frame in container.decode(stream):
                    if source_frame.pts is None:
                        raise ValueError(
                            "source VIDEO lost PTS during the execution decode"
                        )
                    metadata.put(
                        (int(source_frame.pts), Fraction(source_frame.time_base))
                    )
                    yield source_frame.to_ndarray(format="rgba").tobytes()

        with (
            av.open(str(source_path), mode="r") as metadata_container,
            av.open(
                str(video_only),
                mode="w",
                format="mp4",
                options={"movflags": "use_metadata_tags+faststart"},
            ) as video_output,
        ):
            source_stream = metadata_container.streams.video[0]
            out_video = video_output.add_stream("libx264", rate=rate)
            out_video.width = target_width
            out_video.height = target_height
            out_video.pix_fmt = "yuv420p"
            out_video.codec_context.max_b_frames = 0
            out_video.codec_context.thread_count = 1
            out_video.codec_context.time_base = Fraction(source_stream.time_base)
            out_video.options = {
                "crf": _float_arg(crf),
                "preset": "medium",
                "threads": "1",
            }
            from .skin_finish_p1 import _copy_sdr_color_metadata

            copied_color = _copy_sdr_color_metadata(
                source_stream.codec_context, out_video.codec_context
            )

            def consume(frame_index: int, raw: bytes) -> None:
                nonlocal encoded_count
                if frame_index != encoded_count:
                    raise RuntimeError("DLSS-NR file output frame order changed")
                try:
                    pts, frame_time_base = metadata.get(timeout=5)
                except queue.Empty as exc:
                    raise RuntimeError(
                        "DLSS-NR output outran source frame metadata"
                    ) from exc
                rgba = np.frombuffer(raw, dtype=np.uint8).reshape(
                    target_height, target_width, 4
                )
                target = av.VideoFrame.from_ndarray(rgba, format="rgba")
                target.pts = pts
                target.time_base = frame_time_base
                for packet in out_video.encode(target):
                    video_output.mux(packet)
                encoded_count += 1

            _raw, process_report = _stream_raw_frames(
                command,
                raw_inputs(),
                frame_count=frame_count,
                output_frame_bytes=target_width * target_height * 4,
                requested_motion_engine=motion_engine,
                process_factory=process_factory,
                interrupt_check=interrupt_check,
                timeout_seconds=timeout_seconds,
                output_consumer=consume,
            )
            for packet in out_video.encode():
                video_output.mux(packet)
        if encoded_count != frame_count or not metadata.empty():
            raise RuntimeError(
                "DLSS-NR file stream did not preserve exact frame accounting"
            )
        _packet_copy_video_and_audio(video_only, source_path, combined)
        validation = _validate_final_file(
            combined,
            frame_count=frame_count,
            width=target_width,
            height=target_height,
            rate=rate,
            source_audio_packets=source_contract["audio_packets"],
            source_audio_pcm=source_contract["audio_pcm"],
        )
        os.replace(combined, output)
    except Exception:
        for temporary in (video_only, combined):
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    finally:
        try:
            video_only.unlink(missing_ok=True)
        except OSError:
            pass
    report = {
        "schema": "t8.dlss_nr.video_file.v1",
        "status": "EXP_CANDIDATE_ATOMICALLY_PUBLISHED",
        "mode": mode,
        "scale": float(scale),
        "runtime_version": str(runtime_handle["runtime_version"]),
        "quality_profile": quality_profile,
        "requested_nr_parameters": requested_profile,
        "nr_parameters": effective_profile,
        "sr_only_execution": _sr_only_execution_report(mode),
        "sr_preset": sr_preset,
        "motion_engine": process_report["motion_engine_resolved"],
        "motion_engine_requested": motion_engine,
        "motion_engine_resolved": process_report["motion_engine_resolved"],
        "source_path": str(source_path),
        "output_path": str(output),
        "source": {
            key: value
            for key, value in source_contract.items()
            if key not in {"pts", "rate", "time_base"}
        },
        "video": {
            "input_frame_count": frame_count,
            "output_frame_count": encoded_count,
            "input_dimensions": [width, height],
            "output_dimensions": [target_width, target_height],
            "fps": float(rate),
            "duration_seconds": float(Fraction(frame_count, 1) / rate),
            "codec": "libx264",
            "crf": float(crf),
            "copied_sdr_color_metadata": copied_color,
            "single_persistent_dlss_process": True,
            "full_image_batch_materialized": False,
        },
        "audio": {
            "method": "source_packet_payload_copy",
            "packet_payload_exact": validation["audio_packet_payload_exact"],
            "packet_timeline_exact": validation["audio_packet_timeline_exact"],
            "decoded_pcm_exact": validation["audio_decoded_pcm_exact"],
            "decoded_timeline_exact": validation["audio_decoded_timeline_exact"],
            "reencoded": False,
            "source_packets": source_contract["audio_packets"],
            "source_pcm": source_contract["audio_pcm"],
        },
        "validation": validation,
        "atomic_publish": True,
        "source_overwritten": False,
        "no_bilinear_rgb_fallback": True,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "process": process_report,
    }
    return output, source_video, report
