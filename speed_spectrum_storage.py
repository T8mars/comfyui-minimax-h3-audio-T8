from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from safetensors import safe_open
from safetensors.torch import save_file

from .speed_advanced import (
    _spectrum_dataset_public_report,
    _validate_spectrum_dataset,
    canonical_json,
)


SPEED_SPECTRUM_STORAGE_SCHEMA = "minimax_h3_speed_spectrum_dataset_file_t8_v1"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Fingerprint source must be a regular file: {path}")
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"Fingerprint source must be a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest().upper()}"


def _dataset_path(root: Path, dataset_name: str) -> Path:
    raw_name = str(dataset_name)
    name = raw_name.strip()
    if name != raw_name:
        raise ValueError("dataset_name cannot have leading or trailing whitespace")
    if not _SAFE_NAME.fullmatch(name) or name in {".", ".."}:
        raise ValueError(
            "dataset_name must use 1-128 ASCII letters, numbers, dot, underscore or dash"
        )
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.safetensors"
    if path.parent.resolve() != root:
        raise ValueError("Spectrum dataset path escaped its storage root")
    if path.is_symlink():
        raise ValueError("Spectrum dataset files cannot be symbolic links")
    return path


def spectrum_dataset_file_fingerprint(*, root: Path, dataset_name: str) -> str:
    """Return a cache key that changes whenever the dataset file bytes change."""

    path = _dataset_path(root, dataset_name)
    if not path.exists():
        return f"missing:{path}"
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Spectrum dataset source must be a regular file: {path}")
    return f"{path}:{sha256_file(path)}"


def save_spectrum_dataset_file(
    dataset: Mapping[str, Any],
    *,
    root: Path,
    dataset_name: str,
    overwrite: bool,
) -> tuple[dict[str, Any], str, str]:
    _validate_spectrum_dataset(dataset)
    path = _dataset_path(root, dataset_name)
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_descriptor: int | None = None
    temporary: str | None = None
    existed_before = False
    try:
        lock_descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        os.write(lock_descriptor, str(os.getpid()).encode("ascii"))
        os.fsync(lock_descriptor)
        os.close(lock_descriptor)
        lock_descriptor = None
        existed_before = path.exists()
        if existed_before and not overwrite:
            raise FileExistsError(
                f"Spectrum dataset already exists: {path}; enable overwrite explicitly"
            )
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        os.unlink(temporary)
        metadata = _spectrum_dataset_public_report(dataset)
        save_file(
            {"power_sum": dataset["power_sum"].contiguous()},
            temporary,
            metadata={
                "storage_schema": SPEED_SPECTRUM_STORAGE_SCHEMA,
                "dataset_json": canonical_json(metadata, indent=None),
            },
        )
        # Windows rejects fsync on a read-only descriptor; r+b provides the same
        # durability barrier without changing the already written safetensors bytes.
        with open(temporary, "r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        try:
            os.unlink(lock_path)
        except OSError:
            pass
    report = {
        "schema": SPEED_SPECTRUM_STORAGE_SCHEMA,
        "action": "save",
        "path": str(path),
        "overwrote_existing": existed_before,
        "independent_clip_count": int(dataset["independent_clip_count"]),
        "batch_count": int(dataset["batch_count"]),
        "power_sum_sha256": dataset["power_sum_sha256"],
        "source_latents_saved": False,
    }
    return dict(dataset), str(path), canonical_json(report)


def load_spectrum_dataset_file(
    *, root: Path, dataset_name: str
) -> tuple[dict[str, Any], str, str]:
    path = _dataset_path(root, dataset_name)
    if not path.is_file():
        raise FileNotFoundError(f"Spectrum dataset does not exist: {path}")
    if path.is_symlink():
        raise ValueError("Spectrum dataset files cannot be symbolic links")
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"power_sum"}:
            raise ValueError("Spectrum dataset file must contain only power_sum")
        metadata = handle.metadata() or {}
        if metadata.get("storage_schema") != SPEED_SPECTRUM_STORAGE_SCHEMA:
            raise ValueError("Spectrum dataset storage schema mismatch")
        try:
            dataset = json.loads(metadata["dataset_json"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError("Spectrum dataset metadata is missing or malformed") from exc
        dataset["power_sum"] = handle.get_tensor("power_sum")
    _validate_spectrum_dataset(dataset)
    report = {
        "schema": SPEED_SPECTRUM_STORAGE_SCHEMA,
        "action": "load",
        "path": str(path),
        "independent_clip_count": int(dataset["independent_clip_count"]),
        "batch_count": int(dataset["batch_count"]),
        "power_sum_sha256": dataset["power_sum_sha256"],
        "source_latents_loaded": False,
    }
    return dataset, str(path), canonical_json(report)
