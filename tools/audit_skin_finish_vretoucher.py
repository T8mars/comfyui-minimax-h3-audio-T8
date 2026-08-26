from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


OFFICIAL_FILENAME = "gen_best.pth"
OFFICIAL_SIZE_BYTES = 630_172_363
REPORT_SCHEMA = "h3_t8_vretoucher_weight_audit/v1"
STRUCTURE_REPORT_SCHEMA = "h3_t8_vretoucher_source_structure_audit/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_checkpoint() -> Path:
    configured = os.environ.get("COMFYUI_MODEL_PATH", "").strip()
    if configured:
        model_root = Path(configured)
    else:
        model_root = Path(__file__).resolve().parents[3] / "models"
    return model_root / "facerestore_models" / "VRetouchEr" / OFFICIAL_FILENAME


def _state_structure(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    entries = [
        {
            "key": key,
            "shape": [int(value) for value in tensor.shape],
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
        }
        for key, tensor in state.items()
    ]
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {
        "state_tensor_count": len(entries),
        "state_numel": sum(item["numel"] for item in entries),
        "state_structure_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "entries": entries,
    }


def _load_expected_structure(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path).expanduser().resolve()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read expected structure report {path}: {error}") from error
    if (
        not isinstance(value, dict)
        or value.get("schema") != STRUCTURE_REPORT_SCHEMA
        or value.get("status")
        not in {
            "META_STRUCTURE_PASS_CHECKPOINT_NOT_VALIDATED",
            "META_STRUCTURE_AND_FORWARD_SHAPE_PASS_CHECKPOINT_NOT_VALIDATED",
        }
        or not isinstance(value.get("entries"), list)
        or not isinstance(value.get("state_structure_sha256"), str)
    ):
        raise ValueError(
            "expected structure report must be a completed pinned-source meta audit with full entries"
        )
    return value


def audit_checkpoint(
    checkpoint: Path,
    *,
    expected_size_bytes: int | None = OFFICIAL_SIZE_BYTES,
    expected_sha256: str | None = None,
    expected_structure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(checkpoint).expanduser().resolve()
    base: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "checkpoint": str(path),
        "official_filename": OFFICIAL_FILENAME,
        "official_reference_size_bytes": OFFICIAL_SIZE_BYTES,
        "safe_loader": "torch.load(weights_only=True,map_location=cpu)",
        "model_constructed": False,
        "inference_run": False,
    }
    if not path.is_file():
        return {
            **base,
            "status": "MISSING_CHECKPOINT",
            "detail": "Download the official gen_best.pth manually before model validation.",
        }
    size = path.stat().st_size
    actual_sha256 = _sha256(path)
    base.update({"size_bytes": size, "sha256": actual_sha256})
    if expected_size_bytes is not None and size != int(expected_size_bytes):
        return {
            **base,
            "status": "REJECTED_SIZE_MISMATCH",
            "detail": f"expected {int(expected_size_bytes)} bytes, got {size}",
        }
    expected_hash = str(expected_sha256 or "").strip().lower()
    if expected_hash and actual_sha256.lower() != expected_hash:
        return {
            **base,
            "status": "REJECTED_SHA256_MISMATCH",
            "expected_sha256": expected_hash,
        }
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        return {
            **base,
            "status": "REJECTED_SAFE_LOAD_FAILED",
            "detail": f"{type(error).__name__}: {error}",
        }
    if not isinstance(state, dict) or not state:
        return {
            **base,
            "status": "REJECTED_NOT_A_NONEMPTY_STATE_DICT",
        }
    invalid_keys = [
        str(key)
        for key, value in state.items()
        if not isinstance(key, str) or not isinstance(value, torch.Tensor)
    ]
    if invalid_keys:
        return {
            **base,
            "status": "REJECTED_NON_TENSOR_STATE_DICT_ENTRY",
            "invalid_entry_samples": invalid_keys[:8],
        }
    dtypes: dict[str, int] = {}
    for tensor in state.values():
        name = str(tensor.dtype)
        dtypes[name] = dtypes.get(name, 0) + 1
    observed_structure = _state_structure(state)
    structural = {
        "tensor_count": observed_structure["state_tensor_count"],
        "parameter_numel": observed_structure["state_numel"],
        "state_structure_sha256": observed_structure["state_structure_sha256"],
        "dtype_tensor_counts": dtypes,
        "first_keys": sorted(state)[:8],
        "last_keys": sorted(state)[-8:],
    }
    if expected_structure is not None:
        expected_entries = expected_structure.get("entries")
        expected_structure_sha256 = expected_structure.get("state_structure_sha256")
        expected_tensor_count = expected_structure.get("state_tensor_count")
        if (
            observed_structure["entries"] != expected_entries
            or observed_structure["state_structure_sha256"] != expected_structure_sha256
            or observed_structure["state_tensor_count"] != expected_tensor_count
        ):
            del state
            return {
                **base,
                **structural,
                "status": "REJECTED_STATE_STRUCTURE_MISMATCH",
                "expected_state_structure_sha256": expected_structure_sha256,
                "expected_state_tensor_count": expected_tensor_count,
            }
        structural.update(
            {
                "expected_state_structure_sha256": expected_structure_sha256,
                "exact_state_structure_match": True,
            }
        )
    del state
    if not expected_hash:
        return {
            **base,
            **structural,
            "status": "UNVERIFIED_WEIGHT_HASH_REQUIRED",
            "detail": (
                "The file safely loads as a tensor state dict, but size alone is not identity. "
                "Record a trusted official SHA-256 before model construction."
            ),
        }
    final_status = "STRUCTURE_AND_TRUSTED_HASH_PASS_MODEL_NOT_LOADED"
    if expected_structure is not None:
        final_status = "EXACT_STRUCTURE_AND_TRUSTED_HASH_PASS_MODEL_NOT_LOADED"
    return {
        **base,
        **structural,
        "status": final_status,
        "expected_sha256": expected_hash,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only, weights-only preflight for the official VRetouchEr checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, default=_default_checkpoint())
    parser.add_argument("--expected-sha256", default="")
    parser.add_argument(
        "--expected-structure-report",
        type=Path,
        help="Full report from audit_skin_finish_vretoucher_structure.py",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        expected_structure = _load_expected_structure(args.expected_structure_report)
    except ValueError as error:
        report = {
            "schema": REPORT_SCHEMA,
            "status": "REJECTED_EXPECTED_STRUCTURE_REPORT_INVALID",
            "detail": str(error),
            "checkpoint": str(args.checkpoint),
            "model_constructed": False,
            "inference_run": False,
        }
    else:
        report = audit_checkpoint(
            args.checkpoint,
            expected_sha256=args.expected_sha256 or None,
            expected_structure=expected_structure,
        )
    if args.report is not None:
        _write_json_atomic(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if "STRUCTURE_AND_TRUSTED_HASH_PASS" in report["status"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
