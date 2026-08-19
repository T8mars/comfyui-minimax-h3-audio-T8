#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


CURATION_SCHEMA = "minimax_h3_speed_source_curation_v2"
MANIFEST_SCHEMA = "minimax_h3_speed_spectrum_accumulation_manifest_v1"
FETCH_SCHEMA = "minimax_h3_speed_external_corpus_fetch_v1"
DATASET_PROVENANCE_SCHEMA = "minimax_h3_speed_dataset_provenance_t8_v1"
SOURCE_ENTRY_SCHEMA = "minimax_h3_speed_source_entry_t8_v1"
REVIEW_SCHEMA = "minimax_h3_speed_corpus_visual_review_v1"
TASK_FAMILIES = {"T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA", "Hybrid"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _decoded_window_sha256(item: Mapping[str, Any]) -> str:
    direct = item.get("decoded_window_sha256")
    signature = item.get("decoded_signature")
    nested = signature.get("raw_sha256") if isinstance(signature, Mapping) else None
    value = str(direct or nested or "").strip().upper()
    if len(value) != 64 or any(character not in "0123456789ABCDEF" for character in value):
        raise ValueError("A provisional candidate is missing a complete decoded-window SHA-256")
    return value


def _source_set_sha256(rows: list[Mapping[str, Any]]) -> str:
    values = sorted(
        (
            {
                "source_file_sha256": str(row["source_file_sha256"]).upper(),
                "decoded_window_sha256": str(row["decoded_window_sha256"]).upper(),
            }
            for row in rows
        ),
        key=lambda item: (item["source_file_sha256"], item["decoded_window_sha256"]),
    )
    payload = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _external_corpus_provenance(
    fetch_reports: list[tuple[Path, Mapping[str, Any]]],
    *,
    selected_rows: list[Mapping[str, Any]],
    curation_report_sha256: str,
    selection_policy: str,
    independence_reviewed: bool,
    content_diversity_reviewed: bool,
    review_report: tuple[Path, Mapping[str, Any]] | None,
) -> dict[str, Any] | None:
    if not fetch_reports:
        return None
    datasets = []
    fetched_hashes: set[str] = set()
    shards = []
    for path, report in fetch_reports:
        if report.get("schema") != FETCH_SCHEMA:
            raise ValueError(f"Fetch report schema must be {FETCH_SCHEMA}: {path}")
        dataset = report.get("dataset")
        if not isinstance(dataset, Mapping):
            raise ValueError(f"Fetch report is missing dataset provenance: {path}")
        identity = (
            str(dataset.get("id", "")).strip(),
            str(dataset.get("revision", "")).strip(),
            str(dataset.get("license", "")).strip().lower(),
        )
        if not all(identity):
            raise ValueError(f"Fetch report dataset identity is incomplete: {path}")
        datasets.append(identity)
        shard = str(dataset.get("shard", "")).strip().replace("\\", "/")
        lfs_oid = str(dataset.get("shard_lfs_oid", "")).strip().upper()
        if not shard or len(lfs_oid) != 64:
            raise ValueError(f"Fetch report shard identity is incomplete: {path}")
        shards.append(
            {
                "shard": shard,
                "lfs_oid": lfs_oid,
                "fetch_report_sha256": _sha256(path),
            }
        )
        for item in report.get("files", []):
            if isinstance(item, Mapping):
                value = str(item.get("sha256", "")).strip().upper()
                if len(value) == 64:
                    fetched_hashes.add(value)
    if len(set(datasets)) != 1:
        raise ValueError("All fetch reports must describe the same dataset id/revision/license")
    selected_hashes = {str(row["source_file_sha256"]).upper() for row in selected_rows}
    missing = sorted(selected_hashes.difference(fetched_hashes))
    if missing:
        raise ValueError(
            "Selected candidates are not fully covered by the fixed-revision fetch reports"
        )
    if review_report is None:
        if independence_reviewed or content_diversity_reviewed:
            raise ValueError("Reviewed flags require a bound visual review report")
        return None
    review_path, review = review_report
    if review.get("schema") != REVIEW_SCHEMA:
        raise ValueError(f"Review report schema must be {REVIEW_SCHEMA}")
    selected_source_set_sha256 = _source_set_sha256(selected_rows)
    if str(review.get("selected_source_set_sha256", "")).upper() != selected_source_set_sha256:
        raise ValueError("Review report selected source set does not match the manifest selection")
    if int(review.get("selected_source_count", 0)) != len(selected_rows):
        raise ValueError("Review report selected source count does not match the manifest")
    if review.get("independence_reviewed") is not True or not independence_reviewed:
        raise ValueError("Independent-corpus review must be explicit in the report and CLI")
    if review.get("content_diversity_reviewed") is not True or not content_diversity_reviewed:
        raise ValueError("Content-diversity review must be explicit in the report and CLI")
    dataset_id, revision, license_id = datasets[0]
    return {
        "schema": DATASET_PROVENANCE_SCHEMA,
        "source_kind": "independent_natural_video_corpus",
        "dataset_id": dataset_id,
        "dataset_revision": revision,
        "dataset_license": license_id,
        "source_shards": sorted(shards, key=lambda item: item["shard"]),
        "curation_report_sha256": curation_report_sha256,
        "review_report_sha256": _sha256(review_path),
        "selection_policy": selection_policy,
        "selected_source_count": len(selected_rows),
        "selected_source_set_sha256": selected_source_set_sha256,
        "independence_reviewed": bool(independence_reviewed),
        "content_diversity_reviewed": bool(content_diversity_reviewed),
        "raw_media_redistributed": False,
    }


def build_manifest(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    input_root: Path,
    dataset_name: str,
    task_family: str,
    video_vae_name: str,
    checkpoint_fingerprint: str,
    vae_fingerprint: str,
    minimum_formal_clips: int = 100,
    maximum_entries: int = 0,
    selection_policy: str = "sha256_rank",
    fetch_reports: list[tuple[Path, Mapping[str, Any]]] | None = None,
    independence_reviewed: bool = False,
    content_diversity_reviewed: bool = False,
    review_report: tuple[Path, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if report.get("schema") != CURATION_SCHEMA:
        raise ValueError(f"Curation report schema must be {CURATION_SCHEMA}")
    if report.get("mode") != "signature":
        raise ValueError("Only a strict signature-mode curation report can build a manifest")
    if task_family not in TASK_FAMILIES:
        raise ValueError(f"Unsupported task_family: {task_family}")
    if int(minimum_formal_clips) < 100:
        raise ValueError("minimum_formal_clips cannot be lower than 100")
    if int(maximum_entries) < 0:
        raise ValueError("maximum_entries cannot be negative")
    if selection_policy != "sha256_rank":
        raise ValueError("selection_policy must be sha256_rank")
    required_text = {
        "dataset_name": dataset_name,
        "video_vae_name": video_vae_name,
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "vae_fingerprint": vae_fingerprint,
    }
    if any(not str(value).strip() for value in required_text.values()):
        raise ValueError("Dataset name, VAE name and fingerprints must be non-empty")

    input_root = input_root.resolve()
    rows = []
    for item in report.get("items", []):
        if not isinstance(item, Mapping) or item.get("status") != "provisional_candidate":
            continue
        strict_decode = item.get("strict_decode")
        if not isinstance(strict_decode, Mapping) or not strict_decode.get("passed"):
            raise ValueError("A provisional candidate is missing a passing strict decode")
        source = Path(str(item.get("path", ""))).resolve()
        try:
            relative = source.relative_to(input_root)
        except ValueError as exc:
            raise ValueError(f"Candidate is outside the declared ComfyUI input root: {source}") from exc
        file_hash = str(item.get("file_sha256", "")).strip().upper()
        if len(file_hash) != 64:
            raise ValueError(f"Candidate is missing a complete file SHA-256: {source}")
        rows.append(
            {
                "file": relative.as_posix(),
                "batch_id": f"natural_{file_hash[:20].lower()}",
                "source_file_sha256": file_hash,
                "decoded_window_sha256": _decoded_window_sha256(item),
            }
        )
    rows.sort(key=lambda row: (row["source_file_sha256"], row["decoded_window_sha256"]))
    if maximum_entries:
        rows = rows[: int(maximum_entries)]
    if not rows:
        raise ValueError("The curation report has no provisional candidates")
    file_hashes = [row["source_file_sha256"] for row in rows]
    if len(file_hashes) != len(set(file_hashes)):
        raise ValueError("The selected candidates contain an exact file duplicate")
    decoded_hashes = [row["decoded_window_sha256"] for row in rows]
    if len(decoded_hashes) != len(set(decoded_hashes)):
        raise ValueError("The selected candidates contain an exact decoded-window duplicate")
    curation_report_sha256 = _sha256(report_path)
    dataset_provenance = _external_corpus_provenance(
        fetch_reports or [],
        selected_rows=rows,
        curation_report_sha256=curation_report_sha256,
        selection_policy=selection_policy,
        independence_reviewed=independence_reviewed,
        content_diversity_reviewed=content_diversity_reviewed,
        review_report=review_report,
    )
    formal = bool(
        len(rows) >= int(minimum_formal_clips)
        and dataset_provenance is not None
        and independence_reviewed
        and content_diversity_reviewed
    )
    entries = [
        {
            **row,
            "source_entry": {
                "schema": SOURCE_ENTRY_SCHEMA,
                "batch_id": row["batch_id"],
                "source_file_sha256": row["source_file_sha256"],
                "decoded_window_sha256": row["decoded_window_sha256"],
            },
        }
        for row in rows
    ]
    return {
        "schema": MANIFEST_SCHEMA,
        "dataset_name": str(dataset_name).strip(),
        "task_family": task_family,
        "video_vae_name": str(video_vae_name).strip(),
        "checkpoint_fingerprint": str(checkpoint_fingerprint).strip(),
        "vae_fingerprint": str(vae_fingerprint).strip(),
        "width": int(report["policy"]["target_width"]),
        "height": int(report["policy"]["target_height"]),
        "length": int(report["policy"]["required_frames"]),
        "max_temporal_samples": 32,
        "provenance": {
            "source_kind": (
                "independent_natural_video_corpus"
                if dataset_provenance is not None
                else "curated_video_corpus_proxy"
            ),
            "curation_report_sha256": curation_report_sha256,
            "curation_mode": "signature",
            "selected_provisional_candidates": len(rows),
            "selection_policy": selection_policy,
            "selected_source_set_sha256": _source_set_sha256(rows),
            "minimum_formal_clips": int(minimum_formal_clips),
            "formal_dataset_authorized": formal,
            "independence_validated": bool(independence_reviewed),
            "content_diversity_reviewed": bool(content_diversity_reviewed),
            "warning": (
                "Mechanical uniqueness does not prove semantic independence. Formal use also "
                "requires fixed-revision fetch coverage plus explicit independence and diversity review."
            ),
        },
        "dataset_provenance": dataset_provenance,
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a fail-closed H3 SPEED accumulation manifest from a signature curation report."
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--task-family", choices=sorted(TASK_FAMILIES), required=True)
    parser.add_argument("--video-vae-name", required=True)
    parser.add_argument("--checkpoint-fingerprint", required=True)
    parser.add_argument("--vae-fingerprint", required=True)
    parser.add_argument("--minimum-formal-clips", type=int, default=100)
    parser.add_argument("--maximum-entries", type=int, default=0)
    parser.add_argument("--selection-policy", choices=["sha256_rank"], default="sha256_rank")
    parser.add_argument("--fetch-report", type=Path, action="append", default=[])
    parser.add_argument("--independence-reviewed", action="store_true")
    parser.add_argument("--content-diversity-reviewed", action="store_true")
    parser.add_argument("--review-report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    manifest = build_manifest(
        report,
        report_path=args.report,
        input_root=args.input_root,
        dataset_name=args.dataset_name,
        task_family=args.task_family,
        video_vae_name=args.video_vae_name,
        checkpoint_fingerprint=args.checkpoint_fingerprint,
        vae_fingerprint=args.vae_fingerprint,
        minimum_formal_clips=args.minimum_formal_clips,
        maximum_entries=args.maximum_entries,
        selection_policy=args.selection_policy,
        fetch_reports=[
            (path, json.loads(path.read_text(encoding="utf-8")))
            for path in args.fetch_report
        ],
        independence_reviewed=args.independence_reviewed,
        content_diversity_reviewed=args.content_diversity_reviewed,
        review_report=(
            (
                args.review_report,
                json.loads(args.review_report.read_text(encoding="utf-8")),
            )
            if args.review_report is not None
            else None
        ),
    )
    _write_json_atomic(args.output, manifest)
    print(
        json.dumps(
            {
                "entries": len(manifest["entries"]),
                "formal_dataset_authorized": manifest["provenance"][
                    "formal_dataset_authorized"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
