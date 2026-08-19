#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
from typing import Any, BinaryIO, Mapping
import urllib.request


OUTPUT_SCHEMA = "minimax_h3_speed_external_corpus_fetch_v1"
DEFAULT_DATASET_ID = "Vchitect/Vchitect_T2V_DataVerse"
DEFAULT_REVISION = "e068be25f4d06a837992a1e9096fd00105c83f2c"
DEFAULT_SHARD = "00000/000000.tar"
DEFAULT_SHARD_LFS_OID = (
    "8f73aa683f527fff5221327d334e415e1b4715658f6c2f9bd152827b5de8f339"
)
DEFAULT_LICENSE = "apache-2.0"
MAX_MEMBER_BYTES = 512 * 1024 * 1024
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


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


def _safe_output_name(*, member_name: str, revision: str, shard: str) -> str:
    pure = PurePosixPath(member_name)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Unsafe archive member path: {member_name}")
    name = pure.name
    suffix = Path(name).suffix.lower()
    if not name or suffix not in VIDEO_EXTENSIONS:
        raise ValueError(f"Archive member is not a supported video: {member_name}")
    stem = Path(name).stem
    if not stem.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"Archive member has an unsafe filename: {member_name}")
    shard_stem = Path(PurePosixPath(shard).name).stem
    return f"vchitect_{revision[:8]}_{shard_stem}_{stem}{suffix}"


def extract_video_members(
    archive_stream: BinaryIO,
    *,
    target_root: Path,
    maximum_videos: int,
    revision: str,
    shard: str,
    skipped_members: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    maximum_videos = int(maximum_videos)
    if maximum_videos < 100 or maximum_videos > 5000:
        raise ValueError("maximum_videos must be in [100, 5000]")
    target_root = target_root.resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    with tarfile.open(fileobj=archive_stream, mode="r|*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            suffix = Path(PurePosixPath(member.name).name).suffix.lower()
            if suffix not in VIDEO_EXTENSIONS:
                continue
            if member.size <= 0 or member.size > MAX_MEMBER_BYTES:
                if skipped_members is not None:
                    skipped_members.append(
                        {
                            "member": member.name,
                            "bytes": member.size,
                            "reason": "video_size_outside_safe_bound",
                        }
                    )
                continue
            output_name = _safe_output_name(
                member_name=member.name,
                revision=revision,
                shard=shard,
            )
            if output_name.casefold() in seen_names:
                raise ValueError(f"Archive contains a duplicate output filename: {output_name}")
            seen_names.add(output_name.casefold())
            target = target_root / output_name
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Unable to read archive video member: {member.name}")
            if target.exists():
                if target.stat().st_size != member.size:
                    raise FileExistsError(
                        f"Existing target has a different size; refusing overwrite: {target}"
                    )
                records.append(
                    {
                        "member": member.name,
                        "file": target.name,
                        "bytes": target.stat().st_size,
                        "sha256": _sha256(target),
                        "action": "reused_existing",
                    }
                )
            else:
                descriptor, temporary = tempfile.mkstemp(
                    prefix=f".{output_name}.", suffix=".part", dir=target_root
                )
                digest = hashlib.sha256()
                written = 0
                try:
                    with os.fdopen(descriptor, "wb") as handle:
                        while True:
                            block = source.read(1024 * 1024)
                            if not block:
                                break
                            written += len(block)
                            if written > member.size or written > MAX_MEMBER_BYTES:
                                raise ValueError(
                                    f"Archive member exceeded its declared safe size: {member.name}"
                                )
                            digest.update(block)
                            handle.write(block)
                        handle.flush()
                        os.fsync(handle.fileno())
                    if written != member.size:
                        raise ValueError(
                            f"Archive member was truncated: {member.name} ({written}/{member.size})"
                        )
                    os.replace(temporary, target)
                except BaseException:
                    try:
                        os.unlink(temporary)
                    except OSError:
                        pass
                    raise
                records.append(
                    {
                        "member": member.name,
                        "file": target.name,
                        "bytes": written,
                        "sha256": digest.hexdigest().upper(),
                        "action": "extracted",
                    }
                )
            if len(records) >= maximum_videos:
                break
    if len(records) != maximum_videos:
        raise ValueError(
            f"Shard ended before {maximum_videos} videos were available; got {len(records)}"
        )
    return records


def fetch_corpus(
    *,
    target_root: Path,
    output_report: Path,
    maximum_videos: int,
    dataset_id: str = DEFAULT_DATASET_ID,
    revision: str = DEFAULT_REVISION,
    shard: str = DEFAULT_SHARD,
    shard_lfs_oid: str = DEFAULT_SHARD_LFS_OID,
    license_id: str = DEFAULT_LICENSE,
    opener: Any = urllib.request.urlopen,
) -> dict[str, Any]:
    if not dataset_id or len(revision) != 40 or len(shard_lfs_oid) != 64:
        raise ValueError("Dataset id, fixed 40-character revision and 64-character LFS oid are required")
    url = f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/{shard}?download=true"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "minimax-h3-speed-calibration/1.0"},
    )
    skipped_members: list[dict[str, Any]] = []
    with opener(request, timeout=120) as response:
        records = extract_video_members(
            response,
            target_root=target_root,
            maximum_videos=maximum_videos,
            revision=revision,
            shard=shard,
            skipped_members=skipped_members,
        )
    report = {
        "schema": OUTPUT_SCHEMA,
        "dataset": {
            "id": dataset_id,
            "revision": revision,
            "shard": shard,
            "shard_lfs_oid": shard_lfs_oid,
            "license": license_id,
            "source_url": url,
        },
        "target_root": str(target_root.resolve()),
        "stream_stopped_after_selected_videos": True,
        "full_shard_downloaded": False,
        "selected_video_count": len(records),
        "selected_total_bytes": sum(int(row["bytes"]) for row in records),
        "skipped_member_count": len(skipped_members),
        "skipped_members": skipped_members,
        "files": records,
        "usage_boundary": (
            "Local spectrum calibration only. Raw dataset media is not part of the plugin and "
            "must not be committed or redistributed by this tool. Mechanical extraction does "
            "not replace strict decode, duplicate, diversity or content review."
        ),
    }
    _write_json_atomic(output_report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a bounded, revision-pinned VChitect video sample for H3 SPEED spectrum "
            "calibration. The full WebDataset shard is not downloaded."
        )
    )
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--maximum-videos", type=int, default=250)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--shard", default=DEFAULT_SHARD)
    parser.add_argument("--shard-lfs-oid", default=DEFAULT_SHARD_LFS_OID)
    parser.add_argument("--license", dest="license_id", default=DEFAULT_LICENSE)
    args = parser.parse_args()
    report = fetch_corpus(
        target_root=args.target_root,
        output_report=args.output_report,
        maximum_videos=args.maximum_videos,
        dataset_id=args.dataset_id,
        revision=args.revision,
        shard=args.shard,
        shard_lfs_oid=args.shard_lfs_oid,
        license_id=args.license_id,
    )
    print(
        json.dumps(
            {
                "selected_video_count": report["selected_video_count"],
                "selected_total_bytes": report["selected_total_bytes"],
                "full_shard_downloaded": report["full_shard_downloaded"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
