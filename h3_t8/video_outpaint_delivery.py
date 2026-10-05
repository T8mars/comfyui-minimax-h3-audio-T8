"""Persistent, hash-bound delivery records; no claim of a two-file atomic rename."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from .video_outpaint_plan import canonical


MAX_REPORT_BYTES = 8*1024*1024


def delivery_report_path(video_path, *, report_kind="outpaint"):
    if report_kind not in {"outpaint", "postprocess"}:
        raise ValueError("unknown delivery report kind")
    return Path(video_path).with_suffix(f".mp4.{report_kind}.json")


def publish_with_delivery_report(temporary_video, target, report, *, interrupt_check=None, report_kind="outpaint"):
    """Write/fsync report first, then no-replace-link the validated video.

    A crash between links can leave a report with no video: that is explicitly
    not a completed delivery. Normal failures remove only the report link we
    created. Existing files, including orphan reports, are never overwritten.
    """
    temporary_video, target = Path(temporary_video), Path(target)
    sidecar = delivery_report_path(target, report_kind=report_kind)
    if target.exists() or sidecar.exists():
        raise FileExistsError("output video or delivery report already exists; choose a new output name")
    data = {**report, "delivery_report_path": str(sidecar),
            "publication_verification": "matching_video_bytes_required_report_alone_is_not_completion"}
    data["delivery_report_sha256"] = hashlib.sha256(canonical(data).encode()).hexdigest()
    blob = canonical(data).encode()
    if len(blob) > MAX_REPORT_BYTES:
        raise ValueError("outpaint delivery report exceeds the 8MiB metadata limit")
    descriptor, name = tempfile.mkstemp(prefix=f".{target.stem}.receipt-", suffix=".json", dir=target.parent)
    private = Path(name)
    linked = published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        if interrupt_check:
            interrupt_check()
        os.link(private, sidecar)
        linked = True
        if interrupt_check:
            interrupt_check()
        os.link(temporary_video, target)
        published = True
    finally:
        if linked and not published and sidecar.exists() and os.path.samefile(private, sidecar):
            sidecar.unlink()
        private.unlink(missing_ok=True)
    return data


def read_delivery_report(video_path, *, report_kind="outpaint"):
    """Verify the on-disk report and actual video, without trusting its path field.

    This detects stale/damaged/mismatched files, not a maliciously forged report;
    hashes are integrity bindings, not signatures or new perceptual validation.
    """
    video = Path(video_path).resolve(strict=True)
    path = delivery_report_path(video, report_kind=report_kind)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError("outpaint delivery report missing, oversized or not a regular file")
    data = json.loads(path.read_text(encoding="utf-8"))
    digest = data.pop("delivery_report_sha256", None)
    schema = ("t8.h3.video_outpaint.final_file/v1" if report_kind == "outpaint"
              else "t8.h3.postprocess.final_file/v1")
    if (data.get("schema") != schema
            or digest != hashlib.sha256(canonical(data).encode()).hexdigest()
            or Path(data.get("path", "")).resolve() != video
            or Path(data.get("delivery_report_path", "")).resolve() != path):
        raise ValueError("outpaint delivery report integrity/path mismatch")
    actual = hashlib.sha256()
    with video.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            actual.update(block)
    if actual.hexdigest() != data.get("sha256"):
        raise ValueError("outpaint delivered video differs from its report")
    return {**data, "delivery_report_sha256": digest}
