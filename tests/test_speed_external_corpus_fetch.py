from __future__ import annotations

import io
import tarfile

import pytest

from h3_audio_t8_pkg.tools.fetch_h3_speed_calibration_corpus import (
    DEFAULT_REVISION,
    DEFAULT_SHARD,
    extract_video_members,
)


def _archive(*, count: int, unsafe: bool = False, empty_first: bool = False) -> io.BytesIO:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for index in range(count):
            payload = b"" if empty_first and index == 0 else f"video-{index}".encode()
            name = f"./{index:010d}.mp4"
            if unsafe and index == 0:
                name = "../escape.mp4"
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    stream.seek(0)
    return stream


def test_stream_extractor_stops_at_bound_and_reuses_exact_size_files(tmp_path):
    skipped: list[dict] = []
    records = extract_video_members(
        _archive(count=106, empty_first=True),
        target_root=tmp_path,
        maximum_videos=100,
        revision=DEFAULT_REVISION,
        shard=DEFAULT_SHARD,
        skipped_members=skipped,
    )
    assert len(records) == 100
    assert skipped == [
        {
            "member": "./0000000000.mp4",
            "bytes": 0,
            "reason": "video_size_outside_safe_bound",
        }
    ]
    assert all(row["action"] == "extracted" for row in records)
    assert len(list(tmp_path.glob("*.mp4"))) == 100
    repeated = extract_video_members(
        _archive(count=106, empty_first=True),
        target_root=tmp_path,
        maximum_videos=100,
        revision=DEFAULT_REVISION,
        shard=DEFAULT_SHARD,
    )
    assert all(row["action"] == "reused_existing" for row in repeated)


def test_stream_extractor_refuses_unsafe_members_and_small_requested_counts(tmp_path):
    with pytest.raises(ValueError, match=r"\[100, 5000\]"):
        extract_video_members(
            _archive(count=5),
            target_root=tmp_path,
            maximum_videos=5,
            revision=DEFAULT_REVISION,
            shard=DEFAULT_SHARD,
        )
    with pytest.raises(ValueError, match="Unsafe archive member"):
        extract_video_members(
            _archive(count=100, unsafe=True),
            target_root=tmp_path,
            maximum_videos=100,
            revision=DEFAULT_REVISION,
            shard=DEFAULT_SHARD,
        )
