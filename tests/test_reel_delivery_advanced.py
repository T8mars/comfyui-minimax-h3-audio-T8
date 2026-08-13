from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pytest

from h3_audio_t8_pkg.nodes_reel_delivery_advanced import (
    REEL_DELIVERY_ADVANCED_NODE_CLASSES,
)
from h3_audio_t8_pkg.reel_delivery_advanced import (
    build_reel_delivery_plan,
    compose_reel_delivery,
    _cleanup_orphan_temporary_files,
    _reel_execution_lock,
    _cleanup_temporary,
    _unlink_with_retry,
    validate_reel_plan,
)


def _write_clip(path: Path, value: int, tone: float, frames: int = 24) -> None:
    import av

    with av.open(str(path), mode="w", format="mp4") as output:
        video = output.add_stream("libx264", rate=Fraction(24, 1))
        video.width = 64
        video.height = 64
        video.pix_fmt = "yuv420p"
        audio = output.add_stream("aac", rate=48000, layout="stereo")
        for index in range(frames):
            array = np.full((64, 64, 3), value + index % 3, dtype=np.uint8)
            output.mux(video.encode(av.VideoFrame.from_ndarray(array, format="rgb24")))
        output.mux(video.encode(None))
        wave = np.full((2, round(frames / 24 * 48000)), tone, dtype=np.float32)
        frame = av.AudioFrame.from_ndarray(wave, format="fltp", layout="stereo")
        frame.sample_rate = 48000
        frame.time_base = Fraction(1, 48000)
        output.mux(audio.encode(frame))
        output.mux(audio.encode(None))


def _configure(monkeypatch, tmp_path):
    import h3_audio_t8_pkg.reel_delivery_advanced as delivery

    output = tmp_path / "output"
    input_dir = tmp_path / "input"
    temp = tmp_path / "temp"
    output.mkdir()
    input_dir.mkdir()
    temp.mkdir()
    monkeypatch.setattr(
        delivery.folder_paths, "get_output_directory", lambda: str(output)
    )
    monkeypatch.setattr(
        delivery.folder_paths, "get_input_directory", lambda: str(input_dir)
    )
    monkeypatch.setattr(delivery.folder_paths, "get_temp_directory", lambda: str(temp))
    return input_dir, output


def _plan(monkeypatch, tmp_path):
    input_dir, output = _configure(monkeypatch, tmp_path)
    first = input_dir / "first.mp4"
    second = input_dir / "second.mp4"
    music = input_dir / "music.mp4"
    _write_clip(first, 20, 0.10)
    _write_clip(second, 220, 0.15)
    _write_clip(music, 100, 0.05, frames=48)
    payload = {
        "clips": [
            {
                "id": "a",
                "path": "input:first.mp4",
                "crossfade_to_next_seconds": 0.25,
                "source_audio_gain_db": -6,
            },
            {
                "id": "b",
                "path": "input:second.mp4",
                "source_audio_gain_db": -6,
            },
        ],
        "audio_lanes": [
            {
                "id": "music",
                "role": "music",
                "events": [
                    {
                        "id": "score",
                        "path": "input:music.mp4",
                        "start_seconds": 0,
                        "trim_in_seconds": 0,
                        "trim_out_seconds": 1.75,
                        "gain_db": -12,
                        "fade_out_seconds": 0.25,
                    }
                ],
            }
        ],
    }
    plan = build_reel_delivery_plan(
        "reel_test",
        json.dumps(payload),
        48000,
        1.0,
        64.0,
    )
    return plan, first, second, output


def test_reel_plan_is_read_only_hash_bound_and_multilane(monkeypatch, tmp_path):
    plan, first, second, _output = _plan(monkeypatch, tmp_path)
    assert plan["total_frames"] == 42
    assert plan["total_samples"] == 84000
    assert plan["clip_count"] == 2
    assert plan["audio_event_count"] == 3
    assert {event["role"] for event in plan["audio_events"]} == {"dialogue", "music"}
    validate_reel_plan(plan)
    first.write_bytes(first.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="source changed"):
        validate_reel_plan(plan)
    assert second.is_file()


def test_reel_plan_rejects_path_escape_and_oversized_transition(monkeypatch, tmp_path):
    input_dir, _output = _configure(monkeypatch, tmp_path)
    clip = input_dir / "clip.mp4"
    _write_clip(clip, 80, 0.1)
    with pytest.raises(ValueError, match="inside ComfyUI"):
        build_reel_delivery_plan(
            "escape",
            json.dumps({"clips": [{"path": str(tmp_path.parent / "outside.mp4")}]}),
            48000,
            1.0,
            64.0,
        )
    with pytest.raises(ValueError, match="exceeds maximum"):
        build_reel_delivery_plan(
            "transition",
            json.dumps(
                {
                    "clips": [
                        {
                            "path": "input:clip.mp4",
                            "crossfade_to_next_seconds": 1.5,
                        },
                        {"path": "input:clip.mp4"},
                    ]
                }
            ),
            48000,
            1.0,
            64.0,
        )


def test_reel_compose_requires_confirmation_then_streams_and_resumes(
    monkeypatch, tmp_path
):
    plan, first, second, _output = _plan(monkeypatch, tmp_path)
    first_hash = plan["clips"][0]["source_sha256"]
    second_hash = plan["clips"][1]["source_sha256"]
    path, report = compose_reel_delivery(
        plan,
        False,
        "H3_Reel_Test",
        18,
        "block_if_clipping",
    )
    assert path == ""
    assert report["status"] == "planned_not_composed"

    path, report = compose_reel_delivery(
        plan,
        True,
        "H3_Reel_Test",
        18,
        "block_if_clipping",
    )
    output = Path(path)
    assert output.is_file()
    assert report["frame_count"] == 42
    assert report["audio_samples"] == 84000
    assert report["source_files_mutated"] is False
    assert first_hash == plan["clips"][0]["source_sha256"]
    assert second_hash == plan["clips"][1]["source_sha256"]
    import av

    with av.open(str(output)) as container:
        decoded_frames = sum(1 for _ in container.decode(container.streams.video[0]))
        assert decoded_frames == 42
        assert container.streams.audio

    second_path, second_report = compose_reel_delivery(
        plan,
        True,
        "H3_Reel_Test",
        18,
        "block_if_clipping",
    )
    assert second_path == path
    assert second_report["video"]["resumed"] is True
    assert second_report["audio"]["resumed"] is True

    third_path, third_report = compose_reel_delivery(
        plan,
        True,
        "H3_Reel_Test",
        17,
        "block_if_clipping",
    )
    assert third_path == path
    assert third_report["video"].get("resumed") is not True
    assert third_report["audio"]["resumed"] is True
    state = json.loads(Path(third_report["state_path"]).read_text(encoding="utf-8"))
    assert state["video_crf"] == 17


def test_temporary_cleanup_retries_transient_windows_lock(monkeypatch, tmp_path):
    target = tmp_path / "locked.tmp"
    target.write_bytes(b"partial")
    original_unlink = Path.unlink
    attempts = 0

    def transient_lock(path, *args, **kwargs):
        nonlocal attempts
        if path == target and attempts < 2:
            attempts += 1
            raise PermissionError("simulated transient Windows handle")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", transient_lock)
    assert _unlink_with_retry(target, timeout_seconds=1.0, retry_seconds=0.001) is True
    assert attempts == 2
    assert not target.exists()


def test_temporary_cleanup_preserves_primary_error_when_lock_outlives_budget(
    monkeypatch, tmp_path
):
    target = tmp_path / "locked.tmp"
    target.write_bytes(b"partial")
    monkeypatch.setattr(
        "h3_audio_t8_pkg.reel_delivery_advanced._unlink_with_retry",
        lambda _path: False,
    )
    with pytest.raises(RuntimeError, match="primary failure") as captured:
        try:
            raise RuntimeError("primary failure")
        except RuntimeError as captured_error:
            _cleanup_temporary(target, captured_error)
            raise
    assert any("remained locked" in note for note in captured.value.__notes__)


def test_orphan_cleanup_covers_all_reel_temporary_types(tmp_path):
    prefix = "Reel"
    names = [
        ".Reel.token.mp4.tmp",
        "..Reel.token.video.mp4.tmp",
        "..Reel.token.wav.tmp",
        ".Reel.token.filters.txt",
        ".Reel.state.json.token.tmp",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"orphan")
    (tmp_path / "unrelated.tmp").write_bytes(b"keep")
    assert _cleanup_orphan_temporary_files(tmp_path, prefix) == sorted(names)
    assert not any((tmp_path / name).exists() for name in names)
    assert (tmp_path / "unrelated.tmp").is_file()


def test_reel_execution_lock_rejects_concurrent_same_project(tmp_path):
    with _reel_execution_lock(tmp_path):
        with pytest.raises(TimeoutError, match="project is busy"):
            with _reel_execution_lock(tmp_path, timeout_seconds=0.01):
                raise AssertionError("a second project owner must not enter")


def test_reel_nodes_are_append_only_safe_defaults():
    schemas = [node.define_schema() for node in REEL_DELIVERY_ADVANCED_NODE_CLASSES]
    assert [schema.node_id for schema in schemas] == [
        "MiniMaxH3ReelDeliveryPlanT8Advanced",
        "MiniMaxH3ReelDeliveryComposeT8Advanced",
    ]
    assert all(schema.is_experimental for schema in schemas)
    compose_inputs = {item.id: item for item in schemas[1].inputs}
    assert compose_inputs["confirm_compose"].default is False
    assert compose_inputs["peak_policy"].default == "block_if_clipping"
