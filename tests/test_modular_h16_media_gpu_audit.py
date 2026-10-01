"""A VHS intermediate MP4 cannot be confused with the delivered H16 AV file."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.audit_modular_h16_media_gpu import _effect_report, _media


def _fixture(tmp_path: Path, monkeypatch):
    output = tmp_path / "output/MiniMaxH3"
    output.mkdir(parents=True)
    final = output / "Probe_00001-audio.mp4"
    silent = output / "Probe_00001.mp4"
    final.write_bytes(b"muxed-fixture")
    silent.write_bytes(b"video-only-fixture")
    history = {"outputs": {"21": {"gifs": [{
        "fullpath": str(final), "filename": final.name,
        "type": "output", "format": "video/h265-mp4",
    }]}}}

    def fake_run(argv, **_kwargs):
        if argv[0] == "ffprobe":
            streams = [
                {"codec_type": "video", "width": 448, "height": 448,
                 "nb_read_frames": "124", "avg_frame_rate": "24/1"},
                {"codec_type": "audio", "duration": "5.152000"},
            ]
            return SimpleNamespace(stdout=json.dumps({"streams": streams}),
                                   stderr=b"", returncode=0)
        if "f32le" in argv:
            return SimpleNamespace(stdout=np.array([0.1, -0.2], dtype="<f4").tobytes(),
                                   stderr=b"", returncode=0)
        return SimpleNamespace(stdout=b"", stderr=b"", returncode=0)

    monkeypatch.setattr("tools.audit_modular_h16_media_gpu.subprocess.run", fake_run)
    return final, history


def test_h16_media_audit_binds_vhs_muxed_delivery(tmp_path, monkeypatch):
    final, history = _fixture(tmp_path, monkeypatch)
    report = _media(tmp_path, history)
    assert report["path"] == str(final)
    assert report["frames"] == 124
    assert report["decoded_audio_rms"] > 0
    assert report["strict_decode"] is True


def test_h16_media_audit_rejects_wrong_history_path(tmp_path, monkeypatch):
    _final, history = _fixture(tmp_path, monkeypatch)
    changed = deepcopy(history)
    changed["outputs"]["21"]["gifs"][0]["fullpath"] = str(tmp_path / "other.mp4")
    with pytest.raises(FileNotFoundError):
        _media(tmp_path, changed)


def test_h16_media_audit_rejects_additional_delivery(tmp_path, monkeypatch):
    _final, history = _fixture(tmp_path, monkeypatch)
    (tmp_path / "output/MiniMaxH3/unexpected.mp4").write_bytes(b"extra")
    with pytest.raises(ValueError, match="inventory differs"):
        _media(tmp_path, history)


def _effect_history():
    report = {
        "schema": "t8.modular-sampling.eav-audit.v1",
        "h16_window_index": 6,
        "h16_plan_sha256": "plan",
        "audio_output": "refined_exp",
        "config": {"mode": "apply_exp"},
        "status": "observed_apply_exp",
        "completed_forwards": 4,
        "planned_forwards": 4,
        "clock_match": True,
        "relay_required": True,
        "relay_attention_calls": 200,
        "selector_calls": 200,
        "sparse_producer_calls": 0,
        "forward_plan": {"forwards": [{"attention_blocks": 50}] * 4},
        "feta": {"aborted": None, "model_forward_count": 4,
                 "active_forward_count": 3, "g_max": 1.3},
    }
    return {"outputs": {"80": {"text": [json.dumps(report)]}}}


def test_h16_effect_audit_requires_real_relay_and_active_feta_calls():
    history = _effect_history()
    accepted = _effect_report(history, "80", 6, "apply_exp")
    assert accepted["relay_attention_calls"] == 200
    assert accepted["g_max"] > 1
    missing_relay = deepcopy(history)
    report = json.loads(missing_relay["outputs"]["80"]["text"][0])
    report["relay_attention_calls"] = 0
    missing_relay["outputs"]["80"]["text"][0] = json.dumps(report)
    with pytest.raises(ValueError, match="actual call audit failed"):
        _effect_report(missing_relay, "80", 6, "apply_exp")
    no_gain = deepcopy(history)
    report = json.loads(no_gain["outputs"]["80"]["text"][0])
    report["feta"]["g_max"] = 1.0
    no_gain["outputs"]["80"]["text"][0] = json.dumps(report)
    with pytest.raises(ValueError, match="actual call audit failed"):
        _effect_report(no_gain, "80", 6, "apply_exp")
