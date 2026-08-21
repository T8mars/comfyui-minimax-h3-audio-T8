from __future__ import annotations

import json
import subprocess

from h3_audio_t8_pkg.tools import analyze_h3_speed_quality_pairs as analysis_tool
from h3_audio_t8_pkg.tools.analyze_h3_speed_quality_pairs import build_blind_review


def test_blind_review_contains_three_anonymous_pairs_and_reveal(tmp_path):
    pairs = {}
    for name in ("t2va", "fl2va", "ref2va"):
        baseline = tmp_path / f"{name}_baseline.mp4"
        speed = tmp_path / f"{name}_speed.mp4"
        baseline.write_bytes(b"baseline-" + name.encode())
        speed.write_bytes(b"speed-" + name.encode())
        pairs[name] = {"baseline": baseline, "speed": speed}
    blind = tmp_path / "blind"
    reveal = build_blind_review(pairs, output_dir=blind, seed=123)
    assert set(reveal["pairs"]) == {"t2va", "fl2va", "ref2va"}
    for mapping in reveal["pairs"].values():
        assert set(mapping) == {"A", "B"}
        assert set(mapping.values()) == {"baseline", "speed"}
    html = (blind / "blind_review.html").read_text(encoding="utf-8")
    assert "MiniMax H3 SPEED" in html
    assert "漏填按“平”处理" in html
    assert "h3_speed_blind_review.json" in html
    saved = json.loads((blind / "reveal.json").read_text(encoding="utf-8"))
    assert saved == reveal
    assert len(list(blind.glob("*.mp4"))) == 6


def test_strict_decode_makes_ffmpeg_decoder_errors_fatal(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(analysis_tool.subprocess, "run", fake_run)
    result = analysis_tool._strict_decode(tmp_path / "sample.mp4", "ffmpeg", attempts=1)
    assert result["passed"] is True
    assert "-xerror" in commands[0]
    assert commands[0][commands[0].index("-err_detect") + 1] == "explode"
    assert commands[0][commands[0].index("-threads") + 1] == "1"
