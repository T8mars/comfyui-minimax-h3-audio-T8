from __future__ import annotations

import json

import pytest

from h3_audio_t8_pkg.tools import build_external_bridge_blind_review as blind_tool


def _contract():
    return {
        "width": 1152,
        "height": 640,
        "fps": 24.0,
        "frame_count": 22,
        "video_duration_seconds": 22 / 24,
        "sample_rate": 32000,
        "channels": 2,
        "audio_duration_seconds": 0.896,
    }


def test_external_bridge_package_is_private_deterministic_and_reference_aware(
    monkeypatch, tmp_path
):
    control = tmp_path / "reveals-native32b.mp4"
    candidate = tmp_path / "reveals-clipproj8b.mp4"
    reference = tmp_path / "reference.png"
    control.write_bytes(b"control-media")
    candidate.write_bytes(b"candidate-media")
    reference.write_bytes(b"reference-image")
    monkeypatch.setattr(blind_tool, "_media_contract", lambda _path: _contract())
    manifest = {
        "schema": blind_tool.MANIFEST_SCHEMA,
        "review_id": "external-bridges-v1",
        "pairs": [
            {
                "pair_id": "i2va",
                "label": "第1组",
                "task_type": "I2VA",
                "prompt": "same prompt",
                "control": control.name,
                "candidate": candidate.name,
                "control_method": "native 32B",
                "candidate_method": "ClipProj 8B",
                "reference_images": [reference.name],
                "reference_metrics": ["first_frame", "identity"],
            }
        ],
    }
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_key = blind_tool.build_package(manifest, tmp_path, first, 1234)
    second_key = blind_tool.build_package(manifest, tmp_path, second, 1234)
    assert first_key["pairs"][0]["sides"] == second_key["pairs"][0]["sides"]
    html = (first / "blind_review.html").read_text(encoding="utf-8")
    assert "JSON.stringify(payload,null,2)+'\\n'" in html
    assert str(control) not in html
    assert str(candidate) not in html
    assert "native 32B" not in html
    assert "ClipProj 8B" not in html
    assert '"arm":"control"' not in html
    assert '"arm":"candidate"' not in html
    assert blind_tool.REVIEW_SCHEMA in html
    assert "同步静音播放" in html
    assert "先选择本组是否可判断" in html
    assert "原素材/参考本身不足" in html
    assert "playback_problem" in html
    assert "assessability:value.assessability||'unsure'" in html
    assert "参考图 1" in html
    key = json.loads((first / "blind_key.json").read_text(encoding="utf-8"))
    assert {side["method"] for side in key["pairs"][0]["sides"]} == {
        "native 32B",
        "ClipProj 8B",
    }
    for side in key["pairs"][0]["sides"]:
        copied = first / "media" / f"pair-01-{side['code']}.mp4"
        assert blind_tool._sha256_file(copied) == side["sha256"]
    copied_ref = first / "media" / "pair-01-ref-01.png"
    assert blind_tool._sha256_file(copied_ref) == key["pairs"][0]["references"][0][
        "sha256"
    ]


def test_external_bridge_package_rejects_contract_and_metric_mismatch(
    monkeypatch, tmp_path
):
    control = tmp_path / "control.mp4"
    candidate = tmp_path / "candidate.mp4"
    control.write_bytes(b"control")
    candidate.write_bytes(b"candidate")

    def contract(path):
        value = _contract()
        if path == candidate:
            value["channels"] = 1
        return value

    monkeypatch.setattr(blind_tool, "_media_contract", contract)
    manifest = {
        "schema": blind_tool.MANIFEST_SCHEMA,
        "review_id": "review",
        "pairs": [
            {
                "pair_id": "pair",
                "control": str(control),
                "candidate": str(candidate),
                "reference_metrics": ["unknown"],
            }
        ],
    }
    with pytest.raises(ValueError, match="differs at channels"):
        blind_tool.build_package(manifest, tmp_path, tmp_path / "blind", 1)
    monkeypatch.setattr(blind_tool, "_media_contract", lambda _path: _contract())
    with pytest.raises(ValueError, match="unsupported reference_metrics"):
        blind_tool.build_package(manifest, tmp_path, tmp_path / "blind-2", 1)


def test_external_bridge_package_preserves_existing_review_evidence(
    monkeypatch, tmp_path
):
    control = tmp_path / "control.mp4"
    candidate = tmp_path / "candidate.mp4"
    control.write_bytes(b"control")
    candidate.write_bytes(b"candidate")
    monkeypatch.setattr(blind_tool, "_media_contract", lambda _path: _contract())
    manifest = {
        "schema": blind_tool.MANIFEST_SCHEMA,
        "review_id": "review-v1",
        "pairs": [
            {
                "pair_id": "pair",
                "control": str(control),
                "candidate": str(candidate),
                "control_method": "control",
                "candidate_method": "candidate",
                "reference_metrics": [],
            }
        ],
    }
    output = tmp_path / "blind"
    original_key = blind_tool.build_package(manifest, tmp_path, output, 1234)
    original_key_bytes = (output / "blind_key.json").read_bytes()
    original_page_bytes = (output / "blind_review.html").read_bytes()

    # Rebuilding the exact package is safe because its private mapping is unchanged.
    monkeypatch.setattr(blind_tool, "_document", lambda **_kwargs: "changed")
    assert blind_tool.build_package(manifest, tmp_path, output, 1234) == original_key
    assert (output / "blind_key.json").read_bytes() == original_key_bytes
    assert (output / "blind_review.html").read_bytes() == original_page_bytes

    changed_id = dict(manifest, review_id="review-v2")
    with pytest.raises(ValueError, match="different immutable key"):
        blind_tool.build_package(changed_id, tmp_path, output, 1234)
    with pytest.raises(ValueError, match="different immutable key"):
        blind_tool.build_package(manifest, tmp_path, output, 4321)
    assert (output / "blind_key.json").read_bytes() == original_key_bytes


def test_external_bridge_package_rejects_unkeyed_nonempty_output(
    monkeypatch, tmp_path
):
    control = tmp_path / "control.mp4"
    candidate = tmp_path / "candidate.mp4"
    control.write_bytes(b"control")
    candidate.write_bytes(b"candidate")
    monkeypatch.setattr(blind_tool, "_media_contract", lambda _path: _contract())
    manifest = {
        "schema": blind_tool.MANIFEST_SCHEMA,
        "review_id": "review",
        "pairs": [
            {
                "pair_id": "pair",
                "control": str(control),
                "candidate": str(candidate),
                "reference_metrics": [],
            }
        ],
    }
    output = tmp_path / "blind"
    output.mkdir()
    (output / "old-page.html").write_text("stale", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty but has no blind_key"):
        blind_tool.build_package(manifest, tmp_path, output, 1)


def test_external_bridge_package_supports_hash_bound_custom_review_copy(
    monkeypatch, tmp_path
):
    control = tmp_path / "control.mp4"
    candidate = tmp_path / "candidate.mp4"
    control.write_bytes(b"control")
    candidate.write_bytes(b"candidate")
    monkeypatch.setattr(blind_tool, "_media_contract", lambda _path: _contract())
    manifest = {
        "schema": blind_tool.MANIFEST_SCHEMA,
        "review_id": "creator-review-v1",
        "page_title": "Creator <AV> 匿名评审",
        "page_intro": "先看画面，再分别听完整音轨。",
        "export_filename": "creator_av_blind_review.json",
        "analysis_generalization": "This result applies only to one fixed Creator AV pair.",
        "pairs": [
            {
                "pair_id": "pair",
                "control": str(control),
                "candidate": str(candidate),
                "reference_metrics": [],
            }
        ],
    }
    output = tmp_path / "blind"
    key = blind_tool.build_package(manifest, tmp_path, output, 9)
    html = (output / "blind_review.html").read_text(encoding="utf-8")

    assert "Creator &lt;AV&gt; 匿名评审" in html
    assert "先看画面，再分别听完整音轨。" in html
    assert 'link.download="creator_av_blind_review.json"' in html
    assert key["display_contract"] == {
        "page_title": "Creator <AV> 匿名评审",
        "page_intro": "先看画面，再分别听完整音轨。",
        "export_filename": "creator_av_blind_review.json",
    }
    assert key["analysis_contract"] == {
        "generalization": "This result applies only to one fixed Creator AV pair."
    }

    changed = dict(manifest, page_title="不同标题")
    with pytest.raises(ValueError, match="different immutable key"):
        blind_tool.build_package(changed, tmp_path, output, 9)

    invalid = dict(manifest, export_filename="../review.json")
    with pytest.raises(ValueError, match="safe ASCII"):
        blind_tool.build_package(invalid, tmp_path, tmp_path / "invalid", 9)

    invalid_analysis = dict(manifest, analysis_generalization="")
    with pytest.raises(ValueError, match="analysis_generalization"):
        blind_tool.build_package(
            invalid_analysis, tmp_path, tmp_path / "invalid-analysis", 9
        )
