from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg.nodes_skin_finish_stream_quality import (
    MiniMaxH3SkinFinishQualityVideoStreamT8Advanced,
)
from h3_audio_t8_pkg.skin_finish_stream_quality import (
    QUALITY_STREAM_MIN_AVAILABLE_RAM_MIB,
    QUALITY_STREAM_REVIEWED_INCREMENTAL_PEAK_MIB,
    SKIN_FINISH_QUALITY_STREAM_REPORT_SCHEMA,
    SKIN_FINISH_QUALITY_STREAM_RAM_PREFLIGHT_SCHEMA,
    _QualityChunkProcessor,
    _quality_stream_ram_preflight,
    stream_skin_finish_quality_video,
)


class _FakeModel:
    def to(self, **_kwargs):
        return self


def _install_deterministic_pipeline(monkeypatch):
    import h3_audio_t8_pkg.skin_finish_stream_quality as module

    monkeypatch.setattr(
        module,
        "_load_pinned_parsenet",
        lambda: (_FakeModel(), Path("parsing_parsenet.pth"), "A" * 64),
    )
    monkeypatch.setattr(
        module,
        "_parser_logits",
        lambda _model, _crop: torch.zeros((1, 19, 512, 512)),
    )
    monkeypatch.setattr(
        module,
        "_semantic_local_masks",
        lambda *_args, **_kwargs: (
            torch.ones((1, 512, 512)),
            torch.zeros((1, 512, 512)),
            {},
        ),
    )
    monkeypatch.setattr(
        module,
        "_process_chunk",
        lambda source, mask, **_kwargs: source
        + mask.unsqueeze(-1) * source.new_tensor([0.01, 0.005, 0.002]),
    )

    def frequency(source, candidate, mask, **_kwargs):
        return (
            candidate,
            source,
            source,
            None,
            mask,
            torch.zeros_like(mask),
            torch.zeros((*mask.shape, 3), dtype=torch.float16),
            json.dumps({"rejected_frame_count": 0}),
        )

    def guard(source, candidate, mask, **_kwargs):
        return (
            candidate,
            source,
            source,
            None,
            mask,
            torch.zeros_like(mask),
            torch.zeros((*mask.shape, 3), dtype=torch.float16),
            json.dumps({"rejected_frame_count": 0}),
        )

    audit_frame_counts = []

    def audit(source, candidate, mask, **_kwargs):
        audit_frame_counts.append(int(source.shape[0]))
        report = {
            "summary": {
                "maximum_observed_temporal_effect_jump": 0.001,
                "failed_frame_indices": [],
            }
        }
        return (
            source,
            candidate,
            source,
            None,
            True,
            0,
            source[:1],
            json.dumps(report),
        )

    monkeypatch.setattr(module, "separate_skin_finish_frequencies", frequency)
    monkeypatch.setattr(module, "guard_skin_finish_candidate", guard)
    monkeypatch.setattr(module, "audit_skin_finish_candidate", audit)
    return audit_frame_counts


def _processor():
    return _QualityChunkProcessor(
        preset="subtle",
        amount=0.30,
        texture_keep=0.95,
        shine_control=0.25,
        crop_expansion=1.0,
        minimum_class_probability=0.55,
        feature_protection_px=4,
        mask_feather_px=0,
        proxy_long_side=128,
        low_frequency_strength=1.0,
        source_detail_gain=1.0,
        separation_radius_percent=1.0,
        maximum_radius_px=32,
        shadow_protection=0.10,
        highlight_protection=0.94,
        minimum_texture_ratio=0.78,
        maximum_temporal_effect_jump=0.04,
    )


def test_quality_processor_is_chunk_bounded_and_carries_one_audit_frame(monkeypatch):
    audit_frame_counts = _install_deterministic_pipeline(monkeypatch)
    processor = _processor()
    frames = torch.zeros((4, 32, 32, 3), dtype=torch.float32)
    records = [[{"box": [10.0, 10.0, 22.0, 22.0], "weight": 1.0}]] * 4

    first, first_mask = processor(frames[:2], records[:2], 0)
    second, second_mask = processor(frames[2:], records[2:], 2)
    processor.close()
    report = processor.report()

    assert tuple(first.shape) == (2, 32, 32, 3)
    assert tuple(second.shape) == (2, 32, 32, 3)
    assert bool((first_mask > 0).any()) and bool((second_mask > 0).any())
    assert audit_frame_counts == [2, 3]
    assert report["summary"]["chunk_count"] == 2
    assert report["summary"]["peak_chunk_frames"] == 2
    assert report["summary"]["source_frame_count"] == 4
    assert report["summary"]["semantic_ready_frame_count"] == 4
    assert report["summary"]["maximum_temporal_effect_jump"] == pytest.approx(0.001)
    assert report["parser"]["loaded"] is True
    assert report["parser"]["released_after_execute"] is True


def test_quality_processor_preserves_source_when_no_semantic_face(monkeypatch):
    _install_deterministic_pipeline(monkeypatch)
    processor = _processor()
    frames = torch.rand((2, 32, 32, 3), generator=torch.Generator().manual_seed(91))

    output, mask = processor(frames, [[], []], 0)
    processor.close()

    assert torch.equal(output, frames)
    assert int(torch.count_nonzero(mask)) == 0
    assert processor.report()["summary"]["source_only_frame_count"] == 2


def test_quality_stream_default_never_loads_parser_or_writes(monkeypatch):
    import h3_audio_t8_pkg.skin_finish_stream_quality as module

    sentinel = object()

    def source_only(video, **kwargs):
        assert video is sentinel
        assert kwargs == {"accept_candidate": False}
        return (
            video,
            "",
            json.dumps(
                {
                    "schema": "old",
                    "status": "SOURCE_SELECTED_NO_ANALYSIS_OR_FILE_WRITTEN",
                }
            ),
            None,
        )

    monkeypatch.setattr(module, "stream_skin_finish_video", source_only)
    monkeypatch.setattr(
        module,
        "_load_pinned_parsenet",
        lambda: (_ for _ in ()).throw(AssertionError("parser must stay unloaded")),
    )
    monkeypatch.setattr(
        module,
        "_quality_stream_ram_preflight",
        lambda: (_ for _ in ()).throw(AssertionError("preflight must stay skipped")),
    )

    video, path, report_json, saved = stream_skin_finish_quality_video(
        sentinel,
        accept_candidate=False,
    )
    report = json.loads(report_json)

    assert video is sentinel
    assert path == "" and saved is None
    assert report["schema"] == SKIN_FINISH_QUALITY_STREAM_REPORT_SCHEMA
    assert report["quality_pipeline"]["executed"] is False
    assert report["quality_pipeline"]["parser_loaded"] is False
    assert report["resource_preflight"] == {
        "allowed": False,
        "measurement_performed": False,
        "schema": SKIN_FINISH_QUALITY_STREAM_RAM_PREFLIGHT_SCHEMA,
        "status": "SKIPPED_SOURCE_SELECTED_BY_DEFAULT",
    }


def test_quality_stream_ram_preflight_uses_the_reviewed_floor():
    passed = _quality_stream_ram_preflight({"host_available_mib": 3072.0})
    blocked = _quality_stream_ram_preflight({"host_available_mib": 1024.0})
    unavailable = _quality_stream_ram_preflight({"pid": 7})

    assert QUALITY_STREAM_MIN_AVAILABLE_RAM_MIB == pytest.approx(2048.0)
    assert QUALITY_STREAM_REVIEWED_INCREMENTAL_PEAK_MIB == pytest.approx(1163.129)
    assert passed["allowed"] is True
    assert passed["status"] == "PASS_REVIEWED_AVAILABLE_RAM_FLOOR"
    assert passed["reviewed_floor_headroom_mib"] == pytest.approx(884.871)
    assert blocked["allowed"] is False
    assert blocked["status"] == "ABSTAIN_INSUFFICIENT_SYSTEM_RAM_NO_FILE_WRITTEN"
    assert unavailable["allowed"] is True
    assert unavailable["measurement_available"] is False
    assert unavailable["available_mib"] is None


def test_quality_stream_insufficient_ram_abstains_before_analysis(monkeypatch):
    import h3_audio_t8_pkg.skin_finish_stream_quality as module

    sentinel = object()

    def source_only(video, **kwargs):
        assert video is sentinel
        assert kwargs == {"accept_candidate": False}
        return (
            video,
            "",
            json.dumps({"schema": "old", "status": "SOURCE_SELECTED"}),
            None,
        )

    monkeypatch.setattr(module, "stream_skin_finish_video", source_only)
    monkeypatch.setattr(
        module,
        "_quality_stream_ram_preflight",
        lambda: {
            "schema": SKIN_FINISH_QUALITY_STREAM_RAM_PREFLIGHT_SCHEMA,
            "status": "ABSTAIN_INSUFFICIENT_SYSTEM_RAM_NO_FILE_WRITTEN",
            "allowed": False,
            "available_mib": 1024.0,
            "minimum_available_mib": 2048.0,
        },
    )
    monkeypatch.setattr(
        module,
        "_load_pinned_parsenet",
        lambda: (_ for _ in ()).throw(AssertionError("parser must stay unloaded")),
    )

    video, path, report_json, saved = stream_skin_finish_quality_video(
        sentinel,
        accept_candidate=True,
    )
    report = json.loads(report_json)

    assert video is sentinel
    assert path == "" and saved is None
    assert report["status"] == "ABSTAIN_INSUFFICIENT_SYSTEM_RAM_NO_FILE_WRITTEN"
    assert report["quality_pipeline"] == {
        "executed": False,
        "parser_loaded": False,
        "reason": "insufficient_system_ram_preflight",
    }
    assert report["resource_preflight"]["available_mib"] == pytest.approx(1024.0)
    assert report["human_review_required"] is False


def test_quality_stream_preserves_analysis_abstain_without_execution_block(monkeypatch):
    import h3_audio_t8_pkg.skin_finish_stream_quality as module

    sentinel = object()

    def no_face(video, **kwargs):
        assert video is sentinel
        assert kwargs["accept_candidate"] is True
        assert callable(kwargs["_chunk_processor"])
        return (
            video,
            "",
            json.dumps(
                {
                    "schema": "old",
                    "status": "ABSTAIN_NO_RELIABLE_FACE_NO_FILE_WRITTEN",
                    "two_pass_executed": False,
                }
            ),
            None,
        )

    monkeypatch.setattr(module, "stream_skin_finish_video", no_face)
    monkeypatch.setattr(
        module,
        "_quality_stream_ram_preflight",
        lambda: _quality_stream_ram_preflight({"host_available_mib": 4096.0}),
    )

    video, path, report_json, saved = stream_skin_finish_quality_video(
        sentinel,
        accept_candidate=True,
    )
    report = json.loads(report_json)

    assert video is sentinel
    assert path == "" and saved is None
    assert report["status"] == "ABSTAIN_NO_RELIABLE_FACE_NO_FILE_WRITTEN"
    assert report["quality_pipeline"]["executed"] is False
    assert report["quality_pipeline"]["parser"]["loaded"] is False
    assert report["resource_preflight"]["allowed"] is True
    assert report["execution"]["full_semantic_mask_batch_materialized"] is False
    assert report["execution"]["full_candidate_image_batch_materialized"] is False


def test_quality_stream_schema_is_review_gated_and_conservative():
    schema = MiniMaxH3SkinFinishQualityVideoStreamT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}

    assert schema.is_experimental is True
    assert schema.is_output_node is True
    assert inputs["accept_candidate"].default is False
    assert inputs["chunk_frames"].default == 2
    assert inputs["mask_feather_px"].default == 0
    assert inputs["source_detail_gain"].default == pytest.approx(1.0)
    assert inputs["maximum_temporal_effect_jump"].default == pytest.approx(0.04)


def test_quality_stream_workflow_is_importable_documented_and_source_safe():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "examples"
        / "workflows"
        / "17-skin-finish"
        / "2026-08-25_H3_Skin_Finish_Quality_Stream_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    node = next(
        item
        for item in workflow["nodes"]
        if item["type"] == "MiniMaxH3SkinFinishQualityVideoStreamT8Advanced"
    )
    schema = MiniMaxH3SkinFinishQualityVideoStreamT8Advanced.define_schema()

    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(item["id"] for item in workflow["nodes"])
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert [item["name"] for item in node["inputs"]] == ["source_video"]
    assert len(node["widgets_values"]) == len(schema.inputs) - 1
    assert node["widgets_values"][0:4] == ["subtle", 0.30, 0.95, 0.25]
    assert node["widgets_values"][13] == 0
    assert node["widgets_values"][23] == 2
    assert node["widgets_values"][-1] is False
    assert sum(item["type"] == "MarkdownNote" for item in workflow["nodes"]) == 6
    note_text = "\n".join(
        item["widgets_values"][0]
        for item in workflow["nodes"]
        if item["type"] == "MarkdownNote"
    )
    assert "GetVideoComponents" in note_text
    assert "accept_candidate=false" in note_text
    assert "2048MiB" in note_text
    assert "ABSTAIN_INSUFFICIENT_SYSTEM_RAM_NO_FILE_WRITTEN" in note_text
    assert "ParseNet" in note_text
    assert "口型" in note_text

    mirror = (
        root.parents[1]
        / "user"
        / "default"
        / "workflows"
        / "MiniMax H3 T8"
        / "17-skin-finish"
        / path.name
    )
    assert mirror.read_bytes() == path.read_bytes()


def test_oil_control_stream_workflow_is_importable_reviewed_and_source_safe():
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "examples"
        / "workflows"
        / "17-skin-finish"
        / "2026-08-25_H3_Skin_Finish_Oil_Control_Stream_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    node = next(
        item
        for item in workflow["nodes"]
        if item["type"] == "MiniMaxH3SkinFinishQualityVideoStreamT8Advanced"
    )
    schema = MiniMaxH3SkinFinishQualityVideoStreamT8Advanced.define_schema()

    assert workflow["version"] == 0.4
    assert [item["name"] for item in node["inputs"]] == ["source_video"]
    assert len(node["widgets_values"]) == len(schema.inputs) - 1
    assert node["widgets_values"][0:4] == ["oil_control", 0.35, 0.90, 0.35]
    assert node["widgets_values"][13] == 0
    assert node["widgets_values"][23] == 2
    assert node["widgets_values"][24] == "MiniMaxH3/SkinFinish/oil_control_stream"
    assert node["widgets_values"][25] == pytest.approx(16.0)
    assert node["widgets_values"][-1] is False
    assert sum(item["type"] == "MarkdownNote" for item in workflow["nodes"]) == 6
    note_text = "\n".join(
        item["widgets_values"][0]
        for item in workflow["nodes"]
        if item["type"] == "MarkdownNote"
    )
    for required in (
        "v1.0八步LoRA",
        "124/124",
        "packet和PCM精确",
        "来源没有明显油光",
        "蜡像感",
        "accept_candidate=true",
    ):
        assert required in note_text

    mirror = (
        root.parents[1]
        / "user"
        / "default"
        / "workflows"
        / "MiniMax H3 T8"
        / "17-skin-finish"
        / path.name
    )
    assert mirror.read_bytes() == path.read_bytes()
