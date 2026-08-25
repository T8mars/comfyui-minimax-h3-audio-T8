from __future__ import annotations

import base64
import importlib
import json
from io import BytesIO
from pathlib import Path

from PIL import Image
import torch

from h3_audio_t8_pkg.face_refine_advanced import source_proxy_sha256
from h3_audio_t8_pkg.nodes_skin_finish import (
    MiniMaxH3SkinFinishAdvancedT8,
    MiniMaxH3SkinFinishPreviewAuditT8Advanced,
    MiniMaxH3SkinFinishT8,
    _encode_preview_frame,
)
from h3_audio_t8_pkg.skin_finish import (
    SKIN_FINISH_IMAGE_RAM_FIXED_HEADROOM_MIB,
    SKIN_FINISH_IMAGE_RAM_PREFLIGHT_SCHEMA,
    SKIN_FINISH_IMAGE_RAM_SAFETY_FACTOR,
    SKIN_FINISH_REPORT_SCHEMA,
    _estimate_skin_finish_image_ram,
    _skin_finish_image_ram_preflight,
    build_skin_finish_review,
    run_skin_finish,
)


def _frames(frame_count: int = 5, channels: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(240824)
    rgb = torch.rand((frame_count, 64, 96, 3), generator=generator)
    rgb[:, 20:36, 32:64] = torch.linspace(0.25, 0.95, 32).view(1, 1, 32, 1)
    if channels == 3:
        return rgb
    alpha = torch.linspace(0.0, 1.0, 64 * 96).view(1, 64, 96, 1)
    return torch.cat([rgb, alpha.expand(frame_count, -1, -1, -1)], dim=-1)


def _audio() -> dict:
    return {
        "waveform": torch.linspace(-0.5, 0.5, 3200).view(1, 1, -1),
        "sample_rate": 32000,
    }


def _external_mask(frame_count: int = 5) -> torch.Tensor:
    mask = torch.zeros((frame_count, 64, 96))
    mask[:, 12:52, 24:72] = 1.0
    return mask


def test_missing_external_mask_abstains_to_exact_source_and_audio_object():
    frames = _frames()
    audio = _audio()
    candidate, source, selected, used, rejected, difference, _, audio_out, report = (
        run_skin_finish(frames, mask=None, audio=audio)
    )
    parsed = json.loads(report)
    assert parsed["schema"] == SKIN_FINISH_REPORT_SCHEMA
    assert parsed["status"] == "ABSTAIN_EXTERNAL_MASK_MISSING"
    assert candidate is frames
    assert source is frames
    assert selected is frames
    assert audio_out is audio
    assert torch.count_nonzero(used) == 0
    assert torch.count_nonzero(rejected) == 0
    assert torch.count_nonzero(difference) == 0
    assert parsed["resource_preflight"]["status"] == "SKIPPED_NO_CANDIDATE_PATH"


def test_image_ram_estimate_is_shape_derived_and_non_user_lowerable():
    frames = _frames()
    estimate = _estimate_skin_finish_image_ram(
        frames,
        chunk_frames=4,
        proxy_long_side=640,
        mask_source="external_exact",
    )
    assert estimate["frame_count"] == 5
    assert estimate["height"] == 64
    assert estimate["width"] == 96
    assert estimate["chunk_frames"] == 4
    assert estimate["components_mib"]["candidate"] > 0.0
    assert estimate["components_mib"]["difference_fp16"] > 0.0
    assert estimate["safety_factor"] == SKIN_FINISH_IMAGE_RAM_SAFETY_FACTOR
    assert estimate["fixed_headroom_mib"] == SKIN_FINISH_IMAGE_RAM_FIXED_HEADROOM_MIB
    assert estimate["required_available_mib"] >= 512.0


def test_image_ram_preflight_pass_block_and_unavailable_are_explicit():
    frames = _frames()
    estimate = _estimate_skin_finish_image_ram(
        frames,
        chunk_frames=4,
        proxy_long_side=640,
        mask_source="face_refine_plan",
    )
    required = estimate["required_available_mib"]
    passed = _skin_finish_image_ram_preflight(
        frames,
        chunk_frames=4,
        proxy_long_side=640,
        mask_source="face_refine_plan",
        snapshot={
            "host_available_mib": required + 1.0,
            "commit_available_mib": required + 2.0,
        },
    )
    blocked = _skin_finish_image_ram_preflight(
        frames,
        chunk_frames=4,
        proxy_long_side=640,
        mask_source="face_refine_plan",
        snapshot={
            "host_available_mib": required - 1.0,
            "commit_available_mib": required + 2.0,
        },
    )
    unavailable = _skin_finish_image_ram_preflight(
        frames,
        chunk_frames=4,
        proxy_long_side=640,
        mask_source="face_refine_plan",
        snapshot={"pid": 7},
    )
    assert passed["schema"] == SKIN_FINISH_IMAGE_RAM_PREFLIGHT_SCHEMA
    assert passed["status"] == "PASS_ESTIMATED_INCREMENTAL_RAM_FLOOR"
    assert passed["allowed"] is True
    assert blocked["status"] == (
        "ABSTAIN_INSUFFICIENT_SYSTEM_RAM_NO_CANDIDATE_ALLOCATED"
    )
    assert blocked["allowed"] is False
    assert blocked["blocked_by"] == ["host_available_mib"]
    assert unavailable["status"] == "ALLOW_MEASUREMENT_UNAVAILABLE_BOUNDED_CPU_ROUTE"
    assert unavailable["allowed"] is True
    assert unavailable["measurement_available"] is False


def test_insufficient_image_ram_abstains_before_mask_or_candidate_allocation(monkeypatch):
    import h3_audio_t8_pkg.skin_finish as module

    frames = _frames()
    mask = _external_mask()
    monkeypatch.setattr(
        module,
        "_memory_snapshot",
        lambda: {"host_available_mib": 1.0, "commit_available_mib": 1.0},
    )
    monkeypatch.setattr(
        module,
        "_prepare_mask",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("mask preparation must stay skipped")
        ),
    )
    monkeypatch.setattr(
        module,
        "_process_chunk",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("candidate processing must stay skipped")
        ),
    )
    candidate, source, selected, used, rejected, difference, _, _, report = (
        run_skin_finish(frames, mask=mask)
    )
    parsed = json.loads(report)
    assert parsed["status"] == (
        "ABSTAIN_INSUFFICIENT_SYSTEM_RAM_NO_CANDIDATE_ALLOCATED"
    )
    assert candidate is frames
    assert source is frames
    assert selected is frames
    assert torch.count_nonzero(used) == 0
    assert torch.count_nonzero(rejected) == 0
    assert torch.count_nonzero(difference) == 0
    assert used.untyped_storage().nbytes() == 4
    assert difference.untyped_storage().nbytes() == 6
    assert parsed["resource_preflight"]["allowed"] is False
    assert parsed["resource_preflight"]["blocked_by"] == [
        "commit_available_mib",
        "host_available_mib",
    ]


def test_external_mask_is_inside_only_and_preserves_alpha_exactly():
    frames = _frames(channels=4)
    mask = _external_mask()
    candidate, source, selected, used, _, difference, _, _, report = run_skin_finish(
        frames,
        mask=mask,
        preset="oil_control",
        amount=0.75,
        shine_control=0.80,
        accept_candidate=False,
    )
    parsed = json.loads(report)
    assert parsed["status"] == "CANDIDATE_READY"
    assert source is frames
    assert selected is frames
    assert torch.isfinite(candidate).all()
    assert torch.equal(candidate[..., 3:], frames[..., 3:])
    outside = used <= 0
    assert torch.equal(candidate[..., :3][outside], frames[..., :3][outside])
    assert float(difference[used > 0].mean()) > 0.0
    assert parsed["mechanical_gates"]["outside_mask_bit_exact"] is True
    assert parsed["mechanical_gates"]["alpha_or_aux_channels_preserved"] is True


def test_mask_area_gate_rejects_an_unbounded_full_frame_mask():
    frames = _frames()
    full = torch.ones((5, 64, 96))
    candidate, _, _, used, rejected, _, _, _, report = run_skin_finish(
        frames,
        mask=full,
        maximum_mask_area=0.45,
    )
    parsed = json.loads(report)
    assert parsed["status"] == "ABSTAIN_NO_RELIABLE_SKIN_MASK"
    assert candidate is frames
    assert torch.count_nonzero(used) == 0
    assert torch.equal(rejected, full)


def test_review_only_builds_candidate_but_never_selects_it():
    frames = _frames()
    candidate, _, selected, used, _, _, _, _, report = run_skin_finish(
        frames,
        mask=_external_mask(),
        execution_mode="review_only",
        accept_candidate=True,
    )
    parsed = json.loads(report)
    assert parsed["status"] == "CANDIDATE_READY"
    assert "review_only_forces_source_selection" in parsed["findings"]
    assert torch.count_nonzero(used) > 0
    assert candidate is not frames
    assert selected is frames
    assert parsed["mechanical_gates"]["candidate_selected"] is False


def test_face_refine_plan_route_is_source_bound_and_protects_features():
    frames = _frames(frame_count=3)
    plan = {
        "schema": "h3_t8_face_refine_plan/v1",
        "source": {"proxy_sha256": source_proxy_sha256(frames)},
        "frames": [
            {
                "state": "detected",
                "paste_weight": 1.0,
                "source_face_box_xyxy": [28.0, 8.0, 68.0, 56.0],
            }
            for _ in range(3)
        ],
    }
    candidate, _, _, used, _, _, _, _, report = run_skin_finish(
        frames,
        mask_source="face_refine_plan",
        face_plan=plan,
        protect_features=True,
    )
    parsed = json.loads(report)
    assert parsed["status"] == "CANDIDATE_READY"
    assert parsed["mask_source"]["feature_protection"] is True
    assert "not semantic skin parsing" in parsed["mask_source"]["warning"]
    assert torch.count_nonzero(used) > 0
    # Approximate eye/mouth zones are explicitly excluded from the proxy mask.
    assert used[:, 24:30, 38:44].max() == 0
    assert candidate.shape == frames.shape


def test_face_refine_plan_source_mismatch_abstains():
    frames = _frames(frame_count=3)
    plan = {
        "schema": "h3_t8_face_refine_plan/v1",
        "source": {"proxy_sha256": "0" * 64},
        "frames": [{} for _ in range(3)],
    }
    candidate, _, _, used, _, _, _, _, report = run_skin_finish(
        frames,
        mask_source="face_refine_plan",
        face_plan=plan,
    )
    assert json.loads(report)["status"] == "ABSTAIN_FACE_PLAN_SOURCE_MISMATCH"
    assert candidate is frames
    assert torch.count_nonzero(used) == 0


def test_preview_requires_explicit_accept_and_pcm_exact_audio():
    frames = _frames()
    audio = _audio()
    candidate, source, _, used, rejected, _, state, audio_out, report = run_skin_finish(
        frames,
        mask=_external_mask(),
        audio=audio,
    )
    values = build_skin_finish_review(
        source,
        candidate,
        used,
        rejected,
        state,
        report,
        frame_index=2,
        comparison_position=0.4,
        accept_candidate=True,
        audio_source=audio,
        audio_passthrough=audio_out,
    )
    selected, split, source_crop, candidate_crop, mask_view, diff, loop, audio_final, review = values
    parsed = json.loads(review)
    assert selected is candidate
    assert split.shape == (1, 64, 96, 3)
    assert source_crop.shape == candidate_crop.shape
    assert mask_view.shape == (1, 64, 96, 3)
    assert diff.shape == (1, 64, 96, 3)
    assert loop.shape[0] == 5
    assert audio_final is audio
    assert parsed["status"] == "ACCEPTED_CANDIDATE"
    assert parsed["audio_status"] == "pcm_exact"
    assert parsed["automatic_accept"] is False


def test_preview_node_emits_bounded_browser_proxy_without_changing_review_contract():
    frames = _frames()
    audio = _audio()
    candidate, source, _, used, rejected, _, state, audio_out, report = run_skin_finish(
        frames,
        mask=_external_mask(),
        audio=audio,
    )
    result = MiniMaxH3SkinFinishPreviewAuditT8Advanced.execute(
        source_frames=source,
        candidate_frames=candidate,
        used_mask=used,
        rejected_mask=rejected,
        skin_finish_state=state,
        gate_report_json=report,
        frame_index=2,
        comparison_position=0.4,
        accept_candidate=False,
        audio_source=audio,
        audio_passthrough=audio_out,
    )
    review = json.loads(result.ui["text"][0])
    payload = json.loads(result.ui["t8_skin_finish_preview"][0])
    assert review["status"] == "REVIEW_REQUIRED"
    assert review["accepted_candidate"] is False
    assert payload["schema"] == "h3_t8_skin_finish_preview_ui/v1"
    assert payload["status"] == "READY"
    assert payload["proxy_only"] is True
    assert payload["full_resolution_outputs_available"] is True
    assert payload["comparison_position"] == 0.4
    assert payload["automatic_accept"] is False
    assert max(payload["proxy_width"], payload["proxy_height"]) <= 512
    for key in ("source_data_url", "candidate_data_url"):
        prefix, encoded = payload[key].split(",", 1)
        assert prefix == "data:image/jpeg;base64"
        with Image.open(BytesIO(base64.b64decode(encoded))) as image:
            assert image.mode == "RGB"
            assert image.size == (payload["proxy_width"], payload["proxy_height"])


def test_skin_finish_preview_frontend_is_local_explicit_and_non_accepting():
    source = (
        Path(__file__).resolve().parents[1] / "web" / "skin_finish_preview.js"
    ).read_text(encoding="utf-8")
    assert 'const NODE_ID = "MiniMaxH3SkinFinishPreviewAuditT8Advanced"' in source
    assert "innerHTML" not in source
    assert "textContent" in source
    assert "clipPath" in source
    assert 'item.name === "comparison_position"' in source
    assert "widget.value = value" in source
    assert "widget.callback?.(value)" in source
    assert "serialize: false" in source
    assert "queuePrompt" not in source
    assert 'item.name === "accept_candidate"' not in source


def test_skin_finish_preview_proxy_is_bounded_to_512_long_side():
    frames = torch.zeros((1, 600, 800, 3), dtype=torch.float32)
    data_url, width, height = _encode_preview_frame(frames, 0)
    assert (width, height) == (512, 384)
    prefix, encoded = data_url.split(",", 1)
    assert prefix == "data:image/jpeg;base64"
    with Image.open(BytesIO(base64.b64decode(encoded))) as image:
        assert image.size == (512, 384)


def test_skin_finish_node_schemas_are_safe_by_default():
    basic = MiniMaxH3SkinFinishT8.define_schema()
    advanced = MiniMaxH3SkinFinishAdvancedT8.define_schema()
    preview = MiniMaxH3SkinFinishPreviewAuditT8Advanced.define_schema()
    basic_inputs = {item.id: item for item in basic.inputs}
    advanced_inputs = {item.id: item for item in advanced.inputs}
    preview_inputs = {item.id: item for item in preview.inputs}
    assert basic_inputs["preset"].default == "subtle"
    assert basic_inputs["skin_mask"].optional is True
    assert advanced.is_experimental is True
    assert advanced_inputs["mask_source"].default == "external_exact"
    assert advanced_inputs["accept_candidate"].default is False
    assert advanced_inputs["chunk_frames"].default == 4
    assert preview.is_output_node is True
    assert preview_inputs["accept_candidate"].default is False


def test_skin_finish_modules_publish_no_persistent_tensor_model_or_lru_cache():
    module_names = (
        "skin_finish",
        "skin_finish_parser",
        "skin_finish_multiface_parser",
        "skin_finish_p1",
        "skin_finish_p2",
        "skin_finish_person_profiles",
        "skin_finish_safety_audit",
        "skin_finish_frequency",
        "skin_finish_specular_frequency",
        "skin_finish_surface",
        "skin_finish_timeline",
        "skin_finish_stream_quality",
    )
    for module_name in module_names:
        module = importlib.import_module(f"h3_audio_t8_pkg.{module_name}")
        public_items = {
            name: value
            for name, value in vars(module).items()
            if not name.startswith("__")
        }
        persistent_tensors_or_models = [
            name
            for name, value in public_items.items()
            if isinstance(value, (torch.Tensor, torch.nn.Module))
        ]
        cached_callables = [
            name
            for name, value in public_items.items()
            if callable(value) and callable(getattr(value, "cache_info", None))
        ]
        named_mutable_caches = [
            name
            for name, value in public_items.items()
            if "cache" in name.lower() and isinstance(value, (dict, list, set))
        ]
        assert persistent_tensors_or_models == [], module_name
        assert cached_callables == [], module_name
        assert named_mutable_caches == [], module_name
