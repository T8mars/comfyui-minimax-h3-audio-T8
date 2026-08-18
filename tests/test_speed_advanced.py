from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

import comfy.nested_tensor

from h3_audio_t8_pkg.nodes_speed_advanced import SPEED_ADVANCED_NODE_CLASSES
from h3_audio_t8_pkg.speed_advanced import (
    SPEED_PLAN_SCHEMA,
    SPEED_PROFILE_SCHEMA,
    SPEED_SOURCE_SCHEMA,
    StageConditioning,
    _build_empty_t2va_stage,
    _profile_binding,
    _task_support,
    activation_time,
    align_sigma,
    build_spectrum_profile,
    build_speed_plan,
    build_speed_source,
    dct_expand_official,
    fit_h3_spatial_power_spectrum,
    kappa,
    power_spectrum,
    recover_raw_flow_state,
    reindex_joint_audio_state,
    resolve_stage_shapes,
    solve_segment_noise,
)


def test_official_speed_equations_match_reference_formulas():
    power = power_spectrum(12.0, 219.484718, 2.422687)
    expected_activation = 1.0 / (
        1.0 + math.sqrt(0.01 / (power * (1.0 + power - 0.01)))
    )
    expected_kappa = 2.0 / (1.0 + 0.75)
    assert activation_time(power, 0.01) == pytest.approx(expected_activation)
    assert kappa(0.75, 2.0) == pytest.approx(expected_kappa)
    assert align_sigma(0.75, 2.0) == pytest.approx(0.75 * expected_kappa)


def test_stage_shapes_snap_to_32_without_aspect_distortion():
    stages = resolve_stage_shapes(1056, 608, [0.5, 1.0])
    assert [(stage["width"], stage["height"]) for stage in stages] == [
        (544, 320),
        (1056, 608),
    ]
    assert stages[0]["latent_width"] == 34
    assert stages[0]["latent_height"] == 20
    assert stages[-1]["latent_width"] % 2 == 0
    assert stages[-1]["latent_height"] % 2 == 0
    assert stages[0]["snap_anisotropy"] < 0.05


def test_manual_plan_preserves_exact_nfe_and_aligns_each_transition():
    plan, report_json = build_speed_plan(
        width=1056,
        height=608,
        steps=20,
        scales="0.5,1.0",
        transition_mode="manual_sigmas",
        manual_transition_sigmas="0.85",
        delta=0.01,
        shift_video=12.0,
        transform="dct",
        profile_policy="require_validated_profile",
        fallback_policy="error",
    )
    assert plan["schema"] == SPEED_PLAN_SCHEMA
    assert plan["nfe"] == 20
    assert sum(segment["nfe"] for segment in plan["segments"]) == 20
    transition = plan["transitions"][0]
    assert transition["ratio"] == pytest.approx(2.0)
    assert transition["actual_grid_ratio"] == pytest.approx(
        math.sqrt((66 / 34) * (38 / 20))
    )
    assert transition["actual_grid_ratio"] != pytest.approx(transition["ratio"])
    assert transition["aligned_sigma"] > transition["sigma"]
    assert transition["aligned_sigma"] == pytest.approx(
        align_sigma(transition["sigma"], transition["ratio"])
    )
    assert json.loads(report_json)["official_method"]["wan_constants_reused"] is False


def test_three_stage_plan_has_strict_segments_and_same_total_nfe():
    plan, _ = build_speed_plan(
        width=1344,
        height=768,
        steps=20,
        scales="0.4,0.7,1.0",
        transition_mode="manual_sigmas",
        manual_transition_sigmas="0.94,0.78",
        delta=0.01,
        shift_video=12.0,
        transform="dct",
        profile_policy="require_validated_profile",
        fallback_policy="error",
    )
    assert len(plan["stages"]) == 3
    assert len(plan["transitions"]) == 2
    assert len(plan["segments"]) == 3
    assert plan["nfe"] == 20
    assert all(segment["nfe"] > 0 for segment in plan["segments"])
    assert plan["transitions"][0]["step_index"] < plan["transitions"][1]["step_index"]


def test_delta_optimal_rejects_single_clip_probe_unless_explicitly_allowed():
    profile = {
        "schema": SPEED_PROFILE_SCHEMA,
        "profile_name": "probe",
        "status": "research_probe_only",
        "validated_for_delta_optimal": False,
        "checkpoint_fingerprint": "sha256:test",
        "vae_fingerprint": "sha256:test-vae",
        "fit": {"amplitude": 200.0, "beta": 2.0, "r_squared": 0.95},
    }
    kwargs = dict(
        width=1056,
        height=608,
        steps=20,
        scales="0.5,1.0",
        transition_mode="delta_optimal",
        manual_transition_sigmas="0.85",
        delta=0.01,
        shift_video=12.0,
        transform="dct",
        fallback_policy="error",
        spectrum_profile=profile,
    )
    with pytest.raises(ValueError, match="research probe"):
        build_speed_plan(profile_policy="require_validated_profile", **kwargs)
    plan, _ = build_speed_plan(profile_policy="allow_research_profile", **kwargs)
    assert plan["profile"]["amplitude"] == 200.0


def test_delta_profile_binding_rejects_task_and_fingerprint_mismatches():
    plan = {
        "transition_mode": "delta_optimal",
        "profile_policy": "require_validated_profile",
        "profile": {
            "task_family": "T2VA",
            "checkpoint_fingerprint": "sha256:model-a",
            "vae_fingerprint": "sha256:vae-a",
        },
    }
    source = {
        "resolved_task": "t2va",
        "checkpoint_fingerprint": "sha256:model-a",
        "vae_fingerprint": "sha256:vae-a",
    }
    assert _profile_binding(plan, source)["status"] == "matched"
    with pytest.raises(ValueError, match="task mismatch"):
        _profile_binding(plan, {**source, "resolved_task": "ref2va"})
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _profile_binding(plan, {**source, "vae_fingerprint": "sha256:vae-b"})
    with pytest.raises(ValueError, match="requires runtime source fingerprints"):
        _profile_binding(plan, {**source, "checkpoint_fingerprint": "unrecorded"})


def test_spectrum_fit_recovers_a_positive_power_law_and_marks_probe_status():
    torch.manual_seed(7)
    latent = torch.randn(1, 24, 8, 16, 24)
    flattened = latent.reshape(-1, 1, 16, 24)
    for _ in range(3):
        flattened = torch.nn.functional.avg_pool2d(
            flattened, kernel_size=3, stride=1, padding=1
        )
    latent = flattened.reshape(1, 24, 8, 16, 24)
    fit = fit_h3_spatial_power_spectrum(latent, max_temporal_samples=4)
    assert fit["amplitude"] > 0.0
    assert fit["beta"] > 0.0
    assert math.isfinite(fit["r_squared"])
    profile, report_json = build_spectrum_profile(
        latent,
        profile_name="unit_probe",
        task_family="T2VA",
        checkpoint_fingerprint="sha256:model",
        vae_fingerprint="sha256:vae",
        independent_clip_count=1,
        minimum_r_squared=0.0,
        max_temporal_samples=4,
    )
    assert profile["validated_for_delta_optimal"] is False
    assert profile["status"] == "research_probe_only"
    assert json.loads(report_json)["schema"] == SPEED_PROFILE_SCHEMA


def test_declared_spectrum_evidence_cannot_promote_one_actual_clip():
    torch.manual_seed(17)
    latent = torch.randn(1, 24, 4, 16, 24)
    flattened = latent.reshape(-1, 1, 16, 24)
    for _ in range(3):
        flattened = torch.nn.functional.avg_pool2d(
            flattened, kernel_size=3, stride=1, padding=1
        )
    latent = flattened.reshape(1, 24, 4, 16, 24)
    profile, _ = build_spectrum_profile(
        latent,
        profile_name="false_dataset_claim",
        task_family="T2VA",
        checkpoint_fingerprint="sha256:model",
        vae_fingerprint="sha256:vae",
        independent_clip_count=100,
        minimum_r_squared=0.0,
        max_temporal_samples=4,
    )
    assert profile["actual_batch_entries"] == 1
    assert profile["declared_evidence_present_in_input"] is False
    assert profile["provenance_complete"] is True
    assert profile["validated_for_delta_optimal"] is False


def test_strict_scope_is_exactly_t2va_native_stock20():
    source = {"resolved_task": "t2va", "audio_mode": "native"}
    assert _task_support(source, "strict_t2va_stock20", 20, 12.0, 3.0)[0] is True
    assert _task_support(source, "strict_t2va_stock20", 8, 12.0, 3.0)[0] is False
    assert _task_support(source, "strict_t2va_stock20", 20, 11.0, 3.0)[0] is False
    assert _task_support(source, "strict_t2va_stock20", 20, 12.0, 4.0)[0] is False
    assert _task_support(
        {**source, "resolved_task": "fl2va"}, "strict_t2va_stock20", 20, 12.0, 3.0
    )[0] is False
    assert _task_support(
        {**source, "audio_mode": "lock_source"}, "strict_t2va_stock20", 20, 12.0, 3.0
    )[0] is False
    assert _task_support(
        {**source, "first_frame": torch.zeros(1, 32, 32, 3)},
        "strict_t2va_stock20",
        20,
        12.0,
        3.0,
    )[0] is False
    assert _task_support(
        {**source, "ref_images": {"ref_image_0": torch.zeros(1, 32, 32, 3)}},
        "strict_t2va_stock20",
        20,
        12.0,
        3.0,
    )[0] is False


def test_strict_t2va_stage_reuses_text_and_rebuilds_only_empty_av_canvas():
    positive = object()
    mux_audio = object()
    template = StageConditioning(
        positive=positive,
        latent={},
        mux_audio=mux_audio,
        conditioned_prompt="fixed prompt",
        media_map="{}",
        report="initial stage",
    )
    stage = _build_empty_t2va_stage(
        {"length": 124},
        width=544,
        height=320,
        template=template,
    )
    video, audio = stage.latent["samples"].unbind()
    assert stage.positive is positive
    assert stage.mux_audio is mux_audio
    assert stage.conditioned_prompt == "fixed prompt"
    assert stage.route == "reused_t2va_text_plus_stage_empty_av"
    assert tuple(video.shape) == (1, 24, 37, 20, 34)
    assert tuple(audio.shape) == (1, 32, 2, 207)
    assert torch.count_nonzero(video) == 0
    assert torch.count_nonzero(audio) == 0


def test_dct_expansion_matches_scipy_reference_and_is_deterministic():
    scipy_fft = pytest.importorskip("scipy.fft")
    source = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) / 10.0
    expanded, aligned, report = dct_expand_official(
        source,
        5,
        7,
        sigma=0.6,
        ratio=math.sqrt((5 / 3) * (7 / 4)),
        seed=123,
        chunk_size=1,
    )
    generator = torch.Generator(device="cpu").manual_seed(123)
    coefficients = torch.randn((1, 5, 7), generator=generator).numpy() * 0.6
    coefficients[0, :3, :4] = scipy_fft.dctn(source[0].numpy(), type=2, norm="ortho")
    reference = scipy_fft.idctn(coefficients[0], type=2, norm="ortho")
    reference *= kappa(0.6, math.sqrt((5 / 3) * (7 / 4)))
    assert torch.allclose(expanded[0], torch.from_numpy(reference), atol=2e-5, rtol=2e-5)
    repeated, repeated_aligned, _ = dct_expand_official(
        source,
        5,
        7,
        sigma=0.6,
        ratio=math.sqrt((5 / 3) * (7 / 4)),
        seed=123,
        chunk_size=1,
    )
    assert torch.equal(expanded, repeated)
    assert aligned == pytest.approx(repeated_aligned)
    assert report["new_coefficients"] == 23
    other_chunk, _, _ = dct_expand_official(
        source,
        5,
        7,
        sigma=0.6,
        ratio=math.sqrt((5 / 3) * (7 / 4)),
        seed=123,
        chunk_size=2,
    )
    assert torch.equal(expanded, other_chunk)


def test_audio_reindex_and_segment_transport_round_trip():
    video = torch.randn(1, 24, 2, 4, 6)
    audio = torch.randn(1, 32, 2, 8)
    external = comfy.nested_tensor.NestedTensor((video, audio))
    sigma = 0.6
    audio_scale = 4.0
    raw = recover_raw_flow_state(external, sigma=sigma, audio_scale=audio_scale)
    raw_video, raw_audio = raw.unbind()
    assert torch.allclose(raw_video, video * 0.4)
    assert torch.allclose(raw_audio, audio * 1.6)

    target_video = torch.randn_like(video)
    target_audio = torch.randn_like(audio)
    target = comfy.nested_tensor.NestedTensor((target_video, target_audio))
    sigma_to = 0.8
    reindexed_audio = reindex_joint_audio_state(
        raw_audio,
        target_audio * audio_scale,
        sigma_from=sigma,
        sigma_to=sigma_to,
    )
    desired = comfy.nested_tensor.NestedTensor((raw_video, reindexed_audio))
    noise = solve_segment_noise(
        desired,
        target,
        sigma=sigma_to,
        audio_scale=audio_scale,
        noise_scale=1.0,
    )
    noise_video, noise_audio = noise.unbind()
    reconstructed_video = sigma_to * noise_video + (1 - sigma_to) * target_video
    reconstructed_audio = sigma_to * noise_audio + (1 - sigma_to) * target_audio * audio_scale
    assert torch.allclose(reconstructed_video, raw_video)
    assert torch.allclose(reconstructed_audio, reindexed_audio)


def test_source_resolves_all_media_without_encoding_and_nodes_are_advanced():
    dummy = object()
    image = torch.zeros(1, 64, 64, 3)
    source, report_json = build_speed_source(
        clip=dummy,
        video_vae=dummy,
        audio_vae=dummy,
        prompt="test",
        length=124,
        task_type="FL2VA",
        audio_mode="native",
        audio_denoise_strength=0.35,
        add_source_as_reference=True,
        prompt_primary_audio_ordinal=1,
        strict_prompt_tags=True,
        ref_image_size="match",
        reference_video_policy="official_2_to_15s",
        checkpoint_fingerprint="unrecorded",
        vae_fingerprint="unrecorded",
        drive_audio=None,
        final_audio=None,
        first_frame=image,
        last_frame=image,
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
    )
    assert source["schema"] == SPEED_SOURCE_SCHEMA
    assert source["resolved_task"] == "fl2va"
    assert json.loads(report_json)["resolved_task"] == "fl2va"
    assert len(SPEED_ADVANCED_NODE_CLASSES) == 4
    for node in SPEED_ADVANCED_NODE_CLASSES:
        schema = node.define_schema()
        assert schema.is_experimental is True
        assert schema.node_id.endswith("Advanced")
        assert schema.category == "T8/MiniMax H3/SPEED/Experimental"


@pytest.mark.parametrize(
    ("filename", "task", "scope", "image_count"),
    [
        ("H3_SPEED_T2VA_Stock20_Advanced_EXP.json", "T2VA", "strict_t2va_stock20", 0),
        ("H3_SPEED_FL2VA_Stock20_Advanced_EXP.json", "FL2VA", "multimodal_research_exp", 2),
        ("H3_SPEED_Ref2VA_Stock20_Advanced_EXP.json", "Ref2VA", "multimodal_research_exp", 1),
    ],
)
def test_speed_frontend_workflows_are_importable_and_self_documenting(
    filename, task, scope, image_count
):
    path = Path(__file__).resolve().parents[1] / "examples" / "workflows" / filename
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert len(nodes) == len(workflow["nodes"])
    assert any(node["type"] == "MiniMaxH3SPEEDPlanT8Advanced" for node in nodes.values())
    source = next(
        node for node in nodes.values() if node["type"] == "MiniMaxH3SPEEDSourceT8Advanced"
    )
    sampler = next(
        node for node in nodes.values() if node["type"] == "MiniMaxH3SPEEDSamplerT8Advanced"
    )
    assert source["widgets_values"][2] == task
    assert sampler["widgets_values"][2] == scope
    assert sum(node["type"] == "LoadImage" for node in nodes.values()) == image_count
    notes = [node for node in nodes.values() if node["type"] == "MarkdownNote"]
    assert notes and task in notes[0]["widgets_values"][0]
    links = {link[0]: link for link in workflow["links"]}
    assert len(links) == len(workflow["links"])
    for link_id, origin, origin_slot, target, target_slot, link_type in workflow["links"]:
        assert link_id in links
        assert origin in nodes and target in nodes
        assert link_id in (nodes[origin]["outputs"][origin_slot].get("links") or [])
        assert nodes[target]["inputs"][target_slot]["link"] == link_id
        assert link_type
