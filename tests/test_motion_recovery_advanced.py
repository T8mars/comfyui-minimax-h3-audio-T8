from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import comfy.nested_tensor

from h3_audio_t8_pkg.core import AUDIO_LATENT_FPS, FPS, video_latent_t
from h3_audio_t8_pkg.motion_recovery_advanced import (
    MOTION_PLAN_SCHEMA,
    MOTION_WINDOW_SCHEMA,
    analyze_motion_overload,
    collect_motion_window,
    compose_motion_recovery,
    plan_motion_segment,
    prepare_motion_retiming,
    recover_motion_av,
    route_automatic_motion_recovery,
    validate_motion_plan,
)
from h3_audio_t8_pkg.nodes_motion_recovery_advanced import (
    MOTION_RECOVERY_ADVANCED_NODE_CLASSES,
    MiniMaxH3MotionAutoGateT8Advanced,
)
from helpers import FakeAudioVAE, FakeVideoVAE, make_audio


def _av_latent(frames: int = 124, height: int = 64, width: int = 96, moving=False):
    video = torch.zeros(1, 24, video_latent_t(frames), height // 16, width // 16)
    if moving:
        for token in range(video.shape[2]):
            video[:, :, token, :, token % video.shape[-1]] = float((token % 7) + 1)
    audio = torch.zeros(1, 32, 2, round(frames / FPS * AUDIO_LATENT_FPS))
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def _frames(frames: int = 124, height: int = 64, width: int = 96, moving=False):
    value = torch.zeros(frames, height, width, 3)
    if moving:
        for frame in range(frames):
            left = (frame * 3) % max(1, width - 12)
            value[frame, 16:40, left : left + 12] = 1.0
    return value


def _manual_plan(frames=124, ranges="20-60:4"):
    plan, _preview, should_repair, expanded, _report = analyze_motion_overload(
        _av_latent(frames),
        _frames(frames),
        "manual_ranges",
        ranges,
        3.5,
        2.0,
        0.01,
        4,
        2,
        5,
        24.0,
    )
    assert should_repair is True
    assert expanded > frames
    return plan


def test_stationary_clip_abstains_instead_of_forcing_quantile_expansion():
    plan, preview, should_repair, expanded, report_json = analyze_motion_overload(
        _av_latent(),
        _frames(),
        "auto_conservative_exp",
        "",
        3.5,
        2.0,
        0.01,
        4,
        2,
        5,
        24.0,
    )
    assert plan["schema"] == MOTION_PLAN_SCHEMA
    assert plan["status"] == "abstained"
    assert should_repair is False
    assert expanded == 124
    assert plan["holds"] == [1] * 124
    assert preview.shape == (124, 64, 96, 3)
    report = json.loads(report_json)
    assert report["automatic_gate"] is False
    assert report["claims"]["read_only"] is True
    assert report["claims"]["physical_jerk"] is False


def test_auto_gate_abstain_is_exact_and_never_requests_lazy_second_pass():
    plan, _preview, should_repair, _expanded, _report = analyze_motion_overload(
        _av_latent(),
        _frames(),
        "auto_conservative_exp",
        "",
        3.5,
        2.0,
        0.01,
        4,
        2,
        5,
        24.0,
    )
    baseline_frames = _frames()
    baseline_audio = make_audio(seconds=124 / FPS, sample_rate=32000, value=0.125)
    assert should_repair is False
    assert MiniMaxH3MotionAutoGateT8Advanced.check_lazy_status(
        should_repair,
        baseline_frames,
        baseline_audio,
        plan,
        None,
        None,
    ) == []
    frames, audio, did_repair, report_json = route_automatic_motion_recovery(
        should_repair,
        baseline_frames,
        baseline_audio,
        plan,
        None,
        None,
    )
    assert frames is baseline_frames
    assert audio is baseline_audio
    assert did_repair is False
    report = json.loads(report_json)
    assert report["status"] == "abstained"
    assert report["second_pass_requested"] is False
    assert report["baseline_object_passthrough"] is True


def test_auto_gate_requests_both_lazy_inputs_only_when_repair_is_selected():
    plan = _manual_plan(ranges="20-30:3")
    baseline_frames = _frames()
    baseline_audio = make_audio(seconds=124 / FPS, sample_rate=32000, value=0.125)
    repaired_frames = baseline_frames.clone() + 0.01
    repaired_audio = make_audio(seconds=124 / FPS, sample_rate=32000, value=0.25)
    assert MiniMaxH3MotionAutoGateT8Advanced.check_lazy_status(
        True,
        baseline_frames,
        baseline_audio,
        plan,
        None,
        None,
    ) == ["repaired_frames", "repaired_audio"]
    assert MiniMaxH3MotionAutoGateT8Advanced.check_lazy_status(
        True,
        baseline_frames,
        baseline_audio,
        plan,
        repaired_frames,
        None,
    ) == ["repaired_audio"]
    frames, audio, did_repair, report_json = route_automatic_motion_recovery(
        True,
        baseline_frames,
        baseline_audio,
        plan,
        repaired_frames,
        repaired_audio,
    )
    assert frames is repaired_frames
    assert torch.equal(audio["waveform"], repaired_audio["waveform"])
    assert did_repair is True
    assert json.loads(report_json)["second_pass_requested"] is True


def test_fast_moving_synthetic_clip_can_cross_the_conservative_auto_gate():
    plan, _preview, should_repair, expanded, report_json = analyze_motion_overload(
        _av_latent(moving=True),
        _frames(moving=True),
        "auto_conservative_exp",
        "",
        2.0,
        1.2,
        0.005,
        4,
        2,
        5,
        24.0,
    )
    report = json.loads(report_json)
    assert report["automatic_gate"] is True
    assert plan["status"] == "ready"
    assert should_repair is True
    assert expanded > 124


def test_manual_plan_uses_h3_grid_and_rejects_payload_tampering():
    plan = _manual_plan(ranges="20-30:4,2.0s-2.5s:3")
    assert plan["expanded_length"] == sum(plan["holds"])
    assert (plan["expanded_length"] - 5) % 17 == 0
    assert plan["legal_padding_anchor"] is not None
    assert validate_motion_plan(plan)["plan_sha256"] == plan["plan_sha256"]
    changed = dict(plan)
    changed["holds"] = list(plan["holds"])
    changed["holds"][20] += 1
    with pytest.raises(ValueError, match="SHA-256"):
        validate_motion_plan(changed)


def test_prepare_composer_and_safe_recover_keep_original_audio():
    plan = _manual_plan(ranges="20-30:3")
    frames = _frames()
    source_audio = make_audio(seconds=124 / FPS, sample_rate=32000, value=0.125, channels=2)
    prepared, smeared, smeared_audio, same_plan, prepare_report = prepare_motion_retiming(
        frames,
        plan,
        FakeVideoVAE(),
        None,
        None,
        "none_invent_exp",
    )
    assert smeared.shape[0] == plan["expanded_length"]
    assert smeared_audio["waveform"].shape[-1] == round(
        plan["expanded_length"] / FPS * 32000
    )
    assert prepared["h3_t8_motion_plan_sha256"] == plan["plan_sha256"]
    assert json.loads(prepare_report)["audio_seed_mode"] == "none_invent_exp"

    sigmas = torch.linspace(1.0, 0.0, 21)
    composed, pass2_sigmas, composed_plan, actual_nfe, composer_report = compose_motion_recovery(
        prepared,
        sigmas,
        same_plan,
        "apply_exp",
        0.48,
        2,
    )
    assert composed is not prepared
    assert actual_nfe == 10
    assert pass2_sigmas.numel() == 11
    assert composed_plan["plan_sha256"] == plan["plan_sha256"]
    assert json.loads(composer_report)["automatic_stacking"] is False

    generated = torch.arange(plan["expanded_length"], dtype=torch.float32)
    generated = generated[:, None, None, None].expand(-1, 8, 8, 3).clone()
    generated_audio = make_audio(
        seconds=plan["expanded_length"] / FPS,
        sample_rate=32000,
        value=0.5,
        channels=2,
    )
    recovered, audio, recover_report = recover_motion_av(
        generated,
        generated_audio,
        source_audio,
        plan,
        "pass1_original",
        0.8,
    )
    starts = []
    cursor = 0
    for hold in plan["holds"]:
        starts.append(cursor)
        cursor += hold
    assert recovered.shape[0] == 124
    assert torch.equal(recovered[:, 0, 0, 0], torch.tensor(starts, dtype=torch.float32))
    assert audio["waveform"].shape[-1] == round(124 / FPS * 32000)
    assert torch.all(audio["waveform"] == 0.125)
    assert audio["h3_t8_motion_audio_mode"] == "pass1_original"
    assert audio["h3_t8_motion_plan_sha256"] == plan["plan_sha256"]
    report = json.loads(recover_report)
    assert report["safe_default"] is True
    assert report["phase_vocoder_used"] is False
    assert report["delivery_status"] == "stable_default_exact_pass1"
    assert "none_observed" in report["known_audio_risk"]


def test_experimental_audio_reports_preserve_the_human_listening_boundary():
    plan = _manual_plan(ranges="34-40:2")
    generated_frames = _frames(plan["expanded_length"])
    pass1_audio = make_audio(seconds=124 / FPS, sample_rate=32000, value=0.1, channels=2)
    generated_audio = make_audio(
        seconds=plan["expanded_length"] / FPS,
        sample_rate=32000,
        value=0.2,
        channels=2,
    )

    _frames_pass2, _audio_pass2, report_pass2 = recover_motion_av(
        generated_frames,
        generated_audio,
        pass1_audio,
        plan,
        "pass2_recovered_exp",
        0.8,
    )
    pass2 = json.loads(report_pass2)
    assert pass2["safe_default"] is False
    assert pass2["delivery_status"] == "diagnostic_only_failed_single_clip_listening"
    assert "suddenly become distant" in pass2["known_audio_risk"]

    _frames_blend, _audio_blend, report_blend = recover_motion_av(
        generated_frames,
        generated_audio,
        pass1_audio,
        plan,
        "blend_exp",
        0.8,
    )
    blend = json.loads(report_blend)
    assert blend["safe_default"] is False
    assert blend["delivery_status"] == "opt_in_single_clip_pass_at_mix_0p8"
    assert "does not generalize" in blend["known_audio_risk"]


def test_seeded_audio_prepare_uses_signed_plan_and_expected_audio_clock():
    plan = _manual_plan(ranges="34-40:2")
    source_audio = make_audio(seconds=124 / FPS, sample_rate=32000, value=0.05, channels=2)
    prepared, _smeared, seed_audio, _plan, report_json = prepare_motion_retiming(
        _frames(),
        plan,
        FakeVideoVAE(),
        FakeAudioVAE(),
        source_audio,
        "follow_original_0p5",
    )
    video, audio = tuple(prepared["samples"].unbind())
    video_mask, audio_mask = tuple(prepared["noise_mask"].unbind())
    assert video.shape[2] == video_latent_t(plan["expanded_length"])
    assert audio.shape[-1] == round(plan["expanded_length"] / FPS * AUDIO_LATENT_FPS)
    assert torch.all(video_mask == 1.0)
    assert torch.all(audio_mask == 0.5)
    assert seed_audio["waveform"].shape[-1] == round(
        plan["expanded_length"] / FPS * 32000
    )
    assert "phase-vocoder" in json.loads(report_json)["audio_warning"]


def test_segment_windows_bank_and_resume_without_rerendering_completed_items(tmp_path):
    frames = _frames()
    audio = make_audio(seconds=124 / FPS, sample_rate=32000, value=0.1, channels=2)
    sample_count = audio["waveform"].shape[-1]
    ramp = torch.linspace(-0.75, 0.75, sample_count, dtype=torch.float32)
    audio["waveform"] = ramp[None, None].expand(1, 2, -1).clone()
    parent = _manual_plan(ranges="20-95:4")
    first = plan_motion_segment(frames, audio, parent, 90, 0, 4, "hot_ranges_only")
    window_count = first[5]
    assert window_count > 1
    complete = False
    windows_on_disk = 0
    for index in range(window_count):
        (
            segment,
            window_plan,
            _first,
            _last,
            segment_audio,
            count,
            selected,
            _report,
        ) = plan_motion_segment(
            frames, audio, parent, 90, index, 4, "hot_ranges_only"
        )
        assert window_plan["schema"] == MOTION_WINDOW_SCHEMA
        assert selected == index
        assert count == window_count
        assert segment_audio["waveform"].shape[-1] == round(
            window_plan["world_length"] / FPS * 32000
        )
        segment_audio["h3_t8_motion_audio_mode"] = "pass1_original"
        segment_audio["h3_t8_motion_plan_sha256"] = window_plan["plan_sha256"]
        output_frames, output_audio, complete, windows_on_disk, report_json = collect_motion_window(
            frames,
            audio,
            segment,
            segment_audio,
            window_plan,
            "unit_test",
            str(tmp_path),
            True,
            "float32_exact",
            2,
        )
        assert output_frames.shape == frames.shape
        assert output_audio["waveform"].shape == audio["waveform"].shape
        assert json.loads(report_json)["status"] in {"waiting", "complete"}
    assert complete is True
    assert windows_on_disk == window_count
    complete_report = json.loads(report_json)
    assert complete_report["exact_pass1_audio_bypass_windows"] == list(
        range(window_count)
    )
    assert torch.equal(output_audio["waveform"], audio["waveform"])

    segment, window_plan, *_ = plan_motion_segment(
        frames, audio, parent, 90, 0, 4, "hot_ranges_only"
    )
    window = window_plan["window"]
    start_sample = round(window["source_start"] / FPS * 32000)
    end_sample = round((window["source_end"] + 1) / FPS * 32000)
    rebuilt = collect_motion_window(
        frames,
        audio,
        segment,
        {
            "waveform": audio["waveform"][..., start_sample:end_sample],
            "sample_rate": 32000,
        },
        window_plan,
        "unit_test",
        str(tmp_path),
        False,
        "float32_exact",
        2,
    )
    assert rebuilt[2] is True
    assert json.loads(rebuilt[4])["resume_capable"] is True
    assert json.loads(rebuilt[4])["store_dtype"] == "float32_exact"


def test_append_only_motion_recovery_node_schemas_are_safe_by_default():
    schemas = [node.define_schema() for node in MOTION_RECOVERY_ADVANCED_NODE_CLASSES]
    assert [schema.node_id for schema in schemas] == [
        "MiniMaxH3MotionOverloadAnalyzeT8Advanced",
        "MiniMaxH3MotionRetimingPrepareT8Advanced",
        "MiniMaxH3MotionRecoveryComposerT8Advanced",
        "MiniMaxH3MotionRecoverAVT8Advanced",
        "MiniMaxH3MotionSegmentPlanT8Advanced",
        "MiniMaxH3MotionWindowCollectT8Advanced",
        "MiniMaxH3MotionAutoGateT8Advanced",
    ]
    assert all(schema.is_experimental for schema in schemas)
    analyze_inputs = {item.id: item for item in schemas[0].inputs}
    composer_inputs = {item.id: item for item in schemas[2].inputs}
    recover_inputs = {item.id: item for item in schemas[3].inputs}
    collect_inputs = {item.id: item for item in schemas[5].inputs}
    gate_inputs = {item.id: item for item in schemas[6].inputs}
    assert analyze_inputs["mode"].default == "auto_conservative_exp"
    assert composer_inputs["mode"].default == "report_only"
    assert recover_inputs["audio_mode"].default == "pass1_original"
    assert collect_inputs["store_dtype"].default == "float32_exact"
    assert schemas[4].is_output_node is False
    assert schemas[5].is_output_node is False
    assert gate_inputs["repaired_frames"].lazy is True
    assert gate_inputs["repaired_audio"].lazy is True


@pytest.mark.parametrize(
    ("filename", "windowed"),
    [
        ("2026-08-22_H3_Motion_Recovery_Fullclip_Stock20_Advanced_EXP.json", False),
        ("2026-08-22_H3_Motion_Recovery_Windowed_Stock20_Advanced_EXP.json", True),
    ],
)
def test_motion_recovery_frontend_workflows_are_importable_and_safely_wired(
    filename, windowed
):
    root = Path(__file__).resolve().parents[1]
    workflow = json.loads(
        (
            root
            / "examples"
            / "workflows"
            / "07-motion-detail"
            / filename
        ).read_text(encoding="utf-8")
    )
    nodes = {node["id"]: node for node in workflow["nodes"]}
    by_type = {node["type"]: node for node in workflow["nodes"]}
    link_ids = [int(link[0]) for link in workflow["links"]]
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link_ids)
    assert len(link_ids) == len(set(link_ids))
    assert sum(node["type"] == "MarkdownNote" for node in nodes.values()) == 5
    assert by_type["MiniMaxH3MotionOverloadAnalyzeT8Advanced"]["widgets_values"][:2] == [
        "auto_conservative_exp",
        "",
    ]
    assert by_type["MiniMaxH3MotionRecoveryComposerT8Advanced"]["widgets_values"] == [
        "apply_exp",
        0.48,
        2,
    ]
    assert by_type["MiniMaxH3MotionRecoverAVT8Advanced"]["widgets_values"][0] == (
        "pass1_original"
    )
    gate = by_type["MiniMaxH3MotionAutoGateT8Advanced"]
    analyzer = by_type["MiniMaxH3MotionOverloadAnalyzeT8Advanced"]
    decode1 = next(
        node
        for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3AVDecodeT8" and "pass 1" in node["title"].lower()
    )
    assert any(link[1:5] == [analyzer["id"], 2, gate["id"], 0] for link in workflow["links"])
    assert any(link[1:5] == [decode1["id"], 0, gate["id"], 1] for link in workflow["links"])
    assert any(link[1:5] == [decode1["id"], 1, gate["id"], 2] for link in workflow["links"])
    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert link_id in link_ids
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[target]["inputs"][input_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
    if windowed:
        segment = by_type["MiniMaxH3MotionSegmentPlanT8Advanced"]
        collect = by_type["MiniMaxH3MotionWindowCollectT8Advanced"]
        assert segment["widgets_values"] == [209, 0, 12, "hot_ranges_only"]
        assert collect["widgets_values"] == [
            "motion_recovery_stock20_demo",
            "",
            True,
            "float32_exact",
            6,
        ]
        retiming = by_type["MiniMaxH3MotionRetimingPrepareT8Advanced"]
        source_audio = next(
            item for item in retiming["inputs"] if item["name"] == "source_audio"
        )
        source_audio_slot = retiming["inputs"].index(source_audio)
        assert any(
            link[1:5] == [segment["id"], 4, retiming["id"], source_audio_slot]
            for link in workflow["links"]
        )
    else:
        conditioning = by_type["MiniMaxH3AudioConditioningT8"]
        assert conditioning["widgets_values"][1:4] == [736, 416, 124]


def test_windowed_stock20_api_fixture_uses_the_safe_audio_and_budget_contracts():
    root = Path(__file__).resolve().parents[1]
    prompt = json.loads(
        (
            root
            / "tests"
            / "fixtures"
            / "api"
            / "motion_recovery_windowed_stock20_api.json"
        ).read_text(encoding="utf-8")
    )
    assert prompt["5"]["inputs"]["width"] * prompt["5"]["inputs"]["height"] == 737280
    assert prompt["6"]["inputs"]["steps"] == 20
    assert prompt["9"]["inputs"]["sigmas"] == ["6", 2]
    assert prompt["11"]["inputs"]["mode"] == "auto_conservative_exp"
    assert prompt["12"]["inputs"]["max_expanded_frames"] == 209
    assert prompt["13"]["inputs"]["source_audio"] == ["12", 4]
    assert prompt["22"]["inputs"]["av_latent"] == ["13", 0]
    assert prompt["14"]["inputs"]["sigmas"] == ["22", 2]
    assert prompt["16"]["inputs"]["sampler"] == ["22", 1]
    assert prompt["18"]["inputs"]["audio_mode"] == "pass1_original"
    assert prompt["19"]["inputs"]["store_dtype"] == "float32_exact"
    assert prompt["20"]["inputs"]["images"] == ["24", 0]
    assert prompt["24"]["inputs"]["should_repair"] == ["11", 2]
    assert prompt["24"]["inputs"]["repaired_frames"] == ["19", 0]
