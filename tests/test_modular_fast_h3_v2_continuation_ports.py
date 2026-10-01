"""Accepted-parent V2 ports keep LOW picture and post-reconcile HIGH boundaries explicit."""

import json

from comfy.nested_tensor import NestedTensor
import pytest
import torch

from h3_audio_t8_pkg import long_video
from h3_audio_t8_pkg.audio_ops import trim_av_output
from h3_audio_t8_pkg import long_video_dual_model_runner as legacy_runner
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg import progressive_continuation_relay as legacy_relay
from h3_audio_t8_pkg.modular_sampling import continuation, node_classes
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2_continuation import (
    accepted_delivery_window, accepted_phase_context, lock_accepted_high_prefix, plan_accepted_window,
    project_accepted_relay, reconcile_accepted_high,
)
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2_continuation_nodes import (
    MiniMaxH3FastH3V2AcceptedContextEXPT8,
    MiniMaxH3FastH3V2AcceptedDeliveryWindowEXPT8,
    MiniMaxH3FastH3V2AcceptedHighPrefixEXPT8,
    MiniMaxH3FastH3V2AcceptedReconcileEXPT8,
    MiniMaxH3FastH3V2AcceptedRelayProjectEXPT8,
    MiniMaxH3FastH3V2AcceptedWindowEXPT8,
)
from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE
from test_progressive_continuation import accepted  # noqa: F401


def contexts_for(case):
    source = continuation.capture_source(case.root, **case.request)
    return continuation.prepare_contexts(source, FakeVideoVAE())


def raw_high_conditions(contexts):
    request = contexts.source.binding["request"]
    high_context = accepted_phase_context(contexts, "high")[0]
    return long_video.build_long_video_conditioning(
        clip=FakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        context=high_context, segment_index=request["segment_index"],
        context_frames=request["context_frames"], context_audio="video_and_audio",
        prompt="Continue walking.", width=request["width"], height=request["height"],
        length=124, return_details=True)


def raw_high_latent(contexts):
    return raw_high_conditions(contexts)[1]


def test_accepted_phase_ports_expose_distinct_legacy_sources(accepted):  # noqa: F811
    contexts = contexts_for(accepted)
    low = accepted_phase_context(contexts, "low")
    high = accepted_phase_context(contexts, "high")
    assert low[0] is contexts.low and high[0] is contexts.high
    assert low[1:3] == high[1:3] == (1, accepted.request["context_frames"])
    assert low[3:5] == (accepted.request["low_width"], accepted.request["low_height"])
    assert high[3:5] == (accepted.request["width"], accepted.request["height"])
    assert json.loads(low[5])["source"] == "accepted_rgb24_resize_vae"
    assert json.loads(high[5])["source"] == "accepted_completed_av_context"
    assert low[0]["audio_tail"] is high[0]["audio_tail"]
    assert {cls.__name__ for cls in node_classes()} >= {
        "MiniMaxH3FastH3V2AcceptedContextEXPT8", "MiniMaxH3FastH3V2AcceptedHighPrefixEXPT8",
        "MiniMaxH3FastH3V2AcceptedReconcileEXPT8", "MiniMaxH3FastH3V2AcceptedRelayProjectEXPT8",
        "MiniMaxH3FastH3V2AcceptedWindowEXPT8", "MiniMaxH3FastH3V2AcceptedDeliveryWindowEXPT8"}
    assert MiniMaxH3FastH3V2AcceptedContextEXPT8.GET_NODE_INFO_V1()["output"][0] == long_video.CONTEXT_TYPE_NAME
    assert MiniMaxH3FastH3V2AcceptedHighPrefixEXPT8.GET_NODE_INFO_V1()["output"] == ["LATENT", "STRING"]


def test_accepted_window_uses_parent_clock_and_native_grid(accepted):  # noqa: F811
    contexts = contexts_for(accepted)
    length, new_frames, report_json = plan_accepted_window(contexts, 192)
    report = json.loads(report_json)
    assert (length, new_frames) == (90, 68)
    assert report["accepted_start_frame"] == 124 and report["context_frames"] == 22
    assert MiniMaxH3FastH3V2AcceptedWindowEXPT8.GET_NODE_INFO_V1()["output"] == ["INT", "INT", "STRING"]
    with pytest.raises(ValueError, match=r"17n\+5"):
        plan_accepted_window(contexts, 193)


def test_accepted_old_fixed_window_keeps_124_render_and_68_delivery(accepted):  # noqa: F811
    contexts = contexts_for(accepted)
    length, new_frames, report_json = plan_accepted_window(contexts, 192, "old_fixed_124")
    assert (length, new_frames) == (124, 68)
    report = json.loads(report_json)
    assert report["discarded_suffix_frames"] == 34
    delivery = accepted_delivery_window(contexts, 192, "old_fixed_124")
    assert (round(delivery[5] * 24), round(delivery[6] * 24)) == (22, 68)
    assert json.loads(delivery[-1])["render_frames"] == 124
    with pytest.raises(ValueError, match="exact accepted 8s"):
        plan_accepted_window(contexts, 209, "old_fixed_124")


def test_delivery_port_trims_exact_remainder_and_revalidates_parent(accepted):  # noqa: F811
    contexts = contexts_for(accepted)
    chain, index, parent, revision, timeline, trim, duration, save_context, report_json = \
        accepted_delivery_window(contexts, 192)
    assert (chain, index, parent, revision, save_context) == ("chain", 1, "parent", 1, False)
    assert (round(timeline * 24), round(trim * 24), round(duration * 24)) == (124, 22, 68)
    report = json.loads(report_json)
    assert (report["render_frames"], report["trim_start_frame"], report["new_frames"]) == (90, 22, 68)
    assert report["accepted_source_sha256"] == contexts.source.sha256
    assert MiniMaxH3FastH3V2AcceptedDeliveryWindowEXPT8.GET_NODE_INFO_V1()["output"] == [
        "STRING", "INT", "STRING", "INT", "FLOAT", "FLOAT", "FLOAT", "BOOLEAN", "STRING"]
    frames = torch.arange(90, dtype=torch.float32).view(90, 1, 1, 1).expand(-1, 2, 2, 3)
    waveform = torch.arange(120000, dtype=torch.float32).view(1, 1, -1).expand(-1, 2, -1)
    trimmed, audio, _ = trim_av_output(frames, trim, duration,
                                       {"waveform": waveform, "sample_rate": 32000}, 24.)
    assert len(trimmed) == 68 and float(trimmed[0, 0, 0, 0]) == 22.
    assert audio["waveform"].shape == (1, 2, 90667)
    assert float(audio["waveform"][0, 0, 0]) == 29333.
    accepted.manifest["revision"] += 1
    accepted.manifest_path.write_text(json.dumps(accepted.manifest))
    with pytest.raises(ValueError, match="revision"):
        accepted_delivery_window(contexts, 192)


def test_high_prefix_after_reconcile_preserves_audio_and_old_ramp(accepted):  # noqa: F811
    contexts = contexts_for(accepted)
    latent = raw_high_latent(contexts)
    before_video, before_audio = latent["samples"].unbind()
    output, text = lock_accepted_high_prefix(contexts, latent)
    video, audio = output["samples"].unbind()
    video_mask, audio_mask = output["noise_mask"].unbind()
    steps = long_video.CONTEXT_FRAME_STEPS[accepted.request["context_frames"]]
    assert video is not before_video and audio is before_audio
    torch.testing.assert_close(video[:, :, :steps], contexts.high["video_tail"][:, :, -steps:])
    torch.testing.assert_close(video[:, :, steps:], before_video[:, :, steps:])
    assert torch.all(video_mask[:, :, :steps] == 0)
    assert [float(video_mask[0, 0, steps + index, 0, 0]) for index in range(3)] == [.25, .5, .75]
    assert torch.all(video_mask[:, :, steps + 3:] == 1)
    assert torch.all(audio_mask == 1)
    assert json.loads(text)["after"] == "external_learned_upscale_and_reconcile"
    with pytest.raises(ValueError, match="already owned"):
        lock_accepted_high_prefix(contexts, output)


def test_port_revalidates_changed_parent_and_rejects_wrong_high_geometry(accepted):  # noqa: F811
    contexts = contexts_for(accepted)
    latent = raw_high_latent(contexts)
    video, audio = latent["samples"].unbind()
    wrong = {**latent, "samples": NestedTensor((video[..., :-1], audio))}
    with pytest.raises(ValueError):
        lock_accepted_high_prefix(contexts, wrong)
    accepted.manifest["revision"] += 1
    accepted.manifest_path.write_text(json.dumps(accepted.manifest))
    with pytest.raises(ValueError, match="revision"):
        accepted_phase_context(contexts, "low")
    with pytest.raises(ValueError, match="revision"):
        lock_accepted_high_prefix(contexts, latent)


def test_accepted_relay_projects_exact_old_global_frame_clock(accepted):  # noqa: F811
    contexts = contexts_for(accepted)
    global_plan = relay.build_prompt_relay_plan("Scene.", "Walk.\nStop.\nTurn.", 345,
        "auto_equal", "", "paper_v1", .1, False, False)[0]
    source_copy = json.loads(json.dumps(global_plan))
    actual, prompt, report_json = project_accepted_relay(contexts, global_plan, 124)
    expected = legacy_relay.project_for_source(contexts.source, global_plan, 124)
    assert actual == expected and prompt == expected["compiled_prompt"]
    assert global_plan == source_copy
    report = json.loads(report_json)
    assert report["accepted_source_sha256"] == contexts.source.sha256
    assert report["projection"]["render_start_frame"] == 124 - accepted.request["context_frames"]
    assert MiniMaxH3FastH3V2AcceptedRelayProjectEXPT8.GET_NODE_INFO_V1()["output"][0] == relay.PROMPT_RELAY_PLAN_TYPE


def test_fixed_window_relay_uses_accepted_end_not_discarded_suffix(accepted):  # noqa: F811
    contexts = contexts_for(accepted)
    global_plan = relay.build_prompt_relay_plan("Scene.", "Walk.\nStop.\nTurn.", 193,
        "auto_equal", "", "paper_v1", .1, False, False)[0]
    actual, _, report_json = project_accepted_relay(contexts, global_plan, 124, 192)
    expected = legacy_relay.project_for_source(contexts.source, global_plan, 124, 192)
    assert actual == expected
    assert json.loads(report_json)["length"] == 124
    with pytest.raises(ValueError, match="old fixed render window"):
        project_accepted_relay(contexts, global_plan, 90, 192)


@pytest.mark.parametrize("masked", [False, True])
def test_accepted_joint_audio_reconcile_is_exact_old_legacy_policy(accepted, masked):  # noqa: F811
    contexts = contexts_for(accepted)
    positive, template, *_ = raw_high_conditions(contexts)
    template_video, template_audio = template["samples"].unbind()
    if masked:
        audio_mask = torch.ones_like(template_audio)
        audio_mask[..., :4] = 0
        audio_mask[..., 4:6] = .5
        template = {**template, "noise_mask": NestedTensor((torch.ones_like(template_video), audio_mask))}
    learned = {"samples": NestedTensor((template_video + .25, template_audio + 2))}
    old_runner = object.__new__(legacy_runner.DualModelSegmentRunner)
    old_runner.audio = ("legacy_policy", 0.)
    old_av, old_positive, old_report = old_runner._reconcile(learned, template, positive)
    new_av, new_positive, new_report = reconcile_accepted_high(contexts, learned, template, positive)
    for old_part, new_part in zip(old_av["samples"].unbind(), new_av["samples"].unbind(), strict=True):
        assert torch.equal(old_part, new_part)
    assert ("noise_mask" in old_av) == ("noise_mask" in new_av) == masked
    if masked:
        for old_mask, new_mask in zip(old_av["noise_mask"].unbind(), new_av["noise_mask"].unbind(), strict=True):
            assert torch.equal(old_mask, new_mask)
        assert torch.all(new_av["samples"].unbind()[1][..., :4] == template_audio[..., :4])
        assert torch.all(new_av["samples"].unbind()[1][..., 4:] == learned["samples"].unbind()[1][..., 4:])
    assert torch.equal(old_positive[0][0], new_positive[0][0])
    assert json.loads(old_report).get("dual_audio_handoff") == json.loads(new_report).get("dual_audio_handoff")
    assert json.loads(new_report)["accepted_source_sha256"] == contexts.source.sha256
    assert MiniMaxH3FastH3V2AcceptedReconcileEXPT8.GET_NODE_INFO_V1()["output"] == [
        "LATENT", "CONDITIONING", "STRING"]
