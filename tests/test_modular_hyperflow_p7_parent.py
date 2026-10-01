"""P7 accepted-parent contexts match the existing two-segment runner."""
import json
import hashlib
from pathlib import Path
from types import MethodType, SimpleNamespace

import comfy.patcher_extension
import pytest
import torch
import torch.nn.functional as F
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise

from h3_audio_t8_pkg import long_video_delivery as delivery
from h3_audio_t8_pkg import long_video
from h3_audio_t8_pkg import long_video_dual_picture_context as picture
from h3_audio_t8_pkg import progressive_continuation
from h3_audio_t8_pkg.core import empty_av_latent
from h3_audio_t8_pkg.hyperflow_long_video_exp import runner
from h3_audio_t8_pkg.long_video_in_node_loop_effects_advanced import _write_effects_audit
from h3_audio_t8_pkg.modular_sampling import hyperflow_p7 as p7
from h3_audio_t8_pkg.modular_sampling import hyperflow_p7_nodes as public
from h3_audio_t8_pkg.modular_sampling import hyperflow_fresh as fresh
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from test_progressive_continuation import accepted  # noqa: F401
from test_modular_hyperflow import inputs as hyperflow_inputs
from helpers import FakeClip, FakeVideoVAE, FakeAudioVAE


@pytest.fixture
def p7_chain(accepted):  # noqa: F811
    """Real accepted MP4 and context files; fake diffusion and encoder only."""
    case = accepted
    job = case.request["job_sha256"]
    summary = "8-step dual_clock_euler/native_flow shift12/3; P7 fixture"
    high, _ = empty_av_latent(128, 64, 124)
    high_video, high_audio = high["samples"].unbind()
    high_video.fill_(.1)
    high_audio.fill_(.2)
    high_path = case.descriptor.parent / "p7-high.context.safetensors"
    high_record = delivery._write_context_candidate(high, high_path, "chain", 0, "tiny", summary)
    case.accepted_context.write_bytes(high_path.read_bytes())
    low, _ = empty_av_latent(64, 32, 124)
    low_video, low_audio = low["samples"].unbind()
    low_video.fill_(.3)
    low_audio.copy_(high_audio)
    low_path = case.descriptor.parent / "low.context.safetensors"
    low_record = delivery._write_context_candidate(low, low_path, "chain", 0, "tiny", job)
    case.info.update(context_path=high_path.relative_to(case.root).as_posix(),
                     context_sha256=high_record["sha256"], sampling_summary=summary)
    case.manifest["segments"][0].update(context_sha256=high_record["sha256"], sampling_summary=summary)
    case.descriptor.write_text(json.dumps(case.info), encoding="utf-8")
    case.manifest_path.write_text(json.dumps(case.manifest), encoding="utf-8")
    _write_effects_audit(case.descriptor, {"contract_sha256": job, "segment_index": 0,
        "candidate_id": "parent", "sampling_plan": {"mode": runner.RECIPE,
            "hyperflow": {"intervals": [[0, 4], [4, 8]]},
            "dual_model": {"low_context": {
                "path": low_path.relative_to(case.root).as_posix(),
                "sha256": low_record["sha256"], "audio_source": "completed_second_pass_output"}}}})
    case.low_path = low_path
    case.high_path = high_path
    case.summary = summary
    return case


def request(case, **changes):
    values = {**case.request, "previous_job_sha256": case.request["job_sha256"]}
    values.pop("job_sha256")
    return {**values, **changes}


def encoder():
    return SimpleNamespace(encode=lambda _images: torch.full((1, 24, 12, 2, 4), .25))


def test_real_p7_style_parent_cannot_use_progressive_source(p7_chain):
    with pytest.raises(ValueError, match="execution contract"):
        progressive_continuation.capture_continuation_source(p7_chain.root, **p7_chain.request)
    parent = p7.capture_parent(p7_chain.root, **request(p7_chain))
    assert parent.binding["accepted_context_sha256"] == p7_chain.info["context_sha256"]
    assert parent.revalidate() == parent.binding


@pytest.mark.parametrize("frames", [5, 22, 39])
def test_accepted_low_and_high_contexts_equal_original_p7_runner(p7_chain, frames):
    case = p7_chain
    parent = p7.capture_parent(case.root, **request(case, context_frames=frames))
    current = parent.prepare_contexts(encoder())
    current.verify()
    original = runner.HyperFlowLongVideoSegmentRunner(object(), object(),
        contract={"hyperflow": {"source_sha256": "h" * 64}}, low_width=64,
        low_height=32, upscaler_model="unused")
    high_path = case.root / case.manifest["segments"][0]["context_path"]
    high, _ = delivery._load_accepted_context_file(high_path, "chain", 0, 1)
    low, low_sha = original._low_context(case.root, "chain", SimpleNamespace(index=1),
                                         high, "parent", case.request["job_sha256"])
    media, source = picture.accepted_source(case.root, "parent", 1, "chain")
    expected, _ = picture.prepare_context(low, media, source, encoder(), 64, 32)
    assert low_sha == delivery._sha256_file(case.low_path)
    for name in ("video_tail", "audio_tail"):
        assert torch.equal(current.low[name], expected[name])
        assert torch.equal(current.high[name], high[name])
    assert current.low["metadata"] == expected["metadata"]
    assert current.high["metadata"] == high["metadata"]
    assert current.preparation["additional_sampling_nfe"] == 0
    assert current.low["metadata"]["audio_overhang"] == current.high["metadata"]["audio_overhang"]


@pytest.mark.parametrize("fault", ["revision", "candidate", "accepted_video", "accepted_context",
                                   "low_context", "audit", "summary"])
def test_parent_change_fails_before_reuse(p7_chain, fault):
    case = p7_chain
    parent = p7.capture_parent(case.root, **request(case))
    if fault == "revision":
        case.manifest["revision"] += 1
        case.manifest_path.write_text(json.dumps(case.manifest), encoding="utf-8")
    elif fault == "candidate":
        case.manifest["segments"][0]["candidate_id"] = "other"
        case.manifest_path.write_text(json.dumps(case.manifest), encoding="utf-8")
    elif fault == "accepted_video":
        case.accepted_media.write_bytes(b"changed")
    elif fault == "accepted_context":
        case.accepted_context.write_bytes(b"changed")
    elif fault == "low_context":
        case.low_path.write_bytes(b"changed")
    elif fault == "audit":
        (case.descriptor.parent / "effects_audit.json").write_text("{}", encoding="utf-8")
    else:
        case.info["sampling_summary"] = "changed"
        case.descriptor.write_text(json.dumps(case.info), encoding="utf-8")
    with pytest.raises((ValueError, KeyError)):
        parent.revalidate()


def test_parent_changes_during_vaeprep_and_context_mutation_are_rejected(p7_chain):
    case = p7_chain
    parent = p7.capture_parent(case.root, **request(case))

    def encode(_images):
        case.manifest["revision"] += 1
        case.manifest_path.write_text(json.dumps(case.manifest), encoding="utf-8")
        return torch.full((1, 24, 12, 2, 4), .25)

    with pytest.raises(ValueError, match="revision"):
        parent.prepare_contexts(SimpleNamespace(encode=encode))


def test_prepared_context_audio_change_is_detected(p7_chain):
    parent = p7.capture_parent(p7_chain.root, **request(p7_chain))
    contexts = parent.prepare_contexts(encoder())
    contexts.low["audio_tail"][0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="completed audio"):
        contexts.verify()


def test_actual_two_segment_p7_accepted_parent_is_readable_when_installed():
    root = (Path(__file__).resolve().parents[1] / "artifacts/development"
            / "hyperflow-long-video-p7-4ce4ba4e60c8/output/minimax_h3_t8_long_video"
            / "hf_long_video_p7_4ce4ba4e60c8")
    if not root.is_dir():
        pytest.skip("Optional previously generated P7 accepted media is absent")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    selected = manifest["segments"][0]
    candidate = (root / "candidates/segment_00000" / selected["candidate_id"] / "candidate.json")
    audit = json.loads((candidate.parent / "effects_audit.json").read_text(encoding="utf-8"))
    parent = p7.capture_parent(root, chain_id=manifest["chain_id"], segment_index=1,
        parent_candidate_id=selected["candidate_id"], parent_revision=manifest["revision"],
        previous_job_sha256=audit["contract_sha256"], context_frames=22,
        width=selected["width"], height=selected["height"],
        low_width=selected["width"] // 2, low_height=selected["height"] // 2)
    assert parent.revalidate() == parent.binding
    assert parent.binding["accepted_video_sha256"] == selected["video_sha256"]
    assert parent.binding["low_context_sha256"] == audit["sampling_plan"]["dual_model"]["low_context"]["sha256"]


def test_public_p7_context_ports_and_existing_chain_only(p7_chain, monkeypatch):
    monkeypatch.setattr(public.delivery, "long_video_chain_root", lambda _chain: p7_chain.root)
    source_schema = public.MiniMaxH3HyperFlowP7AcceptedParentEXPT8.define_schema()
    contexts_schema = public.MiniMaxH3HyperFlowP7PrepareContextsEXPT8.define_schema()
    assert source_schema.node_id == "MiniMaxH3HyperFlowP7AcceptedParentEXPT8"
    assert contexts_schema.node_id == "MiniMaxH3HyperFlowP7PrepareContextsEXPT8"
    assert {item.id for item in contexts_schema.inputs} == {"accepted_parent", "video_vae"}
    args = request(p7_chain)
    source_output = public.MiniMaxH3HyperFlowP7AcceptedParentEXPT8.execute(**args).result
    parent, report = source_output[:2]
    assert type(parent) is p7.P7Parent and report == parent.binding_json
    assert tuple(source_output[2:]) == (128, 64, 64, 32)
    assert public.MiniMaxH3HyperFlowP7AcceptedParentEXPT8.fingerprint_inputs(**args) == parent.sha256
    contexts, report = public.MiniMaxH3HyperFlowP7PrepareContextsEXPT8.execute(
        parent, encoder()).result
    assert type(contexts) is p7.P7Contexts and report == contexts.contract_json
    assert contexts.verify()["parent_sha256"] == parent.sha256
    with pytest.raises(ValueError, match="authenticated P7"):
        public.MiniMaxH3HyperFlowP7PrepareContextsEXPT8.execute(object(), encoder())
    p7_chain.manifest["revision"] += 1
    p7_chain.manifest_path.write_text(json.dumps(p7_chain.manifest), encoding="utf-8")
    stale = public.MiniMaxH3HyperFlowP7AcceptedParentEXPT8.fingerprint_inputs(**args)
    assert stale != stale


@pytest.mark.parametrize("frames", [22, 39])
@pytest.mark.parametrize("phase", ["low", "high"])
def test_p7_phase_matches_original_reference_only_conditioning(p7_chain, frames, phase):
    parent = p7.capture_parent(p7_chain.root, **request(p7_chain, context_frames=frames))
    contexts = parent.prepare_contexts(encoder())
    clip, video_vae, audio_vae = FakeClip(), FakeVideoVAE(), FakeAudioVAE()
    chosen = p7.prepare_phase(contexts, phase, clip=clip, video_vae=video_vae,
                              audio_vae=audio_vae, prompt="A pelican rides.", length=124)
    selected_context = contexts.low if phase == "low" else contexts.high
    width, height = (64, 32) if phase == "low" else (128, 64)
    expected = long_video.build_long_video_conditioning(
        clip=clip, video_vae=video_vae, audio_vae=audio_vae,
        context=selected_context, segment_index=1, context_frames=frames,
        context_audio="video_and_audio", prompt="A pelican rides.", width=width, height=height,
        length=124, task_type="T2VA", audio_mode="native", audio_denoise_strength=.35,
        add_source_as_reference=False, prompt_primary_audio_ordinal=0,
        strict_prompt_tags=True, ref_image_size="match",
        reference_video_policy="official_2_to_15s", first_frame_reuse="segment0_only",
        persistent_identity_strategy="single_reference", persistent_identity_interval=1,
        return_details=True)
    assert p7.snapshot(chosen.result) == p7.snapshot(expected)
    assert chosen.verify()["contexts_sha256"] == contexts.verify()["sha256"]
    node_out = public.MiniMaxH3HyperFlowP7ConditioningEXPT8.execute(
        contexts, phase, clip, video_vae, audio_vae, "A pelican rides.", 124).result
    assert type(node_out[0]) is p7.P7Phase and len(node_out) == 8
    assert p7.snapshot(node_out[0].result) == p7.snapshot(expected)
    changed = p7.prepare_phase(contexts, phase, clip=clip, video_vae=video_vae,
                               audio_vae=audio_vae, prompt="Another scene.", length=124)
    assert chosen.verify()["sha256"] != changed.verify()["sha256"]


def test_p7_phase_rejects_invalid_context_or_mutated_output(p7_chain):
    parent = p7.capture_parent(p7_chain.root, **request(p7_chain))
    contexts = parent.prepare_contexts(encoder())
    args = dict(clip=FakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
                prompt="Walking.", length=124)
    with pytest.raises(ValueError, match="authenticated P7"):
        p7.prepare_phase(object(), "low", **args)
    with pytest.raises(ValueError, match="newly generated"):
        p7.prepare_phase(contexts, "high", **{**args, "length": 22})
    prepared = p7.prepare_phase(contexts, "low", **args)
    prepared.result[1]["samples"].unbind()[0][0, 0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="Prepared P7 phase changed"):
        prepared.verify()


@pytest.mark.parametrize("phase", ["low", "high"])
def test_segment_zero_is_read_only_and_matches_original_empty_context(tmp_path, monkeypatch, phase):
    root = tmp_path / "fresh"
    monkeypatch.setattr(public.delivery, "long_video_chain_root", lambda _chain: root)
    node = public.MiniMaxH3HyperFlowP7InitialSegmentEXPT8
    contexts, report, *_ = node.execute("fresh", 22, 128, 64, 64, 32).result
    assert type(contexts) is p7.P7InitialContexts and report == contexts.binding_json
    assert not root.exists() and contexts.low == long_video._empty_context("fresh", 0)
    assert contexts.high == long_video._empty_context("fresh", 0)
    clip, video_vae, audio_vae = FakeClip(), FakeVideoVAE(), FakeAudioVAE()
    prepared = p7.prepare_phase(contexts, phase, clip=clip, video_vae=video_vae,
                                audio_vae=audio_vae, prompt="First scene.", length=124)
    width, height = (64, 32) if phase == "low" else (128, 64)
    expected = long_video.build_long_video_conditioning(
        clip=clip, video_vae=video_vae, audio_vae=audio_vae,
        context=long_video._empty_context("fresh", 0), segment_index=0, context_frames=0,
        context_audio="video_and_audio", prompt="First scene.", width=width, height=height,
        length=124, task_type="T2VA", audio_mode="native", audio_denoise_strength=.35,
        add_source_as_reference=False, prompt_primary_audio_ordinal=0, strict_prompt_tags=True,
        ref_image_size="match", reference_video_policy="official_2_to_15s",
        first_frame_reuse="segment0_only", persistent_identity_strategy="single_reference",
        persistent_identity_interval=1, return_details=True)
    assert p7.snapshot(prepared.result) == p7.snapshot(expected)
    assert prepared.verify()["contexts_sha256"] == contexts.verify()["sha256"]
    assert not root.exists()  # Neither source nor native conditions creates the chain.


def test_segment_zero_rejects_existing_or_changed_manifest(tmp_path, p7_chain):
    with pytest.raises(ValueError, match="accepted segments"):
        p7.capture_initial(p7_chain.root, chain_id="chain", context_frames=22,
                           width=128, height=64, low_width=64, low_height=32)
    root = tmp_path / "empty"
    root.mkdir()
    path = root / delivery.MANIFEST_NAME
    manifest = delivery._new_manifest("empty")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    contexts = p7.capture_initial(root, chain_id="empty", context_frames=22,
                                  width=128, height=64, low_width=64, low_height=32)
    assert contexts.verify()["manifests"][delivery.MANIFEST_NAME]
    manifest["revision"] += 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="initial chain changed"):
        contexts.verify()


def test_initial_low_is_one_real_tiny_core_stage_with_bound_denoised_result(tmp_path, monkeypatch):
    contexts = p7.capture_initial(tmp_path / "fresh", chain_id="fresh", context_frames=22,
                                  width=128, height=64, low_width=64, low_height=32)
    phase = p7.prepare_phase(contexts, "low", clip=FakeClip(), video_vae=FakeVideoVAE(),
                             audio_vae=FakeAudioVAE(), prompt="A first scene.", length=5)
    low, _, _, _ = hyperflow_inputs(monkeypatch, distinct=True)
    selected, sampler, sigmas, source, stage, positive, negative, report = p7.setup_low(
        phase, low, phase.result[0], phase.result[0])
    assert json.loads(report)["sampling_calls"] == 0
    assert stage.start == 0 and stage.end == 4 and positive is negative
    assert source[p7.LOW_SOURCE_KEY]["phase_sha256"] == phase.verify()["sha256"]
    noise = RandomNoise.execute(53).result[0]
    guider = BasicGuider.execute(selected, positive).result[0]
    first = sample_stage(noise, guider, sampler, sigmas, source, stage)[2]
    receipt = first.verify()
    assert receipt["verified_recipe_completion"]
    assert receipt["portable_identity"] is True
    assert receipt["request"]["model"]["long_video_patch"]["version"] == long_video.LONG_VIDEO_PATCH_VERSION
    assert receipt["execution"]["hyperflow_fresh"]["absolute_apply_intervals"] == [0, 1, 2, 3]
    bound = p7.bind_low(phase, first)
    assert bound.verify()["stage_receipt_sha256"] == receipt["receipt_sha256"]
    node_out = public.MiniMaxH3HyperFlowP7LowResultEXPT8.execute(phase, first).result
    assert node_out[0].verify() == bound.verify()
    assert node_out[1] is first.denoised_output and node_out[1] is not first.output
    other = p7.prepare_phase(contexts, "low", clip=FakeClip(), video_vae=FakeVideoVAE(),
                             audio_vae=FakeAudioVAE(), prompt="Another first scene.", length=5)
    with pytest.raises(ValueError, match="not from this P7"):
        p7.bind_low(other, first)
    assert not (tmp_path / "fresh").exists()


def test_p7_low_setup_rejects_high_phase_and_changed_motion_guides(tmp_path, monkeypatch):
    contexts = p7.capture_initial(tmp_path / "fresh", chain_id="fresh", context_frames=22,
                                  width=128, height=64, low_width=64, low_height=32)
    high = p7.prepare_phase(contexts, "high", clip=FakeClip(), video_vae=FakeVideoVAE(),
                            audio_vae=FakeAudioVAE(), prompt="First scene.", length=5)
    low = p7.prepare_phase(contexts, "low", clip=FakeClip(), video_vae=FakeVideoVAE(),
                           audio_vae=FakeAudioVAE(), prompt="First scene.", length=5)
    model, _, _, _ = hyperflow_inputs(monkeypatch)
    with pytest.raises(ValueError, match="LOW phase"):
        p7.setup_low(high, model, high.result[0], high.result[0])
    altered = [[value, {**metadata, "minimax_frame_count": 999}]
               for value, metadata in low.result[0]]
    with pytest.raises(ValueError, match="changed authenticated continuation guides"):
        p7.setup_low(low, model, altered, low.result[0])


def test_fresh_p7_identity_keeps_real_wrappers_and_rejects_foreign_extra_conds(tmp_path, monkeypatch):
    contexts = p7.capture_initial(tmp_path / "fresh", chain_id="fresh", context_frames=22,
                                  width=128, height=64, low_width=64, low_height=32)
    phase = p7.prepare_phase(contexts, "low", clip=FakeClip(), video_vae=FakeVideoVAE(),
                             audio_vae=FakeAudioVAE(), prompt="First.", length=5)
    model, _, _, _ = hyperflow_inputs(monkeypatch)
    selected = p7.setup_low(phase, model, phase.result[0], phase.result[0])[0]
    before = fresh.live_identity(selected, "unverified test")
    selected.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL,
                                  "external", lambda executor, *a, **k: executor(*a, **k))
    after = fresh.live_identity(selected, "unverified test")
    assert before["execution_selection"] != after["execution_selection"]

    def foreign(_self, **_kwargs):
        return {}

    monkeypatch.setattr(selected.model, "extra_conds", MethodType(foreign, selected.model))
    with pytest.raises(ValueError, match="extra_conds owner changed"):
        fresh.live_identity(selected, "unverified test")


def test_fresh_p7_identity_keeps_foreign_live_forward(tmp_path, monkeypatch):
    contexts = p7.capture_initial(tmp_path / "fresh", chain_id="fresh", context_frames=22,
                                  width=128, height=64, low_width=64, low_height=32)
    phase = p7.prepare_phase(contexts, "low", clip=FakeClip(), video_vae=FakeVideoVAE(),
                             audio_vae=FakeAudioVAE(), prompt="First.", length=5)
    model, _, _, _ = hyperflow_inputs(monkeypatch)
    selected = p7.setup_low(phase, model, phase.result[0], phase.result[0])[0]
    before = fresh.live_identity(selected, "unverified test")
    block = selected.model.diffusion_model.blocks[0]
    monkeypatch.setattr(block, "forward", lambda *_args, **_kwargs: None)
    after = fresh.live_identity(selected, "unverified test")
    assert before["execution_selection"] != after["execution_selection"]


def test_p7_separate_lift_reconcile_and_real_high_stage(tmp_path, monkeypatch):
    contexts = p7.capture_initial(tmp_path / "fresh", chain_id="fresh", context_frames=22,
                                  width=128, height=64, low_width=64, low_height=32)
    clip, video_vae, audio_vae = FakeClip(), FakeVideoVAE(), FakeAudioVAE()
    low_phase = p7.prepare_phase(contexts, "low", clip=clip, video_vae=video_vae,
                                 audio_vae=audio_vae, prompt="LOW scene.", length=5)
    high_phase = p7.prepare_phase(contexts, "high", clip=clip, video_vae=video_vae,
                                  audio_vae=audio_vae, prompt="HIGH scene.", length=5)
    low_model, high_model, _, _ = hyperflow_inputs(monkeypatch, distinct=True)
    selected, sampler, sigmas, source, stage, positive, _, _ = p7.setup_low(
        low_phase, low_model, low_phase.result[0], low_phase.result[0])
    first = sample_stage(RandomNoise.execute(53).result[0],
                         BasicGuider.execute(selected, positive).result[0],
                         sampler, sigmas, source, stage)[2]
    bound_low = p7.bind_low(low_phase, first)
    model_file = tmp_path / "tiny-upscaler.safetensors"
    model_file.write_bytes(b"P7 fake learned model for CPU routing test")
    model_sha = hashlib.sha256(model_file.read_bytes()).hexdigest()
    monkeypatch.setattr(p7.folder_paths, "get_full_path_or_raise", lambda _kind, _name: str(model_file))
    calls = []

    def fake_lift(latent, model_name, size_mode, scale_by, target_mp, width, height,
                  aspect, anisotropy, precision, release):
        calls.append((model_name, size_mode, scale_by, target_mp, width, height,
                      aspect, anisotropy, precision, release))
        video, audio = latent["samples"].unbind()
        enlarged = {"samples": NestedTensor((F.interpolate(video, size=(video.shape[2], 4, 8),
                                                              mode="nearest"), audio.clone()))}
        report = {"status": "ok", "node": "MiniMaxH3LearnedLatentUpscaleT8Advanced",
                  "geometry": {"size_mode": size_mode, "aspect_policy": aspect,
                               "source_width": 64, "source_height": 32,
                               "output_width": width, "output_height": height},
                  "model": {"name": model_name, "path": str(model_file), "sha256": model_sha,
                            "precision": precision},
                  "release_policy": release, "audio_preserved": True}
        return enlarged, width, height, json.dumps(report)

    monkeypatch.setattr(p7, "learned_upscale_h3_av_latent", fake_lift)
    lifted = p7.lift_low(bound_low, model_file.name)
    assert calls == [(model_file.name, "target_dimensions", 2.0, 1.0, 128, 64,
                      "honor_dimensions_exp", 1.05, "fp16", "offload_after")]
    assert lifted.verify()["low_sha256"] == bound_low.verify()["sha256"]
    handoff = p7.handoff_high(lifted, high_phase, high_phase.result[0], high_phase.result[0])
    assert handoff.verify()["lift_sha256"] == lifted.verify()["sha256"]
    assert json.loads(handoff.reconcile_json)["audio_policy"]["effective_source"] == "legacy_policy"
    selected, sampler, sigmas, source, stage, positive, _, _ = p7.setup_high(handoff, high_model)
    assert stage.start == 4 and stage.end == 8
    second = sample_stage(RandomNoise.execute(54).result[0],
                          BasicGuider.execute(selected, positive).result[0],
                          sampler, sigmas, source, stage)[2]
    assert second.verify()["portable_identity"] is True
    bound_high = p7.bind_high(handoff, second)
    assert bound_high.verify()["stage_receipt_sha256"] == second.verify()["receipt_sha256"]
    assert public.MiniMaxH3HyperFlowP7HighResultEXPT8.execute(handoff, second).result[1] is second.output
    with pytest.raises(ValueError, match="not from this P7 handoff"):
        p7.bind_high(handoff, first)
    assert not (tmp_path / "fresh").exists()
    model_file.write_bytes(b"another upscaler")
    with pytest.raises(ValueError, match="upscaler file changed"):
        handoff.verify()
