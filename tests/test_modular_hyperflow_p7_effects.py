"""P7 external Relay/EAV must route real stage calls, not just expose sockets."""
import json
from pathlib import Path
import subprocess
import sys

from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
import pytest

from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.modular_sampling import eav
from h3_audio_t8_pkg.modular_sampling import hyperflow_p7 as p7
from h3_audio_t8_pkg.modular_sampling import hyperflow_p7_effects as effects
from h3_audio_t8_pkg.modular_sampling import hyperflow_p7_effect_nodes as public
from h3_audio_t8_pkg.modular_sampling import hyperflow_p7_storage as frozen
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from helpers import FakeVideoVAE, FakeAudioVAE
from test_modular_hyperflow import inputs as hyperflow_inputs
from test_modular_hyperflow_p7_storage import fake_lift_for
from test_modular_hyperflow_p7_parent import p7_chain, request, encoder  # noqa: F401
from test_progressive_continuation import accepted  # noqa: F401
from test_prompt_relay_long_video_advanced import NativeLikeFakeClip


def initial(tmp_path, monkeypatch):
    contexts = p7.capture_initial(tmp_path / "fresh", chain_id="fresh", context_frames=22,
                                  width=128, height=64, low_width=64, low_height=32)
    global_plan = relay.build_prompt_relay_plan("Scene.", "Walk.\nTurn.", 22,
        "auto_equal", "", "paper_v1", .1, False, False)[0]
    projected = effects.project_relay(contexts, global_plan, 22)
    clip = NativeLikeFakeClip()
    phase = p7.prepare_phase(contexts, "low", clip=clip, video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(), prompt=projected.projected["compiled_prompt"], length=22)
    low_model, high_model, _, _ = hyperflow_inputs(monkeypatch, distinct=True)
    return contexts, global_plan, projected, clip, phase, low_model, high_model


def test_initial_low_external_relay_and_eav_real_calls(tmp_path, monkeypatch):
    contexts, global_plan, projected, clip, phase, model, _ = initial(tmp_path, monkeypatch)
    projected_node = public.MiniMaxH3HyperFlowP7RelayProjectEXPT8.execute(
        contexts, global_plan, 22).result
    assert projected_node[0].verify() == projected.verify()
    assert projected_node[1] == phase.result[3]
    selected, positive, negative, report = public.MiniMaxH3HyperFlowP7RelayApplyEXPT8.execute(
        model, phase, clip, projected, query_chunk_rows=64).result
    assert json.loads(report)["sampling_calls"] == 0
    selected, sampler, sigmas, source, context, positive, negative, _ = p7.setup_low(
        phase, selected, positive, negative)
    selected, telemetry, _ = eav.apply_stage_eav(selected, sigmas, source, context,
        eav.EAVConfig("apply_exp", .2, 0., 1., 32, 3.))
    result = sample_stage(RandomNoise.execute(53).result[0],
        BasicGuider.execute(selected, positive).result[0], sampler, sigmas, source, context)[2]
    receipt = result.verify()
    actual = receipt["execution"]["hyperflow_fresh"]
    assert receipt["verified_recipe_completion"]
    assert receipt["portable_identity"] is True
    assert actual["absolute_apply_intervals"] == [0, 1, 2, 3]
    assert actual["eav"]["selector_calls"] == 200
    assert actual["relay"]["routed_attention_calls"] == 200
    assert telemetry.snapshot()["completed_forwards"] == 4
    bound = p7.bind_low(phase, result)
    assert bound.verify()["stage_receipt_sha256"] == receipt["receipt_sha256"]
    path, digest = frozen.save_low(bound, output_root=tmp_path / "output")[2:4]
    assert frozen.load_low(phase, path, digest, output_root=tmp_path / "output")[0].verify() == bound.verify()


def test_p7_stage_eav_requires_matching_prepared_phase(tmp_path, monkeypatch):
    contexts, _, _, clip, low_phase, model, _ = initial(tmp_path, monkeypatch)
    selected, _, sigmas, source, context, _, _, _ = p7.setup_low(low_phase, model)
    high_phase = p7.prepare_phase(contexts, "high", clip=clip,
        video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(), prompt="HIGH scene.", length=22)
    config = eav.EAVConfig("report_only", .2, 0., 1., 32, 3.)
    with pytest.raises(ValueError, match="HIGH source"):
        public.MiniMaxH3HyperFlowP7StageEAVApplyEXPT8.execute(
            selected, sigmas, source, context, config, high_phase)
    altered = {**source, p7.LOW_SOURCE_KEY: {**source[p7.LOW_SOURCE_KEY], "segment_index": 1}}
    with pytest.raises(ValueError, match="LOW source"):
        public.MiniMaxH3HyperFlowP7StageEAVApplyEXPT8.execute(
            selected, sigmas, altered, context, config, low_phase)
    patched, _, report = public.MiniMaxH3HyperFlowP7StageEAVApplyEXPT8.execute(
        selected, sigmas, source, context, config, low_phase).result
    assert patched is not selected
    assert json.loads(report)["long_video_contract"]["segment_index"] == 0


def test_unpaired_relay_and_bypassed_eav_do_not_claim_completed_effects(tmp_path, monkeypatch):
    _, _, projected, clip, phase, model, _ = initial(tmp_path, monkeypatch)
    selected, positive, negative, _ = effects.apply_relay(model, phase, clip, projected,
                                                          query_chunk_rows=64)
    selected, sampler, sigmas, source, context, _, _, _ = p7.setup_low(
        phase, selected, positive, negative)
    incorrect = [[value, {**meta, relay.PROMPT_RELAY_BINDING_KEY: {"foreign": True}}]
                 for value, meta in positive]
    with pytest.raises(ValueError, match="paired"):
        sample_stage(RandomNoise.execute(53).result[0],
            BasicGuider.execute(selected, incorrect).result[0], sampler, sigmas, source, context)
    selected, telemetry, _ = eav.apply_stage_eav(selected, sigmas, source, context,
        eav.EAVConfig("apply_exp", .2, 0., 1., 32, 3.))
    calls = []

    def foreign(func, *args, **kwargs):
        calls.append(True)
        return func(*args, **kwargs)

    selected.model_options["transformer_options"]["optimized_attention_override"] = foreign
    result = sample_stage(RandomNoise.execute(53).result[0],
        BasicGuider.execute(selected, positive).result[0], sampler, sigmas, source, context)[2]
    actual = result.verify()["execution"]["hyperflow_fresh"]
    assert calls and telemetry.snapshot()["completed_forwards"] == 4
    assert actual["eav"]["selector_calls"] == 0
    assert not actual["composition_verified"]
    assert result.verify()["portable_identity"] is False
    with pytest.raises(ValueError, match="unverified executable stack"):
        frozen.save_low(p7.bind_low(phase, result), output_root=tmp_path / "output")


def test_projected_prompt_and_motion_guides_must_match(tmp_path, monkeypatch):
    contexts, plan, projected, clip, phase, model, _ = initial(tmp_path, monkeypatch)
    assert projected.verify()["contexts_sha256"] == contexts.verify()["sha256"]
    wrong = p7.prepare_phase(contexts, "low", clip=clip, video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(), prompt="Other scene.", length=22)
    with pytest.raises(ValueError, match="encoded prompt"):
        effects.apply_relay(model, wrong, clip, projected)
    plan["compiled_prompt"] = "Changed"
    assert projected.verify()["global_plan"]["compiled_prompt"] != plan["compiled_prompt"]


@pytest.mark.parametrize("scope", ["low", "high", "both"])
def test_external_effects_are_per_phase_and_high_is_independent(tmp_path, monkeypatch, scope):
    contexts, _, projected, clip, low_phase, low_model, high_model = initial(tmp_path, monkeypatch)
    low_positive = low_negative = low_phase.result[0]
    if scope in ("low", "both"):
        low_model, low_positive, low_negative, _ = effects.apply_relay(
            low_model, low_phase, clip, projected, query_chunk_rows=64)
    model, sampler, sigmas, source, context, positive, negative, _ = p7.setup_low(
        low_phase, low_model, low_positive, low_negative)
    if scope in ("low", "both"):
        model, _, _ = eav.apply_stage_eav(model, sigmas, source, context,
            eav.EAVConfig("apply_exp", .2, 0., 1., 32, 3.))
    low_sampled = sample_stage(RandomNoise.execute(53).result[0],
        BasicGuider.execute(model, positive).result[0], sampler, sigmas, source, context)[2]
    low = p7.bind_low(low_phase, low_sampled)
    upscaler = tmp_path / "upscaler.safetensors"
    upscaler.write_bytes(b"P7 effects routing-only upscaler")
    monkeypatch.setattr(p7.folder_paths, "get_full_path_or_raise", lambda _kind, _name: str(upscaler))
    monkeypatch.setattr(p7, "learned_upscale_h3_av_latent", fake_lift_for(upscaler))
    lift = p7.lift_low(low, upscaler.name)
    high_prompt = projected.projected["compiled_prompt"] if scope in ("high", "both") else "High scene."
    high_phase = p7.prepare_phase(contexts, "high", clip=clip, video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(), prompt=high_prompt, length=22)
    high_positive = high_negative = high_phase.result[0]
    if scope in ("high", "both"):
        high_model, high_positive, high_negative, _ = effects.apply_relay(
            high_model, high_phase, clip, projected, query_chunk_rows=64)
    handoff = p7.handoff_high(lift, high_phase, high_positive, high_negative)
    model, sampler, sigmas, source, context, positive, negative, _ = p7.setup_high(handoff, high_model)
    if scope in ("high", "both"):
        model, _, _ = eav.apply_stage_eav(model, sigmas, source, context,
            eav.EAVConfig("apply_exp", .2, 0., 1., 32, 3.))
    high_sampled = sample_stage(RandomNoise.execute(54).result[0],
        BasicGuider.execute(model, positive).result[0], sampler, sigmas, source, context)[2]
    high = p7.bind_high(handoff, high_sampled)
    for sampled, enabled in ((low.sampled, scope in ("low", "both")),
                             (high.sampled, scope in ("high", "both"))):
        receipt = sampled.verify()
        actual = receipt["execution"]["hyperflow_fresh"]
        assert receipt["verified_recipe_completion"] and receipt["portable_identity"]
        assert ("eav" in actual) is enabled and ("relay" in actual) is enabled
        if enabled:
            assert actual["eav"]["selector_calls"] == 200
            assert actual["relay"]["routed_attention_calls"] == 200
    assert high.verify()["completed_audio"] == "second_pass_joint_av"
    assert frozen.save_high(high, output_root=tmp_path / "output")[0] is high


def test_accepted_parent_relay_projection_uses_manifest_frame_boundary(p7_chain):  # noqa: F811
    parent = p7.capture_parent(p7_chain.root, **request(p7_chain))
    contexts = parent.prepare_contexts(encoder())
    global_plan = relay.build_prompt_relay_plan("Scene.", "Walk.\nTurn.", 345,
        "auto_equal", "", "paper_v1", .1, False, False)[0]
    projected = effects.project_relay(contexts, global_plan, 124)
    accepted_end = p7_chain.manifest["segments"][0]["timeline_end_frame"]
    assert projected.projected["long_video_projection"]["accepted_start_frame"] == accepted_end
    assert projected.verify()["contexts_sha256"] == contexts.verify()["sha256"]
    path = Path(p7_chain.manifest_path)
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="accepted parent changed|manifest changed"):
        projected.verify()


def test_effectful_low_loads_in_new_process_and_only_high_samples(tmp_path, monkeypatch):
    contexts, _, projected, clip, phase, model, _ = initial(tmp_path, monkeypatch)
    model, positive, negative, _ = effects.apply_relay(model, phase, clip, projected,
                                                       query_chunk_rows=64)
    model, sampler, sigmas, source, stage, positive, _, _ = p7.setup_low(
        phase, model, positive, negative)
    model, _, _ = eav.apply_stage_eav(model, sigmas, source, stage,
        eav.EAVConfig("apply_exp", .2, 0., 1., 32, 3.))
    low = p7.bind_low(phase, sample_stage(RandomNoise.execute(53).result[0],
        BasicGuider.execute(model, positive).result[0], sampler, sigmas, source, stage)[2])
    output = tmp_path / "output"
    path, digest = frozen.save_low(low, output_root=output)[2:4]
    upscaler = tmp_path / "upscaler.safetensors"
    upscaler.write_bytes(b"P7 new-process effects upscaler")
    code = r'''
import json, runpy, sys
runpy.run_path('tools/check_director_d1_cpu.py', run_name='config')
runpy.run_path('tests/conftest.py')
import pytest
from pathlib import Path
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
from helpers import FakeVideoVAE, FakeAudioVAE
from test_modular_hyperflow import inputs
from test_modular_hyperflow_p7_storage import fake_lift_for
from test_prompt_relay_long_video_advanced import NativeLikeFakeClip
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.modular_sampling import hyperflow_p7 as p7
from h3_audio_t8_pkg.modular_sampling import hyperflow_p7_effects as effects
from h3_audio_t8_pkg.modular_sampling import hyperflow_p7_storage as frozen
from h3_audio_t8_pkg.modular_sampling import eav
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
with pytest.MonkeyPatch.context() as patch:
    root, output, upscaler, path, digest = map(Path, sys.argv[1:6])
    contexts = p7.capture_initial(root, chain_id='fresh', context_frames=22,
                                  width=128, height=64, low_width=64, low_height=32)
    plan = relay.build_prompt_relay_plan('Scene.', 'Walk.\nTurn.', 22,
        'auto_equal', '', 'paper_v1', .1, False, False)[0]
    projected = effects.project_relay(contexts, plan, 22)
    clip = NativeLikeFakeClip()
    phase = p7.prepare_phase(contexts, 'low', clip=clip, video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(), prompt=projected.projected['compiled_prompt'], length=22)
    patch.setattr(p7, 'setup_low', lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('LOW replay')))
    low, _, _ = frozen.load_low(phase, str(path), str(digest), output_root=output)
    patch.setattr(p7.folder_paths, 'get_full_path_or_raise', lambda _kind, _name: str(upscaler))
    patch.setattr(p7, 'learned_upscale_h3_av_latent', fake_lift_for(upscaler))
    lift = p7.lift_low(low, upscaler.name)
    high_phase = p7.prepare_phase(contexts, 'high', clip=clip, video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(), prompt=projected.projected['compiled_prompt'], length=22)
    _, high_model, _, _ = inputs(patch, distinct=True)
    high_model, positive, negative, _ = effects.apply_relay(high_model, high_phase, clip,
        projected, query_chunk_rows=64)
    handoff = p7.handoff_high(lift, high_phase, positive, negative)
    model, sampler, sigmas, source, stage, positive, _, _ = p7.setup_high(handoff, high_model)
    model, _, _ = eav.apply_stage_eav(model, sigmas, source, stage,
        eav.EAVConfig('apply_exp', .2, 0., 1., 32, 3.))
    result = p7.bind_high(handoff, sample_stage(RandomNoise.execute(54).result[0],
        BasicGuider.execute(model, positive).result[0], sampler, sigmas, source, stage)[2])
    receipt = result.sampled.verify()
    observed = receipt['execution']['hyperflow_fresh']
    assert receipt['portable_identity'] and observed['relay']['routed_attention_calls'] == 200
    assert observed['eav']['selector_calls'] == 200
    print('RESULT=' + json.dumps({'low_calls': 0, 'high_calls': 4,
        'low_receipt': low.sampled.verify()['receipt_sha256']}))
'''
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path / "fresh"), str(output),
                            str(upscaler), path, digest], cwd=Path(__file__).resolve().parents[1],
                           capture_output=True, text=True, timeout=120)
    assert child.returncode == 0, child.stdout + child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == {"low_calls": 0, "high_calls": 4,
                      "low_receipt": low.sampled.verify()["receipt_sha256"]}
