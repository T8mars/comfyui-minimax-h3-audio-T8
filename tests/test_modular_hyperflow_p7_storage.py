"""P7-only frozen LOW/HIGH stages keep parent and sampled-source provenance."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import MethodType

from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
import pytest
import torch.nn.functional as F

from h3_audio_t8_pkg import long_video_delivery as delivery
from h3_audio_t8_pkg.modular_sampling import hyperflow_p7 as p7
from h3_audio_t8_pkg.modular_sampling import hyperflow
from h3_audio_t8_pkg.modular_sampling import hyperflow_p7_storage as frozen
from h3_audio_t8_pkg.modular_sampling import hyperflow_p7_storage_nodes as public
from h3_audio_t8_pkg.modular_sampling import storage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from test_modular_hyperflow import inputs as hyperflow_inputs
from helpers import FakeClip, FakeVideoVAE, FakeAudioVAE


def fake_lift_for(upscaler):
    upscaler = Path(upscaler)
    digest = hashlib.sha256(upscaler.read_bytes()).hexdigest()

    def fake_lift(latent, name, mode, scale, mp, width, height, aspect, anisotropy, precision, release):
        video, audio = latent["samples"].unbind()
        enlarged = {"samples": NestedTensor((F.interpolate(video, size=(video.shape[2], 4, 8),
                                                              mode="nearest"), audio.clone()))}
        report = {"status": "ok", "node": "MiniMaxH3LearnedLatentUpscaleT8Advanced",
                  "geometry": {"size_mode": mode, "aspect_policy": aspect,
                               "source_width": 64, "source_height": 32,
                               "output_width": width, "output_height": height},
                  "model": {"name": name, "path": str(upscaler), "sha256": digest,
                            "precision": precision},
                  "release_policy": release, "audio_preserved": True}
        return enlarged, width, height, json.dumps(report)

    return fake_lift


@pytest.fixture
def pair(tmp_path, monkeypatch):
    output = tmp_path / "output"
    chain_root = tmp_path / "fresh"
    monkeypatch.setattr(frozen.folder_paths, "get_output_directory", lambda: str(output))
    contexts = p7.capture_initial(chain_root, chain_id="fresh", context_frames=22,
                                  width=128, height=64, low_width=64, low_height=32)
    clip, video_vae, audio_vae = FakeClip(), FakeVideoVAE(), FakeAudioVAE()
    low_phase = p7.prepare_phase(contexts, "low", clip=clip, video_vae=video_vae,
                                 audio_vae=audio_vae, prompt="LOW scene.", length=5)
    high_phase = p7.prepare_phase(contexts, "high", clip=clip, video_vae=video_vae,
                                  audio_vae=audio_vae, prompt="HIGH scene.", length=5)
    low_model, high_model, _, _ = hyperflow_inputs(monkeypatch, distinct=True)
    low_setup = p7.setup_low(low_phase, low_model, low_phase.result[0], low_phase.result[0])
    model, sampler, sigmas, source, context, positive, _, _ = low_setup
    sampled_low = sample_stage(RandomNoise.execute(53).result[0],
        BasicGuider.execute(model, positive).result[0], sampler, sigmas, source, context)[2]
    low = p7.bind_low(low_phase, sampled_low)
    assert sampled_low.verify()["portable_identity"] is True
    upscaler = tmp_path / "upscaler.safetensors"
    upscaler.write_bytes(b"P7 routing-only fake upscaler")
    monkeypatch.setattr(p7.folder_paths, "get_full_path_or_raise", lambda _kind, _name: str(upscaler))
    monkeypatch.setattr(p7, "learned_upscale_h3_av_latent", fake_lift_for(upscaler))
    lift = p7.lift_low(low, upscaler.name)
    handoff = p7.handoff_high(lift, high_phase, high_phase.result[0], high_phase.result[0])
    model, sampler, sigmas, source, context, positive, _, _ = p7.setup_high(handoff, high_model)
    sampled_high = sample_stage(RandomNoise.execute(54).result[0],
        BasicGuider.execute(model, positive).result[0], sampler, sigmas, source, context)[2]
    high = p7.bind_high(handoff, sampled_high)
    assert sampled_high.verify()["portable_identity"] is True
    return {"output": output, "chain_root": chain_root, "low_phase": low_phase,
            "high_phase": high_phase, "low": low, "high": high, "handoff": handoff,
            "contexts": contexts, "upscaler": upscaler, "selected_low_model": low_setup[0]}


def test_p7_frozen_low_and_high_are_separate_and_resume_without_low_sampling(pair, monkeypatch):
    low = pair["low"]
    high = pair["high"]
    saved_low = public.MiniMaxH3HyperFlowP7LowSaveEXPT8.execute(low).result
    assert saved_low[0] is low and saved_low[1] is low.sampled.denoised_output
    assert "modular_p7_stages" in str(frozen.stage_root(pair["low_phase"]))
    assert not pair["chain_root"].exists()
    assert json.loads(saved_low[4])["automatic_cache_reuse"] is False
    saved_high = public.MiniMaxH3HyperFlowP7HighSaveEXPT8.execute(high).result
    assert saved_high[0] is high and saved_high[1] is high.sampled.output
    monkeypatch.setattr(p7, "setup_low", lambda *_a, **_k: pytest.fail("LOW sampled during resume"))
    loaded_low = public.MiniMaxH3HyperFlowP7LowLoadEXPT8.execute(
        pair["low_phase"], saved_low[2], saved_low[3]).result
    assert loaded_low[0].sampled.verify()["receipt_sha256"] == low.sampled.verify()["receipt_sha256"]
    assert loaded_low[1]["samples"].unbind()[0].equal(saved_low[1]["samples"].unbind()[0])
    assert public.MiniMaxH3HyperFlowP7LowLoadEXPT8.fingerprint_inputs(
        pair["low_phase"], saved_low[2], saved_low[3]) == frozen.fingerprint(
            pair["low_phase"], saved_low[2])
    loaded_high = public.MiniMaxH3HyperFlowP7HighLoadEXPT8.execute(
        pair["handoff"], saved_high[2], saved_high[3]).result
    assert loaded_high[0].sampled.verify()["receipt_sha256"] == high.sampled.verify()["receipt_sha256"]
    assert loaded_high[1]["samples"].unbind()[1].equal(high.sampled.output["samples"].unbind()[1])


def test_p7_unbound_core_fingerprint_is_exact_for_both_stages(pair):
    low_path, low_sha = frozen.save_low(pair["low"])[2:4]
    high_path, high_sha = frozen.save_high(pair["high"])[2:4]
    low_key = frozen.fingerprint_selected(low_path, frozen.LOW_STAGE)
    high_key = frozen.fingerprint_selected(high_path, frozen.HIGH_STAGE)
    assert public.MiniMaxH3HyperFlowP7LowLoadEXPT8.fingerprint_inputs(
        None, low_path, low_sha) == low_key
    assert public.MiniMaxH3HyperFlowP7HighLoadEXPT8.fingerprint_inputs(
        None, high_path, high_sha) == high_key
    assert low_key == frozen.fingerprint_selected(low_path, frozen.LOW_STAGE)
    assert high_key == frozen.fingerprint_selected(high_path, frozen.HIGH_STAGE)
    with pytest.raises(ValueError, match="exact P7"):
        frozen.fingerprint_selected("../" + low_path, frozen.LOW_STAGE)
    with pytest.raises(ValueError, match="exact P7"):
        frozen.fingerprint_selected(high_path, frozen.LOW_STAGE)


def test_p7_frozen_low_rejects_edited_phase_and_wrong_stage(pair):
    low = pair["low"]
    high = pair["high"]
    path, digest = frozen.save_low(low)[2:4]
    changed = p7.prepare_phase(pair["contexts"], "low", clip=FakeClip(),
                               video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
                               prompt="Edited LOW scene.", length=5)
    with pytest.raises(ValueError, match="not from this P7 source/phase"):
        frozen.load_low(changed, path, digest)
    with pytest.raises(ValueError, match="LOW phase"):
        frozen.load_low(pair["high_phase"], path, digest)
    high_path, high_digest = frozen.save_high(high)[2:4]
    with pytest.raises(ValueError, match="not the expected stage"):
        frozen.load_low(pair["low_phase"], high_path, high_digest)


def test_p7_frozen_low_rejects_corruption_escape_and_parent_change(pair):
    path, digest = frozen.save_low(pair["low"])[2:4]
    root = frozen.stage_root(pair["low_phase"])
    with pytest.raises(ValueError):
        frozen.load_low(pair["low_phase"], "../" + path, digest)
    with pytest.raises(ValueError, match="SHA"):
        frozen.load_low(pair["low_phase"], path, "0" * 64)
    state = root / path.replace("manifest.json", "state.safetensors")
    with state.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="size changed|SHA mismatch"):
        frozen.load_low(pair["low_phase"], path, digest)
    pair["chain_root"].mkdir()
    manifest = delivery._new_manifest("fresh")
    (pair["chain_root"] / delivery.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="initial chain changed"):
        frozen.load_low(pair["low_phase"], path, digest)


def test_p7_cancel_before_commit_never_exposes_completed_manifest(pair, monkeypatch):
    root = frozen.stage_root(pair["low_phase"])
    real_replace = storage.os.replace

    def cancel_manifest(source, target):
        if str(target).endswith("manifest.json"):
            raise RuntimeError("cancel before commit")
        return real_replace(source, target)

    monkeypatch.setattr(storage.os, "replace", cancel_manifest)
    with pytest.raises(RuntimeError, match="cancel before commit"):
        frozen.save_low(pair["low"])
    assert not list(root.rglob("manifest.json"))
    assert list(root.rglob("manifest.json.partial"))


def test_p7_portable_projection_does_not_promote_unknown_extra_conds(pair, monkeypatch):
    model = pair["selected_low_model"]
    assert hyperflow.model_selection(model)["portable_cache_reuse"] is True

    def foreign(_self, **_kwargs):
        return {}

    monkeypatch.setitem(model.object_patches, "extra_conds", MethodType(foreign, model.model))
    assert hyperflow.model_selection(model)["portable_cache_reuse"] is False


def test_p7_frozen_low_resumes_high_in_new_process_without_low_sampling(pair):
    path, digest = frozen.save_low(pair["low"])[2:4]
    code = r'''
import json, runpy, sys
runpy.run_path('tools/check_director_d1_cpu.py', run_name='config')
runpy.run_path('tests/conftest.py')
import pytest
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
from helpers import FakeClip, FakeVideoVAE, FakeAudioVAE
from test_modular_hyperflow import inputs
from test_modular_hyperflow_p7_storage import fake_lift_for
from h3_audio_t8_pkg.modular_sampling import hyperflow_p7 as p7
from h3_audio_t8_pkg.modular_sampling import hyperflow_p7_storage as frozen
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from pathlib import Path
with pytest.MonkeyPatch.context() as patch:
    chain_root, output, upscaler, path, digest = map(Path, sys.argv[1:6])
    contexts = p7.capture_initial(chain_root, chain_id='fresh', context_frames=22,
                                  width=128, height=64, low_width=64, low_height=32)
    clip, video_vae, audio_vae = FakeClip(), FakeVideoVAE(), FakeAudioVAE()
    low_phase = p7.prepare_phase(contexts, 'low', clip=clip, video_vae=video_vae,
                                 audio_vae=audio_vae, prompt='LOW scene.', length=5)
    patch.setattr(p7, 'setup_low', lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('LOW replay')))
    bound, _, _ = frozen.load_low(low_phase, str(path), str(digest), output_root=output)
    patch.setattr(p7.folder_paths, 'get_full_path_or_raise', lambda _kind, _name: str(upscaler))
    patch.setattr(p7, 'learned_upscale_h3_av_latent', fake_lift_for(upscaler))
    lift = p7.lift_low(bound, upscaler.name)
    high_phase = p7.prepare_phase(contexts, 'high', clip=clip, video_vae=video_vae,
                                  audio_vae=audio_vae, prompt='HIGH scene.', length=5)
    handoff = p7.handoff_high(lift, high_phase, high_phase.result[0], high_phase.result[0])
    _, high_model, _, _ = inputs(patch, distinct=True)
    model, sampler, sigmas, source, stage, positive, _, _ = p7.setup_high(handoff, high_model)
    completed = sample_stage(RandomNoise.execute(54).result[0],
        BasicGuider.execute(model, positive).result[0], sampler, sigmas, source, stage)[2]
    result = p7.bind_high(handoff, completed)
    assert result.verify()['selected_socket'] == 'output'
    assert completed.verify()['execution']['hyperflow_fresh']['absolute_apply_intervals'] == [4, 5, 6, 7]
    print('RESULT=' + json.dumps({'high_calls': 4, 'low_calls': 0,
        'low_receipt': bound.sampled.verify()['receipt_sha256']}))
'''
    child = subprocess.run([sys.executable, "-c", code, str(pair["chain_root"]), str(pair["output"]),
                            str(pair["upscaler"]), path, digest],
                           cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=90)
    assert child.returncode == 0, child.stdout + child.stderr
    report = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert report == {"high_calls": 4, "low_calls": 0,
                      "low_receipt": pair["low"].sampled.verify()["receipt_sha256"]}
