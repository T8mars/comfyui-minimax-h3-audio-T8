"""Portable identity of exact own middle-frame closures, not model admission.

Actual tiny Core50+2, miniature Core video Encoder, original HyperFlow sampler.
Only text embeddings and tiny adapter tensors are synthetic. No GPU claim.
"""
from copy import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import comfy.sample
import pytest
import torch

from h3_audio_t8_pkg.modular_sampling import hyperflow as stages
from h3_audio_t8_pkg.modular_sampling import hyperflow_effects as effects
from h3_audio_t8_pkg.modular_sampling import hyperflow_identity as identity
from h3_audio_t8_pkg.modular_sampling import hyperflow_storage as storage
from h3_audio_t8_pkg import multikeyframe_advanced as multi
from h3_audio_t8_pkg.multikeyframe_advanced import MULTIKEYFRAME_PATCH_VERSION, append_keyframe_plan
from h3_audio_t8_pkg.nodes_director_keyframes import MiniMaxH3DirectorKeyframeConditioningEXPT8
from helpers import FakeAudioVAE
from test_h21_control_regions import RealTinyEncoderOwner
from test_h21_director_keyframe_runtime import CountingNativeLikeFakeClip
from test_h21_hyperflow_keyframe_runtime import prepare, check_original
from test_hyperflow_advanced import _tiny_native_model, _tiny_weights
from test_progressive_sampling_runtime import tiny_model


def real_encoder():
    # A reproducible actual tiny Encoder in a fresh process, not saved latents
    # or monkeypatched encode output. Do not initialize CUDA RNG devices.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21102195)
        return RealTinyEncoderOwner()


@pytest.fixture(scope="module")
def actual_encoder():
    return real_encoder()


def real_base(monkeypatch):
    _, diffusion = _tiny_native_model(monkeypatch)
    base = tiny_model()
    base.model.diffusion_model = diffusion
    original = {key: value.detach().clone() for key, value in base.model_state_dict().items()}
    return base, _tiny_weights(diffusion), original


@pytest.fixture
def actual_base(monkeypatch):
    return real_base(monkeypatch)


def actual_head(built):
    model, positive, source = built[:3]
    bound = effects.bind_head(model, source, positive, [], 4)
    calls = []
    boundary = stages.sample_head(bound[0], source, positive, [],
        comfy.sample.prepare_noise(source["samples"], 21102195), seed=21102195,
        callback=lambda step, _p, _x, total: calls.append((step, total)))
    receipt = boundary.verify()
    assert calls == [(index, 8) for index in range(4)]
    assert receipt["execution"]["actual_apply_intervals"] == list(range(4))
    assert receipt["execution"]["actual_forward_coverage"] is True
    assert all(torch.isfinite(part).all() for part in boundary.scaffold["samples"].unbind())
    assert not torch.cuda.is_initialized()
    return boundary, bound[0]


@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("nonuniform", [False, True])
def test_exact_own_advanced_closures_have_actual_portable_head(
        actual_base, actual_encoder, relay, nonuniform):
    base, weights, original = actual_base
    built = prepare(base, weights, actual_encoder, relay=relay, nonuniform=nonuniform)
    boundary, _ = actual_head(built)
    check_original(base, original)
    receipt = boundary.verify()
    assert receipt["execution"]["portable_identity"] is True, receipt["request"]["model"]
    assert ("per_condition_forward" in receipt["request"]["model"]["advanced_patch"]) is nonuniform


@pytest.mark.parametrize("target", ["extra_conds", "diffusion_model._forward"])
def test_unknown_foreign_delegate_runs_but_copied_marker_cannot_certify_it(
        actual_base, actual_encoder, target):
    base, weights, original = actual_base
    built = prepare(base, weights, actual_encoder, relay=False, nonuniform=True)
    model = built[0]
    delegate = model.get_model_object(target)
    calls = []

    def foreign(_self, *args, **kwargs):
        calls.append(1)
        return delegate(*args, **kwargs)

    foreign._t8_multikeyframe_patch_version = MULTIKEYFRAME_PATCH_VERSION
    owner = model.model if target == "extra_conds" else model.model.diffusion_model
    selected = types.MethodType(foreign, owner)
    model.add_object_patch(target, selected)
    boundary, _ = actual_head(built)
    assert boundary.verify()["execution"]["portable_identity"] is False
    assert calls and model.get_model_object(target) is selected
    check_original(base, original)


@pytest.mark.parametrize("helper", ["_condition_rows_with_augs", "_repair_layout_positions", "call_h3_final_layer"])
def test_runtime_replaced_helper_is_not_certified_from_unchanged_file_sha(
        actual_base, actual_encoder, monkeypatch, helper):
    base, weights, original = actual_base
    native_helper = getattr(multi, helper)
    calls = []

    def selected_helper(*args, **kwargs):
        calls.append(1)
        return native_helper(*args, **kwargs)

    monkeypatch.setattr(multi, helper, selected_helper)
    built = prepare(base, weights, actual_encoder, relay=False, nonuniform=True)
    boundary, _ = actual_head(built)
    assert calls
    assert boundary.verify()["execution"]["portable_identity"] is False
    check_original(base, original)


@pytest.mark.parametrize("target,field", [
    ("extra_conds", "original_extra_conds"),
    ("diffusion_model._forward", "original_forward"),
])
def test_exact_closure_with_another_native_delegate_owner_still_cannot_certify(
        actual_base, actual_encoder, target, field):
    base, weights, original = actual_base
    built = prepare(base, weights, actual_encoder, relay=False, nonuniform=True)
    selected = built[0].get_model_object(target).__func__
    cell = dict(zip(selected.__code__.co_freevars, selected.__closure__, strict=True))[field]
    delegate = cell.cell_contents
    # Actual native function but a different owner object; shallow tensors are
    # intentionally equal, so a mere class/function/content-name check is weak.
    cell.cell_contents = types.MethodType(delegate.__func__, copy(delegate.__self__))
    boundary, _ = actual_head(built)
    assert boundary.verify()["execution"]["portable_identity"] is False
    check_original(base, original)


def test_identity_projection_and_warm_core_cleanup_never_mutate_live_owners(actual_base, actual_encoder):
    base, weights, original = actual_base
    built = prepare(base, weights, actual_encoder, relay=True, nonuniform=True)
    selected = effects.bind_head(built[0], built[2], built[1], [], 4)[0]
    before_extra = base.model.extra_conds
    before_forward = base.model.diffusion_model._forward
    before_objects = dict(selected.object_patches)
    before = identity.model_identity(selected)
    assert before["portable_cache_reuse"] is True
    assert base.model.extra_conds == before_extra
    assert base.model.diffusion_model._forward == before_forward
    assert all(selected.object_patches[key] is value for key, value in before_objects.items())
    boundary = stages.sample_head(selected, built[2], built[1], [],
        comfy.sample.prepare_noise(built[2]["samples"], 21102195), seed=21102195)
    assert boundary.verify()["execution"]["portable_identity"] is True
    assert identity.model_identity(selected) == before
    assert base.model.extra_conds == before_extra
    assert base.model.diffusion_model._forward == before_forward
    assert all(selected.object_patches[key] is value for key, value in before_objects.items())
    check_original(base, original)


def prepare_changed_high(base, weights, encoder, change):
    from h3_audio_t8_pkg.modular_sampling.hyperflow_fresh import install
    selected = install(base, weights)[0]
    clip = CountingNativeLikeFakeClip()
    image = torch.linspace(.05, .95, 32 * 32 * 3).reshape(1, 32, 32, 3)
    plan = append_keyframe_plan(None, image * (.7 if change == "asset" else .9),
        "frame", 2 if change == "time" else 1, .92, "stretch", True)
    return MiniMaxH3DirectorKeyframeConditioningEXPT8.execute(model=selected, clip=clip,
        video_vae=encoder, audio_vae=FakeAudioVAE(), prompt="A person moves.",
        width=32, height=32, length=22, task_type="FL2VA", audio_mode="native",
        audio_denoise_strength=.35, add_source_as_reference=False, prompt_primary_audio_ordinal=0,
        strict_prompt_tags=True, ref_image_size="match", reference_video_policy="official_2_to_15s",
        first_frame_noise_aug=.999, last_frame_noise_aug=.999, reference_visual_noise_aug=.999,
        first_frame=image, last_frame=image * .8, keyframe_plan=plan).result


@pytest.mark.parametrize("change", ["time", "asset"])
def test_high_time_or_asset_change_is_allowed_but_cannot_reuse_old_request_identity(
        actual_base, actual_encoder, change):
    base, weights, original = actual_base
    low = prepare(base, weights, actual_encoder, relay=False, nonuniform=True)
    boundary, _ = actual_head(low)
    high = prepare(base, weights, actual_encoder, relay=False, nonuniform=True)
    first = effects.bind_tail(boundary, high[0], high[1], [])
    _, first_text = stages.sample_tail(boundary, first[0], first[1], [], seed=21102195)
    changed = prepare_changed_high(base, weights, actual_encoder, change)
    second = effects.bind_tail(boundary, changed[0], changed[1], [])
    output, second_text = stages.sample_tail(boundary, second[0], second[1], [], seed=21102195)
    first_report, second_report = json.loads(first_text), json.loads(second_text)
    assert first_report["execution"]["portable_identity"] is True
    assert second_report["execution"]["portable_identity"] is True
    assert first_report["request"]["model"]["common_base_sha256"] == second_report["request"]["model"]["common_base_sha256"]
    assert first_report["request"]["inputs"] != second_report["request"]["inputs"]
    assert stages.sha(first_report["request"]) != stages.sha(second_report["request"])
    assert all(torch.isfinite(value).all() for value in output["samples"].unbind())
    check_original(base, original)


@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("low_nonuniform", [False, True])
def test_uniform_and_nonuniform_high_are_independent_valid_selected_conditions(
        actual_base, actual_encoder, relay, low_nonuniform):
    base, weights, original = actual_base
    low = prepare(base, weights, actual_encoder, relay=relay, nonuniform=low_nonuniform)
    boundary, _ = actual_head(low)
    high = prepare(base, weights, actual_encoder, relay=relay, nonuniform=not low_nonuniform)
    bound = effects.bind_tail(boundary, high[0], high[1], [])
    output, text = stages.sample_tail(boundary, bound[0], bound[1], [], seed=21102195)
    report = json.loads(text)
    assert boundary.verify()["execution"]["portable_identity"] is True
    assert report["execution"]["portable_identity"] is True
    assert report["execution"]["actual_apply_intervals"] == [4, 5, 6, 7]
    assert report["head_executed"] is report["fresh_noise"] is False
    assert report["request"]["model"]["common_base_sha256"] == boundary.verify()["request"]["model"]["common_base_sha256"]
    assert report["request"]["model"]["sha256"] != boundary.verify()["request"]["model"]["sha256"]
    assert ("per_condition_forward" in boundary.verify()["request"]["model"]["advanced_patch"]) is low_nonuniform
    assert ("per_condition_forward" in report["request"]["model"]["advanced_patch"]) is not low_nonuniform
    assert all(torch.isfinite(value).all() for value in output["samples"].unbind())
    check_original(base, original)


@pytest.mark.parametrize("fault", ["marker", "helper", "code"])
def test_same_live_model_midstage_mutation_never_gets_a_completion_and_fresh_retry_runs(
        actual_base, actual_encoder, monkeypatch, fault):
    base, weights, original = actual_base
    built = prepare(base, weights, actual_encoder, relay=False, nonuniform=True)
    selected = effects.bind_head(built[0], built[2], built[1], [], 4)[0]
    assert identity.model_identity(selected)["portable_cache_reuse"] is True
    helper = multi._condition_rows_with_augs
    original_helper = types.FunctionType(helper.__code__, helper.__globals__,
        helper.__name__, helper.__defaults__, helper.__closure__)
    changed = compile("def replacement(self, payload, device):\n"
        "    return _h21_original_helper(self, payload, device)\n", "<h21-live-code>", "exec")
    changed_code = next(item for item in changed.co_consts if isinstance(item, types.CodeType))
    calls = []
    with monkeypatch.context() as patch:
        if fault == "code":
            patch.setattr(multi, "_h21_original_helper", original_helper, raising=False)

        def observe(step, _prediction, _state, total):
            calls.append((step, total))
            if step != 1:
                return
            if fault == "marker":
                function = selected.object_patches["extra_conds"].__func__
                patch.setattr(function, "_t8_multikeyframe_patch_version", MULTIKEYFRAME_PATCH_VERSION + 1)
            elif fault == "helper":
                def foreign_helper(*args, **kwargs):
                    return original_helper(*args, **kwargs)
                patch.setattr(multi, "_condition_rows_with_augs", foreign_helper)
            else:
                patch.setattr(helper, "__code__", changed_code)

        with pytest.raises(ValueError, match="MODEL changed"):
            stages.sample_head(selected, built[2], built[1], [],
                comfy.sample.prepare_noise(built[2]["samples"], 21102195),
                seed=21102195, callback=observe)
    assert calls == [(index, 8) for index in range(4)]
    retry, _ = actual_head(built)
    assert retry.verify()["execution"]["portable_identity"] is True
    check_original(base, original)


def test_changed_actual_common_base_content_cannot_continue_old_portable_head(
        actual_base, actual_encoder, monkeypatch):
    base, weights, original = actual_base
    boundary, _ = actual_head(prepare(base, weights, actual_encoder, relay=False, nonuniform=True))
    assert boundary.verify()["execution"]["portable_identity"] is True
    other, other_weights, _ = real_base(monkeypatch)
    with torch.inference_mode():
        other.model.diffusion_model.blocks[0].attn.qkv_proj.weight.add_(.1)
    high = prepare(other, other_weights, actual_encoder, relay=False, nonuniform=True)
    bound = effects.bind_tail(boundary, high[0], high[1], [])
    calls = []
    with pytest.raises(ValueError, match="same common base"):
        stages.sample_tail(boundary, bound[0], bound[1], [], seed=21102195,
            callback=lambda *args: calls.append(args))
    assert calls == []
    boundary.verify()
    check_original(base, original)


def test_saved_middle_head_real_byte_damage_is_rejected_by_selected_sha(
        actual_base, actual_encoder, tmp_path):
    base, weights, original = actual_base
    boundary, _ = actual_head(prepare(base, weights, actual_encoder, relay=False, nonuniform=True))
    assert boundary.verify()["execution"]["portable_identity"] is True
    path, digest, _ = storage.save_boundary(boundary, tmp_path)
    artifact = tmp_path / path
    before = artifact.read_bytes()
    artifact.write_bytes(before[:-1] + bytes([before[-1] ^ 1]))
    with pytest.raises(ValueError, match="SHA mismatch"):
        storage.load_boundary(tmp_path, path, digest)
    check_original(base, original)


@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("recording_profile", ["native", "record", "record_ref"])
def test_saved_middle_head_really_replays_cold_tail_only_without_noise(
        actual_base, actual_encoder, monkeypatch, tmp_path, relay, recording_profile):
    base, weights, original = actual_base
    recording = recording_profile != "native"
    source_as_reference = recording_profile == "record_ref"
    low = prepare(base, weights, actual_encoder, relay=relay, nonuniform=True,
        recording=recording, source_as_reference=source_as_reference)
    boundary, _ = actual_head(low)
    assert boundary.verify()["execution"]["portable_identity"] is True
    path, digest, saved = storage.save_boundary(boundary, tmp_path)
    assert json.loads(saved)["portable_identity"] is True
    high = prepare(base, weights, actual_encoder, relay=relay, nonuniform=True,
        recording=recording, source_as_reference=source_as_reference)
    tail = effects.bind_tail(boundary, high[0], high[1], [])
    expected, text = stages.sample_tail(boundary, tail[0], tail[1], [], seed=21102195)
    assert json.loads(text)["execution"]["portable_identity"] is True
    code = r'''
import gc,json,runpy,sys
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
runpy.run_path('tests/conftest.py')
import pytest,torch,comfy.sample
from test_h21_hyperflow_keyframe_identity import real_base,real_encoder,prepare
from h3_audio_t8_pkg.modular_sampling import hyperflow as stages,hyperflow_effects as effects,hyperflow_storage as storage
def forbidden(*a,**kw):
    raise AssertionError('Cold middle-keyframe replay ran HEAD or prepared fresh noise')
with pytest.MonkeyPatch.context() as patch:
    base,weights,original=real_base(patch)
    encoder=real_encoder()
    high=prepare(base,weights,encoder,relay=sys.argv[4]=='True',nonuniform=True,
        recording=sys.argv[5]!='native',source_as_reference=sys.argv[5]=='record_ref')
    boundary,_=storage.load_boundary(sys.argv[1],sys.argv[2],sys.argv[3])
    tail=effects.bind_tail(boundary,high[0],high[1],[])
    patch.setattr(stages,'sample_head',forbidden)
    patch.setattr(comfy.sample,'prepare_noise',forbidden)
    output,text=stages.sample_tail(boundary,tail[0],tail[1],[],seed=21102195)
    receipt=json.loads(text)
    assert receipt['execution']['portable_identity'] is True
    assert receipt['execution']['actual_apply_intervals']==[4,5,6,7]
    assert receipt['head_executed'] is receipt['fresh_noise'] is False
    assert not torch.cuda.is_initialized()
    print('RESULT='+json.dumps(stages.snapshot(output),sort_keys=True))
del base,weights,original,encoder,high,boundary,tail,output
gc.collect()
'''
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    import folder_paths
    env["PYTHONPATH"] = str(Path(folder_paths.__file__).resolve().parent) + os.pathsep + str(root / "tests") + os.pathsep + env.get("PYTHONPATH", "")
    child = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", code,
        str(tmp_path), path, digest, str(relay), recording_profile], cwd=root, env=env, capture_output=True,
        text=True, encoding="utf-8", timeout=120)
    assert child.returncode == 0, child.stdout + child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == stages.snapshot(expected)
    assert storage.file_sha(tmp_path / path) == digest
    check_original(base, original)
