"""Stage persistence preserves native AV state and never silently resamples."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise

from h3_audio_t8_pkg.modular_sampling.fast_h3_v2 import build_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import save_stage, load_stage, file_sha, fingerprint_stage, _lease
from test_modular_results import inputs
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning


def frozen(root, stage="low_0_4"):
    result = sample_stage(*inputs(stage))[2]
    path, digest, _ = save_stage(result, root, "tests/" + stage)
    return result, path, digest


@pytest.mark.parametrize("stage", ["low_0_4", "high_4_8"])
def test_exact_roundtrip_keeps_both_distinct_outputs_and_metadata(tmp_path, stage):
    args = list(inputs(stage))
    args[4]["inspection_metadata"] = {"name": "音画", "tuple": (2, .5), "bytes": b"abc"}
    result = sample_stage(*args)[2]
    path, digest, _ = save_stage(result, tmp_path)
    output, x0, context, restored, report = load_stage(tmp_path, path, digest, stage)
    assert context.stage == stage
    assert restored.verify() == result.verify()
    assert json.loads(report)["automatic_cache_reuse"] is False
    for current, original in ((output, result.output), (x0, result.denoised_output)):
        assert current["inspection_metadata"] == original["inspection_metadata"]
        for a, b in zip(current["samples"].unbind(), original["samples"].unbind()):
            assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()


def test_saved_low_restores_into_only_high_and_matches_uninterrupted_tiny_path(tmp_path):
    low, path, digest = frozen(tmp_path)
    restored_low = load_stage(tmp_path, path, digest, "low_0_4")[1]

    def high(source):
        prepared, sampler, sigmas, context, _ = build_stage(model(), source, "high_4_8", "dense_compat_exp")
        guider = BasicGuider.execute(prepared, conditioning()).result[0]
        return sample_stage(RandomNoise.execute(17).result[0], guider, sampler, sigmas, source, context)

    expected = high(low.denoised_output)
    restored = high(restored_low)
    assert restored[2].verify()["execution"]["denoiser_evaluations"] == 4
    for a, b in zip(expected[0]["samples"].unbind(), restored[0]["samples"].unbind()):
        assert torch.equal(a, b)
    # Same tiny layout: this does not qualify real learned upscale/GPU media.
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("bad", ["empty_sha", "wrong_sha", "wrong_stage", "tensor", "orphan", "manifest"])
def test_missing_or_corrupt_completion_is_never_loaded(tmp_path, bad):
    _, relative, digest = frozen(tmp_path)
    manifest = tmp_path / relative
    stage = "low_0_4"
    if bad == "empty_sha":
        digest = ""
    elif bad == "wrong_sha":
        digest = "0" * 64
    elif bad == "wrong_stage":
        stage = "high_4_8"
    elif bad == "tensor":
        state = manifest.parent / "state.safetensors"
        with state.open("r+b") as handle:
            handle.seek(-1, 2)
            last = handle.read(1)[0]
            handle.seek(-1, 2)
            handle.write(bytes([last ^ 1]))
    elif bad == "orphan":
        manifest.rename(manifest.with_suffix(".partial"))
    elif bad == "manifest":
        manifest.write_text("{}", encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        load_stage(tmp_path, relative, digest, stage)


@pytest.mark.parametrize("relative", ["../manifest.json", "C:/outside/manifest.json", "a/../manifest.json", "NUL/manifest.json"])
def test_path_escape_and_reserved_names_rejected(tmp_path, relative):
    with pytest.raises(ValueError):
        load_stage(tmp_path, relative, "0" * 64, "low_0_4")


def test_unknown_effect_identity_can_be_preserved_but_not_reused_as_certified_state(tmp_path):
    from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig, apply_stage_eav
    noise, guider, sampler, sigmas, source, context = inputs("high_4_8")
    prepared, _, _ = apply_stage_eav(guider.model_patcher, sigmas, source, context, EAVConfig(tau=.2))
    # A genuine unknown execution owner, not the now-authenticated plain EAV.
    previous = prepared.model_options["transformer_options"]["optimized_attention_override"]
    prepared.model_options["transformer_options"]["optimized_attention_override"] = lambda *args, **kwargs: previous(*args, **kwargs)
    guider = BasicGuider.execute(prepared, conditioning()).result[0]
    result = sample_stage(noise, guider, sampler, sigmas, source, context)[2]
    relative, digest, _ = save_stage(result, tmp_path)
    with pytest.raises(ValueError, match="unverified executable identity"):
        load_stage(tmp_path, relative, digest, "high_4_8")


def test_no_overwrite_and_busy_artifact_does_not_read(tmp_path):
    result, relative, digest = frozen(tmp_path)
    second, _, _ = save_stage(result, tmp_path, "tests/low_0_4")
    assert relative != second and file_sha(tmp_path / relative) == digest
    with _lease((tmp_path / relative).parent):
        with pytest.raises(RuntimeError, match="busy"):
            load_stage(tmp_path, relative, digest, "low_0_4")
    load_stage(tmp_path, relative, digest, "low_0_4")


def test_actual_process_lease_is_released_after_owner_termination(tmp_path):
    _, relative, digest = frozen(tmp_path)
    lock = (tmp_path / relative).parent / ".stage.lock"
    if os.name == "nt":
        acquire = "import msvcrt; msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)"
    else:
        acquire = "import fcntl; fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)"
    code = "import sys,time; handle=open(sys.argv[1],'r+b',buffering=0); " + acquire + "; print('ready',flush=True); time.sleep(30)"
    child = subprocess.Popen([sys.executable, "-c", code, str(lock)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        assert child.poll() is None
        with pytest.raises(RuntimeError, match="busy"):
            load_stage(tmp_path, relative, digest, "low_0_4")
    finally:
        child.terminate()
        child.communicate(timeout=5)
    load_stage(tmp_path, relative, digest, "low_0_4")


def test_failed_write_keeps_only_orphan_never_completed_manifest(tmp_path, monkeypatch):
    from h3_audio_t8_pkg.modular_sampling import storage
    result = sample_stage(*inputs())[2]

    def fail(*args, **kwargs):
        raise OSError("intentional interrupted write")

    monkeypatch.setattr(storage, "save_file", fail)
    with pytest.raises(OSError, match="interrupted"):
        save_stage(result, tmp_path)
    assert not list(tmp_path.rglob("manifest.json"))
    assert list(tmp_path.rglob(".stage.lock"))  # OS lease, not stale-owner guessing.


def test_actual_new_process_load_and_only_high_match_source_process(tmp_path):
    _, path, digest = frozen(tmp_path)
    code = """
import json,runpy,sys
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
runpy.run_path('tests/conftest.py')
from h3_audio_t8_pkg.modular_sampling.storage import load_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2 import build_stage
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning
from comfy_extras.nodes_custom_sampler import BasicGuider,RandomNoise
import torch
low = load_stage(sys.argv[1],sys.argv[2],sys.argv[3],'low_0_4')[1]
prepared,sampler,sigmas,context,_ = build_stage(model(),low,'high_4_8','dense_compat_exp')
guider=BasicGuider.execute(prepared,conditioning()).result[0]
receipt=sample_stage(RandomNoise.execute(17).result[0],guider,sampler,sigmas,low,context)[2].verify()
assert receipt['execution']['denoiser_evaluations']==4 and not torch.cuda.is_initialized()
print('RESULT='+json.dumps(receipt['outputs']))
"""
    root = Path(__file__).resolve().parents[1]
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest], cwd=root,
                           capture_output=True, text=True, timeout=40)
    assert child.returncode == 0, child.stderr
    outputs = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    low = load_stage(tmp_path, path, digest, "low_0_4")[1]
    prepared, sampler, sigmas, context, _ = build_stage(model(), low, "high_4_8", "dense_compat_exp")
    guider = BasicGuider.execute(prepared, conditioning()).result[0]
    receipt = sample_stage(RandomNoise.execute(17).result[0], guider, sampler, sigmas, low, context)[2].verify()
    assert outputs == receipt["outputs"]


def test_file_fingerprint_changes_when_external_state_is_replaced(tmp_path):
    _, relative, digest = frozen(tmp_path)
    original = fingerprint_stage(tmp_path, relative)
    assert original == fingerprint_stage(tmp_path, relative)
    state = (tmp_path / relative).parent / "state.safetensors"
    with state.open("ab") as handle:
        handle.write(b"tamper")
    assert fingerprint_stage(tmp_path, relative) != original
    with pytest.raises(ValueError):
        load_stage(tmp_path, relative, digest, "low_0_4")


@pytest.mark.parametrize("change", ["schema", "stage_context", "receipt_sha256", "extra", "state_bytes", "cache_flag"])
def test_matching_outer_sha_cannot_bypass_inner_contract(tmp_path, change):
    _, relative, _ = frozen(tmp_path)
    path = tmp_path / relative
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if change == "schema":
        manifest["schema"] = "future.v999"
    elif change == "stage_context":
        manifest["stage_context"]["video_shape"][3] += 1
    elif change == "receipt_sha256":
        manifest["receipt_sha256"] = "0" * 64
    elif change == "extra":
        manifest["unknown"] = True
    elif change == "state_bytes":
        manifest["state_bytes"] = True
    elif change == "cache_flag":
        manifest["automatic_cache_reuse"] = True
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_stage(tmp_path, relative, file_sha(path), "low_0_4")


def test_public_nodes_execute_sampler_save_load_with_stable_external_fingerprint(tmp_path, monkeypatch):
    from h3_audio_t8_pkg.modular_sampling import nodes
    monkeypatch.setattr(nodes, "_stage_store_root", lambda: tmp_path)
    sampled = nodes.MiniMaxH3StageSamplerEXPT8.execute(*inputs()).result
    saved = nodes.MiniMaxH3StageSaveEXPT8.execute(sampled[2], prefix="public/LOW").result
    restored = nodes.MiniMaxH3StageLoadEXPT8.execute(saved[2], saved[3], "low_0_4").result
    assert restored[3].verify() == sampled[2].verify()
    assert restored[2].stage == "low_0_4"
    first = nodes.MiniMaxH3StageLoadEXPT8.fingerprint_inputs(saved[2], saved[3])
    assert first == nodes.MiniMaxH3StageLoadEXPT8.fingerprint_inputs(saved[2], saved[3])
    assert json.loads(saved[4])["automatic_cache_reuse"] is False


def test_native_video_audio_masks_survive_save_load_exactly(tmp_path):
    from comfy.nested_tensor import NestedTensor
    args = list(inputs())
    vm, am = (torch.ones_like(value) for value in args[4]["samples"].unbind())
    vm[:, :, :1] = 0
    am[..., :4] = 0
    args[4]["noise_mask"] = NestedTensor((vm, am))
    result = sample_stage(*args)[2]
    path, digest, _ = save_stage(result, tmp_path)
    output, denoised, _, _, _ = load_stage(tmp_path, path, digest, "low_0_4")
    for value in (output, denoised):
        assert torch.equal(value["noise_mask"].unbind()[0], vm)
        assert torch.equal(value["noise_mask"].unbind()[1], am)
    assert args[4]["noise_mask"].unbind()[0] is vm


def test_missing_external_fingerprint_does_not_create_a_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        fingerprint_stage(tmp_path, "nonexistent/manifest.json")
    assert list(tmp_path.iterdir()) == []
