"""Explicit SPEED frozen-stage integrity; tiny CPU Core, not trained output."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from h3_audio_t8_pkg import speed_advanced as old
from h3_audio_t8_pkg.modular_sampling import speed_stages as split
from h3_audio_t8_pkg.modular_sampling import speed_storage as frozen
from h3_audio_t8_pkg.modular_sampling import speed_storage_nodes as public
from h3_audio_t8_pkg.modular_sampling.results import _input_identity, sha
from test_modular_speed_stages import _conditioning, _model, _plan, _source, _relay_case


def _first(monkeypatch):
    monkeypatch.setattr(split, "_build_stage", lambda source, width, height: _conditioning(source, width, height))
    monkeypatch.setattr(split, "_build_empty_t2va_stage",
                        lambda source, width, height, prior: _conditioning(source, width, height, prior))
    monkeypatch.setattr(split, "_apply_speed_scoped_headroom", lambda: (None, {"applied": False}))
    plan, source, seed = _plan(), _source(), 724
    low = split.prepare_speed_stage(
        _model(), plan, source, stage_index=0, shift_audio=3.0,
        seed=seed, execution_scope="strict_t2va_stock20",
    )
    output, result, _ = split.sample_speed_stage(
        low[0], low[1], low[2], low[3], low[4], old.H3ModalityStableNoise(seed), low[5],
    )
    return plan, source, seed, low, output, result


def test_frozen_speed_stage_restores_previous_text_and_exact_dct_handoff(monkeypatch, tmp_path):
    plan, source, seed, low, _, live = _first(monkeypatch)
    _, path, digest, _ = frozen.save_speed_stage(live, output_root=tmp_path)
    loaded, prior, report = frozen.load_speed_stage(plan, seed, 3.0, 1, path, digest, output_root=tmp_path)
    assert json.loads(report)["sampling_calls"] == 0
    assert prior.stage.positive[0][0].shape == low[1][0][0].shape
    for actual, expected in zip(loaded.external_output.unbind(), live.external_output.unbind()):
        assert torch.equal(actual, expected)
    high = split.prepare_speed_stage(
        _model(), plan, source, stage_index=1, shift_audio=3.0,
        seed=seed, execution_scope="strict_t2va_stock20", previous_spec=prior,
    )
    assert high[9]["conditioning_route"] == "reused"
    original_noise, _ = split.handoff_speed_stage(live, high[5], dct_chunk_size=4)
    restored_noise, _ = split.handoff_speed_stage(loaded, high[5], dct_chunk_size=4)
    for actual, expected in zip(restored_noise.value.unbind(), original_noise.value.unbind()):
        assert torch.equal(actual, expected)


def test_public_speed_save_load_nodes_use_exact_artifact_and_same_typed_ports(monkeypatch, tmp_path):
    plan, _, seed, _, _, live = _first(monkeypatch)
    original_root = frozen.stage_root
    monkeypatch.setattr(frozen, "stage_root", lambda **kwargs: original_root(output_root=tmp_path))
    saved = public.MiniMaxH3SPEEDStageSaveEXPT8.execute(live).result
    assert saved[0] is live
    loaded = public.MiniMaxH3SPEEDStageLoadEXPT8.execute(
        plan, seed, 3.0, 1, saved[1], saved[2],
    ).result
    assert loaded[0].verify_live()["receipt_sha256"] == live.verify_live()["receipt_sha256"]
    assert loaded[1].index == 0
    assert public.MiniMaxH3SPEEDStageLoadEXPT8.fingerprint_inputs(
        plan, seed, 3.0, 1, saved[1], saved[2],
    ) == frozen.fingerprint_speed_stage(saved[1])


def test_frozen_speed_stage_rejects_wrong_sha_plan_and_seed(monkeypatch, tmp_path):
    plan, source, seed, low, _, live = _first(monkeypatch)
    _, path, digest, _ = frozen.save_speed_stage(live, output_root=tmp_path)
    with pytest.raises(ValueError, match="SHA mismatch"):
        frozen.load_speed_stage(plan, seed, 3.0, 1, path, "0" * 64, output_root=tmp_path)
    with pytest.raises(ValueError, match="receipt or current next-stage identity"):
        frozen.load_speed_stage(plan, seed + 1, 3.0, 1, path, digest, output_root=tmp_path)
    with pytest.raises(ValueError, match="manifest does not belong"):
        frozen.load_speed_stage(_plan("0.4,0.7,1.0", "0.94,0.78", 96), seed, 3.0,
                                1, path, digest, output_root=tmp_path)


@pytest.mark.parametrize("mode", ["report_only", "apply_exp"])
def test_frozen_speed_stage_accepts_completed_external_eav(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(split, "_build_stage", lambda source, width, height: _conditioning(source, width, height))
    monkeypatch.setattr(split, "_build_empty_t2va_stage",
                        lambda source, width, height, prior: _conditioning(source, width, height, prior))
    monkeypatch.setattr(split, "_apply_speed_scoped_headroom", lambda: (None, {"applied": False}))
    from h3_audio_t8_pkg.modular_sampling import eav
    plan, source, seed = _plan(), _source(), 724
    low = split.prepare_speed_stage(_model(), plan, source, stage_index=0, shift_audio=3.0,
                                    seed=seed, execution_scope="strict_t2va_stock20")
    effect_model, runtime, _ = eav.apply_stage_eav(
        low[0], low[4], low[2], low[10], eav.EAVConfig(mode, 0.1, 0.0, 1.0, 32, 3.0),
    )
    _, live, _ = split.sample_speed_stage(
        effect_model, low[1], low[2], low[3], low[4], old.H3ModalityStableNoise(seed), low[5],
    )
    assert live.verify_live()["portable_identity"] is True
    assert runtime.snapshot()["completed_forwards"] == low[5].nfe
    _, path, digest, _ = frozen.save_speed_stage(live, output_root=tmp_path)
    restored, prior, _ = frozen.load_speed_stage(plan, seed, 3.0, 1, path, digest, output_root=tmp_path)
    high = split.prepare_speed_stage(_model(), plan, source, stage_index=1, shift_audio=3.0,
                                     seed=seed, execution_scope="strict_t2va_stock20", previous_spec=prior)
    live_noise, _ = split.handoff_speed_stage(live, high[5], dct_chunk_size=4)
    restored_noise, _ = split.handoff_speed_stage(restored, high[5], dct_chunk_size=4)
    assert all(torch.equal(a, b) for a, b in zip(live_noise.value.unbind(), restored_noise.value.unbind()))


@pytest.mark.parametrize("with_eav", [False, True])
def test_frozen_speed_stage_accepts_completed_external_relay(monkeypatch, tmp_path, with_eav):
    monkeypatch.setattr(split, "_apply_speed_scoped_headroom", lambda: (None, {"applied": False}))
    _, _, low, model, positive, latent = _relay_case()
    if with_eav:
        from h3_audio_t8_pkg.modular_sampling import eav
        model, runtime, _ = eav.apply_stage_eav(
            model, low[4], latent, low[10], eav.EAVConfig("report_only", 0.1, 0.0, 1.0, 32, 3.0),
        )
    _, live, _ = split.sample_speed_stage(
        model, positive, latent, low[3], low[4], old.H3ModalityStableNoise(123), low[5],
    )
    evidence = live.verify_live()["execution"]["effects"]
    assert evidence["kind"] == ("eav" if with_eav else "relay")
    assert live.verify_live()["portable_identity"] is True
    _, path, digest, _ = frozen.save_speed_stage(live, output_root=tmp_path)
    restored, _, _ = frozen.load_speed_stage(low[5].plan, 123, 3.0, 1, path, digest, output_root=tmp_path)
    assert restored.verify_live()["receipt_sha256"] == live.verify_live()["receipt_sha256"]


def test_frozen_relay_low_supports_independently_edited_high_relay_and_eav(monkeypatch, tmp_path):
    from h3_audio_t8_pkg.modular_sampling import eav, speed_effects
    from h3_audio_t8_pkg.prompt_relay_advanced import build_prompt_relay_plan, prompt_relay_model_contract
    from h3_audio_t8_pkg.long_video_dual_identity import content_identity
    monkeypatch.setattr(split, "_apply_speed_scoped_headroom", lambda: (None, {"applied": False}))
    low_plan, source, low, low_model, low_positive, low_latent = _relay_case()
    _, live_low, _ = split.sample_speed_stage(
        low_model, low_positive, low_latent, low[3], low[4], old.H3ModalityStableNoise(123), low[5],
    )
    _, path, digest, _ = frozen.save_speed_stage(live_low, output_root=tmp_path)
    frozen_low, frozen_spec, _ = frozen.load_speed_stage(
        low[5].plan, 123, 3.0, 1, path, digest, output_root=tmp_path,
    )
    high_plan, *_ = build_prompt_relay_plan(
        global_prompt="A new high-resolution scene instruction",
        local_prompts="The woman turns.\nShe pauses under rain.",
        length=22, timing_mode="auto_equal", time_ranges="", math_profile="paper_v1",
        epsilon=0.1, allow_gaps=False, allow_overlaps=False,
    )
    assert high_plan["plan_hash"] != low_plan["plan_hash"]
    high = split.prepare_speed_stage(
        _model(), low[5].plan, source, stage_index=1, shift_audio=3.0,
        seed=123, execution_scope="strict_t2va_stock20", previous_spec=frozen_spec,
    )
    high_model, high_positive, high_latent, *_ = speed_effects.apply_relay(
        high[0], high[5], source, high_plan,
    )
    binding = prompt_relay_model_contract(high_model)["binding"]
    assert binding["plan_hash"] == high_plan["plan_hash"]
    high_model, runtime, _ = eav.apply_stage_eav(
        high_model, high[4], high_latent, high[10], eav.EAVConfig("report_only", 0.1, 0.0, 1.0, 32, 3.0),
    )
    live_noise, _ = split.handoff_speed_stage(live_low, high[5], dct_chunk_size=4)
    frozen_noise, _ = split.handoff_speed_stage(frozen_low, high[5], dct_chunk_size=4)
    assert all(torch.equal(a, b) for a, b in zip(live_noise.value.unbind(), frozen_noise.value.unbind()))
    _, live_high, _ = split.sample_speed_stage(
        high_model, high_positive, high_latent, high[3], high[4], live_noise, high[5],
    )
    _, resumed_high, _ = split.sample_speed_stage(
        high_model, high_positive, high_latent, high[3], high[4], frozen_noise, high[5],
    )
    assert all(torch.equal(a, b) for a, b in
               zip(live_high.external_output.unbind(), resumed_high.external_output.unbind()))
    assert runtime.snapshot()["status"] == "observed_report_only"
    assert (resumed_high.verify_live()["request"]["model"]["stage_effects"]["relay"]["binding"]
            == content_identity(binding))


def test_frozen_speed_fingerprint_rechecks_corrupted_tensor_file(monkeypatch, tmp_path):
    plan, _, seed, _, _, live = _first(monkeypatch)
    _, path, digest, _ = frozen.save_speed_stage(live, output_root=tmp_path)
    before = frozen.fingerprint_speed_stage(path, output_root=tmp_path)
    state = frozen.stage_root(output_root=tmp_path) / Path(path).parent / frozen.FILENAME
    with state.open("ab") as stream:
        stream.write(b"corrupt")
    assert frozen.fingerprint_speed_stage(path, output_root=tmp_path) != before
    with pytest.raises(ValueError, match="SHA or size mismatch"):
        frozen.load_speed_stage(plan, seed, 3.0, 1, path, digest, output_root=tmp_path)
    with pytest.raises(ValueError, match="escapes|relative|artifact"):
        frozen.load_speed_stage(plan, seed, 3.0, 1, "../other/manifest.json", digest, output_root=tmp_path)


def test_frozen_speed_interrupted_write_never_commits_manifest(monkeypatch, tmp_path):
    _, _, _, _, _, live = _first(monkeypatch)
    def interrupted(*_args, **_kwargs):
        raise RuntimeError("simulated write interruption")
    monkeypatch.setattr(frozen, "save_file", interrupted)
    with pytest.raises(RuntimeError, match="write interruption"):
        frozen.save_speed_stage(live, output_root=tmp_path)
    assert not list(frozen.stage_root(output_root=tmp_path).rglob("manifest.json"))


@pytest.mark.parametrize("with_eav", [False, True])
def test_frozen_speed_stage_new_process_runs_only_next_stage(monkeypatch, tmp_path, with_eav):
    plan, source, seed, low, _, live = _first(monkeypatch)
    if with_eav:
        from h3_audio_t8_pkg.modular_sampling import eav
        effect_model, _, _ = eav.apply_stage_eav(
            low[0], low[4], low[2], low[10], eav.EAVConfig("apply_exp", 0.1, 0.0, 1.0, 32, 3.0),
        )
        _, live, _ = split.sample_speed_stage(
            effect_model, low[1], low[2], low[3], low[4], old.H3ModalityStableNoise(seed), low[5],
        )
    _, path, digest, _ = frozen.save_speed_stage(live, output_root=tmp_path)
    high = split.prepare_speed_stage(
        _model(), plan, source, stage_index=1, shift_audio=3.0,
        seed=seed, execution_scope="strict_t2va_stock20", previous_spec=low[5],
    )
    live_noise, _ = split.handoff_speed_stage(live, high[5], dct_chunk_size=4)
    _, live_high, _ = split.sample_speed_stage(
        high[0], high[1], high[2], high[3], high[4], live_noise, high[5],
    )
    expected = sha(_input_identity(live_high.external_output))
    root = Path(__file__).resolve().parents[1]
    code = """
import json, runpy, sys
from pathlib import Path
import comfy.cli_args
comfy.cli_args.args.cpu = True
comfy.cli_args.args.use_pytorch_cross_attention = True
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'tests'))
runpy.run_path(str(root / 'tests' / 'conftest.py'))
import test_modular_speed_stages as helper
from h3_audio_t8_pkg.modular_sampling import speed_stages as stages, speed_storage as store
from h3_audio_t8_pkg.modular_sampling.results import _input_identity, sha
stages._build_empty_t2va_stage = lambda source, width, height, prior: helper._conditioning(source, width, height, prior)
stages._apply_speed_scoped_headroom = lambda: (None, {'applied': False})
plan, source, seed = helper._plan(), helper._source(), 724
prior_result, prior_spec, report = store.load_speed_stage(plan, seed, 3.0, 1, sys.argv[3], sys.argv[4], output_root=sys.argv[2])
high = stages.prepare_speed_stage(helper._model(), plan, source, stage_index=1, shift_audio=3.0, seed=seed,
                                  execution_scope='strict_t2va_stock20', previous_spec=prior_spec)
noise, _ = stages.handoff_speed_stage(prior_result, high[5], dct_chunk_size=4)
_, result, _ = stages.sample_speed_stage(high[0], high[1], high[2], high[3], high[4], noise, high[5])
print('SPEED_RESUME_RESULT=' + json.dumps({'low_calls': json.loads(report)['sampling_calls'],
    'high_calls': result.verify_live()['execution']['callbacks'],
    'output_sha256': sha(_input_identity(result.external_output))}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root.parents[1])
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    process = subprocess.run(
        [sys.executable, "-c", code, str(root), str(tmp_path), path, digest],
        cwd=root, env=env, capture_output=True, text=True, timeout=90, check=False,
    )
    assert process.returncode == 0, process.stderr[-2500:]
    marker = next(line.split("=", 1)[1] for line in process.stdout.splitlines()
                  if line.startswith("SPEED_RESUME_RESULT="))
    result = json.loads(marker)
    assert result == {"low_calls": 0, "high_calls": list(range(high[5].nfe)),
                      "output_sha256": expected}


def test_frozen_speed_rejects_claimed_effect_coverage_without_calls(monkeypatch, tmp_path):
    from h3_audio_t8_pkg.modular_sampling import eav
    from h3_audio_t8_pkg.modular_sampling.results import canonical
    plan, _, seed, low, _, _ = _first(monkeypatch)
    effect_model, _, _ = eav.apply_stage_eav(
        low[0], low[4], low[2], low[10], eav.EAVConfig("apply_exp", 0.1, 0.0, 1.0, 32, 3.0),
    )
    _, live, _ = split.sample_speed_stage(
        effect_model, low[1], low[2], low[3], low[4], old.H3ModalityStableNoise(seed), low[5],
    )
    receipt = live.verify_live()
    receipt["execution"]["effects"]["selector_calls"] = 0
    receipt["receipt_sha256"] = sha({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    forged = split.SpeedStageResult(live.spec, live.external_output, live.video_version,
                                    live.audio_version, canonical(receipt))
    with pytest.raises(ValueError, match="portable execution/effect evidence"):
        frozen.save_speed_stage(forged, output_root=tmp_path)


def test_three_stage_middle_snapshot_restores_only_final_transition(monkeypatch, tmp_path):
    monkeypatch.setattr(split, "_build_stage", lambda source, width, height: _conditioning(source, width, height))
    monkeypatch.setattr(split, "_build_empty_t2va_stage",
                        lambda source, width, height, prior: _conditioning(source, width, height, prior))
    monkeypatch.setattr(split, "_apply_speed_scoped_headroom", lambda: (None, {"applied": False}))
    plan, source, seed = _plan("0.4,0.7,1.0", "0.94,0.78", 96), _source(), 819
    prior_spec = prior_result = None
    for index in (0, 1):
        prepared = split.prepare_speed_stage(
            _model(), plan, source, stage_index=index, shift_audio=3.0,
            seed=seed, execution_scope="strict_t2va_stock20", previous_spec=prior_spec,
        )
        noise = (old.H3ModalityStableNoise(seed) if index == 0 else
                 split.handoff_speed_stage(prior_result, prepared[5], dct_chunk_size=4)[0])
        _, prior_result, _ = split.sample_speed_stage(
            prepared[0], prepared[1], prepared[2], prepared[3], prepared[4], noise, prepared[5],
        )
        prior_spec = prepared[5]
    assert prior_result.verify_live()["portable_identity"] is True
    _, path, digest, _ = frozen.save_speed_stage(prior_result, output_root=tmp_path)
    restored, template, _ = frozen.load_speed_stage(plan, seed, 3.0, 2, path, digest, output_root=tmp_path)
    final = split.prepare_speed_stage(
        _model(), plan, source, stage_index=2, shift_audio=3.0,
        seed=seed, execution_scope="strict_t2va_stock20", previous_spec=template,
    )
    live_noise, _ = split.handoff_speed_stage(prior_result, final[5], dct_chunk_size=4)
    loaded_noise, _ = split.handoff_speed_stage(restored, final[5], dct_chunk_size=4)
    for actual, expected in zip(loaded_noise.value.unbind(), live_noise.value.unbind()):
        assert torch.equal(actual, expected)
