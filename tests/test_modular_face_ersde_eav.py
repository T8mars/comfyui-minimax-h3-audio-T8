"""Exact-source Core ER-SDE Face EAV and completed-stage identity."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, BasicScheduler, KSamplerSelect, RandomNoise

from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.modular_sampling import native_explicit
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig, apply_stage_eav, audit_stage_eav
from h3_audio_t8_pkg.modular_sampling.face_stage import (
    bind_parity_face_stage, bind_multiface_face_stage, bind_window_face_stage,
)
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import load_stage, save_stage
from test_fast_h3_v2_core_sampler import model
from test_modular_face_multiface import _job as multiface_job
from test_modular_face_parity import _inputs as parity_inputs
from test_modular_face_window import _job as window_job
from test_face_refine_parity_advanced import _locked_av
from test_progressive_sampling_runtime import conditioning


def _bind(variant, denoise):
    if variant == "parity":
        frames, plan, latent, prepared, sampler, sigmas = parity_inputs()
        if denoise != .45:
            sigmas = BasicScheduler.execute(prepared, "simple", 8, denoise).result[0]
        bound = bind_parity_face_stage(plan, frames, prepared, sampler, sigmas, latent)
    else:
        if variant == "multiface":
            parent, source, plan = multiface_job()
            latent = _locked_av()
        else:
            parent, audio, window_plan, source, window_audio, mapping, plan = window_job()
            latent = _locked_av(frame_count=22)
        prepared, _, _ = sampling.setup_dual_clock_sampling(
            model(), latent, 8, 12., 3., "er_sde", "simple")
        sampler = KSamplerSelect.execute("er_sde").result[0]
        sigmas = BasicScheduler.execute(prepared, "simple", 8, denoise).result[0]
        if variant == "multiface":
            bound = bind_multiface_face_stage(plan, source, parent, prepared, sampler, sigmas, latent)
        else:
            bound = bind_window_face_stage(plan, source, parent, window_plan, mapping,
                                           audio, window_audio, prepared, sampler, sigmas, latent)
    return bound, latent


@pytest.mark.parametrize("variant", ("parity", "multiface", "window"))
@pytest.mark.parametrize("denoise", (.45, 1.))
@pytest.mark.parametrize("mode", ("report_only", "apply_exp"))
def test_exact_core_ersde_face_stage_eav_observes_all_forwards(variant, denoise, mode):
    (bound, sampler, sigmas, context, bind_report), latent = _bind(variant, denoise)
    assert json.loads(bind_report)["portable_completion_sampler_adapted"] is True
    assert json.loads(context.profile)["forward_plan"]["known"] is True
    effected, runtime, _ = apply_stage_eav(bound, sigmas, latent, context,
        EAVConfig(mode, tau=.2, start_video_progress=0., end_video_progress=1.,
                  g_hard_limit=3.))
    result = sample_stage(RandomNoise.execute(812).result[0],
                          BasicGuider.execute(effected, conditioning()).result[0],
                          sampler, sigmas, latent, context)[2]
    _, report_json = audit_stage_eav(result.output, runtime)
    report = json.loads(report_json)
    assert report["status"] == ("observed_apply_exp" if mode == "apply_exp" else "observed_report_only")
    assert report["clock_match"] is True
    assert report["completed_forwards"] == context.steps
    assert report["selector_calls"] >= context.steps * runtime.blocks
    assert result.verify()["portable_identity"] is True


def test_changed_core_source_fails_closed_to_unverified_eav(monkeypatch):
    monkeypatch.setattr(native_explicit, "ER_SDE_SOURCE_SHA256", "unknown-core-source")
    (bound, _, _, context, bind_report), _ = _bind("parity", .45)
    assert json.loads(bind_report)["eav_forward_coverage_adapter"] == "none"
    assert json.loads(context.profile)["forward_plan"]["known"] is False
    assert json.loads(bind_report)["portable_completion_sampler_adapted"] is False


def test_core_source_change_after_binding_invalidates_forward_plan(monkeypatch):
    import comfy.k_diffusion.sampling as k_sampling

    (bound, _, _, context, _), _ = _bind("parity", .45)
    assert native_explicit.forward_plan(context)["known"] is True
    monkeypatch.setattr(k_sampling, "sample_er_sde", lambda *args, **kwargs: None)
    assert native_explicit.forward_plan(context)["known"] is False


@pytest.mark.parametrize("variant", ("parity", "multiface", "window"))
def test_exact_source_ersde_face_artifact_loads_and_audits_in_fresh_cpu_process(variant, tmp_path):
    torch.manual_seed(411)
    (bound, sampler, sigmas, context, _), latent = _bind(variant, .45)
    result = sample_stage(RandomNoise.execute(812).result[0],
        BasicGuider.execute(bound, conditioning()).result[0], sampler, sigmas, latent, context)[2]
    receipt = result.verify()
    assert receipt["portable_identity"] and receipt["verified_recipe_completion"]
    path, digest, _ = save_stage(result, tmp_path, f"face-{variant}-er-sde")
    _, _, _, loaded, _ = load_stage(tmp_path, path, digest, "native_high")
    assert loaded.verify()["receipt_sha256"] == receipt["receipt_sha256"]
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(root.parents[1]), str(root)))
    env.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    completed = subprocess.run([sys.executable, str(root / "tests/modular_face_ersde_cold_worker.py"),
                                str(tmp_path), path, digest, variant],
                               text=True, capture_output=True, env=env, timeout=90, check=True)
    assert '"status": "cold_face_source_bound"' in completed.stdout
    assert receipt["receipt_sha256"] in completed.stdout


@pytest.mark.parametrize("mutation", ("core_source", "sampler_options"))
def test_er_sde_unknown_executable_settings_do_not_gain_portable_completion(monkeypatch, mutation, tmp_path):
    (bound, sampler, sigmas, context, _), latent = _bind("parity", .45)
    if mutation == "core_source":
        monkeypatch.setattr(native_explicit, "ER_SDE_SOURCE_SHA256", "unknown-core-source")
    else:
        sampler.extra_options = {"max_stage": 2}
    result = sample_stage(RandomNoise.execute(813).result[0],
        BasicGuider.execute(bound, conditioning()).result[0], sampler, sigmas, latent, context)[2]
    assert result.verify()["portable_identity"] is False
    with pytest.raises(ValueError, match="incomplete or unknown recipe"):
        save_stage(result, tmp_path, "face-unverified")


def test_er_sde_source_guard_is_rechecked_after_the_actual_sampler_return(monkeypatch):
    from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

    (bound, sampler, sigmas, context, _), latent = _bind("parity", .45)
    original = SamplerCustomAdvanced.execute

    def changed_after_return(cls, *args, **kwargs):
        value = original(*args, **kwargs)
        monkeypatch.setattr(native_explicit, "ER_SDE_SOURCE_SHA256", "changed-after-sampling")
        return value

    monkeypatch.setattr(SamplerCustomAdvanced, "execute", classmethod(changed_after_return))
    with pytest.raises(ValueError, match="sampler ownership changed"):
        sample_stage(RandomNoise.execute(814).result[0],
            BasicGuider.execute(bound, conditioning()).result[0], sampler, sigmas, latent, context)
