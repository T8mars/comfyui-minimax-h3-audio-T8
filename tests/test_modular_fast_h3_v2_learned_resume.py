"""Frozen V2 LOW -> real learned handoff code -> HIGH, with a CPU lifter double.

The model loader/3D network are doubled, not the upscaler geometry, AV
reconciliation, V2 samplers or frozen-stage protocol. This is not pretrained
GPU/image/audio qualification or whole-graph PromptExecutor execution.
"""

import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import comfy.cli_args
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise

from h3_audio_t8_pkg import learned_latent_upscale_advanced as learned
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2 import build_stage
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2_handoff import completed_low_x0
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import save_stage
from test_fast_h3_v2_core_sampler import model
from test_learned_latent_upscale_advanced import _patch_model_manager, _ResizeNetwork
from test_modular_results import inputs
from test_progressive_sampling_runtime import conditioning


def _high_after_learned(low_result, monkeypatch):
    low_x0, low_report = completed_low_x0(low_result)
    _patch_model_manager(monkeypatch, _ResizeNetwork())
    enlarged, width, height, upscale_report = learned.learned_upscale_h3_av_latent(
        low_x0, "cpu-double.safetensors", "target_dimensions", 2., 1.,
        256, 128, "honor_dimensions_exp", 1.05, "fp32", "offload_after")
    low_video, low_audio = low_x0["samples"].unbind()
    high_video, high_audio = enlarged["samples"].unbind()
    assert (width, height) == (256, 128)
    assert tuple(high_video.shape[-2:]) == (8, 16)
    assert tuple(low_video.shape[-2:]) == (4, 8)
    assert high_audio is low_audio
    assert json.loads(upscale_report)["audio_preserved"] is True
    template = {"samples": NestedTensor((torch.zeros_like(high_video), torch.zeros_like(high_audio)))}
    high_input, positive, reconcile_report = learned.reconcile_two_pass_h3_latent(
        enlarged, template, conditioning(), "auto", "legacy_policy", 0.)
    assert high_input["samples"].unbind()[1] is low_audio
    assert json.loads(reconcile_report)["audio_source"] == "first_pass"
    prepared, sampler, sigmas, context, _ = build_stage(
        model(), high_input, "high_4_8", "dense_compat_exp")
    guider = BasicGuider.execute(prepared, positive).result[0]
    result = sample_stage(RandomNoise.execute(17).result[0], guider,
                          sampler, sigmas, high_input, context)[2]
    receipt = result.verify()
    assert receipt["verified_recipe_completion"] is True
    assert receipt["execution"]["denoiser_evaluations"] == 4
    return {"low_request_sha256": json.loads(low_report)["request_sha256"],
            "high_outputs": receipt["outputs"],
            "high_request_sha256": receipt["request_sha256"]}


def test_frozen_low_only_high_in_fresh_process_matches_uninterrupted_learned_path(tmp_path):
    low = sample_stage(*inputs("low_0_4"))[2]
    path, digest, _ = save_stage(low, tmp_path)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with pytest.MonkeyPatch.context() as patch:
            uninterrupted = _high_after_learned(low, patch)
    finally:
        torch.set_num_threads(previous_threads)

    code = """
import json,runpy,sys
import comfy.cli_args
comfy.cli_args.args.use_pytorch_cross_attention = sys.argv[4] == '1'
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
runpy.run_path('tests/conftest.py')
import pytest,torch
from h3_audio_t8_pkg.modular_sampling.storage import load_stage
from test_modular_fast_h3_v2_learned_resume import _high_after_learned
torch.set_num_threads(1)
frozen = load_stage(sys.argv[1],sys.argv[2],sys.argv[3],'low_0_4')[3]
with pytest.MonkeyPatch.context() as patch:
    report = _high_after_learned(frozen, patch)
assert not torch.cuda.is_initialized()
print('RESULT='+json.dumps(report,sort_keys=True))
"""
    root = Path(__file__).resolve().parents[1]
    attention_backend = str(int(bool(comfy.cli_args.args.use_pytorch_cross_attention)))
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest,
                            attention_backend],
                           cwd=root, capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stderr
    resumed = json.loads(next(line[7:] for line in child.stdout.splitlines()
                              if line.startswith("RESULT=")))
    assert resumed == uninterrupted
    assert not torch.cuda.is_initialized()
