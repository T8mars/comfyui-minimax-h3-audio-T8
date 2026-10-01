"""Actual VDN LOW plus a distinct clean native HIGH; not native8 impersonating VDN."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import comfy.nested_tensor
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

from h3_audio_t8_pkg import learned_latent_upscale_advanced as learned, vdn_h3_advanced as vdn
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import save_stage
from test_modular_vdn_stages import effect_args, inputs, source
from test_modular_native_explicit import inputs as native_inputs
from test_progressive_sampling_runtime import conditioning, latent as high_template


def native_high(first, refine, effects="relay_eav"):
    # Explicit synthetic 4x4 -> 4x8 spatial handoff, not a learned weight claim.
    video, audio = first["samples"].unbind()
    enlarged = {**first, "samples": comfy.nested_tensor.NestedTensor((video.repeat_interleave(2, -1), audio))}
    high, _, _ = learned.reconcile_two_pass_h3_latent(enlarged, high_template(), conditioning(), "first_pass",
        second_pass_audio_source="first_pass", second_pass_audio_strength=0.)
    return native_inputs(("lbh", 4, refine), "native_high", effects, source=high)


@pytest.mark.parametrize("training", ["stage_dmd_8nfe", "stage_b_50nfe"])
@pytest.mark.parametrize("refine", [3, 4, 5])
def test_real_vdn_then_independent_original_native_high_parity_and_relay(training, refine):
    low_args, _, _ = inputs(training)
    low = sample_stage(*low_args)[2]
    assert low.verify()["execution"]["denoiser_evaluations"] == vdn.STAGES[training]["steps"]
    args, runtime, original = native_high(low.output, refine)
    high_model = args[1].model_patcher
    assert not high_model.additional_models and high_model.get_attachment(vdn.ATTACHMENT_KEY) is None
    assert not high_model.get_wrappers("diffusion_model", vdn.WRAPPER_KEY)
    expected = SamplerCustomAdvanced.execute(*original).result
    actual = sample_stage(*args)[2]
    for wanted, value in zip(expected, (actual.output, actual.denoised_output)):
        assert all(torch.equal(x, y) for x, y in zip(wanted["samples"].unbind(), value["samples"].unbind()))
    assert actual.verify()["portable_identity"] and actual.verify()["execution"]["denoiser_evaluations"] == refine
    report = runtime.snapshot()
    assert report["status"] == "observed_report_only" and report["relay_attention_calls"] == refine
    audited, _ = learned.audit_two_pass_h3_audio(args[4], actual.output, 0., True, 1e-5)
    assert torch.equal(audited["samples"].unbind()[1], low.output["samples"].unbind()[1])


@pytest.mark.parametrize("training", ["stage_dmd_8nfe", "stage_b_50nfe"])
@pytest.mark.parametrize("refine", [3, 4, 5])
def test_new_process_load_vdn_complete_only_runs_native_high_relay_eav(tmp_path, training, refine):
    args, _ = effect_args(training, "vdn_complete", source())
    low = sample_stage(*args)[2]
    path, digest, _ = save_stage(low, tmp_path)
    args, runtime, _ = native_high(low.output, refine, "relay_eav_apply")
    result = sample_stage(*args)[2]
    assert runtime.snapshot()["status"] == "observed_apply_exp"
    expected = result.verify()["outputs"]
    code = """
import json,runpy,sys
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
runpy.run_path('tests/conftest.py')
from test_modular_vdn_native import native_high
from h3_audio_t8_pkg.modular_sampling.storage import load_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
import torch
first=load_stage(sys.argv[1],sys.argv[2],sys.argv[3],'vdn_complete')[0]
args,runtime,_=native_high(first,int(sys.argv[4]),'relay_eav_apply')
receipt=sample_stage(*args)[2].verify()
assert receipt['execution']['denoiser_evaluations']==int(sys.argv[4]) and receipt['portable_identity']
assert runtime.snapshot()['status']=='observed_apply_exp'
assert not args[1].model_patcher.additional_models and not torch.cuda.is_initialized()
print('RESULT='+json.dumps(receipt['outputs']))
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest, str(refine)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == expected
