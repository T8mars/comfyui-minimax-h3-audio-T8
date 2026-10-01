"""Actual Relay-enabled VDN LOW feeding a separate original native HIGH."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import save_stage
from test_modular_vdn_relay import inputs
from test_modular_vdn_native import native_high


@pytest.mark.parametrize("training", ["stage_dmd_8nfe", "stage_b_50nfe"])
@pytest.mark.parametrize("refine", [3, 4, 5])
def test_vdn_relay_eav_low_to_independent_native_relay_eav_high(training, refine):
    args, runtime, eav, _ = inputs(with_eav=True, training=training)
    low = sample_stage(*args)[2]
    assert runtime.snapshot()["status"] == eav.snapshot()["status"] == "observed_apply_exp"
    args, runtime, original = native_high(low.output, refine)
    assert not args[1].model_patcher.additional_models
    expected = SamplerCustomAdvanced.execute(*original).result
    result = sample_stage(*args)[2]
    for a, b in zip(expected, (result.output, result.denoised_output)):
        assert all(torch.equal(x, y) for x, y in zip(a["samples"].unbind(), b["samples"].unbind()))
    assert result.verify()["portable_identity"] and result.verify()["execution"]["denoiser_evaluations"] == refine
    assert runtime.snapshot()["relay_attention_calls"] == refine


@pytest.mark.parametrize("training", ["stage_dmd_8nfe", "stage_b_50nfe"])
def test_new_process_reads_relay_vdn_low_only_runs_clean_native_high(tmp_path, training):
    low = sample_stage(*inputs(with_eav=True, training=training)[0])[2]
    path, digest, _ = save_stage(low, tmp_path)
    args, runtime, _ = native_high(low.output, 4, "relay_eav_apply")
    expected = sample_stage(*args)[2].verify()["outputs"]
    assert runtime.snapshot()["status"] == "observed_apply_exp"
    code = """
import json,runpy,sys
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
runpy.run_path('tests/conftest.py')
from test_modular_vdn_native import native_high
from h3_audio_t8_pkg.modular_sampling.storage import load_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
import torch
first=load_stage(sys.argv[1],sys.argv[2],sys.argv[3],'vdn_complete')[0]
args,runtime,_=native_high(first,4,'relay_eav_apply')
receipt=sample_stage(*args)[2].verify()
assert receipt['execution']['denoiser_evaluations']==4 and receipt['portable_identity']
assert runtime.snapshot()['status']=='observed_apply_exp'
assert not args[1].model_patcher.additional_models and not torch.cuda.is_initialized()
print('RESULT='+json.dumps(receipt['outputs']))
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == expected
