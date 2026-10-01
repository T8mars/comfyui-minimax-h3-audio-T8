"""Real new-process replay of effect-bearing HyperFlow HEAD; no HEAD rerun."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from h3_audio_t8_pkg.modular_sampling import hyperflow as stages, hyperflow_effects as effects, eav
from h3_audio_t8_pkg.modular_sampling import hyperflow_storage as storage
from test_modular_hyperflow import inputs
from test_modular_hyperflow_effects import case
from test_progressive_relay import paired


@pytest.mark.parametrize("split", [1, 4, 7])
@pytest.mark.parametrize("with_relay", [False, True])
def test_new_process_saved_effects_and_independent_tail_config(tmp_path, monkeypatch, split, with_relay):
    _, _, _, _, _, boundary, _, _ = case(monkeypatch, "apply_exp", with_relay, split)
    path, digest, _ = storage.save_boundary(boundary, tmp_path)
    _, high, _, positive = inputs(monkeypatch, distinct=True)
    if with_relay:
        high, positive, _ = paired(high, query_route="video_only_paper")
    positive[0][0].add_(.3)
    bound = effects.bind_tail(boundary, high, positive, [])
    high, _, _ = eav.apply_stage_eav(bound[0], bound[4], bound[3], bound[5], eav.EAVConfig("report_only", .7, .2, 1., 32, 3.))
    expected, _ = stages.sample_tail_result(boundary, high, positive, [], seed=19)
    code = r'''
import json, runpy, sys
runpy.run_path('tools/check_director_d1_cpu.py', run_name='config')
runpy.run_path('tests/conftest.py')
import pytest, torch, comfy.sample
from test_modular_hyperflow import inputs
from test_progressive_relay import paired
from h3_audio_t8_pkg.modular_sampling import hyperflow as stages, hyperflow_effects as effects, eav
from h3_audio_t8_pkg.modular_sampling import hyperflow_storage as storage
from h3_audio_t8_pkg import hyperflow_two_pass_advanced as legacy
def forbidden(*a, **kw):
    raise AssertionError('HEAD/whole split/fresh noise during effect-bearing TAIL-only replay')
with pytest.MonkeyPatch.context() as patch:
    _, high, _, positive = inputs(patch, distinct=True)
    if sys.argv[4] == 'True':
        high, positive, _ = paired(high, query_route='video_only_paper')
    positive[0][0].add_(.3)
    boundary, _ = storage.load_boundary(sys.argv[1], sys.argv[2], sys.argv[3])
    bound = effects.bind_tail(boundary, high, positive, [])
    high, _, _ = eav.apply_stage_eav(bound[0], bound[4], bound[3], bound[5], eav.EAVConfig('report_only', .7, .2, 1., 32, 3.))
    patch.setattr(stages, 'sample_head', forbidden)
    patch.setattr(legacy, 'sample_hyperflow_split', forbidden)
    patch.setattr(comfy.sample, 'prepare_noise', forbidden)
    result, _ = stages.sample_tail_result(boundary, high, positive, [], seed=19)
    path, digest, _ = storage.save_result(result, sys.argv[1])
    patch.setattr(stages, 'sample_tail', forbidden)
    loaded, _ = storage.load_result(sys.argv[1], path, digest)
    assert loaded.receipt_json == result.receipt_json
    assert effects.audit(boundary, 'head')['eav']['config']['mode'] == 'apply_exp'
    audit = effects.audit(loaded, 'tail')
    assert audit['composition_verified'] and audit['eav']['config']['mode'] == 'report_only'
    assert result.verify()['sampling']['execution']['portable_identity'] is True
    assert not torch.cuda.is_initialized()
    print('RESULT=' + json.dumps({'output': stages.snapshot(result.output), 'audit': audit}))
'''
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest, str(with_relay)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=70)
    assert child.returncode == 0, child.stdout + child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == {"output": stages.snapshot(expected.output), "audit": effects.audit(expected, "tail")}
    assert not torch.cuda.is_initialized()
