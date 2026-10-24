"""Actual author CPU LoRA merge and existing-projection contract, no FP8 kernel claim."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess

import pytest
import torch


SCRIPT = r'''
import hashlib, json, os, sys
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
repo, source_root, vdn_root, output, guarded = sys.argv[1:]
os.environ['FREEVIDEO_VDN_ROOT'] = vdn_root
sys.path[:0] = [repo, source_root, str(Path(vdn_root)/'diffusers/src'), vdn_root]
import torch
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
from safetensors.torch import save_file, load_file
from h3_t8.freevideo_h21.source import Source, digest
from h3_t8.freevideo_h21.assets import offline_tables, modulation_strategy, prepare_loras
from h3_t8.freevideo_h21.plans import plan, clock
from freevideo_engine import adaln_assets, lora_cache
source = Source(source_root)
selected=plan(source,'max',256,256,39)
root = Path(output)
cache = root/'cache'
(cache/'adaln').mkdir(parents=True)
target = 'transformer_blocks.0.adaln_proj.linear'
original = torch.arange(72, dtype=torch.float32).reshape(36,2)/100
save_file({'time_embedder.linear_1.weight': torch.eye(2)}, str(cache/'root.safetensors'))
save_file({target+'.weight': original}, str(cache/'adaln/00.safetensors'))
def row(group):
    path=cache/(group+'.safetensors')
    return dict(group=group,file=group+'.safetensors',bytes=path.stat().st_size,sha256=digest(path))
root_row, projection_row = row('root'), row('adaln/00')
manifest = dict(format='freevideo-fp8-slim-v1',precision='fp8',source_id='actual-small-projection-fixture',
    groups=[root_row], linears={}, scale_granularity='rowwise',
    adaln_sources=dict(groups=[projection_row]))
(cache/'manifest.json').write_text(json.dumps(manifest),encoding='utf8')
before = {str(p):digest(p) for p in cache.rglob('*') if p.is_file()}
adapter = root/'adapter.safetensors'
a=torch.tensor([[.25,-.5]])
b=torch.arange(36,dtype=torch.float32)[:,None]/100
save_file({target+'.lora_A.weight':a,target+'.lora_B.weight':b},str(adapter))
selection=dict(path=str(adapter),bytes=adapter.stat().st_size,sha256=digest(adapter),strength=-.75)
from contextlib import nullcontext
scope=offline_tables(source,[]) if guarded=='yes' else nullcontext()
with scope:
    derived, report=prepare_loras(cache,[selection],dict(inference_kernels=False, head_chunk=0),root/'owned-variants')
assert report['selected_preparation']=='author_fused' and report['explicit_engine_options_preserved']
actual_manifest=json.loads((derived/'manifest.json').read_text(encoding='utf8'))
assert 'format' not in actual_manifest and not actual_manifest.get('adaln_tables')
actual=load_file(str(derived/'adaln/00.safetensors'))[target+'.weight']
expected=original+(b@a)*-.75
assert torch.equal(actual,expected)
# Run an actual linear operation using the actually merged parameter tensor.
layer=torch.nn.Linear(2,36,bias=False)
layer.weight=torch.nn.Parameter(actual,requires_grad=False)
x=torch.tensor([[.5,-.25]])
assert torch.equal(layer(x),torch.nn.functional.linear(x,expected))
assert adaln_assets.weight_identity(actual_manifest)!=adaln_assets.weight_identity(manifest)
# The actual weight identity really changed. It must still retain the merged
# parameter and use the real dynamic projection API, not be banned for no table.
dynamic, dynamic_manifest, options, tables, decision=modulation_strategy(source,derived,actual_manifest,{},
    selected,dict(adaln_cache=False,adaln_disk_cache=False),dict(hidden_size=2),
    sampling_root=root,home=root/'owned-home')
assert dynamic==derived and dynamic_manifest==actual_manifest
assert not options['adaln_cache'] and not tables and not decision['optimization_enabled']
assert decision['mode']=='actual_dynamic_projections'
assert torch.equal(load_file(str(dynamic/'adaln/00.safetensors'))[target+'.weight'],expected)
# An existing complete slim source can likewise expose a verified full-storage
# view without changing its weights, inventing constants, or touching originals.
view, view_manifest, view_options, tables, decision=modulation_strategy(source,cache,manifest,{},
    selected,dict(adaln_cache=False),dict(hidden_size=2),
    sampling_root=root,home=root/'owned-home')
assert 'format' not in view_manifest and 'adaln_tables' not in view_manifest
assert not view_options['adaln_cache'] and not tables
assert torch.equal(load_file(str(view/'adaln/00.safetensors'))[target+'.weight'],original)
assert os.path.samefile(view/'adaln/00.safetensors',cache/'adaln/00.safetensors')
# Published optional artifacts are not installed assets. A normal absence can
# use actual projections; a missing asset claimed by the model manifest cannot.
identity=adaln_assets.identity(adaln_assets.weight_identity(actual_manifest),
    clock(source,selected,'SINGLE')['modulation_timesteps'],2)
missing=dict(identity=identity,files=[dict(index=0,file='uninstalled/00.safetensors',bytes=128,sha256='0'*64)])
dynamic,_,options,tables,decision=modulation_strategy(source,derived,actual_manifest,
    dict(optional_adaln=dict(tables=[missing])),selected,dict(adaln_cache=True),dict(hidden_size=2),
    sampling_root=root,home=root/'owned-home')
assert dynamic==derived and not options['adaln_cache'] and not tables
assert 'has not been installed' in decision['reason']
try:
    modulation_strategy(source,derived,dict(actual_manifest,adaln_tables=[missing]),{},selected,
        dict(adaln_cache=True),dict(hidden_size=2),sampling_root=root,home=root/'owned-home')
except ValueError as error:
    assert 'Installed exact v4 modulation table is missing' in str(error)
else:
    raise AssertionError('Model-declared missing installed constant was silently bypassed')
assert all(digest(path)==sha for path,sha in before.items())
assert not torch.cuda.is_initialized()
print(json.dumps(dict(actual_CPU_merge=True,actual_CPU_forward=True,original_bytes_unchanged=True,
    derived_has_real_projections=True,derived_is_full_format=True,CUDA_initialized=False,
    FP8_kernel_tested=False,trained_model_tested=False,guarded=guarded,report=report)))
'''


@pytest.mark.parametrize("guarded", ["no", "yes"])
def test_actual_existing_projection_LoRA_merge_is_not_banned_as_implicit_download(tmp_path, guarded):
    values = [os.environ.get(name) for name in ("T8_H21_FREEVIDEO_PYTHON", "T8_H21_FREEVIDEO_SOURCE", "T8_H21_FREEVIDEO_VDN")]
    if not all(values):
        pytest.skip("Select the existing external VDN environment; no implicit install/download")
    python, source_root, vdn_root = values
    repo = str(Path(__file__).resolve().parents[1])
    command = [python, "-I", "-B", "-X", "utf8", "-c", SCRIPT, repo, source_root, vdn_root, str(tmp_path), guarded]
    result = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                            capture_output=True, text=True, encoding="utf8", timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"original_bytes_unchanged": true' in result.stdout
    assert not torch.cuda.is_initialized()
