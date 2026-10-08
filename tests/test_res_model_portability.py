"""Two new raw-INT8/fresh-process identity checks; no prior tests repeated."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from h3_audio_t8_pkg import res_history_exp as res
from h3_audio_t8_pkg.res_model_identity import loaded_model_identity
from test_res_model_identity import model, source_latent, prepared, run


def test_actual_INT8_ConvRot_original_storage_survives_real_Core_LoRA_materialization(tmp_path):
    from test_fast_h3_v2_real_lora_identity import _branch
    quantized, layer = _branch(.25)
    bare = model()
    key = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    bare.model.diffusion_model.blocks[0].attn.qkv_proj = layer
    assert bare.add_patches({key: quantized.patches[key][0][1]}, .25) == [key]
    selected = prepared(bare, source_latent(), tmp_path, mode="disabled")[0]
    before = loaded_model_identity(selected)
    raw, scale = layer.weight._qdata.clone(), layer.weight.params.scale.clone()
    bare.patch_weight_to_device(key, torch.device("cpu"))
    assert not torch.equal(layer.weight._qdata, raw) or not torch.equal(layer.weight.params.scale, scale)
    after = loaded_model_identity(selected)
    assert before == after and after["portable_cache_reuse"]
    assert key in bare.backup
    assert torch.equal(bare.backup[key].weight._qdata, raw)
    assert torch.equal(bare.backup[key].weight.params.scale, scale)
    assert not torch.cuda.is_initialized()


def test_new_process_reads_real_RES_file_with_automatic_weights_and_remaining4(tmp_path):
    bare, source = model(), source_latent(masked=True)
    checkpoint = prepared(bare, source, tmp_path, mode="checkpoint", confirm_checkpoint_write=True)
    expected = run(*checkpoint[:3], source)
    boundary = res.read_checkpoint(tmp_path / "boundaries", "native.h3res.safetensors")
    program = r'''
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); work=Path(sys.argv[2])
sys.path[:0]=[str(root.parents[1]),str(root),str(root/'tests')]
sys.argv=['fresh-res-identity','--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch
torch.set_num_threads(2)
import conftest
from test_res_model_identity import model,source_latent,prepared,run
bare=model(); source=source_latent(masked=True)
selected=prepared(bare,source,work,mode='resume')
result=run(*selected[:3],source)
def digest(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
record={'result':[[digest(t) for t in slot['samples'].unbind()] for slot in result],
 'remaining_schedule_steps':len(selected[2])-1,'CUDA_initialized':torch.cuda.is_initialized()}
assert not record['CUDA_initialized']
with (work/'fresh-result.json').open('x',encoding='utf8') as f: json.dump(record,f)
'''
    root = Path(__file__).resolve().parents[1]
    child = subprocess.run([sys.executable, "-B", "-c", program, str(root), str(tmp_path)],
        text=True, capture_output=True, timeout=120)
    assert child.returncode == 0, child.stdout + child.stderr
    actual = json.loads((tmp_path / "fresh-result.json").read_bytes())

    def digest(tensor):
        return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    assert actual["result"] == [[digest(t) for t in slot["samples"].unbind()] for slot in expected]
    assert actual["remaining_schedule_steps"] == 4 and actual["CUDA_initialized"] is False
    assert res.read_checkpoint(tmp_path / "boundaries", "native.h3res.safetensors")["file_sha256"] == boundary["file_sha256"]
    assert not torch.cuda.is_initialized()
