"""NEW RES history checks only; no old12-suite or pretrained GPU rerun."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
import comfy.k_diffusion.sampling as core

from h3_audio_t8_pkg import res_history_exp as res
from h3_audio_t8_pkg.sampling import native_flow_sigmas

CONTRACT = {"model": "analytic_nonlinear_not_pretrained", "seed": 42,
            "condition": {"spoken_once": "你好，今天天气真好。"},
            "clock": {"video": 12., "audio": 3.}}


class Analytic:
    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.inner_model = SimpleNamespace(model_patcher=SimpleNamespace(
            get_model_object=lambda name: SimpleNamespace(noise_scale=1.)))

    def __call__(self, values, sigma, **options):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("injected fifth-call interruption")
        sigma = sigma.reshape((-1,) + (1,) * (values.ndim - 1))
        return .2 * values + .03 * values.square() + .3 * sigma.sin()


def fixture(dtype=torch.float32):
    sigmas = native_flow_sigmas(8, 12.).to(dtype)
    noise = torch.tensor([[[.2, -.1, .4, -.3]]], dtype=dtype)
    latent = noise * .25
    mask = torch.tensor([[[1., .5, 0., 1.]]], dtype=dtype)
    return sigmas, noise, latent, mask


def checkpoint(root, *, fail=None, step=4, name="state.h3res.safetensors"):
    sigmas, noise, latent, mask = fixture()
    saved = []

    def post(state):
        if state.completed_steps == step:
            saved.append(res.save_checkpoint(root, name, state, sigmas,
                original_noise=noise, original_latent_image=latent,
                denoise_mask=mask, run_contract=CONTRACT, hash_chunk_bytes=128))

    result = res.sample_res_history(Analytic(fail), noise.clone(), sigmas,
        extra_args={"seed": 42}, post_step=post, disable=True)
    return result, saved[0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_new_loop_matches_actual_Core_every_pre_and_post_boundary(dtype):
    sigmas, noise, _, _ = fixture(dtype)
    old_callbacks, new_callbacks, boundaries = [], [], []
    full_model, new_model = Analytic(), Analytic()
    full = core.sample_res_multistep(full_model, noise.clone(), sigmas,
        disable=True, callback=lambda v: old_callbacks.append({k: v[k].clone()
            if isinstance(v[k], torch.Tensor) else v[k] for k in ("i", "x", "denoised")}))
    actual = res.sample_res_history(new_model, noise.clone(), sigmas,
        disable=True, callback=lambda v: new_callbacks.append({k: v[k].clone()
            if isinstance(v[k], torch.Tensor) else v[k] for k in ("i", "x", "denoised")}),
        post_step=lambda s: boundaries.append(res.RESHistory(s.completed_steps,
            s.state_x.clone(), s.old_denoised.clone(), s.old_sigma_down.clone())))
    assert full_model.calls == new_model.calls == 8
    assert torch.equal(full, actual) and torch.equal(noise, fixture(dtype)[1])
    for original, new in zip(old_callbacks, new_callbacks, strict=True):
        assert original["i"] == new["i"]
        assert torch.equal(original["x"], new["x"])
        assert torch.equal(original["denoised"], new["denoised"])
    assert [s.completed_steps for s in boundaries] == list(range(1, 9))
    assert torch.equal(boundaries[-1].state_x, full)
    # This fixture actually distinguishes post-update x from callback x/x0.
    assert not torch.equal(boundaries[3].state_x, old_callbacks[3]["x"])
    assert torch.equal(boundaries[3].old_denoised, old_callbacks[3]["denoised"])
    assert torch.equal(boundaries[3].old_sigma_down, sigmas[4])
    assert not torch.cuda.is_initialized()


def test_interruption_keeps_step4_and_fresh_CPU_process_finishes_only_remaining4(tmp_path):
    sigmas, noise, latent, mask = fixture()
    with pytest.raises(RuntimeError, match="fifth-call interruption"):
        checkpoint(tmp_path, fail=5)
    saved = res.read_checkpoint(tmp_path, "state.h3res.safetensors", expected_contract=CONTRACT,
                                hash_chunk_bytes=128)
    before = saved["file_sha256"]
    assert saved["history"].completed_steps == 4
    assert torch.equal(saved["tensors"]["original_noise"], noise)
    assert torch.equal(saved["tensors"]["original_latent_image"], latent)
    assert torch.equal(saved["tensors"]["denoise_mask"], mask)
    code = r'''
import importlib.util,json,os,pathlib,sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
root=pathlib.Path(sys.argv[1]); checkpoint_root=pathlib.Path(sys.argv[2])
sys.path[:0]=[str(root.parents[1]),str(root/'tests')]
sys.argv=['res-history-fresh-worker','--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch
torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('h3_audio_t8_pkg',root/'__init__.py',submodule_search_locations=[str(root)])
package=importlib.util.module_from_spec(spec);sys.modules[spec.name]=package;spec.loader.exec_module(package)
from h3_audio_t8_pkg import res_history_exp as res
from test_res_history_exp import Analytic,CONTRACT
from safetensors.torch import save_file
data=res.read_checkpoint(checkpoint_root,'state.h3res.safetensors',expected_contract=CONTRACT,hash_chunk_bytes=128)
model=Analytic(); history=data['history']; full=data['tensors']['full_sigmas']
result=res.sample_res_history(model,torch.zeros_like(history.state_x),full[4:],full_sigmas=full,
                             resume=history,disable=True)
assert model.calls==4 and not torch.cuda.is_initialized()
save_file({'actual':result},str(checkpoint_root/'fresh-result.safetensors'))
print(json.dumps({'calls':model.calls,'CUDA_initialized':torch.cuda.is_initialized(),'completed_before_resume':history.completed_steps}))
'''
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    worker = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", code,
        str(Path(__file__).resolve().parents[1]), str(tmp_path)], env=env,
        capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert worker.returncode == 0, worker.stdout + worker.stderr
    assert '"calls": 4' in worker.stdout and '"CUDA_initialized": false' in worker.stdout
    expected = core.sample_res_multistep(Analytic(), noise.clone(), sigmas, disable=True)
    assert torch.equal(load_file(str(tmp_path / "fresh-result.safetensors"))["actual"], expected)
    assert res.read_checkpoint(tmp_path, "state.h3res.safetensors")["file_sha256"] == before
    assert not torch.cuda.is_initialized()


def test_create_only_atomic_boundary_and_failed_publish_never_overwrites(tmp_path, monkeypatch):
    _, saved = checkpoint(tmp_path)
    before = saved["file_sha256"]
    with pytest.raises(FileExistsError, match="already exists"):
        checkpoint(tmp_path)
    assert res.read_checkpoint(tmp_path, "state.h3res.safetensors")["file_sha256"] == before
    def fail_link(source, target):
        raise OSError("filesystem does not support atomic hard-link publish")
    monkeypatch.setattr(res.os, "link", fail_link)
    with pytest.raises(OSError, match="atomic hard-link"):
        checkpoint(tmp_path, name="new.h3res.safetensors")
    assert not (tmp_path / "new.h3res.safetensors").exists()
    assert not list(tmp_path.glob(".res-*.tmp"))
    assert res.read_checkpoint(tmp_path, "state.h3res.safetensors")["file_sha256"] == before


def test_file_digest_contract_source_and_history_are_checked(tmp_path):
    _, saved = checkpoint(tmp_path)
    changed = copy.deepcopy(CONTRACT)
    changed["condition"]["spoken_once"] = "改变对白"
    with pytest.raises(ValueError, match="contract changed"):
        res.read_checkpoint(tmp_path, "state.h3res.safetensors", expected_contract=changed)
    original = saved["payload"]
    for name in ("old_denoised", "state_x", "full_sigmas", "old_sigma_down", "denoise_mask"):
        tensors = {k: v.clone() for k, v in saved["tensors"].items()}
        tensors[name].reshape(-1)[0] += .01
        path = tmp_path / f"{name}.h3res.safetensors"
        save_file(tensors, str(path), metadata={res.METADATA_KEY: json.dumps(original)})
        with pytest.raises(ValueError):
            res.read_checkpoint(tmp_path, path.name)
    bad = copy.deepcopy(original)
    bad["source_contract"]["res_multistep"] = "0" * 64
    save_file(saved["tensors"], str(tmp_path / "source.h3res.safetensors"),
              metadata={res.METADATA_KEY: json.dumps(bad)})
    with pytest.raises(ValueError, match="source changed"):
        res.read_checkpoint(tmp_path, "source.h3res.safetensors")
    with safe_open(str(saved["path"]), framework="pt", device="cpu") as handle:
        assert (handle.metadata() or {}).get(res.METADATA_KEY)
    assert res.read_checkpoint(tmp_path, "state.h3res.safetensors")["file_sha256"] == saved["file_sha256"]


def test_wrong_paths_Euler_record_and_file_budget_rejected_before_tensor_load(tmp_path, monkeypatch):
    checkpoint(tmp_path)
    for path in ("../escape.h3res.safetensors", "C:/escape.h3res.safetensors",
                 "CON.h3res.safetensors", "wrong.h3nfe.safetensors"):
        with pytest.raises(ValueError):
            res.resolve_path(tmp_path, path)
    save_file({"state_x": torch.ones(1)}, str(tmp_path / "euler.h3res.safetensors"),
              metadata={res.METADATA_KEY: json.dumps({"schema": "t8.minimax_h3.nfe_resume.v1"})})
    with pytest.raises(ValueError, match="deterministic RES"):
        res.read_checkpoint(tmp_path, "euler.h3res.safetensors")
    monkeypatch.setattr(res, "MAX_FILE_BYTES", 8)
    with pytest.raises(ValueError, match="file budget"):
        res.read_checkpoint(tmp_path, "state.h3res.safetensors")


def test_global_remaining_table_history_dtype_and_terminal_boundary_are_not_rebased(tmp_path):
    _, saved = checkpoint(tmp_path)
    full = saved["tensors"]["full_sigmas"]
    history = saved["history"]
    wrong = full[4:].clone()
    wrong[0] -= .01
    with pytest.raises(ValueError, match="remaining schedule"):
        res.sample_res_history(Analytic(), history.state_x, wrong, full_sigmas=full,
                               resume=history, disable=True)
    with pytest.raises(ValueError, match="shape/dtype"):
        res.sample_res_history(Analytic(), history.state_x.double(), full[4:],
                               full_sigmas=full, resume=history, disable=True)
    wrong_history = res.RESHistory(4, history.state_x, history.old_denoised, full[3])
    with pytest.raises(ValueError, match="global sigma boundary"):
        res.sample_res_history(Analytic(), history.state_x, full[4:], full_sigmas=full,
                               resume=wrong_history, disable=True)
    _, final = checkpoint(tmp_path, step=8, name="final.h3res.safetensors")
    with pytest.raises(ValueError, match="no remaining steps"):
        res.sample_res_history(Analytic(), final["history"].state_x, full[8:],
                               full_sigmas=full, resume=final["history"], disable=True)


def test_callback_interruption_is_not_committed_as_an_unfinished_update(tmp_path):
    full, noise, latent, mask = fixture()
    def callback(record):
        if record["i"] == 4:
            raise RuntimeError("interrupt before fifth update")
    def post(state):
        if state.completed_steps == 4:
            res.save_checkpoint(tmp_path, "safe.h3res.safetensors", state, full,
                original_noise=noise, original_latent_image=latent, denoise_mask=mask,
                run_contract=CONTRACT)
    with pytest.raises(RuntimeError, match="fifth update"):
        res.sample_res_history(Analytic(), noise.clone(), full,
                               callback=callback, post_step=post, disable=True)
    saved = res.read_checkpoint(tmp_path, "safe.h3res.safetensors")
    model = Analytic()
    actual = res.sample_res_history(model, noise.clone(), full[4:], full_sigmas=full,
                                   resume=saved["history"], disable=True)
    assert model.calls == 4
    assert torch.equal(actual, core.sample_res_multistep(Analytic(), noise.clone(), full, disable=True))


def test_loaded_Core_code_change_is_rejected_even_if_source_text_stays_pinned(monkeypatch):
    original = core.to_d.__code__
    constants = list(original.co_consts)
    constants[0] = "changed live constant while the source file is unchanged"
    monkeypatch.setattr(core.to_d, "__code__", original.replace(co_consts=tuple(constants)))
    full, noise, _, _ = fixture()
    selected = Analytic()
    with pytest.raises(ValueError, match="executable differs"):
        res.sample_res_history(selected, noise, full, disable=True)
    assert selected.calls == 0
