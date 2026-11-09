"""Actual tiny Core sampler reuse, dependency invalidation and delivery retry."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from h3_audio_t8_pkg.modular_sampling import reuse, results, storage
from test_modular_results import inputs


def run(root, args=None, mode="reuse_completed"):
    values = reuse.sample_or_reuse(*(inputs() if args is None else args), root, mode)
    return values, json.loads(values[3])


def test_actual_completed_hit_skips_sampler_and_restores_after_decode_failure(tmp_path, monkeypatch):
    first, report = run(tmp_path)
    assert report["status"] == "miss_saved" and report["current_denoiser_evaluations"] == 4
    # A downstream delivery failure must not erase successfully sampled state.
    def failed_decoder():
        raise RuntimeError("decoder OOM")
    with pytest.raises(RuntimeError):
        failed_decoder()
    monkeypatch.setattr(results, "sample_stage", lambda *a: pytest.fail("a verified hit must not sample"))
    second, report = run(tmp_path)
    assert report["status"] == "verified_hit" and report["current_denoiser_evaluations"] == 0
    for a, b in zip(first[0]["samples"].unbind(), second[0]["samples"].unbind()):
        assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()
    # Editing a returned tensor cannot corrupt the on-disk completed state.
    second[0]["samples"].unbind()[0].zero_()
    third, _ = run(tmp_path)
    assert results._input_identity(first[0]) == results._input_identity(third[0])


@pytest.mark.parametrize("changed", ["seed", "text", "source", "reference", "weights"])
def test_content_changes_invalidate_only_the_affected_request(tmp_path, changed):
    _, old = run(tmp_path)
    args = list(inputs())
    if changed == "seed":
        args[0].seed += 1
    elif changed == "text":
        args[1].original_conds["positive"][0]["cross_attn"].add_(.01)
    elif changed == "source":
        args[4]["samples"].unbind()[0].add_(.01)
    elif changed == "reference":
        # Native condition metadata is content-addressed even with the same name.
        args[1].original_conds["positive"][0]["reference_content"] = torch.ones(2, 2)
    else:
        with torch.no_grad():
            next(args[1].model_patcher.model.diffusion_model.parameters()).add_(.01)
    _, current = run(tmp_path, args)
    assert current["status"] == "miss_saved"
    assert current["request_sha256"] != old["request_sha256"]
    _, unaffected = run(tmp_path)
    assert unaffected["status"] == "verified_hit"


def test_unknown_sampler_delegates_even_if_a_known_request_exists(tmp_path):
    run(tmp_path)
    args = list(inputs())
    calls = []
    def foreign(denoiser, x, sigmas, **kwargs):
        calls.append(True)
        return x
    args[2].sampler_function = foreign
    _, report = run(tmp_path, args)
    assert calls == [True] and report["status"] == "uncertified_executed"
    assert report["artifact_path"] is None


def test_corrupt_tensor_is_not_reused_and_explicit_refresh_keeps_history(tmp_path):
    _, report = run(tmp_path)
    original = tmp_path / report["artifact_path"]
    state = original.parent / "state.safetensors"
    with state.open("r+b") as handle:
        handle.seek(-1, 2)
        old = handle.read(1)[0]
        handle.seek(-1, 2)
        handle.write(bytes([old ^ 1]))
    with pytest.raises(ValueError, match="SHA mismatch"):
        run(tmp_path)
    _, fresh = run(tmp_path, mode="refresh")
    assert fresh["status"] == "refreshed" and fresh["artifact_path"] != report["artifact_path"]
    assert original.exists() and run(tmp_path)[1]["status"] == "verified_hit"


def test_interrupted_save_does_not_publish_cache_index(tmp_path, monkeypatch):
    def fail(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(storage, "save_file", fail)
    with pytest.raises(OSError, match="disk full"):
        run(tmp_path)
    assert not list(tmp_path.rglob("entry.json"))
    assert not list(tmp_path.rglob("manifest.json"))


def test_disabled_does_not_create_store(tmp_path):
    root = tmp_path / "absent"
    _, report = run(root, mode="disabled")
    assert report["status"] == "disabled_executed" and not root.exists()


def test_prepared_lookup_hint_cannot_authorize_other_live_inputs(tmp_path, monkeypatch):
    args = list(inputs())
    hint = reuse.request_identity(*args)
    run(tmp_path, args)
    args[0].seed += 1
    monkeypatch.setattr(results, "sample_stage", lambda *a: pytest.fail("wrong hint must never sample or return"))
    with pytest.raises(ValueError, match="does not match"):
        reuse.sample_or_reuse(*args, tmp_path, "reuse_completed", _prepared_identity=hint)


def _write_adapter(path, seed):
    from safetensors.torch import save_file
    from test_fast_h3_v2_core_sampler import model
    key = 'diffusion_model.blocks.0.attn.qkv_proj.weight'
    rows, columns = model().model_state_dict()[key].shape
    rng = torch.Generator().manual_seed(seed)
    prefix = 'blocks.0.attn.qkv_proj'
    save_file({prefix+'.lora_down.weight': torch.randn(2, columns, generator=rng) * .1,
               prefix+'.lora_up.weight': torch.randn(rows, 2, generator=rng) * .1}, str(path))


def _lora_inputs(paths, strengths):
    from h3_audio_t8_pkg.h3_lora_compat_advanced import load_minimax_h3_lora_model
    from h3_audio_t8_pkg.modular_sampling.fast_h3_v2 import build_stage
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
    from test_fast_h3_v2_core_sampler import model
    from test_progressive_sampling_runtime import conditioning, latent
    selected = model()
    for path, strength in zip(paths, strengths, strict=True):
        selected, raw = load_minimax_h3_lora_model(selected, path, strength)
        assert json.loads(raw)['applied_patch_count'] == 1
    source = latent()
    selected, sampler, sigmas, context, _ = build_stage(selected, source, 'low_0_4', 'dense_compat_exp')
    guider = BasicGuider.execute(selected, conditioning()).result[0]
    return RandomNoise.execute(31).result[0], guider, sampler, sigmas, source, context


@pytest.mark.parametrize('change', ['order', 'strength', 'same_file_contents'])
def test_actual_multiple_loaded_loras_invalidate_by_order_strength_and_contents(tmp_path, change):
    first, second = tmp_path/'first.safetensors', tmp_path/'second.safetensors'
    _write_adapter(first, 71)
    _write_adapter(second, 72)
    store = tmp_path/'store'
    output, previous = run(store, _lora_inputs([first, second], [.4, .7]))
    assert previous['status'] == 'miss_saved'
    _, unchanged = run(store, _lora_inputs([first, second], [.4, .7]))
    assert unchanged['status'] == 'verified_hit'
    paths, strengths = [first, second], [.4, .7]
    if change == 'order':
        paths, strengths = paths[::-1], strengths[::-1]
    elif change == 'strength':
        strengths = [.8, .7]
    else:
        _write_adapter(first, 73)  # Same filename, byte length and target; different actual tensors.
    changed, current = run(store, _lora_inputs(paths, strengths))
    assert current['status'] == 'miss_saved' and current['request_sha256'] != previous['request_sha256']
    assert current['current_denoiser_evaluations'] == 4
    if change != 'order':
        assert not torch.equal(output[0]['samples'].unbind()[0], changed[0]['samples'].unbind()[0])
    assert (store/previous['artifact_path']).exists()  # Keep accepted/history candidates.


def test_changed_parent_take_invalidates_only_actual_dependent_stage(tmp_path):
    from h3_audio_t8_pkg.modular_sampling.fast_h3_v2 import build_stage
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
    from test_fast_h3_v2_core_sampler import model
    from test_progressive_sampling_runtime import conditioning

    def successor(parent):
        selected, sampler, sigmas, context, _ = build_stage(model(), parent, 'high_4_8', 'dense_compat_exp')
        return RandomNoise.execute(53).result[0], BasicGuider.execute(selected, conditioning()).result[0], sampler, sigmas, parent, context

    a, a_report = run(tmp_path)
    _, b_report = run(tmp_path, successor(a[1]))
    control = list(inputs())
    control[0].seed = 99
    _, control_report = run(tmp_path, control)
    new_parent_inputs = list(inputs())
    new_parent_inputs[0].seed = 44
    new_parent, _ = run(tmp_path, new_parent_inputs)
    _, new_successor = run(tmp_path, successor(new_parent[1]))
    assert new_successor['status'] == 'miss_saved'
    assert new_successor['request_sha256'] != b_report['request_sha256']
    assert run(tmp_path, successor(a[1]))[1]['status'] == 'verified_hit'
    assert run(tmp_path, control)[1]['request_sha256'] == control_report['request_sha256']
    assert run(tmp_path, control)[1]['status'] == 'verified_hit'
    assert (tmp_path/a_report['artifact_path']).exists()


def test_new_process_rebuilds_identical_request_and_reuses_without_sampling(tmp_path):
    import comfy.cli_args
    first, report = run(tmp_path)
    core = Path(comfy.cli_args.__file__).resolve().parents[1]
    root = Path(__file__).resolve().parents[1]
    code = r'''
import json,os,runpy,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path[:0]=[sys.argv[1],sys.argv[2]]
import comfy.cli_args
comfy.cli_args.args.cpu=True
import torch
torch.set_num_threads(2)
def forbid(): raise RuntimeError('CPU child forbids CUDA initialization')
torch.cuda._lazy_init=forbid
import comfy_kitchen
comfy_kitchen.int8_attention_is_available=lambda *a,**k:False
runpy.run_path('tests/conftest.py')
from h3_audio_t8_pkg.modular_sampling import results,reuse
from test_modular_results import inputs
def cannot_sample(*a,**k): raise AssertionError('A new-process verified hit must not sample')
results.sample_stage=cannot_sample
out=reuse.sample_or_reuse(*inputs(),sys.argv[3],'reuse_completed')
print('RESULT='+json.dumps(dict(report=json.loads(out[3]),output=results._input_identity(out[0]),cuda=torch.cuda.is_initialized())))
'''
    child = subprocess.run([sys.executable, '-c', code, str(core), str(root), str(tmp_path)],
                           cwd=root, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                           capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert child.returncode == 0, child.stderr
    result = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith('RESULT=')))
    assert result['report']['status'] == 'verified_hit'
    assert result['report']['request_sha256'] == report['request_sha256']
    assert result['output'] == results._input_identity(first[0])
    assert result['cuda'] is False


def test_public_node_executes_then_rechecks_persisted_completion(tmp_path, monkeypatch):
    import math
    from h3_audio_t8_pkg import nodes_stage_reuse as public
    monkeypatch.setattr(public, '_stage_store_root', lambda: tmp_path)
    first = public.MiniMaxH3StageReuseEXPT8.execute(*inputs(), mode='reuse_completed').result
    second = public.MiniMaxH3StageReuseEXPT8.execute(*inputs(), mode='reuse_completed').result
    assert json.loads(first[3])['status'] == 'miss_saved'
    assert json.loads(second[3])['status'] == 'verified_hit'
    assert results._input_identity(first[0]) == results._input_identity(second[0])
    assert math.isnan(public.MiniMaxH3StageReuseEXPT8.fingerprint_inputs(mode='reuse_completed'))


def test_partial_index_cannot_redirect_write_outside_store(tmp_path):
    root = tmp_path/'store'
    args = inputs()
    key = results.sha(reuse.request_identity(*args))
    index_dir = root/'reuse-index'/key
    index_dir.mkdir(parents=True)
    protected = tmp_path/'protected.txt'
    protected.write_text('keep original', encoding='utf-8')
    try:
        (index_dir/'entry.partial').symlink_to(protected)
    except OSError as error:
        pytest.skip(f'This OS account cannot create symlinks: {error}')
    with pytest.raises(ValueError, match='link|junction'):
        run(root, args)
    assert protected.read_text(encoding='utf-8') == 'keep original'
    assert not (index_dir/'entry.json').exists()
