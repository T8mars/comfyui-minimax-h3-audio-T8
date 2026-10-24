"""Actual tiny native Euler first-segment Relay checkpoint integration, CPU."""

import json
import os
from pathlib import Path
import subprocess
import sys

import comfy.samplers
import pytest
import torch
from safetensors.torch import load_file, save_file

from h3_audio_t8_pkg.progressive_checkpoint import ProgressiveCheckpointSession
from h3_audio_t8_pkg.progressive_relay_checkpoint import ProgressiveRelayCheckpointBinding
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from test_progressive_checkpoint import lora_pair, same
from test_progressive_relay import paired, sample, conditioning, tiny_model
from test_progressive_masking import memory_nodes, installed_sol, kj  # noqa: F401
from test_progressive_sampling_runtime import stub_lifter  # noqa: F401


def models(backend, memory_nodes, installed_sol):  # noqa: F811
    low, high = lora_pair(backend, memory_nodes, installed_sol)
    patches = low.patches
    low = low.clone()
    low.patches = {}
    low, positive, _ = paired(low)
    low.patches = patches
    return low, high, positive


@pytest.mark.parametrize('backend', ['plain', 'kj_memory', 'sol'])
@pytest.mark.parametrize('eav_mode', ['disabled', 'apply_exp'])
def test_first_relay_checkpoint_resume_preserves_complete_av(
        tmp_path, stub_lifter, memory_nodes, installed_sol, backend, eav_mode):  # noqa: F811
    low, high = lora_pair(backend, memory_nodes, installed_sol)
    # Standalone Relay's public contract binds before weight patches. Retain
    # the real LoRAAdapter stack from the fixture and apply it downstream.
    patches = low.patches
    low = low.clone()
    low.patches = {}
    low, positive, _ = paired(low)
    low.patches = patches
    options = dict(model_hires=high, eav_mode=eav_mode, eav_tau=.2)
    baseline, _ = sample(low, positive, **options)

    def cancel(step, *_):
        if step == 5:
            raise InterruptedError('HIGH interrupted')
    with ProgressiveCheckpointSession(tmp_path).exclusive() as checkpoint:
        with pytest.raises(InterruptedError):
            sample(low, positive, **options, callback=cancel, checkpoint=checkpoint)
    callbacks = []
    with ProgressiveCheckpointSession(tmp_path).exclusive() as checkpoint:
        output, text = sample(low, positive, **options,
            callback=lambda step, *_: callbacks.append(step), checkpoint=checkpoint)
    same(output, baseline)
    report = json.loads(text)
    assert callbacks == [4, 5, 6, 7]
    assert report['counts']['actual_forwards'] == {'low': 0, 'high': 4}
    assert report['prompt_relay']['low']['completed_calls'] == {'forward': 0, 'routed_attention': 0}
    assert report['prompt_relay']['high']['completed_calls'] == {'forward': 4, 'routed_attention': 4}


@pytest.mark.parametrize('route', ['video_only_paper', 'joint_av_exp'])
def test_first_relay_route_changes_cache_identity(tmp_path, stub_lifter, route):  # noqa: F811
    # Same embeddings/seed/weights; routing differs and must not share LOW.
    model, positive, _ = paired(tiny_model(), query_route='video_only_paper')
    with ProgressiveCheckpointSession(tmp_path).exclusive() as checkpoint:
        sample(model, positive, checkpoint=checkpoint)
    model, positive, _ = paired(tiny_model(), query_route=route)
    with ProgressiveCheckpointSession(tmp_path).exclusive() as checkpoint:
        _, text = sample(model, positive, checkpoint=checkpoint)
    assert json.loads(text)['checkpoint']['reused_low'] == (route == 'video_only_paper')


@pytest.mark.parametrize('fault', ['paired_text', 'binding', 'weight'])
def test_original_relay_mutation_is_rejected(fault):
    model, positive, _ = paired(tiny_model())
    bound = ProgressiveRelayCheckpointBinding(model, model, positive, conditioning(), comfy.samplers.ksampler('euler'))
    if fault == 'paired_text':
        positive[0][0].add_(.1)
    elif fault == 'binding':
        model.get_attachment(relay.PROMPT_RELAY_WRAPPER_KEY)['binding']['events'][0]['midpoint'] += 1
    else:
        with torch.no_grad():
            next(model.model.parameters()).add_(.1)
    with pytest.raises((ValueError, RuntimeError)):
        bound.verify()


@pytest.mark.parametrize('change', ['chunk', 'event_time', 'high_weights'])
def test_bound_relay_configuration_invalidates_cache(tmp_path, stub_lifter, change):  # noqa: F811
    import copy
    base, high = tiny_model(), tiny_model()
    model, positive, plain = paired(base)
    with ProgressiveCheckpointSession(tmp_path).exclusive() as checkpoint:
        sample(model, positive, model_hires=high, checkpoint=checkpoint)
    binding = copy.deepcopy(relay.prompt_relay_model_contract(model)['binding'])
    if change == 'event_time':
        from h3_audio_t8_pkg.long_video_in_node_loop_advanced import _sha256_json
        binding['events'][0]['midpoint'] += 1
        binding.pop('binding_hash')
        binding['binding_hash'] = _sha256_json(binding)
    if change == 'high_weights':
        with torch.no_grad():
            next(high.model.parameters()).add_(.01)
    model, _ = relay.patch_prompt_relay_model(base, binding, 64 if change == 'chunk' else 32)
    marked = [[embedding, {**meta, relay.PROMPT_RELAY_BINDING_KEY: binding}] for embedding, meta in plain]
    positive = relay._attach_binding_model_cond(marked, binding['binding_hash'])
    with ProgressiveCheckpointSession(tmp_path).exclusive() as checkpoint:
        _, text = sample(model, positive, model_hires=high, checkpoint=checkpoint)
    assert not json.loads(text)['checkpoint']['reused_low']
    assert len(list(tmp_path.glob('low-boundary-*.json'))) == 2


def test_first_relay_change_during_high_refuses_delivery(tmp_path, stub_lifter):  # noqa: F811
    model, positive, _ = paired(tiny_model())
    def change(step, *_):
        if step == 5:
            positive[0][0].add_(.1)
    with ProgressiveCheckpointSession(tmp_path).exclusive() as checkpoint:
        with pytest.raises(ValueError, match='inputs changed'):
            sample(model, positive, callback=change, checkpoint=checkpoint)
    assert len(list(tmp_path.glob('low-boundary-*.json'))) == 1


@pytest.mark.parametrize('backend', ['plain', 'kj_memory', 'sol'])
def test_first_relay_eav_fresh_process_restores_complete_av(
        tmp_path, stub_lifter, memory_nodes, installed_sol, backend):  # noqa: F811
    low, high, positive = models(backend, memory_nodes, installed_sol)
    options = dict(model_hires=high, eav_mode='apply_exp', eav_tau=.2)
    expected, _ = sample(low, positive, **options)
    def cancel(step, *_):
        if step == 5:
            raise InterruptedError('HIGH interrupted')
    cache = tmp_path / 'cache'
    with ProgressiveCheckpointSession(cache).exclusive() as checkpoint:
        with pytest.raises(InterruptedError):
            sample(low, positive, **options, checkpoint=checkpoint, callback=cancel)
    output = str(tmp_path / 'child')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1', 'T8_FIRST_RELAY_WORKER': json.dumps(
        dict(cache=str(cache), output=output, backend=backend))}
    code = ("import runpy;runpy.run_path('tools/check_director_d1_cpu.py',run_name='config');"
            f"import torch;torch.set_num_threads({torch.get_num_threads()});import pytest;"
            "result=pytest.main(['-q','tests/test_progressive_relay_checkpoint.py::test_first_relay_worker','--tb=short']);"
            "assert not torch.cuda.is_initialized();raise SystemExit(result)")
    result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                            env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    tensors = load_file(output + '.safetensors')
    for name, tensor in zip(('video', 'audio'), expected['samples'].unbind(), strict=True):
        torch.testing.assert_close(tensors[name], tensor, rtol=0, atol=0)
    report = json.loads(Path(output + '.json').read_text())
    assert report['counts']['actual_forwards'] == {'low': 0, 'high': 4}
    assert report['eav']['low']['execution_scope'] == 'restored_low_not_current_forwards'


@pytest.mark.skipif('T8_FIRST_RELAY_WORKER' not in os.environ, reason='fresh-process worker entry only')
def test_first_relay_worker(stub_lifter, memory_nodes, installed_sol):  # noqa: F811
    args = json.loads(os.environ['T8_FIRST_RELAY_WORKER'])
    low, high, positive = models(args['backend'], memory_nodes, installed_sol)
    callbacks = []
    with ProgressiveCheckpointSession(args['cache']).exclusive() as checkpoint:
        output, text = sample(low, positive, model_hires=high, eav_mode='apply_exp', eav_tau=.2,
            checkpoint=checkpoint, callback=lambda step, *_: callbacks.append(step))
    assert callbacks == [4, 5, 6, 7]
    video, audio = output['samples'].unbind()
    save_file(dict(video=video, audio=audio), args['output'] + '.safetensors')
    Path(args['output'] + '.json').write_text(text)
