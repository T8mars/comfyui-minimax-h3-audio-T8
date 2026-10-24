"""Real native CPU boundary persistence/replay; trained models are not used."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType

import comfy.samplers
import pytest
import torch
from safetensors.torch import load_file, save_file

from h3_audio_t8_pkg import progressive_sampling_runtime as runtime
from h3_audio_t8_pkg.progressive_checkpoint import ProgressiveCheckpointSession, native_model_identity
from h3_audio_t8_pkg.progressive_continuation import capture_continuation_source
from h3_audio_t8_pkg.sampling import native_flow_sigmas, setup_dual_clock_sampling
from test_progressive_sampling_runtime import tiny_model, conditioning, latent, stub_lifter  # noqa: F401
from test_progressive_continuation import accepted, capture  # noqa: F401
from test_progressive_continuation_runtime import run
from test_progressive_continuation_relay import arguments
from test_progressive_masking import memory_nodes, installed_sol, kj  # noqa: F401


def sample(model, root=None, *, callback=None, seed=19):
    def execute(checkpoint):
        return runtime.sample_progressive_h3(model, conditioning(), conditioning(), latent(),
            comfy.samplers.ksampler('euler'), native_flow_sigmas(8, 12.),
            upscaler_model='test', seed=seed, low_evaluations=4, callback=callback, checkpoint=checkpoint)
    if root is None:
        return execute(None)
    with ProgressiveCheckpointSession(root).exclusive() as checkpoint:
        return execute(checkpoint)


def same(left, right):
    for a, b in zip(left['samples'].unbind(), right['samples'].unbind(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_low_boundary_reuse_skips_real_low_without_changing_video_or_audio(tmp_path, stub_lifter):  # noqa: F811
    model = tiny_model()
    baseline, _ = sample(model)
    first, first_report = sample(model, tmp_path)
    callbacks = []
    replay, text = sample(model, tmp_path, callback=lambda i, *_: callbacks.append(i))
    same(first, baseline)
    same(replay, baseline)
    report = json.loads(text)
    assert callbacks == [4, 5, 6, 7]
    assert report['counts']['actual_forwards'] == {'low': 0, 'high': 4}
    assert report['checkpoint']['reused_evaluations'] == 4
    assert not json.loads(first_report)['checkpoint']['reused_low']
    record = json.loads(next(tmp_path.glob('low-boundary-*.json')).read_text())
    tensors = load_file(str(tmp_path / record['tensor_file']))
    assert set(tensors) == {'clean_video', 'audio_next'}
    assert tensors['audio_next'].dtype == torch.float32
    assert not torch.equal(tensors['audio_next'], baseline['samples'].unbind()[1])


@pytest.mark.parametrize('change', ['seed', 'weight', 'lora'])
def test_changed_execution_never_reuses_old_boundary(tmp_path, stub_lifter, change):  # noqa: F811
    model = tiny_model()
    sample(model, tmp_path)
    seed = 19
    if change == 'seed':
        seed += 1
    elif change == 'weight':
        with torch.no_grad():
            next(model.model.parameters()).add_(.01)
    else:
        key, weight = next(iter(model.model.named_parameters()))
        model.add_patches({key: ('diff', (torch.full_like(weight, .01),))}, .7)
    _, text = sample(model, tmp_path, seed=seed)
    assert not json.loads(text)['checkpoint']['reused_low']
    assert len(list(tmp_path.glob('low-boundary-*.json'))) == 2


def test_cancel_before_low_completion_does_not_publish_receipt(tmp_path, stub_lifter):  # noqa: F811
    def cancel(step, *_):
        if step == 3:
            raise InterruptedError('LOW not completed')
    with pytest.raises(InterruptedError):
        sample(tiny_model(), tmp_path, callback=cancel)
    assert not list(tmp_path.glob('low-boundary-*.json'))


@pytest.mark.parametrize('fault', ['tensor_bytes', 'missing_audio', 'nonfinite', 'path', 'report'])
def test_corrupt_checkpoint_is_rejected_instead_of_rerunning_low(tmp_path, stub_lifter, fault):  # noqa: F811
    from h3_audio_t8_pkg.progressive_checkpoint import digest
    from h3_audio_t8_pkg.long_video_delivery import _sha256_file
    model = tiny_model()
    sample(model, tmp_path)
    path = next(tmp_path.glob('low-boundary-*.json'))
    record = json.loads(path.read_text())
    tensor_path = tmp_path / record['tensor_file']
    if fault == 'tensor_bytes':
        tensor_path.write_bytes(tensor_path.read_bytes() + b'corrupt')
    elif fault in ('missing_audio', 'nonfinite'):
        tensors = load_file(str(tensor_path))
        if fault == 'missing_audio':
            tensors.pop('audio_next')
        else:
            tensors['audio_next'].flatten()[0] = float('nan')
        save_file(tensors, str(tensor_path))
        record['tensor_sha256'] = _sha256_file(tensor_path)
    elif fault == 'path':
        record['tensor_file'] = '../outside.safetensors'
    else:
        record['low_report']['actual_forwards'] = 3
        record['report_sha256'] = digest(record['low_report'])
    path.write_text(json.dumps(record))
    before = len(stub_lifter)
    with pytest.raises(ValueError):
        sample(model, tmp_path)
    assert len(stub_lifter) == before


def test_session_requires_actual_exclusive_lock(tmp_path):
    first, second = ProgressiveCheckpointSession(tmp_path), ProgressiveCheckpointSession(tmp_path)
    with pytest.raises(RuntimeError, match='locked'):
        first.load_low()
    with first.exclusive():
        with pytest.raises(RuntimeError):
            with second.exclusive():
                pytest.fail('Two live owners obtained the same lock')
    with second.exclusive():
        assert second.active


def test_native_sampling_object_patch_is_bound_without_weakening_old_identity():
    from h3_audio_t8_pkg.long_video_dual_identity import stage_model_identity
    model, sampler, _ = setup_dual_clock_sampling(tiny_model(), latent(), 8, 12., 3., 'euler', 'native_flow')
    before = dict(model.object_patches)
    first = native_model_identity(model, sampler)
    assert model.object_patches == before and 'model_sampling' in before
    generic = stage_model_identity(model)
    assert generic['portable_cache_reuse'] is False
    assert model.object_patches == before
    model.get_model_object('model_sampling').set_noise_scale(.9)
    assert native_model_identity(model, sampler) != first


@pytest.mark.parametrize('fault', ['sampling_method', 'sampling_state', 'latent_method'])
def test_unknown_coordinate_mutations_cannot_get_cache_identity(fault):
    model, sampler = tiny_model(), comfy.samplers.ksampler('euler')
    if fault == 'sampling_method':
        sampling = model.get_model_object('model_sampling')
        sampling.timestep = MethodType(lambda self, x: x * 1000, sampling)
    elif fault == 'sampling_state':
        model.get_model_object('model_sampling').future_behavior = .2
    else:
        latent_format = model.get_model_object('latent_format')
        latent_format.process_in = MethodType(lambda self, x: x, latent_format)
    with pytest.raises(ValueError):
        native_model_identity(model, sampler)


def test_changed_model_during_high_is_rejected_without_relabeling_low(tmp_path, stub_lifter):  # noqa: F811
    model = tiny_model()

    def change(step, *_):
        if step == 5:
            key, weight = next(iter(model.model.named_parameters()))
            model.add_patches({key: ('diff', (torch.full_like(weight, .02),))}, .8)
    with pytest.raises(ValueError, match='execution inputs changed'):
        sample(model, tmp_path, callback=change)
    assert len(list(tmp_path.glob('low-boundary-*.json'))) == 1


def lora_pair(backend, memory_nodes, installed_sol):  # noqa: F811
    from comfy.weight_adapter.lora import LoRAAdapter
    low, high = tiny_model(), tiny_model()
    if backend == 'kj_memory':
        lowmem, sage, _ = memory_nodes

        def patch(model):
            model = sage.MiniMaxH3MemoryEfficientSageAttentionPatch.execute(model).result[0]
            model = lowmem.MiniMaxLowVRAMAttention.execute(model, 2).result[0]
            return lowmem.MiniMaxChunkFeedForward.execute(model, 2, 256).result[0]
    elif backend == 'sol':
        def patch(model):
            return installed_sol.SolAttentionPatch().patch(model, True, .5, min_tokens=256)[0]
    else:
        assert backend == 'plain'
        def patch(model):
            return model
    low, high = patch(low), patch(high)
    for model in (low, high):
        identity = native_model_identity(model, comfy.samplers.ksampler('euler'))
        assert identity == json.loads(json.dumps(identity))
    for model in (low, high):
        key, weight = next((k, w) for k, w in model.model.named_parameters() if w.ndim == 2)
        for amount in (.1, .2):
            adapter = LoRAAdapter(set(), (torch.full((weight.shape[0], 2), amount),
                torch.full((2, weight.shape[1]), .1), 2., None, None, None))
            assert model.add_patches({key: adapter}, .7)
    return low, high


@pytest.mark.parametrize('backend', ['kj_memory', 'sol'])
def test_relay_eav_checkpoint_reuses_actual_lora_pair_with_backend(
        accepted, stub_lifter, memory_nodes, installed_sol, tmp_path, backend):  # noqa: F811
    low, high = lora_pair(backend, memory_nodes, installed_sol)
    source = capture(accepted)
    baseline, _ = run(source, low, model_hires=high, **arguments())
    cache = tmp_path / 'cache'
    with ProgressiveCheckpointSession(cache).exclusive() as checkpoint:
        first, _ = run(source, low, model_hires=high, **arguments(), checkpoint=checkpoint)
    with ProgressiveCheckpointSession(cache).exclusive() as checkpoint:
        second, text = run(source, low, model_hires=high, **arguments(), checkpoint=checkpoint)
    same(first, baseline)
    same(second, baseline)
    report = json.loads(text)
    assert report['checkpoint']['reused_low']
    assert report['prompt_relay']['low']['completed_calls'] == {'forward': 0, 'routed_attention': 0}
    assert report['prompt_relay']['high']['completed_calls'] == {'forward': 4, 'routed_attention': 4}
    assert report['stage_models']['low']['weight_patch_entries'] == 2
    assert report['stage_models']['high']['weight_patch_entries'] == 2


@pytest.mark.parametrize('cancel_step', [None, 1, 5])
def test_memory_stage_restores_live_objects_before_identity_or_reuse(
        accepted, stub_lifter, memory_nodes, cancel_step):  # noqa: F811
    import comfy.utils
    lowmem, sage, _ = memory_nodes
    models = []
    for _ in range(2):
        model = sage.MiniMaxH3MemoryEfficientSageAttentionPatch.execute(tiny_model()).result[0]
        model = lowmem.MiniMaxLowVRAMAttention.execute(model, 2).result[0]
        models.append(lowmem.MiniMaxChunkFeedForward.execute(model, 2, 256).result[0])
    snapshots = [{path: comfy.utils.get_attr(model.model, path) for path in model.object_patches}
                 for model in models]

    def cancel(step, *_):
        if step == cancel_step:
            raise InterruptedError('cancel memory stage')

    def execute():
        return run(capture(accepted), models[0], model_hires=models[1], **arguments(), callback=cancel)
    if cancel_step is None:
        execute()
    else:
        with pytest.raises(InterruptedError):
            execute()
    for model, snapshot in zip(models, snapshots, strict=True):
        assert not model.object_patches_backup
        for path, original in snapshot.items():
            assert comfy.utils.get_attr(model.model, path) == original, path
        native_model_identity(model, comfy.samplers.ksampler('euler'))


def test_stage_cleanup_refuses_foreign_live_patch_without_mutation():
    model = tiny_model()
    block = model.model.diffusion_model.blocks[0]
    path = 'diffusion_model.blocks.0.forward'
    original = block.forward
    own = MethodType(lambda self, *a, **k: original(*a, **k), block)
    foreign = MethodType(lambda self, *a, **k: original(*a, **k), block)
    model.add_object_patch(path, own)
    model.patch_model(load_weights=False)
    block.forward = foreign
    with pytest.raises(RuntimeError, match='owned by another MODEL'):
        runtime._restore_stage_objects(model)
    assert block.forward is foreign
    assert model.object_patches_backup[path] == original
    block.forward = own
    runtime._restore_stage_objects(model)
    assert block.forward == original


@pytest.mark.parametrize('backend', ['plain', 'kj_memory', 'sol'])
def test_object_cleanup_keeps_previous_sampling_math_bitwise(
        accepted, stub_lifter, memory_nodes, installed_sol, backend, monkeypatch):  # noqa: F811
    source = capture(accepted)
    old_low, old_high = lora_pair(backend, memory_nodes, installed_sol)
    staged = []
    restore = runtime._restore_stage_objects
    try:
        with monkeypatch.context() as context:
            # Exactly the old lifecycle: native sampling leaves object patches
            # resident. Keep strong references solely to restore the fixture.
            context.setattr(runtime, '_restore_stage_objects', lambda model: staged.append(model))
            previous, _ = run(source, old_low, model_hires=old_high, **arguments())
    finally:
        for model in staged:
            restore(model)
    low, high = lora_pair(backend, memory_nodes, installed_sol)
    current, _ = run(source, low, model_hires=high, **arguments())
    same(previous, current)


@pytest.mark.parametrize('backend', ['plain', 'kj_memory', 'sol'])
def test_continuation_relay_eav_fresh_process_resumes_noisy_audio_boundary(
        accepted, stub_lifter, tmp_path, memory_nodes, installed_sol, backend):  # noqa: F811
    source = capture(accepted)
    low, high = lora_pair(backend, memory_nodes, installed_sol)
    baseline, _ = run(source, low, model_hires=high, **arguments())
    cache = tmp_path / 'cache'

    def cancel(step, *_):
        if step == 5:
            raise InterruptedError('interrupted HIGH')
    with pytest.raises(InterruptedError):
        with ProgressiveCheckpointSession(cache).exclusive() as checkpoint:
            run(source, low, model_hires=high, **arguments(), checkpoint=checkpoint, callback=cancel)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1', 'T8_PROGRESSIVE_CHECKPOINT_CHILD': json.dumps(
        {'root': str(accepted.root), 'request': accepted.request, 'cache': str(cache),
         'output': str(tmp_path / 'child'), 'backend': backend})}
    code = ("import runpy;runpy.run_path('tools/check_director_d1_cpu.py',run_name='config');"
            f"import torch;torch.set_num_threads({torch.get_num_threads()});import pytest;"
            "result=pytest.main(['-q','tests/test_progressive_checkpoint.py::test_fresh_process_worker','--tb=short']);"
            "assert not torch.cuda.is_initialized();raise SystemExit(result)")
    result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).parents[1],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    tensors = load_file(str(tmp_path / 'child.safetensors'))
    for key, expected in zip(('video', 'audio'), baseline['samples'].unbind(), strict=True):
        torch.testing.assert_close(tensors[key], expected, rtol=0, atol=0)
    report = json.loads((tmp_path / 'child.json').read_text())
    assert report['counts']['actual_forwards'] == {'low': 0, 'high': 4}
    assert report['eav']['low']['execution_scope'] == 'restored_low_not_current_forwards'


@pytest.mark.skipif('T8_PROGRESSIVE_CHECKPOINT_CHILD' not in os.environ, reason='isolated fresh-process worker only')
def test_fresh_process_worker(stub_lifter, memory_nodes, installed_sol):  # noqa: F811
    args = json.loads(os.environ['T8_PROGRESSIVE_CHECKPOINT_CHILD'])
    source = capture_continuation_source(args['root'], **args['request'])
    callbacks = []
    low, high = lora_pair(args['backend'], memory_nodes, installed_sol)
    with ProgressiveCheckpointSession(args['cache']).exclusive() as checkpoint:
        output, text = run(source, low, model_hires=high, **arguments(), checkpoint=checkpoint,
                           callback=lambda i, *_: callbacks.append(i))
    assert callbacks == [4, 5, 6, 7]
    video, audio = output['samples'].unbind()
    save_file({'video': video, 'audio': audio}, args['output'] + '.safetensors')
    Path(args['output'] + '.json').write_text(text)
