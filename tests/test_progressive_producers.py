"""Native wrapper/real tokenizer identity tests with tiny, non-generating networks."""

import json
from types import MethodType

import comfy.sd
import comfy.model_patcher
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
from comfy.text_encoders.minimax import MiniMaxH3TEModel, MiniMaxH3Tokenizer
import pytest
import torch

from h3_audio_t8_pkg.progressive_producers import NativeProgressiveProducers, native_producer_identity
from test_progressive_sampling_runtime import tiny_model, conditioning, latent, stub_lifter  # noqa: F401
from test_progressive_continuation import accepted, capture  # noqa: F401


def component(role):
    # Native class shells, not pretrained architectures or an encoder forward.
    cls = {'clip': MiniMaxH3TEModel, 'video_vae': MiniMaxH3VideoVAE, 'audio_vae': MiniMaxH3AudioVAE}[role]
    network = cls.__new__(cls)
    torch.nn.Module.__init__(network)
    network.register_parameter('probe', torch.nn.Parameter(torch.arange(4, dtype=torch.float32)))
    network.eval()
    wrapper = (comfy.sd.CLIP if role == 'clip' else comfy.sd.VAE).__new__(
        comfy.sd.CLIP if role == 'clip' else comfy.sd.VAE)
    wrapper.patcher = comfy.model_patcher.ModelPatcher(network, torch.device('cpu'), torch.device('cpu'))
    if role == 'clip':
        wrapper.cond_stage_model = network
        wrapper.tokenizer = MiniMaxH3Tokenizer()
        wrapper.tokenizer_options = {}
        wrapper.layer_idx = None
        wrapper.use_clip_schedule = False
        wrapper.apply_hooks_to_conds = None
    else:
        wrapper.first_stage_model = network
        wrapper.vae_dtype = torch.float32
        wrapper.crop_input = True
        wrapper.size = None
        wrapper.process_input = lambda image: image * 2 - 1
    return wrapper


@pytest.mark.parametrize('role', ['clip', 'video_vae', 'audio_vae'])
def test_native_producer_identity_binds_real_weights_and_roundtrips(role):
    value = component(role)
    first = native_producer_identity(value, role)
    assert first == native_producer_identity(component(role), role)
    assert first == json.loads(json.dumps(first))
    with torch.no_grad():
        next(value.patcher.model.parameters()).add_(.1)
    assert native_producer_identity(value, role) != first


@pytest.mark.parametrize('change', ['dtype', 'crop', 'network_setting', 'process', 'vocabulary', 'options', 'buffer'])
def test_configuration_and_tokenizer_changes_invalidate(change):
    role = 'clip' if change in {'vocabulary', 'options'} else 'video_vae'
    value = component(role)
    first = native_producer_identity(value, role)
    if change == 'dtype':
        value.vae_dtype = torch.float16
    elif change == 'crop':
        value.crop_input = False
    elif change == 'network_setting':
        value.patcher.model.clip_length = 39
    elif change == 'process':
        value.process_input = lambda image: image * 3 - 1
    elif change == 'vocabulary':
        value.tokenizer.qwen3vl_32b.tokenizer.add_tokens(['producer_identity_probe'])
    elif change == 'buffer':
        value.patcher.model.register_buffer('pixel_mean', torch.ones(3), persistent=False)
    else:
        value.tokenizer_options['new_option'] = True
    assert native_producer_identity(value, role) != first


@pytest.mark.parametrize('fault', ['method', 'forward', 'hook', 'patch', 'wrong_role', 'unknown'])
def test_unknown_owners_are_not_silently_hashed(fault):
    value = component('video_vae')
    if fault == 'method':
        value.encode = MethodType(lambda self, x: x, value)
    elif fault == 'forward':
        value.patcher.model.forward = lambda x: x
    elif fault == 'hook':
        value.patcher.model.register_forward_pre_hook(lambda *a: None)
    elif fault == 'patch':
        value.patcher.add_patches({'probe': ('diff', (torch.ones(4),))})
    elif fault == 'unknown':
        value.unknown_execution = object()
    if fault == 'wrong_role':
        with pytest.raises(ValueError):
            native_producer_identity(value, 'audio_vae')
    else:
        from h3_audio_t8_pkg.patch_stack_policy import model_identity_matches
        first = native_producer_identity(value, 'video_vae')
        second = native_producer_identity(value, 'video_vae')
        assert first['portable_cache_reuse'] is False
        assert first['sha256'] != second['sha256']
        assert model_identity_matches(first, second)


def test_bound_producers_revalidate_objects_and_contents():
    values = {role: component(role) for role in ('clip', 'video_vae', 'audio_vae')}
    bound = NativeProgressiveProducers(**values)
    assert bound.verify(**values) == bound.identity
    with pytest.raises(ValueError, match='objects differ'):
        bound.verify(**{**values, 'video_vae': component('video_vae')})
    values['video_vae'].crop_input = False
    with pytest.raises(ValueError, match='producer changed'):
        bound.verify()


def test_same_latents_do_not_reuse_checkpoint_after_producer_change(tmp_path, stub_lifter):  # noqa: F811
    import comfy.samplers
    from h3_audio_t8_pkg.progressive_checkpoint import ProgressiveCheckpointSession
    from h3_audio_t8_pkg.progressive_sampling_runtime import sample_progressive_h3
    from h3_audio_t8_pkg.sampling import native_flow_sigmas
    values = {role: component(role) for role in ('clip', 'video_vae', 'audio_vae')}
    producers = NativeProgressiveProducers(**values)
    model = tiny_model()

    def run(binding, callback=None):
        with ProgressiveCheckpointSession(tmp_path).exclusive() as cache:
            return sample_progressive_h3(model, conditioning(), conditioning(), latent(),
                comfy.samplers.ksampler('euler'), native_flow_sigmas(8, 12.),
                upscaler_model='test', seed=19, low_evaluations=4, checkpoint=cache,
                producers=binding, callback=callback)
    baseline, _ = run(producers)
    callbacks = []
    restored, report = run(producers, lambda i, *_: callbacks.append(i))
    assert callbacks == [4, 5, 6, 7]
    assert json.loads(report)['producers'] == producers.identity
    for a, b in zip(baseline['samples'].unbind(), restored['samples'].unbind(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    values['video_vae'].crop_input = False
    with pytest.raises(ValueError, match='producer changed'):
        run(producers)
    _, report = run(NativeProgressiveProducers(**values))
    assert not json.loads(report)['checkpoint']['reused_low']
    assert len(list(tmp_path.glob('low-boundary-*.json'))) == 2


def test_fabricated_producer_labels_fail_before_sampling(stub_lifter):  # noqa: F811
    import comfy.samplers
    from h3_audio_t8_pkg.progressive_sampling_runtime import sample_progressive_h3
    from h3_audio_t8_pkg.sampling import native_flow_sigmas
    with pytest.raises(ValueError, match='not a supplied identity label'):
        sample_progressive_h3(tiny_model(), conditioning(), conditioning(), latent(),
            comfy.samplers.ksampler('euler'), native_flow_sigmas(8, 12.), upscaler_model='test', seed=19,
            producers={'clip': '0' * 64})
    assert not stub_lifter


def test_live_native_encode_method_change_is_identity_bound(monkeypatch):
    value = component('video_vae')
    first = native_producer_identity(value, 'video_vae')
    monkeypatch.setattr(MiniMaxH3VideoVAE, 'encode', lambda self, x: x)
    assert native_producer_identity(value, 'video_vae') != first


@pytest.mark.parametrize('role,network_type', [('video_vae', MiniMaxH3VideoVAE), ('audio_vae', MiniMaxH3AudioVAE)])
def test_actual_native_vae_configuration_reaches_materialized_weight_gate(role, network_type):
    # Build the complete real architecture/configuration on meta, with no GPU,
    # weight allocation, checkpoint download or encoding. Meta weights MUST
    # remain rejected; this tests compatibility of the preceding config walk.
    with torch.device('meta'):
        network = network_type()
        wrapper = comfy.sd.VAE(sd=network.state_dict(), device=torch.device('meta'), dtype=torch.float32)
    with pytest.raises(ValueError, match='unsupported materializing conversion'):
        native_producer_identity(wrapper, role)


def test_actual_native_clip_configuration_reaches_materialized_weight_gate():
    wrapper = component('clip')
    with torch.device('meta'):
        network = MiniMaxH3TEModel(device=torch.device('meta'), dtype=torch.float32)
    wrapper.cond_stage_model = network
    wrapper.patcher = comfy.model_patcher.ModelPatcher(network, torch.device('cpu'), torch.device('cpu'))
    with pytest.raises(ValueError, match='unsupported materializing conversion'):
        native_producer_identity(wrapper, 'clip')


def test_native_clip_transient_state_is_not_a_changed_producer():
    from comfy.text_encoders.minimax import MiniMaxQwen3VL, MiniMaxH3ClipModel
    wrapper = component('clip')
    clip = MiniMaxH3ClipModel.__new__(MiniMaxH3ClipModel)
    torch.nn.Module.__init__(clip)
    clip.options_default = ('last', None, True)
    clip.execution_device = None
    transformer = MiniMaxQwen3VL.__new__(MiniMaxQwen3VL)
    torch.nn.Module.__init__(transformer)
    clip.transformer = transformer
    wrapper.cond_stage_model.add_module('fixture_clip', clip)
    first = native_producer_identity(wrapper, 'clip')
    clip.execution_device = torch.device('cpu')
    clip.layer, clip.layer_idx, clip.return_projected_pooled = 'last', None, True
    transformer.last_token_tags = torch.ones(9, dtype=torch.long)
    assert native_producer_identity(wrapper, 'clip') == first
    wrapper.layer_idx = -2
    assert native_producer_identity(wrapper, 'clip') != first


@pytest.mark.parametrize('fault', [None, 'pin_alias', 'weight_function', 'lora_function', 'seed_key', 'cast', 'allocation', 'resident_view'])
def test_real_dynamic_patcher_cpu_preparation_preserves_producer_identity(fault):
    import comfy.ops
    import importlib
    import comfy_aimdo.control
    import comfy_aimdo.host_buffer
    if comfy_aimdo.host_buffer.lib is None:
        # Load native host-buffer functions only; do not call init_devices.
        assert comfy_aimdo.control.init()
        importlib.reload(comfy_aimdo.host_buffer)
    # __new__ normally routes CPU to the legacy patcher. Instantiate the exact
    # dynamic class explicitly to exercise its real load/bookkeeping on CPU;
    # _vbar_get handles CPU without allocating a CUDA VBAR.
    wrapper = component('video_vae')
    network = wrapper.first_stage_model
    layer = comfy.ops.manual_cast.Linear(128, 128)
    with torch.no_grad():
        layer.weight.fill_(.1)
        layer.bias.zero_()
    network.add_module('fixture_linear', layer)
    patcher = object.__new__(comfy.model_patcher.ModelPatcherDynamic)
    comfy.model_patcher.ModelPatcherDynamic.__init__(patcher, network, torch.device('cpu'), torch.device('cpu'))
    wrapper.patcher = patcher
    before = native_producer_identity(wrapper, 'video_vae')
    try:
        patcher.load(torch.device('cpu'))
        assert layer._pin_state is network.dynamic_pins[torch.device('cpu')]
        assert native_producer_identity(wrapper, 'video_vae') == before
        if fault is not None:
            if fault == 'pin_alias':
                layer._pin_state = dict(layer._pin_state)
            elif fault == 'weight_function':
                layer.weight_function = [lambda value: value]
            elif fault == 'lora_function':
                layer.weight_lowvram_function = lambda value: value
            elif fault == 'seed_key':
                layer.seed_key = 'another-module'
            elif fault == 'cast':
                layer.comfy_cast_weights = False
            elif fault == 'resident_view':
                layer._v_weight = layer.weight.detach().clone()
            else:
                layer._v = (object(), 0, 64)
            if fault in ('weight_function', 'lora_function'):
                assert native_producer_identity(wrapper, 'video_vae')['portable_cache_reuse'] is False
            else:
                with pytest.raises(ValueError):
                    native_producer_identity(wrapper, 'video_vae')
            return
        with torch.no_grad():
            layer.weight[0, 0] += .1
        assert native_producer_identity(wrapper, 'video_vae') != before
    finally:
        patcher.restore_loaded_backups()
        # CPU has no VBAR; the GPU-only detach path intentionally refuses it.
        network.dynamic_patchers.discard(id(patcher))
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('role', ['clip', 'video_vae', 'audio_vae'])
def test_actual_core_load_does_not_invalidate_unchanged_producer(role):
    import comfy.model_management as mm
    import comfy.ops
    from h3_audio_t8_pkg.long_video_dual_residency import release_stage_residency
    wrapper = component(role)
    # Real Core layers inherit empty weight/bias function lists from the class.
    # The legacy CPU loader materializes those same lists on the instance;
    # a plain probe Parameter alone never exercises that residency transition.
    layer = comfy.ops.manual_cast.Linear(4, 4)
    with torch.no_grad():
        layer.weight.fill_(.1)
        layer.bias.zero_()
    wrapper.patcher.model.add_module('fixture_linear', layer)
    if role == 'clip':
        wrapper.patcher.set_model_compute_dtype(torch.float32)
    first = native_producer_identity(wrapper, role)
    try:
        mm.load_models_gpu([wrapper.patcher], force_full_load=True)
        assert native_producer_identity(wrapper, role) == first
    finally:
        release_stage_residency(wrapper)
    assert native_producer_identity(wrapper, role) == first


@pytest.mark.parametrize('fault', ['swapped', 'changed_during_prepare'])
def test_continuation_rechecks_actual_producers_before_and_after_preparation(accepted, monkeypatch, fault):  # noqa: F811
    import comfy.samplers
    from h3_audio_t8_pkg.progressive_continuation import ProgressiveContinuationSource
    from h3_audio_t8_pkg.progressive_continuation_runtime import sample_progressive_continuation
    from h3_audio_t8_pkg.sampling import native_flow_sigmas
    values = {role: component(role) for role in ('clip', 'video_vae', 'audio_vae')}
    bound = NativeProgressiveProducers(**values)
    prepared = []

    def prepare(*a, **k):
        prepared.append(True)
        values['video_vae'].crop_input = False
        # Must reject the changed producer before consuming these placeholders.
        return None, None, {}
    monkeypatch.setattr(ProgressiveContinuationSource, 'prepare_conditions', prepare)
    selected = values if fault == 'changed_during_prepare' else {**values, 'clip': component('clip')}
    with pytest.raises(ValueError, match='producer objects differ|producer changed'):
        sample_progressive_continuation(capture(accepted), tiny_model(), comfy.samplers.ksampler('euler'),
            native_flow_sigmas(8, 12.), **selected, producers=bound, prompt='test', length=124,
            upscaler_model='unused', seed=19)
    assert bool(prepared) == (fault == 'changed_during_prepare')


@pytest.mark.parametrize('key', ['weight_function', 'bias_function'])
def test_legacy_function_lists_are_not_ignored_when_execution_is_changed(key):
    import comfy.ops
    wrapper = component('video_vae')
    layer = comfy.ops.manual_cast.Linear(4, 4)
    with torch.no_grad():
        layer.weight.fill_(.1)
        layer.bias.zero_()
    wrapper.first_stage_model.add_module('fixture_linear', layer)
    first = native_producer_identity(wrapper, 'video_vae')
    setattr(layer, key, [])
    assert native_producer_identity(wrapper, 'video_vae') == first
    setattr(layer, key, [lambda value: value + .1])
    assert native_producer_identity(wrapper, 'video_vae') != first


@pytest.mark.parametrize('key', ['comfy_cast_weights', 'prev_comfy_cast_weights'])
def test_legacy_cast_alias_is_bound_to_real_class_default_and_selected_policy(key):
    import comfy.model_management as mm
    import comfy.ops
    from h3_audio_t8_pkg.long_video_dual_residency import release_stage_residency
    wrapper = component('clip')
    layer = comfy.ops.manual_cast.Linear(4, 4)
    with torch.no_grad():
        layer.weight.fill_(.1)
        layer.bias.zero_()
    wrapper.patcher.model.add_module('fixture_linear', layer)
    wrapper.patcher.set_model_compute_dtype(torch.float32)
    before = native_producer_identity(wrapper, 'clip')
    try:
        mm.load_models_gpu([wrapper.patcher], force_full_load=True)
        assert layer.comfy_cast_weights is layer.prev_comfy_cast_weights is True
        assert native_producer_identity(wrapper, 'clip') == before
        setattr(layer, key, False)
        assert native_producer_identity(wrapper, 'clip') != before
    finally:
        release_stage_residency(wrapper)
