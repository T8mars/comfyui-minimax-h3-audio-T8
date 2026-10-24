"""Portable real-Core CPU VBAR transport regression; no trained files or GPU.

Regression for Core VBAR transport state. This uses the public
implementation and original tiny-model fixtures, not a private source override.
"""
import ctypes
from types import MethodType

import comfy.memory_management as memory
import comfy.ops as ops
import pytest
import torch

from h3_audio_t8_pkg import reference_step_cache as primitive
from h3_audio_t8_pkg import reference_step_cache_advanced as feature
from test_h21_reference_step_cache import positive, sample, two_layers


@pytest.fixture(autouse=True)
def CPU_only_transport(monkeypatch):
    from comfy.ldm.minimax import model as native
    from comfy.ldm.modules import attention
    monkeypatch.setattr(native, 'optimized_attention', attention.attention_pytorch)
    with torch.no_grad():
        yield
    assert not torch.cuda.is_initialized()


def resolve_with_real_Core(layer):
    """Call original Core resolver on real tiny packed CPU parameter storage.

    A ctypes signature models the documented fault return, not a real CUDA
    VBAR allocation. Core, not the fixture, performs every `_v_*` transition.
    """
    geometries = memory.tensors_to_geometries([layer.weight, layer.bias])
    packed = torch.zeros(memory.vram_aligned_size(geometries), dtype=torch.uint8)
    views = memory.interpret_gathered_like(geometries, packed)
    for source, view in zip((layer.weight, layer.bias), views, strict=True):
        if source is not None:
            view.copy_(source)
    layer._prefetch = dict(signature=(ctypes.c_uint32 * 3)(1, 2, 3), resident=False,
                           needs_cast=False, cast_geometry=geometries, xfer_dest=packed)
    return ops.resolve_cast_module_with_vbar(
        layer, torch.float32, torch.device('cpu'), torch.float32, torch.float32, False)


@pytest.mark.parametrize('bias', [False, True])
def test_real_Core_transport_migration_preserves_structure_parameters_and_math(bias):
    layer = ops.manual_cast.Linear(3, 2, bias=bias, device='cpu', dtype=torch.float32)
    layer.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10)
    if layer.bias is not None:
        layer.bias.copy_(torch.tensor([.2, -.1]))
    # The initial state is the state produced by Core ModelPatcher.set_dirty.
    layer._v_signature = None
    assert '_v_weight' not in vars(layer) and '_v_bias' not in vars(layer)
    before = primitive.module_contract(layer)
    versions, digest = primitive.parameter_versions(layer), primitive.weight_digest(layer)
    weight, returned_bias = resolve_with_real_Core(layer)
    assert primitive.module_contract(layer) == before
    assert primitive.parameter_versions(layer) == versions
    assert primitive.weight_digest(layer) == digest
    assert isinstance(layer._v_signature, ctypes.Array)
    assert isinstance(layer._v_weight, torch.Tensor)
    assert (layer._v_bias is None) == (not bias)
    assert all(key not in primitive.module_contract(layer)[0][4]
               for key in ('_v_weight', '_v_bias', '_v_signature'))
    x = torch.tensor([[1., 2., 3.], [-1., .5, 2.]])
    assert torch.equal(torch.nn.functional.linear(x, weight, returned_bias), layer(x))


def test_actual_Core_four_evaluations_survive_original_resolver_writes():
    bare = two_layers()
    layer = bare.model.diffusion_model.final_layer.video_out
    layer._v_signature = None
    selected, paired, _ = feature.configure_reference_step_cache(bare, positive(), 'approximate', 2)
    calls = []

    def transport(*_):
        calls.append(1)
        resolve_with_real_Core(layer)

    handle = layer.register_forward_hook(transport)
    try:
        output = sample(selected, paired)
    finally:
        handle.remove()
    report = feature.audit_reference_step_cache(selected)
    assert len(calls) == report['completed_model_evaluations'] == 4
    assert report['approximate_block_hits'] == 4
    assert report['failed_model_evaluations'] == 0
    assert report['cleanup_observed'] and not report['request_active']
    assert report['cached_bytes_after_request'] == 0 and not report['portable_cache_identity']
    assert all(torch.isfinite(value).all() for value in output['samples'].unbind())


@pytest.mark.parametrize('change', ['eps', 'forward', 'code', 'module', 'new_private_None'])
def test_true_structure_or_executable_change_is_not_transport(change, monkeypatch):
    bare = two_layers()
    selected, paired, _ = feature.configure_reference_step_cache(bare, positive(), 'approximate', 2)

    class ChangedComputation:
        def forward(self, *args, **kwargs):
            return args[0] + (1 if __class__ is not None else 2)

    def mutate(*_):
        module = bare.model.diffusion_model.blocks[0].norm1
        if change == 'eps':
            module.eps += .001
        elif change == 'forward':
            previous = module.forward

            def foreign(self, value):
                return previous(value)

            module.forward = MethodType(foreign, module)
        elif change == 'code':
            live = module.forward.__func__
            old_id, old_code = id(live), live.__code__
            assert live.__code__.co_freevars == ChangedComputation.forward.__code__.co_freevars
            monkeypatch.setattr(live, '__code__', ChangedComputation.forward.__code__)
            assert id(live) == old_id and live.__code__ is not old_code
        elif change == 'module':
            bare.model.diffusion_model.blocks[0].norm1 = torch.nn.RMSNorm(24, eps=module.eps)
        else:
            module._not_Core_transport = None

    handle = bare.model.diffusion_model.final_layer.register_forward_hook(mutate)
    try:
        with pytest.raises(ValueError, match='executable module ownership/structure changed'):
            sample(selected, paired)
    finally:
        handle.remove()
    report = feature.audit_reference_step_cache(selected)
    assert report['failed_model_evaluations'] == 1
    assert report['cleanup_observed'] and report['cached_bytes_after_request'] == 0


@pytest.mark.parametrize('change', ['version', 'data_bytes'])
def test_actual_selected_parameter_version_and_final_bytes_still_rejected(change):
    bare = two_layers()
    selected, paired, _ = feature.configure_reference_step_cache(bare, positive(), 'approximate', 2)
    calls = []

    def mutate(*_):
        calls.append(1)
        if change == 'version':
            bare.model.diffusion_model.final_layer.video_out.weight.add_(.01)
        elif len(calls) == 4:
            bare.model.diffusion_model.final_layer.video_out.weight.data.add_(.01)

    handle = bare.model.diffusion_model.final_layer.register_forward_hook(mutate)
    try:
        with pytest.raises(ValueError, match='weight version changed' if change == 'version' else 'weight content SHA changed'):
            sample(selected, paired)
    finally:
        handle.remove()
    report = feature.audit_reference_step_cache(selected)
    assert report['failed_model_evaluations'] == 1
    assert report['cleanup_observed'] and report['cached_bytes_after_request'] == 0


def test_foreign_forward_or_changed_Core_resolver_does_not_authenticate_transport(monkeypatch):
    layer = ops.manual_cast.Linear(3, 2, bias=False, device='cpu', dtype=torch.float32)
    layer._v_signature = None
    previous = layer.forward

    def foreign(*args, **kwargs):
        return previous(*args, **kwargs)

    layer.forward = foreign
    assert primitive.module_contract(layer)[0][4]['_v_signature'] is None
    del layer.forward
    assert '_v_signature' not in primitive.module_contract(layer)[0][4]
    original_resolver = ops.resolve_cast_module_with_vbar

    def replacement(*args, **kwargs):
        return original_resolver(*args, **kwargs)

    monkeypatch.setattr(ops, 'resolve_cast_module_with_vbar', replacement)
    assert primitive.module_contract(layer)[0][4]['_v_signature'] is None


@pytest.mark.parametrize('missing', ['resolver', 'cast', 'owner', 'non_type_owner'])
def test_missing_new_Core_transport_API_only_disables_exemption(missing, monkeypatch):
    # Ordinary no-cast CPU execution does not need newer VBAR diagnostics.
    layer = ops.disable_weight_init.Linear(3, 2, bias=False, device='cpu', dtype=torch.float32)
    layer.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3) / 10)
    layer._v_signature = None
    x = torch.tensor([[1., 2., 3.], [-1., .5, 2.]])
    expected = layer(x)
    if missing == 'non_type_owner':
        monkeypatch.setattr(ops, 'CastWeightBiasOp', object())
    else:
        name = {'resolver': 'resolve_cast_module_with_vbar', 'cast': 'cast_bias_weight',
                'owner': 'CastWeightBiasOp'}[missing]
        monkeypatch.delattr(ops, name)
    assert primitive.module_contract(layer)[0][4]['_v_signature'] is None
    assert torch.equal(layer(x), expected)
