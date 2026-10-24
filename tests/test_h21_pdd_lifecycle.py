"""Actual tiny baked safetensors -> native Core loader -> AV forward, CPU only.

Random miniature weights exercise file/config/head lifecycle, not trained PDD
quality, the full vendor checkpoint, or GPU qualification.
"""
from copy import deepcopy
import json

import comfy.model_management
import comfy.sd
import comfy.utils
from comfy.weight_adapter.lora import LoRAAdapter
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced
from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention
from safetensors.torch import save_file
import pytest
import torch

from h3_audio_t8_pkg import pdd_advanced as pdd
from h3_audio_t8_pkg.h3_weight_diagnostics import describe_h3_payload, inspect_h3_weight_file
from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.modular_sampling import pdd_stages as split
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from test_fast_h3_v2_core_sampler import model
from test_modular_rf_restart import source_latent
from test_progressive_sampling_runtime import conditioning


def baked_file(tmp_path):
    plain = model().model.diffusion_model
    baked = deepcopy(plain)
    generator = torch.Generator().manual_seed(211032)
    for name in ('video_out', 'audio_out'):
        head = getattr(baked.final_layer, name)
        for key in ('weight', 'bias'):
            base = getattr(head, key).detach()
            absolute = base.unsqueeze(0) + .005*torch.randn((32, *base.shape), generator=generator)
            encoded = pdd._encode_native_pdd_head_bank(absolute)
            setattr(head, key, torch.nn.Parameter(encoded, requires_grad=False))
    baked.requires_grad_(False)
    path = tmp_path/'arbitrary-baked.safetensors'
    save_file({key: value.contiguous() for key, value in baked.state_dict().items()}, str(path),
              metadata={'config': json.dumps({'transformer': {'pdd_steps': 32}})})
    return path, plain, baked


def load_cpu(path):
    loaded = comfy.sd.load_diffusion_model(str(path), model_options={'dtype': torch.float32, 'load_device': torch.device('cpu')})
    assert loaded is not None
    comfy.model_management.load_model_gpu(loaded)
    assert not torch.cuda.is_initialized()
    return loaded


def inputs():
    generator = torch.Generator().manual_seed(211033)
    x = [torch.randn((1, 24, 2, 4, 4), generator=generator), torch.randn((1, 32, 2, 3), generator=generator)]
    return x, torch.tensor([500.]), torch.randn((1, 2, 8), generator=generator)


def forward(diffusion, schedule):
    x, timestep, context = inputs()
    return diffusion(x, timestep, context, transformer_options={'sample_sigmas': schedule},
                     minimax_payload={'layout': native.PackedLayout(2, 2, 4, 4, 3)})


def test_baked_pdd_native_loader_consumes_config_head_bank_and_next_plain_request(tmp_path, monkeypatch):
    monkeypatch.setattr(native, 'optimized_attention', attention.attention_pytorch)
    path, plain, baked = baked_file(tmp_path)
    before = {key: value.clone() for key, value in plain.state_dict().items()}
    loaded = load_cpu(path)
    diffusion = loaded.model.diffusion_model
    assert diffusion.final_layer.video_out.weight.shape[0] == 32*plain.final_layer.video_out.weight.shape[0]
    assert diffusion.final_layer.audio_out.weight.shape[0] == 32*plain.final_layer.audio_out.weight.shape[0]
    for key, value in baked.state_dict().items():
        torch.testing.assert_close(diffusion.state_dict()[key], value, rtol=0, atol=0)
    with torch.inference_mode():
        expected = forward(baked, pdd.pdd_runtime_sigmas())
        actual = forward(diffusion, pdd.pdd_runtime_sigmas())
    for left, right in zip(expected, actual):
        assert torch.isfinite(right).all()
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    # A later ordinary checkpoint must not inherit 32-head configuration.
    ordinary_path = tmp_path/'ordinary-after-baked.safetensors'
    save_file({key: value.contiguous() for key, value in plain.state_dict().items()}, str(ordinary_path))
    ordinary = load_cpu(ordinary_path).model.diffusion_model
    assert ordinary.final_layer.video_out.weight.shape == plain.final_layer.video_out.weight.shape
    assert ordinary.final_layer.audio_out.weight.shape == plain.final_layer.audio_out.weight.shape
    with torch.inference_mode():
        expected_plain = forward(plain, torch.tensor([1., .5, 0.]))
        actual_plain = forward(ordinary, torch.tensor([1., .5, 0.]))
    for left, right in zip(expected_plain, actual_plain):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert all(torch.equal(value, plain.state_dict()[key]) for key, value in before.items())


def test_baked_pdd_header_explicit_config_is_diagnostic_not_generic_lora(tmp_path):
    path, _, _ = baked_file(tmp_path)
    report = inspect_h3_weight_file(path)
    assert report['status'] == 'inspected' and report['tensor_values_loaded'] is False
    requirements = report['special_runtime_requirements']
    assert any(item['kind'] == 'pdd' and item.get('representation') == 'native_stacked_heads_metadata'
               for item in requirements)


@pytest.mark.parametrize('config', ['not json', '[]', 'null', '{"transformer":null}',
                                   '{"transformer":{"pdd_steps":true}}', '{"transformer":{"pdd_steps":1}}'])
def test_unrelated_or_malformed_config_never_fabricates_pdd_runtime_or_a_load_ban(config):
    report = describe_h3_payload(['final_layer.video_out.weight'], {'config': config})
    assert not report['special_runtime_requirements']
    assert report['compatibility'] == 'not_established_by_header'
    assert report['identity_policy'] == 'structural_diagnostics_not_unknown_model_gate'


@pytest.mark.parametrize('stage', split.STAGES)
def test_actual_native_baked_file_runs_absolute_windows_without_fake_lora_receipt(tmp_path, monkeypatch, stage):
    monkeypatch.setattr(native, 'optimized_attention', attention.attention_pytorch)
    path, _, _ = baked_file(tmp_path)
    loaded = load_cpu(path)
    assert loaded.get_attachment(pdd.PDD_ATTACHMENT_KEY) is None and not loaded.patches
    source, positive = source_latent(True), conditioning()
    full = pdd.pdd_runtime_sigmas()
    bound, sampler, sigmas, context, report = split.build_stage(loaded, source, full, stage)
    assert json.loads(context.profile)['pdd']['mode'] == split.BAKED
    assert json.loads(report)['native_baked_heads_content_bound'] is True
    old, old_sampler, _ = sampling.setup_dual_clock_sampling(loaded, source, 8, 12., 3.,
        'euler' if stage == split.STAGES[0] else 'dual_clock_euler',
        'simple' if stage == split.STAGES[0] else 'native_flow')
    noise = RandomNoise.execute(211034).result[0]
    reference = SamplerCustomAdvanced.execute(noise, BasicGuider.execute(old, positive).result[0], old_sampler, sigmas, source).result
    result = sample_stage(noise, BasicGuider.execute(bound, positive).result[0], sampler, sigmas, source, context)[2]
    for actual, expected in zip((result.output, result.denoised_output), reference):
        assert all(torch.equal(a, b) for a, b in zip(actual['samples'].unbind(), expected['samples'].unbind()))
    receipt = result.verify()
    assert receipt['execution']['denoiser_evaluations'] == 4
    assert receipt['portable_identity'] and receipt['verified_recipe_completion']
    assert loaded.get_attachment(pdd.PDD_ATTACHMENT_KEY) is None and not loaded.patches


def test_plain_user_model_can_run_windows_without_false_pdd_receipt_or_portable_cache(caplog):
    source, bare = source_latent(True), model()
    bound, sampler, sigmas, context, report = split.build_stage(bare, source, pdd.pdd_runtime_sigmas())
    assert json.loads(report)['pdd_head_contract_recognized'] is False
    result = sample_stage(RandomNoise.execute(211035).result[0], BasicGuider.execute(bound, conditioning()).result[0],
                          sampler, sigmas, source, context)[2]
    receipt = result.verify()
    assert receipt['execution']['denoiser_evaluations'] == 4
    assert not receipt['portable_identity'] and not receipt['verified_recipe_completion']
    assert bare.get_attachment(pdd.PDD_ATTACHMENT_KEY) is None and 'not PDD recipe qualification' in caplog.text


def test_baked_head_lora_survives_native_load_and_does_not_invalidate_original_bank_identity(tmp_path):
    path, _, _ = baked_file(tmp_path)
    loaded = load_cpu(path)
    key = 'diffusion_model.final_layer.video_out.weight'
    weight = loaded.model_state_dict()[key]
    original = weight.detach().clone()
    adapter = LoRAAdapter(set(), (torch.full((weight.shape[0], 1), .003),
                                  torch.full((1, weight.shape[1]), .004), 1., None, None, None))
    selected = loaded.clone()
    assert selected.add_patches({key: adapter}, .7) == [key]
    source = source_latent(True)
    bound, sampler, sigmas, context, _ = split.build_stage(selected, source, pdd.pdd_runtime_sigmas())
    try:
        result = sample_stage(RandomNoise.execute(211036).result[0], BasicGuider.execute(bound, conditioning()).result[0],
                              sampler, sigmas, source, context)[2]
        assert result.verify()['portable_identity'] is True
        assert bound.patches[key][0][1] is adapter
    finally:
        bound.unpatch_model()
    torch.testing.assert_close(loaded.model_state_dict()[key], original, rtol=0, atol=0)
