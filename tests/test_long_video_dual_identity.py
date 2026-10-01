from collections import namedtuple

import pytest
import torch

from h3_audio_t8_pkg.long_video_dual_identity import stage_model_identity, content_identity
from h3_audio_t8_pkg.h3_memory_advanced import (
    ATTENTION_WRAPPER_KEY,
    RUNTIME_TOKEN_KEY,
    configure_chunk_feed_forward,
    configure_low_vram_attention,
)
from test_relay_kj_memory import small_model, patched_source, memory_nodes  # noqa: F401
from test_relay_kj_backend import kj  # noqa: F401
from test_relay_sol_backend import installed_sol  # noqa: F401


def test_same_base_different_lora_strength_or_order_changes_identity():
    a = small_model()
    tensor = torch.ones(3)
    a.patches = {"diffusion_model.blocks.0.attn.qkv_proj.weight": [(1., ("lora", (tensor,)), 1., None, None)]}
    b = a.clone()
    first = stage_model_identity(a)
    assert stage_model_identity(b)["sha256"] == first["sha256"]
    b.patches[next(iter(b.patches))][0] = (.5, ("lora", (tensor,)), 1., None, None)
    assert stage_model_identity(b)["sha256"] != first["sha256"]
    assert len(a.patches[next(iter(a.patches))]) == 1


def test_unsampled_weight_bytes_are_hashed_not_just_64_positions():
    model = small_model()
    before = stage_model_identity(model)["sha256"]
    with torch.no_grad():
        model.model.diffusion_model.blocks[0].attn.qkv_proj.weight.view(-1)[57] += .001
    assert stage_model_identity(model)["sha256"] != before


def test_pre_lora_backup_provides_stable_base_identity():
    model = small_model()
    before = stage_model_identity(model)["sha256"]
    name = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    weight = model.model_state_dict()[name]
    backup = namedtuple("Backup", "weight inplace_update")
    model.backup[name] = backup(weight.clone(), False)
    with torch.no_grad():
        weight.add_(1.)
    assert stage_model_identity(model)["sha256"] == before


def test_installed_kj_memory_selection_is_retained_without_faking_portable_configuration(request):
    memory_fixture = request.getfixturevalue("memory_nodes")
    model = patched_source(memory_fixture, "sage_lowmem_ffn", 2)
    methods = dict(model.object_patches)
    assert model.model_options['transformer_options']['minimax_head_chunks'] == 2
    first = stage_model_identity(model)
    other = patched_source(memory_fixture, "sage_lowmem_ffn", 3)
    second = stage_model_identity(other)
    assert model.model_options['transformer_options']['minimax_head_chunks'] == 2
    assert other.model_options['transformer_options']['minimax_head_chunks'] == 3
    assert set(methods) == set(model.object_patches)
    assert all(model.object_patches[key] is value for key, value in methods.items())
    if first.get('portable_cache_reuse') is False:
        # Installed third-party implementation changes are allowed to run, but
        # are not an excuse to mint a portable certificate or erase selection.
        assert first['memory'] is None and first['backend']['kind'] == 'user_selected_unverified'
        assert second['portable_cache_reuse'] is False
        assert first['execution_selection']['model_options'] != second['execution_selection']['model_options']
        assert first['execution_selection']['object_patches']
        assert first['opaque_internal_state_verified'] is False
    else:
        assert first['memory']['head_chunks'] == 2 and second['memory']['head_chunks'] == 3
        assert second['sha256'] != first['sha256']


def test_t8_memory_nodes_are_authenticated_and_stage_specific():
    first, _ = configure_low_vram_attention(small_model(), 2)
    first, _ = configure_chunk_feed_forward(first, 2, 4096)
    identity = stage_model_identity(first)
    assert identity["memory"] == {
        "kind": "t8_h3_memory",
        "head_chunks": 2,
        "ffn_settings": [2, 4096],
        "source_sha256s": identity["memory"]["source_sha256s"],
    }

    second, _ = configure_low_vram_attention(small_model(), 4)
    second, _ = configure_chunk_feed_forward(second, 3, 8192)
    second_identity = stage_model_identity(second)
    assert second_identity["memory"]["head_chunks"] == 4
    assert second_identity["memory"]["ffn_settings"] == [3, 8192]
    assert second_identity["sha256"] != identity["sha256"]


def test_t8_memory_identity_rejects_replaced_runtime_token_or_wrapper():
    patched, _ = configure_low_vram_attention(small_model(), 2)
    patched.model_options["transformer_options"][RUNTIME_TOKEN_KEY]["attention"] = object()
    with pytest.raises(RuntimeError, match="runtime token changed"):
        stage_model_identity(patched)

    patched, _ = configure_low_vram_attention(small_model(), 2)
    patched.wrappers["diffusion_model"][ATTENTION_WRAPPER_KEY] = [lambda *args: None]
    with pytest.raises(RuntimeError, match="wrapper ownership changed"):
        stage_model_identity(patched)


def test_core_pytorch_backend_is_accepted_as_the_stage_attention_owner():
    from comfy.ldm.modules import attention
    from h3_audio_t8_pkg.h3_core_compat import set_h3_attention_backend

    patched = small_model().clone()
    set_h3_attention_backend(patched, attention.attention_pytorch)
    identity = stage_model_identity(patched)
    assert identity["backend"]["kind"] == "core_plain"
    assert identity["backend"]["name"] == "pytorch"


def test_installed_sol_selector_configuration_is_preserved_without_certifying_an_unknown_model_stack(request):
    sol_module = request.getfixturevalue("installed_sol")
    patched = sol_module.SolAttentionPatch().patch(
        small_model(),
        True,
        0.5,
        min_tokens=4096,
        strict=True,
        thresh_type="diag",
        int8_qk=False,
        int8_pv=False,
    )[0]
    from h3_audio_t8_pkg.relay_sol_backend import capture_composed_backend
    selected = patched.model_options['transformer_options']['optimized_attention_override']
    selector = capture_composed_backend(selected)
    assert selector is not None
    configuration = {
        "tau": 0.5, "min_tokens": 4096, "strict": True,
        "thresh_type": "diag", "int8_qk": False, "int8_pv": False,
    }
    assert selector.report()['configuration'] == configuration
    before = dict(patched.object_patches)
    identity = stage_model_identity(patched)
    assert patched.model_options['transformer_options']['optimized_attention_override'] is selected
    assert set(before) == set(patched.object_patches)
    assert all(patched.object_patches[key] is value for key, value in before.items())
    assert selector.report()['configuration'] == configuration
    if identity.get('portable_cache_reuse') is False:
        assert identity['backend']['kind'] == 'user_selected_unverified'
        assert identity['execution_selection']['model_options']
        assert identity['opaque_internal_state_verified'] is False
    else:
        assert identity['backend']['kind'] == 'audited_sol_attn_selector'
        assert identity['backend']['configuration'] == configuration


def test_unknown_live_hook_cannot_share_resume_identity():
    model = small_model()
    model.model.diffusion_model.blocks[0].register_forward_pre_hook(lambda *args: None)
    identity = stage_model_identity(model)
    assert identity["portable_cache_reuse"] is False
    assert stage_model_identity(model)["sha256"] != identity["sha256"]


def test_media_reordering_and_content_replacement_changes_identity():
    a, b = torch.zeros(100), torch.ones(100)
    before = content_identity([a, b])
    assert content_identity([b, a]) != before
    a[57] = .001
    assert content_identity([a, b]) != before


def test_actual_loader_metadata_is_hashed_but_not_an_execution_patch_bypass():
    model = small_model()
    model.set_attachments("t8_h3_lora_metadata", {"format": "comfy", "training": "EMA B"})
    first = stage_model_identity(model)["sha256"]
    model.set_attachments("t8_h3_lora_metadata", {"format": "comfy", "training": "other"})
    assert stage_model_identity(model)["sha256"] != first
    model.set_attachments("foreign_runtime", {"claimed_safe": True})
    assert stage_model_identity(model)["portable_cache_reuse"] is False


def test_loader_metadata_cannot_hide_a_callback():
    model = small_model()
    model.set_attachments("t8_h3_lora_metadata", {"callback": lambda: None})
    with pytest.raises(ValueError, match="plain safetensors string map"):
        stage_model_identity(model)


def test_lora_compat_diagnostic_is_inert_but_content_bound():
    model = small_model()
    baseline = stage_model_identity(model)["sha256"]
    report = {"schema": "t8.minimax_h3.lora_compat.v1", "status": "applied",
              "full_algorithm_verified": False, "model_class": type(model.model).__name__,
              "file": {"name": "EMA-B.safetensors",
                       "identity_policy": "display_only_not_a_load_gate_no_hash_scan"},
              "applied_patch_count": 1}
    model.set_attachments("t8_h3_lora_compat_report", report)
    first = stage_model_identity(model)
    assert first.get("portable_cache_reuse", True) is True
    assert first["sha256"] != baseline
    model.set_attachments("t8_h3_lora_compat_report", {**report, "applied_patch_count": 2})
    assert stage_model_identity(model)["sha256"] != first["sha256"]
    model.set_attachments("t8_h3_lora_compat_report", {**report, "extra_callback": lambda: None})
    assert stage_model_identity(model)["portable_cache_reuse"] is False
    model.set_attachments("t8_h3_lora_compat_report", {**report, "full_algorithm_verified": True})
    assert stage_model_identity(model)["portable_cache_reuse"] is False


def test_current_core_lora_adapter_and_legacy_tuples_are_both_content_bound():
    from comfy.weight_adapter.lora import LoRAAdapter
    weights = (torch.ones(2, 1), torch.ones(1, 3), 1., None, None, None)
    adapter = LoRAAdapter({'up', 'down'}, weights)
    before = content_identity({'patch': [(1., adapter, 1., None, None)]})
    weights[0][0, 0] = 2.
    assert content_identity({'patch': [(1., adapter, 1., None, None)]}) != before
    assert content_identity(('lora', weights))['type'] == 'tuple'
    adapter.h = lambda *args: None
    from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
    with pytest.raises(UnverifiedModelStack, match='additional execution state'):
        content_identity(adapter)


def test_current_core_external_lora_adapters_are_accepted_and_stage_specific():
    from comfy.weight_adapter.lora import LoRAAdapter

    key = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    weights = (torch.ones(2, 1), torch.ones(1, 3), 1., None, None, None)
    first = small_model()
    first.patches = {key: [(0.7, LoRAAdapter({'up', 'down'}, weights), 1., None, None)]}
    second = small_model()
    second.patches = {key: [(0.2, LoRAAdapter({'up', 'down'}, weights), 1., None, None)]}
    assert stage_model_identity(first)["lora_target_count"] == 1
    assert stage_model_identity(second)["lora_target_count"] == 1
    assert stage_model_identity(first)["sha256"] != stage_model_identity(second)["sha256"]


def test_float8_raw_identity_is_bounded_and_finite_on_cpu():
    import hashlib
    value = torch.ones(1024 * 1024 + 3).to(torch.float8_e4m3fn)
    identity = content_identity(value)
    assert identity['tensor_sha256'] == hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
    assert identity['dtype'] == 'torch.float8_e4m3fn'
    value.view(torch.uint8)[-1] = 127
    with pytest.raises(ValueError, match='nonfinite'):
        content_identity(value)


def test_same_lora_path_replaced_only_changes_execution_after_reload(tmp_path):
    from safetensors.torch import load_file, save_file
    from comfy.weight_adapter.lora import LoRAAdapter
    path = tmp_path / 'same-name.safetensors'
    key = 'diffusion_model.blocks.0.attn.qkv_proj.weight'
    save_file({'up': torch.ones(2, 1), 'down': torch.ones(1, 3)}, str(path))
    base = small_model()
    def loaded():
        values = {k: v.clone() for k, v in load_file(str(path)).items()}
        model = base.clone()
        adapter = LoRAAdapter({'up', 'down'}, (values['up'], values['down'], 1., None, None, None))
        model.patches = {key: [(1., adapter, 1., None, None)]}
        return model
    first = loaded()
    identity = stage_model_identity(first)['sha256']
    save_file({'up': torch.full((2, 1), 2.), 'down': torch.ones(1, 3)}, str(path))
    assert stage_model_identity(first)['sha256'] == identity
    assert stage_model_identity(loaded())['sha256'] != identity


def test_actual_backend_a_b_a_reuses_only_matching_stage(tmp_path):
    from comfy.ldm.modules import attention
    from h3_audio_t8_pkg.h3_core_compat import set_h3_attention_backend
    from h3_audio_t8_pkg.long_video_dual_stage_cache import AVStageCache
    from test_long_video_dual_model_stages import latent
    model = small_model()
    cache = AVStageCache(tmp_path)
    a = stage_model_identity(model)
    cache.save('low_x0', a, latent(2.), {})
    other = model.clone()
    set_h3_attention_backend(other, attention.attention_pytorch)
    b = stage_model_identity(other)
    assert a['sha256'] != b['sha256'] and cache.load('low_x0', b) is None
    assert stage_model_identity(model)['sha256'] == a['sha256']
    assert cache.load('low_x0', a) is not None
