from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import MethodType

import pytest
import torch

from comfy import ops
from comfy.ldm.minimax import model as core_h3
from comfy.ldm.minimax.model import DiTBlock
from comfy.ldm.modules import attention as core_attention
from comfy.patcher_extension import WrapperExecutor

import h3_audio_t8_pkg
from h3_audio_t8_pkg import prompt_relay_advanced as prompt_relay
from h3_audio_t8_pkg.h3_memory_advanced import (
    ATTACHMENT_KEY,
    ATTENTION_WRAPPER_KEY,
    FFN_WRAPPER_KEY,
    RUNTIME_TOKEN_KEY,
    configure_chunk_feed_forward,
    configure_low_vram_attention,
    inspect_t8_memory_composition,
)
from h3_audio_t8_pkg.nodes_h3_memory_advanced import (
    MiniMaxH3ChunkFeedForwardT8Advanced,
    MiniMaxH3LowVRAMAttentionT8Advanced,
)
from test_prompt_relay_core_compat import model_fixture
from test_prompt_relay_core_compat import bound_layout


def _small_model(block_count=2):
    model = model_fixture()
    model.model.diffusion_model.blocks = torch.nn.ModuleList(
        [
            DiTBlock(
                24,
                3,
                8,
                32,
                24,
                1e-6,
                1e-6,
                dtype=torch.float32,
                device="cpu",
                operations=ops.disable_weight_init,
            )
            for _ in range(block_count)
        ]
    )
    generator = torch.Generator().manual_seed(915)
    with torch.no_grad():
        for value in model.model.parameters():
            value.copy_(torch.randn(value.shape, generator=generator) * 0.1)
        for block in model.model.diffusion_model.blocks:
            block.attn.q_norm.weight.fill_(1.0)
            block.attn.k_norm.weight.fill_(1.0)
    return model


def _block_args(rows=17):
    return (
        torch.randn(rows, 24, generator=torch.Generator().manual_seed(916)),
        torch.randn(1, 24, generator=torch.Generator().manual_seed(917)),
        [(0, rows, 0)],
        None,
    )


def test_node_schemas_have_deliberately_small_public_surface():
    attention = MiniMaxH3LowVRAMAttentionT8Advanced.define_schema()
    ffn = MiniMaxH3ChunkFeedForwardT8Advanced.define_schema()
    assert attention.node_id == "MiniMaxH3LowVRAMAttentionT8Advanced"
    assert [item.id for item in attention.inputs] == ["model", "head_chunks"]
    attention_inputs = {item.id: item for item in attention.inputs}
    assert attention_inputs["head_chunks"].default == 4
    assert [item.id for item in ffn.inputs] == ["model", "chunks", "seq_threshold"]
    ffn_inputs = {item.id: item for item in ffn.inputs}
    assert ffn_inputs["chunks"].default == 2
    assert ffn_inputs["seq_threshold"].default == 4096
    assert [item.id for item in attention.outputs] == ["model", "report_json"]
    assert [item.id for item in ffn.outputs] == ["model", "report_json"]


def test_ffn_chunks_one_is_an_exact_object_identity_bypass():
    sentinel = object()
    returned, report = configure_chunk_feed_forward(sentinel, 1, 4096)
    assert returned is sentinel
    assert report["status"] == "identity"
    assert report["bit_exact_claim"] is True


def test_attention_head_chunks_one_is_still_an_active_early_release_patch():
    source = _small_model()
    patched, report = configure_low_vram_attention(source, 1)
    assert patched is not source
    assert report["status"] == "active"
    assert report["head_chunks"] == 1
    assert len(patched.object_patches) == 4
    assert source.object_patches == {}


def test_two_nodes_compose_in_either_order_without_mutating_source():
    for order in ("attention_first", "ffn_first"):
        source = _small_model()
        if order == "attention_first":
            middle, _ = configure_low_vram_attention(source, 2)
            patched, _ = configure_chunk_feed_forward(middle, 3, 256)
        else:
            middle, _ = configure_chunk_feed_forward(source, 3, 256)
            patched, _ = configure_low_vram_attention(middle, 2)
        assert source.object_patches == {}
        assert len(middle.object_patches) in (2, 4)
        assert len(patched.object_patches) == 6
        receipt = patched.get_attachment(ATTACHMENT_KEY)
        assert receipt.attention is not None and receipt.ffn is not None
        tokens = patched.model_options["transformer_options"][RUNTIME_TOKEN_KEY]
        assert tokens["attention"] is receipt.attention.token
        assert tokens["ffn"] is receipt.ffn.token


def test_two_t8_memory_nodes_compose_with_prompt_relay_and_execute_all_guards():
    source, _ = configure_low_vram_attention(_small_model(1), 4)
    source, _ = configure_chunk_feed_forward(source, 2, 4096)
    binding, layout = bound_layout("joint_av_exp")
    patched, _ = prompt_relay.patch_prompt_relay_model(source, binding, 32)

    contract = prompt_relay.prompt_relay_model_contract(patched)
    memory = contract["memory_composition"]
    assert memory["kind"] == "t8_h3_memory"
    assert memory["head_chunks"] == 4
    assert memory["ffn_settings"] == [2, 4096]
    assert set(memory["wrapper_keys"]) == {
        ATTENTION_WRAPPER_KEY,
        FFN_WRAPPER_KEY,
    }

    patched.patch_model(load_weights=False)
    try:
        wrappers = patched.get_all_wrappers("diffusion_model")
        executor = WrapperExecutor.new_executor(lambda *args, **kwargs: "ok", wrappers)
        result = executor.execute(
            [torch.zeros(1)],
            None,
            None,
            patched.model_options["transformer_options"],
            minimax_payload={"layout": layout},
            **{prompt_relay.PROMPT_RELAY_PAYLOAD_KEY: binding["binding_hash"]},
        )
        assert result == "ok"
        assert prompt_relay.PROMPT_RELAY_RUNTIME_KEY not in patched.model_options[
            "transformer_options"
        ]
    finally:
        patched.unpatch_model(unpatch_weights=False)


def test_prompt_relay_t8_memory_keeps_later_user_wrapper(caplog):
    source, _ = configure_low_vram_attention(_small_model(1), 4)
    source, _ = configure_chunk_feed_forward(source, 2, 4096)
    binding, layout = bound_layout("joint_av_exp")
    patched, _ = prompt_relay.patch_prompt_relay_model(source, binding, 32)
    patched.add_wrapper_with_key("diffusion_model", "foreign", lambda executor, *a, **k: executor(*a, **k))

    assert prompt_relay.prompt_relay_model_contract(patched)["memory_composition"] is None

    patched.patch_model(load_weights=False)
    try:
        executor = WrapperExecutor.new_executor(
            lambda *args, **kwargs: "delegated",
            patched.get_all_wrappers("diffusion_model"),
        )
        assert executor.execute(
                [torch.zeros(1)],
                None,
                None,
                patched.model_options["transformer_options"],
                minimax_payload={"layout": layout},
                **{prompt_relay.PROMPT_RELAY_PAYLOAD_KEY: binding["binding_hash"]},
            ) == "delegated"
        assert "advisory" in caplog.text
    finally:
        patched.unpatch_model(unpatch_weights=False)


@pytest.mark.parametrize("head_chunks", [1, 2, 3, 56])
@pytest.mark.parametrize("use_ffn", [False, True])
def test_patched_block_matches_native_h3_equation_on_cpu(
    monkeypatch, head_chunks, use_ffn
):
    monkeypatch.setattr(core_h3, "optimized_attention", core_attention.attention_pytorch)
    source = _small_model(1)
    block = source.model.diffusion_model.blocks[0]
    x, t_emb, segments, rope = _block_args(19)
    expected = block(
        x.clone(),
        t_emb,
        segments,
        rope,
        transformer_options={},
    )
    patched, _ = configure_low_vram_attention(source, head_chunks)
    if use_ffn:
        patched, _ = configure_chunk_feed_forward(patched, 3, 256)
    patched.patch_model(load_weights=False)
    try:
        options = patched.model_options["transformer_options"]
        actual = block(
            x.clone(),
            t_emb,
            segments,
            rope,
            transformer_options=options,
        )
    finally:
        patched.unpatch_model(unpatch_weights=False)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_attention_calls_one_backend_delegate_per_effective_group(monkeypatch):
    source = _small_model(1)
    calls = []

    def delegate(func, q, k, v, heads, **kwargs):
        calls.append((heads, tuple(q.shape)))
        return core_attention.attention_pytorch(
            q, k, v, heads, **{**kwargs, "_inside_attn_wrapper": True}
        )

    source.model_options["transformer_options"]["optimized_attention_override"] = delegate
    patched, report = configure_low_vram_attention(source, 2)
    block = patched.model.diffusion_model.blocks[0]
    patched.patch_model(load_weights=False)
    try:
        x = torch.randn(11, 24)
        result = block.attn(
            x,
            transformer_options=patched.model_options["transformer_options"],
        )
    finally:
        patched.unpatch_model(unpatch_weights=False)
    assert result.shape == x.shape
    assert [item[0] for item in calls] == [2, 1]
    assert "callable_global_override" in report["attention_backend"]


def test_ffn_threshold_is_inclusive_and_chunks_only_above_it(monkeypatch):
    source = _small_model(1)
    patched, _ = configure_chunk_feed_forward(source, 3, 256)
    mlp = patched.model.diffusion_model.blocks[0].mlp
    original = comfy_linear = ops.linear_input_act
    rows = []

    def counted(*args, **kwargs):
        rows.append(int(args[1].shape[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(ops, "linear_input_act", counted)
    patched.patch_model(load_weights=False)
    try:
        mlp(torch.randn(256, 24))
        assert rows == [256]
        rows.clear()
        mlp(torch.randn(257, 24))
        assert rows == [86, 86, 85]
    finally:
        patched.unpatch_model(unpatch_weights=False)
        monkeypatch.setattr(ops, "linear_input_act", comfy_linear)


@pytest.mark.parametrize("kind", ["attention", "ffn"])
def test_runtime_guard_warns_and_preserves_later_forward(kind, caplog):
    source = _small_model(1)
    if kind == "attention":
        patched, _ = configure_low_vram_attention(source, 2)
        wrapper_key = ATTENTION_WRAPPER_KEY
        owner = patched.model.diffusion_model.blocks[0].attn
    else:
        patched, _ = configure_chunk_feed_forward(source, 2, 256)
        wrapper_key = FFN_WRAPPER_KEY
        owner = patched.model.diffusion_model.blocks[0].mlp
    patched.patch_model(load_weights=False)
    try:
        owner.forward = MethodType(lambda self, x, **kwargs: x, owner)
        wrappers = patched.get_wrappers("diffusion_model", wrapper_key)
        executor = WrapperExecutor.new_executor(
            lambda *args, **kwargs: owner.forward(torch.ones(2, 24)),
            wrappers,
        )
        actual = executor.execute(
                None,
                None,
                None,
                patched.model_options["transformer_options"],
            )
        assert torch.equal(actual, torch.ones(2, 24))
        assert "replaced after binding" in caplog.text
    finally:
        patched.unpatch_model(unpatch_weights=False)


def test_duplicate_t8_nodes_are_rejected_but_foreign_mlp_owner_is_delegated(caplog):
    attention, _ = configure_low_vram_attention(_small_model(1), 2)
    with pytest.raises(RuntimeError, match="already installed"):
        configure_low_vram_attention(attention, 2)
    ffn, _ = configure_chunk_feed_forward(_small_model(1), 2, 256)
    with pytest.raises(RuntimeError, match="already installed"):
        configure_chunk_feed_forward(ffn, 2, 256)

    foreign = _small_model(1)
    calls = []
    def upstream(x):
        calls.append(len(x))
        return x * 2
    path = "diffusion_model.blocks.0.mlp.forward"
    foreign.object_patches[path] = upstream
    patched, report = configure_chunk_feed_forward(foreign, 2, 256)
    assert foreign.object_patches[path] is upstream
    assert report["delegated_mlp_paths"] == [path]
    assert report["compatibility_warnings"]
    assert "your own risk" in caplog.text
    patched.patch_model(load_weights=False)
    try:
        mlp = patched.model.diffusion_model.blocks[0].mlp
        for rows in (256, 257):
            x = torch.randn(rows, 24)
            torch.testing.assert_close(mlp(x), x * 2, rtol=0, atol=0)
        assert calls == [256, 129, 128]
    finally:
        patched.unpatch_model(unpatch_weights=False)


def test_dit_replacement_and_weight_lora_metadata_are_preserved(caplog):
    source = _small_model(1)
    source.patches["diffusion_model.blocks.0.mlp.fc1.weight"] = [(1.0, (object(),))]
    patched, _ = configure_chunk_feed_forward(source, 2, 256)
    assert patched.patches == source.patches
    assert patched.patches is not source.patches
    source.model_options["transformer_options"]["patches_replace"] = {
        "dit": {("double_block", 0): object()}
    }
    patched, report = configure_chunk_feed_forward(source, 2, 256)
    assert patched.model_options["transformer_options"]["patches_replace"] == (
        source.model_options["transformer_options"]["patches_replace"]
    )
    assert report["compatibility_warnings"]
    assert "not rejected" in caplog.text


@pytest.mark.parametrize("runtime_patch", [False, True])
def test_foreign_attention_owner_remains_active_with_chunk_ffn(monkeypatch, caplog, runtime_patch):
    monkeypatch.setattr(core_h3, "optimized_attention", core_attention.attention_pytorch)
    source = _small_model(1)
    block = source.model.diffusion_model.blocks[0]
    original = block.attn.forward
    calls = []
    def kj_like(self, x, rope_freqs=None, transformer_options={}):
        calls.append(len(x))
        return original(x, rope_freqs=rope_freqs, transformer_options=transformer_options)
    upstream = MethodType(kj_like, block.attn)
    path = "diffusion_model.blocks.0.attn.forward"
    if runtime_patch:
        block.attn.forward = upstream
    else:
        source.add_object_patch(path, upstream)
    x, t_emb, segments, rope = _block_args(257)
    patched, report = configure_chunk_feed_forward(source, 2, 256)
    assert report["compatibility_warnings"]
    if not runtime_patch:
        assert patched.object_patches[path] is upstream
        assert source.object_patches[path] is upstream
    patched.patch_model(load_weights=False)
    try:
        executor = WrapperExecutor.new_executor(
            lambda *args: block(x.clone(), t_emb, segments, rope, transformer_options=args[3]),
            patched.get_all_wrappers("diffusion_model"),
        )
        actual = executor.execute(None, None, None, patched.model_options["transformer_options"])
        assert actual.shape == x.shape and torch.isfinite(actual).all()
        assert calls == [257]
        assert block.attn.forward is upstream
    finally:
        patched.unpatch_model(unpatch_weights=False)
    assert "not rejected" in caplog.text


def test_low_vram_attention_keeps_foreign_attention_with_warning(caplog):
    foreign = _small_model(1)
    foreign.object_patches["diffusion_model.blocks.0.attn.forward"] = lambda x: x
    patched, report = configure_low_vram_attention(foreign, 2)
    assert patched.object_patches == foreign.object_patches
    assert report["preserved_foreign_blocks"] == 1
    assert "advisory" in caplog.text


def test_permissive_ffn_does_not_authenticate_unknown_owners_for_cached_composition():
    foreign = _small_model(1)
    foreign.object_patches["diffusion_model.blocks.0.attn.forward"] = lambda x: x
    patched, report = configure_chunk_feed_forward(foreign, 2, 256)
    assert report["compatibility_warnings"]
    assert inspect_t8_memory_composition(patched) is None


def test_registration_is_append_only_and_features_match():
    ids = [
        node.define_schema().node_id
        for node in asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    ]
    feature_ids = json.loads(
        (Path(__file__).resolve().parents[1] / "features.json").read_text(
            encoding="utf-8"
        )
    )["nodes"]
    assert len(ids) == len(set(ids)) == 676
    assert ids == feature_ids
    assert ids[334:336] == [
        "MiniMaxH3LowVRAMAttentionT8Advanced",
        "MiniMaxH3ChunkFeedForwardT8Advanced",
    ]
    # The original 343-node prefix stays in place as new nodes append.
    assert ids[339:343] == ['SolAttnMiniMax', 'MiniMaxH3SemanticBridgeConfigT8',
                        'MiniMaxH3SemanticBridgeApplyT8', 'MiniMaxH3LTXLatentAdapterEXPT8']


def test_report_json_is_deterministic_and_does_not_serialize_runtime_tokens():
    source = _small_model(1)
    output = MiniMaxH3LowVRAMAttentionT8Advanced.execute(source, 4)
    report = json.loads(output.result[1])
    assert report["head_chunks"] == 4
    assert "token" not in output.result[1].lower()
    output = MiniMaxH3ChunkFeedForwardT8Advanced.execute(source, 1, 4096)
    assert output.result[0] is source
    assert json.loads(output.result[1])["status"] == "identity"
