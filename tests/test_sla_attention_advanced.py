from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import h3_audio_t8_pkg.sla_attention_advanced as sla
from h3_audio_t8_pkg.nodes_sla_attention_advanced import (
    MiniMaxH3LightX2VSLAAuditT8Advanced,
    MiniMaxH3LightX2VSLAKJSageComposerT8Advanced,
    MiniMaxH3LightX2VSLAT8Advanced,
)
from h3_audio_t8_pkg.sla_attention_advanced import (
    SLA_EXPECTED_BLOCKS,
    SLA_EXPECTED_NFE,
    SLARuntime,
    _compose_kj_sage_forward,
    _inspect_kj_sage_contract,
    _model_dtype_contract,
    _validate_sigmas,
    finalize_sla_runtime,
    lightx2v_block_map,
    mean_pool_blocks,
    route_sla_attention,
)
from h3_audio_t8_pkg.sampling import native_flow_sigmas


def test_mean_pool_blocks_uses_exact_tail_divisor():
    values = torch.arange(10, dtype=torch.float32).view(1, 1, 10, 1)
    pooled = mean_pool_blocks(values, 4)
    assert pooled.flatten().tolist() == pytest.approx([1.5, 5.5, 8.5])


def test_lightx2v_router_matches_direct_reference_and_floor_topk():
    generator = torch.Generator().manual_seed(123)
    q = torch.randn(1, 3, 257, 8, generator=generator)
    k = torch.randn(1, 3, 257, 8, generator=generator)
    sparse_map, topk = lightx2v_block_map(q, k)

    smooth_k = k - k.mean(dim=-2, keepdim=True)
    pooled_q = mean_pool_blocks(q, sla.SLA_Q_BLOCK)
    pooled_k = mean_pool_blocks(smooth_k, sla.SLA_K_BLOCK)
    scores = pooled_q @ pooled_k.transpose(-1, -2)
    expected_topk = max(1, int(sla.SLA_KEEP_RATIO * scores.shape[-1]))
    indices = torch.topk(scores, expected_topk, dim=-1, sorted=False).indices
    expected = torch.zeros_like(scores, dtype=torch.int8)
    expected.scatter_(-1, indices, 1)

    assert topk == expected_topk == 1
    assert torch.equal(sparse_map, expected)
    assert torch.equal(sparse_map.sum(dim=-1), torch.ones_like(scores[..., 0]))


def test_sigma_contract_requires_exact_4step_shift6_native_flow():
    report = _validate_sigmas(native_flow_sigmas(4, 6.0))
    assert report["nfe"] == 4
    with pytest.raises(ValueError, match="4 NFE"):
        _validate_sigmas(native_flow_sigmas(8, 6.0))
    with pytest.raises(ValueError, match="video shift 6.0"):
        _validate_sigmas(native_flow_sigmas(4, 12.0))


def test_model_dtype_contract_does_not_mistake_quantized_compute_dtype_for_bf16():
    class FakeWeight:
        dtype = torch.bfloat16

    class FakeProjection:
        weight = FakeWeight()
        quant_format = "INT8 ConvRot"

    class FakeAttention:
        qkv_proj = FakeProjection()

    class FakeBlock:
        attn = FakeAttention()

    class FakeDiffusion:
        blocks = [FakeBlock()]

    class FakeBase:
        diffusion_model = FakeDiffusion()

    class FakeModel:
        model = FakeBase()

        @staticmethod
        def model_size():
            return 123

    report = _model_dtype_contract(FakeModel(), "auto_detect_exp")
    assert report["observed_qkv_dtype"] == "torch.bfloat16"
    assert report["observed_quant_format"] == "INT8 ConvRot"
    assert report["quantized_base_observed"] is True
    assert report["official_bf16_base_observed"] is False
    assert report["compatibility_status"] == "quantized_base_experimental"
    with pytest.raises(RuntimeError, match="upstream BF16 FL2VA base"):
        _model_dtype_contract(FakeModel(), "official_bf16_only")


def _runtime(mode: str) -> SLARuntime:
    return SLARuntime(
        {
            "mode": mode,
            "sparsity_ratio_requested": sla.SLA_SPARSITY_RATIO,
        }
    )


def minimax_sageattn_forward(self, x, rope_freqs=None, transformer_options=None):
    qkv = self.qkv_proj(x)
    _chunks = (transformer_options or {}).get("minimax_head_chunks", 1)
    output = _sageattn_int8_fp8_nhd(qkv, x.dtype)  # noqa: F821
    return self.out_proj(output)


def test_kj_sage_contract_requires_complete_consistent_bound_50_block_patch():
    class FakeAttention:
        qkv_proj = None
        out_proj = None

    class FakeBlock:
        def __init__(self):
            self.attn = FakeAttention()

    class FakeDiffusion:
        def __init__(self):
            self.blocks = [FakeBlock() for _ in range(SLA_EXPECTED_BLOCKS)]

    class FakeBase:
        def __init__(self):
            self.diffusion_model = FakeDiffusion()

    class FakeModel:
        def __init__(self):
            self.model = FakeBase()
            self.object_patches = {
                f"diffusion_model.blocks.{index}.attn.forward": (
                    minimax_sageattn_forward.__get__(
                        block.attn, type(block.attn)
                    )
                )
                for index, block in enumerate(self.model.diffusion_model.blocks)
            }

    model = FakeModel()
    contract = _inspect_kj_sage_contract(model)
    assert contract["patch_count"] == SLA_EXPECTED_BLOCKS
    assert contract["source_sha256"]
    model.object_patches.pop("diffusion_model.blocks.49.attn.forward")
    with pytest.raises(RuntimeError, match="exactly 50"):
        _inspect_kj_sage_contract(model)


def test_kj_sage_composer_dispatches_one_backend_per_call():
    class FakeAttention:
        def forward(self, x, **_kwargs):
            return ("sla_stock", x)

    module = FakeAttention()

    def kj_forward(x, **_kwargs):
        return ("kj_sage", x)

    runtime = SLARuntime(
        {"mode": "dense_lora_control", "external_attention_policy": "compose_kj_sage"}
    )
    index = runtime.begin_forward(
        {
            "task": "FL2VA",
            "seq_len": 8,
            "pixel_frames": 22,
            "latent_t": 6,
            "latent_h": 4,
            "latent_w": 4,
        }
    )
    route = {
        "runtime": runtime,
        "forward_index": index,
        "mode": "apply_lightx2v_sla",
        "seq_len": 8,
    }
    composed = _compose_kj_sage_forward(
        module, kj_forward, source_sha256="test-fingerprint"
    )
    x = torch.zeros(8, 16)
    assert composed(x, transformer_options={sla.SLA_RUNTIME_KEY: route})[0] == "sla_stock"
    assert composed(x, transformer_options={})[0] == "kj_sage"

    route["mode"] = "dense_lora_control"
    assert composed(x, transformer_options={sla.SLA_RUNTIME_KEY: route})[0] == "kj_sage"
    report = runtime.snapshot(consume=False)
    assert report["dense_control_calls_per_forward"] == [1]
    assert report["external_sage_calls_per_forward"] == [1]


def test_sparse_attention_route_uses_exact_map_without_dense_fallback(monkeypatch):
    runtime = _runtime("apply_lightx2v_sla")
    route = {
        "task": "FL2VA",
        "seq_len": 129,
        "pixel_frames": 22,
        "latent_t": 6,
        "latent_h": 4,
        "latent_w": 4,
    }
    forward_index = runtime.begin_forward(route)
    route.update(
        {
            "runtime": runtime,
            "forward_index": forward_index,
            "mode": "apply_lightx2v_sla",
            "max_router_workspace_mib": 512,
        }
    )
    observed = {}

    def fake_sparse(q, k, v, **kwargs):
        observed["mask"] = kwargs["mask_id"].clone()
        observed["kwargs"] = dict(kwargs)
        return q.clone()

    import spas_sage_attn

    monkeypatch.setattr(
        spas_sage_attn, "block_sparse_sage2_attn_cuda", fake_sparse
    )
    q = torch.randn(1, sla.SLA_HEADS, 129, sla.SLA_HEAD_DIM)
    output = route_sla_attention(
        q,
        q.clone(),
        q.clone(),
        sla.SLA_HEADS,
        skip_reshape=True,
        transformer_options={sla.SLA_RUNTIME_KEY: route},
    )
    expected_map, topk = lightx2v_block_map(q, q)

    assert output.shape == (1, 129, sla.SLA_HEADS * sla.SLA_HEAD_DIM)
    assert torch.equal(observed["mask"], expected_map)
    assert observed["kwargs"]["smooth_k"] is True
    assert observed["kwargs"]["pvthreshd"] == 1.0e6
    report = runtime.snapshot(consume=False)
    assert report["sparse_kernel_calls_per_forward"] == [1]
    assert report["dense_control_calls_per_forward"] == [0]
    assert report["forwards"][0]["retained_key_blocks_min"] == topk
    assert report["forwards"][0]["retained_key_blocks_max"] == topk


def test_dense_control_delegates_and_records_without_router(monkeypatch):
    runtime = _runtime("dense_lora_control")
    route = {
        "task": "FL2VA",
        "seq_len": 8,
        "pixel_frames": 22,
        "latent_t": 6,
        "latent_h": 4,
        "latent_w": 4,
    }
    forward_index = runtime.begin_forward(route)
    route.update(
        {
            "runtime": runtime,
            "forward_index": forward_index,
            "mode": "dense_lora_control",
            "max_router_workspace_mib": 512,
        }
    )

    def fake_dense(q, _k, _v, heads, **_kwargs):
        return torch.zeros(q.shape[0], q.shape[-2], heads * q.shape[-1])

    monkeypatch.setattr(sla.attention_module, "optimized_attention", fake_dense)
    q = torch.randn(1, sla.SLA_HEADS, 8, sla.SLA_HEAD_DIM)
    output = route_sla_attention(
        q,
        q,
        q,
        sla.SLA_HEADS,
        skip_reshape=True,
        transformer_options={sla.SLA_RUNTIME_KEY: route},
    )
    assert output.shape == (1, 8, sla.SLA_HEADS * sla.SLA_HEAD_DIM)
    report = runtime.snapshot(consume=False)
    assert report["dense_control_calls_per_forward"] == [1]
    assert report["sparse_kernel_calls_per_forward"] == [0]


@pytest.mark.parametrize(
    ("mode", "sparse"),
    [("apply_lightx2v_sla", True), ("dense_lora_control", False)],
)
def test_runtime_audit_requires_four_forwards_and_fifty_blocks(mode, sparse):
    runtime = _runtime(mode)
    for _forward in range(SLA_EXPECTED_NFE):
        index = runtime.begin_forward(
            {
                "task": "FL2VA",
                "seq_len": 1024,
                "pixel_frames": 124,
                "latent_t": 31,
                "latent_h": 8,
                "latent_w": 8,
            }
        )
        for _block in range(SLA_EXPECTED_BLOCKS):
            runtime.record_attention(
                index,
                sparse=sparse,
                workspace_bytes=4096 if sparse else 0,
                key_blocks=16 if sparse else 0,
                retained_key_blocks=2 if sparse else 0,
            )
    latent = {"samples": torch.zeros(1)}
    returned, report_json = finalize_sla_runtime(latent, runtime)
    report = json.loads(report_json)
    assert returned is latent
    assert report["model_forward_count"] == 4
    assert report["status"].endswith("verified")


def test_runtime_audit_refuses_missing_block():
    runtime = _runtime("apply_lightx2v_sla")
    for _forward in range(4):
        index = runtime.begin_forward(
            {
                "task": "FL2VA",
                "seq_len": 1024,
                "pixel_frames": 124,
                "latent_t": 31,
                "latent_h": 8,
                "latent_w": 8,
            }
        )
        for _block in range(49):
            runtime.record_attention(
                index,
                sparse=True,
                workspace_bytes=4096,
                key_blocks=16,
                retained_key_blocks=2,
            )
    with pytest.raises(RuntimeError, match="50 main attention"):
        finalize_sla_runtime({"samples": torch.zeros(1)}, runtime)


def test_runtime_audit_verifies_kj_sage_dense_control_dispatch():
    runtime = SLARuntime(
        {"mode": "dense_lora_control", "external_attention_policy": "compose_kj_sage"}
    )
    for _forward in range(SLA_EXPECTED_NFE):
        index = runtime.begin_forward(
            {
                "task": "FL2VA",
                "seq_len": 1024,
                "pixel_frames": 124,
                "latent_t": 31,
                "latent_h": 8,
                "latent_w": 8,
            }
        )
        for _block in range(SLA_EXPECTED_BLOCKS):
            runtime.record_attention(
                index, sparse=False, external_backend="kj_sage"
            )
    _latent, report_json = finalize_sla_runtime(
        {"samples": torch.zeros(1)}, runtime
    )
    assert json.loads(report_json)["status"] == "dense_lora_kj_sage_control_verified"


def test_new_node_ids_are_append_safe_advanced_ids():
    schemas = [
        MiniMaxH3LightX2VSLAT8Advanced.define_schema(),
        MiniMaxH3LightX2VSLAAuditT8Advanced.define_schema(),
        MiniMaxH3LightX2VSLAKJSageComposerT8Advanced.define_schema(),
    ]
    assert [schema.node_id for schema in schemas] == [
        "MiniMaxH3LightX2VSLAT8Advanced",
        "MiniMaxH3LightX2VSLAAuditT8Advanced",
        "MiniMaxH3LightX2VSLAKJSageComposerT8Advanced",
    ]
    assert all(schema.node_id.endswith("Advanced") for schema in schemas)


def test_public_sla_workflow_is_importable_and_uses_one_attention_owner():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "15-sla-attention"
        / "2026-08-22_H3_LightX2V_SLA_FL2VA_4Step_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    by_type = {node["type"]: node for node in workflow["nodes"]}
    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(node["id"] for node in workflow["nodes"])
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert sum(node["type"] == "MarkdownNote" for node in workflow["nodes"]) == 3
    assert by_type["MiniMaxH3DualClockSamplerT8"]["widgets_values"][:5] == [
        4,
        6.0,
        3.0,
        "dual_clock_euler",
        "native_flow",
    ]
    assert by_type["MiniMaxH3LightX2VSLAT8Advanced"]["widgets_values"] == [
        sla.SLA_LORA_FILENAME,
        "apply_lightx2v_sla",
        "auto_detect_exp",
        512,
    ]
    forbidden = {
        "LoraLoaderModelOnly",
        "LoraLoaderBypassModelOnly",
        "MiniMaxH3EnhanceAVideoT8Advanced",
        "MiniMaxH3EnhanceAVideoSageComposerT8Advanced",
    }
    assert forbidden.isdisjoint(by_type)


def test_public_sla_kj_composer_workflow_has_one_conditional_attention_owner():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "15-sla-attention"
        / "2026-08-22_H3_LightX2V_SLA_KJ_Sage_Composer_FL2VA_4Step_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    by_type = {node["type"]: node for node in workflow["nodes"]}
    dual_clock = by_type["MiniMaxH3DualClockSamplerT8"]
    kj_sage = by_type["MiniMaxH3MemoryEfficientSageAttentionPatch"]
    composer = by_type["MiniMaxH3LightX2VSLAKJSageComposerT8Advanced"]
    assert workflow["version"] == 0.4
    assert sum(node["type"] == "MarkdownNote" for node in workflow["nodes"]) == 3
    links = {int(link[0]): link for link in workflow["links"]}
    assert links[int(kj_sage["inputs"][0]["link"])][1:5] == [
        dual_clock["id"],
        0,
        kj_sage["id"],
        0,
    ]
    assert links[int(composer["inputs"][0]["link"])][1:5] == [
        kj_sage["id"],
        0,
        composer["id"],
        0,
    ]
    assert composer["widgets_values"] == [
        sla.SLA_LORA_FILENAME,
        "apply_lightx2v_sla",
        "auto_detect_exp",
        512,
    ]
    assert "ModelAttentionBackend" not in by_type
    assert "SolAttnMiniMax" not in by_type
