from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

import h3_audio_t8_pkg.enhance_a_video_advanced as eav_module
import h3_audio_t8_pkg.detail_sampling_advanced as detail_module
import h3_audio_t8_pkg.prompt_relay_advanced as prompt_relay_module
from h3_audio_t8_pkg.conditioning import build_packed_layout
from h3_audio_t8_pkg.enhance_a_video_advanced import (
    EAV_REFERENCE_TASKS,
    EAVRuntime,
    _reference_segment_contract,
    _runtime_route,
    _validate_stock20_sigmas,
    build_eav_block_cache_model,
    build_eav_long_video_model,
    build_eav_model,
    build_eav_prompt_relay_model,
    build_eav_stg_model,
    exact_chunked_cfi,
    finalize_eav_runtime,
    route_eav_attention,
    route_eav_prompt_relay_attention,
)
from h3_audio_t8_pkg.nodes_enhance_a_video_advanced import (
    MiniMaxH3EnhanceAVideoBlockCacheComposerT8Advanced,
    MiniMaxH3EnhanceAVideoLongVideoComposerT8Advanced,
    MiniMaxH3EnhanceAVideoPromptRelayComposerT8Advanced,
    MiniMaxH3EnhanceAVideoReferenceComposerT8Advanced,
    MiniMaxH3EnhanceAVideoSageComposerT8Advanced,
    MiniMaxH3EnhanceAVideoSTGComposerT8Advanced,
)
from h3_audio_t8_pkg.long_video import (
    CONTEXT_FRAME_STEPS,
    LONG_VIDEO_PATCH_VERSION,
    MOTION_FRAME_INDEX,
    patch_long_video_model,
    step_offsets,
)
from h3_audio_t8_pkg.prompt_relay_advanced import patch_prompt_relay_model
from h3_audio_t8_pkg.tools.build_eav_reference_probe_prompts import build_prompt
from h3_audio_t8_pkg.tools.build_eav_sage_probe_prompt import (
    build_prompt as build_sage_prompt,
)
from comfy.model_patcher import ModelPatcher, create_model_options_clone
from comfy.patcher_extension import PatcherInjection
from comfy.weight_adapter.bypass import BypassInjectionManager


class MiniMaxH3Model(torch.nn.Module):
    pass


class _NativeH3Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = MiniMaxH3Model()

    def extra_conds(self, **_kwargs):
        return {}


_NativeH3Base.extra_conds.__module__ = "comfy.model_base"


def _model_patcher() -> ModelPatcher:
    return ModelPatcher(
        _NativeH3Base(),
        load_device=torch.device("cpu"),
        offload_device=torch.device("cpu"),
    )


def _allow_fixture_core(monkeypatch):
    monkeypatch.setattr(eav_module, "_source_sha256", lambda _source: "fixture")
    monkeypatch.setattr(eav_module, "ATTENTION_FORWARD_SHA256S", {"fixture"})
    monkeypatch.setattr(eav_module, "PACKED_LAYOUT_SHA256S", {"fixture"})
    monkeypatch.setattr(eav_module, "MODEL_FORWARD_SHA256S", {"fixture"})
    monkeypatch.setattr(eav_module, "PATCHIFY_VIDEO_SHA256S", {"fixture"})
    monkeypatch.setattr(eav_module, "BLOCK_CACHE_OUTER_WRAPPER_SHA256S", {"fixture"})
    monkeypatch.setattr(
        eav_module, "BLOCK_CACHE_DIFFUSION_WRAPPER_SHA256S", {"fixture"}
    )
    monkeypatch.setattr(eav_module, "BLOCK_CACHE_CLASS_SHA256S", {"fixture"})
    monkeypatch.setattr(eav_module, "BLOCK_CACHE_PATCH_CALL_SHA256S", {"fixture"})
    monkeypatch.setattr(
        eav_module, "BLOCK_CACHE_CONFIG_CLASS_SHA256S", {"fixture"}
    )


def _allow_prompt_relay_fixture_core(monkeypatch):
    monkeypatch.setattr(prompt_relay_module, "_source_sha256", lambda _source: "fixture")
    monkeypatch.setattr(prompt_relay_module, "ATTENTION_FORWARD_SHA256S", {"fixture"})
    monkeypatch.setattr(prompt_relay_module, "PACKED_LAYOUT_SHA256S", {"fixture"})
    monkeypatch.setattr(prompt_relay_module, "TOKENIZER_SHA256S", {"fixture"})
    monkeypatch.setattr(prompt_relay_module, "EXTRA_CONDS_SHA256S", {"fixture"})


def _relay_binding(*, task="t2va", query_route="video_only_paper"):
    binding = {
        "schema": prompt_relay_module.PROMPT_RELAY_PATCH_VERSION,
        "plan_hash": "plan-fixture",
        "compiled_prompt_sha256": "prompt-fixture",
        "text_len": 4,
        "prompt_token_count": 4,
        "prompt_token_sha256": "tokens-fixture",
        "events": [
            {
                "event_index": 0,
                "text_key_start": 0,
                "text_key_end": 2,
                "midpoint": 1.0,
                "window": 1.0,
                "sigma": 1.0,
            },
            {
                "event_index": 1,
                "text_key_start": 2,
                "text_key_end": 4,
                "midpoint": 4.0,
                "window": 1.0,
                "sigma": 1.0,
            },
        ],
        "query_route": query_route,
        "task": task,
        "keyframe_count": 0,
        "reference_block_count": 0,
        "layout_contract": {
            "schema": 1,
            "signature": [4, 3, 2, 2, 1],
            "seq_len": 12,
            "segments": [[0, 4, "text"], [4, 6, "audio"], [6, 12, "video"]],
            "position_shape": [12, 3],
            "position_dtype": "torch.int64",
            "position_sha256": "positions-fixture",
            "contract_hash": "layout-fixture",
        },
    }
    binding["binding_hash"] = prompt_relay_module._sha256_json(binding)
    return binding


def _relay_model(monkeypatch, *, task="t2va", query_route="video_only_paper"):
    _allow_prompt_relay_fixture_core(monkeypatch)
    return patch_prompt_relay_model(
        _model_patcher(),
        _relay_binding(task=task, query_route=query_route),
        query_chunk_rows=64,
    )[0]


def _stock20_sigmas():
    return torch.cat((torch.linspace(1.0, 0.05, 20), torch.zeros(1)))


def _turbo8_sigmas():
    return torch.cat((torch.linspace(1.0, 0.08, 8), torch.zeros(1)))


def _add_alpha8_bypass(model, *, hook_count=208, strength=1.0):
    manager = BypassInjectionManager()
    manager.hooks = [
        SimpleNamespace(multiplier=strength, module=object()) for _ in range(hook_count)
    ]

    def inject_all(_model_patcher):
        return len(manager.hooks)

    def eject_all(_model_patcher):
        return len(manager.hooks)

    model.set_injections(
        "bypass_lora", [PatcherInjection(inject=inject_all, eject=eject_all)]
    )
    return model


class _FixtureBlockCacheConfig:
    def __init__(self, *, cache_device="cpu"):
        self.residual_diff_threshold = 0.12
        self.start_percent = 0.08
        self.end_percent = 0.95
        self.max_consecutive_hits = 2
        self.cache_device = cache_device
        self.metric_stride = 8
        self.verbose = False


class _FixtureBlockCache:
    def __init__(self, *, cache_device="cpu", decision="full"):
        self.config = _FixtureBlockCacheConfig(cache_device=cache_device)
        self.total_blocks = 50
        self.total_forwards = 0
        self.full_forwards = 0
        self.cache_hits = 0
        self.decision = decision

    def clone(self):
        return _FixtureBlockCache(
            cache_device=self.config.cache_device, decision=self.decision
        )


class _FixtureBlockPatch:
    def __init__(self, block_index):
        self.block_index = block_index

    def __call__(self, args, extra_options):
        return extra_options["original_block"](args)


def _fixture_block_cache_outer(executor, *args, **kwargs):
    return executor(*args, **kwargs)


def _fixture_block_cache_diffusion(executor, *args, **kwargs):
    runtime_cache = args[3][eav_module.BLOCK_CACHE_KEY]
    runtime_cache.total_forwards += 1
    if runtime_cache.decision == "hit":
        runtime_cache.cache_hits += 1
    else:
        runtime_cache.full_forwards += 1
    return executor(*args, **kwargs)


def _block_cache_model(monkeypatch, *, cache_device="cpu"):
    _allow_fixture_core(monkeypatch)
    model = _model_patcher()
    transformer = model.model_options["transformer_options"].copy()
    transformer[eav_module.BLOCK_CACHE_KEY] = _FixtureBlockCache(
        cache_device=cache_device
    )
    model.model_options["transformer_options"] = transformer
    model.set_model_patch_replace(
        _FixtureBlockPatch(0), "dit", "double_block", 0
    )
    model.set_model_patch_replace(
        _FixtureBlockPatch(49), "dit", "double_block", 49
    )
    model.add_wrapper_with_key(
        eav_module.comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        eav_module.BLOCK_CACHE_WRAPPER_KEY,
        _fixture_block_cache_outer,
    )
    model.add_wrapper_with_key(
        eav_module.comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        eav_module.BLOCK_CACHE_WRAPPER_KEY,
        _fixture_block_cache_diffusion,
    )
    return model


def _dense_off_diagonal_cfi(q, k, frames, spatial_tokens):
    _, heads, _, dim = q.shape
    q_grid = q[0].reshape(heads, frames, spatial_tokens, dim).permute(2, 0, 1, 3)
    k_grid = k[0].reshape(heads, frames, spatial_tokens, dim).permute(2, 0, 1, 3)
    attention = torch.matmul(q_grid * (dim**-0.5), k_grid.transpose(-2, -1))
    attention = attention.to(torch.float32).softmax(dim=-1)
    diagonal = torch.eye(frames, dtype=torch.bool)[None, None]
    return attention.masked_fill(diagonal, 0).sum() / (
        spatial_tokens * heads * frames * (frames - 1)
    )


def test_exact_chunked_cfi_matches_dense_off_diagonal_formula():
    torch.manual_seed(41)
    frames, spatial, heads, dim = 7, 11, 3, 8
    q = torch.randn((1, heads, frames * spatial, dim))
    k = torch.randn_like(q)
    expected = _dense_off_diagonal_cfi(q, k, frames, spatial)
    actual, chunk, workspace = exact_chunked_cfi(
        q,
        k,
        frames=frames,
        spatial_tokens=spatial,
        max_workspace_mib=4,
    )
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert 1 <= chunk <= spatial
    assert workspace <= 4 * 1024 * 1024


def test_chunk_size_does_not_change_cfi():
    torch.manual_seed(42)
    frames, spatial, heads, dim = 9, 97, 4, 16
    q = torch.randn((1, heads, frames * spatial, dim))
    k = torch.randn_like(q)
    small, *_ = exact_chunked_cfi(
        q, k, frames=frames, spatial_tokens=spatial, max_workspace_mib=4
    )
    large, *_ = exact_chunked_cfi(
        q, k, frames=frames, spatial_tokens=spatial, max_workspace_mib=64
    )
    assert torch.allclose(small, large, atol=1e-7, rtol=1e-6)


def test_time_major_reshape_tracks_each_spatial_position_across_frames():
    frames, spatial, heads, dim = 3, 2, 1, 2
    q = torch.zeros((1, heads, frames * spatial, dim))
    k = torch.zeros_like(q)
    # Time-major rows are [t0s0,t0s1,t1s0,t1s1,t2s0,t2s1].
    for time in range(frames):
        q[0, 0, time * spatial + 0] = torch.tensor([float(time + 1), 0.0])
        k[0, 0, time * spatial + 0] = torch.tensor([float(time + 1), 0.0])
        q[0, 0, time * spatial + 1] = torch.tensor([0.0, float(time + 1)])
        k[0, 0, time * spatial + 1] = torch.tensor([0.0, float(time + 1)])
    actual, *_ = exact_chunked_cfi(
        q, k, frames=frames, spatial_tokens=spatial, max_workspace_mib=4
    )
    expected = _dense_off_diagonal_cfi(q, k, frames, spatial)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)


def test_attention_router_scales_only_target_video_rows(monkeypatch):
    seq_len, audio_start, audio_end, video_start, video_end = 12, 4, 6, 6, 12
    frames, spatial, heads, dim = 3, 2, 1, 2
    q = torch.randn((1, heads, seq_len, dim))
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    baseline = torch.arange(seq_len * 4, dtype=torch.float32).reshape(1, seq_len, 4)

    delegated = []

    def delegate(*_args, **_kwargs):
        value = baseline.clone()
        delegated.append(value)
        return value

    monkeypatch.setattr(eav_module.attention_module, "optimized_attention", delegate)
    runtime = EAVRuntime({"mode": "apply_exp"})
    route = {
        "seq_len": seq_len,
        "audio_start": audio_start,
        "audio_end": audio_end,
        "video_start": video_start,
        "video_end": video_end,
        "frames": frames,
        "spatial_tokens": spatial,
        "active": True,
        "max_workspace_mib": 4,
        "tau": 4.0,
        "g_hard_limit": 3.0,
        "runtime": runtime,
        "forward_index": runtime.begin_forward(
            sigma_video=0.5,
            progress_video=0.5,
            route={
                "active": True,
                "frames": frames,
                "spatial_tokens": spatial,
                "seq_len": seq_len,
                "audio_start": audio_start,
                "audio_end": audio_end,
                "video_start": video_start,
                "video_end": video_end,
            },
        ),
        "mode": "apply_exp",
    }
    output = route_eav_attention(
        q,
        k,
        v,
        heads,
        skip_reshape=True,
        transformer_options={eav_module.EAV_RUNTIME_KEY: route},
    )
    assert output is delegated[0]
    assert torch.equal(output[:, :video_start], baseline[:, :video_start])
    assert torch.equal(output[:, audio_start:audio_end], baseline[:, audio_start:audio_end])
    assert not torch.equal(output[:, video_start:video_end], baseline[:, video_start:video_end])


def test_report_only_is_output_identity(monkeypatch):
    seq_len, frames, spatial, heads, dim = 12, 3, 2, 1, 2
    q = torch.randn((1, heads, seq_len, dim))
    k = torch.randn_like(q)
    baseline = torch.randn((1, seq_len, 4))
    monkeypatch.setattr(
        eav_module.attention_module,
        "optimized_attention",
        lambda *_args, **_kwargs: baseline,
    )
    runtime = EAVRuntime({"mode": "report_only"})
    base_route = {
        "active": True,
        "frames": frames,
        "spatial_tokens": spatial,
        "seq_len": seq_len,
        "audio_start": 4,
        "audio_end": 6,
        "video_start": 6,
        "video_end": 12,
    }
    route = {
        **base_route,
        "max_workspace_mib": 4,
        "tau": 4.0,
        "g_hard_limit": 3.0,
        "runtime": runtime,
        "forward_index": runtime.begin_forward(
            sigma_video=0.5, progress_video=0.5, route=base_route
        ),
        "mode": "report_only",
    }
    output = route_eav_attention(
        q,
        k,
        torch.randn_like(q),
        heads,
        skip_reshape=True,
        transformer_options={eav_module.EAV_RUNTIME_KEY: route},
    )
    assert output is baseline


def test_strict_sage_router_is_authoritative_and_audited(monkeypatch):
    seq_len, frames, spatial, heads, dim = 12, 3, 2, 1, 2
    q = torch.randn((1, heads, seq_len, dim))
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    baseline = torch.randn((1, seq_len, heads * dim))
    calls = []

    def strict_backend(*_args, **kwargs):
        calls.append(kwargs)
        return baseline

    monkeypatch.setattr(eav_module, "_strict_sage_attention", strict_backend)
    monkeypatch.setattr(
        eav_module.attention_module,
        "optimized_attention",
        lambda *_args, **_kwargs: pytest.fail("native backend must not run"),
    )
    runtime = EAVRuntime({"mode": "report_only", "attention_backend": "strict_sage_hnd"})
    base_route = {
        "active": True,
        "frames": frames,
        "spatial_tokens": spatial,
        "seq_len": seq_len,
        "audio_start": 4,
        "audio_end": 6,
        "video_start": 6,
        "video_end": 12,
    }
    route = {
        **base_route,
        "attention_backend": "strict_sage_hnd",
        "max_workspace_mib": 4,
        "tau": 4.0,
        "g_hard_limit": 3.0,
        "runtime": runtime,
        "forward_index": runtime.begin_forward(
            sigma_video=0.5, progress_video=0.5, route=base_route
        ),
        "mode": "report_only",
    }
    output = route_eav_attention(
        q,
        k,
        v,
        heads,
        skip_reshape=True,
        transformer_options={eav_module.EAV_RUNTIME_KEY: route},
    )
    assert output is baseline
    assert len(calls) == 1
    report = runtime.snapshot(consume=False)
    assert report["strict_sage_call_count"] == 1
    assert report["strict_sage_calls_per_forward"] == [1]
    assert report["strict_sage_failure_count"] == 0
    assert report["strict_sage_fallback_count"] == 0


def test_strict_sage_rejects_non_native_tensor_contract_before_kernel():
    q = torch.zeros((1, 1, 4, 8), dtype=torch.float16)
    with pytest.raises(RuntimeError, match="one CUDA device"):
        eav_module._strict_sage_attention(
            q,
            q,
            q,
            1,
            mask=None,
            skip_reshape=True,
            skip_output_reshape=False,
        )
    with pytest.raises(RuntimeError, match="HND input/output"):
        eav_module._strict_sage_attention(
            q,
            q,
            q,
            1,
            mask=None,
            skip_reshape=False,
            skip_output_reshape=False,
        )


def test_strict_sage_architecture_guard_blocks_sm120_high_token_profile():
    eav_module._strict_sage_architecture_guard(49_999, (12, 0))
    eav_module._strict_sage_architecture_guard(200_000, (8, 9))

    with pytest.raises(RuntimeError, match="pure-noise output failure"):
        eav_module._strict_sage_architecture_guard(50_000, (12, 0))


def test_runtime_route_accepts_stable_visual_tasks_and_uses_final_video_grid():
    text_len, frames, height, width, audio_t = 8, 7, 8, 10, 12
    x = [
        torch.zeros((1, 24, frames, height, width)),
        torch.zeros((1, 32, 2, audio_t)),
    ]
    latent = torch.zeros((1, 24, 1, height, width))
    tasks = {
        "T2VA": [],
        "I2VA": [{"resolved_frame_index": 0, "latent": latent}],
        "L2VA": [{"resolved_frame_index": 21, "latent": latent}],
        "FL2VA": [
            {"resolved_frame_index": 0, "latent": latent},
            {"resolved_frame_index": 21, "latent": latent},
        ],
    }
    for expected_task, keyframes in tasks.items():
        layout = build_packed_layout(
            text_len, frames, height, width, audio_t, keyframes=keyframes
        )
        route = _runtime_route(
            x=x,
            timestep=torch.tensor([500.0]),
            context=torch.zeros((1, text_len, 4)),
            payload={"layout": layout, "keyframes": keyframes, "refs": []},
            denoise_mask=None,
            audio_denoise_mask=None,
            start_progress=0.0,
            end_progress=1.0,
        )
        assert route["task"] == expected_task
        assert route["frames"] == frames
        assert route["spatial_tokens"] == (height // 2) * (width // 2)
        assert route["audio_end"] == route["video_start"]
        assert route["video_end"] == layout.seq_len

    bad_keyframes = [{"resolved_frame_index": 10, "latent": latent}]
    bad_layout = build_packed_layout(
        text_len, frames, height, width, audio_t, keyframes=bad_keyframes
    )
    with pytest.raises(RuntimeError, match="positions"):
        _runtime_route(
            x=x,
            timestep=torch.tensor([500.0]),
            context=torch.zeros((1, text_len, 4)),
            payload={"layout": bad_layout, "keyframes": bad_keyframes, "refs": []},
            denoise_mask=None,
            audio_denoise_mask=None,
            start_progress=0.0,
            end_progress=1.0,
        )

    plain_layout = build_packed_layout(text_len, frames, height, width, audio_t)
    with pytest.raises(RuntimeError, match="Ref2VA/Hybrid"):
        _runtime_route(
            x=x,
            timestep=torch.tensor([500.0]),
            context=torch.zeros((1, text_len, 4)),
            payload={"layout": plain_layout, "keyframes": [], "refs": [object()]},
            denoise_mask=None,
            audio_denoise_mask=None,
            start_progress=0.0,
            end_progress=1.0,
        )


def test_reference_segment_contract_matches_native_packed_layout():
    refs = [
        {"kind": "image", "latent_h": 6, "latent_w": 8},
        {"kind": "audio", "ref_audio_t": 5},
        {
            "kind": "video_audio",
            "latent_t": 3,
            "latent_h": 4,
            "latent_w": 6,
            "ref_audio_t": 7,
        },
    ]
    assert _reference_segment_contract(refs) == [
        ("ref_img", 12),
        ("ref_audio", 10),
        ("ref_audio", 14),
        ("ref_img", 18),
    ]


def test_reference_composer_routes_ref2va_and_hybrid_without_touching_legacy_scope():
    text_len, frames, height, width, audio_t = 8, 7, 8, 10, 12
    x = [
        torch.zeros((1, 24, frames, height, width)),
        torch.zeros((1, 32, 2, audio_t)),
    ]
    context = torch.zeros((1, text_len, 4))
    image_ref = {"kind": "image", "latent_h": 6, "latent_w": 8}

    ref_layout = build_packed_layout(
        text_len, frames, height, width, audio_t, refs=[image_ref]
    )
    ref_route = _runtime_route(
        x=x,
        timestep=torch.tensor([500.0]),
        context=context,
        payload={"layout": ref_layout, "keyframes": [], "refs": [image_ref]},
        denoise_mask=None,
        audio_denoise_mask=None,
        start_progress=0.0,
        end_progress=1.0,
        allowed_tasks=EAV_REFERENCE_TASKS,
        allow_reference_blocks=True,
    )
    assert ref_route["task"] == "Ref2VA"
    assert ref_route["reference_block_count"] == 1
    assert ref_route["audio_end"] == ref_route["video_start"]
    assert ref_route["video_end"] == ref_layout.seq_len

    keyframe = {
        "resolved_frame_index": 0,
        "latent": torch.zeros((1, 24, 1, height, width)),
    }
    hybrid_layout = build_packed_layout(
        text_len,
        frames,
        height,
        width,
        audio_t,
        keyframes=[keyframe],
        refs=[image_ref],
    )
    hybrid_route = _runtime_route(
        x=x,
        timestep=torch.tensor([500.0]),
        context=context,
        payload={
            "layout": hybrid_layout,
            "keyframes": [keyframe],
            "refs": [image_ref],
        },
        denoise_mask=None,
        audio_denoise_mask=None,
        start_progress=0.0,
        end_progress=1.0,
        allowed_tasks=EAV_REFERENCE_TASKS,
        allow_reference_blocks=True,
    )
    assert hybrid_route["task"] == "Hybrid"

    bad_layout = build_packed_layout(
        text_len,
        frames,
        height,
        width,
        audio_t,
        refs=[{"kind": "image", "latent_h": 4, "latent_w": 4}],
    )
    with pytest.raises(RuntimeError, match="segment order/sizes"):
        _runtime_route(
            x=x,
            timestep=torch.tensor([500.0]),
            context=context,
            payload={"layout": bad_layout, "keyframes": [], "refs": [image_ref]},
            denoise_mask=None,
            audio_denoise_mask=None,
            start_progress=0.0,
            end_progress=1.0,
            allowed_tasks=EAV_REFERENCE_TASKS,
            allow_reference_blocks=True,
        )


def test_reference_composer_node_is_append_only_stock20_and_disabled_is_identity():
    schema = MiniMaxH3EnhanceAVideoReferenceComposerT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert schema.is_experimental is True
    assert schema.node_id == "MiniMaxH3EnhanceAVideoReferenceComposerT8Advanced"
    assert "sampling_profile" not in inputs
    model = object()
    output = MiniMaxH3EnhanceAVideoReferenceComposerT8Advanced.execute(
        model=model,
        sigmas=_stock20_sigmas(),
        mode="disabled",
        tau=4.0,
        start_video_progress=0.0,
        end_video_progress=1.0,
        max_workspace_mib=32,
        g_hard_limit=1.5,
    )
    returned_model, runtime, report_json = output.result
    report = json.loads(report_json)
    assert returned_model is model
    assert isinstance(runtime, EAVRuntime)
    assert report["task_scope"] == ["Ref2VA", "Hybrid"]
    assert report["allow_reference_blocks"] is True
    assert report["sampling_profile"] == "stock20"


def test_strict_sage_composer_is_append_only_and_disabled_is_exact_identity(monkeypatch):
    schema = MiniMaxH3EnhanceAVideoSageComposerT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert schema.node_id == "MiniMaxH3EnhanceAVideoSageComposerT8Advanced"
    assert schema.is_experimental is True
    assert inputs["task_scope"].default == "visual"
    assert inputs["sampling_profile"].default == "stock20"

    monkeypatch.setattr(
        eav_module,
        "_strict_sage_contract",
        lambda: pytest.fail("disabled must not inspect or load Sage"),
    )
    model = object()
    output = MiniMaxH3EnhanceAVideoSageComposerT8Advanced.execute(
        model=model,
        sigmas=_stock20_sigmas(),
        task_scope="visual",
        mode="disabled",
        tau=4.0,
        start_video_progress=0.0,
        end_video_progress=1.0,
        max_workspace_mib=32,
        g_hard_limit=1.5,
        sampling_profile="stock20",
    )
    returned_model, runtime, report_json = output.result
    report = json.loads(report_json)
    assert returned_model is model
    assert isinstance(runtime, EAVRuntime)
    assert report["attention_backend"] == "strict_sage_hnd"
    assert report["task_scope"] == ["T2VA", "I2VA", "FL2VA", "L2VA"]
    assert report["notes"][0].startswith("disabled returns the original MODEL")


def test_strict_sage_composer_binds_audited_backend_and_rejects_reference_turbo(
    monkeypatch,
):
    _allow_fixture_core(monkeypatch)
    monkeypatch.setattr(
        eav_module,
        "_strict_sage_contract",
        lambda: {
            "backend": "sageattention.sageattn",
            "package_version": "fixture",
            "silent_fallback": False,
        },
    )
    output = MiniMaxH3EnhanceAVideoSageComposerT8Advanced.execute(
        model=_model_patcher(),
        sigmas=_stock20_sigmas(),
        task_scope="visual",
        mode="report_only",
        tau=4.0,
        start_video_progress=0.0,
        end_video_progress=1.0,
        max_workspace_mib=32,
        g_hard_limit=1.5,
        sampling_profile="stock20",
    )
    patched, _runtime, report_json = output.result
    report = json.loads(report_json)
    assert report["attention_backend"] == "strict_sage_hnd"
    assert report["attention_backend_contract"]["silent_fallback"] is False
    installed = patched.model_options["transformer_options"][
        "optimized_attention_override"
    ]
    assert installed._t8_h3_eav_patch_version == eav_module.EAV_PATCH_VERSION

    with pytest.raises(ValueError, match="reference scope currently requires stock20"):
        MiniMaxH3EnhanceAVideoSageComposerT8Advanced.execute(
            model=object(),
            sigmas=_turbo8_sigmas(),
            task_scope="reference",
            mode="disabled",
            tau=4.0,
            start_video_progress=0.0,
            end_video_progress=1.0,
            max_workspace_mib=32,
            g_hard_limit=1.5,
            sampling_profile="turbo8_alpha8",
        )


def test_prompt_relay_composer_schema_is_append_only_and_explains_order():
    schema = MiniMaxH3EnhanceAVideoPromptRelayComposerT8Advanced.define_schema()
    assert schema.node_id == "MiniMaxH3EnhanceAVideoPromptRelayComposerT8Advanced"
    assert [item.id for item in schema.inputs] == [
        "model",
        "sigmas",
        "mode",
        "tau",
        "start_video_progress",
        "end_video_progress",
        "max_workspace_mib",
        "g_hard_limit",
        "sampling_profile",
    ]
    assert "does not add model forwards" in schema.description


def test_prompt_relay_composer_disabled_preserves_exact_relay_model(monkeypatch):
    _allow_fixture_core(monkeypatch)
    relay_model = _relay_model(monkeypatch)
    returned, runtime, report_json = build_eav_prompt_relay_model(
        relay_model,
        _stock20_sigmas(),
        mode="disabled",
        tau=4.0,
        start_video_progress=0.0,
        end_video_progress=1.0,
        max_workspace_mib=32,
        g_hard_limit=1.5,
        sampling_profile="stock20",
    )
    assert returned is relay_model
    assert isinstance(runtime, EAVRuntime)
    report = json.loads(report_json)
    assert report["prompt_relay_contract"]["task"] == "T2VA"
    assert report["prompt_relay_contract"]["adds_model_forwards"] is False
    override = returned.model_options["transformer_options"][
        "optimized_attention_override"
    ]
    assert override._t8_prompt_relay_binding_hash == _relay_binding()["binding_hash"]


def test_prompt_relay_composer_replaces_exact_relay_owner_and_keeps_provenance(
    monkeypatch,
):
    _allow_fixture_core(monkeypatch)
    relay_model = _relay_model(monkeypatch, query_route="joint_av_exp")
    patched, runtime, report_json = build_eav_prompt_relay_model(
        relay_model,
        _stock20_sigmas(),
        mode="apply_exp",
        tau=4.0,
        start_video_progress=0.0,
        end_video_progress=1.0,
        max_workspace_mib=32,
        g_hard_limit=1.5,
        sampling_profile="stock20",
    )
    report = json.loads(report_json)
    assert report["composer_profile"] == "prompt_relay_t2va_joint_av_exp_v1"
    assert report["prompt_relay_contract"]["query_route"] == "joint_av_exp"
    assert report["prompt_relay_contract"]["event_count"] == 2
    wrapper_type = eav_module.comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    assert patched.get_wrappers(
        wrapper_type, prompt_relay_module.PROMPT_RELAY_WRAPPER_KEY
    ) == []
    assert patched.get_wrappers(wrapper_type, eav_module.EAV_WRAPPER_KEY) == []
    assert len(
        patched.get_wrappers(wrapper_type, eav_module.EAV_PROMPT_RELAY_WRAPPER_KEY)
    ) == 1
    override = patched.model_options["transformer_options"][
        "optimized_attention_override"
    ]
    assert (
        override._t8_h3_eav_prompt_relay_patch_version
        == eav_module.EAV_PROMPT_RELAY_PATCH_VERSION
    )
    assert override._t8_h3_eav_patch_version == eav_module.EAV_PATCH_VERSION
    assert override._t8_prompt_relay_binding_hash == _relay_binding(
        query_route="joint_av_exp"
    )["binding_hash"]
    assert patched.get_attachment(prompt_relay_module.PROMPT_RELAY_WRAPPER_KEY) is None
    attachment = patched.get_attachment(eav_module.EAV_PROMPT_RELAY_WRAPPER_KEY)
    assert attachment["prompt_relay"]["adds_model_forwards"] is False
    assert runtime.config["attention_backend"] == "native_optimized"


def test_prompt_relay_composer_rejects_tampered_binding_and_unaudited_turbo_task(
    monkeypatch,
):
    _allow_fixture_core(monkeypatch)
    tampered = _relay_model(monkeypatch)
    tampered.get_attachment(prompt_relay_module.PROMPT_RELAY_WRAPPER_KEY)[
        "binding"
    ]["task"] = "i2va"
    with pytest.raises(RuntimeError, match="binding hash is invalid"):
        build_eav_prompt_relay_model(
            tampered,
            _stock20_sigmas(),
            mode="report_only",
            tau=4.0,
            start_video_progress=0.0,
            end_video_progress=1.0,
            max_workspace_mib=32,
            g_hard_limit=1.5,
            sampling_profile="stock20",
        )

    relay_i2va = _relay_model(monkeypatch, task="i2va")
    with pytest.raises(ValueError, match="limited to audited T2VA"):
        build_eav_prompt_relay_model(
            relay_i2va,
            _turbo8_sigmas(),
            mode="disabled",
            tau=4.0,
            start_video_progress=0.0,
            end_video_progress=1.0,
            max_workspace_mib=32,
            g_hard_limit=1.5,
            sampling_profile="turbo8_alpha8",
        )


def test_combined_attention_runs_relay_then_scales_only_target_video(monkeypatch):
    torch.manual_seed(901)
    seq_len, heads, dim = 12, 1, 2
    q = torch.randn((1, heads, seq_len, dim))
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def _identity_attention(query, *_args, **_kwargs):
        return query.permute(0, 2, 1, 3).reshape(1, query.shape[2], -1).clone()

    monkeypatch.setattr(
        prompt_relay_module.attention_module,
        "optimized_attention",
        _identity_attention,
    )
    monkeypatch.setattr(
        prompt_relay_module.attention_module,
        "attention_pytorch",
        _identity_attention,
    )
    runtime = EAVRuntime({"mode": "apply_exp"})
    base_route = {
        "active": True,
        "frames": 3,
        "spatial_tokens": 2,
        "seq_len": seq_len,
        "audio_start": 4,
        "audio_end": 6,
        "video_start": 6,
        "video_end": 12,
        "task": "T2VA",
    }
    forward = runtime.begin_forward(
        sigma_video=0.5,
        progress_video=0.5,
        route=base_route,
    )
    eav_route = {
        **base_route,
        "max_workspace_mib": 4,
        "tau": 4.0,
        "g_hard_limit": 3.0,
        "runtime": runtime,
        "forward_index": forward,
        "mode": "apply_exp",
    }
    query_times = torch.arange(6, dtype=torch.float32)
    relay_route = {
        "binding_hash": "fixture",
        "query_route": "video_only_paper",
        "seq_len": seq_len,
        "audio_start": 4,
        "audio_end": 6,
        "video_start": 6,
        "video_end": 12,
        "query_segments": (
            {
                "kind": "video",
                "start": 6,
                "end": 12,
                "query_times": query_times,
            },
        ),
        "events": (
            {
                "text_key_start": 0,
                "text_key_end": 2,
                "midpoint": 1.0,
                "window": 1.0,
                "sigma": 1.0,
            },
            {
                "text_key_start": 2,
                "text_key_end": 4,
                "midpoint": 4.0,
                "window": 1.0,
                "sigma": 1.0,
            },
        ),
    }
    output = route_eav_prompt_relay_attention(
        q,
        k,
        v,
        heads,
        skip_reshape=True,
        transformer_options={
            eav_module.EAV_RUNTIME_KEY: eav_route,
            prompt_relay_module.PROMPT_RELAY_RUNTIME_KEY: relay_route,
        },
        query_chunk_rows=64,
    )
    baseline = _identity_attention(q)
    assert torch.equal(output[:, :6], baseline[:, :6])
    assert not torch.equal(output[:, 6:], baseline[:, 6:])
    snapshot = runtime.snapshot(consume=False)
    assert snapshot["attention_measurement_count"] == 1
    assert snapshot["forwards"][0]["attention_count"] == 1


@pytest.mark.parametrize("task", ["Ref2VA", "Hybrid"])
@pytest.mark.parametrize("mode", ["disabled", "apply_exp"])
def test_reference_probe_prompts_are_same_seed_same_nfe_controlled_pairs(task, mode):
    graph = build_prompt(task, mode)
    conditioning = graph["5"]["inputs"]
    composer = graph["7"]
    assert conditioning["task_type"] == task
    assert [conditioning[key] for key in ("width", "height", "length")] == [1152, 640, 124]
    assert conditioning["ref_images.ref_image_0"] == ["14", 0]
    assert ("first_frame" in conditioning) is (task == "Hybrid")
    assert composer["class_type"] == "MiniMaxH3EnhanceAVideoReferenceComposerT8Advanced"
    assert composer["inputs"]["mode"] == mode
    assert graph["6"]["inputs"]["steps"] == 20
    assert graph["8"]["inputs"]["noise_seed"] == (
        2608217302 if task == "Ref2VA" else 2608217303
    )


def test_disabled_returns_exact_original_model_and_tau_zero_is_not_used_as_off_switch():
    model = object()
    returned, runtime, report_json = build_eav_model(
        model,
        _stock20_sigmas(),
        mode="disabled",
        tau=0.0,
        start_video_progress=0.0,
        end_video_progress=1.0,
        max_workspace_mib=32,
        g_hard_limit=1.5,
    )
    assert returned is model
    assert isinstance(runtime, EAVRuntime)
    report = json.loads(report_json)
    assert any("tau=0 is not an off switch" in note for note in report["notes"])


def test_equivalent_eav_core_source_text_change_is_not_a_compatibility_gate(
    monkeypatch,
):
    monkeypatch.setattr(eav_module, "_source_sha256", lambda _value: "unknown-source")
    _patched, _runtime, report_json = build_eav_model(
        _model_patcher(),
        _stock20_sigmas(),
        mode="report_only",
        tau=4.0,
        start_video_progress=0.0,
        end_video_progress=1.0,
        max_workspace_mib=32,
        g_hard_limit=1.5,
    )
    contract = json.loads(report_json)["core_contract"]
    assert contract["source_hash_policy"] == "diagnostic_only_not_a_compatibility_gate"
    assert set(contract["source_hashes"].values()) == {"unknown-source"}


def test_stock20_and_model_conflicts_fail_closed(monkeypatch):
    _validate_stock20_sigmas(_stock20_sigmas())
    with pytest.raises(ValueError, match="21 sigma"):
        _validate_stock20_sigmas(torch.linspace(1.0, 0.0, 9))

    _allow_fixture_core(monkeypatch)
    model = _model_patcher()
    model.model_options["transformer_options"]["optimized_attention_override"] = object()
    with pytest.raises(RuntimeError, match="attention override"):
        build_eav_model(
            model,
            _stock20_sigmas(),
            mode="apply_exp",
            tau=4.0,
            start_video_progress=0.0,
            end_video_progress=1.0,
            max_workspace_mib=32,
            g_hard_limit=1.5,
        )


def test_turbo8_reports_alpha8_bypass_contract_without_model_gate(monkeypatch):
    _allow_fixture_core(monkeypatch)
    model = _add_alpha8_bypass(_model_patcher())
    patched, _runtime, report_json = build_eav_model(
        model,
        _turbo8_sigmas(),
        mode="report_only",
        tau=4.0,
        start_video_progress=0.0,
        end_video_progress=1.0,
        max_workspace_mib=32,
        g_hard_limit=1.5,
        sampling_profile="turbo8_alpha8",
    )
    assert patched is not model
    report = json.loads(report_json)
    assert report["sigma_contract"]["nfe"] == 8
    assert report["turbo_contract"]["hook_count"] == 208
    assert report["turbo_contract"]["strength_min"] == pytest.approx(1.0)

    _patched, _runtime, report_json = build_eav_model(
            _add_alpha8_bypass(_model_patcher(), hook_count=207),
            _turbo8_sigmas(),
            mode="apply_exp",
            tau=4.0,
            start_video_progress=0.0,
            end_video_progress=1.0,
            max_workspace_mib=32,
            g_hard_limit=1.5,
            sampling_profile="turbo8_alpha8",
        )
    report = json.loads(report_json)
    assert report["turbo_contract"]["reference_hook_count_match"] is False
    _patched, _runtime, report_json = build_eav_model(
            _add_alpha8_bypass(_model_patcher(), strength=0.75),
            _turbo8_sigmas(),
            mode="apply_exp",
            tau=4.0,
            start_video_progress=0.0,
            end_video_progress=1.0,
            max_workspace_mib=32,
            g_hard_limit=1.5,
            sampling_profile="turbo8_alpha8",
        )
    report = json.loads(report_json)
    assert report["turbo_contract"]["reference_strength_match"] is False


def test_block_cache_composer_is_append_only_cpu_stock20_and_disabled_is_identity(
    monkeypatch,
):
    schema = MiniMaxH3EnhanceAVideoBlockCacheComposerT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert schema.node_id == "MiniMaxH3EnhanceAVideoBlockCacheComposerT8Advanced"
    assert schema.is_experimental is True
    assert "sampling_profile" not in inputs
    assert inputs["mode"].default == "report_only"

    source = _block_cache_model(monkeypatch)
    returned, runtime, report_json = build_eav_block_cache_model(
        source,
        _stock20_sigmas(),
        mode="disabled",
        tau=4.0,
        start_video_progress=0.0,
        end_video_progress=1.0,
        max_workspace_mib=32,
        g_hard_limit=1.5,
    )
    assert returned is source
    assert isinstance(runtime, EAVRuntime)
    report = json.loads(report_json)
    assert report["composer_profile"] == "block_cache_visual_stock20_v1"
    assert report["block_cache_contract"]["cache_device"] == "cpu"
    assert report["block_cache_contract"]["boundary_blocks"] == [0, 49]


def test_equivalent_block_cache_source_text_change_is_not_a_compatibility_gate(
    monkeypatch,
):
    source = _block_cache_model(monkeypatch)
    monkeypatch.setattr(eav_module, "_source_sha256", lambda _value: "unknown-source")
    _patched, _runtime, report_json = build_eav_block_cache_model(
        source,
        _stock20_sigmas(),
        mode="report_only",
        tau=4.0,
        start_video_progress=0.0,
        end_video_progress=1.0,
        max_workspace_mib=32,
        g_hard_limit=1.5,
    )
    contract = json.loads(report_json)["block_cache_contract"]
    assert contract["source_hash_policy"] == "diagnostic_only_not_a_compatibility_gate"
    assert set(contract["source_hashes"].values()) == {"unknown-source"}


def test_block_cache_composer_replaces_only_diffusion_owner_and_keeps_outer_lifecycle(
    monkeypatch,
):
    source = _block_cache_model(monkeypatch)
    patched, runtime, report_json = build_eav_block_cache_model(
        source,
        _stock20_sigmas(),
        mode="apply_exp",
        tau=4.0,
        start_video_progress=0.0,
        end_video_progress=1.0,
        max_workspace_mib=32,
        g_hard_limit=1.5,
    )
    wrapper_type = eav_module.comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    outer_type = eav_module.comfy.patcher_extension.WrappersMP.OUTER_SAMPLE
    assert patched.get_wrappers(wrapper_type, eav_module.BLOCK_CACHE_WRAPPER_KEY) == []
    assert patched.get_wrappers(wrapper_type, eav_module.EAV_WRAPPER_KEY) == []
    assert len(
        patched.get_wrappers(wrapper_type, eav_module.EAV_BLOCK_CACHE_WRAPPER_KEY)
    ) == 1
    assert patched.get_wrappers(
        outer_type, eav_module.BLOCK_CACHE_WRAPPER_KEY
    ) == [_fixture_block_cache_outer]
    transformer = patched.model_options["transformer_options"]
    assert transformer[eav_module.BLOCK_CACHE_KEY] is source.model_options[
        "transformer_options"
    ][eav_module.BLOCK_CACHE_KEY]
    assert set(transformer["patches_replace"]["dit"]) == {
        ("double_block", 0),
        ("double_block", 49),
    }
    override = transformer["optimized_attention_override"]
    assert (
        override._t8_h3_eav_block_cache_patch_version
        == eav_module.EAV_BLOCK_CACHE_PATCH_VERSION
    )
    assert patched.get_attachment(eav_module.EAV_WRAPPER_KEY) is None
    attachment = patched.get_attachment(eav_module.EAV_BLOCK_CACHE_WRAPPER_KEY)
    assert attachment["block_cache"]["adds_model_forwards"] is False
    assert runtime.config["attention_backend"] == "native_optimized"
    assert json.loads(report_json)["block_cache_contract"]["total_blocks"] == 50


def test_block_cache_combined_wrapper_records_full_and_hit_transitions(monkeypatch):
    patched, runtime, _report_json = build_eav_block_cache_model(
        _block_cache_model(monkeypatch),
        _stock20_sigmas(),
        mode="report_only",
        tau=4.0,
        start_video_progress=0.0,
        end_video_progress=1.0,
        max_workspace_mib=32,
        g_hard_limit=1.5,
    )
    wrapper = patched.get_wrappers(
        eav_module.comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        eav_module.EAV_BLOCK_CACHE_WRAPPER_KEY,
    )[0]
    text_len, frames, height, width, audio_t = 8, 7, 8, 10, 12
    x = [
        torch.zeros((1, 24, frames, height, width)),
        torch.zeros((1, 32, 2, audio_t)),
    ]
    context = torch.zeros((1, text_len, 4))
    layout = build_packed_layout(text_len, frames, height, width, audio_t)
    options = create_model_options_clone(patched.model_options)["transformer_options"]
    runtime_cache = options[eav_module.BLOCK_CACHE_KEY].clone()
    options[eav_module.BLOCK_CACHE_KEY] = runtime_cache

    class _Executor:
        wrappers = [wrapper]
        class_obj = object()

        def __call__(self, *_args, **_kwargs):
            return [torch.ones(1), torch.ones(1)]

    payload = {"layout": layout, "keyframes": [], "refs": []}
    wrapper(
        _Executor(),
        x,
        torch.tensor([500.0]),
        context,
        options,
        minimax_payload=payload,
    )
    runtime_cache.decision = "hit"
    wrapper(
        _Executor(),
        x,
        torch.tensor([400.0]),
        context,
        options,
        minimax_payload=payload,
    )
    snapshot = runtime.snapshot(consume=False)
    assert [row["block_cache_decision"] for row in snapshot["forwards"]] == [
        "full",
        "hit",
    ]


def test_block_cache_composer_rejects_gpu_cache_and_additional_wrapper(monkeypatch):
    with pytest.raises(RuntimeError, match="cache_device=cpu"):
        build_eav_block_cache_model(
            _block_cache_model(monkeypatch, cache_device="gpu"),
            _stock20_sigmas(),
            mode="disabled",
            tau=4.0,
            start_video_progress=0.0,
            end_video_progress=1.0,
            max_workspace_mib=32,
            g_hard_limit=1.5,
        )

    conflict = _block_cache_model(monkeypatch)
    conflict.add_wrapper_with_key(
        eav_module.comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        "unknown",
        lambda executor, *args, **kwargs: executor(*args, **kwargs),
    )
    with pytest.raises(RuntimeError, match="additional model/sample wrappers"):
        build_eav_block_cache_model(
            conflict,
            _stock20_sigmas(),
            mode="disabled",
            tau=4.0,
            start_video_progress=0.0,
            end_video_progress=1.0,
            max_workspace_mib=32,
            g_hard_limit=1.5,
        )


def test_block_cache_runtime_audit_uses_actual_hit_miss_measurement_counts():
    runtime = EAVRuntime(
        {
            "mode": "apply_exp",
            "sampling_profile": "stock20",
            "sigma_contract": {"nfe": 4},
            "block_cache_contract": {
                "config": {"max_consecutive_hits": 2}
            },
        }
    )
    route = {
        "active": True,
        "frames": 37,
        "spatial_tokens": 299,
        "seq_len": 12000,
        "audio_start": 500,
        "audio_end": 914,
        "video_start": 914,
        "video_end": 12000,
        "task": "T2VA",
    }
    decisions = ("full", "hit", "hit", "full")
    for index, decision in enumerate(decisions):
        forward = runtime.begin_forward(
            sigma_video=1.0 - index / 4,
            progress_video=index / 4,
            route=route,
        )
        runtime.record_block_cache_decision(forward, decision)
        for _ in range(1 if decision == "hit" else 50):
            runtime.record(forward, g=1.1, cfi=0.02, chunk_rows=8, workspace=1024)
    latent = {"samples": torch.zeros(1)}
    returned, report_json = finalize_eav_runtime(latent, runtime)
    assert returned is latent
    report = json.loads(report_json)
    assert report["status"] == "apply_exp_block_cache_verified"
    assert report["block_cache"]["cache_hits"] == 2
    assert report["block_cache"]["full_forwards"] == 2
    assert report["attention_calls_per_active_forward"] == [50, 1, 1, 50]
    assert report["attention_measurement_count"] == 102


def test_stg_composer_schema_is_append_only_and_conservative():
    schema = MiniMaxH3EnhanceAVideoSTGComposerT8Advanced.define_schema()
    inputs = {item.id: item for item in schema.inputs}
    assert schema.node_id == "MiniMaxH3EnhanceAVideoSTGComposerT8Advanced"
    assert inputs["mode"].default == "report_only"
    assert inputs["stg_scale"].default == pytest.approx(0.35)
    assert inputs["stg_double_blocks"].default == "25"
    assert inputs["stg_start_progress"].default == pytest.approx(0.25)
    assert inputs["stg_end_progress"].default == pytest.approx(0.85)
    assert inputs["shift_video"].default == pytest.approx(12.0)
    assert inputs["rescale"].default == pytest.approx(0.0)


def test_stg_composer_owns_one_eav_wrapper_and_one_post_cfg_hook(monkeypatch):
    _allow_fixture_core(monkeypatch)
    monkeypatch.setattr(detail_module, "MiniMaxH3Model", MiniMaxH3Model)
    source = _model_patcher()
    patched, runtime, report_json = build_eav_stg_model(
        source,
        _stock20_sigmas(),
        mode="report_only",
        tau=4.0,
        start_video_progress=0.0,
        end_video_progress=1.0,
        max_workspace_mib=32,
        g_hard_limit=1.5,
        stg_scale=0.35,
        stg_double_blocks="25",
        stg_start_progress=0.25,
        stg_end_progress=0.85,
        shift_video=12.0,
        rescale=0.0,
    )
    wrapper_type = eav_module.comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    assert not patched.get_wrappers(wrapper_type, eav_module.EAV_WRAPPER_KEY)
    assert len(patched.get_wrappers(wrapper_type, eav_module.EAV_STG_WRAPPER_KEY)) == 1
    assert patched.model_options.get("sampler_post_cfg_function")
    report = json.loads(report_json)
    contract = report["stg_contract"]
    assert contract["base_nfe"] == 20
    assert contract["expected_weak_forwards"] > 0
    assert contract["expected_total_forwards"] == 20 + contract["expected_weak_forwards"]
    assert contract["weak_feta_measurements_when_active"] == 49
    assert runtime.config["composer_profile"] == "stg_visual_stock20_v1"


def test_plain_eav_rejects_existing_post_cfg_guidance(monkeypatch):
    _allow_fixture_core(monkeypatch)
    source = _model_patcher()
    source.set_model_sampler_post_cfg_function(lambda args: args["denoised"])
    with pytest.raises(RuntimeError, match="sampler_post_cfg_function"):
        build_eav_model(
            source,
            _stock20_sigmas(),
            mode="report_only",
            tau=4.0,
            start_video_progress=0.0,
            end_video_progress=1.0,
            max_workspace_mib=32,
            g_hard_limit=1.5,
        )


def test_stg_runtime_audit_requires_exact_main_weak_sequence_and_counts():
    expected_branches = ["main", "main", "stg_weak", "main", "stg_weak", "main"]
    runtime = EAVRuntime(
        {
            "mode": "apply_exp",
            "sampling_profile": "stock20",
            "sigma_contract": {"nfe": 4},
            "stg_contract": {
                "applied": True,
                "base_nfe": 4,
                "expected_weak_forwards": 2,
                "expected_total_forwards": 6,
                "expected_branches": expected_branches,
                "double_blocks": [25],
            },
        }
    )
    route = {
        "active": True,
        "frames": 37,
        "spatial_tokens": 299,
        "seq_len": 12000,
        "audio_start": 500,
        "audio_end": 914,
        "video_start": 914,
        "video_end": 12000,
        "task": "T2VA",
    }
    for index, branch in enumerate(expected_branches):
        skipped = (25,) if branch == "stg_weak" else ()
        forward = runtime.begin_forward(
            sigma_video=1.0 - index / len(expected_branches),
            progress_video=index / len(expected_branches),
            route=route,
            branch=branch,
            skipped_blocks=skipped,
        )
        for _ in range(49 if branch == "stg_weak" else 50):
            runtime.record(forward, g=1.1, cfi=0.02, chunk_rows=8, workspace=1024)
    latent = {"samples": torch.zeros(1)}
    returned, report_json = finalize_eav_runtime(latent, runtime)
    assert returned is latent
    report = json.loads(report_json)
    assert report["status"] == "apply_exp_stg_verified"
    assert report["stg"] == {
        "base_nfe": 4,
        "weak_forwards": 2,
        "total_joint_av_forwards": 6,
        "skipped_double_blocks": [25],
        "active_main_measurements": 50,
        "active_weak_measurements": 49,
        "eav_applied_to_main_and_weak": True,
    }


def test_long_video_composer_is_append_only_stock20_and_segment_bound(monkeypatch):
    schema = MiniMaxH3EnhanceAVideoLongVideoComposerT8Advanced.define_schema()
    assert schema.node_id == "MiniMaxH3EnhanceAVideoLongVideoComposerT8Advanced"
    assert [item.id for item in schema.inputs[:4]] == [
        "model",
        "sigmas",
        "segment_index",
        "context_frames",
    ]
    _allow_fixture_core(monkeypatch)
    source = patch_long_video_model(_model_patcher())
    patched, runtime, report_json = build_eav_long_video_model(
        source,
        _stock20_sigmas(),
        segment_index=1,
        context_frames=22,
        mode="report_only",
        tau=4.0,
        start_video_progress=0.0,
        end_video_progress=1.0,
        max_workspace_mib=32,
        g_hard_limit=1.5,
    )
    wrapper_type = eav_module.comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    assert not patched.get_wrappers(wrapper_type, eav_module.EAV_WRAPPER_KEY)
    assert len(
        patched.get_wrappers(wrapper_type, eav_module.EAV_LONG_VIDEO_WRAPPER_KEY)
    ) == 1
    assert "extra_conds" in patched.object_patches
    report = json.loads(report_json)
    contract = report["long_video_contract"]
    assert contract["segment_index"] == 1
    assert contract["context_frames"] == 22
    assert contract["expected_motion_latent_steps"] > 0
    assert contract["resume_scope"] == "execution_local_eav_runtime_per_segment"
    assert runtime.config["composer_profile"] == "long_video_segment_stock20_v1"


def test_long_video_composer_disabled_preserves_exact_scoped_model(monkeypatch):
    _allow_fixture_core(monkeypatch)
    source = patch_long_video_model(_model_patcher())
    returned, runtime, _report = build_eav_long_video_model(
        source,
        _stock20_sigmas(),
        segment_index=0,
        context_frames=0,
        mode="disabled",
        tau=4.0,
        start_video_progress=0.0,
        end_video_progress=1.0,
        max_workspace_mib=32,
        g_hard_limit=1.5,
    )
    assert returned is source
    _latent, report_json = finalize_eav_runtime({"samples": torch.zeros(1)}, runtime)
    assert json.loads(report_json)["status"] == "eav_disabled_long_video_passthrough"


def test_long_video_composer_rejects_unscoped_model_and_wrong_context(monkeypatch):
    _allow_fixture_core(monkeypatch)
    with pytest.raises(RuntimeError, match="Long Video Conditioning"):
        build_eav_long_video_model(
            _model_patcher(),
            _stock20_sigmas(),
            segment_index=0,
            context_frames=0,
            mode="disabled",
            tau=4.0,
            start_video_progress=0.0,
            end_video_progress=1.0,
            max_workspace_mib=32,
            g_hard_limit=1.5,
        )
    scoped = patch_long_video_model(_model_patcher())
    with pytest.raises(ValueError, match="segment 0 requires context_frames=0"):
        build_eav_long_video_model(
            scoped,
            _stock20_sigmas(),
            segment_index=0,
            context_frames=22,
            mode="disabled",
            tau=4.0,
            start_video_progress=0.0,
            end_video_progress=1.0,
            max_workspace_mib=32,
            g_hard_limit=1.5,
        )


def test_long_video_runtime_audit_is_segment_local():
    contract = {
        "segment_index": 3,
        "context_frames": 22,
        "binding_hash": "segment-three-fixture",
    }
    runtime = EAVRuntime(
        {
            "mode": "apply_exp",
            "sampling_profile": "stock20",
            "sigma_contract": {"nfe": 20},
            "long_video_contract": contract,
        }
    )
    route = {
        "active": True,
        "frames": 37,
        "spatial_tokens": 299,
        "seq_len": 12000,
        "audio_start": 500,
        "audio_end": 914,
        "video_start": 914,
        "video_end": 12000,
        "task": "LongVideoMotion",
    }
    for index in range(20):
        forward = runtime.begin_forward(
            sigma_video=1.0 - index / 20,
            progress_video=index / 20,
            route=route,
        )
        for _ in range(50):
            runtime.record(forward, g=1.1, cfi=0.02, chunk_rows=8, workspace=1024)
    _latent, report_json = finalize_eav_runtime({"samples": torch.zeros(1)}, runtime)
    report = json.loads(report_json)
    assert report["status"] == "apply_exp_long_video_segment_verified"
    assert report["long_video"] == {
        "segment_index": 3,
        "context_frames": 22,
        "binding_hash": "segment-three-fixture",
        "model_forwards": 20,
        "execution_local_runtime_consumed": True,
    }


def test_long_video_runtime_route_accepts_exact_22_frame_motion_offsets():
    text_len, latent_t, latent_h, latent_w, audio_t = 7, 37, 8, 8, 207
    offsets = step_offsets(CONTEXT_FRAME_STEPS[22])
    keyframes = [
        {
            "resolved_frame_index": 0,
            MOTION_FRAME_INDEX: offset,
            "latent": torch.zeros((1, 24, 1, latent_h, latent_w)),
        }
        for offset in offsets
    ]
    layout = build_packed_layout(
        text_len,
        latent_t,
        latent_h,
        latent_w,
        audio_t,
        keyframes=keyframes,
        refs=[],
        frame_count=124,
    )
    route = _runtime_route(
        x=[
            torch.zeros((1, 24, latent_t, latent_h, latent_w)),
            torch.zeros((1, 32, 2, audio_t)),
        ],
        timestep=torch.tensor([500.0]),
        context=torch.zeros((1, text_len, 4)),
        payload={
            "layout": layout,
            "keyframes": keyframes,
            "refs": [],
            "t8_long_video_patch_version": LONG_VIDEO_PATCH_VERSION,
        },
        denoise_mask=None,
        audio_denoise_mask=None,
        start_progress=0.0,
        end_progress=1.0,
        allowed_tasks=("LongVideoSegment0", "LongVideoMotion"),
        allow_reference_blocks=True,
        long_video_contract={"segment_index": 1, "context_frames": 22},
    )
    assert route["task"] == "LongVideoMotion"
    assert route["active"] is True
    assert route["frames"] == 37


def test_runtime_audit_requires_20_forwards_and_50_blocks_each():
    runtime = EAVRuntime(
        {
            "mode": "apply_exp",
            "sampling_profile": "stock20",
            "sigma_contract": {"nfe": 20},
        }
    )
    route = {
        "active": True,
        "frames": 37,
        "spatial_tokens": 299,
        "seq_len": 12000,
        "audio_start": 500,
        "audio_end": 914,
        "video_start": 914,
        "video_end": 12000,
        "task": "T2VA",
    }
    for index in range(20):
        forward = runtime.begin_forward(
            sigma_video=1.0 - index / 20,
            progress_video=index / 20,
            route=route,
        )
        for _ in range(50):
            runtime.record(forward, g=1.1, cfi=0.02, chunk_rows=8, workspace=1024)
    latent = {"samples": torch.zeros(1)}
    returned, report_json = finalize_eav_runtime(latent, runtime)
    assert returned is latent
    report = json.loads(report_json)
    assert report["status"] == "apply_exp_verified"
    assert report["g_min"] == pytest.approx(1.1)
    assert report["attention_measurement_count"] == 1000
    assert len(report["forwards"]) == 20
    assert report["forwards"][0]["attention_count"] == 50
    assert report["forwards"][0]["task"] == "T2VA"
    assert "g_values" not in report["forwards"][0]


def test_runtime_audit_requires_all_strict_sage_calls_without_fallback():
    runtime = EAVRuntime(
        {
            "mode": "apply_exp",
            "sampling_profile": "stock20",
            "attention_backend": "strict_sage_hnd",
            "sigma_contract": {"nfe": 20},
        }
    )
    route = {
        "active": True,
        "frames": 37,
        "spatial_tokens": 299,
        "seq_len": 12000,
        "audio_start": 500,
        "audio_end": 914,
        "video_start": 914,
        "video_end": 12000,
        "task": "T2VA",
    }
    for index in range(20):
        forward = runtime.begin_forward(
            sigma_video=1.0 - index / 20,
            progress_video=index / 20,
            route=route,
        )
        for _ in range(50):
            runtime.record(forward, g=1.1, cfi=0.02, chunk_rows=8, workspace=1024)
            runtime.record_strict_sage_call(forward)
    _latent, report_json = finalize_eav_runtime({"samples": torch.zeros(1)}, runtime)
    report = json.loads(report_json)
    assert report["status"] == "apply_exp_verified"
    assert report["strict_sage_call_count"] == 1000
    assert report["strict_sage_calls_per_forward"] == [50] * 20
    assert report["strict_sage_fallback_count"] == 0


def test_frontend_workflow_is_stock20_t2va_opt_in_and_link_consistent():
    path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "07-motion-detail"
        / "2026-08-21_H3_Enhance_A_Video_FETA_Stock20_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    by_type = {node["type"]: node for node in workflow["nodes"]}
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert sum(node["type"] == "MarkdownNote" for node in nodes.values()) == 3
    assert "LoraLoaderBypassModelOnly" not in by_type
    assert "MiniMaxH3PromptRelayConditioningT8Advanced" not in by_type

    conditioning = by_type["MiniMaxH3AudioConditioningT8"]
    dual = by_type["MiniMaxH3DualClockSamplerT8"]
    eav = by_type["MiniMaxH3EnhanceAVideoT8Advanced"]
    audit = by_type["MiniMaxH3EnhanceAVideoAuditT8Advanced"]
    assert conditioning["widgets_values"][1:5] == [1152, 640, 124, "T2VA"]
    assert dual["widgets_values"] == [
        20,
        12.0,
        3.0,
        "dual_clock_euler",
        "native_flow",
    ]
    assert eav["widgets_values"] == [
        "apply_exp",
        4.0,
        0.0,
        1.0,
        32,
        1.5,
        "stock20",
    ]

    links = {link[0]: link for link in workflow["links"]}

    def source_for_input(node, name):
        item = next(value for value in node["inputs"] if value["name"] == name)
        link = links[item["link"]]
        return nodes[link[1]], link[2]

    assert source_for_input(eav, "model") == (dual, 0)
    assert source_for_input(eav, "sigmas") == (dual, 2)
    assert source_for_input(audit, "runtime") == (eav, 1)
    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_strict_sage_frontend_workflow_uses_one_composer_and_three_notes():
    path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "07-motion-detail"
        / "2026-08-21_H3_Enhance_A_Video_FETA_Strict_Sage_T2VA_Stock20_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    by_type = {node["type"]: node for node in workflow["nodes"]}
    composer = by_type["MiniMaxH3EnhanceAVideoSageComposerT8Advanced"]
    assert "MiniMaxH3EnhanceAVideoT8Advanced" not in by_type
    assert "MiniMaxH3MemoryEfficientSageAttentionPatch" not in by_type
    assert [item["name"] for item in composer["inputs"]] == [
        "model",
        "sigmas",
    ]
    assert composer["widgets_values"] == [
        "visual",
        "apply_exp",
        4.0,
        0.0,
        1.0,
        32,
        1.5,
        "stock20",
    ]
    assert sum(node["type"] == "MarkdownNote" for node in nodes.values()) == 3
    notes = "\n".join(
        node["widgets_values"]
        for node in nodes.values()
        if node["type"] == "MarkdownNote"
    )
    assert "failure=0" in notes
    assert "fallback=0" in notes
    assert workflow["extra"]["t8_enhance_a_video"]["quality_status"].endswith(
        "claims_false"
    )


def test_strict_sage_real_probe_is_same_seed_canvas_and_nfe_as_existing_t2va_pair():
    graph = build_sage_prompt()
    conditioning = graph["5"]["inputs"]
    composer = graph["7"]
    assert [conditioning[key] for key in ("width", "height", "length")] == [
        1152,
        640,
        124,
    ]
    assert conditioning["task_type"] == "T2VA"
    assert graph["6"]["inputs"]["steps"] == 20
    assert graph["8"]["inputs"]["noise_seed"] == 2608217001
    assert composer["class_type"] == "MiniMaxH3EnhanceAVideoSageComposerT8Advanced"
    assert composer["inputs"]["task_scope"] == "visual"
    assert composer["inputs"]["mode"] == "apply_exp"
    assert composer["inputs"]["sampling_profile"] == "stock20"


def test_prompt_relay_eav_frontend_workflow_has_one_owner_and_audited_handoff():
    path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "07-motion-detail"
        / "2026-08-21_H3_Enhance_A_Video_FETA_Prompt_Relay_T2VA_Stock20_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    by_type = {node["type"]: node for node in workflow["nodes"]}
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert sum(node["type"] == "MarkdownNote" for node in nodes.values()) == 3
    assert "MiniMaxH3EnhanceAVideoT8Advanced" not in by_type
    assert "MiniMaxH3EnhanceAVideoSageComposerT8Advanced" not in by_type
    assert "LoraLoaderBypassModelOnly" not in by_type
    relay = by_type["MiniMaxH3PromptRelayConditioningT8Advanced"]
    dual = by_type["MiniMaxH3DualClockSamplerT8"]
    composer = by_type["MiniMaxH3EnhanceAVideoPromptRelayComposerT8Advanced"]
    audit = by_type["MiniMaxH3EnhanceAVideoAuditT8Advanced"]
    guider = by_type["BasicGuider"]
    sampler = by_type["SamplerCustomAdvanced"]
    decode = by_type["MiniMaxH3AVDecodeT8"]
    assert relay["widgets_values"][-2:] == ["apply_exp", 256]
    assert dual["widgets_values"] == [
        20,
        12,
        3,
        "dual_clock_euler",
        "native_flow",
    ]
    assert composer["widgets_values"] == [
        "apply_exp",
        4.0,
        0.0,
        1.0,
        32,
        1.5,
        "stock20",
    ]
    links = {link[0]: link for link in workflow["links"]}

    def source_for_input(node, name):
        item = next(value for value in node["inputs"] if value["name"] == name)
        link = links[item["link"]]
        return nodes[link[1]], link[2]

    assert source_for_input(dual, "model") == (relay, 0)
    assert source_for_input(composer, "model") == (dual, 0)
    assert source_for_input(composer, "sigmas") == (dual, 2)
    assert source_for_input(guider, "model") == (composer, 0)
    assert source_for_input(audit, "av_latent") == (sampler, 0)
    assert source_for_input(audit, "runtime") == (composer, 1)
    assert source_for_input(decode, "av_latent") == (audit, 0)
    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_block_cache_eav_frontend_workflow_has_one_owner_and_audited_handoff():
    path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "07-motion-detail"
        / "2026-08-22_H3_Enhance_A_Video_FETA_BlockCache_T2VA_Stock20_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    by_type = {node["type"]: node for node in workflow["nodes"]}
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert sum(node["type"] == "MarkdownNote" for node in nodes.values()) == 3
    assert "MiniMaxH3EnhanceAVideoT8Advanced" not in by_type
    assert "MiniMaxH3EnhanceAVideoSageComposerT8Advanced" not in by_type
    assert "MiniMaxH3EnhanceAVideoPromptRelayComposerT8Advanced" not in by_type

    unet = by_type["UNETLoader"]
    cache = by_type["MiniMaxH3BlockCacheT8"]
    dual = by_type["MiniMaxH3DualClockSamplerT8"]
    composer = by_type["MiniMaxH3EnhanceAVideoBlockCacheComposerT8Advanced"]
    guider = by_type["BasicGuider"]
    sampler = by_type["SamplerCustomAdvanced"]
    audit = by_type["MiniMaxH3EnhanceAVideoAuditT8Advanced"]
    decode = by_type["MiniMaxH3AVDecodeT8"]
    assert cache["widgets_values"] == [0.08, 0.08, 0.95, 2, "cpu", 8, False]
    assert composer["widgets_values"] == ["apply_exp", 4.0, 0.0, 1.0, 32, 1.5]
    assert [item["name"] for item in composer["inputs"]] == [
        "model",
        "sigmas",
    ]
    links = {link[0]: link for link in workflow["links"]}

    def source_for_input(node, name):
        item = next(value for value in node["inputs"] if value["name"] == name)
        link = links[item["link"]]
        return nodes[link[1]], link[2]

    assert source_for_input(cache, "model") == (unet, 0)
    assert source_for_input(dual, "model") == (cache, 0)
    assert source_for_input(composer, "model") == (dual, 0)
    assert source_for_input(composer, "sigmas") == (dual, 2)
    assert source_for_input(guider, "model") == (composer, 0)
    assert source_for_input(audit, "av_latent") == (sampler, 0)
    assert source_for_input(audit, "runtime") == (composer, 1)
    assert source_for_input(decode, "av_latent") == (audit, 0)

    notes = "\n".join(
        node["widgets_values"]
        for node in nodes.values()
        if node["type"] == "MarkdownNote"
    )
    assert "UNET → BlockCache → DualClock → EAV+BlockCache Composer" in notes
    assert "full前向记录50次FETA测量" in notes
    assert "cache hit只记录" in notes
    assert "不宣称提速、提质、音频非劣或16GB显存安全" in notes
    report = workflow["extra"]["t8_enhance_a_video_block_cache"]
    assert report["validation_status"] == "deterministic_low_load_contract_pass"
    assert report["quality_claim"] is False
    assert report["audio_noninferiority_claim"] is False
    assert report["performance_claim"] is False
    assert report["memory_safe_claim"] is False

    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_stg_eav_frontend_workflow_has_one_owner_and_exact_branch_audit():
    path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "07-motion-detail"
        / "2026-08-22_H3_Enhance_A_Video_FETA_STG_T2VA_Stock20_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    by_type = {node["type"]: node for node in workflow["nodes"]}
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert sum(node["type"] == "MarkdownNote" for node in nodes.values()) == 3
    assert "MiniMaxH3EnhanceAVideoT8Advanced" not in by_type
    assert "MiniMaxH3SpatioTemporalGuidanceT8Advanced" not in by_type
    assert "MiniMaxH3EnhanceAVideoBlockCacheComposerT8Advanced" not in by_type

    dual = by_type["MiniMaxH3DualClockSamplerT8"]
    composer = by_type["MiniMaxH3EnhanceAVideoSTGComposerT8Advanced"]
    guider = by_type["BasicGuider"]
    sampler = by_type["SamplerCustomAdvanced"]
    audit = by_type["MiniMaxH3EnhanceAVideoAuditT8Advanced"]
    decode = by_type["MiniMaxH3AVDecodeT8"]
    assert dual["widgets_values"] == [
        20,
        12.0,
        3.0,
        "dual_clock_euler",
        "native_flow",
    ]
    assert composer["widgets_values"] == [
        "apply_exp",
        4.0,
        0.0,
        1.0,
        32,
        1.5,
        0.35,
        "25",
        0.25,
        0.85,
        12.0,
        0.0,
    ]
    links = {link[0]: link for link in workflow["links"]}

    def source_for_input(node, name):
        item = next(value for value in node["inputs"] if value["name"] == name)
        link = links[item["link"]]
        return nodes[link[1]], link[2]

    assert source_for_input(composer, "model") == (dual, 0)
    assert source_for_input(composer, "sigmas") == (dual, 2)
    assert source_for_input(guider, "model") == (composer, 0)
    assert source_for_input(audit, "av_latent") == (sampler, 0)
    assert source_for_input(audit, "runtime") == (composer, 1)
    assert source_for_input(decode, "av_latent") == (audit, 0)
    notes = "\n".join(
        node["widgets_values"]
        for node in nodes.values()
        if node["type"] == "MarkdownNote"
    )
    assert "主分支执行50块" in notes
    assert "弱分支执行49块" in notes
    assert "不宣称提质、音频非劣、提速、省显存或通用16GB安全" in notes
    report = workflow["extra"]["t8_enhance_a_video_stg"]
    assert report["validation_status"] == "deterministic_low_load_contract_pass"
    assert report["quality_claim"] is False
    assert report["audio_noninferiority_claim"] is False
    assert report["performance_claim"] is False
    assert report["memory_safe_claim"] is False

    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_long_video_eav_frontend_workflow_is_stock20_segment_bound_and_audited():
    path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "04-long-video"
        / "2026-08-22_H3_Enhance_A_Video_Long_Video_Accepted_22F_Stock20_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    by_type = {node["type"]: node for node in workflow["nodes"]}
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert sum(node["type"] == "MarkdownNote" for node in nodes.values()) == 3
    assert "LoraLoaderBypassModelOnly" not in by_type
    assert "MiniMaxH3EnhanceAVideoT8Advanced" not in by_type

    unet = by_type["UNETLoader"]
    planner = by_type["MiniMaxH3LongVideoPlannerT8"]
    conditioning = by_type["MiniMaxH3LongVideoConditioningT8"]
    dual = by_type["MiniMaxH3DualClockSamplerT8"]
    composer = by_type["MiniMaxH3EnhanceAVideoLongVideoComposerT8Advanced"]
    guider = by_type["BasicGuider"]
    sampler = by_type["SamplerCustomAdvanced"]
    audit = by_type["MiniMaxH3EnhanceAVideoAuditT8Advanced"]
    decode = by_type["MiniMaxH3AVDecodeT8"]
    assert dual["widgets_values"] == [
        20,
        12.0,
        3.0,
        "dual_clock_euler",
        "native_flow",
    ]
    assert composer["widgets_values"] == ["apply_exp", 4.0, 0.0, 1.0, 32, 1.5]
    links = {link[0]: link for link in workflow["links"]}

    def source_for_input(node, name):
        item = next(value for value in node["inputs"] if value["name"] == name)
        link = links[item["link"]]
        return nodes[link[1]], link[2]

    assert source_for_input(conditioning, "model") == (unet, 0)
    assert source_for_input(composer, "model") == (dual, 0)
    assert source_for_input(composer, "sigmas") == (dual, 2)
    assert source_for_input(composer, "segment_index") == (planner, 1)
    assert source_for_input(composer, "context_frames") == (planner, 3)
    assert source_for_input(guider, "model") == (composer, 0)
    assert source_for_input(audit, "av_latent") == (sampler, 0)
    assert source_for_input(audit, "runtime") == (composer, 1)
    assert source_for_input(decode, "av_latent") == (audit, 0)
    report = workflow["extra"]["t8_enhance_a_video_long_video"]
    assert report["validation_status"] == "deterministic_low_load_contract_pass"
    assert report["context_frames"] == [0, 5, 22, 39]
    assert report["quality_claim"] is False
    assert report["audio_noninferiority_claim"] is False
    assert report["performance_claim"] is False
    assert report["memory_safe_claim"] is False

    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


@pytest.mark.parametrize(
    ("filename", "task", "profile", "first_image", "last_image", "steps"),
    [
        (
            "2026-08-21_H3_Enhance_A_Video_FETA_I2VA_Stock20_Advanced_EXP.json",
            "I2VA",
            "stock20",
            True,
            False,
            20,
        ),
        (
            "2026-08-21_H3_Enhance_A_Video_FETA_FL2VA_Stock20_Advanced_EXP.json",
            "FL2VA",
            "stock20",
            True,
            True,
            20,
        ),
        (
            "2026-08-21_H3_Enhance_A_Video_FETA_L2VA_Stock20_Advanced_EXP.json",
            "L2VA",
            "stock20",
            False,
            True,
            20,
        ),
        (
            "2026-08-21_H3_Enhance_A_Video_FETA_T2VA_Turbo8_Advanced_EXP.json",
            "T2VA",
            "turbo8_alpha8",
            False,
            False,
            8,
        ),
    ],
)
def test_extended_eav_workflows_are_importable_and_strictly_wired(
    filename, task, profile, first_image, last_image, steps
):
    path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "07-motion-detail"
        / filename
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    by_type = {}
    for node in workflow["nodes"]:
        by_type.setdefault(node["type"], []).append(node)
    links = {link[0]: link for link in workflow["links"]}
    conditioning = by_type["MiniMaxH3AudioConditioningT8"][0]
    dual = by_type["MiniMaxH3DualClockSamplerT8"][0]
    eav = by_type["MiniMaxH3EnhanceAVideoT8Advanced"][0]
    assert conditioning["widgets_values"][1:5] == [1152, 640, 124, task]
    assert dual["widgets_values"] == [
        steps,
        12.0,
        3.0,
        "dual_clock_euler",
        "native_flow",
    ]
    assert eav["widgets_values"][-1] == profile
    assert len(by_type.get("LoadImage", [])) == int(first_image) + int(last_image)
    conditioning_inputs = {item["name"]: item for item in conditioning["inputs"]}
    assert bool(conditioning_inputs["first_frame"]["link"] is not None) is first_image
    assert bool(conditioning_inputs["last_frame"]["link"] is not None) is last_image
    assert len(by_type.get("MarkdownNote", [])) == 3

    loras = by_type.get("LoraLoaderBypassModelOnly", [])
    if profile == "turbo8_alpha8":
        assert len(loras) == 1
        assert loras[0]["widgets_values"] == [
            "minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8-T8-convert.safetensors",
            1.0,
        ]
        dual_model_link = links[dual["inputs"][0]["link"]]
        assert dual_model_link[1] == loras[0]["id"]
    else:
        assert not loras

    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


@pytest.mark.parametrize(
    ("task", "has_first"),
    [("Ref2VA", False), ("Hybrid", True)],
)
def test_reference_eav_workflows_use_the_isolated_composer(task, has_first):
    path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "07-motion-detail"
        / f"2026-08-21_H3_Enhance_A_Video_FETA_{task}_Stock20_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    by_type = {}
    for node in workflow["nodes"]:
        by_type.setdefault(node["type"], []).append(node)
    links = {link[0]: link for link in workflow["links"]}
    conditioning = by_type["MiniMaxH3AudioConditioningT8"][0]
    composer = by_type["MiniMaxH3EnhanceAVideoReferenceComposerT8Advanced"][0]
    assert "MiniMaxH3EnhanceAVideoT8Advanced" not in by_type
    assert conditioning["widgets_values"][1:5] == [1152, 640, 124, task]
    assert composer["widgets_values"] == ["apply_exp", 4.0, 0.0, 1.0, 32, 1.5]
    assert [item["name"] for item in composer["inputs"]] == ["model", "sigmas"]
    conditioning_inputs = {item["name"]: item for item in conditioning["inputs"]}
    assert bool(conditioning_inputs["first_frame"]["link"] is not None) is has_first
    assert conditioning_inputs["last_frame"]["link"] is None
    reference_input = conditioning_inputs["ref_images.ref_image_0"]
    assert reference_input["link"] is not None
    assert len(by_type["LoadImage"]) == 1 + int(has_first)
    assert len(by_type["MarkdownNote"]) == 3
    report = workflow["extra"]["t8_enhance_a_video"]
    assert "real 0.7MP A/B remains pending" in report["real_probe"]

    reference_link = links[reference_input["link"]]
    assert nodes[reference_link[1]]["type"] == "LoadImage"
    reference_slot = conditioning["inputs"].index(reference_input)
    assert reference_link[2:] == [0, conditioning["id"], reference_slot, "IMAGE"]
    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type
