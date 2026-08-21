from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

import h3_audio_t8_pkg.enhance_a_video_advanced as eav_module
from h3_audio_t8_pkg.conditioning import build_packed_layout
from h3_audio_t8_pkg.enhance_a_video_advanced import (
    EAV_REFERENCE_TASKS,
    EAVRuntime,
    _reference_segment_contract,
    _runtime_route,
    _validate_stock20_sigmas,
    build_eav_model,
    exact_chunked_cfi,
    finalize_eav_runtime,
    route_eav_attention,
)
from h3_audio_t8_pkg.nodes_enhance_a_video_advanced import (
    MiniMaxH3EnhanceAVideoReferenceComposerT8Advanced,
    MiniMaxH3EnhanceAVideoSageComposerT8Advanced,
)
from h3_audio_t8_pkg.tools.build_eav_reference_probe_prompts import build_prompt
from h3_audio_t8_pkg.tools.build_eav_sage_probe_prompt import (
    build_prompt as build_sage_prompt,
)
from comfy.model_patcher import ModelPatcher
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


def test_turbo8_requires_exact_alpha8_bypass_contract(monkeypatch):
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

    with pytest.raises(RuntimeError, match="208-module"):
        build_eav_model(
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
    with pytest.raises(RuntimeError, match="strength 1.0"):
        build_eav_model(
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
        "task_scope",
        "mode",
        "tau",
        "start_video_progress",
        "end_video_progress",
        "max_workspace_mib",
        "g_hard_limit",
        "sampling_profile",
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
    assert bool(conditioning["inputs"][17]["link"] is not None) is first_image
    assert bool(conditioning["inputs"][18]["link"] is not None) is last_image
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
    assert composer["inputs"][-1]["name"] == "g_hard_limit"
    assert bool(conditioning["inputs"][17]["link"] is not None) is has_first
    assert conditioning["inputs"][18]["link"] is None
    assert conditioning["inputs"][19]["name"] == "ref_images.ref_image_0"
    assert conditioning["inputs"][19]["link"] is not None
    assert len(by_type["LoadImage"]) == 1 + int(has_first)
    assert len(by_type["MarkdownNote"]) == 3
    report = workflow["extra"]["t8_enhance_a_video"]
    assert "real 0.7MP A/B remains pending" in report["real_probe"]

    reference_link = links[conditioning["inputs"][19]["link"]]
    assert nodes[reference_link[1]]["type"] == "LoadImage"
    assert reference_link[2:] == [0, conditioning["id"], 19, "IMAGE"]
    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type
