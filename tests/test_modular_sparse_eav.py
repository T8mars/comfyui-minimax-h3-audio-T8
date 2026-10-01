"""Native producer wiring + exact CPU FETA math, not a CUDA-kernel certificate.

Only the CUDA eligibility and final Kitchen kernel are explicit test doubles in
integration tests. Native Core VSA tiling/producer/gates/pooled-state paths run.
"""
from types import SimpleNamespace

import pytest
import torch
import comfy.model_management
import comfy.quant_ops
from comfy.ldm.minimax.model import rope_rotation_table
from comfy_extras import nodes_sparse_attention as sparse

from h3_audio_t8_pkg import enhance_a_video_advanced as feta
from h3_audio_t8_pkg.modular_sampling.sparse_eav import projected_cfi
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig, apply_stage_eav
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2 import build_stage
from test_fast_h3_v2_core_sampler import model
from test_modular_eav import sample, assert_equal
from test_progressive_sampling_runtime import latent


def statistic_fixture(frames=8, spatial=64):
    generator = torch.Generator().manual_seed(101)
    attn = model().model.diffusion_model.blocks[0].attn
    prefix, suffix = 5, 7
    hidden = torch.randn(prefix + frames * spatial + suffix, 24, generator=generator)
    angles = torch.randn(hidden.shape[0], 32, generator=generator)
    freqs = rope_rotation_table(angles, hidden.dtype)
    route = dict(frames=frames, spatial_tokens=spatial, video_start=prefix, video_end=prefix + frames * spatial,
                 max_workspace_mib=4)
    return attn, hidden, freqs, route


def reference_cfi(attn, hidden, freqs, route):
    fused = attn.qkv_proj(hidden)
    q, k, _ = fused.split(attn.heads * attn.head_dim, dim=-1)
    q = q.view(1, hidden.shape[0], attn.heads, attn.head_dim)
    k = k.view_as(q)
    comfy.quant_ops.ck.rms_rope_split_half_(q, k, freqs, attn.q_norm.weight, attn.k_norm.weight,
                                          epsilon=attn.q_norm.eps, rot_dim=freqs.shape[-3] * 2)
    a, b = route["video_start"], route["video_end"]
    return feta.exact_chunked_cfi(q.transpose(1, 2)[:, :, a:b], k.transpose(1, 2)[:, :, a:b],
                                 frames=route["frames"], spatial_tokens=route["spatial_tokens"],
                                 max_workspace_mib=4)[0]


def test_streamed_native_projection_statistic_matches_full_qk_without_full_allocation():
    attn, hidden, freqs, route = statistic_fixture()
    expected = reference_cfi(attn, hidden, freqs, route)
    projected_rows = []
    original = attn.qkv_proj

    def record(value):
        projected_rows.append(value.shape[0])
        return original(value)

    view = SimpleNamespace(**{name: getattr(attn, name) for name in ("heads", "head_dim", "q_norm", "k_norm")}, qkv_proj=record)
    actual, columns, workspace = projected_cfi(view, hidden, freqs, route)
    assert torch.allclose(actual, expected, rtol=0, atol=2e-7)
    assert columns < route["spatial_tokens"] and workspace <= 4 * 1024 * 1024
    assert max(projected_rows) <= route["frames"] * columns
    assert sum(projected_rows) == route["frames"] * route["spatial_tokens"]
    assert not torch.cuda.is_initialized()


def test_sparse_statistic_honors_cancellation_and_rejects_too_small_workspace(monkeypatch):
    attn, hidden, freqs, route = statistic_fixture()
    with pytest.raises(ValueError, match="workspace"):
        projected_cfi(attn, hidden, freqs, {**route, "max_workspace_mib": 0})

    def cancel():
        raise RuntimeError("intentional statistic cancellation")

    monkeypatch.setattr(comfy.model_management, "throw_exception_if_processing_interrupted", cancel)
    with pytest.raises(RuntimeError, match="intentional statistic cancellation"):
        projected_cfi(attn, hidden, freqs, route)


@pytest.fixture
def native_producer_cpu_kernel_double(monkeypatch):
    calls = []
    monkeypatch.setattr(sparse, "h3_eligible", lambda *args: True)

    def kernel(chunks, n, heads, freqs, norms, *, kmean=None, vscale=None, tau, topk_ratio,
               token_aug, sink_blocks, sink_q, rope_eps, tail, block_len, coarse_gate):
        # Not a sparse attention numerical reference: observe the REAL native
        # chunk producer, carrier shapes, tiling and original kernel arguments.
        projected = torch.cat(list(chunks()), dim=0)
        assert projected.shape == (n, 3 * heads * 128)
        assert coarse_gate.shape == (1, n, heads, 128)
        assert block_len.sum() <= n and sink_blocks == sink_q
        assert tau == 1. and topk_ratio == .2 and token_aug == 0 and tail is False
        calls.append({"n": n, "heads": heads, "bootstrap": kmean is None, "gate": coarse_gate.clone(),
                      "block_len": block_len.clone(), "prefix": tuple(sink_blocks)})
        value = projected.chunk(3, dim=-1)[2].view(1, n, heads, 128).clone()
        return value, torch.zeros(heads, 128), torch.ones(heads, 128)

    monkeypatch.setattr(sparse.ck, "sol_attn_chunked", kernel)
    return calls


@pytest.mark.parametrize("stage", ["low_0_4", "high_4_8"])
@pytest.mark.parametrize("mode", ["report_only", "apply_exp"])
@pytest.mark.parametrize("head_chunks", [1, 2])
def test_original_vsa_plan_gate_and_sparse_producer_run_without_selector_or_dense_replacement(
        native_producer_cpu_kernel_double, stage, mode, head_chunks):
    source = latent()
    bare = model()
    if head_chunks == 2:
        from h3_audio_t8_pkg.h3_memory_advanced import configure_low_vram_attention
        bare = configure_low_vram_attention(bare, 2)[0]
    prepared, sampler, sigmas, context, _ = build_stage(bare, source, stage, "trained_vsa_exp")
    original = sample(prepared, sampler, sigmas, source)[0]
    calls = native_producer_cpu_kernel_double
    initial = list(calls)
    calls.clear()
    config = EAVConfig(mode, tau=.2, start_video_progress=0., end_video_progress=1., g_hard_limit=3.)
    applied, runtime, _ = apply_stage_eav(prepared, sigmas, source, context, config)
    actual = sample(applied, sampler, sigmas, source)[0]
    report = runtime.snapshot()
    assert report["sparse_producer_calls"] == 4 and report["selector_calls"] == 0
    assert report["status"] == "observed_" + mode
    assert report["feta"]["attention_measurement_count"] == 4
    assert len(calls) == len(initial) == 4 * head_chunks
    assert [entry["bootstrap"] for entry in calls] == [True] * head_chunks + [False] * (3 * head_chunks)
    assert [entry["prefix"] for entry in calls] == [entry["prefix"] for entry in initial]
    if mode == "report_only":
        assert_equal(actual, original)
        for current, prior in zip(calls, initial):
            assert torch.equal(current["gate"], prior["gate"])
            assert torch.equal(current["block_len"], prior["block_len"])
    else:
        assert not torch.equal(actual["samples"].unbind()[0], original["samples"].unbind()[0])
    assert not torch.cuda.is_initialized()  # eligibility/kernel are explicit doubles!


def test_sparse_gain_scales_only_video_before_output_projection(native_producer_cpu_kernel_double):
    source, bare = latent(), model()
    attn = bare.model.diffusion_model.blocks[0].attn
    observed = []
    hook = attn.out_proj.register_forward_pre_hook(lambda module, args: observed.append(args[0].detach().clone()))
    try:
        prepared, sampler, sigmas, context, _ = build_stage(bare, source, "high_4_8", "trained_vsa_exp")
        sample(prepared, sampler, sigmas, source)
        reference = observed[0]
        observed.clear()
        applied, runtime, _ = apply_stage_eav(prepared, sigmas, source, context,
            EAVConfig("apply_exp", tau=.2, start_video_progress=0., end_video_progress=1., g_hard_limit=3.))
        sample(applied, sampler, sigmas, source)
        first = runtime.snapshot()["feta"]["forwards"][0]
        video_start = first["seq_len"] - first["video_rows"]  # Native T2VA video is the final segment.
        assert torch.equal(observed[0][:video_start], reference[:video_start])
        expected = reference[video_start:] * first["g_mean"]
        assert torch.allclose(observed[0][video_start:], expected, rtol=0, atol=2e-7)
        assert first["g_mean"] > 1
    finally:
        hook.remove()


def test_original_sparse_kernel_failure_and_gain_guard_propagate(native_producer_cpu_kernel_double, monkeypatch):
    source = latent()
    prepared, sampler, sigmas, context, _ = build_stage(model(), source, "high_4_8", "trained_vsa_exp")
    applied, runtime, _ = apply_stage_eav(prepared, sigmas, source, context,
        EAVConfig("apply_exp", tau=4., start_video_progress=0., end_video_progress=1., g_hard_limit=1.))
    with pytest.raises(RuntimeError, match="hard limit"):
        sample(applied, sampler, sigmas, source)
    assert runtime.snapshot()["status"] == "aborted"

    def broken(*args, **kwargs):
        raise RuntimeError("intentional original sparse kernel failure")

    monkeypatch.setattr(sparse.ck, "sol_attn_chunked", broken)
    applied, runtime, _ = apply_stage_eav(prepared, sigmas, source, context, EAVConfig(tau=.2))
    with pytest.raises(RuntimeError, match="intentional original sparse kernel failure"):
        sample(applied, sampler, sigmas, source)
    assert runtime.snapshot()["status"] == "aborted"


def test_actual_sparse_branch_does_not_certify_unimplemented_relay_bias(native_producer_cpu_kernel_double):
    from test_progressive_relay import paired
    source = latent()
    relay_model, positive, _ = paired(model())
    prepared, sampler, sigmas, context, _ = build_stage(relay_model, source, "high_4_8", "trained_vsa_exp")
    applied, runtime, _ = apply_stage_eav(prepared, sigmas, source, context,
        EAVConfig(tau=.2, start_video_progress=0., end_video_progress=1., g_hard_limit=3.))
    sample(applied, sampler, sigmas, source, positive)
    report = runtime.snapshot()
    assert report["sparse_producer_calls"] == 4
    assert report["relay_attention_calls"] == 0 and report["relay_required"] is True
    assert report["status"] == "unverified_relay_coverage"


def test_foreign_composed_producer_remains_selected_without_false_sparse_coverage(native_producer_cpu_kernel_double):
    bare, source, calls = model(), latent(), []

    def foreign(args, extra):
        calls.append(True)
        return extra["original_block"](args)

    bare.set_model_patch_replace(foreign, "dit", "double_block", 0)
    prepared, sampler, sigmas, context, _ = build_stage(bare, source, "high_4_8", "trained_vsa_exp")
    original = prepared.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)]
    applied, runtime, _ = apply_stage_eav(prepared, sigmas, source, context, EAVConfig(tau=.2))
    assert applied.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)] is original
    sample(applied, sampler, sigmas, source)
    assert len(calls) == 4
    assert runtime.snapshot()["status"] == "unverified_incomplete_stage_coverage"


def test_disabled_sparse_eav_is_exact_model_identity_without_new_producers():
    prepared, _, sigmas, context, _ = build_stage(model(), latent(), "low_0_4", "trained_vsa_exp")
    original = dict(prepared.model_options["transformer_options"]["patches_replace"]["dit"])
    applied, runtime, _ = apply_stage_eav(prepared, sigmas, latent(), context, EAVConfig("disabled"))
    assert applied is prepared
    assert prepared.model_options["transformer_options"]["patches_replace"]["dit"] == original
    assert runtime.snapshot()["sparse_producer_calls"] == 0
