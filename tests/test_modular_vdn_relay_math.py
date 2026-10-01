"""Small exact-window golden and deliberately slow per-query original scans."""
from dataclasses import replace

import pytest
import torch

from h3_audio_t8_pkg import vdn_h3_advanced as vdn, prompt_relay_advanced as relay
from h3_audio_t8_pkg.modular_sampling import vdn_relay_math as math
from test_modular_vdn_baseline import tiny_vdn


def layout_route(frames=16, route_kind="joint_av_exp", neutral=False):
    layout = vdn.VDNSequenceLayout(6 + frames * 4, 6, frames, 4, 2, 2, 0, 3, ("ref_img",))
    events = tuple(dict(text_key_start=i, text_key_end=i+1, midpoint=float(i*5),
                        window=1000. if neutral else .1, sigma=2.) for i in (1, 2))
    segments = [{"kind": "video", "start": 6, "end": layout.seq_len,
                 "query_times": torch.arange(frames).float().repeat_interleave(4)}]
    if route_kind == "joint_av_exp":
        segments.insert(0, {"kind": "audio", "start": 4, "end": 6, "query_times": torch.tensor([0., 8.])})
    return layout, {"seq_len": layout.seq_len, "video_start": 6, "video_end": layout.seq_len,
                    "events": events, "query_segments": segments}


def dense_golden(q, k, v, layout, bounds, scale, route):
    mask = torch.zeros((layout.seq_len, layout.seq_len), dtype=q.dtype)
    for frame in range(layout.num_frames):
        if frame in (0, layout.num_frames - 1):
            continue
        for other in range(1, layout.num_frames - 1):
            if not bounds[frame][0] <= other <= bounds[frame][1]:
                qa, ka = layout.video_start + frame * 4, layout.video_start + other * 4
                mask[qa:qa+4, ka:ka+4] = -torch.inf
    for segment in route["query_segments"]:
        mask[segment["start"]:segment["end"]] += relay.make_prompt_relay_bias(
            segment["query_times"], layout.seq_len, route["events"], dtype=q.dtype)
    probabilities = (torch.einsum("qhd,khd->hqk", q, k) * scale + mask).softmax(-1)
    return torch.einsum("hqk,khd->qhd", probabilities, v)


@pytest.mark.parametrize("frames", [2, 8, 16])
@pytest.mark.parametrize("chunk", [1, 7, 32])
@pytest.mark.parametrize("route_kind", ["video_only_paper", "joint_av_exp"])
def test_actual_window_keys_anchor_global_rows_and_joint_temporal_bias_match_dense_golden(frames, chunk, route_kind):
    layout, route = layout_route(frames, route_kind)
    generator = torch.Generator().manual_seed(192)
    q, k, v = [torch.randn(layout.seq_len, 2, 8, generator=generator, dtype=torch.float64) for _ in range(3)]
    bounds = vdn.window_bounds(frames)
    stats = {}
    actual = math.window(q, k, v, layout, bounds, .25, route, chunk, 8192, stats)
    expected = dense_golden(q, k, v, layout, bounds, .25, route)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert stats["window_calls"] > 0 and stats["bias_peak_bytes"] <= 8192
    # Text and reference queries are untouched by Relay (their own outputs).
    original = vdn.window_softmax_sdpa(q, k, v, layout, bounds, .25)
    torch.testing.assert_close(actual[:4], original[:4], rtol=1e-12, atol=1e-12)
    assert not torch.equal(actual[layout.video_start:], original[layout.video_start:])


@pytest.mark.parametrize("frames", [2, 16])
def test_neutral_window_is_original_bit_exact(frames):
    layout, route = layout_route(frames, neutral=True)
    q = torch.arange(layout.seq_len * 16).reshape(layout.seq_len, 2, 8).float() / 100.
    bounds, stats = vdn.window_bounds(frames), {}
    assert torch.equal(math.window(q, q, q, layout, bounds, .25, route, 1, 8192, stats),
                       vdn.window_softmax_sdpa(q, q, q, layout, bounds, .25))
    assert stats == {"neutral_window": 1}


def slow_linear(module, x, layout, bounds, qkv, text, text_qkv, route):
    f, s, h, d = layout.num_frames - 2, layout.tokens_per_frame, module.heads, module.head_dim
    xv = x[s:-s]
    q, k, v = module._features(tuple(z[s:-s] for z in qkv), f, (layout.frame_height, layout.frame_width))
    q = q.view(f, s, h, d)
    k, v = [z.view(f, s, h, d).permute(0, 2, 1, 3) for z in (k, v)]
    beta = torch.sigmoid(module.beta_proj(xv)).view(f, s, h).permute(0, 2, 1)
    a, b = vdn._frame_statistics(k, v, beta)
    alpha = module.alpha(xv.view(f, s, -1).mean(1, dtype=torch.float32))
    inner_bounds = [(lo - 1, hi - 1) for lo, hi in bounds[1:-1]]
    kt = module._activate(text_qkv[1], True).view(len(text), h, d).permute(1, 0, 2)[None]
    vt = module._activate(text_qkv[2], False).view(len(text), h, d).permute(1, 0, 2)[None]
    bt = torch.sigmoid(module.beta_proj(text)).T[None]
    times = route["query_segments"][-1]["query_times"].reshape(f + 2, s)[:, 0]
    readout = []
    for frame in range(f):
        weights = torch.ones(len(text))
        for event in route["events"]:
            penalty = relay.prompt_relay_penalty(times[frame + 1:frame + 2], event)
            weights[event["text_key_start"]:event["text_key_end"]] = torch.exp(-penalty)
        ta, tb = vdn._frame_statistics(kt, vt, bt * weights[None, None, :])
        _, seed = vdn._vdn_factor(torch.ones(1, h, d), ta, tb)
        seed = seed[0] * module.TEXT_STATE_SCALE
        # Full ORIGINAL scans rerun for each query seed: slow independent oracle.
        prefix, suffix = vdn._run_scans(alpha, a, b, seed)
        state = vdn._gather_linear_state(prefix, suffix, alpha, inner_bounds, seed)[frame].to(x.dtype)
        readout.append(torch.einsum("hvk,shk->shv", state, q[frame]))
    output = module.norm(torch.stack(readout).reshape(f * s, h, d))
    output = (output * module.output_gate(xv)).reshape(f * s, h * d)
    return torch.cat((torch.zeros(s, h * d), output, torch.zeros(s, h * d)))


@pytest.mark.parametrize("workspace", [16 << 20, 64 << 20])
@pytest.mark.parametrize("frames", [12, 16])
@torch.no_grad()
def test_weighted_nonlinear_seed_scan_basis_matches_slow_original_per_query_scans(workspace, frames):
    _, branch = tiny_vdn()
    module = branch.model.blocks[0].linear_attention
    layout, route = layout_route(frames)
    generator = torch.Generator().manual_seed(731)
    x = torch.randn(frames * 4, 24, generator=generator)
    qkv = tuple(torch.randn(frames * 4, 3, 128, generator=generator) * .2 for _ in range(3))
    text = torch.randn(3, 24, generator=generator)
    text_qkv = tuple(torch.randn(3, 3, 128, generator=generator) * .2 for _ in range(3))
    stats, bounds = {}, vdn.window_bounds(frames)
    actual = math.linear(module, x, layout, bounds, qkv, text, text_qkv, route, workspace, stats)
    expected = slow_linear(module, x, layout, bounds, qkv, text, text_qkv, route)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert stats["linear_frames"] == frames - 2 and stats["linear_peak_estimate_bytes"] <= workspace
    assert torch.count_nonzero(actual[:4]) == torch.count_nonzero(actual[-4:]) == 0
    original = module(x, layout, bounds, qkv, text, text_qkv)
    assert not torch.equal(actual, original)
    _, neutral_route = layout_route(frames, neutral=True)
    assert torch.equal(math.linear(module, x, layout, bounds, qkv, text, text_qkv, neutral_route, workspace, {}), original)
    neutral_seed = math.weighted_seed(module, text, text_qkv, torch.ones(3), slice(None))
    assert torch.equal(neutral_seed, module._text_state(text, text_qkv))


def test_bad_route_and_real_bias_workspace_errors_not_silenced():
    layout, route = layout_route()
    with pytest.raises(ValueError, match="actual VDN layout"):
        math.validate_route(route, replace(layout, video_start=7))
    route["events"][0]["text_key_end"] = 4
    with pytest.raises(ValueError, match="text keys"):
        math.validate_route(route, layout)
    layout, route = layout_route()
    q = torch.zeros(layout.seq_len, 2, 8)
    with pytest.raises(ValueError, match="one selected-key"):
        math.window(q, q, q, layout, vdn.window_bounds(16), .25, route, 32, 1, {})
