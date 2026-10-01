"""Experimental temporal Relay on VDN's real window and linear paths.

Softmax retains the original key sets and adds the existing per-query penalty.
Linear attention has no logits: beta_weighted_text_seed_exp_v1 weights text beta
by exp(-penalty), recomputes the original nonlinear text factor per query frame,
and propagates that seed through the unchanged video scan. This is an explicit
VDN extension, NOT a claim of paper-softmax equivalence or trained quality.
"""
import torch

from .. import vdn_h3_advanced as vdn, prompt_relay_advanced as relay
from ..vdn_sdpa_backend import vdn_sdpa_context

PROFILE = "beta_weighted_text_seed_exp_v1"


def validate_route(route, layout):
    if (route["seq_len"] != layout.seq_len or route["video_start"] != layout.video_start
            or route["video_end"] != layout.video_end):
        raise ValueError("VDN Relay route does not match the actual VDN layout")
    for event in route["events"]:
        if not layout.text_start <= event["text_key_start"] < event["text_key_end"] <= layout.text_start + layout.text_len:
            raise ValueError("VDN Relay event span must address original text keys only")
    cursor = 0
    for segment in route["query_segments"]:
        start, end = segment["start"], segment["end"]
        if (not cursor <= start < end <= layout.seq_len or segment["kind"] not in ("video", "audio")
                or segment["query_times"].numel() != end - start):
            raise ValueError("VDN Relay query segments are invalid")
        cursor = end
    segments = [s for s in route["query_segments"] if s["kind"] == "video"]
    if len(segments) != 1 or (segments[0]["start"], segments[0]["end"]) != (layout.video_start, layout.video_end):
        raise ValueError("VDN Relay needs its actual target-video segment")
    times = segments[0]["query_times"].reshape(layout.num_frames, layout.tokens_per_frame)
    if not torch.equal(times, times[:, :1].expand_as(times)):
        raise ValueError("VDN Relay video frame contains inconsistent query times")
    return times[:, 0]


def neutral(route):
    return all(not bool(torch.any(relay.prompt_relay_penalty(segment["query_times"], event)))
               for segment in route["query_segments"] for event in route["events"])


def window(query, key, value, layout, bounds, scale, route, query_chunk_rows, workspace_bytes, stats):
    validate_route(route, layout)
    if neutral(route):
        stats["neutral_window"] = stats.get("neutral_window", 0) + 1
        return vdn.window_softmax_sdpa(query, key, value, layout, bounds, scale)
    if type(query_chunk_rows) is not int or query_chunk_rows < 1:
        raise ValueError("VDN Relay query chunk must be positive")
    out = torch.empty_like(query)
    global_idx = layout.global_index(query.device)
    vs, ve, per_frame = layout.video_start, layout.video_end, layout.tokens_per_frame

    def rows(query_ids, key_ids):
        row_bytes = key_ids.numel() * max(4, query.element_size())
        count = min(query_chunk_rows, int(workspace_bytes // max(1, row_bytes)))
        if count < 1:
            raise ValueError("VDN Relay workspace cannot hold one selected-key bias row")
        for start in range(0, query_ids.numel(), count):
            selected = query_ids[start:start + count]
            bias = torch.zeros((selected.numel(), key_ids.numel()), device=query.device, dtype=query.dtype)
            for segment in route["query_segments"]:
                positions = (selected >= segment["start"]) & (selected < segment["end"])
                indices = positions.nonzero().flatten()
                if not indices.numel():
                    continue
                times = segment["query_times"][selected[indices] - segment["start"]]
                for event in route["events"]:
                    columns = ((key_ids >= event["text_key_start"]) & (key_ids < event["text_key_end"])).nonzero().flatten()
                    bias[indices[:, None], columns[None, :]] = -relay.prompt_relay_penalty(times, event).to(bias)[:, None]
            stats["bias_peak_bytes"] = max(stats.get("bias_peak_bytes", 0), bias.numel() * bias.element_size())
            stats["window_calls"] = stats.get("window_calls", 0) + 1
            # Preserve the original safe SDPA context. Unsupported selected
            # CUDA kernels fail normally; no quadratic/dense fallback.
            with vdn_sdpa_context(query.device):
                result = torch.nn.functional.scaled_dot_product_attention(
                    query[selected].permute(1, 0, 2).unsqueeze(0),
                    key[key_ids].permute(1, 0, 2).unsqueeze(0),
                    value[key_ids].permute(1, 0, 2).unsqueeze(0),
                    attn_mask=bias, dropout_p=0., is_causal=False, scale=scale)
            out[selected] = result.squeeze(0).permute(1, 0, 2)

    anchors = torch.cat((global_idx, torch.arange(vs, vs + per_frame, device=query.device),
                         torch.arange(ve - per_frame, ve, device=query.device))).unique(sorted=True)
    rows(anchors, torch.arange(layout.seq_len, device=query.device))
    frame = 1
    while frame < layout.num_frames - 1:
        end = frame + 1
        while end < layout.num_frames - 1 and bounds[end] == bounds[frame]:
            end += 1
        lo, hi = bounds[frame]
        selected_frames = sorted(set(range(max(lo, 0), min(hi, layout.num_frames - 1) + 1)) | {0, layout.num_frames - 1})
        video_keys = (torch.tensor(selected_frames, device=query.device)[:, None] * per_frame
                      + torch.arange(per_frame, device=query.device)[None, :] + vs).flatten()
        rows(torch.arange(vs + frame * per_frame, vs + end * per_frame, device=query.device),
             torch.cat((global_idx, video_keys)))
        frame = end
    return out


def text_weights(times, text_len, text_start, events):
    weights = torch.ones((times.numel(), text_len), device=times.device, dtype=torch.float32)
    for event in events:
        start, end = event["text_key_start"] - text_start, event["text_key_end"] - text_start
        weights[:, start:end] = torch.exp(-relay.prompt_relay_penalty(times, event))[:, None]
    return weights


def text_features(module, text_x, text_qkv, head_slice):
    length, heads, dim = text_x.shape[0], module.heads, module.head_dim
    # Original QKV is [tokens,heads,dim]; normalization is per head. Slice
    # before activation so the additional text features are head-chunk bounded.
    key = module._activate(text_qkv[1].view(length, heads, dim)[:, head_slice], True).permute(1, 0, 2)
    value = module._activate(text_qkv[2].view(length, heads, dim)[:, head_slice], False).permute(1, 0, 2)
    beta = torch.sigmoid(module.beta_proj(text_x)).T[head_slice]
    return key, value, beta


def seed_from_features(features, weights, scale):
    key, value, beta = features
    beta = beta * weights[None, :]
    a, b = vdn._frame_statistics(key[None], value[None], beta[None])
    _, injection = vdn._vdn_factor(torch.ones((1, key.shape[0], key.shape[-1]), device=a.device), a, b)
    return scale * injection[0]


def weighted_seed(module, text_x, text_qkv, weights, head_slice):
    return seed_from_features(text_features(module, text_x, text_qkv, head_slice), weights, module.TEXT_STATE_SCALE)


def scan_basis(alpha, a, b):
    """Affine scan components: state(seed) = state(0) + seed @ product(T)."""
    transitions, injections = vdn._vdn_factor(alpha, a, b)
    frames, heads, dim, _ = transitions.shape
    zero = torch.zeros((heads, dim, dim), device=a.device, dtype=injections.dtype)
    eye = torch.eye(dim, device=a.device, dtype=injections.dtype).expand(heads, -1, -1)
    prefix, suffix = torch.empty_like(transitions), torch.empty_like(transitions)
    prefix_op, suffix_op = torch.empty_like(transitions), torch.empty_like(transitions)
    state, operator = zero, eye
    for frame in range(frames):
        torch.baddbmm(injections[frame], state, transitions[frame], out=prefix[frame])
        torch.bmm(operator, transitions[frame], out=prefix_op[frame])
        state, operator = prefix[frame], prefix_op[frame]
    state, operator = zero, eye
    for frame in range(frames - 1, -1, -1):
        torch.baddbmm(injections[frame], state, transitions[frame], out=suffix[frame])
        torch.bmm(operator, transitions[frame], out=suffix_op[frame])
        state, operator = suffix[frame], suffix_op[frame]
    return prefix, suffix, prefix_op, suffix_op


def state_for_frame(basis, alpha, bounds, frame, seed):
    prefix, suffix, prefix_op, suffix_op = basis
    lo, hi = bounds[frame]
    left, right = lo - 1, hi + 1
    before = seed if left < 0 else prefix[left] + torch.bmm(seed, prefix_op[left])
    after = seed if right >= len(alpha) else suffix[right] + torch.bmm(seed, suffix_op[right])
    logs = torch.cat((torch.zeros_like(alpha[:1]), torch.log(alpha.clamp_min(1e-12)).cumsum(0)))
    before_scale = torch.exp(logs[frame + 1] - logs[max(0, left + 1)])
    after_scale = torch.exp(logs[min(right, len(alpha))] - logs[frame])
    return before * before_scale.unsqueeze(1) + after * after_scale.unsqueeze(1)


def linear(module, video_x, layout, bounds, video_qkv, text_x, text_qkv, route, workspace_bytes, stats):
    times = validate_route(route, layout)
    if neutral({**route, "query_segments": [s for s in route["query_segments"] if s["kind"] == "video"]}):
        stats["neutral_linear"] = stats.get("neutral_linear", 0) + 1
        return module(video_x, layout, bounds, video_qkv, text_x, text_qkv)
    frames, tokens, heads, dim = layout.num_frames, layout.tokens_per_frame, module.heads, module.head_dim
    if frames <= 2:
        return video_x.new_zeros(frames * tokens, heads * dim)
    inner, count = slice(tokens, (frames - 1) * tokens), frames - 2
    xv = video_x[inner]
    query, key, value = module._features(tuple(x[inner] for x in video_qkv), count, (layout.frame_height, layout.frame_width))
    shape = (count, tokens, heads, dim)
    queries = query.view(shape)
    keys, values = (x.view(shape).permute(0, 2, 1, 3) for x in (key, value))
    beta = torch.sigmoid(module.beta_proj(xv)).view(count, tokens, heads).permute(0, 2, 1)
    alpha = module.alpha(xv.view(count, tokens, -1).mean(dim=1, dtype=torch.float32))
    inner_bounds = [(lo - 1, hi - 1) for lo, hi in bounds[1:-1]]
    # Account for chunk-owned scan/factor matrices plus text/solve temporaries.
    # Original features/output and backend allocator/workspace are not included.
    per_head = (10 * count + 24) * dim * dim * 4 + text_x.shape[0] * dim * 16
    chunk = min(heads, int(workspace_bytes // max(1, per_head)))
    if chunk < 1:
        raise ValueError("VDN Relay workspace cannot fit one linear head; raise the explicit adapter budget")
    stats["linear_peak_estimate_bytes"] = max(stats.get("linear_peak_estimate_bytes", 0), per_head * chunk)
    readout = torch.empty_like(queries)
    for start in range(0, heads, chunk):
        selected = slice(start, min(start + chunk, heads))
        a, b = vdn._frame_statistics(keys[:, selected], values[:, selected], beta[:, selected])
        basis = scan_basis(alpha[:, selected], a, b)
        features = text_features(module, text_x, text_qkv, selected)
        for frame in range(count):
            weights = text_weights(times[frame + 1:frame + 2], layout.text_len, layout.text_start, route["events"])[0]
            seed = seed_from_features(features, weights, module.TEXT_STATE_SCALE)
            state = state_for_frame(basis, alpha[:, selected], inner_bounds, frame, seed).to(xv.dtype)
            readout[frame, :, selected] = torch.einsum("hvk,shk->shv", state, queries[frame, :, selected])
        stats["linear_head_chunks"] = stats.get("linear_head_chunks", 0) + 1
        stats["linear_seed_factorizations"] = stats.get("linear_seed_factorizations", 0) + count
    stats["linear_frames"] = stats.get("linear_frames", 0) + count
    readout = module.norm(readout.reshape(count * tokens, heads, dim))
    readout = (readout * module.output_gate(xv)).reshape(count * tokens, heads * dim)
    output = readout.new_zeros(frames * tokens, heads * dim)
    output[inner] = readout
    return output
