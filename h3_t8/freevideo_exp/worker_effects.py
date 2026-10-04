"""FreeVideo-local effects on the exact pinned head-chunk producer.

VDN temporal Relay is an experimental beta-weighted nonlinear text seed, not
softmax equivalence. The original video features, alpha bridge, anchor pruning,
head partition and final inference epilogue remain authoritative. No extra NFE.
"""
import math
import types

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def penalty(times, event):
    return (times.float() - event["midpoint"]).abs().sub(event["window"]).clamp_min(0).square() / (2 * event["sigma"] ** 2)


def temporal_trace(q, k, frames, spatial, budget):
    """Exact paper trace for one original head group, bounded spatial columns."""
    rows, heads, dim = q.shape
    if k.shape != q.shape or rows != frames * spatial or frames < 2:
        raise ValueError("FreeVideo FETA Q/K differs from the target-video grid")
    per_column = heads * frames * frames * 12
    count = min(spatial, budget // per_column)
    if count < 1:
        raise ValueError("FreeVideo EAV budget cannot hold one temporal column")
    q = q.view(frames, spatial, heads, dim).permute(1, 2, 0, 3)
    k = k.view(frames, spatial, heads, dim).permute(1, 2, 0, 3)
    trace = torch.zeros((), device=q.device, dtype=torch.float64)
    for start in range(0, spatial, count):
        logits = ((q[start:start + count] * dim ** -.5) @ k[start:start + count].transpose(-1, -2)).float()
        probabilities = logits.softmax(-1)
        trace += probabilities.diagonal(dim1=-2, dim2=-1).sum(dtype=torch.float64)
    return trace, count * per_column


def projected_gain(attn, hidden, rotary, config, chunk):
    from src.models.softmax_attention.kernels import _qk_prep
    from src.models.ops.fp8_linear import Fp8Linear, quantize_activation
    from freevideo_engine.fp8_ops import sliced_projection
    layout = attn.layout
    projections = (attn.orig.to_q, attn.orig.to_k)
    # Keep the original full-row activation quantization and original head GEMM
    # shapes. Gathering fewer input rows could silently change FP8 rounding.
    quantized = quantize_activation(hidden) if all(isinstance(p, Fp8Linear) for p in (*projections, attn.orig.to_v)) else None
    trace = torch.zeros((), device=hidden.device, dtype=torch.float64)
    peak = 0
    for start in range(0, attn.num_heads, chunk):
        end = min(start + chunk, attn.num_heads)
        channels = slice(start * attn.head_dim, end * attn.head_dim)
        raw = [sliced_projection(p, hidden, channels, quantized).view(len(hidden), end - start, attn.head_dim) for p in projections]
        q = _qk_prep(raw[0], attn.orig.norm_q.weight, attn.orig.norm_q.eps, *rotary)
        k = _qk_prep(raw[1], attn.orig.norm_k.weight, attn.orig.norm_k.eps, *rotary)
        part, workspace = temporal_trace(q[layout.video_start:layout.video_end], k[layout.video_start:layout.video_end],
            layout.num_frames, layout.tokens_per_frame, config["workspace_mib"] << 20)
        trace += part
        peak = max(peak, workspace)
        del raw, q, k
    matrices = layout.tokens_per_frame * attn.num_heads
    cfi = ((matrices * layout.num_frames - trace) / (matrices * layout.num_frames * (layout.num_frames - 1))).float()
    gain = ((layout.num_frames + config["tau"]) * cfi).clamp_min(1.)
    value = float(gain)
    if not math.isfinite(value) or value > config["g_hard_limit"]:
        raise ValueError("FreeVideo EAV gain exceeded the explicit hard limit or is not finite")
    return gain, dict(gain=value, cfi=float(cfi), statistics_peak_bytes=peak)


def scan_basis(backend, alpha, a, b):
    """State(seed)=state(0)+seed@operator, using the actual delta backend."""
    with torch.autocast(device_type=a.device.type, enabled=False):
        trans, injection = backend.factor_apply(alpha, a, b)
        frames, heads, dim, _ = trans.shape
        zero = torch.zeros_like(injection[0])
        eye = torch.eye(dim, device=a.device, dtype=trans.dtype).expand(heads, -1, -1)
        prefix, suffix, left_op, right_op = (torch.empty_like(trans) for _ in range(4))
        state, op = zero, eye
        for f in range(frames):
            torch.baddbmm(injection[f], state, trans[f], out=prefix[f])
            torch.bmm(op, trans[f], out=left_op[f])
            state, op = prefix[f], left_op[f]
        state, op = zero, eye
        for f in range(frames - 1, -1, -1):
            torch.baddbmm(injection[f], state, trans[f], out=suffix[f])
            torch.bmm(op, trans[f], out=right_op[f])
            state, op = suffix[f], right_op[f]
    return prefix, suffix, left_op, right_op


def state_for_frame(basis, alpha, bounds, f, seed, bridge):
    prefix, suffix, left_op, right_op = basis
    lo, hi = bounds[f]
    left, right = lo - 1, hi + 1
    before = seed if left < 0 else prefix[left] + seed @ left_op[left]
    after = seed if right >= len(alpha) else suffix[right] + seed @ right_op[right]
    if bridge == "alpha":
        logs = torch.cat((torch.zeros_like(alpha[:1]), alpha.clamp_min(1e-12).log().cumsum(0)))
        before = before * (logs[f + 1] - logs[max(left + 1, 0)]).exp().unsqueeze(-2)
        after = after * (logs[min(right, len(alpha))] - logs[f]).exp().unsqueeze(-2)
    elif bridge != "none":
        raise ValueError("Unknown VDN alpha bridge")
    return before + after


def relay_readout(module, xv, frames, spatial, bounds, raw, frame_size, text_x, text_raw,
                  *, heads, beta, gate, frame_mean, text_beta, times, binding, stats):
    """Head-sliced upstream inference body with per-query nonlinear text seeds."""
    from src.models.linear_attention.scan import frame_statistics
    from src.models.linear_attention.kernels import linear_epilogue
    n_heads, dim = heads.stop - heads.start, module.head_dim
    length, budget = text_raw[0].shape[0], binding.get("workspace_mib", 64) << 20
    per_head = (10 * frames + 24) * dim * dim * 4 + length * dim * 16
    per_seed_frame = 12 * dim * dim * 4 + length * dim * 16
    choices = []
    for count in range(1, n_heads + 1):
        batch = min(frames, (budget - count * per_head) // max(1, count * per_seed_frame))
        if batch >= 1:
            choices.append((math.ceil(n_heads / count) * math.ceil(frames / batch), -count, count, batch))
    if not choices:
        raise ValueError("FreeVideo Relay linear budget cannot hold one head")
    _, _, subchunk, seed_batch = min(choices)
    query, key, value = module._features(raw, frames, frame_size, inference=True, query_fhsd=(frames, spatial), heads=heads)
    keys, values = [v.view(frames, spatial, n_heads, dim).permute(0, 2, 1, 3) for v in (key, value)]
    beta = beta.view(frames, spatial, n_heads).permute(0, 2, 1)
    alpha = module.alpha(frame_mean, heads=heads)
    readout = torch.empty_like(query)
    video_backend = module._delta_backend("backend", spatial)
    for start in range(0, n_heads, subchunk):
        end = min(start + subchunk, n_heads)
        selected = slice(start, end)
        a, b = frame_statistics(keys[:, selected], values[:, selected], beta[:, selected], a_fp32=module.a_fp32, inference=True)
        basis = scan_basis(video_backend, alpha[:, selected], a, b)
        del a, b
        # Cache the original un-convolved text features once per head slice.
        # Batch only the small text statistics/solve, with an explicit bound;
        # avoid thousands of redundant SiLU/L2Norm and Cholesky launches.
        text_key = module._feature_one(text_raw[1][:, selected], "k", None, None, use_conv=False).transpose(0, 1)
        text_value = module._feature_one(text_raw[2][:, selected], "v", None, None, use_conv=False).transpose(0, 1)
        text_backend = module._delta_backend("text_backend", length)
        for begin in range(0, frames, seed_batch):
            stop = min(begin + seed_batch, frames)
            weights = torch.ones(stop - begin, length, device=times.device, dtype=torch.float32)
            for event in binding["events"]:
                weights[:, event["text_key_start"]:event["text_key_end"]] = (-penalty(times[begin:stop], event)).exp()[:, None]
            weighted_beta = text_beta[:, selected].T[None] * weights[:, None, :]
            # Original text backend, SiLU/L2Norm, A symmetrization, 0.5 scale;
            # weighting beta happens BEFORE the nonlinear inverse/solve.
            ta, tb = frame_statistics(text_key[None].expand(stop - begin, -1, -1, -1),
                text_value[None].expand(stop - begin, -1, -1, -1), weighted_beta, a_fp32=module.a_fp32)
            with torch.autocast(device_type=ta.device.type, enabled=False):
                unused_transition, seeds = text_backend.factor_apply(torch.ones(stop - begin, end - start, dim, device=ta.device, dtype=ta.dtype), ta, tb)
                del unused_transition
                seeds = seeds * module.TEXT_STATE_SCALE
            for f in range(begin, stop):
                state = state_for_frame(basis, alpha[:, selected], bounds, f, seeds[f - begin], module.bridge).to(gate.dtype)
                readout[f, selected] = query[f, selected] @ state.transpose(-1, -2)
            stats["linear_seed_solver_batches"] += 1
            del ta, tb, seeds, weighted_beta, weights, state
        stats["linear_seed_factorizations"] += frames * (end - start)
        stats["linear_peak_estimate_bytes"] = max(stats["linear_peak_estimate_bytes"], (per_head + seed_batch * per_seed_frame) * (end - start))
        del basis, text_key, text_value
    return linear_epilogue(readout, module.norm.weight, gate, module.norm.eps, inference=True, fhsd=True)


class EffectPolicy:
    def __init__(self, original, runtime):
        self.original, self.runtime = original, runtime

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, q, k, v, layout, bounds, scale, anchor_frames="none"):
        state = self.runtime
        binding = state.binding
        if binding is None or binding["mode"] != "apply_exp" or state.neutral:
            output = self.original(q, k, v, layout, bounds, scale, anchor_frames)
        else:
            plan = self.original.prepare(layout, bounds, q.device, anchor_frames)
            output = torch.empty_like(q)
            def run(rows, keys, window):
                count = min(binding["query_chunk_rows"], (binding.get("workspace_mib", 64) << 20) // max(1, len(keys) * 4))
                if count < 1:
                    raise ValueError("FreeVideo Relay bias budget cannot fit a selected-key query row")
                for start in range(0, len(rows), count):
                    selected = rows[start:start + count]
                    bias = torch.zeros((len(selected), len(keys)), device=q.device, dtype=q.dtype)
                    for segment in state.query_segments:
                        local = ((selected >= segment["start"]) & (selected < segment["end"])).nonzero().flatten()
                        if not len(local):
                            continue
                        times = segment["times"][selected[local] - segment["start"]]
                        for event in binding["events"]:
                            columns = ((keys >= event["text_key_start"]) & (keys < event["text_key_end"])).nonzero().flatten()
                            bias[local[:, None], columns[None, :]] = -penalty(times, event).to(bias)[:, None]
                    state.stats["bias_peak_bytes"] = max(state.stats["bias_peak_bytes"], bias.numel() * bias.element_size())
                    # An explicit pinned cuDNN path; errors propagate, no dense/
                    # quadratic math-backend or foreign Sage fallback.
                    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                        result = torch.nn.functional.scaled_dot_product_attention(q[selected].transpose(0, 1).unsqueeze(0),
                            k[keys].transpose(0, 1).unsqueeze(0), v[keys].transpose(0, 1).unsqueeze(0),
                            attn_mask=bias, scale=scale, dropout_p=0., is_causal=False)
                    output[selected] = result[0].transpose(0, 1)
                    state.stats["biased_window_calls" if window else "biased_global_calls"] += 1
            if len(plan.dense_q):
                # Preserve original unmasked dense-query GEMM shapes for text,
                # references and non-routed audio. Only overwrite routed rows.
                chunk = self.original.query_chunk or len(plan.dense_q)
                for start in range(0, len(plan.dense_q), chunk):
                    rows = plan.dense_q[start:start + chunk]
                    output[rows] = self.original.dense(q[rows], k, v, scale)
                routed = torch.zeros_like(plan.dense_q, dtype=torch.bool)
                for segment in state.query_segments:
                    routed |= (plan.dense_q >= segment["start"]) & (plan.dense_q < segment["end"])
                run(plan.dense_q[routed], torch.arange(layout.seq_len, device=q.device), False)
            if plan.has_windows:
                qa, ka = self.original.offsets
                for i in range(len(qa) - 1):
                    run(plan.win_q[qa[i]:qa[i + 1]], plan.kv_gather[ka[i]:ka[i + 1]], True)
        if state.gain is not None and state.eav["mode"] == "apply_exp":
            output[layout.video_start:layout.video_end].mul_(state.gain.to(output))
        return output


class Runtime:
    def __init__(self, engine, effects, canvas):
        self.engine, self.eav, self.binding, self.canvas = engine, effects.get("eav"), effects.get("relay"), canvas
        self.gain, self.query_segments, self.video_times = None, [], None
        self.neutral = True
        self.forwards, self.blocks, self.eav_measurements = 0, 0, []
        self.stats = dict(bias_peak_bytes=0, linear_peak_estimate_bytes=0, linear_seed_factorizations=0,
                          linear_seed_solver_batches=0,
                          biased_global_calls=0, biased_window_calls=0)
        self.handles, self.originals = [], []

    def before(self, model, args, kwargs):
        from src.models.hybrid_transform import iter_hybrids
        layout = next(iter_hybrids(model)).layout
        positions = kwargs["position_ids"]
        if (layout.video_end != len(positions) or layout.text_start != 0
                or layout.num_frames != (self.canvas["frames"] - 5) // 17 * 5 + 2
                or layout.tokens_per_frame != (self.canvas["width"] // 32) * (self.canvas["height"] // 32)
                or (self.binding and layout.text_len != self.binding["embeddings"]["shape"][1])):
            raise ValueError("FreeVideo effects differ from the actual packed layout")
        self.video_times = positions[layout.video_start:layout.video_end, 0]
        self.video_times = (self.video_times - self.video_times[0]).float()
        grid = self.video_times.view(layout.num_frames, layout.tokens_per_frame)
        if not torch.equal(grid, grid[:, :1].expand_as(grid)):
            raise ValueError("FreeVideo Relay video frame has inconsistent native query times")
        if self.binding:
            from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MINIMAX_H3_TEXT_TAG
            for event in self.binding["events"]:
                if not bool((kwargs["token_tags"][event["text_key_start"]:event["text_key_end"]] == MINIMAX_H3_TEXT_TAG).all()):
                    raise ValueError("FreeVideo Relay event overlaps a non-text presentation token")
        self.query_segments = [dict(start=layout.video_start, end=layout.video_end, times=self.video_times)]
        if self.binding and self.binding["query_route"] == "joint_av_exp":
            audio_rows = 2 * round(self.canvas["frames"] / 24 * 40)
            actual = kwargs["audio_indices"][-audio_rows:]
            start, end = int(actual[0]), int(actual[-1]) + 1
            if len(actual) != audio_rows or start < layout.text_len or end != layout.video_start or not torch.equal(actual, torch.arange(start, end, device=actual.device)):
                raise ValueError("FreeVideo target-audio layout is invalid")
            times = positions[start:end, 0]
            self.query_segments.insert(0, dict(start=start, end=end, times=(times - times[0]).float()))
        self.neutral = self.binding is None or len(self.binding["events"]) <= 1 or all(
            not bool(penalty(segment["times"], event).any()) for segment in self.query_segments for event in self.binding["events"])
        index = kwargs["timestep_indices"][layout.video_start]
        self.progress = float(kwargs["timestep"][index])  # H3 t=1-sigma, not Core's 1000*sigma.
        self.forwards += 1

    def snapshot(self):
        return dict(schema="t8-freevideo-effects-audit-v1", installed_blocks=len(self.originals),
            forwards=self.forwards, completed_blocks=self.blocks, eav=self.eav,
            eav_measurements=self.eav_measurements, relay_binding_sha256=self.binding["sha256"] if self.binding else None,
            relay_mode=self.binding["mode"] if self.binding else None, stats=self.stats,
            linear_profile="beta_weighted_text_seed_exp_v1", quality_approved=False,
            boundary="Workspace limits explicit statistics/bias/scan matrices, not FP8 input/QK/base features or total VRAM. No extra NFE; nonlinear seed extension is not paper softmax equivalence.")

    def close(self):
        for handle in self.handles:
            handle.remove()
        for attn, attention, branch_forward in self.originals:
            attn._hybrid_forward = attention
            attn.linear_attention._readout_inference = branch_forward
        self.gain = None


def install(engine, effects, canvas):
    from freevideo_exp.effects import validate_eav, validate_binding
    from freevideo_engine.head_chunk import install_head_chunks
    from src.models.hybrid_transform import iter_hybrids
    if effects.get("eav") is not None:
        validate_eav(effects["eav"])
    if effects.get("relay") is not None:
        validate_binding(effects["relay"])
    state = Runtime(engine, effects, canvas)
    if not any(effects.values()):
        return state
    if engine.config["head_parallelism"] != 1 or engine.config["attention"] not in ("cudnn", "cudnn/cudnn"):
        raise ValueError("FreeVideo effects require the explicit serial cuDNN head profile")
    policy = EffectPolicy(engine.attention, state)
    state.originals = [(attn, attn._hybrid_forward, attn.linear_attention._readout_inference)
                       for attn in iter_hybrids(engine.transformer)]
    install_head_chunks(engine.transformer, policy, engine.config["head_chunk"],
        cpu_outputs=engine.config["attention_cpu_outputs"], projection_chunk=engine.config["projection_chunk"] or 1024,
        grouped_outputs=engine.config["grouped_attention_outputs"], parallelism=1)
    state.handles.append(engine.transformer.register_forward_pre_hook(state.before, with_kwargs=True))
    for attn in iter_hybrids(engine.transformer):
        attention, readout = attn._hybrid_forward, attn.linear_attention._readout_inference
        def wrapped(attn, hidden, rotary, original=attention):
            state.gain = None
            if state.eav and state.eav["mode"] != "disabled" and state.eav["start"] <= state.progress <= state.eav["end"]:
                state.gain, measured = projected_gain(attn, hidden, rotary, state.eav, engine.config["head_chunk"])
                state.eav_measurements.append(dict(forward=state.forwards, block=state.blocks % 50, progress=state.progress, **measured))
            output = original(hidden, rotary)
            state.blocks += 1
            state.gain = None
            return output
        attn._hybrid_forward = types.MethodType(wrapped, attn)
        def linear(module, *args, original=readout, owner=attn, **kwargs):
            binding = state.binding
            if binding is None or binding["mode"] != "apply_exp" or state.neutral:
                return original(*args, **kwargs)
            times = state.video_times.view(-1, owner.layout.tokens_per_frame)[:, 0]
            frames = args[1]
            if len(times) == frames + 2:
                times = times[1:-1]  # The parent forward pruned both anchors.
            if len(times) != frames:
                raise ValueError("FreeVideo Relay linear frame grid differs from the original anchor pruning")
            if all(not bool(penalty(times, event).any()) for event in binding["events"]):
                return original(*args, **kwargs)
            return relay_readout(module, *args, **kwargs, times=times, binding=binding, stats=state.stats)
        attn.linear_attention._readout_inference = types.MethodType(linear, attn.linear_attention)
        def gain_hook(module, args, output):
            if state.gain is not None and state.eav["mode"] == "apply_exp":
                output.mul_(state.gain.to(output))
            return output
        state.handles.append(attn.linear_attention.register_forward_hook(gain_hook))
    if len(state.originals) != 50:
        state.close()
        raise ValueError("FreeVideo effects did not install on all 50 pinned hybrid blocks")
    return state
