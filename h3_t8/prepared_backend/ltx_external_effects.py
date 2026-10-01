"""Instance-only adapters over the pinned upstream LTX Attention protocol.

The native forward retains projection, RoPE, masks, perturbation and gating.
Only video attn1 (FETA) and video attn2 (Relay) are touched; original audio
context and audio/intermodal delegates are left intact. No global/Core patch.
"""
from collections import Counter
from contextlib import contextmanager
import functools
import inspect
import math

import torch

try:  # Isolated worker scripts and ordinary package tests use the same module.
    from .ltx_effects_contract import validate_effects
    from .backend_files import sha
except ImportError:
    from ltx_effects_contract import validate_effects
    from backend_files import sha


def temporal_cfi(q, k, heads, frames, spatial, budget):
    if (q.ndim != 3 or k.shape != q.shape or q.shape[1] != frames * spatial
            or type(heads) is not int or heads < 1 or q.shape[-1] % heads):
        raise ValueError("Prepared FETA requires aligned native video Q/K")
    dim = q.shape[-1] // heads
    per_row = 4 * (4 * frames * dim + 4 * frames * frames + 8 * frames)
    rows = min(spatial, budget // per_row)
    if rows < 1:
        raise ValueError("Prepared FETA budget cannot hold a temporal row")
    q = q.view(q.shape[0], frames, spatial, heads, dim)
    k = k.view(k.shape[0], frames, spatial, heads, dim)
    trace = torch.zeros((), dtype=torch.float64, device=q.device)
    for batch in range(q.shape[0]):
        for head in range(heads):
            for start in range(0, spatial, rows):
                a = q[batch, :, start:start + rows, head].transpose(0, 1).float()
                b = k[batch, :, start:start + rows, head].transpose(0, 1).float()
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise ValueError("Prepared FETA has non-finite Q/K")
                probability = (torch.matmul(a, b.transpose(-1, -2)) * dim ** -.5).softmax(-1)
                if not torch.isfinite(probability).all():
                    raise ValueError("Prepared FETA statistic is non-finite")
                trace += probability.diagonal(dim1=-2, dim2=-1).sum(dtype=torch.float64)
    count = q.shape[0] * heads * spatial
    return float(((count * frames - trace) / (count * frames * (frames - 1))).cpu()), rows * per_row


def relay_penalty(times, event, epsilon):
    start = event["start_frame"] * (5. / 3.)
    end = (event["end_frame_exclusive"] - 1) * (5. / 3.)
    midpoint, half = (start + end) / 2, (end - start) / 2
    window = max(half - 2, 0)
    sigma = (half - window) / math.sqrt(2 * math.log(1 / epsilon))
    return (times.float() - midpoint).abs().sub(window).clamp_min(0).square() / (2 * sigma * sigma)


def _mask_rows(mask, start, stop, batch, query, keys):
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor) or mask.ndim not in (2, 3, 4):
        raise ValueError("Prepared Relay cannot interpret the selected attention mask")
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[:, None]
    if mask.shape[0] not in (1, batch) or mask.shape[-1] != keys or mask.shape[-2] not in (1, query):
        raise ValueError("Prepared Relay mask does not broadcast to native Q/K")
    return mask if mask.shape[-2] == 1 else mask[..., start:stop, :]


class PreparedLTXEffects:
    def __init__(self, effects, geometry, prompt, identities):
        validate_effects(effects, geometry, prompt, set(identities))
        self.effects, self.geometry = effects, geometry
        self.frames = (geometry["frames"] - 1) // 8 + 1
        self.spatial = geometry["height"] // 32 * (geometry["width"] // 32)
        self.sigma = None
        self.times = None
        self.counts = Counter()
        self.reasons = Counter()
        self.gains = []
        self.peak_statistic_bytes = self.peak_bias_bytes = 0
        self.contexts = []
        relay = effects["relay"]
        if relay is not None and relay["caches"]:
            from runtime.prompt_cache import load_cache
            for cache in relay["caches"]:
                if sha(cache["path"]) != cache["sha256"] or identities[cache["path"]] != cache["sha256"]:
                    raise ValueError("Prepared Relay cache content identity differs")
                loaded = load_cache(cache["path"], prompt=cache["prompt"], torch_module=torch)
                self.contexts.append(loaded.payload["contexts"]["video"])

    def bind_context(self, video_context):
        relay = self.effects["relay"]
        if (relay is None or not relay["caches"] or relay["mode"] != "apply_exp"
                or not self.counts["relay_installed_blocks"]):
            return video_context
        pieces = [video_context] + [value.to(video_context.device) for value in self.contexts]
        if any(value.dtype != video_context.dtype or value.shape[0] != video_context.shape[0]
               or value.shape[-1] != video_context.shape[-1] for value in pieces):
            raise ValueError("Prepared Relay event contexts differ from selected video context")
        self.counts["video_context_segments_bound"] = len(pieces)
        return torch.cat(pieces, dim=1)

    def denoiser(self, original):
        @functools.wraps(original)
        def delegate(transformer, video_state, audio_state, sigmas, step_index):
            self.sigma = float(sigmas[step_index].detach().cpu())
            positions = video_state.positions
            if positions.ndim != 4 or positions.shape[1] != 3 or positions.shape[2] != self.frames * self.spatial:
                raise ValueError("Prepared effects require actual upstream video token positions")
            self.times = positions[0, 0, :, 0].detach() * self.geometry["fps"] * (5. / 3.)
            if not torch.isfinite(self.times).all():
                raise ValueError("Prepared effect time coordinates are non-finite")
            try:
                return original(transformer, video_state, audio_state, sigmas, step_index)
            finally:
                self.sigma, self.times = None, None
        return delegate

    def _wrap(self, attention, original, masked, kind):
        def selected(q, k, v, heads, mask=None):
            self.counts[kind + "_selector_calls"] += 1
            if kind == "eav":
                config = self.effects["eav"]
                progress = 1 - self.sigma if self.sigma is not None else None
                if progress is None or not config["start_video_progress"] <= progress <= config["end_video_progress"]:
                    self.reasons["eav_outside_progress_window"] += 1
                elif q.shape == k.shape and q.ndim == 3 and q.shape[1] == self.frames * self.spatial:
                    cfi, workspace = temporal_cfi(q, k, heads, self.frames, self.spatial,
                                                 config["max_workspace_mib"] * 1024 ** 2)
                    gain = max(1., (self.frames + config["tau"]) * cfi)
                    if not math.isfinite(gain) or gain > config["g_hard_limit"]:
                        raise RuntimeError("Prepared FETA gain exceeds the configured hard limit")
                    self.counts["eav_measured"] += 1
                    self.gains.append(gain)
                    self.peak_statistic_bytes = max(self.peak_statistic_bytes, workspace)
                    attention._t8_prepared_gain = gain
                else:
                    self.reasons["eav_unsupported_qk_layout"] += 1
                return masked(q, k, v, heads, mask) if mask is not None else original(q, k, v, heads)
            relay = self.effects["relay"]
            if relay["mode"] != "apply_exp" or not relay["caches"]:
                return masked(q, k, v, heads, mask) if mask is not None else original(q, k, v, heads)
            if (self.times is None or q.ndim != 3 or q.shape[1] != len(self.times)
                    or k.shape[1] != 1024 * (1 + len(self.contexts))):
                self.reasons["relay_unsupported_qk_layout"] += 1
                return masked(q, k, v, heads, mask) if mask is not None else original(q, k, v, heads)
            mask_heads = mask.shape[1] if isinstance(mask, torch.Tensor) and mask.ndim == 4 else 1
            if mask_heads not in (1, heads):
                raise ValueError("Prepared Relay mask has incompatible head geometry")
            per_row = q.shape[0] * mask_heads * k.shape[1] * 4 * 4
            rows = min(q.shape[1], relay["max_workspace_mib"] * 1024 ** 2 // per_row)
            if rows < 1:
                raise ValueError("Prepared Relay workspace cannot hold one attention bias row")
            results = []
            for start in range(0, q.shape[1], rows):
                stop = min(start + rows, q.shape[1])
                bias = torch.zeros((q.shape[0], 1, stop - start, k.shape[1]), device=q.device, dtype=q.dtype)
                for index, event in enumerate(relay["plan"]["events"], 1):
                    cost = relay_penalty(self.times[start:stop], event, relay["plan"]["epsilon"])
                    bias[..., index * 1024:(index + 1) * 1024] = -cost[None, None, :, None].to(bias.dtype)
                existing = _mask_rows(mask, start, stop, q.shape[0], q.shape[1], k.shape[1])
                if existing is not None:
                    if existing.dtype == torch.bool:
                        bias = bias.masked_fill(~existing, -torch.inf)
                    elif existing.is_floating_point():
                        bias = bias + existing
                    else:
                        raise ValueError("Prepared Relay mask must be boolean or additive floating")
                self.peak_bias_bytes = max(self.peak_bias_bytes, bias.numel() * bias.element_size())
                results.append(masked(q[:, start:stop], k, v, heads, bias))
                self.counts["relay_bias_chunks"] += 1
            self.counts["relay_applied"] += 1
            return torch.cat(results, dim=1)
        return selected

    @contextmanager
    def installed(self, model):
        from ltx_core.model.transformer.attention import Attention
        changes = []
        try:
            for block in model.transformer_blocks:
                for kind, name in (("eav", "attn1"), ("relay", "attn2")):
                    config = self.effects[kind]
                    if config is None or config["mode"] == "disabled":
                        continue
                    attention = getattr(block, name, None)
                    forward = getattr(attention, "forward", None)
                    if (type(attention) is not Attention or getattr(forward, "__func__", None) is not Attention.forward
                            or not callable(attention.attention_function) or not callable(attention.masked_attention_function)):
                        self.reasons[kind + "_unknown_producer_preserved"] += 1
                        continue
                    original, masked = attention.attention_function, attention.masked_attention_function
                    selected = self._wrap(attention, original, masked, kind)
                    old_forward = attention.__dict__.get("forward")
                    changes.append((attention, original, masked, old_forward))
                    attention.attention_function = selected
                    attention.masked_attention_function = selected
                    if kind == "eav":
                        signature = inspect.signature(forward)
                        def wrapper(*args, _attention=attention, _forward=forward, _signature=signature, **kwargs):
                            bound = _signature.bind(*args, **kwargs)
                            bound.apply_defaults()
                            _attention._t8_prepared_gain = None
                            try:
                                result = _forward(*args, **kwargs)
                                gain = _attention._t8_prepared_gain
                                if self.effects["eav"]["mode"] == "apply_exp" and gain is not None:
                                    # all_perturbed bypasses the selector, so gain stays None.
                                    self.counts["eav_applied"] += 1
                                    return result * gain
                                return result
                            finally:
                                del _attention._t8_prepared_gain
                        attention.forward = wrapper
                    self.counts[kind + "_installed_blocks"] += 1
            yield self
        finally:
            for attention, original, masked, forward in reversed(changes):
                attention.attention_function, attention.masked_attention_function = original, masked
                if forward is None:
                    attention.__dict__.pop("forward", None)
                else:
                    attention.forward = forward

    def report(self):
        return {"schema": "t8.prepared_ltx.effects.observations.v1", "counts": dict(self.counts),
                "uncovered": dict(self.reasons), "gain_min": min(self.gains) if self.gains else None,
                "gain_max": max(self.gains) if self.gains else None,
                "peak_statistic_bytes": self.peak_statistic_bytes, "peak_bias_bytes": self.peak_bias_bytes,
                "native_mask_and_projection_delegated": True, "audio_context_unchanged": True,
                "event_text_boundary": "actual_independent_post_connector_1024_key_cache_per_event",
                "quality_accepted": False, "paper_ltx_qualified": False}
