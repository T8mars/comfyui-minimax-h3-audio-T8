# T8 adaptation, 2026-11-05; upstream BenderBlender/ComfyUI-NAG-MiniMaxH3
# c63a04674047cfaa3c538286ad2041d6375735d2, MIT.
# See THIRD_PARTY_NOTICES/NAG-H3-LICENSE.txt. T8 changes: capability checks,
# identity bypass, scoped evidence and reference/patch preservation.
"""NAG for the MiniMax H3 audio-video DiT.

H3 is single-stream: text, condition/reference rows, audio and video share one packed
sequence ([text | cond | audio | video]) and one attention per block. NAG runs a negative
text stream next to it, the way ComfyUI-NAG does for Flux single blocks: every block gets
the negative text rows appended to the packed sequence, the non-text rows attend once to the
positive text and once to the negative text, and their attention output is replaced by the
NAG mix of the two. The non-text rows are shared by both streams, so the cost is one extra
attention per block over the non-text rows plus the negative text.

The negative text sits on the RoPE time axis so that it ends where the positive text ends,
right before the cond/target rows. RoPE only sees relative positions, so this is exactly the
layout a native sequence built around the negative prompt would have, shifted as a whole.

Everything is installed through the model patcher (a DIFFUSION_MODEL wrapper plus
dit/double_block patches), so stock samplers and guiders keep handling the nested AV latent.
"""

import functools
import logging
import math
from contextlib import nullcontext
from contextvars import ContextVar

import torch

import comfy.model_prefetch
import comfy.patcher_extension
from comfy.ldm.minimax.model import rope_rotation_table

from .nag_math import nag_

NAG_WRAPPER_KEY = "t8_nag_minimax_h3"
TEXT_TAG = 1  # modality tag of plain text rows, see MiniMaxH3Model._forward


def get_minimax_h3_model(model_patcher):
    model = model_patcher.get_model_object("diffusion_model")
    model = getattr(model, "_orig_mod", model)
    required = ("blocks", "preprocess_text_embeds", "rope_freqs")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError("NAG needs actual packed-text H3 interfaces: " + ", ".join(missing))
    return model


def pause_allocations():
    factory = getattr(comfy.model_prefetch, "pause_malloc_graph", None)
    return factory() if factory is not None else nullcontext()


def _text_mod_segments(start, length, row_base, tags):
    # same tag runs MiniMaxH3Model._forward builds for the positive text span
    if tags is None:
        return [(start, start + length, row_base + TEXT_TAG)]
    segments = []
    run_start = 0
    for i in range(1, length + 1):
        if i == length or tags[i] != tags[run_start]:
            segments.append((start + run_start, start + i, row_base + tags[run_start]))
            run_start = i
    return segments


def delegated_nag_attention(x, rope_freqs=None, transformer_options=None, *, forward,
                            seq_len, text_len, scale, tau, alpha, observed,
                            projection=None, preferred_attention=None):
    """Keep the selected QKV/RoPE/forward and head groups; guide its kernel calls.

    The original override still owns the positive stream. Only the additional
    negative stream uses dense attention because it has a different token layout.
    """
    options = dict(transformer_options or {})
    previous = options.get("optimized_attention_override")
    owner = getattr(forward, "__self__", forward)
    projection = projection if projection is not None else getattr(owner, "out_proj", None)
    if not callable(getattr(projection, "register_forward_pre_hook", None)):
        raise ValueError("NAG needs the selected attention's actual output projection hook")
    preferred_attention = (preferred_attention if preferred_attention is not None
                           else getattr(owner, "comfy_attention", None))
    preferred = getattr(preferred_attention, "function", None)
    positive_parts = []
    projection_calls = []
    call_scope = ContextVar("t8_nag_attention_scope", default=False)

    def guide(original, q, k, v, heads, **kwargs):
        # Core unwraps owned containers before invoking this ordinary override.
        if not kwargs.get("skip_reshape", False):
            q, k, v = [p.reshape(p.shape[0], p.shape[1], heads, -1).transpose(1, 2) for p in (q, k, v)]
            kwargs = {**kwargs, "skip_reshape": True}
        if q.shape[-2] != x.shape[0] or k.shape[-2] != x.shape[0]:
            raise ValueError("NAG attention consumer requires the complete packed positive + negative token rows")
        positive_options = dict(options)
        if previous is None:
            positive_options.pop("optimized_attention_override", None)
        else:
            positive_options["optimized_attention_override"] = previous
        positive_kwargs = {**kwargs, "transformer_options": positive_options}
        dense = preferred or original
        positive = (previous(original, q[..., :seq_len, :], k[..., :seq_len, :], v[..., :seq_len, :], heads,
                             **positive_kwargs) if previous is not None else
                    dense(q[..., :seq_len, :], k[..., :seq_len, :], v[..., :seq_len, :], heads,
                             **positive_kwargs))
        dense_options = {key: value for key, value in options.items() if key != "optimized_attention_override"}
        negative = dense(q[..., text_len:, :], k[..., text_len:, :], v[..., text_len:, :], heads,
                            **{**kwargs, "transformer_options": dense_options})
        if kwargs.get("skip_output_reshape", False):
            positive = positive.transpose(1, 2).reshape(positive.shape[0], seq_len, -1)
            negative = negative.transpose(1, 2).reshape(negative.shape[0], x.shape[0] - text_len, -1)
        # Head grouping must not change NAG's L1 norm over the full hidden axis.
        # Save positive groups, and guide the assembled negative rows once, at
        # the selected forward's real output projection (before its weights).
        positive_parts.append(positive)
        out = torch.cat([positive[:, :text_len], negative], dim=1)
        observed()
        if kwargs.get("skip_output_reshape", False):
            return out.reshape(out.shape[0], out.shape[1], heads, -1).transpose(1, 2)
        return out

    def before_projection(module, args):
        if not call_scope.get():
            return
        if not positive_parts:
            raise ValueError("NAG cannot guide this attention: its output projection ran without an optimized attention consumer")
        output = args[0]
        positive = torch.cat(positive_parts, dim=-1).squeeze(0)
        expected = (x.shape[0], positive.shape[-1])
        if output.shape != expected or projection_calls:
            raise ValueError("NAG requires one full packed attention output before its projection; "
                             f"expected {expected}, got {tuple(output.shape)}")
        nag_(positive[text_len:], output[text_len:seq_len], scale, tau, alpha)
        projection_calls.append(True)
        positive_parts.clear()

    handle = projection.register_forward_pre_hook(before_projection)
    token = call_scope.set(True)
    try:
        result = forward(x, rope_freqs=rope_freqs,
                         transformer_options={**options, "optimized_attention_override": guide})
        if not projection_calls:
            raise ValueError("NAG requires the selected attention to consume its exposed output projection")
        return result
    finally:
        call_scope.reset(token)
        handle.remove()


class NAGMiniMaxH3State:
    """Negative text stream of one diffusion model call, carried from block to block."""

    def __init__(self, negative):
        self.negative = negative  # [L_neg, hidden] negative text rows entering the next block
        self.packed = None  # last [positive sequence | negative text] block output
        self.mod_segments_src = None
        self.mod_segments = None
        self.rope_freqs_src = None
        self.rope_freqs = None

    def pack(self, h):
        packed = self.packed
        if (packed is not None and packed.shape[0] == h.shape[0] + self.negative.shape[0]
                and packed.data_ptr() == h.data_ptr() and packed.stride() == h.stride()):
            # h is the positive view of the previous block output, which the blocks update in place
            return packed
        return torch.cat([h, self.negative.to(h.dtype)])

    def unpack(self, out, seq_len):
        self.packed = out
        self.negative = out[seq_len:]

    def get_mod_segments(self, mod_segments, seq_len, tags):
        if self.mod_segments_src is not mod_segments:
            # text rows run at the video timestep: their mod row is t_row * 3 + modality tag
            row_base = mod_segments[0][2] // 3 * 3
            self.mod_segments = list(mod_segments) + _text_mod_segments(
                seq_len, self.negative.shape[0], row_base, tags)
            self.mod_segments_src = mod_segments
        return self.mod_segments

    def get_rope_freqs(self, rope_freqs, model, text_len):
        if rope_freqs is None:
            return None
        if self.rope_freqs_src is not rope_freqs:
            n = self.negative.shape[0]
            position_ids = torch.zeros(n, 3, dtype=torch.float64)
            position_ids[:, 0] = torch.arange(text_len - n, text_len, dtype=torch.float64)
            negative = rope_rotation_table(model.rope_freqs(position_ids, rope_freqs.device), rope_freqs.dtype)
            self.rope_freqs = torch.cat([rope_freqs, negative], dim=1)
            self.rope_freqs_src = rope_freqs
        return self.rope_freqs


class NAGMiniMaxH3Patch:
    def __init__(self, nag_negative, nag_scale, nag_tau, nag_alpha, nag_sigma_end):
        if len(nag_negative) != 1 or nag_negative[0][0].ndim != 3 or nag_negative[0][0].shape[0] != 1:
            raise ValueError("NAG needs one batch-1 text conditioning entry; merge text explicitly before encoding")
        self.negative_context = nag_negative[0][0].detach().clone()
        if self.negative_context.shape[1] == 0 or not bool(torch.isfinite(self.negative_context).all()):
            raise ValueError("NAG negative text must contain finite encoded tokens (an encoded empty prompt is valid)")
        tags = nag_negative[0][1].get("minimax_token_tags", None)
        self.negative_tags = None
        if tags is not None:
            if not torch.is_tensor(tags) or tags.numel() != self.negative_context.shape[1]:
                raise ValueError("NAG negative token tags must contain exactly one modality tag per text token")
            self.negative_tags = [int(t) for t in tags.view(-1).tolist()]
            if any(tag not in (0, 1, 2) for tag in self.negative_tags):
                raise ValueError("NAG negative token tags must use native H3 modality indices 0, 1 or 2")
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        self.nag_sigma_end = nag_sigma_end
        self.model = None
        self.negative_embeds = {}
        self._state = ContextVar("t8_nag_state", default=None)
        self.warned_attention_override = False
        self.warned_no_blocks = False
        self.observed_calls = 0
        self.observed_blocks = 0
        self.bypassed_calls = 0
        self.failed_calls = 0
        self.attention_kernel_calls = 0

    @property
    def state(self):
        return self._state.get()

    def snapshot(self):
        return dict(schema="t8.nag.observed.v1", scope="since_this_configuration",
                    active_diffusion_calls=self.observed_calls, guided_blocks=self.observed_blocks,
                    bypassed_diffusion_calls=self.bypassed_calls, failed_calls=self.failed_calls,
                    negative_tokens=int(self.negative_context.shape[1]),
                    attention_kernel_calls=self.attention_kernel_calls,
                    scale=self.nag_scale, tau=self.nag_tau, alpha=self.nag_alpha,
                    sigma_end=self.nag_sigma_end, negative_attention="dense",
                    quality_verified=False, state_retained=self.state is not None)

    def register(self, model_patcher):
        self.model = get_minimax_h3_model(model_patcher)
        # applying NAG again replaces the earlier settings: its block patches stay but go idle
        model_patcher.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, NAG_WRAPPER_KEY)
        model_patcher.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, NAG_WRAPPER_KEY,
                                           self.diffusion_model_wrapper)
        blocks_replace = model_patcher.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
        for i in range(len(self.model.blocks)):
            previous = blocks_replace.get(("double_block", i))
            if isinstance(previous, NAGMiniMaxH3BlockPatch):
                previous = previous.previous
            model_patcher.set_model_patch_replace(NAGMiniMaxH3BlockPatch(self, i, previous), "dit", "double_block", i)

    def is_active(self, x, transformer_options):
        if self.nag_scale <= 1.0 or x[0].shape[0] != 1:
            return False
        sigmas = transformer_options.get("sigmas", None)
        if sigmas is None or not bool(torch.all(sigmas >= self.nag_sigma_end)):
            return False
        return 0 in transformer_options.get("cond_or_uncond", [])

    def get_negative_embeds(self, context):
        key = (context.device, context.dtype)
        if key not in self.negative_embeds:
            # condition_proj + token refiner, as MiniMaxH3.extra_conds does for the positive prompt
            negative = self.negative_context.to(device=context.device, dtype=context.dtype)
            self.negative_embeds[key] = self.model.preprocess_text_embeds(negative)[0]
        return self.negative_embeds[key]

    def diffusion_model_wrapper(self, executor, x, timestep, context, transformer_options={}, **kwargs):
        token = self._state.set(None)
        try:
            if self.is_active(x, transformer_options):
                # The negative stream outlives single blocks: keep it out of
                # the allocation scope. Cleanup also covers preprocessing errors.
                with pause_allocations():
                    self._state.set(NAGMiniMaxH3State(self.get_negative_embeds(context)))
                self.observed_calls += 1
            else:
                self.bypassed_calls += 1
            out = executor(x, timestep, context, transformer_options, **kwargs)
            if self.state is not None and self.state.packed is None and not self.warned_no_blocks:
                logging.warning("NAG (MiniMax H3): no DiT block ran with NAG; a node applied after NAG "
                                "(e.g. Model Sparse Attention) probably replaced its block patches")
                self.warned_no_blocks = True
            return out
        except BaseException:
            self.failed_calls += 1
            raise
        finally:
            self._state.reset(token)

    def run_block(self, block_index, args, original_block):
        state = self.state
        h = args["img"]
        layout = args["layout"]
        seq_len = h.shape[0]
        _, text_len, kind = layout.segments[0]
        if state is None or kind != "text" or layout.seq_len != seq_len:
            return original_block(args)
        with pause_allocations():
            packed = state.pack(h)
            mod_segments = state.get_mod_segments(args["mod_segments"], seq_len, self.negative_tags)
            rope_freqs = state.get_rope_freqs(args["rope_freqs"], self.model, text_len)
        calls = []
        def observed():
            calls.append(True)
            self.attention_kernel_calls += 1
        attn = self.model.blocks[block_index].attn
        selected_attention = args.get("attention") or attn
        attention = functools.partial(delegated_nag_attention, forward=selected_attention,
            seq_len=seq_len, text_len=text_len, scale=self.nag_scale, tau=self.nag_tau,
            alpha=self.nag_alpha, observed=observed, projection=attn.out_proj,
            preferred_attention=getattr(attn, "comfy_attention", None))

        out = original_block({**args, "img": packed, "mod_segments": mod_segments,
                              "rope_freqs": rope_freqs, "attention": attention})
        state.unpack(out["img"], seq_len)
        self.observed_blocks += bool(calls)
        if not calls and not self.warned_attention_override:
            logging.warning("NAG: selected attention did not expose an optimized attention call; NAG mixing was not applied")
            self.warned_attention_override = True
        return {**out, "img": out["img"][:seq_len]}

    def to(self, device_or_dtype):
        if isinstance(device_or_dtype, torch.device):
            self.negative_embeds = {}
        return self

    def cleanup(self):
        self.negative_embeds = {}


class NAGMiniMaxH3BlockPatch:
    def __init__(self, nag_patch, block_index, previous):
        self.nag_patch = nag_patch
        self.block_index = block_index
        self.previous = previous

    def __call__(self, args, extra_args):
        if self.nag_patch.state is not None:
            # NAG goes innermost, right around the block, so other patches keep seeing the positive sequence only
            original_block = extra_args["original_block"]
            extra_args = {**extra_args, "original_block": functools.partial(
                self.nag_patch.run_block, self.block_index, original_block=original_block)}
        if self.previous is None:
            return extra_args["original_block"](args)
        return self.previous(args, extra_args)

    def to(self, device_or_dtype):
        self.nag_patch.to(device_or_dtype)
        if hasattr(self.previous, "to"):
            self.previous = self.previous.to(device_or_dtype)
        return self

    def cleanup(self):
        self.nag_patch.cleanup()
        if hasattr(self.previous, "cleanup"):
            self.previous.cleanup()

    def models(self):
        if hasattr(self.previous, "models"):
            return self.previous.models()
        return []


def apply_nag(model, nag_negative, nag_scale, nag_tau, nag_alpha, nag_sigma_end):
    parameters = (nag_scale, nag_tau, nag_alpha, nag_sigma_end)
    if not all(type(v) in (int, float) and math.isfinite(v) for v in parameters):
        raise ValueError("NAG parameters must be finite numbers")
    if nag_tau <= 0 or not 0 <= nag_alpha <= 1 or nag_sigma_end < 0:
        raise ValueError("NAG requires tau > 0, alpha in [0,1] and a nonnegative sigma cutoff")
    if nag_scale <= 1 or nag_alpha == 0:
        return model
    m = model.clone()
    patch = NAGMiniMaxH3Patch(nag_negative, nag_scale, nag_tau, nag_alpha, nag_sigma_end)
    patch.register(m)
    m.set_attachments(NAG_WRAPPER_KEY, patch)
    return m
