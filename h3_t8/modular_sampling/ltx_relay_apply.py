"""Opt-in LTX video cross-attention Prompt Relay, delegated to Core.

Only video attn2.forward is wrapped. The original callable performs Q/K/V,
norm, gates and output projection. A local optimized-attention selector adds
bounded event penalties to the selected positive video queries. It preserves
the pre-existing selector and mask; opaque non-delegating producers are
reported as unverified rather than rejected solely for their presence.
"""
from collections import Counter
import functools
import inspect
import json
import math

import torch
import comfy.patcher_extension as extension
from comfy.ldm.lightricks.symmetric_patchifier import latent_to_pixel_coords
from comfy.model_management import throw_exception_if_processing_interrupted

from ..patch_stack_policy import merge_attention_bias, slice_attention_mask
from ..prompt_relay_advanced import _paper_parameters, prompt_relay_penalty
from .ltx_eav import _latent_shape, _sigmas, effective_object
from .ltx_relay_text import validate_ltx_relay_binding


KEY = "t8_ltx_video_prompt_relay_v1"
RUNTIME_TYPE = "T8_LTX_RELAY_RUNTIME"
MODES = ("disabled", "report_only", "apply_exp")


def _native_query_times(model, shape):
    diffusion = effective_object(model, "diffusion_model")
    patchifier = diffusion.patchifier
    patch = tuple(patchifier.patch_size)
    scales = tuple(diffusion.vae_scale_factors)
    if (len(patch) != 3 or patch[0] != 1 or any(type(p) is not int or p < 1 for p in patch)
            or any(size % p for size, p in zip(shape[2:], patch))
            or len(scales) != 3 or scales[0] != 8):
        return None, "unsupported_ltx_time_grid"
    coords = patchifier.get_latent_coords(shape[2], shape[3], shape[4], 1, torch.device("cpu"))
    pixel = latent_to_pixel_coords(coords, scales, causal_fix=bool(diffusion.causal_temporal_positioning))
    times = pixel[0, 0, :, 0].to(torch.float32).contiguous() * (5. / 3.)
    if not torch.isfinite(times).all() or times.numel() != shape[2] * (shape[3] // patch[1]) * (shape[4] // patch[2]):
        return None, "unsupported_ltx_time_grid"
    return times, None


class LTXRelayRuntime:
    def __init__(self, shape, sigmas, binding, mode, max_workspace_mib, paths, times, grid_reason):
        self.shape, self.sigmas = shape, sigmas
        self.binding, self.mode = binding, mode
        self.max_workspace_mib = max_workspace_mib
        self.paths, self.times, self.grid_reason = tuple(paths), times, grid_reason
        self.events = tuple({**_paper_parameters(event["start_frame"], event["end_frame_exclusive"],
                                                  binding.plan["epsilon"]),
                             "text_key_start": binding.token_spans[index + 1][0],
                             "text_key_end": binding.token_spans[index + 1][1]}
                            for index, event in enumerate(binding.plan["events"]))
        self.expected_wrappers = {}
        self.closed, self.run_count = True, 0
        self._reset()

    def _reset(self):
        self.forward_calls = [0] * len(self.paths)
        self.observed_calls = [0] * len(self.paths)
        self.applied_calls = [0] * len(self.paths)
        self.backend_delegate_calls = 0
        self.bias_chunks = 0
        self.peak_bias_bytes = 0
        self.reasons = Counter()
        self.lifecycle_observed = False
        self.extra_text_keys = set()

    def prepare(self, _patcher, _timestep, _options):
        if self.closed:
            self._reset()
            self.run_count += 1
            self.closed = False
        self.lifecycle_observed = True

    def cleanup(self, _patcher):
        self.closed = True

    def report(self, model):
        replaced = [path for path, wrapper in self.expected_wrappers.items()
                    if effective_object(model, path) is not wrapper]
        return {"schema": "t8.ltx.relay.observations.v1", "mode": self.mode,
                "status": "disabled_identity" if self.mode == "disabled" else
                          "observed_not_certified" if sum(self.observed_calls) else "unverified_no_effect_observed",
                "plan_hash": self.binding.plan["plan_hash"], "latent_shape": self.shape,
                "sigmas": self.sigmas, "text_key_spans": self.binding.token_spans,
                "raw_text_key_count": self.binding.token_count,
                "extra_connector_key_counts": sorted(self.extra_text_keys),
                "block_paths": self.paths, "forward_calls": self.forward_calls[:],
                "observed_calls": self.observed_calls[:], "applied_calls": self.applied_calls[:],
                "bias_chunks": self.bias_chunks, "selected_backend_delegate_calls": self.backend_delegate_calls,
                "peak_explicit_bias_bytes": self.peak_bias_bytes,
                "skip_or_uncovered": dict(self.reasons), "replaced_forward_paths": replaced,
                "native_time_grid": self.grid_reason or "Core_ltx_pixel_coords",
                "lifecycle_observed": self.lifecycle_observed, "run_count": self.run_count,
                "candidate_provenance_verified": False, "sampler_completion_verified": False,
                "paper_qualified_for_ltx": False, "portable_cache_reuse_authorized": False,
                "quality_accepted": False,
                "warning": "Opaque producers may ignore the supplied bias. Text connectors can append registers; "
                           "only bound raw segment keys are biased. Telemetry is not a candidate receipt."}


def _normalize_mask(mask, batch, query, keys):
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        return None
    if mask.ndim == 1:
        mask = mask[None, None, None, :]
    elif mask.ndim == 2:
        mask = (mask[:, None, None, :] if mask.shape[0] in (1, batch)
                else mask[None, None, :, :])
    elif mask.ndim == 3:
        mask = mask[:, None, :, :]
    elif mask.ndim != 4:
        return None
    if mask.shape[-1] != keys or mask.shape[-2] not in (1, query) or mask.shape[0] not in (1, batch):
        return None
    return mask


def _slice_batch_mask(mask, batch_start, batch_end):
    if mask is None or mask.shape[0] == 1:
        return mask
    return mask[batch_start:batch_end]


def _forward_wrapper(original, runtime, index, interrupt):
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        signature = None

    @functools.wraps(original)
    def forward(*args, **kwargs):
        runtime.forward_calls[index] += 1
        interrupt()
        if signature is None:
            runtime.reasons["opaque_forward_signature"] += 1
            return original(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        x, context = bound.arguments.get("x"), bound.arguments.get("context")
        options = bound.arguments.get("transformer_options")
        if (not isinstance(options, dict) or not isinstance(x, torch.Tensor)
                or not isinstance(context, torch.Tensor) or x.ndim != 3 or context.ndim != 3):
            runtime.reasons["unsupported_cross_attention_call"] += 1
            return original(*args, **kwargs)
        if runtime.times is None:
            runtime.reasons[runtime.grid_reason] += 1
            return original(*args, **kwargs)
        if len(runtime.events) < 2:
            runtime.reasons["zero_or_one_event_identity"] += 1
            return original(*args, **kwargs)
        target = runtime.times.numel()
        if x.shape[1] < target or context.shape[1] < runtime.binding.token_count:
            runtime.reasons["unsupported_video_or_text_layout"] += 1
            return original(*args, **kwargs)
        branches = options.get("cond_or_uncond")
        if (not isinstance(branches, (list, tuple)) or not branches
                or any(type(branch) is not int or branch not in (0, 1) for branch in branches)
                or x.shape[0] % len(branches) or context.shape[0] != x.shape[0]):
            runtime.reasons["unknown_cfg_branch_layout"] += 1
            return original(*args, **kwargs)
        sigmas = options.get("sigmas")
        if not isinstance(sigmas, torch.Tensor) or not 1 <= sigmas.numel() <= 1024:
            runtime.reasons["missing_native_sigma"] += 1
            return original(*args, **kwargs)
        values = sigmas.detach().reshape(-1)
        sigma = float(values[0].cpu())
        if (not math.isfinite(sigma) or not torch.all(values == values[0])
                or not runtime.sigmas[-1] - 1e-7 <= sigma <= runtime.sigmas[0] + 1e-7):
            runtime.reasons["outside_bound_sigma_window"] += 1
            return original(*args, **kwargs)
        previous = options.get("optimized_attention_override")
        if previous is not None and not callable(previous):
            raise TypeError("Existing LTX optimized_attention_override must be callable")
        selector_calls = 0

        def attention(func, q, k, v, heads, *a, **kw):
            nonlocal selector_calls
            selector_calls += 1
            mask = kw.get("mask")
            q_tokens, k_tokens = q.shape[-2], k.shape[-2]
            normalized = _normalize_mask(mask, q.shape[0], q_tokens, k_tokens)
            if (a or q.ndim not in (3, 4) or k.ndim != q.ndim or v.ndim != q.ndim
                    or q_tokens < target or k_tokens < runtime.binding.token_count
                    or (mask is not None and normalized is None)
                    or q.shape[0] != x.shape[0]):
                runtime.reasons["selector_layout_or_mask_uncovered"] += 1
                return previous(func, q, k, v, heads, *a, **kw) if previous else func(q, k, v, heads, *a, **kw)
            runtime.observed_calls[index] += 1
            runtime.extra_text_keys.add(k_tokens - runtime.binding.token_count)

            def delegate(q_local, k_local, v_local, mask_local):
                local = dict(kw)
                local["mask"] = mask_local
                before = runtime.backend_delegate_calls
                def counted(*inside_args, **inside_kwargs):
                    runtime.backend_delegate_calls += 1
                    return func(*inside_args, **inside_kwargs)
                output = (previous(counted, q_local, k_local, v_local, heads, **local)
                          if previous else counted(q_local, k_local, v_local, heads, **local))
                if previous and runtime.backend_delegate_calls == before:
                    runtime.reasons["existing_override_backend_unverified"] += 1
                return output

            if runtime.mode == "report_only" or 0 not in branches:
                if 0 not in branches:
                    runtime.reasons["unconditional_only"] += 1
                return previous(func, q, k, v, heads, *a, **kw) if previous else func(q, k, v, heads, *a, **kw)

            # Preserve negative CFG rows bit-for-bit by obtaining their native
            # full-batch result before replacing only positive video rows.
            mixed = 1 in branches
            original_output = delegate(q, k, v, mask) if mixed else None
            group_size = q.shape[0] // len(branches)
            results = []
            for branch_index, branch in enumerate(branches):
                begin, end = branch_index * group_size, (branch_index + 1) * group_size
                if branch == 1:
                    continue
                q_group, k_group, v_group = q[begin:end], k[begin:end], v[begin:end]
                group_mask = _slice_batch_mask(normalized, begin, end)
                # Covers explicit bias and mask materialization only; the
                # selected attention backend owns its separate workspace.
                budget = runtime.max_workspace_mib * 1024 ** 2
                per_row = max(1, group_size * k_tokens * 4 * 4)
                rows = min(target, budget // per_row)
                if rows < 1:
                    raise ValueError("LTX Relay workspace cannot hold one attention bias row")
                pieces = []
                for start in range(0, target, rows):
                    interrupt()
                    stop = min(start + rows, target)
                    times = runtime.times[start:stop].to(device=q.device)
                    bias = torch.zeros((group_size, 1, stop - start, k_tokens),
                                       device=q.device, dtype=q.dtype)
                    for event in runtime.events:
                        penalty = prompt_relay_penalty(times, event).to(dtype=bias.dtype)
                        bias[:, :, :, event["text_key_start"]:event["text_key_end"]] = -penalty[None, None, :, None]
                    base_mask = slice_attention_mask(group_mask, start, stop)
                    merged = merge_attention_bias(bias, base_mask)
                    runtime.peak_bias_bytes = max(runtime.peak_bias_bytes, merged.numel() * merged.element_size())
                    pieces.append(delegate(q_group[..., start:stop, :], k_group, v_group, merged))
                    runtime.bias_chunks += 1
                if q_tokens > target:
                    tail_mask = slice_attention_mask(group_mask, target, q_tokens)
                    pieces.append(delegate(q_group[..., target:, :], k_group, v_group, tail_mask))
                results.append((begin, end, torch.cat(pieces, dim=-2)))
            if mixed:
                result = original_output.clone()
                for begin, end, value in results:
                    if result.ndim == 3:
                        result[begin:end, :target] = value[:, :target]
                    else:
                        result[begin:end, :, :target] = value[:, :, :target]
                # Keep guide-query tail and negative rows from the full native call.
            else:
                result = torch.cat([value for _, _, value in results], dim=0)
            runtime.applied_calls[index] += 1
            return result

        local_options = dict(options)
        local_options["optimized_attention_override"] = attention
        bound.arguments["transformer_options"] = local_options
        output = original(*bound.args, **bound.kwargs)
        if not selector_calls:
            runtime.reasons["producer_bypassed_selector"] += 1
        return output
    return forward


def apply_ltx_relay(model, ltx_latent, sigmas, positive, text_binding,
                    mode="report_only", max_workspace_mib=32, *, interrupt=None):
    shape, schedule = _latent_shape(ltx_latent), _sigmas(sigmas)
    binding = validate_ltx_relay_binding(text_binding, positive, ltx_latent)
    if mode not in MODES:
        raise ValueError(f"Unknown LTX Relay mode {mode!r}")
    if type(max_workspace_mib) is not int or not 1 <= max_workspace_mib <= 512:
        raise ValueError("LTX Relay max_workspace_mib must be within 1..512")
    if model.get_attachment(KEY) is not None:
        raise ValueError("This MODEL already has an LTX Relay owner; branch before applying again")
    blocks = effective_object(model, "diffusion_model.transformer_blocks")
    if blocks is None or not len(blocks):
        raise ValueError("LTX Relay requires video transformer blocks")
    paths = [f"diffusion_model.transformer_blocks.{i}.attn2.forward" for i in range(len(blocks))]
    times, grid_reason = _native_query_times(model, shape)
    runtime = LTXRelayRuntime(shape, schedule, binding, mode, max_workspace_mib,
                              paths, times, grid_reason)
    if mode == "disabled":
        return model, positive, runtime, json.dumps(runtime.report(model))
    interrupt = interrupt or throw_exception_if_processing_interrupted
    clone = model.clone()
    for index, path in enumerate(paths):
        original = effective_object(model, path)
        if not callable(original):
            raise TypeError("LTX video cross-attention forward must be callable")
        wrapper = _forward_wrapper(original, runtime, index, interrupt)
        clone.object_patches.pop(path, None)
        clone.add_object_patch(path, wrapper)
        runtime.expected_wrappers[path] = wrapper
    clone.set_attachments(KEY, runtime)
    clone.add_callback_with_key(extension.CallbacksMP.ON_PREPARE_STATE, KEY, runtime.prepare)
    clone.add_callback_with_key(extension.CallbacksMP.ON_CLEANUP, KEY, runtime.cleanup)
    return clone, positive, runtime, json.dumps(runtime.report(clone))


def audit_ltx_relay(model, candidate_latent, runtime):
    if type(runtime) is not LTXRelayRuntime:
        raise TypeError("LTX Relay audit requires its live runtime token")
    if runtime.mode != "disabled" and model.get_attachment(KEY) is not runtime:
        raise ValueError("LTX Relay runtime does not belong to connected MODEL")
    if _latent_shape(candidate_latent) != runtime.shape:
        raise ValueError("LTX Relay candidate geometry differs from bound stage")
    if not torch.isfinite(candidate_latent["samples"]).all():
        raise ValueError("LTX Relay candidate contains non-finite values")
    return candidate_latent, json.dumps(runtime.report(model))
