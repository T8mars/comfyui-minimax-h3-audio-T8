"""Opt-in LTX video FETA, delegating the complete installed attention forward.

No Core/global patch, audio hook, sampler, schedule change or persisted receipt.
The local selector observes post-norm/post-RoPE Q/K; the original forward keeps
its masks, backend, STG and gates. Unknown producers remain callable and are
reported as uncovered, not replaced with a different attention implementation.
"""
from collections import Counter
from dataclasses import asdict
import functools
import inspect
import json
import math

import torch
import comfy.patcher_extension as extension
from comfy.ldm.lightricks.model import GuideAttentionMask
from comfy.model_management import throw_exception_if_processing_interrupted

from .eav import EAVConfig

KEY = "t8_ltx_video_eav_v1"
RUNTIME_TYPE = "T8_LTX_EAV_RUNTIME"


def effective_object(model, path):
    """Respect pending parent replacements as well as exact method patches.

    Core applies object patches in insertion order. A later parent replacement
    supersedes an earlier child patch; get_model_object alone misses that case.
    """
    matching = [(key, value) for key, value in model.object_patches.items()
                if path == key or path.startswith(key + ".")]
    if not matching:
        return model.get_model_object(path)
    key, value = matching[-1]
    for part in path[len(key):].strip(".").split(".") if path != key else ():
        value = getattr(value, part)
    return value


def _latent_shape(latent):
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if (not isinstance(samples, torch.Tensor) or samples.ndim != 5
            or not samples.is_floating_point() or samples.shape[1] != 128
            or min(samples.shape) < 1 or samples.shape[2] < 2):
        raise ValueError("LTX EAV requires video LATENT [B,128,T,H,W] with T >= 2")
    return tuple(samples.shape)


def _sigmas(sigmas):
    if (not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1
            or not sigmas.is_floating_point() or not 2 <= sigmas.numel() <= 10000):
        raise ValueError("LTX EAV requires finite descending one-dimensional SIGMAS")
    values = tuple(float(x) for x in sigmas.detach().cpu())
    if (not all(math.isfinite(x) and 0 <= x <= 1 for x in values)
            or not all(a > b for a, b in zip(values, values[1:]))):
        raise ValueError("LTX EAV SIGMAS must strictly descend within [0,1]")
    return values


def chunked_cfi(q, k, frames, spatial, budget_bytes, interrupt):
    """Exact off-diagonal temporal softmax mean, batching heads/positions.

    q/k are [B,H,T*S,D] views. The budget bounds explicit statistic tensors,
    not the selected attention backend, model activations or BLAS workspace.
    FP32 statistics match the FETA definition; no full spatial Q/K copy.
    """
    if (q.ndim != 4 or k.shape != q.shape or q.shape[2] != frames * spatial
            or frames < 2 or spatial < 1 or q.shape[-1] < 1
            or q.device != k.device or not q.is_floating_point() or not k.is_floating_point()):
        raise ValueError("LTX EAV Q/K temporal layout is invalid")
    batch, heads, _, dim = q.shape
    # Q/K FP32 copies + score/probability matrices + reduction/finite masks,
    # including conservative scratch headroom. Backend workspace is separate.
    per_row = 4 * (4 * frames * dim + 4 * frames * frames + 8 * frames)
    if per_row > budget_bytes:
        raise ValueError("LTX EAV workspace cannot hold one temporal statistic row")
    rows = min(spatial, int(budget_bytes) // per_row)
    trace_sum = torch.zeros((), device=q.device, dtype=torch.float64)
    with torch.no_grad():
        for b in range(batch):
            for h in range(heads):
                q_grid = q[b, h].reshape(frames, spatial, dim).transpose(0, 1)
                k_grid = k[b, h].reshape(frames, spatial, dim).transpose(0, 1)
                for start in range(0, spatial, rows):
                    interrupt()
                    q_chunk = q_grid[start:start + rows].to(torch.float32)
                    k_chunk = k_grid[start:start + rows].to(torch.float32)
                    if not torch.isfinite(q_chunk).all() or not torch.isfinite(k_chunk).all():
                        raise ValueError("LTX EAV Q/K contains non-finite values")
                    scores = torch.matmul(q_chunk, k_chunk.transpose(-1, -2))
                    scores.mul_(dim ** -.5)
                    probabilities = scores.softmax(dim=-1)
                    if not torch.isfinite(probabilities).all():
                        raise ValueError("LTX EAV temporal statistic is non-finite")
                    trace_sum += probabilities.diagonal(dim1=-2, dim2=-1).sum(dtype=torch.float64)
                    del q_chunk, k_chunk, scores, probabilities
        count = batch * heads * spatial
        cfi = float(((count * frames - trace_sum) / (count * frames * (frames - 1))).cpu())
    interrupt()
    return cfi, rows, rows * per_row


class LTXEAVRuntime:
    """Bounded observations only; not proof that a candidate used this model."""
    def __init__(self, shape, sigmas, config, paths, frames, spatial):
        self.shape, self.sigmas, self.config = shape, sigmas, config
        self.paths, self.frames, self.spatial = tuple(paths), frames, spatial
        self.expected_wrappers = {}
        self.closed = True
        self.run_count = 0
        self._reset()

    def _reset(self):
        self.calls = [0] * len(self.paths)
        self.measured = [0] * len(self.paths)
        self.applied = [0] * len(self.paths)
        self.reasons = Counter()
        self.max_gain = self.min_gain = None
        self.workspace_bytes = 0
        self.lifecycle_observed = False

    def prepare(self, _patcher, _timestep, _options):
        if self.closed:
            self._reset()
            self.run_count += 1
            self.closed = False
        self.lifecycle_observed = True

    def cleanup(self, _patcher):
        self.closed = True

    def observe(self, index, q, k, heads, skip_reshape, interrupt):
        if (not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor)
                or type(heads) is not int or heads < 1):
            self.reasons["unsupported_qk_layout"] += 1
            return None
        if not skip_reshape and q.ndim == k.ndim == 3:
            if q.shape[-1] % heads or k.shape[-1] != q.shape[-1]:
                self.reasons["unsupported_qk_layout"] += 1
                return None
            q = q.reshape(q.shape[0], q.shape[1], heads, -1).transpose(1, 2)
            k = k.reshape(k.shape[0], k.shape[1], heads, -1).transpose(1, 2)
        tokens = self.frames * self.spatial
        if (q.ndim != 4 or k.ndim != 4 or q.shape[1] != heads
                or q.shape[2] != tokens or k.shape[2] < tokens
                or q.shape[0] < self.shape[0] or q.shape[0] % self.shape[0]
                or k[:, :, :tokens].shape != q.shape):
            self.reasons["unsupported_qk_layout"] += 1
            return None
        cfi, _rows, workspace = chunked_cfi(q, k[:, :, :tokens], self.frames,
            self.spatial, self.config.max_workspace_mib * 1024 ** 2, interrupt)
        gain = max(1., (self.frames + self.config.tau) * cfi)
        if not math.isfinite(gain) or gain > self.config.g_hard_limit:
            raise RuntimeError("LTX EAV gain exceeds configured hard limit; no clamp or fallback")
        self.measured[index] += 1
        self.min_gain = gain if self.min_gain is None else min(self.min_gain, gain)
        self.max_gain = gain if self.max_gain is None else max(self.max_gain, gain)
        self.workspace_bytes = max(self.workspace_bytes, workspace)
        return gain

    def report(self, model):
        replaced = [path for path, wrapper in self.expected_wrappers.items()
                    if effective_object(model, path) is not wrapper]
        return {"schema": "t8.ltx.eav.observations.v1", "config": asdict(self.config),
            "status": "disabled_identity" if self.config.mode == "disabled" else
                      "observed_not_certified" if sum(self.measured) else "unverified_no_effect_observed",
            "latent_shape": self.shape, "sigmas": self.sigmas,
            "progress_clock": "absolute_1_minus_native_sigma", "frames": self.frames,
            "spatial_tokens": self.spatial, "block_paths": self.paths,
            "forward_calls": self.calls[:], "measured_calls": self.measured[:],
            "applied_calls": self.applied[:], "skip_or_uncovered": dict(self.reasons),
            "unobserved_blocks": [i for i, n in enumerate(self.measured) if not n],
            "replaced_forward_paths": replaced, "gain_min": self.min_gain, "gain_max": self.max_gain,
            "statistic_workspace_peak_bytes": self.workspace_bytes,
            "statistic_masks_applied": False, "attention_masks_preserved": True,
            "lifecycle_observed": self.lifecycle_observed, "run_count": self.run_count,
            "sampler_completion_verified": False, "candidate_provenance_verified": False,
            "portable_cache_reuse_authorized": False, "quality_accepted": False,
            "warning": "CFI is a pre-mask temporal Q/K statistic. Opaque producers can bypass it. "
                       "Telemetry is local/shared by MODEL clones, not an execution receipt."}


def _forward_wrapper(original, runtime, index, interrupt):
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        signature = None

    @functools.wraps(original)
    def forward(*args, **kwargs):
        runtime.calls[index] += 1
        interrupt()
        if signature is None:
            runtime.reasons["opaque_forward_signature"] += 1
            return original(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if "transformer_options" not in bound.arguments or "x" not in bound.arguments:
            runtime.reasons["opaque_forward_signature"] += 1
            return original(*args, **kwargs)
        options = bound.arguments["transformer_options"]
        if not isinstance(options, dict):
            runtime.reasons["unsupported_transformer_options"] += 1
            return original(*args, **kwargs)
        if options.get("stg_skip_self_attn") or bound.arguments.get("context") is not None:
            runtime.reasons["stg_or_cross_attention_bypass"] += 1
            return original(*args, **kwargs)
        sigmas = options.get("sigmas")
        if not isinstance(sigmas, torch.Tensor) or not 1 <= sigmas.numel() <= 1024:
            runtime.reasons["missing_native_sigma"] += 1
            return original(*args, **kwargs)
        values = sigmas.detach().reshape(-1)
        sigma = float(values[0].cpu())
        if not math.isfinite(sigma) or not torch.all(values == values[0]):
            runtime.reasons["mixed_or_invalid_sigma"] += 1
            return original(*args, **kwargs)
        if not runtime.sigmas[-1] - 1e-7 <= sigma <= runtime.sigmas[0] + 1e-7:
            runtime.reasons["outside_bound_sigma_window"] += 1
            return original(*args, **kwargs)
        if not runtime.config.start_video_progress <= 1 - sigma <= runtime.config.end_video_progress:
            runtime.reasons["outside_effect_window"] += 1
            return original(*args, **kwargs)
        x, mask = bound.arguments["x"], bound.arguments.get("mask")
        tokens = runtime.frames * runtime.spatial
        guided = isinstance(mask, GuideAttentionMask)
        if (not isinstance(x, torch.Tensor) or x.ndim != 3
                or (guided and (mask.guide_start != tokens or x.shape[1] < tokens))
                or (not guided and x.shape[1] != tokens)):
            runtime.reasons["unsupported_target_token_layout"] += 1
            return original(*args, **kwargs)
        previous = options.get("optimized_attention_override")
        if previous is not None and not callable(previous):
            raise ValueError("Existing optimized_attention_override is not callable")
        gain = None
        selector_calls = 0

        def attention(func, q, k, v, heads, *a, **kw):
            nonlocal gain, selector_calls
            # In Core's GuideAttentionMask split, the first call is the target
            # prefix. Guide calls are delegated but never included in FETA.
            if selector_calls == 0:
                gain = runtime.observe(index, q, k, heads, kw.get("skip_reshape", False), interrupt)
            selector_calls += 1
            return previous(func, q, k, v, heads, *a, **kw) if previous else func(q, k, v, heads, *a, **kw)

        local_options = dict(options)
        local_options["optimized_attention_override"] = attention
        bound.arguments["transformer_options"] = local_options
        output = original(*bound.args, **bound.kwargs)
        if not selector_calls:
            runtime.reasons["producer_bypassed_selector"] += 1
        if gain is None or runtime.config.mode == "report_only":
            return output
        if not isinstance(output, torch.Tensor) or output.shape != x.shape:
            runtime.reasons["unsupported_output_layout"] += 1
            return output
        # Out-of-place: a foreign forward may return an alias, including x.
        # Only noisy video rows are enhanced; guide rows and audio are intact.
        runtime.applied[index] += 1
        scaled = output[:, :tokens] * gain
        return torch.cat((scaled, output[:, tokens:]), dim=1) if guided else scaled
    return forward


def apply_ltx_eav(model, ltx_latent, sigmas, eav_config, *, interrupt=None):
    if type(eav_config) is not EAVConfig:
        raise TypeError("LTX EAV requires the external Stage EAV config")
    shape, schedule = _latent_shape(ltx_latent), _sigmas(sigmas)
    if model.get_attachment(KEY) is not None:
        raise ValueError("This MODEL already has this LTX EAV owner; branch before applying another config")
    try:
        patch = tuple(effective_object(model, "diffusion_model.patchifier").patch_size)
        blocks = effective_object(model, "diffusion_model.transformer_blocks")
    except AttributeError as error:
        raise ValueError("LTX EAV needs a video patchifier and transformer_blocks") from error
    if (len(patch) != 3 or patch[0] != 1 or any(type(n) is not int or n < 1 for n in patch)
            or any(size % n for size, n in zip(shape[2:], patch)) or blocks is None or not len(blocks)):
        raise ValueError("LTX EAV needs a video patchifier and transformer_blocks with aligned token geometry")
    paths = [f"diffusion_model.transformer_blocks.{i}.attn1.forward" for i in range(len(blocks))]
    runtime = LTXEAVRuntime(shape, schedule, eav_config, paths, shape[2],
                           (shape[3] // patch[1]) * (shape[4] // patch[2]))
    if eav_config.mode == "disabled":
        return model, runtime, json.dumps(runtime.report(model))
    interrupt = interrupt or throw_exception_if_processing_interrupted
    clone = model.clone()
    for index, path in enumerate(paths):
        original = effective_object(model, path)
        if not callable(original):
            raise TypeError("LTX video attention forward must be callable")
        wrapper = _forward_wrapper(original, runtime, index, interrupt)
        # Move only this method entry after existing parent object patches.
        clone.object_patches.pop(path, None)
        clone.add_object_patch(path, wrapper)
        runtime.expected_wrappers[path] = wrapper
    clone.set_attachments(KEY, runtime)
    clone.add_callback_with_key(extension.CallbacksMP.ON_PREPARE_STATE, KEY, runtime.prepare)
    clone.add_callback_with_key(extension.CallbacksMP.ON_CLEANUP, KEY, runtime.cleanup)
    return clone, runtime, json.dumps(runtime.report(clone))


def audit_ltx_eav(model, candidate_latent, runtime):
    if type(runtime) is not LTXEAVRuntime:
        raise TypeError("LTX EAV audit requires its live local runtime token")
    if runtime.config.mode != "disabled" and model.get_attachment(KEY) is not runtime:
        raise ValueError("LTX EAV runtime does not belong to the connected MODEL")
    if _latent_shape(candidate_latent) != runtime.shape:
        raise ValueError("LTX EAV candidate geometry differs from the bound stage")
    if not torch.isfinite(candidate_latent["samples"]).all():
        raise ValueError("LTX EAV candidate contains non-finite values")
    return candidate_latent, json.dumps(runtime.report(model))
