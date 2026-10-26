"""Finite stacked-hop PTD consumer on T8's existing physical-audio clock.

This is the Nynxz a50419a7 tangent-clock contract, NOT Jalen's 1.0 native
carried-variable path. The audio absolute head is corrected by chord/tangent
before the existing PTD audio hook multiplies its tangent. No EPFL grid changes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import math
import threading

import torch
import torch.nn.functional as F
import comfy.model_management
from comfy.ldm.minimax import model as native

from .h3_projection_chunks import _row_local, _source_exact, shared_network_lease
from .patch_stack_policy import warn_patch_stack
from .ptd_advanced import KEY, Expert, PTDController, PTDModelWrapper, _FRAME
from .ptd_compact_checkpoint import CompactCheckpoint, load_compact_file
from .ptd_sampling import (
    T_MAX, audio_sigma, audio_sigma_prime, initial_state, map_streams,
    pair_streams, streams,
)

FORMAT = "t8.taobao_stacked_ptd.tangent0999.v1"
VIDEO_SHIFT, AUDIO_SHIFT = 12.0, 3.0


def _shifted(shift, values):
    return shift * values / (1.0 + (shift - 1.0) * values)


def compact_grid(hops, steps):
    if type(hops) is not int or type(steps) is not int or hops < 1 or steps < 1 or steps % hops:
        raise ValueError("Stacked PTD needs positive divisible fine steps/hops")
    fine = _shifted(VIDEO_SHIFT, torch.linspace(1., 0., steps + 1, dtype=torch.float64))
    width = steps // hops
    return torch.tensor([T_MAX * float(fine[h * width]) for h in range(hops)] + [0.], dtype=torch.float64)


def compact_plans(hops, steps):
    compact_grid(hops, steps)
    width, plans = steps // hops, []
    for shift in (VIDEO_SHIFT, AUDIO_SHIFT):
        times = 1.0 - _shifted(shift, torch.linspace(1., 0., steps + 1, dtype=torch.float64))
        sizes, rows = times.diff(), []
        for hop in range(hops):
            start = hop * width
            block = sizes[start:start + width]
            if not bool(torch.isfinite(block).all()) or not bool((block > 0).all()):
                raise ValueError("Invalid stacked PTD fine interval widths")
            row = torch.zeros(steps, dtype=torch.float64)
            row[start:start + width] = block / block.sum()
            rows.append(row.float())
        plans.append(torch.stack(rows))
    return tuple(plans)


def audio_chord_corrections(grid):
    return tuple((audio_sigma(float(a)) - audio_sigma(float(b))) /
                 (float(a) - float(b)) / audio_sigma_prime(float(a))
                 for a, b in zip(grid[:-1], grid[1:]))


def _finite_strength(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class CompactPTDController(PTDController):
    """Reuse request, exact-hop, lease, fc2/refiner and restoration ownership."""

    format = FORMAT

    def __init__(self, network, checkpoint, *, file_report=None, lora_strength=1., head_strength=1.):
        if not isinstance(checkpoint, CompactCheckpoint):
            raise TypeError("Stacked PTD needs a validated CompactCheckpoint")
        for name in ("blocks", "token_refiner", "get_submodule", "named_modules", "final_layer"):
            if not hasattr(network, name):
                raise TypeError(f"Stacked PTD requires network API {name}")
        if not hasattr(network.token_refiner, "blocks"):
            raise TypeError("Stacked PTD requires token_refiner.blocks")
        if (len(network.blocks), len(network.token_refiner.blocks)) != (checkpoint.blocks, checkpoint.refiners):
            raise ValueError("Stacked PTD expert topology differs from actual network blocks/refiners")
        self.recipe = f"taobao_{checkpoint.hops}hop_{checkpoint.steps}fine_tangent0999"
        self.grid = tuple(compact_grid(checkpoint.hops, checkpoint.steps).tolist())
        self.network, self.hops, self.steps = network, checkpoint.hops, checkpoint.steps
        self.lora_strength = _finite_strength(lora_strength, "LoRA strength")
        self.head_strength = _finite_strength(head_strength, "Head strength")
        self.factor_scale = checkpoint.alpha / checkpoint.rank * self.lora_strength
        if not math.isfinite(self.factor_scale):
            raise ValueError("Stacked PTD alpha/rank/strength overflow")
        bank = defaultdict(lambda: defaultdict(list))
        for pair in checkpoint.pairs:
            module = network.get_submodule(pair.target)
            if not callable(getattr(module, "register_forward_hook", None)):
                raise TypeError(f"Stacked PTD needs output-hook API: {pair.target}")
            for hop in range(self.hops):
                a, b = pair.a[hop], pair.b[hop]
                if (a.shape[1] != getattr(module, "in_features", None) or
                        b.shape[0] * (3 if pair.qkv_index is not None else 1) != getattr(module, "out_features", None)):
                    raise ValueError(f"Stacked PTD actual target dimensions disagree: {pair.target}, hop {hop}")
                if pair.swap_halves:
                    up, gate = b.chunk(2, dim=0)
                    b = torch.cat((gate, up), dim=0)
                bank[pair.target][hop].append(Expert(a.detach().cpu().clone(), b.detach().cpu().clone(), pair.qkv_index))
        self.bank = {path: {hop: tuple(experts) for hop, experts in by_hop.items()}
                     for path, by_hop in bank.items()}
        v_plan, a_plan = compact_plans(self.hops, self.steps)
        self.heads = {}
        for path, (weight, bias) in checkpoint.heads.items():
            module = network.get_submodule(path)
            if not callable(getattr(module, "register_forward_hook", None)):
                raise TypeError(f"Stacked PTD absolute head needs output-hook API: {path}")
            if (getattr(module, "in_features", None), getattr(module, "out_features", None)) != (weight.shape[2], weight.shape[1]):
                raise ValueError(f"Stacked PTD absolute head dimensions disagree: {path}")
            plan = v_plan if path.endswith("video_out") else a_plan
            fused_w = torch.einsum("hn,noi->hoi", plan, weight.float())
            fused_b = None if bias is None else torch.einsum("hn,no->ho", plan, bias.float())
            if not bool(torch.isfinite(fused_w).all()) or (fused_b is not None and not bool(torch.isfinite(fused_b).all())):
                raise ValueError(f"Stacked PTD absolute head fusion overflow: {path}")
            self.heads[path] = (fused_w.contiguous(), None if fused_b is None else fused_b.contiguous())
        self.chord_corrections = audio_chord_corrections(self.grid)
        self.warnings = list(checkpoint.warnings)
        self.file_report = deepcopy(file_report)
        self.storage = checkpoint.storage
        self.selection = ContextVar(f"t8_compact_hop_{id(self)}", default=None)
        self.request = ContextVar(f"t8_compact_request_{id(self)}", default=None)
        self.lock, self.last, self.requests = threading.RLock(), None, 0
        self._bank_seal = self._bank_digest()

    def _bank_digest(self):
        digest = hashlib.sha256(super()._bank_digest().encode("ascii"))
        digest.update(repr((self.factor_scale, self.head_strength, self.chord_corrections)).encode("ascii"))
        for path, values in sorted(self.heads.items()):
            digest.update(path.encode("utf8"))
            for tensor in values:
                if tensor is None:
                    digest.update(b"none")
                    continue
                if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.float32:
                    raise ValueError("Stacked PTD private absolute head bank changed")
                digest.update(repr(tuple(tensor.shape)).encode("ascii"))
                payload = memoryview(tensor.detach().contiguous().view(torch.uint8).numpy()).cast("B")
                for start in range(0, len(payload), 8 * 1024 * 1024):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    digest.update(payload[start:start + 8 * 1024 * 1024])
        return digest.hexdigest()

    def _hook(self, frame, path, hop, experts):
        def addition(module, args, kwargs, output):
            if _FRAME.get() is not frame:
                return output
            comfy.model_management.throw_exception_if_processing_interrupted()
            value = args[0] if args else kwargs.get("input")
            if not isinstance(value, torch.Tensor) or not isinstance(output, torch.Tensor):
                raise TypeError(f"Stacked PTD projection needs tensor input/output: {path}")
            result = output.clone() if any(e.qkv_index is not None for e in experts) else output
            for expert in experts:
                a, b = (factor.to(device=value.device, dtype=output.dtype) for factor in (expert.a, expert.b))
                delta = F.linear(F.linear(value.to(a.dtype), a), b).to(output.dtype) * self.factor_scale
                if expert.qkv_index is None:
                    result = result + delta
                else:
                    start = expert.qkv_index * delta.shape[-1]
                    result[..., start:start + delta.shape[-1]] += delta
            state = self.request.get()
            if state is not None:
                key = f"{hop}:{path}"
                state["expert_calls"][key] = state["expert_calls"].get(key, 0) + 1
            return result
        return addition

    @contextmanager
    def forward_scope(self):
        # Own the shared-network lease BEFORE examining/installing head hooks.
        # The parent scope takes the same reentrant lease for the expert hooks.
        with shared_network_lease(self.network) as acquired:
            if not acquired:
                self.note("shared_network_lease_unavailable_original_forward_retained")
                yield
                return
            with self._head_scope():
                yield

    def capture_head_patch_context(self, model):
        """Keep the active guider's head patches, including patches added later.

        The checkpoint heads are absolute, so adding them after the original
        projection would erase a selected head LoRA. Until a delta composition
        protocol exists, retain that projection and report partial consumption.
        Inspect the actual request's patcher, never the setup clone's old dict.
        """
        patcher = getattr(getattr(model, "inner_model", None), "model_patcher", None)
        mappings = []
        known = patcher is not None
        for name in ("patches", "weight_wrapper_patches", "object_patches"):
            value = getattr(patcher, name, None)
            if not isinstance(value, Mapping):
                known = False
            else:
                mappings.append(value)
        hooks = getattr(patcher, "hook_patches", None)
        if not isinstance(hooks, Mapping):
            known = False
        else:
            for value in hooks.values():
                if isinstance(value, Mapping):
                    mappings.append(value)
                else:
                    known = False
        retained = []
        for path in self.heads:
            target = "diffusion_model." + path
            if not known or any(
                isinstance(key, str) and (
                    key == target or key.startswith(target + ".") or target.startswith(key + ".")
                )
                for mapping in mappings for key in mapping
            ):
                retained.append(path)
                self.note(f"selected_head_patch_{path}_original_retained" if known else
                          f"unknown_head_patch_context_{path}_original_retained")
        self.request.get()["retained_head_paths"] = tuple(retained)

    @contextmanager
    def _head_scope(self):
        hop, _ = self.current()
        handles, frame = [], []
        try:
            for path, (weights, biases) in self.heads.items():
                module = self.network.get_submodule(path)
                # A foreign head output hook must not be overwritten by our
                # absolute replacement. Keep the user's actual computation and
                # report partial consumption, rather than rejecting the model.
                state = self.request.get()
                if state is not None and path in state.get("retained_head_paths", ()):
                    continue
                if (module._forward_hooks or module._forward_pre_hooks or "forward" in vars(module)
                        or not _row_local(module.forward)
                        or torch.nn.modules.module._global_forward_hooks
                        or torch.nn.modules.module._global_forward_pre_hooks):
                    self.note(f"unknown_head_{path}_original_retained")
                    continue
                factor = self.chord_corrections[hop] if path.endswith("audio_out") else 1.

                def head_hook(module, args, kwargs, output, *, path=path, weights=weights, biases=biases, factor=factor):
                    if not frame or _FRAME.get() is not frame[0]:
                        return output
                    value = args[0] if args else kwargs.get("input")
                    if not isinstance(value, torch.Tensor) or not isinstance(output, torch.Tensor):
                        raise TypeError(f"Stacked PTD absolute head needs tensor input/output: {path}")
                    selected = F.linear(value.to(output.dtype), weights[hop].to(value.device, output.dtype),
                        None if biases is None else biases[hop].to(value.device, output.dtype)) * factor
                    result = selected if self.head_strength == 1. else output + (selected - output) * self.head_strength
                    state = self.request.get()
                    if state is not None:
                        calls = state.setdefault("head_calls", {})
                        key = f"{hop}:{path}"
                        calls[key] = calls.get(key, 0) + 1
                    return result
                # Register BEFORE the existing scope's physical-audio derivative.
                handles.append(module.register_forward_hook(head_hook, with_kwargs=True))
            previous_frame = _FRAME.get()
            with super().forward_scope():
                current_frame = _FRAME.get()
                # A contended shared-network lease delegates without creating
                # an owned frame. Never arm absolute heads in that branch.
                if current_frame is not previous_frame:
                    frame.append(current_frame)
                yield
        finally:
            for handle in reversed(handles):
                handle.remove()

    def end_request(self, completed):
        state = self.request.get()
        if completed and state is not None:
            for hop in range(self.hops):
                for path in self.heads:
                    if f"{hop}:{path}" not in state.get("head_calls", {}):
                        self.note(f"absolute_head_not_observed:{hop}:{path}")
        super().end_request(completed)

    def report(self):
        report = super().report()
        report.update(format=self.format, storage=self.storage,
            clock_contract="T_MAX0999_physical_audio_chord_over_tangent_not_native10_carry",
            fine_steps=self.steps, trained_hops=self.hops, lora_strength=self.lora_strength,
            alpha_over_rank_and_strength=self.factor_scale, head_strength=self.head_strength,
            weight_scale=self.factor_scale, audio_chord_corrections=list(self.chord_corrections),
            runtime_int8=False, full_model_or_asset_qualification=False)
        report["continuous_stage_protocol_implemented"] = False
        return report


def compact_sigmas(controller):
    return torch.tensor(controller.grid, dtype=torch.float64)


@torch.no_grad()
def sample_compact(model, x, sigmas, extra_args=None, callback=None, disable=None, *, controller):
    expected = compact_sigmas(controller)
    if (not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or
            sigmas.dtype not in (torch.float32, torch.float64) or sigmas.shape != expected.shape or
            not torch.equal(sigmas.detach().cpu(), expected.to(sigmas.dtype))):
        raise ValueError("Stacked PTD requires its complete checkpoint-derived hop grid")
    extra = {} if extra_args is None else dict(extra_args)
    if extra.get("denoise_mask") is not None:
        raise ValueError("Stacked PTD pure-noise recurrence has no partial-mask initial-state contract")
    original = getattr(model, "latent_image", None)
    if original is not None and any(bool(torch.count_nonzero(part)) for part in streams(original)):
        raise ValueError("Stacked PTD requires empty generated latent; references belong in CONDITIONING")
    noise = getattr(model, "noise", None)
    state = initial_state(noise) if noise is not None else map_streams(x, lambda value: value.float())
    pair_streams(state, x, lambda a, b: a)
    parts = streams(state)
    first = parts[0]
    if first.ndim < 1 or first.shape[0] < 1 or any(part.shape[0] != first.shape[0] or part.device != first.device for part in parts):
        raise ValueError("Stacked PTD streams require the same nonempty batch/device")
    grid = expected.to(first.device)
    controller.begin_request()
    try:
        controller.capture_head_patch_context(model)
        for hop, time in enumerate(controller.grid[:-1]):
            comfy.model_management.throw_exception_if_processing_interrupted()
            with controller.activate(hop, time):
                predicted = model(state, grid[hop].expand(first.shape[0]), **extra)
            velocity = pair_streams(state, predicted, lambda a, b: (a.float() - b.float()) / time)
            if callback is not None:
                callback({"x": state, "i": hop, "sigma": grid[hop], "sigma_hat": grid[hop], "denoised": predicted})
            dt = (grid[hop + 1] - grid[hop]).float()
            state = pair_streams(state, velocity, lambda a, b: a.float() + dt * b.float())
            if any(not bool(torch.isfinite(part).all()) for part in streams(state)):
                raise ValueError(f"Non-finite stacked PTD state after hop {hop}")
            controller.completed_hop(hop)
    except BaseException:
        controller.end_request(False)
        raise
    controller.end_request(True)
    return state


def make_compact_sampler(controller):
    from comfy.samplers import KSAMPLER
    return KSAMPLER(sample_compact, extra_options={"controller": controller})


def configure_compact(model, checkpoint, *, file_report=None, lora_strength=1., head_strength=1.):
    if model.model_options.get(KEY) is not None:
        raise ValueError("This branch already has one PTD hop clock; select an unconfigured branch to replace it")
    network = model.get_model_object("diffusion_model")
    controller = CompactPTDController(network, checkpoint, file_report=file_report,
        lora_strength=lora_strength, head_strength=head_strength)
    selected = model.clone()
    sampling = deepcopy(model.get_model_object("model_sampling"))
    if not hasattr(sampling, "audio_shift") or not hasattr(sampling, "shift"):
        raise TypeError("Stacked PTD requires the AV flow sampler shift/audio_shift API")
    sampling.audio_shift = sampling.shift
    selected.add_object_patch("model_sampling", sampling)
    if not hasattr(network, "condition_proj") or not callable(getattr(network, "preprocess_text_embeds", None)):
        raise TypeError("Stacked PTD requires condition_proj and preprocess_text_embeds APIs")
    width = network.condition_proj.in_features

    def raw_text(value):
        if value.shape[-1] != width:
            raise ValueError("Stacked PTD needs raw text; already refined embeddings cannot be reversed")
        return value

    # The shared network may still carry another loaded clone's raw-text patch.
    # Use the selected patcher's effective object (including user overrides).
    preprocess = model.get_model_object("diffusion_model.preprocess_text_embeds")
    if _source_exact(getattr(preprocess, "__func__", preprocess), native, "MiniMaxH3Model.preprocess_text_embeds"):
        selected.add_object_patch("diffusion_model.preprocess_text_embeds", raw_text)
    else:
        controller.warnings.append("Unknown text preprocess retained; raw-refiner coverage may be partial")
    previous = selected.model_options.get("model_function_wrapper")
    selected.set_model_unet_function_wrapper(PTDModelWrapper(controller, previous))
    selected.model_options[KEY] = controller
    if controller.warnings:
        warn_patch_stack("Stacked PTD retains selected user model/patches; inspect partial-consumption warnings")
    return selected, controller.report()


def load_compact_setup(model, path):
    checkpoint, file_report = load_compact_file(path)
    selected, report = configure_compact(model, checkpoint, file_report=file_report)
    controller = selected.model_options[KEY]
    return selected, make_compact_sampler(controller), compact_sigmas(controller), report
