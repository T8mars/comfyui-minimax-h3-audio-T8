"""Explicit hop-expert MODEL selection; scoped state, never a global patch.

PTD contract: EPFL VITA PTD-inference 43ae22c3 (Apache-2.0); independent T8
integration retains user wrappers, module calls, ordered LoRA and real errors.
"""

from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import threading

import torch
import torch.nn.functional as F
from comfy.ldm.minimax import model as native
import comfy.model_management
import comfy.ops
import comfy.quant_ops

from .h3_projection_chunks import (
    _module_call_native,
    _source_exact,
    shared_network_lease,
)
from .patch_stack_policy import warn_patch_stack
from .ptd_checkpoint import (
    BUFFERS,
    load_ptd_file,
    recipe_grid,
    validate_metadata,
    validate_tensors,
)
from .ptd_sampling import (
    audio_sigma,
    audio_sigma_prime,
    make_sampler,
    native_audio_shift,
    ptd_sigmas,
)

KEY = "t8_h3_ptd_v1"
_FRAME = ContextVar("t8_ptd_forward_frame", default=None)


def _original(module, qualname):
    function = getattr(module.forward, "__func__", module.forward)
    if not _module_call_native(module) or not _source_exact(function, native, qualname):
        return False
    # These substitutions deliberately change the native rounding order. They
    # may not erase an installed helper's behavior merely because the outer
    # module forward still belongs to Core.
    if qualname == "DiTBlock.forward":
        return all(
            _source_exact(getattr(native, name, None), native, name)
            for name in ("_mod_row", "_mod_scale_shift", "_mod_gate")
        )
    if qualname == "MLP.forward":
        return all(
            _source_exact(getattr(comfy.ops, name, None), comfy.ops, name)
            for name in ("linear_input_act", "_eager_input_act", "_swiglu_eager")
        ) and (
            type(comfy.ops.INPUT_ACT_EAGER) is dict
            and comfy.ops.INPUT_ACT_EAGER.get("swiglu") is comfy.ops._swiglu_eager
            and _source_exact(F.silu, F, "silu")
        )
    if qualname == "Attention.forward":
        kitchen = getattr(comfy.quant_ops, "ck", None)
        name = (
            "rms_rope_split_half"
            if comfy.model_management.in_training
            else "rms_rope_split_half_"
        )
        # The explicit split-half math must not erase a user-owned native RoPE
        # call. This authenticates the selected public callable, not arbitrary
        # foreign kernels or a trained-quality/portable qualification.
        return kitchen is not None and _source_exact(
            getattr(kitchen, name, None), kitchen, name
        )
    return True


def _block_math(
    block, x, t_emb, mod_segments, rope_freqs, transformer_options=None, attention=None
):
    values = block.adaln_proj(t_emb)
    sa, ca, ga, sm, cm, gm = values

    def mod(value, shift, scale):
        result = torch.empty_like(value)
        for start, stop, row in mod_segments:
            result[start:stop] = value[start:stop] * (
                1 + scale[row].to(value.dtype)
            ) + shift[row].to(value.dtype)
        return result

    def residual(value, update, gate):
        result = torch.empty_like(value)
        for start, stop, row in mod_segments:
            product = update[start:stop] * gate[row].to(value.dtype)
            result[start:stop] = value[start:stop] + product
        return result

    options = {} if transformer_options is None else transformer_options
    x = residual(
        x,
        (block.attn if attention is None else attention)(
            mod(block.norm1(x), sa, ca),
            rope_freqs=rope_freqs,
            transformer_options=options,
        ),
        ga,
    )
    return residual(x, block.mlp(mod(block.norm2(x), sm, cm)), gm)


def _attention_math(module, x, rope_freqs=None, transformer_options=None):
    from comfy.ldm.modules.attention import AttentionTensorContainer

    q, k, v = [
        part.reshape(x.shape[0], module.heads, module.head_dim)
        for part in module.qkv_proj(x).chunk(3, dim=-1)
    ]
    q, k = module.q_norm(q), module.k_norm(k)
    if rope_freqs is not None:
        cosine = rope_freqs[0, :, 0, :, 0, 0].repeat(1, 2).unsqueeze(1).to(q.dtype)
        sine = rope_freqs[0, :, 0, :, 1, 0].repeat(1, 2).unsqueeze(1).to(q.dtype)
        width = cosine.shape[-1]

        def rotate(value):
            leading = value[..., :width]
            a, b = leading.chunk(2, dim=-1)
            return torch.cat(
                (
                    leading * cosine + torch.cat((-b, a), dim=-1) * sine,
                    value[..., width:],
                ),
                dim=-1,
            )

        q, k = rotate(q), rotate(k)
    values = [
        AttentionTensorContainer(value.transpose(0, 1).unsqueeze(0))
        for value in (q, k, v)
    ]
    result = native.optimized_attention(
        *values,
        module.heads,
        preferred_attention=module.comfy_attention,
        mask=None,
        skip_reshape=True,
        transformer_options={} if transformer_options is None else transformer_options,
    )
    return module.out_proj(result.squeeze(0))


def _mlp_math(module, x):
    # An explicit module call is essential: Core's fused SwiGLU can bypass fc2
    # hooks, including the hop expert. Do not mutate shared comfy_force_forward.
    gate, up = module.fc1(x).chunk(2, dim=-1)
    return module.fc2(F.silu(gate) * up)


@dataclass(frozen=True)
class Expert:
    a: torch.Tensor
    b: torch.Tensor
    qkv_index: int | None


class PTDController:
    def __init__(self, network, tensors, metadata, *, file_report=None):
        self.recipe, warnings = validate_metadata(metadata)
        self.grid = recipe_grid(self.recipe)
        self.network = network
        for name in ("blocks", "token_refiner", "get_submodule", "final_layer"):
            if not hasattr(network, name):
                raise TypeError(
                    f"PTD requires the selected network API {name}, not a particular model name"
                )
        if not hasattr(network.token_refiner, "blocks"):
            raise TypeError("PTD requires token_refiner.blocks to map hop experts")
        pairs, extra = validate_tensors(
            tensors,
            self.recipe,
            blocks=len(network.blocks),
            refiners=len(network.token_refiner.blocks),
        )
        warnings.extend(extra)
        bank = defaultdict(lambda: defaultdict(list))
        for pair in pairs:
            module = network.get_submodule(pair.module)
            if not callable(getattr(module, "register_forward_hook", None)):
                raise TypeError(
                    f"PTD target needs module output-hook API: {pair.module}"
                )
            a, b = tensors[pair.a_key], tensors[pair.b_key]
            if a.shape[1] != getattr(module, "in_features", None) or b.shape[0] * (
                3 if pair.qkv_index is not None else 1
            ) != getattr(module, "out_features", None):
                raise ValueError(
                    f"PTD factor dimensions do not fit selected target {pair.module}: A={tuple(a.shape)}, B={tuple(b.shape)}"
                )
            if pair.swap_halves:
                up, gate = b.chunk(2, dim=0)
                b = torch.cat((gate, up), dim=0)
            bank[pair.module][pair.hop].append(
                Expert(
                    a.detach().cpu().clone(), b.detach().cpu().clone(), pair.qkv_index
                )
            )
        for key, target in BUFFERS.items():
            if key not in tensors:
                continue
            parent, _, attr = target.rpartition(".")
            actual = getattr(network.get_submodule(parent), attr)
            expected = tensors[key]
            if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape:
                raise ValueError(
                    f"PTD reference-buffer dimensions do not match {target}"
                )
            if actual.device.type == "meta" or not torch.equal(
                actual.detach().cpu(), expected.to(device="cpu", dtype=actual.dtype)
            ):
                warnings.append(
                    f"PTD selected base differs from checkpoint reference {target}; original selected base retained"
                )
        if getattr(network, "use_adaln_curves", False):
            warnings.append(
                "PTD on AdaLN curves is unverified; selected implementation retained"
            )
        self.bank = {
            key: {hop: tuple(values) for hop, values in hops.items()}
            for key, hops in bank.items()
        }
        self._bank_seal = self._bank_digest()
        self.warnings = sorted(set(warnings))
        self.file_report = deepcopy(file_report)
        self.selection = ContextVar(f"t8_ptd_hop_{id(self)}", default=None)
        self.request = ContextVar(f"t8_ptd_request_{id(self)}", default=None)
        self.lock = threading.RLock()
        self.last = None
        self.requests = 0

    def _bank_digest(self):
        """Seal every selected hop, including ones no longer used by the loop.

        Factor versions alone miss .data writes, while permanent version checks
        would reject a restored, byte-identical bank on a retry. Hash the actual
        private CPU contents at request boundaries, without a weight-size cap or
        a full-size bytes allocation. A noncontiguous factor may still require
        a single-factor contiguous tensor allocation; neither scan is free.
        """
        digest = hashlib.sha256()

        def label(value):
            encoded = json.dumps(
                value, separators=(",", ":"), ensure_ascii=True
            ).encode()
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)

        label((self.recipe, self.grid))
        if type(self.bank) is not dict or any(
            type(path) is not str for path in self.bank
        ):
            raise ValueError("PTD selected expert bank structure changed")
        for path, hops in sorted(self.bank.items()):
            if type(hops) is not dict or any(type(hop) is not int for hop in hops):
                raise ValueError("PTD selected expert hop structure changed")
            label(path)
            for hop, experts in sorted(hops.items()):
                if type(experts) is not tuple:
                    raise ValueError("PTD selected expert factor structure changed")
                label((hop, len(experts)))
                for expert in experts:
                    if type(expert) is not Expert:
                        raise ValueError("PTD selected expert descriptor changed")
                    label(expert.qkv_index)
                    for factor in (expert.a, expert.b):
                        if (
                            not isinstance(factor, torch.Tensor)
                            or factor.device.type != "cpu"
                            or factor.layout != torch.strided
                            or not factor.is_floating_point()
                        ):
                            raise ValueError("PTD selected expert tensor changed")
                        label((str(factor.dtype), tuple(factor.shape)))
                        payload = memoryview(
                            factor.detach().contiguous().view(torch.uint8).numpy()
                        ).cast("B")
                        for start in range(0, len(payload), 8 * 1024 * 1024):
                            comfy.model_management.throw_exception_if_processing_interrupted()
                            digest.update(payload[start : start + 8 * 1024 * 1024])
        return digest.hexdigest()

    def _verify_bank(self):
        if self._bank_digest() != self._bank_seal:
            raise ValueError("PTD selected expert contents changed during sampling")

    def begin_request(self):
        previous = self.request.get()
        state = {
            "completed": False,
            "hops": [],
            "forwards": [],
            "expert_calls": {},
            "delegated": set(),
            "previous": previous,
        }
        with self.lock:
            self.requests += 1
            state["request_number"] = self.requests
        self.request.set(state)
        try:
            self._verify_bank()
        except BaseException:
            self.note("selected_expert_content_seal_failed")
            self.end_request(False)
            raise

    def completed_hop(self, hop):
        self.request.get()["hops"].append(hop)

    def end_request(self, completed):
        state = self.request.get()
        try:
            if completed:
                self._verify_bank()
                for hop in range(len(self.grid) - 1):
                    for path in self.bank:
                        if f"{hop}:{path}" not in state["expert_calls"]:
                            state["delegated"].add(
                                f"selected_expert_not_observed:{hop}:{path}"
                            )
            state["completed"] = bool(completed)
        except BaseException:
            state["completed"] = False
            state["delegated"].add("selected_expert_content_seal_failed")
            raise
        finally:
            self.request.set(state.pop("previous"))
            with self.lock:
                self.last = deepcopy(state)

    def note(self, message):
        state = self.request.get()
        if state is not None:
            state["delegated"].add(message)

    @contextmanager
    def activate(self, hop, time):
        if type(hop) is not int or not 0 <= hop < len(self.grid) - 1:
            raise ValueError("PTD hop index is outside the selected expert bank")
        if (
            not isinstance(time, (int, float))
            or not math.isfinite(float(time))
            or abs(float(time) - self.grid[hop]) > 2e-6
        ):
            raise ValueError("PTD expert hop must use its exact trained grid time")
        token = self.selection.set((hop, self.grid[hop]))
        try:
            yield self
        finally:
            self.selection.reset(token)

    def current(self):
        value = self.selection.get()
        if value is None:
            raise ValueError(
                "PTD MODEL has no active hop; connect its dedicated PTD sampler and complete sigmas"
            )
        return value

    def _hook(self, frame, path, hop, experts):
        def addition(module, args, kwargs, output):
            if _FRAME.get() is not frame:
                return output
            comfy.model_management.throw_exception_if_processing_interrupted()
            value = args[0] if args else kwargs.get("input")
            if not isinstance(value, torch.Tensor) or not isinstance(
                output, torch.Tensor
            ):
                raise TypeError(f"PTD projection needs tensor input/output: {path}")
            result = (
                output.clone()
                if any(expert.qkv_index is not None for expert in experts)
                else output
            )
            for expert in experts:
                a, b = (
                    factor.to(device=value.device, dtype=output.dtype)
                    for factor in (expert.a, expert.b)
                )
                delta = F.linear(F.linear(value.to(a.dtype), a), b).to(output.dtype)
                if expert.qkv_index is None:
                    result = result + delta
                else:
                    start = expert.qkv_index * delta.shape[-1]
                    result[..., start : start + delta.shape[-1]] += delta
            state = self.request.get()
            if state is not None:
                key = f"{hop}:{path}"
                state["expert_calls"][key] = state["expert_calls"].get(key, 0) + 1
            return result

        return addition

    @contextmanager
    def forward_scope(self):
        hop, time = self.current()
        with shared_network_lease(self.network) as acquired:
            if not acquired:
                self.note("shared_network_lease_unavailable_original_forward_retained")
                yield
                return
            frame, handles, methods = object(), [], []
            token = _FRAME.set(frame)

            def replace(module, function, qualname):
                if (
                    torch.nn.modules.module._global_forward_pre_hooks
                    or torch.nn.modules.module._global_forward_hooks
                    or not _original(module, qualname)
                ):
                    self.note(f"unknown_{qualname}_original_forward_retained")
                    return
                had, old = "forward" in vars(module), module.forward

                def dispatch(*args, **kwargs):
                    return (
                        function(module, *args, **kwargs)
                        if _FRAME.get() is frame
                        else old(*args, **kwargs)
                    )

                module.forward = dispatch
                methods.append((module, had, old, dispatch))

            try:
                for block in self.network.blocks:
                    replace(block, _block_math, "DiTBlock.forward")
                    replace(block.attn, _attention_math, "Attention.forward")
                for path, module in self.network.named_modules():
                    if path.endswith(".mlp"):
                        replace(module, _mlp_math, "MLP.forward")
                embed = getattr(self.network, "time_embedder", None)
                remembered = {}
                if (
                    embed is not None
                    and _original(embed, "TimeEmbedder.forward")
                    and not embed._forward_pre_hooks
                    # Global pre-hooks run before local hooks. Replacing their
                    # already-transformed input would erase the user's actual
                    # computation; retain the full original precision path.
                    and not torch.nn.modules.module._global_forward_pre_hooks
                    and not torch.nn.modules.module._global_forward_hooks
                ):

                    def exact_time(module, args):
                        if _FRAME.get() is not frame:
                            return args
                        values = args[0]
                        candidates = torch.tensor(
                            [
                                1 - time,
                                1 - audio_sigma(time),
                                max(1 - time, 0.999),
                                1.0,
                            ],
                            device=values.device,
                            dtype=torch.float32,
                        ).unique()
                        distances, indices = (
                            (values[:, None] - candidates[None]).abs().min(dim=1)
                        )
                        if not bool((distances <= 2e-6).all()):
                            raise ValueError(
                                "PTD native time labels disagree with the active AV clocks"
                            )
                        return (candidates[indices],) + args[1:]

                    def remember(module, args, output):
                        if _FRAME.get() is frame:
                            remembered["value"] = output

                    handles.extend(
                        (
                            embed.register_forward_pre_hook(exact_time),
                            embed.register_forward_hook(remember),
                        )
                    )
                    for layer in (*self.network.blocks, self.network.final_layer):
                        adaln = getattr(layer, "adaln_proj", None)
                        linear = getattr(adaln, "linear", None)
                        if adaln is None or not getattr(adaln, "apply_silu", True):
                            continue
                        if (
                            not _original(adaln, "AdalnProj.forward")
                            or adaln._forward_pre_hooks
                            or linear._forward_pre_hooks
                        ):
                            self.note("unknown_adaln_input_order_original_retained")
                            continue

                        def precision(module, args):
                            if _FRAME.get() is not frame:
                                return args
                            if "value" not in remembered:
                                raise ValueError(
                                    "PTD AdaLN executed before time embedding"
                                )
                            return (
                                F.silu(remembered["value"]).to(args[0].dtype),
                            ) + args[1:]

                        handles.append(linear.register_forward_pre_hook(precision))
                else:
                    self.note("unknown_time_embedding_precision_original_retained")
                for path, by_hop in self.bank.items():
                    handles.append(
                        self.network.get_submodule(path).register_forward_hook(
                            self._hook(frame, path, hop, by_hop[hop]), with_kwargs=True
                        )
                    )

                def audio_clock(module, args, output):
                    return (
                        output
                        * torch.as_tensor(
                            audio_sigma_prime(time),
                            device=output.device,
                            dtype=output.dtype,
                        )
                        if _FRAME.get() is frame
                        else output
                    )

                handles.append(
                    self.network.final_layer.audio_out.register_forward_hook(
                        audio_clock
                    )
                )
                state = self.request.get()
                if state is not None:
                    state["forwards"].append(
                        {
                            "hop": hop,
                            "video_sigma": time,
                            "audio_sigma": audio_sigma(time),
                            "audio_sigma_prime": audio_sigma_prime(time),
                            "native_audio_shift": native_audio_shift(time),
                        }
                    )
                yield
                if state is not None:
                    missing = [
                        path
                        for path in self.bank
                        if f"{hop}:{path}" not in state["expert_calls"]
                    ]
                    for path in missing:
                        self.note(f"selected_expert_not_observed:{hop}:{path}")
            finally:
                for handle in reversed(handles):
                    handle.remove()
                for module, had, old, dispatch in reversed(methods):
                    if vars(module).get("forward") is dispatch:
                        if had:
                            module.forward = old
                        else:
                            del module.forward
                    else:
                        self.note("later_user_forward_owner_preserved")
                _FRAME.reset(token)

    def report(self):
        with self.lock:
            value = deepcopy(self.last)
        if value is not None:
            value["delegated"] = sorted(value["delegated"])
        partial = bool(self.warnings) or bool(value and value["delegated"])
        return {
            "schema": "t8.h3.ptd.v1",
            "recipe": self.recipe,
            "grid": list(self.grid),
            "status": "selected_not_executed"
            if value is None
            else "observed_partial"
            if value["completed"] and partial
            else "observed_complete"
            if value["completed"]
            else "failed",
            "last_request": value,
            "warnings": list(self.warnings),
            "file": deepcopy(self.file_report),
            "partial": partial,
            "completion_boundary": "Completed means the fixed-hop integrator returned; expert hooks and wrapper entries are observed separately, not invented NFE or trained quality",
            "expert_modules": len(self.bank),
            "expert_content_sha256": self._bank_seal,
            "weight_scale": 1.0,
            "fp32_state_carry": True,
            "portable_cache_identity": False,
            "trained_GPU_qualified": False,
            "quality_accepted": False,
        }


class PTDModelWrapper:
    def __init__(self, controller, previous):
        self.controller, self.previous = controller, previous

    def __call__(self, model_function, kwargs):
        _, time = self.controller.current()
        sigma = kwargs.get("timestep")
        if (
            not isinstance(sigma, torch.Tensor)
            or not sigma.numel()
            or not bool(torch.isfinite(sigma).all())
            or not bool(
                torch.allclose(sigma, torch.full_like(sigma, time), atol=2e-6, rtol=0)
            )
        ):
            raise ValueError("PTD model sigma differs from active expert time")
        forwarded = dict(kwargs)
        conditions = dict(kwargs.get("c", {}))
        options = dict(conditions.get("transformer_options", {}))
        options.update(
            minimax_h3_sigma_shift_video=1.0,
            minimax_h3_sigma_shift_audio=native_audio_shift(time),
        )
        conditions["transformer_options"] = options
        payload = dict(conditions.get("minimax_payload", {}))
        payload["audio_scale"] = 1.0
        conditions["minimax_payload"] = payload
        forwarded["c"] = conditions
        with self.controller.forward_scope():
            return (
                self.previous(model_function, forwarded)
                if self.previous is not None
                else model_function(
                    forwarded["input"], forwarded["timestep"], **conditions
                )
            )

    def to(self, device):
        if callable(getattr(self.previous, "to", None)):
            self.previous = self.previous.to(device)
        return self

    def cleanup(self, **kwargs):
        if callable(getattr(self.previous, "cleanup", None)):
            self.previous.cleanup(**kwargs)

    def models(self):
        return (
            self.previous.models()
            if callable(getattr(self.previous, "models", None))
            else []
        )


def get_controller(model):
    from .ptd_compact import CompactPTDController

    value = model.model_options.get(KEY)
    if type(value) not in (PTDController, CompactPTDController):
        raise ValueError("Connect the MODEL returned by the PTD expert loader")
    return value


def sampler_and_sigmas(controller):
    """Dispatch only the two implemented PTD integrator protocols."""
    from .ptd_compact import CompactPTDController, compact_sigmas, make_compact_sampler

    if type(controller) is CompactPTDController:
        return make_compact_sampler(controller), compact_sigmas(controller)
    if type(controller) is not PTDController:
        raise ValueError("PTD sampler needs an implemented expert-controller protocol")
    return make_sampler(controller), ptd_sigmas(controller.recipe)


def configure_ptd(model, tensors, metadata, *, file_report=None):
    if model.model_options.get(KEY) is not None:
        raise ValueError(
            "PTD expert bank already selected on this branch; select an unconfigured MODEL branch to replace it"
        )
    network = model.get_model_object("diffusion_model")
    controller = PTDController(network, tensors, metadata, file_report=file_report)
    selected = model.clone()
    sampling = deepcopy(model.get_model_object("model_sampling"))
    if not hasattr(sampling, "audio_shift") or not hasattr(sampling, "shift"):
        raise TypeError(
            "PTD needs model_sampling shift/audio_shift API to carry physical audio"
        )
    sampling.audio_shift = sampling.shift
    selected.add_object_patch("model_sampling", sampling)
    if not hasattr(network, "condition_proj") or not callable(
        getattr(network, "preprocess_text_embeds", None)
    ):
        raise TypeError(
            "PTD needs condition_proj and preprocess_text_embeds to recompute text with each hop expert"
        )
    width = network.condition_proj.in_features

    def raw_text(value):
        if value.shape[-1] != width:
            raise ValueError(
                "PTD requires raw text embeddings for per-hop refiner; rebuild CONDITIONING before preprocessing"
            )
        return value

    # A previously sampled clone can leave its object patch resident on the
    # shared network. Resolve this input branch's patch/backup, not that clone's
    # currently installed callable, when rebuilding a checkpoint selection.
    preprocessor = model.get_model_object("diffusion_model.preprocess_text_embeds")
    if _source_exact(
        getattr(preprocessor, "__func__", preprocessor),
        native,
        "MiniMaxH3Model.preprocess_text_embeds",
    ):
        selected.add_object_patch("diffusion_model.preprocess_text_embeds", raw_text)
    else:
        controller.warnings.append(
            "Unknown text preprocess callable retained; per-hop refiner coverage may be partial"
        )
    previous = selected.model_options.get("model_function_wrapper")
    selected.set_model_unet_function_wrapper(PTDModelWrapper(controller, previous))
    selected.model_options[KEY] = controller
    if controller.warnings:
        warn_patch_stack(
            "PTD selected non-published combination; original user choices retained. Inspect report warnings."
        )
    return selected, controller.report()


def load_ptd_setup(model, path, *, recipe=None):
    from .ptd_compact import load_compact_setup
    from .ptd_compact_checkpoint import is_compact_file

    if is_compact_file(path):
        if recipe is not None:
            raise ValueError(
                "Stacked PTD derives its hop grid from checkpoint metadata; an EPFL recipe override has no mapping to its absolute head bank"
            )
        return load_compact_setup(model, path)
    tensors, metadata, file_report = load_ptd_file(path, recipe=recipe)
    selected, report = configure_ptd(model, tensors, metadata, file_report=file_report)
    controller = get_controller(selected)
    return selected, make_sampler(controller), ptd_sigmas(controller.recipe), report
