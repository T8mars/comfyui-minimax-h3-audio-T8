"""RES-only source-bound KJ/Sol attention routing for external effects.

Keep original KJ projections and original Sol gates. Unbiased rows outside the
Sol gate use the original KJ kernel on a private K copy (that kernel centers K
in-place). Biased rows retain Sol's selected mask-capable delegate. No backend
is described as having consumed a mask it cannot implement.
"""
from collections import Counter
import hashlib
from pathlib import Path

import torch

from .patch_stack_policy import UnverifiedModelStack, warn_patch_stack
from .relay_kj_backend import _codes, _live_code_matches
from .relay_kj_memory import HeadGroupedBackend, _make_memory_forward
from .relay_sol_backend import UserSelectedBackend
from .vdn_attention_compat import _factory_closure

KEY = "t8_RES_KJ_Sol_external_effect_memory_v1"


class RESMemoryDelegate:
    def __init__(self, original, kernel, source_sha256):
        self.original, self.kernel, self.source_sha256 = original, kernel, source_sha256
        self.counters = Counter()

    def report(self):
        return dict(kind="RES_source_bound_KJ_Sol_external_effect_delegate", completed_calls=dict(self.counters),
            source_sha256=self.source_sha256, kernel_policy="original_KJ_unbiased_outside_Sol_gate; original_Sol_selected_delegate_for_bias",
            input_K_centering="private_copy_only", CUDA_numerical_qualified=False)

    def attention(self, q, k, v, heads, *, mask=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        from .prompt_relay_advanced import PROMPT_RELAY_RUNTIME_KEY
        from .enhance_a_video_advanced import EAV_RUNTIME_KEY
        options = kwargs.get("transformer_options") or {}
        route = options.get(PROMPT_RELAY_RUNTIME_KEY, options.get(EAV_RUNTIME_KEY, {}))
        gate = options.get("sol_compose")
        # Literal original composer conditions, evaluated against the complete
        # packed sequence, not the short Relay query chunk.
        take = (gate is not None and q.device.type == "cuda" and q.dtype == torch.bfloat16
                and route.get("seq_len", 0) >= gate["min_tokens"])
        if take and options.get("sigmas") is not None:
            sigma = float(options["sigmas"][0])
            take = not (sigma > gate["sigma_start"] or sigma < gate["sigma_end"])
        if q.device.type == "cuda" and not take and mask is None:
            if (not skip_reshape or q.ndim != 4 or q.shape[1] != heads
                    or k.shape[1] != heads or v.shape[1] != heads or kwargs.get("scale") is not None
                    or kwargs.get("enable_gqa", False)):
                raise RuntimeError("RES KJ delegate requires its original full-head packed HND contract")
            # Preserve the K used by later Relay chunks and FETA statistics.
            output = self.kernel([q.transpose(1, 2), k.transpose(1, 2).clone(), v.transpose(1, 2)], v.dtype)
            self.counters["original_KJ:unbiased"] += 1
            return output.transpose(1, 2) if skip_output_reshape else output.reshape(q.shape[0], q.shape[2], -1)
        output = self.original.attention(q, k, v, heads, mask=mask, skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape, **kwargs)
        self.counters["selected_delegate:biased" if mask is not None else "selected_delegate:unbiased"] += 1
        return output


def _source_methods():
    source = Path(__file__).read_bytes()
    import sys
    module = sys.modules[__name__]
    codes = tuple(_codes(compile(source, __file__, "exec", dont_inherit=True)))
    for name in ("__init__", "attention", "report"):
        if not _live_code_matches(vars(RESMemoryDelegate)[name], codes, module):
            raise UnverifiedModelStack("RES memory effect delegate executable changed")
    return hashlib.sha256(source).hexdigest()


def inspect_owner(model, backend=None):
    owner = model.get_attachment(KEY)
    if type(owner) is not dict or set(owner) != {"backend", "original_composers", "source_sha256"}:
        raise UnverifiedModelStack("RES KJ/Sol effect memory has no exact source owner")
    selected = owner["backend"]
    if backend is not None and selected is not backend:
        raise UnverifiedModelStack("RES memory selector and projection belong to different owners")
    if (type(selected) is not HeadGroupedBackend or type(selected.delegate) is not RESMemoryDelegate
            or set(vars(selected)) != {"delegate", "head_chunks", "runtime_token", "expected_methods", "contract"}
            or selected.expected_methods != {} or type(selected.runtime_token) is not object
            or set(vars(selected.delegate)) != {"original", "kernel", "source_sha256", "counters"}
            or type(selected.delegate.counters) is not Counter
            or type(selected.delegate.original) is not UserSelectedBackend
            or type(selected.head_chunks) is not int or not 1 <= selected.head_chunks <= 56
            or owner["source_sha256"] != _source_methods()):
        raise UnverifiedModelStack("RES memory effect runtime delegate changed")
    original = owner["original_composers"]
    blocks = model.model.diffusion_model.blocks
    expected = {f"diffusion_model.blocks.{index}.attn.forward" for index in range(len(blocks))}
    if type(original) is not dict or set(original) != expected:
        raise UnverifiedModelStack("RES memory effects require the complete actual attention set")
    from . import sol_attn_minimax_v2 as sol
    from .res_sol_identity import _composer
    raw = {}
    for index, block in enumerate(blocks):
        path = f"diffusion_model.blocks.{index}.attn.forward"
        delegate = _composer(model.object_patches.get(path), block.attn, sol)
        state = _factory_closure(delegate.__func__, _make_memory_forward, "forward")
        original_delegate = _composer(original[path], block.attn, sol)
        if (state is None or set(state) != {"original", "backend"}
                or state["backend"] is not selected or state["original"] is not original_delegate
                or original_delegate.__func__.__globals__.get("_sageattn_int8_fp8_nhd") is not selected.delegate.kernel
                or selected.delegate.source_sha256 != hashlib.sha256(Path(original_delegate.__func__.__globals__["__file__"]).read_bytes()).hexdigest()):
            raise UnverifiedModelStack("RES KJ projection was replaced or belongs to a different kernel")
        raw[path] = original_delegate
    return raw, dict(provider_sha256=owner["source_sha256"], head_chunks=selected.head_chunks,
        kernel_source_sha256=selected.delegate.source_sha256, actual_paths=sorted(raw),
        mask_policy="original_Sol_selected_mask_capable_delegate", K_input_mutated=False,
        numerical_or_quality_qualification=False)


def prepare(model):
    """Only compose the authenticated original KJ/Sol applied-Relay branch."""
    from . import prompt_relay_advanced as relay
    if not model.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY):
        return model
    contract = relay.prompt_relay_model_contract(model)
    backend = contract["attention_backend"]
    if type(backend) is not UserSelectedBackend:
        return model
    from .res_relay_identity import project_original_sol_delegate
    from .res_sol_identity import inspect_original_sol
    try:
        view, _ = project_original_sol_delegate(model, backend)
        inspected = inspect_original_sol(view)
        if inspected is None or not inspected[0]["unwrapped_attention_delegates_authenticated"]:
            raise UnverifiedModelStack("RES external effects KJ calculation owner is not source-bound")
    except UnverifiedModelStack as error:
        warn_patch_stack(str(error))
        return model
    original = {path: model.object_patches[path] for path in inspected[1]}
    delegates = inspected[1]
    kernels = {method.__func__.__globals__["_sageattn_int8_fp8_nhd"] for method in delegates.values()}
    if len(kernels) != 1:
        warn_patch_stack("RES external effects keep unmatched KJ calculation delegates; coverage unverified")
        return model
    groups = model.model_options["transformer_options"].get("minimax_head_chunks", 1)
    if type(groups) is not int or not 1 <= groups <= 56:
        raise ValueError("RES KJ head chunks are outside the native supported range")
    source = next(iter(delegates.values())).__func__.__globals__["__file__"]
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    selected = HeadGroupedBackend(RESMemoryDelegate(backend, kernels.pop(), digest), groups,
        dict(kind="RES_KJ_Sol_effects", head_chunks=groups, ffn_settings=None, source_sha256s=[digest]))
    branch = model.clone()
    branch.remove_wrappers_with_key("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY)
    branch.remove_attachments(relay.PROMPT_RELAY_WRAPPER_KEY)
    branch.model_options["transformer_options"]["optimized_attention_override"] = backend.override
    from . import sol_attn_minimax_v2 as sol
    for path, delegate in delegates.items():
        branch.add_object_patch(path, sol._compose_module_patch(delegate.__self__, _make_memory_forward(delegate, selected)))
    branch.set_attachments(KEY, dict(backend=selected, original_composers=original, source_sha256=_source_methods()))
    branch, _ = relay._install_prompt_relay_model(branch, contract["binding"], contract["query_chunk_rows"],
        contract["core_hashes"], attention_backend=selected)
    inspect_owner(branch, selected)
    return branch


def raw_delegate(model, path, delegate):
    """Read-only normalization after exact native Sol composer authentication."""
    if model.get_attachment(KEY) is None:
        return delegate, None
    raw, content = inspect_owner(model)
    state = _factory_closure(delegate.__func__, _make_memory_forward, "forward")
    if state is None or state["original"] is not raw[path]:
        raise UnverifiedModelStack("RES Sol inspector saw a different adapted memory delegate")
    return raw[path], content
