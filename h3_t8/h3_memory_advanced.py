"""Independent MiniMax H3 activation-memory patches.

These patches are deliberately local to a cloned ``MODEL``.  They do not
import KJNodes and they never alter ComfyUI's process-global attention
selection.  The attention patch reduces the lifetime of the normalized block
input and can split a self-attention call over head groups.  The FFN patch
splits the packed token axis before the SwiGLU projection.

Both implementations are Experimental: they preserve the H3 equations but a
different GEMM/attention launch shape can change floating-point rounding.
"""

from __future__ import annotations

from .patch_stack_policy import warn_patch_stack

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import inspect
import json
import logging
from pathlib import Path
from types import MethodType
from typing import Any, Literal

import torch

import comfy.model_management as model_management
import comfy.ops
import comfy.quant_ops
from comfy.ldm.minimax import model as core_h3
from comfy.ldm.modules.attention import AttentionTensorContainer


SCHEMA = "t8.minimax_h3.memory_patches.v1"
ATTACHMENT_KEY = "t8_minimax_h3_memory_patches_v1"
RUNTIME_TOKEN_KEY = "t8_minimax_h3_memory_tokens_v1"
ATTENTION_WRAPPER_KEY = "t8_minimax_h3_low_vram_attention_owner_v1"
FFN_WRAPPER_KEY = "t8_minimax_h3_chunk_ffn_owner_v1"
# Keep ordinary memory-node imports independent of the optional V2 route.
# A marker is only a reason to call its real owner authenticator, never proof.
FAST_H3_V2_ATTACHMENT_KEY = "t8_fasth3_v2_owner_v1"
logger = logging.getLogger(__name__)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


@dataclass(frozen=True)
class _PatchRecord:
    kind: Literal["attention", "ffn"]
    token: object
    paths: tuple[str, ...]
    owners: tuple[Any, ...]
    methods: tuple[Any, ...]
    settings: tuple[tuple[str, int], ...]
    wrapper_key: str
    wrapper: Any | None = None


@dataclass(frozen=True)
class _MemoryReceipt:
    schema: str = SCHEMA
    attention: _PatchRecord | None = None
    ffn: _PatchRecord | None = None

    def on_model_patcher_clone(self):
        # Immutable.  Keeping the exact record/token identities across clones
        # lets the runtime owner guard detect copied descriptive metadata.
        return self


def _receipt(model: Any) -> _MemoryReceipt:
    getter = getattr(model, "get_attachment", None)
    value = getter(ATTACHMENT_KEY) if callable(getter) else getattr(
        model, "attachments", {}
    ).get(ATTACHMENT_KEY)
    if value is None:
        return _MemoryReceipt()
    if not isinstance(value, _MemoryReceipt) or value.schema != SCHEMA:
        raise RuntimeError("MiniMax H3 memory patch receipt is foreign or malformed")
    return value


def _set_receipt(model: Any, receipt: _MemoryReceipt) -> None:
    setter = getattr(model, "set_attachments", None)
    if callable(setter):
        setter(ATTACHMENT_KEY, receipt)
        return
    attachments = getattr(model, "attachments", None)
    if not isinstance(attachments, dict):
        raise RuntimeError("MODEL clone does not expose attachment storage")
    attachments[ATTACHMENT_KEY] = receipt


def _diffusion_model(model: Any):
    getter = getattr(model, "get_model_object", None)
    try:
        diffusion = (
            getter("diffusion_model")
            if callable(getter)
            else model.model.diffusion_model
        )
    except Exception as error:
        raise ValueError(
            f"MODEL does not expose diffusion_model: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(diffusion, core_h3.MiniMaxH3Model):
        raise ValueError(
            "MiniMax H3 memory nodes require the current native ComfyUI MiniMaxH3Model"
        )
    blocks = getattr(diffusion, "blocks", None)
    if blocks is None or len(blocks) < 1:
        raise ValueError("MiniMax H3 diffusion model does not expose any DiT blocks")
    return diffusion, tuple(blocks)


def _core_contract(blocks: tuple[Any, ...]) -> dict[str, Any]:
    required = {
        "block": {
            "self",
            "x",
            "t_emb",
            "mod_segments",
            "rope_freqs",
            "transformer_options",
        },
        "attention": {"self", "x", "rope_freqs", "transformer_options"},
        "mlp": {"self", "x"},
    }
    functions = {
        "block": core_h3.DiTBlock.forward,
        "attention": core_h3.Attention.forward,
        "mlp": core_h3.MLP.forward,
    }
    signatures = {}
    for name, function in functions.items():
        parameters = tuple(inspect.signature(function).parameters)
        missing = sorted(required[name] - set(parameters))
        if missing:
            raise RuntimeError(
                f"Unsupported ComfyUI H3 {name} contract; missing parameters: {missing}"
            )
        signatures[name] = list(parameters)
    # Current Core calls DiTBlock.forward without ``attention``.  Older H3
    # builds exposed it as an optional replacement callback.  The wrapper
    # below keeps an optional argument so both call sites remain valid, but it
    # must not require that legacy-only name during capability validation.
    for index, block in enumerate(blocks):
        if not isinstance(block, core_h3.DiTBlock):
            raise ValueError(f"H3 block {index} is not a native DiTBlock")
        if not isinstance(getattr(block, "attn", None), core_h3.Attention):
            raise ValueError(f"H3 block {index} does not expose native Attention")
        if not isinstance(getattr(block, "mlp", None), core_h3.MLP):
            raise ValueError(f"H3 block {index} does not expose native MLP")
        attention = block.attn
        if (
            int(getattr(attention, "heads", 0)) < 1
            or int(getattr(attention, "head_dim", 0)) < 1
            or not hasattr(attention, "qkv_proj")
            or not hasattr(attention, "out_proj")
            or not hasattr(block.mlp, "fc1")
            or not hasattr(block.mlp, "fc2")
        ):
            raise ValueError(f"H3 block {index} has an incomplete Attention/MLP surface")
    return {
        "status": "semantic_contract_validated",
        "block_count": len(blocks),
        "signatures": signatures,
        "implementation": "native_comfyui_minimax_h3",
    }


def _record_paths(record: _PatchRecord | None) -> set[str]:
    return set() if record is None else set(record.paths)


def _validate_existing_receipt(
    model: Any,
    receipt: _MemoryReceipt,
    *,
    allow_foreign_owners: bool = False,
) -> list[str]:
    patches = getattr(model, "object_patches", None)
    if not isinstance(patches, dict):
        raise RuntimeError("MODEL does not expose object patch ownership")
    allowed: set[str] = set()
    replaced: list[str] = []
    for record in (receipt.attention, receipt.ffn):
        if record is None:
            continue
        if not (
            len(record.paths) == len(record.owners) == len(record.methods)
            and record.token is not None
            and record.wrapper is not None
        ):
            raise RuntimeError("MiniMax H3 memory patch receipt is incomplete")
        allowed.update(record.paths)
        for path, method in zip(record.paths, record.methods, strict=True):
            if patches.get(path) is not method:
                if path not in patches:
                    raise RuntimeError(f"MiniMax H3 memory patch receipt lost its path: {path}")
                if not callable(patches[path]):
                    raise TypeError(f"MiniMax H3 memory forward is not callable: {path}")
                replaced.append(path)
                warn_patch_stack(f"MiniMax H3 memory path has a later user-selected owner: {path}")
        wrappers = model.get_wrappers("diffusion_model", record.wrapper_key)
        if wrappers != [record.wrapper]:
            raise RuntimeError(
                "MiniMax H3 memory wrapper ownership changed after binding: "
                f"{record.wrapper_key}"
            )
    relevant = {
        path
        for path in patches
        if path.startswith("diffusion_model.blocks.")
        and path.endswith((".forward", ".attn.forward", ".mlp.forward"))
    }
    unknown = sorted((relevant - allowed) | set(replaced))
    if unknown and not allow_foreign_owners:
        warn_patch_stack('MiniMax H3 memory nodes retain existing block/attention/MLP owners: ' + repr(unknown[:12]))
    return unknown


def _capture_fast_h3_v2_owner(model: Any):
    getter = getattr(model, "get_attachment", None)
    marker = getter(FAST_H3_V2_ATTACHMENT_KEY) if callable(getter) else getattr(
        model, "attachments", {}
    ).get(FAST_H3_V2_ATTACHMENT_KEY)
    if marker is None:
        return None
    from .fast_h3_v2_advanced import capture_fast_h3_v2_owner

    receipt = capture_fast_h3_v2_owner(model)
    if receipt is None:
        raise RuntimeError("FastH3 V2 memory bridge lost its owner receipt")
    return receipt


def _refresh_fast_h3_v2_memory(model: Any, receipt: Any):
    if receipt is None:
        return model
    from .fast_h3_v2_advanced import refresh_fast_h3_v2_memory

    return refresh_fast_h3_v2_memory(model)


def _validate_transformer_options(model: Any, *, attention: bool) -> tuple[dict, str, Any]:
    model_options = getattr(model, "model_options", None)
    if not isinstance(model_options, dict):
        raise RuntimeError("MODEL does not expose model_options")
    options = model_options.get("transformer_options")
    if not isinstance(options, dict):
        raise RuntimeError("MODEL transformer_options must be a dictionary")
    v2_receipt = _capture_fast_h3_v2_owner(model)
    if v2_receipt is not None:
        # capture authenticates wrapper/callback/receipt identity; this also
        # checks the precise block producers against the live options.
        v2_receipt.runtime.validate_options(options)
    replacements = options.get("patches_replace", {})
    if replacements and not isinstance(replacements, Mapping):
        raise RuntimeError("MODEL patches_replace is malformed")
    dit = replacements.get("dit", {}) if isinstance(replacements, Mapping) else {}
    if attention and dit and (v2_receipt is None or v2_receipt.profile == "dense_compat_exp"):
        warn_patch_stack('MiniMax H3 memory nodes cannot stack with DiT block replacement patches')
    if attention:
        patches = options.get("patches", {})
        if isinstance(patches, Mapping) and any(
            patches.get(name) for name in ("attn1_patch", "attn1_output_patch")
        ):
            warn_patch_stack('MiniMax H3 Low VRAM Attention cannot stack with attention hook patches')
        if "minimax_head_chunks" in options:
            warn_patch_stack('MiniMax H3 Low VRAM Attention found another head-chunk owner')
        if "sol_take_forward" in options:
            warn_patch_stack('MiniMax H3 Low VRAM Attention found another Sol forward delegate')
    override = options.get("optimized_attention_override")
    if override is not None and not callable(override):
        raise RuntimeError("optimized_attention_override is present but is not callable")
    backend = (
        f"authenticated_fast_h3_v2:{v2_receipt.profile}"
        if v2_receipt is not None
        else (
            "core_selected_attention"
            if override is None
            else "callable_global_override_preserved_group_safety_requires_backend_validation"
        )
    )
    return options, backend, v2_receipt


def _validate_modules_are_unpatched(
    blocks: tuple[Any, ...], *, allow_foreign_owners: bool = False
) -> list[str]:
    existing = []
    for index, block in enumerate(blocks):
        for label, module in (("block", block), ("attention", block.attn), ("mlp", block.mlp)):
            installed = getattr(module, "__dict__", {}).get("forward")
            if installed is None:
                continue
            function = getattr(installed, "__func__", installed)
            owner = getattr(installed, "__self__", None)
            if function is type(module).forward and owner is module:
                # ModelPatcher restores the native class method as an instance
                # attribute after an earlier run.  That is a clean state.
                continue
            if installed is not None:
                if allow_foreign_owners:
                    suffix = {"block": "forward", "attention": "attn.forward", "mlp": "mlp.forward"}[label]
                    existing.append(f"diffusion_model.blocks.{index}.{suffix}")
                    continue
                warn_patch_stack(f'H3 {label} {index} is still runtime-patched; unload/unpatch the MODEL first')
    return existing


def _runtime_options(args, kwargs) -> dict:
    options = kwargs.get("transformer_options")
    if options is None and len(args) >= 4:
        options = args[3]
    if not isinstance(options, dict):
        raise RuntimeError("MiniMax H3 memory runtime transformer_options are missing")
    return options


def _bind_runtime_guard(model: Any, record: _PatchRecord) -> Any:
    def guard(executor, *args, **kwargs):
        options = _runtime_options(args, kwargs)
        tokens = options.get(RUNTIME_TOKEN_KEY)
        if not isinstance(tokens, Mapping) or tokens.get(record.kind) is not record.token:
            raise RuntimeError(
                f"MiniMax H3 {record.kind} memory runtime receipt is missing or replaced"
            )
        for path, owner, method in zip(
            record.paths, record.owners, record.methods, strict=True
        ):
            if getattr(owner, "forward", None) is not method:
                warn_patch_stack(f'MiniMax H3 {record.kind} memory forward was replaced after binding: {path}')
        return executor(*args, **kwargs)

    model.add_wrapper_with_key("diffusion_model", record.wrapper_key, guard)
    return guard


def inspect_t8_memory_composition(
    model: Any,
    *,
    allowed_wrapper_keys: tuple[str | None, ...] = (),
) -> dict[str, Any] | None:
    """Authenticate T8 memory-node ownership for cache/resume identities."""

    receipt = _receipt(model)
    records = tuple(
        record for record in (receipt.attention, receipt.ffn) if record is not None
    )
    if not records:
        return None
    foreign = _validate_existing_receipt(model, receipt)
    if foreign:
        return None  # Not authenticated for portable cache reuse; execution is allowed.
    options = model.model_options.get("transformer_options")
    if not isinstance(options, dict):
        raise RuntimeError("MiniMax H3 memory transformer options are missing")
    runtime_tokens = options.get(RUNTIME_TOKEN_KEY)
    if not isinstance(runtime_tokens, Mapping):
        raise RuntimeError("MiniMax H3 memory runtime token map is missing")
    methods: dict[str, Any] = {}
    settings: dict[str, dict[str, int]] = {}
    wrapper_keys: list[str] = []
    for record in records:
        if runtime_tokens.get(record.kind) is not record.token:
            raise RuntimeError(
                f"MiniMax H3 {record.kind} memory runtime token changed after binding"
            )
        methods.update(zip(record.paths, record.methods, strict=True))
        settings[record.kind] = dict(record.settings)
        wrapper_keys.append(record.wrapper_key)
    active_wrappers = {
        key
        for wrapper_type in model.wrappers.values()
        for key, values in wrapper_type.items()
        if values
    }
    # Core's anonymous wrapper group uses None, NOT the string "None".
    # Preserve that exact key; coercion both misses real Union2 composition
    # and can accidentally authorize the distinct string-named group.
    allowed_keys = set(wrapper_keys) | {
        None if key is None else str(key) for key in allowed_wrapper_keys
    }
    if active_wrappers != allowed_keys:
        warn_patch_stack('MiniMax H3 memory MODEL contains wrappers outside its authenticated receipt')
        return None
    head_chunks = settings.get("attention", {}).get("head_chunks", 1)
    ffn = settings.get("ffn")
    return {
        "kind": "t8_h3_memory",
        "head_chunks": int(head_chunks),
        "ffn_settings": (
            None
            if ffn is None
            else [int(ffn["chunks"]), int(ffn["seq_threshold"])]
        ),
        "source_sha256s": [
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        ],
        "methods": methods,
        "wrapper_keys": sorted(wrapper_keys),
        "runtime_option_keys": sorted(
            {RUNTIME_TOKEN_KEY}
            | ({"sol_take_forward"} if receipt.attention is not None else set())
        ),
    }


def _install_runtime_token(model: Any, record: _PatchRecord) -> None:
    options = model.model_options["transformer_options"]
    existing = options.get(RUNTIME_TOKEN_KEY, {})
    if not isinstance(existing, Mapping):
        raise RuntimeError("MiniMax H3 memory runtime token map is malformed")
    if record.kind in existing:
        raise RuntimeError(f"MiniMax H3 {record.kind} memory token already exists")
    options[RUNTIME_TOKEN_KEY] = {**existing, record.kind: record.token}


def _attention_impl(self, x, head_chunks: int, rope_freqs=None, transformer_options={}):
    if isinstance(x, list):
        if len(x) != 1 or not torch.is_tensor(x[0]):
            raise ValueError(
                "MiniMax H3 Low VRAM Attention expects a single-tensor hand-off list"
            )
        x = x.pop()
    if not torch.is_tensor(x) or x.ndim != 2:
        raise ValueError(
            "MiniMax H3 Low VRAM Attention expects packed [tokens, hidden] input"
        )
    sequence = int(x.shape[0])
    device = x.device
    dtype = x.dtype
    heads = int(self.heads)
    head_dim = int(self.head_dim)
    qkv = self.qkv_proj(x)
    del x
    q, k, v = qkv.split(heads * head_dim, dim=-1)
    v = v.view(sequence, heads, head_dim)
    if rope_freqs is not None:
        q = q.view(1, sequence, heads, head_dim)
        k = k.view(1, sequence, heads, head_dim)
        q_weight = model_management.cast_to(self.q_norm.weight, device=device)
        k_weight = model_management.cast_to(self.k_norm.weight, device=device)
        rotation_dimension = rope_freqs.shape[-3] * 2
        if model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q,
                k,
                rope_freqs,
                q_weight,
                k_weight,
                epsilon=self.q_norm.eps,
                rot_dim=rotation_dimension,
            )
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q,
                k,
                rope_freqs,
                q_weight,
                k_weight,
                epsilon=self.q_norm.eps,
                rot_dim=rotation_dimension,
            )
        q, k = q[0], k[0]
    else:
        q = self.q_norm(q.view(sequence, heads, head_dim))
        k = self.k_norm(k.view(sequence, heads, head_dim))
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    groups = min(int(head_chunks), heads)
    if groups == 1:
        output = core_h3.optimized_attention(
            AttentionTensorContainer(q),
            AttentionTensorContainer(k),
            AttentionTensorContainer(v),
            heads,
            mask=None,
            skip_reshape=True,
            transformer_options=transformer_options,
        ).squeeze(0)
    else:
        output = torch.empty(
            (sequence, heads * head_dim), dtype=dtype, device=device
        )
        start = 0
        sizes = [
            heads // groups + (1 if index < heads % groups else 0)
            for index in range(groups)
        ]
        for size in sizes:
            stop = start + size
            group_output = core_h3.optimized_attention(
                AttentionTensorContainer(q[:, start:stop]),
                AttentionTensorContainer(k[:, start:stop]),
                AttentionTensorContainer(v[:, start:stop]),
                size,
                mask=None,
                skip_reshape=True,
                transformer_options=transformer_options,
            ).squeeze(0)
            expected = (sequence, size * head_dim)
            if tuple(group_output.shape) != expected:
                raise RuntimeError(
                    "Attention backend returned an incompatible grouped output shape: "
                    f"expected {expected}, got {tuple(group_output.shape)}"
                )
            output[:, start * head_dim : stop * head_dim] = group_output
            start = stop
    del q, k, v, qkv
    return self.out_proj(output)


def _make_attention_forward(head_chunks: int):
    def forward(self, x, rope_freqs=None, transformer_options={}):
        return _attention_impl(
            self,
            x,
            head_chunks,
            rope_freqs=rope_freqs,
            transformer_options=transformer_options,
        )

    # ComfyUI-SolAttn_triton checks this marker when it is connected after us.
    forward._uses_optimized_attention = True
    forward._t8_h3_low_vram_attention = True
    forward._t8_h3_head_chunks = int(head_chunks)
    return forward


def _make_block_forward():
    def forward(
        self,
        x,
        t_emb,
        mod_segments,
        rope_freqs,
        transformer_options={},
        attention=None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln_proj(t_emb)
        )
        hidden = core_h3._mod_scale_shift(
            self.norm1(x), shift_msa, scale_msa, mod_segments
        )
        if attention is None:
            handoff = [hidden]
            attention_output = self.attn(
                handoff,
                rope_freqs=rope_freqs,
                transformer_options=transformer_options,
            )
            if handoff:
                raise RuntimeError("Low VRAM attention did not consume the block hand-off")
        else:
            # Native sparse block hooks inject a gate-aware producer accepting
            # the normalized tensor, not the private early-release list.
            attention_output = attention(
                hidden,
                rope_freqs=rope_freqs,
                transformer_options=transformer_options,
            )
        del hidden
        x = core_h3._mod_gate(x, gate_msa, attention_output, mod_segments)
        hidden = core_h3._mod_scale_shift(
            self.norm2(x), shift_mlp, scale_mlp, mod_segments
        )
        return core_h3._mod_gate(x, gate_mlp, self.mlp(hidden), mod_segments)

    forward._t8_h3_low_vram_block = True
    return forward


def _make_ffn_forward(chunks: int, seq_threshold: int, previous_forward=None):
    def run(self, x):
        if previous_forward is not None:
            return previous_forward(x)
        return comfy.ops.linear_input_act(self.fc2, self.fc1(x), "swiglu")

    def forward(self, x):
        if not torch.is_tensor(x) or x.ndim != 2:
            raise ValueError(
                "MiniMax H3 Chunk FeedForward expects packed [tokens, hidden] input"
            )
        if int(x.shape[0]) <= seq_threshold:
            return run(self, x)
        output = torch.empty_like(x)
        offset = 0
        for chunk in torch.chunk(x, chunks, dim=0):
            stop = offset + int(chunk.shape[0])
            output[offset:stop] = run(self, chunk)
            offset = stop
        if offset != int(x.shape[0]):
            raise RuntimeError("Chunk FeedForward did not cover every packed token")
        return output

    forward._t8_h3_chunk_ffn = True
    forward._t8_h3_chunks = int(chunks)
    forward._t8_h3_seq_threshold = int(seq_threshold)
    return forward


def configure_low_vram_attention(model: Any, head_chunks: int):
    groups = int(head_chunks)
    if groups < 1 or groups > 56:
        raise ValueError("head_chunks must be in 1..56")
    diffusion, blocks = _diffusion_model(model)
    del diffusion
    contract = _core_contract(blocks)
    receipt = _receipt(model)
    if receipt.attention is not None:
        raise RuntimeError("MiniMax H3 Low VRAM Attention is already installed")
    _validate_existing_receipt(model, receipt)
    options, backend, v2_receipt = _validate_transformer_options(model, attention=True)
    del options
    _validate_modules_are_unpatched(blocks)
    if getattr(model, "object_patches_backup", {}):
        warn_patch_stack('MODEL is currently patched; unload it before adding memory nodes')

    cloned = model.clone()
    token = object()
    paths: list[str] = []
    owners: list[Any] = []
    methods: list[Any] = []
    block_forward_function = _make_block_forward()
    attention_forward_function = _make_attention_forward(groups)
    for index, block in enumerate(blocks):
        block_path = f"diffusion_model.blocks.{index}.forward"
        attention_path = f"diffusion_model.blocks.{index}.attn.forward"
        foreign = any(path in model.object_patches for path in (block_path, attention_path))
        foreign = foreign or any(
            getattr(module, "__dict__", {}).get("forward") is not None
            and getattr(module.forward, "__func__", None) is not type(module).forward
            for module in (block, block.attn)
        )
        if foreign:
            warn_patch_stack(f"Low VRAM Attention retains foreign block/attention {index}; head grouping may be bypassed")
            continue
        block_method = MethodType(block_forward_function, block)
        cloned.add_object_patch(block_path, block_method)
        paths.append(block_path)
        owners.append(block)
        methods.append(block_method)

        attention_path = f"diffusion_model.blocks.{index}.attn.forward"
        attention_method = MethodType(attention_forward_function, block.attn)
        cloned.add_object_patch(attention_path, attention_method)
        paths.append(attention_path)
        owners.append(block.attn)
        methods.append(attention_method)
    record = _PatchRecord(
        kind="attention",
        token=token,
        paths=tuple(paths),
        owners=tuple(owners),
        methods=tuple(methods),
        settings=(("head_chunks", groups),),
        wrapper_key=ATTENTION_WRAPPER_KEY,
    )
    wrapper = _bind_runtime_guard(cloned, record)
    record = replace(record, wrapper=wrapper)
    updated_receipt = _MemoryReceipt(
        attention=record,
        ffn=receipt.ffn,
    )
    _set_receipt(cloned, updated_receipt)
    _install_runtime_token(cloned, record)
    # A downstream ComfyUI-SolAttn_triton node can delegate eligible calls to
    # this exact low-memory forward instead of discarding it.
    cloned.model_options["transformer_options"].setdefault("sol_take_forward", attention_forward_function)
    report = {
        "schema": SCHEMA,
        "status": "active",
        "node": "MiniMaxH3LowVRAMAttentionT8Advanced",
        "algorithm": "early_release_plus_head_grouped_attention",
        "head_chunks": groups,
        "effective_groups": f"min({groups}, model_heads)",
        "block_count": len(blocks),
        "core_contract": contract,
        "attention_backend": backend,
        "source_model_unchanged": True,
        "memory_safe_claim": False,
        "bit_exact_claim": False,
        "compatibility": {
            "direct": [
                "native ComfyUI MiniMax H3",
                "ordinary weight LoRA applied before this node",
                "the sibling T8 Chunk FeedForward node in either order",
            ],
            "conditional": [
                "a callable global optimized_attention_override is preserved and receives each head group",
                "ComfyUI-SolAttn_triton connected after this node can use sol_take_forward",
            ],
            "unverified_user_risk": [
                "another block/attention forward owner, including KJ H3 memory Sage/LowVRAM",
                "DiT block replacement hooks",
                "another head-chunk or Sol-forward delegate owner",
            ],
        },
        "installed_attention_blocks": len(paths) // 2,
        "preserved_foreign_blocks": len(blocks) - len(paths) // 2,
        "scientific_boundary": (
            "The equations are preserved, but grouped kernel launch shapes can change floating-point rounding. "
            "The prior 6.47% combined peak-VRAM reduction was one fixed local prototype measurement, "
            "not a universal claim for this node."
        ),
    }
    return _refresh_fast_h3_v2_memory(cloned, v2_receipt), report


def configure_chunk_feed_forward(model: Any, chunks: int, seq_threshold: int):
    count = int(chunks)
    threshold = int(seq_threshold)
    if count < 1 or count > 64:
        raise ValueError("chunks must be in 1..64")
    if threshold < 256 or threshold > 262144:
        raise ValueError("seq_threshold must be in 256..262144")
    if count == 1:
        return model, {
            "schema": SCHEMA,
            "status": "identity",
            "node": "MiniMaxH3ChunkFeedForwardT8Advanced",
            "chunks": 1,
            "seq_threshold": threshold,
            "source_model_unchanged": True,
            "memory_safe_claim": False,
            "bit_exact_claim": True,
            "identity_contract": "returns the exact same MODEL object without validation, clone, patch or receipt",
        }

    diffusion, blocks = _diffusion_model(model)
    del diffusion
    contract = _core_contract(blocks)
    receipt = _receipt(model)
    if receipt.ffn is not None:
        raise RuntimeError("MiniMax H3 Chunk FeedForward is already installed")
    foreign_paths = _validate_existing_receipt(model, receipt, allow_foreign_owners=True)
    _, _, v2_receipt = _validate_transformer_options(model, attention=False)
    runtime_paths = _validate_modules_are_unpatched(blocks, allow_foreign_owners=True)
    if getattr(model, "object_patches_backup", {}):
        warn_patch_stack('MODEL is currently patched; unload it before adding memory nodes')

    options = model.model_options["transformer_options"]
    dit = options.get("patches_replace", {}).get("dit", {})
    compatibility_warnings = []
    if foreign_paths or runtime_paths or dit:
        warning = (
            "MiniMax H3 Chunk FeedForward: existing third-party block/attention/MLP patches "
            "are allowed and preserved/delegated, not rejected. Compatibility, chunking benefit "
            "and numerical equivalence are unverified; use this combination at your own risk."
        )
        logger.warning("%s Object owners: %s; runtime owners: %s; DiT hooks: %s",
                       warning, foreign_paths, runtime_paths, list(dit))
        compatibility_warnings.append(warning)

    cloned = model.clone()
    token = object()
    paths: list[str] = []
    owners: list[Any] = []
    methods: list[Any] = []
    delegated_mlp_paths = []
    for index, block in enumerate(blocks):
        path = f"diffusion_model.blocks.{index}.mlp.forward"
        previous = model.object_patches.get(path, block.mlp.forward)
        native = (
            getattr(previous, "__self__", None) is block.mlp
            and getattr(previous, "__func__", None) is core_h3.MLP.forward
        )
        if not callable(previous):
            raise TypeError(f"Existing MLP forward is not callable: {path}")
        if not native:
            delegated_mlp_paths.append(path)
        ffn_forward_function = _make_ffn_forward(count, threshold, None if native else previous)
        method = MethodType(ffn_forward_function, block.mlp)
        cloned.add_object_patch(path, method)
        paths.append(path)
        owners.append(block.mlp)
        methods.append(method)
    record = _PatchRecord(
        kind="ffn",
        token=token,
        paths=tuple(paths),
        owners=tuple(owners),
        methods=tuple(methods),
        settings=(("chunks", count), ("seq_threshold", threshold)),
        wrapper_key=FFN_WRAPPER_KEY,
    )
    wrapper = _bind_runtime_guard(cloned, record)
    record = replace(record, wrapper=wrapper)
    updated_receipt = _MemoryReceipt(
        attention=receipt.attention,
        ffn=record,
    )
    _set_receipt(cloned, updated_receipt)
    _install_runtime_token(cloned, record)
    report = {
        "schema": SCHEMA,
        "status": "conditional_active",
        "node": "MiniMaxH3ChunkFeedForwardT8Advanced",
        "algorithm": "packed_token_axis_swiglu_chunking",
        "chunks": count,
        "seq_threshold": threshold,
        "activation_policy": "chunk only when packed_rows > seq_threshold",
        "threshold_boundary": "packed_rows <= seq_threshold keeps the native one-call path",
        "block_count": len(blocks),
        "core_contract": contract,
        "source_model_unchanged": True,
        "memory_safe_claim": False,
        "bit_exact_claim": False,
        "compatibility_policy": "warn_and_continue_at_user_risk",
        "compatibility_warnings": compatibility_warnings,
        "existing_forward_owners": foreign_paths,
        "existing_runtime_owners": runtime_paths,
        "existing_dit_hooks": [str(key) for key in dit],
        "delegated_mlp_paths": delegated_mlp_paths,
        "compatibility": {
            "direct": [
                "native ComfyUI MiniMax H3",
                "ordinary weight LoRA applied before this node",
                "attention-only backends that do not replace the DiT block or MLP forward",
                "the sibling T8 Low VRAM Attention node in either order",
            ],
            "allowed_at_user_risk": [
                "existing KJ/Sage/Sol or other block/attention forward owners are preserved",
                "existing MLP forward owners are called on each token chunk",
                "existing DiT block replacement hooks are preserved; a replacement can bypass MLP chunking",
            ],
        },
        "scientific_boundary": (
            "SwiGLU remains token-local, but changing GEMM row shapes can change floating-point rounding. "
            "Savings depend on sequence length, backend fusion, allocator state and other GPU users."
        ),
    }
    return _refresh_fast_h3_v2_memory(cloned, v2_receipt), report
