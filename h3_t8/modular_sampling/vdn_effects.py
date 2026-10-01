"""FETA at original VDN softmax/linear projection inputs, without changing VDN.

The old block/attention/window/linear functions remain authoritative. Ephemeral
attribute views add bounded QKV statistics and target-video gain only. No global
monkeypatch, dense replacement, extra model forward or live module mutation.
"""
from types import SimpleNamespace

from .. import vdn_h3_advanced as vdn, enhance_a_video_advanced as feta
from ..patch_stack_policy import warn_patch_stack, UnverifiedModelStack
from ..vdn_attention_compat import _factory_closure
from . import sparse_eav


def wrap(original, block, branch, runtime, relay_runtime=None):
    def wrapped(args, extra):
        route = args["transformer_options"].get(feta.EAV_RUNTIME_KEY)
        if route is None:
            if relay_runtime is not None:
                from .vdn_relay import execute_block
                return execute_block(block, branch, args, extra["original_block"], relay_runtime)
            return original(args, extra)
        gain = None

        def qkv(hidden):
            nonlocal gain
            value = block.attn.qkv_proj(hidden)
            if route["active"]:
                gain = sparse_eav._gain(block.attn, hidden, args["rope_freqs"], route)
            return value

        def softmax_projection(value):
            if gain is not None and route["mode"] == "apply_exp":
                value[int(route["video_start"]):int(route["video_end"])].mul_(gain.to(value))
            return block.attn.out_proj(value)

        def linear_projection(value):
            if gain is not None and route["mode"] == "apply_exp":
                value.mul_(gain.to(value))  # The original linear output has video rows only.
            return branch.to_out_linear(value)

        attention = SimpleNamespace(heads=block.attn.heads, head_dim=block.attn.head_dim,
            q_norm=block.attn.q_norm, k_norm=block.attn.k_norm, qkv_proj=qkv, out_proj=softmax_projection)
        block_view = SimpleNamespace(attn=attention, adaln_proj=block.adaln_proj,
            norm1=block.norm1, norm2=block.norm2, mlp=block.mlp)
        branch_view = SimpleNamespace(softmax_gate=branch.softmax_gate,
            linear_attention=branch.linear_attention, to_out_linear=linear_projection)
        if relay_runtime is not None:
            from .vdn_relay import execute_block
            before = relay_runtime.completed_blocks
            output = execute_block(block_view, branch_view, args, extra["original_block"], relay_runtime)
            runtime.relay_calls += relay_runtime.completed_blocks - before
        else:
            output = vdn._vdn_block(block_view, branch_view, args, extra["original_block"])
        runtime.sparse_calls += 1
        return output
    return wrapped


def unwrap(hook):
    state = _factory_closure(hook, wrap, "wrapped")
    if state is None:
        return hook
    if (set(state) != {"original", "block", "branch", "runtime", "relay_runtime"} or hook.__defaults__ is not None
            or hook.__kwdefaults__ is not None or vars(hook)):
        raise UnverifiedModelStack("VDN effect wrapper source/fields changed")
    return state["original"]


def install(model, runtime):
    from .vdn_identity import original_hook
    branches = model.additional_models.get(vdn.ADDITIONAL_MODEL_KEY, [])
    if len(branches) != 1:
        warn_patch_stack("VDN effects cannot authenticate its additional branch; retain current producers")
        return
    options = model.model_options["transformer_options"]
    hooks = list(options[vdn.OWNER_HOOKS_KEY])
    active = options.get("patches_replace", {}).get("dit", {})
    for index, (block, branch) in enumerate(zip(model.get_model_object("diffusion_model").blocks, branches[0].model.blocks)):
        original = active.get(("double_block", index))
        state = _factory_closure(original, wrap, "wrapped")
        source = unwrap(original)
        if original is not hooks[index] or not original_hook(source, block, branch):
            warn_patch_stack("VDN effects retain an unknown DiT producer; effect coverage remains unverified")
            continue
        hooks[index] = wrap(source, block, branch, runtime, None if state is None else state["relay_runtime"])
        model.set_model_patch_replace(hooks[index], "dit", "double_block", index)
    model.model_options["transformer_options"][vdn.OWNER_HOOKS_KEY] = tuple(hooks)


def project(model, runtime):
    from .vdn_identity import require, original_hook
    view = model.clone()
    options = view.model_options["transformer_options"]
    hooks = options[vdn.OWNER_HOOKS_KEY]
    branches = view.additional_models[vdn.ADDITIONAL_MODEL_KEY][0].model.blocks
    originals = []
    for index, (block, branch, hook) in enumerate(zip(view.get_model_object("diffusion_model").blocks, branches, hooks)):
        state = _factory_closure(hook, wrap, "wrapped")
        require(state is not None and state.get("runtime") is runtime and state.get("block") is block
                and state.get("branch") is branch
                and options["patches_replace"]["dit"].get(("double_block", index)) is hook,
                "stage effect hook changed its actual runtime/branch owner")
        original = unwrap(hook)
        require(original_hook(original, block, branch), "stage effect lost original Composer hook")
        view.set_model_patch_replace(original, "dit", "double_block", index)
        originals.append(original)
    view.model_options["transformer_options"][vdn.OWNER_HOOKS_KEY] = tuple(originals)
    return view
