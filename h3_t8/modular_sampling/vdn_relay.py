"""Explicit VDN Relay producer adapter, independent of Stage EAV."""
from dataclasses import asdict, dataclass
import json

import comfy.patcher_extension as extension

from .. import vdn_h3_advanced as vdn, prompt_relay_advanced as relay
from ..patch_stack_policy import UnverifiedModelStack, warn_patch_stack
from ..vdn_attention_compat import _factory_closure
from . import vdn_relay_math as numerical, vdn_stages
from .contracts import StageContext

KEY = "t8_modular_vdn_relay_v1"
RUNTIME_TYPE = "T8_VDN_RELAY_RUNTIME"


@dataclass(frozen=True)
class Config:
    mode: str = "apply_exp"
    max_workspace_mib: int = 64

    def __post_init__(self):
        if self.mode not in ("disabled", "apply_exp"):
            raise ValueError("VDN Relay adapter mode must be disabled/apply_exp")
        if type(self.max_workspace_mib) is not int or not 4 <= self.max_workspace_mib <= 1024:
            raise ValueError("VDN Relay workspace must be an integer in [4,1024] MiB")


class Runtime:
    def __init__(self, config, context, binding_hash, query_chunk_rows, blocks):
        self.config, self.context, self.binding_hash = config, context, binding_hash
        self.query_chunk_rows, self.blocks = query_chunk_rows, blocks
        self.installed_blocks = 0
        self.closed = True
        self.completed_blocks = self.neutral_blocks = 0
        self.stats, self.aborted = {}, None

    def prepare(self, patcher, timestep, model_options):
        if self.closed:
            self.completed_blocks = self.neutral_blocks = 0
            self.stats, self.aborted = {}, None
            self.closed = False

    def cleanup(self, patcher):
        self.closed = True

    def snapshot(self):
        expected = self.context.steps * self.blocks
        status = ("disabled_identity" if self.config.mode == "disabled" else "aborted" if self.aborted else
                  "unverified_incomplete_coverage" if self.installed_blocks != self.blocks or self.completed_blocks != expected
                  else "observed_neutral" if self.neutral_blocks == expected else "observed_apply_exp")
        return {"schema": "t8.modular-sampling.vdn-relay-audit.v1", "status": status,
                "config": asdict(self.config), "linear_profile": numerical.PROFILE,
                "stage_context": self.context.to_dict(), "binding_hash": self.binding_hash,
                "completed_blocks": self.completed_blocks, "expected_blocks": expected,
                "neutral_blocks": self.neutral_blocks, "installed_blocks": self.installed_blocks,
                "stats": dict(self.stats), "aborted": self.aborted,
                "boundary": "Original window key sets + temporal logit bias. Linear beta-weighted nonlinear text "
                            "seed is a VDN experimental extension, not softmax equivalence or trained/GPU quality. "
                            "Workspace bounds bias and estimates head-chunk scan/factor buffers, not base features, "
                            "backend allocator or total VRAM. No fallback or extra diffusion NFE."}


def attention(block, branch, x, rope, layout, route, runtime):
    query, key, value, raw = vdn._qkv(block.attn, x, rope)
    bounds = vdn.window_bounds(layout.num_frames, radius=1, chunk=5)
    softmax = numerical.window(query, key, value, layout, bounds, block.attn.head_dim**-.5,
        route, runtime.query_chunk_rows, runtime.config.max_workspace_mib << 20, runtime.stats)
    softmax = softmax * branch.softmax_gate(x)
    output = block.attn.out_proj(softmax.reshape(x.shape[0], -1).type_as(x))
    if all(lo <= 0 and hi >= layout.num_frames - 1 for lo, hi in bounds):
        return output
    vs, ve, ts, te = layout.video_start, layout.video_end, layout.text_start, layout.text_start + layout.text_len
    linear = numerical.linear(branch.linear_attention, x[vs:ve], layout, bounds,
        tuple(z[vs:ve] for z in raw), x[ts:te], tuple(z[ts:te] for z in raw), route,
        runtime.config.max_workspace_mib << 20, runtime.stats)
    output[vs:ve].add_(branch.to_out_linear(linear.type_as(x)))
    return output


def execute_block(block, branch, args, original, runtime):
    route = args["transformer_options"].get(relay.PROMPT_RELAY_RUNTIME_KEY)
    if route is None:
        # Unknown upstream owner may bypass Relay. Keep it; no successful audit.
        return vdn._vdn_block(block, branch, args, original)
    try:
        if route.get("binding_hash") != runtime.binding_hash:
            raise ValueError("VDN Relay MODEL and actual route binding hashes differ")
        layout = args["transformer_options"][vdn.LAYOUT_KEY]
        numerical.validate_route(route, layout)
        if numerical.neutral(route):
            result = vdn._vdn_block(block, branch, args, original)
            runtime.neutral_blocks += 1
        else:
            shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(args["t_emb"])
            hidden = vdn.minimax_model._mod_scale_shift(block.norm1(args["img"]), shift, scale, args["mod_segments"])
            image = vdn.minimax_model._mod_gate(args["img"], gate,
                attention(block, branch, hidden, args["rope_freqs"], layout, route, runtime), args["mod_segments"])
            hidden = vdn.minimax_model._mod_scale_shift(block.norm2(image), shift_mlp, scale_mlp, args["mod_segments"])
            result = {"img": vdn.minimax_model._mod_gate(image, gate_mlp, block.mlp(hidden), args["mod_segments"])}
        runtime.completed_blocks += 1
        return result
    except BaseException as exc:
        runtime.aborted = f"{type(exc).__name__}: {exc}"
        raise


def apply(model, sigmas, av_latent, context, config):
    from . import vdn_effects, vdn_identity
    vdn_stages.validate_stage(model, sigmas, av_latent, context)
    if type(config) is not Config:
        raise TypeError("VDN Relay needs its explicit adapter configuration")
    if config.mode == "disabled":
        runtime = Runtime(config, context, "disabled", 32, len(model.get_model_object("diffusion_model").blocks))
        return model, runtime, json.dumps(runtime.snapshot(), ensure_ascii=False)
    if model.get_attachment(KEY) is not None:
        raise ValueError("VDN Relay already installed on this stage branch")
    contract = relay.prompt_relay_model_contract(model)
    if not contract or not model.get_wrappers("diffusion_model", relay.PROMPT_RELAY_WRAPPER_KEY):
        raise ValueError("Connect paired Relay Conditioning -> VDN Stage -> VDN Relay Apply -> optional Stage EAV")
    if not contract.get("attention_owner_verified") or contract.get("attention_backend") is not None:
        warn_patch_stack("VDN Relay retains the selected external Relay owner; its own window uses original VDN SDPA, "
                         "not a certification that the external backend also runs")
    prepared = model.clone()
    blocks = prepared.get_model_object("diffusion_model").blocks
    runtime = Runtime(config, context, contract["binding_hash"], contract["query_chunk_rows"], len(blocks))
    branches = prepared.additional_models[vdn.ADDITIONAL_MODEL_KEY][0].model.blocks
    options = prepared.model_options["transformer_options"]
    hooks = list(options[vdn.OWNER_HOOKS_KEY])
    active = options.get("patches_replace", {}).get("dit", {})
    for index, (block, branch, original) in enumerate(zip(blocks, branches, hooks)):
        if active.get(("double_block", index)) is not original or not vdn_identity.original_hook(original, block, branch):
            warn_patch_stack("VDN Relay retains an unknown producer; actual coverage remains unverified")
            continue
        hooks[index] = vdn_effects.wrap(original, block, branch, None, runtime)
        prepared.set_model_patch_replace(hooks[index], "dit", "double_block", index)
        runtime.installed_blocks += 1
    prepared.model_options["transformer_options"][vdn.OWNER_HOOKS_KEY] = tuple(hooks)
    prepared.set_attachments(KEY, runtime)
    prepared.add_callback_with_key(extension.CallbacksMP.ON_PREPARE_STATE, KEY, runtime.prepare)
    prepared.add_callback_with_key(extension.CallbacksMP.ON_CLEANUP, KEY, runtime.cleanup)
    return prepared, runtime, json.dumps(runtime.snapshot(), ensure_ascii=False, indent=2)


def project(model):
    """Authenticate then remove only our Relay component on an inspection clone."""
    from . import vdn_effects, vdn_identity
    runtime = model.get_attachment(KEY)
    if runtime is None:
        return model, None
    def require(value, message):
        if not value:
            raise UnverifiedModelStack("VDN Relay " + message)
    require(type(runtime) is Runtime and set(vars(runtime)) == {
        "config", "context", "binding_hash", "query_chunk_rows", "blocks", "installed_blocks", "closed",
        "completed_blocks", "neutral_blocks", "stats", "aborted"}, "runtime contains unknown execution state")
    require(type(runtime.config) is Config and type(runtime.context) is StageContext
            and runtime.config.mode == "apply_exp", "configuration/context changed")
    require(vdn_stages.capture_owner(model).context == runtime.context, "stage owner changed")
    blocks = model.get_model_object("diffusion_model").blocks
    require(len(blocks) == runtime.blocks == runtime.installed_blocks, "producer coverage is incomplete")
    view = model.clone()
    for role, name in ((extension.CallbacksMP.ON_PREPARE_STATE, "prepare"), (extension.CallbacksMP.ON_CLEANUP, "cleanup")):
        callbacks = model.callbacks.get(role, {}).get(KEY, [])
        require(len(callbacks) == 1 and getattr(callbacks[0], "__self__", None) is runtime
                and getattr(callbacks[0], "__func__", None) is getattr(Runtime, name), "lifecycle owner changed")
        view.callbacks[role].pop(KEY)
        if not view.callbacks[role]:
            view.callbacks.pop(role)
    options = model.model_options["transformer_options"]
    hooks = list(options[vdn.OWNER_HOOKS_KEY])
    branches = model.additional_models[vdn.ADDITIONAL_MODEL_KEY][0].model.blocks
    for index, (block, branch, hook) in enumerate(zip(blocks, branches, hooks)):
        state = _factory_closure(hook, vdn_effects.wrap, "wrapped")
        require(state is not None and state.get("relay_runtime") is runtime and state.get("block") is block
                and state.get("branch") is branch
                and options["patches_replace"]["dit"].get(("double_block", index)) is hook, "actual hook owner changed")
        original = vdn_effects.unwrap(hook)
        require(vdn_identity.original_hook(original, block, branch), "original Composer changed")
        hooks[index] = (original if state["runtime"] is None else
                        vdn_effects.wrap(original, block, branch, state["runtime"]))
        view.set_model_patch_replace(hooks[index], "dit", "double_block", index)
    view.model_options["transformer_options"][vdn.OWNER_HOOKS_KEY] = tuple(hooks)
    view.remove_attachments(KEY)
    return view, {"config": asdict(runtime.config), "stage_context": runtime.context.to_dict(),
                  "binding_hash": runtime.binding_hash, "query_chunk_rows": runtime.query_chunk_rows,
                  "linear_profile": numerical.PROFILE, "blocks": runtime.blocks}


def audit(av_latent, runtime):
    if type(runtime) is not Runtime:
        raise TypeError("VDN Relay Audit requires the matching runtime")
    video, audio = vdn_stages.sampling.nested_av_parts(av_latent)
    if tuple(video.shape) != runtime.context.video_shape or tuple(audio.shape) != runtime.context.audio_shape:
        raise ValueError("VDN Relay Audit actual AV geometry differs from its stage")
    report = runtime.snapshot()
    if report["status"] == "aborted":
        raise RuntimeError(report["aborted"])
    return av_latent, json.dumps(report, ensure_ascii=False, indent=2)
