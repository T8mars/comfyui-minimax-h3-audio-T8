"""Value-only bridge to shared native effect encoding, never old sampling/cache."""
from __future__ import annotations

from dataclasses import replace
from contextlib import contextmanager
import json

from .source import canonical


@contextmanager
def pass_canvas(state, canvas, role):
    """Scope only effect geometry to the actual stage, restoring even on error."""
    if state is None:
        yield
        return
    original = state.canvas
    state.canvas = dict(canvas)
    try:
        yield
    finally:
        state.canvas = original


def descriptor(model):
    """Borrow the existing *data* contract, not an older engine producer.

    The shared native builder/transport never loads a runtime from this value.
    H21 dispatch, source, Stage and completion identity stay entirely separate.
    """
    from ..freevideo_exp.runtime import FreeVideoModel
    value = json.loads(model.effects_json)
    return FreeVideoModel(model.config_path, model.config_sha256, model.loras,
        canonical(value["eav"]) if value.get("eav") is not None else None,
        canonical(value["relay"]) if value.get("relay") is not None else None)


def with_eav(model, **kwargs):
    from ..freevideo_exp.effects import validate_eav
    value = dict(mode="report_only", tau=4., start=.15, end=.9, workspace_mib=32, g_hard_limit=1.5)
    value.update(kwargs)
    validate_eav(value)
    effects = json.loads(model.effects_json)
    effects["eav"] = value
    return replace(model, effects_json=canonical(effects))


def build_relay_conditioning(freevideo_model, **kwargs):
    from ..freevideo_exp.effects import build_relay_conditioning as native_builder
    # Fresh single native encoding, retaining authoritative text/physical media
    # identities. An explicit new Relay node can replace the prior plan; it does
    # not concatenate old embeddings or change LoRA/model selections.
    value = descriptor(freevideo_model)
    from .runtime import config_for
    from .assets import model_structure
    text_dim = model_structure(config_for(freevideo_model)["base"])["text_dim"]
    paired, *outputs = native_builder(replace(value, relay=None), text_dim=text_dim, **kwargs)
    effects = json.loads(freevideo_model.effects_json)
    effects["relay"] = json.loads(paired.relay)
    return replace(freevideo_model, effects_json=canonical(effects)), *outputs


def transport(model, conditioning, canvas, *, text_dim=5120):
    from ..freevideo_exp.effects import transport_effects, validate_eav
    value = descriptor(model)
    if type(conditioning) is dict:
        if value.relay is not None:
            raise ValueError("Connect the paired native H21 Relay CONDITIONING; raw JSON has no authoritative token binding")
        return dict(eav=validate_eav(json.loads(value.eav)) if value.eav else None, relay=None)
    return (transport_effects(value, conditioning, canvas) if text_dim == 5120 else
            transport_effects(value, conditioning, canvas, text_dim=text_dim))


def install(engine, effects, canvas):
    """New-family lifecycle, shared equations, actual backend and layer count.

    Legacy string-only cuDNN aliases and 50-layer producer checks are not v4
    input contracts. Unavailable optional effects preserve original execution
    and explicitly report non-consumption, never silently enable kernels.
    """
    import types
    from ..freevideo_exp.effects import validate_eav, validate_binding
    from ..freevideo_exp import worker_effects as shared
    from freevideo_engine.head_chunk import install_head_chunks
    from src.models.hybrid_transform import iter_hybrids

    if effects.get("eav") is not None:
        validate_eav(effects["eav"])
    if effects.get("relay") is not None:
        validate_binding(effects["relay"])
    hybrids = list(iter_hybrids(engine.transformer))
    backend = (getattr(engine.attention, "global_backend", None), getattr(engine.attention, "window_backend", None))

    class Runtime(shared.Runtime):
        unavailable = None

        def snapshot(self):
            value = super().snapshot()
            value.update(producer="t8.freevideo.h21.effects.v4", actual_hybrid_blocks=len(hybrids),
                actual_attention_backends=list(backend), actual_requested_attention=engine.config.get("attention"),
                partial=self.unavailable is not None, optimization_enabled=bool(self.originals),
                effects_consumed=self.blocks > 0, warning=self.unavailable,
                CUDA_kernel_qualification=False, trained_quality_accepted=False)
            return value

    state = Runtime(engine, effects, canvas)
    if not any(effects.values()):
        return state
    config = engine.config
    if (backend != ("cudnn", "cudnn") or config.get("head_parallelism") != 1
            or not config.get("inference_kernels") or type(config.get("head_chunk")) is not int or config["head_chunk"] < 1):
        state.unavailable = ("Optional H21 effects not installed: actual API needs serial cuDNN global/window "
            "with already-enabled inference kernels and a positive explicit head chunk. Original MODEL/backend "
            "execution is preserved; no effect consumption or portable optimization is claimed.")
        return state
    if not hybrids:
        state.unavailable = "Optional H21 effects found no actual HybridAttention residual API; original execution is preserved."
        return state
    state.originals = [(attn, attn._hybrid_forward, attn.linear_attention._readout_inference) for attn in hybrids]
    try:
        policy = shared.EffectPolicy(engine.attention, state)
        install_head_chunks(engine.transformer, policy, config["head_chunk"],
            cpu_outputs=config["attention_cpu_outputs"], projection_chunk=config["projection_chunk"] or 1024,
            grouped_outputs=config["grouped_attention_outputs"], parallelism=1)
        state.handles.append(engine.transformer.register_forward_pre_hook(state.before, with_kwargs=True))
        for index, attn in enumerate(hybrids):
            attention, readout = attn._hybrid_forward, attn.linear_attention._readout_inference
            def wrapped(attn, hidden, rotary, original=attention, block_index=index):
                state.gain = None
                if state.eav and state.eav["mode"] != "disabled" and state.eav["start"] <= state.progress <= state.eav["end"]:
                    state.gain, measured = shared.projected_gain(attn, hidden, rotary, state.eav, config["head_chunk"])
                    state.eav_measurements.append(dict(forward=state.forwards, block=block_index,
                                                      progress=state.progress, **measured))
                output = original(hidden, rotary)
                state.blocks += 1
                state.gain = None
                return output
            attn._hybrid_forward = types.MethodType(wrapped, attn)
            def linear(module, *args, original=readout, owner=attn, **kwargs):
                binding = state.binding
                if binding is None or binding["mode"] != "apply_exp" or state.neutral:
                    return original(*args, **kwargs)
                times = state.video_times.view(-1, owner.layout.tokens_per_frame)[:, 0]
                frames = args[1]
                if len(times) == frames + 2:
                    times = times[1:-1]
                if len(times) != frames:
                    raise ValueError("H21 Relay linear frame grid differs from actual anchor pruning")
                if all(not bool(shared.penalty(times, event).any()) for event in binding["events"]):
                    return original(*args, **kwargs)
                return shared.relay_readout(module, *args, **kwargs, times=times, binding=binding, stats=state.stats)
            attn.linear_attention._readout_inference = types.MethodType(linear, attn.linear_attention)
            def gain_hook(module, args, output):
                if state.gain is not None and state.eav["mode"] == "apply_exp":
                    output.mul_(state.gain.to(output))
                return output
            state.handles.append(attn.linear_attention.register_forward_hook(gain_hook))
    except BaseException:
        state.close()
        raise
    return state
