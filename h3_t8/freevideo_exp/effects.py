"""Explicit FreeVideo branch descriptors and native authoritative Relay encoding.

Core MODEL patches cannot operate on an isolated FreeVideo engine. These frozen
descriptors are transported with the matching CONDITIONING instead; the worker
adapts the pinned head-chunk producer. Nothing is installed on an old MODEL.
"""
from dataclasses import replace
import hashlib
import json
import math

from .runtime import FreeVideoModel, canonical, geometry, tensor_record

RELAY_KEY = "t8_freevideo_relay"


def validate_eav(value):
    if not isinstance(value, dict) or set(value) != {"mode", "tau", "start", "end", "workspace_mib", "g_hard_limit"}:
        raise ValueError("Invalid FreeVideo EAV descriptor")
    if value["mode"] not in ("disabled", "report_only", "apply_exp"):
        raise ValueError("FreeVideo EAV mode must be disabled/report_only/apply_exp")
    for k in ("tau", "start", "end", "g_hard_limit"):
        if type(value[k]) not in (int, float) or not math.isfinite(value[k]):
            raise ValueError("FreeVideo EAV parameters must be finite")
    if not (-32 <= value["tau"] <= 32 and 0 <= value["start"] <= value["end"] <= 1 and 1 <= value["g_hard_limit"] <= 10):
        raise ValueError("FreeVideo EAV parameter range is invalid")
    if type(value["workspace_mib"]) is not int or not 1 <= value["workspace_mib"] <= 1024:
        raise ValueError("FreeVideo EAV workspace must be an integer in [1,1024] MiB")
    return value


def with_eav(model, mode="report_only", tau=4., start=.15, end=.9, workspace_mib=32, g_hard_limit=1.5):
    if not isinstance(model, FreeVideoModel) or model.eav is not None:
        raise ValueError("FreeVideo EAV needs its own branch; do not install it twice")
    value = validate_eav(dict(mode=mode, tau=tau, start=start, end=end, workspace_mib=workspace_mib, g_hard_limit=g_hard_limit))
    return replace(model, eav=canonical(value))


def validate_binding(value):
    if not isinstance(value, dict) or value.get("schema") != "t8-freevideo-relay-v1":
        raise ValueError("Invalid FreeVideo Relay binding")
    unsigned = dict(value)
    sha = unsigned.pop("sha256", None)
    if hashlib.sha256(canonical(unsigned).encode()).hexdigest() != sha:
        raise ValueError("FreeVideo Relay binding identity changed")
    if value.get("mode") not in ("report_only", "apply_exp") or value.get("query_route") not in ("video_only_paper", "joint_av_exp"):
        raise ValueError("Invalid FreeVideo Relay mode/route")
    if type(value.get("query_chunk_rows")) is not int or not 1 <= value["query_chunk_rows"] <= 2048:
        raise ValueError("Invalid FreeVideo Relay query chunk")
    workspace = value.get("workspace_mib", 64)
    if type(workspace) is not int or not 4 <= workspace <= 1024:
        raise ValueError("Invalid FreeVideo Relay workspace")
    geometry(**value["geometry"])
    length = value["embeddings"]["shape"][1]
    for event in value["events"]:
        if not (type(event["text_key_start"]) is int and type(event["text_key_end"]) is int
                and 0 <= event["text_key_start"] < event["text_key_end"] <= length):
            raise ValueError("FreeVideo Relay event is outside authoritative text keys")
        if any(type(event[k]) not in (int, float) or not math.isfinite(event[k]) for k in ("midpoint", "window", "sigma")) or event["window"] < 0 or event["sigma"] <= 0:
            raise ValueError("FreeVideo Relay event has invalid timing")
    return value


def condition_identity(conditioning, canvas):
    import torch
    from .runtime import conditioning_transport
    tensors, meta = conditioning_transport(conditioning, canvas)
    rows = {}
    for name, value in tensors.items():
        rows[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
            sha256=hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
    return hashlib.sha256(canonical(dict(tensors=rows, transport=meta)).encode()).hexdigest()


def transport_effects(model, conditioning, canvas):
    eav = validate_eav(json.loads(model.eav)) if model.eav is not None else None
    supplied = conditioning[0][1].get(RELAY_KEY)
    if (supplied is None) != (model.relay is None):
        raise ValueError("Connect the paired FreeVideo Relay MODEL and CONDITIONING")
    binding = validate_binding(json.loads(model.relay)) if model.relay else None
    if binding is not None:
        if supplied != binding or binding["geometry"] != canvas or tensor_record(conditioning[0][0]) != binding["embeddings"]:
            raise ValueError("FreeVideo Relay conditioning/geometry differs from its bound branch")
        if condition_identity(conditioning, canvas) != binding.get("conditions_sha256"):
            raise ValueError("FreeVideo Relay reference/keyframe/conditioning identity changed")
        tags = conditioning[0][1]["minimax_token_tags"].detach().cpu().contiguous()
        if hashlib.sha256(tags.view(__import__("torch").uint8).numpy().tobytes()).hexdigest() != binding["tags_sha256"]:
            raise ValueError("FreeVideo Relay token tags changed")
    return dict(eav=eav, relay=binding)


def build_relay_conditioning(freevideo_model, prompt_relay_plan, execution_mode="report_only", query_chunk_rows=256, relay_workspace_mib=64, **kwargs):
    from ..conditioning import build_conditioning
    from ..prompt_relay_advanced import _validate_plan, build_prompt_relay_binding
    import torch
    if not isinstance(freevideo_model, FreeVideoModel) or freevideo_model.relay is not None:
        raise ValueError("Use an independent unbound FreeVideo model branch for each Relay plan")
    plan = _validate_plan(prompt_relay_plan)
    if kwargs.get("audio_mode", "native") not in ("native", "reference_only"):
        raise ValueError("FreeVideo Relay supports native/reference-only generation, not Core lock/remix sampling")
    if kwargs.get("audio_mode") == "reference_only" and not kwargs.get("add_source_as_reference", True):
        raise ValueError("reference_only requires add_source_as_reference=true")
    kwargs["prompt"], kwargs["length"] = plan["compiled_prompt"], plan["frame_count"]
    result = build_conditioning(**kwargs, return_details=True)
    conditioning, latent, output_audio, text, media, report, details = result
    native = build_prompt_relay_binding(kwargs["clip"], plan, text, conditioning, details["tokens"])
    tags = conditioning[0][1]["minimax_token_tags"].detach().cpu().contiguous()
    binding = dict(schema="t8-freevideo-relay-v1", mode=execution_mode, query_route=native["query_route"],
        query_chunk_rows=query_chunk_rows, workspace_mib=relay_workspace_mib, events=native["events"], native_binding=native,
        geometry=geometry(kwargs["width"], kwargs["height"], details["frame_count"]),
        embeddings=tensor_record(conditioning[0][0]), tags_sha256=hashlib.sha256(tags.view(torch.uint8).numpy().tobytes()).hexdigest())
    binding["conditions_sha256"] = condition_identity(conditioning, binding["geometry"])
    binding["sha256"] = hashlib.sha256(canonical(binding).encode()).hexdigest()
    validate_binding(binding)
    paired = [[conditioning[0][0], dict(conditioning[0][1], **{RELAY_KEY: binding})]]
    model = replace(freevideo_model, relay=canonical(binding))
    # Validate native task/reference shapes before an engine can be dispatched.
    from .runtime import conditioning_transport
    conditioning_transport(paired, binding["geometry"])
    return model, paired, latent, output_audio, text, media, canonical(dict(binding_sha256=binding["sha256"],
        mode=execution_mode, native_report=report, events=len(binding["events"]),
        linear_profile="beta_weighted_text_seed_exp_v1", quality_approved=False))
