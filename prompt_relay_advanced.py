from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from collections.abc import Mapping

import torch

import comfy.conds
import comfy.patcher_extension
from comfy.ldm.modules import attention as attention_module
from comfy.ldm.minimax.model import Attention, PackedLayout
from comfy.model_base import MiniMaxH3 as MiniMaxH3BaseModel
from comfy.text_encoders.minimax import MiniMaxH3Tokenizer

import node_helpers

from .conditioning import build_conditioning, build_packed_layout
from .core import FPS, align_frame_count, nested_av_parts


PROMPT_RELAY_PLAN_TYPE = "H3_T8_PROMPT_RELAY_PLAN"
PROMPT_RELAY_PLAN_SCHEMA = 1
PROMPT_RELAY_PATCH_VERSION = 4
PROMPT_RELAY_BINDING_KEY = "minimax_prompt_relay_binding"
PROMPT_RELAY_PAYLOAD_KEY = "t8_prompt_relay_binding_hash"
PROMPT_RELAY_RUNTIME_KEY = "t8_prompt_relay_runtime"
PROMPT_RELAY_WRAPPER_KEY = "t8_prompt_relay_v1"

MATH_PROFILES = ("paper_v1", "legacy_repo_compat")
TIMING_MODES = ("auto_equal", "frames", "seconds", "percent")
EXECUTION_MODES = ("report_only", "apply_exp")
APPLY_EXP_TASKS = ("t2va", "i2va", "fl2va", "l2va", "ref2va", "hybrid")
PROMPT_RELAY_QUERY_ROUTES = ("video_only_paper", "joint_av_exp")

ATTENTION_FORWARD_SHA256S = {
    "4e8888f72ea5ccf68fb5ce5b1178ab0ddea66ca61137fcf01df2308ef27bf0be",
}
PACKED_LAYOUT_SHA256S = {
    "1124904e8835c6db068e61e304490d93784e6a8da6ca6b38afd93975611b3af4",
}
TOKENIZER_SHA256S = {
    "c05a3608337ea95d6703909a055a8c20b88a3b38d08e706e142daf5d2ff96b20",
}
EXTRA_CONDS_SHA256S = {
    "e43a26358405187d5a9556c158843d9ffe150ac52591d1034f3cce422e565974",
}

_RANGE_RE = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(?:-|:|–|—)\s*"
    r"([0-9]+(?:\.[0-9]+)?)\s*$"
)


def _canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _layout_contract(layout) -> dict:
    if layout is None or not hasattr(layout, "segments") or not hasattr(layout, "position_ids"):
        raise RuntimeError("Prompt Relay requires the native H3 PackedLayout payload")
    position_ids = layout.position_ids.detach().to(device="cpu").contiguous()
    if position_ids.ndim != 2 or int(position_ids.shape[1]) != 3:
        raise RuntimeError("Prompt Relay received an invalid H3 PackedLayout position grid")
    contract = {
        "schema": 1,
        "signature": [int(value) for value in layout.signature],
        "seq_len": int(layout.seq_len),
        "segments": [
            [int(start), int(end), str(kind)]
            for start, end, kind in layout.segments
        ],
        "position_shape": [int(value) for value in position_ids.shape],
        "position_dtype": str(position_ids.dtype),
        "position_sha256": hashlib.sha256(position_ids.numpy().tobytes()).hexdigest(),
    }
    contract["contract_hash"] = _sha256_json(contract)
    return contract


def _bind_layout_contract(
    binding: Mapping,
    layout,
    *,
    resolved_task: str,
    keyframes,
    refs,
) -> dict:
    bound = dict(binding)
    bound.pop("binding_hash", None)
    bound.update(
        {
            "task": str(resolved_task).lower(),
            "keyframe_count": len(keyframes or ()),
            "reference_block_count": len(refs or ()),
            "layout_contract": _layout_contract(layout),
        }
    )
    bound["binding_hash"] = _sha256_json(bound)
    return bound


def _source_sha256(function) -> str:
    function = getattr(function, "__func__", function)
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def _local_prompt_lines(local_prompts: str) -> list[str]:
    lines = []
    for raw in str(local_prompts).replace("|", "\n").splitlines():
        prompt = raw.strip()
        if prompt:
            lines.append(prompt)
    if len(lines) > 32:
        raise ValueError("Prompt Relay currently supports at most 32 local events")
    return lines


def _parse_range(raw: str) -> tuple[float, float]:
    match = _RANGE_RE.match(raw)
    if match is None:
        raise ValueError(
            f"Invalid Prompt Relay range {raw!r}; use start-end, one range per line"
        )
    start, end = float(match.group(1)), float(match.group(2))
    if end < start:
        raise ValueError(f"Prompt Relay range ends before it starts: {raw!r}")
    return start, end


def _explicit_ranges(
    timing_mode: str,
    time_ranges: str,
    count: int,
    frame_count: int,
) -> list[tuple[int, int]]:
    raw_ranges = [line.strip() for line in str(time_ranges).splitlines() if line.strip()]
    if len(raw_ranges) != count:
        raise ValueError(
            "Prompt Relay needs exactly one time range per local prompt "
            f"({count} prompts, {len(raw_ranges)} ranges)"
        )
    ranges = []
    for raw in raw_ranges:
        start, end = _parse_range(raw)
        if timing_mode == "frames":
            if not start.is_integer() or not end.is_integer():
                raise ValueError(
                    "Prompt Relay frame ranges require integer frame indices"
                )
            start_frame = int(round(start))
            end_frame = int(round(end)) + 1  # UI frame ranges are inclusive.
        elif timing_mode == "seconds":
            start_frame = int(round(start * FPS))
            end_frame = int(round(end * FPS))
        elif timing_mode == "percent":
            if start < 0.0 or end < 0.0 or start > 100.0 or end > 100.0:
                raise ValueError("Prompt Relay percent ranges use 0..100, not 0..1")
            start_frame = int(round(frame_count * start / 100.0))
            end_frame = int(round(frame_count * end / 100.0))
        else:
            raise ValueError(f"Unknown Prompt Relay timing mode {timing_mode!r}")
        ranges.append((start_frame, end_frame))
    return ranges


def _auto_equal_ranges(count: int, frame_count: int) -> list[tuple[int, int]]:
    bounds = [round(i * frame_count / count) for i in range(count + 1)]
    bounds[0], bounds[-1] = 0, frame_count
    return [(bounds[i], bounds[i + 1]) for i in range(count)]


def _validate_ranges(
    ranges: list[tuple[int, int]],
    frame_count: int,
    allow_gaps: bool,
    allow_overlaps: bool,
) -> None:
    for index, (start, end) in enumerate(ranges, 1):
        if start < 0 or end > frame_count or end <= start:
            raise ValueError(
                f"Prompt Relay event {index} resolves to invalid frame interval "
                f"[{start}, {end}) for {frame_count} frames"
            )
        if end - start < 5:
            raise ValueError(
                f"Prompt Relay event {index} is only {end - start} frames; "
                "use at least 5 frames per local event"
            )
    if any(
        ranges[index][0] < ranges[index - 1][0]
        for index in range(1, len(ranges))
    ):
        raise ValueError("Prompt Relay ranges must be listed in chronological order")
    ordered = sorted(ranges)
    if not allow_gaps and (ordered[0][0] != 0 or ordered[-1][1] != frame_count):
        raise ValueError("Prompt Relay ranges must cover the complete target when gaps are disabled")
    previous_end = ordered[0][1]
    for start, end in ordered[1:]:
        if not allow_overlaps and start < previous_end:
            raise ValueError("Prompt Relay ranges overlap while allow_overlaps is disabled")
        if not allow_gaps and start > previous_end:
            raise ValueError("Prompt Relay ranges contain a gap while allow_gaps is disabled")
        previous_end = max(previous_end, end)


def _paper_parameters(start_frame: int, end_frame: int, epsilon: float) -> dict:
    start_coord = (5.0 / 3.0) * start_frame
    end_coord = (5.0 / 3.0) * (end_frame - 1)
    midpoint = (start_coord + end_coord) / 2.0
    half_span = (end_coord - start_coord) / 2.0
    window = max(half_span - 2.0, 0.0)
    sigma = (half_span - window) / math.sqrt(2.0 * math.log(1.0 / epsilon))
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError("Prompt Relay event is too short for a finite paper_v1 sigma")
    endpoint_cost = ((half_span - window) ** 2) / (2.0 * sigma * sigma)
    return {
        "start_coord": start_coord,
        "end_coord": end_coord,
        "midpoint": midpoint,
        "half_span": half_span,
        "window": window,
        "sigma": sigma,
        "endpoint_weight": math.exp(-endpoint_cost),
    }


def _legacy_parameters(start_frame: int, end_frame: int, epsilon: float) -> dict:
    result = _paper_parameters(start_frame, end_frame, epsilon)
    result["sigma"] = 1.0 / math.log(1.0 / epsilon)
    endpoint_cost = (
        (result["half_span"] - result["window"]) ** 2
        / (2.0 * result["sigma"] * result["sigma"])
    )
    result["endpoint_weight"] = math.exp(-endpoint_cost)
    return result


def build_prompt_relay_plan(
    global_prompt: str,
    local_prompts: str,
    length: int,
    timing_mode: str,
    time_ranges: str,
    math_profile: str,
    epsilon: float,
    allow_gaps: bool,
    allow_overlaps: bool,
) -> tuple[dict, str, int, str, str]:
    global_prompt = str(global_prompt).strip()
    if not global_prompt:
        raise ValueError("Prompt Relay global prompt cannot be empty")
    prompts = _local_prompt_lines(local_prompts)
    frame_count = align_frame_count(length)
    if timing_mode not in TIMING_MODES:
        raise ValueError(f"Unknown Prompt Relay timing mode {timing_mode!r}")
    if math_profile not in MATH_PROFILES:
        raise ValueError(f"Unknown Prompt Relay math profile {math_profile!r}")
    if not 0.0 < float(epsilon) < 1.0:
        raise ValueError("Prompt Relay epsilon must be strictly between 0 and 1")

    if not prompts:
        if str(time_ranges).strip():
            raise ValueError(
                "Prompt Relay global-only bypass cannot include time_ranges; clear the "
                "ranges or connect an all-disabled Event chain"
            )
        ranges = []
    elif timing_mode == "auto_equal":
        ranges = _auto_equal_ranges(len(prompts), frame_count)
    else:
        ranges = _explicit_ranges(timing_mode, time_ranges, len(prompts), frame_count)
    if ranges:
        _validate_ranges(ranges, frame_count, allow_gaps, allow_overlaps)

    compiled_prompt = f"Global scene: {global_prompt}"
    events = []
    for index, (local_prompt, (start, end)) in enumerate(zip(prompts, ranges), 1):
        compiled_prompt += "\n"
        char_start = len(compiled_prompt)
        compiled_prompt += f"Event {index}: {local_prompt}"
        char_end = len(compiled_prompt)
        parameters = (
            _paper_parameters(start, end, float(epsilon))
            if math_profile == "paper_v1"
            else _legacy_parameters(start, end, float(epsilon))
        )
        events.append(
            {
                "event_index": index,
                "local_prompt": local_prompt,
                "start_frame": start,
                "end_frame_exclusive": end,
                "start_seconds": start / FPS,
                "end_seconds": end / FPS,
                "prompt_char_start": char_start,
                "prompt_char_end": char_end,
                **parameters,
            }
        )

    plan = {
        "type": PROMPT_RELAY_PLAN_TYPE,
        "schema": PROMPT_RELAY_PLAN_SCHEMA,
        "paper": "Prompt Relay, arXiv:2604.10030v1",
        "math_profile": math_profile,
        "epsilon": float(epsilon),
        "timing_mode": timing_mode,
        "frame_count": frame_count,
        "fps": FPS,
        "global_prompt": global_prompt,
        "compiled_prompt": compiled_prompt,
        "allow_gaps": bool(allow_gaps),
        "allow_overlaps": bool(allow_overlaps),
        "events": events,
    }
    plan["plan_hash"] = _sha256_json(plan)
    timeline = {
        "frame_count": frame_count,
        "fps": FPS,
        "events": [
            {
                "event": event["event_index"],
                "frames": [event["start_frame"], event["end_frame_exclusive"] - 1],
                "seconds": [event["start_seconds"], event["end_seconds"]],
                "prompt": event["local_prompt"],
            }
            for event in events
        ],
    }
    report = {
        "status": "plan_ready" if events else "plan_bypass_no_events",
        "experimental": True,
        "plan_hash": plan["plan_hash"],
        "math_profile": math_profile,
        "paper_equation": math_profile == "paper_v1",
        "epsilon": float(epsilon),
        "event_count": len(events),
        "frame_count": frame_count,
        "endpoint_weights": [event["endpoint_weight"] for event in events],
        "notes": [
            "paper_v1 is the default and derives sigma from epsilon per event",
            "legacy_repo_compat is available only for comparison and is not the paper equation",
            "frame ranges require integers and use inclusive end frames; second/percent ranges use end boundaries",
            "explicit ranges must be listed in chronological start order",
            "zero active events form a global-only no-patch bypass plan",
            "apply_exp with one local event is an automatic no-patch passthrough; normally merge it into global_prompt",
        ],
    }
    return plan, compiled_prompt, frame_count, json.dumps(timeline, ensure_ascii=False, indent=2), json.dumps(report, ensure_ascii=False, indent=2)


def _validate_plan(plan: Mapping) -> dict:
    if not isinstance(plan, Mapping):
        raise ValueError("Prompt Relay plan is not a mapping")
    plan = dict(plan)
    if plan.get("type") != PROMPT_RELAY_PLAN_TYPE or int(plan.get("schema", 0)) != PROMPT_RELAY_PLAN_SCHEMA:
        raise ValueError("Unsupported Prompt Relay plan type/schema")
    claimed = plan.pop("plan_hash", None)
    actual = _sha256_json(plan)
    plan["plan_hash"] = claimed
    if claimed != actual:
        raise ValueError("Prompt Relay plan hash mismatch; rebuild it with the Plan node")
    return plan


def configure_prompt_relay_query_route(
    prompt_relay_plan: Mapping,
    query_route: str,
) -> tuple[dict, str]:
    plan = _validate_plan(prompt_relay_plan)
    query_route = str(query_route)
    if query_route not in PROMPT_RELAY_QUERY_ROUTES:
        raise ValueError(f"Unknown Prompt Relay query route {query_route!r}")
    routed = dict(plan)
    routed.pop("plan_hash", None)
    routed["query_route"] = query_route
    routed["query_route_schema"] = 1
    routed["plan_hash"] = _sha256_json(routed)
    report = {
        "status": "query_route_configured",
        "experimental": query_route == "joint_av_exp",
        "query_route": query_route,
        "plan_hash": routed["plan_hash"],
        "paper_scope": (
            "published_video_query_route"
            if query_route == "video_only_paper"
            else "experimental_h3_joint_audio_video_extension"
        ),
        "notes": [
            (
                "video_only_paper keeps the published Prompt Relay scope: only target "
                "video queries receive local-text timing bias"
            ),
            (
                "joint_av_exp maps the native H3 target-audio packed time grid onto the "
                "same event coordinates; the Prompt Relay paper did not validate audio"
            ),
        ],
    }
    return routed, json.dumps(report, ensure_ascii=False, indent=2)


def _authoritative_entries(tokens: Mapping) -> list:
    if not isinstance(tokens, Mapping) or set(tokens) != {"qwen3vl_32b"}:
        raise RuntimeError("Prompt Relay requires the native MiniMax H3 qwen3vl_32b tokenizer output")
    batches = tokens["qwen3vl_32b"]
    if len(batches) != 1:
        raise RuntimeError("Prompt Relay requires exactly one native H3 token batch")
    return list(batches[0])


def _inner_tokenizer(clip):
    outer = getattr(clip, "tokenizer", None)
    inner = getattr(outer, "qwen3vl_32b", None)
    hf = getattr(inner, "tokenizer", None)
    if inner is None or hf is None or not hasattr(hf, "byte_decoder"):
        raise RuntimeError("Prompt Relay requires ComfyUI's native MiniMax H3 Qwen tokenizer")
    return inner, hf


def _prompt_token_ids(clip, prompt: str) -> tuple[list[int], object]:
    inner, hf = _inner_tokenizer(clip)
    batches = inner.tokenize_with_weights(
        prompt,
        return_word_ids=False,
        disable_weights=True,
    )
    if len(batches) != 1:
        raise RuntimeError("Prompt Relay prompt exceeds the supported single token batch")
    ids = [entry[0] for entry in batches[0]]
    if not ids or not all(isinstance(token, int) for token in ids):
        raise RuntimeError("Prompt Relay prompt tokenization produced a non-text token")
    return ids, hf


def _token_byte_offsets(prompt: str, token_ids: list[int], hf) -> list[tuple[int, int]]:
    offsets = []
    decoded = bytearray()
    byte_decoder = hf.byte_decoder
    for token_id in token_ids:
        token = hf.convert_ids_to_tokens(int(token_id))
        try:
            piece = bytes(byte_decoder[character] for character in token)
        except KeyError as error:
            raise RuntimeError(
                f"Prompt Relay could not reconstruct byte offsets for token {token!r}"
            ) from error
        start = len(decoded)
        decoded.extend(piece)
        offsets.append((start, len(decoded)))
    expected = prompt.encode("utf-8")
    if bytes(decoded) != expected:
        raise RuntimeError(
            "Prompt Relay tokenizer byte reconstruction does not match the compiled prompt"
        )
    return offsets


def _span_to_tokens(
    prompt: str,
    offsets: list[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    byte_start = len(prompt[:char_start].encode("utf-8"))
    byte_end = len(prompt[:char_end].encode("utf-8"))
    selected = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > byte_start and start < byte_end
    ]
    if not selected:
        raise RuntimeError("Prompt Relay local prompt resolved to an empty token span")
    return selected[0], selected[-1] + 1


def build_prompt_relay_binding(
    clip,
    plan: Mapping,
    conditioned_prompt: str,
    conditioning,
    tokens: Mapping,
) -> dict:
    plan = _validate_plan(plan)
    if conditioned_prompt != plan["compiled_prompt"]:
        raise RuntimeError(
            "Prompt Relay compiled prompt was changed during media-tag preprocessing; "
            "use canonical connected tags such as <Picture 1>/<Video 1>/<Audio 1> "
            "inside the Plan text"
        )
    entries = _authoritative_entries(tokens)
    prompt_ids, hf = _prompt_token_ids(clip, conditioned_prompt)
    if len(entries) < len(prompt_ids):
        raise RuntimeError("Prompt Relay authoritative token stream is shorter than the prompt")
    entry_tail = [entry[0] for entry in entries[-len(prompt_ids):]]
    if entry_tail != prompt_ids:
        raise RuntimeError(
            "Prompt Relay prompt tokens are not the exact tail of the authoritative H3 token stream"
        )
    if not conditioning or not conditioning[0] or not torch.is_tensor(conditioning[0][0]):
        raise RuntimeError("Prompt Relay did not receive a native ComfyUI CONDITIONING tensor")
    text_len = int(conditioning[0][0].shape[1])
    prompt_start = text_len - len(prompt_ids)
    if prompt_start < 0:
        raise RuntimeError("Prompt Relay prompt token count exceeds final conditioning length")
    metadata = conditioning[0][1]
    tags = metadata.get("minimax_token_tags") if isinstance(metadata, Mapping) else None
    if not torch.is_tensor(tags) or int(tags.numel()) != text_len:
        raise RuntimeError("Prompt Relay requires native H3 minimax_token_tags provenance")
    if not bool((tags.reshape(-1)[prompt_start:] == 1).all()):
        raise RuntimeError("Prompt Relay local prompt overlaps a visual presentation token")

    offsets = _token_byte_offsets(conditioned_prompt, prompt_ids, hf)
    bound_events = []
    for event in plan["events"]:
        local_start, local_end = _span_to_tokens(
            conditioned_prompt,
            offsets,
            int(event["prompt_char_start"]),
            int(event["prompt_char_end"]),
        )
        bound_events.append(
            {
                "event_index": int(event["event_index"]),
                "text_key_start": prompt_start + local_start,
                "text_key_end": prompt_start + local_end,
                "midpoint": float(event["midpoint"]),
                "window": float(event["window"]),
                "sigma": float(event["sigma"]),
            }
        )

    binding = {
        "schema": PROMPT_RELAY_PATCH_VERSION,
        "plan_hash": plan["plan_hash"],
        "compiled_prompt_sha256": hashlib.sha256(conditioned_prompt.encode("utf-8")).hexdigest(),
        "text_len": text_len,
        "prompt_token_count": len(prompt_ids),
        "prompt_token_sha256": hashlib.sha256(
            _canonical_json(prompt_ids).encode("utf-8")
        ).hexdigest(),
        "events": bound_events,
        "query_route": str(plan.get("query_route", "video_only_paper")),
    }
    if binding["query_route"] not in PROMPT_RELAY_QUERY_ROUTES:
        raise RuntimeError(
            f"Prompt Relay plan contains unknown query route {binding['query_route']!r}"
        )
    binding["binding_hash"] = _sha256_json(binding)
    return binding


def _assert_core_contract(
    model,
    *,
    allowed_live_extra_conds_patch_versions: tuple[int, ...] = (),
) -> dict:
    if not hasattr(model, "clone") or not hasattr(model, "add_wrapper_with_key"):
        raise ValueError("Prompt Relay requires a ComfyUI MODEL patcher")
    base = getattr(model, "model", None)
    if not isinstance(base, MiniMaxH3BaseModel):
        diffusion_name = type(getattr(base, "diffusion_model", None)).__name__
        if diffusion_name != "MiniMaxH3Model":
            raise ValueError("Prompt Relay currently requires a native MiniMax H3 MODEL")

    class_extra_conds = getattr(type(base), "extra_conds", None)
    if class_extra_conds is None:
        raise RuntimeError("Prompt Relay could not locate native MiniMax H3 extra_conds")
    live_extra_conds = getattr(base, "__dict__", {}).get("extra_conds")
    if live_extra_conds is not None:
        live_function = getattr(live_extra_conds, "__func__", live_extra_conds)
        live_version = getattr(live_function, "_t8_long_video_patch_version", None)
        if live_version not in set(allowed_live_extra_conds_patch_versions):
            raise RuntimeError(
                "Prompt Relay detected an active instance-level extra_conds patch and "
                "refused to treat it as native H3"
            )

    hashes = {
        "attention_forward": _source_sha256(Attention.forward),
        "packed_layout": _source_sha256(PackedLayout.__init__),
        "tokenizer": _source_sha256(MiniMaxH3Tokenizer.tokenize_with_weights),
        "extra_conds": _source_sha256(class_extra_conds),
    }
    expected = {
        "attention_forward": ATTENTION_FORWARD_SHA256S,
        "packed_layout": PACKED_LAYOUT_SHA256S,
        "tokenizer": TOKENIZER_SHA256S,
        "extra_conds": EXTRA_CONDS_SHA256S,
    }
    mismatches = [name for name, value in hashes.items() if value not in expected[name]]
    if mismatches:
        raise RuntimeError(
            "Prompt Relay has not validated this ComfyUI H3 core contract: "
            + ", ".join(f"{name}={hashes[name]}" for name in mismatches)
        )

    transformer = getattr(model, "model_options", {}).get("transformer_options", {})
    if "optimized_attention_override" in transformer:
        raise RuntimeError(
            "Prompt Relay cannot stack with an existing optimized_attention_override"
        )
    replacements = transformer.get("patches_replace", {})
    if isinstance(replacements, Mapping) and any(bool(value) for value in replacements.values()):
        raise RuntimeError("Prompt Relay cannot stack with block/attention replacements yet")
    wrappers = getattr(model, "wrappers", {})
    diffusion_wrappers = wrappers.get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {})
    if any(bool(value) for value in diffusion_wrappers.values()):
        raise RuntimeError("Prompt Relay cannot stack with an existing diffusion-model wrapper yet")
    if bool(getattr(model, "patches", {})):
        raise RuntimeError(
            "Prompt Relay input MODEL already has weight patches; bind Prompt Relay first, "
            "then apply LoRA downstream"
        )
    injections = getattr(model, "injections", {})
    active_injections = sorted(
        str(key)
        for key, value in injections.items()
        if bool(value)
    )
    if active_injections:
        raise RuntimeError(
            "Prompt Relay input MODEL already has runtime injections ("
            + ", ".join(active_injections)
            + "); bind Prompt Relay first, then apply LoRA downstream"
        )
    object_patches = getattr(model, "object_patches", {})
    conflicting_objects = sorted(
        key
        for key in object_patches
        if key in {"extra_conds", "diffusion_model._forward", "diffusion_model.forward"}
    )
    if conflicting_objects:
        raise RuntimeError(
            "Prompt Relay cannot stack with existing H3 object patches: "
            + ", ".join(conflicting_objects)
        )
    extra_function = getattr(class_extra_conds, "__func__", class_extra_conds)
    if getattr(extra_function, "__module__", None) != "comfy.model_base":
        raise RuntimeError("Prompt Relay cannot stack with an existing extra_conds object patch")
    return hashes


def _runtime_route(layout, binding: Mapping, device: torch.device) -> dict:
    expected_contract = binding.get("layout_contract")
    actual_contract = _layout_contract(layout)
    if not isinstance(expected_contract, Mapping) or dict(expected_contract) != actual_contract:
        raise RuntimeError(
            "Prompt Relay runtime PackedLayout differs from the layout bound by Conditioning"
        )
    text_segment = layout.segments[0]
    video_segments = [segment for segment in layout.segments if segment[2] == "video"]
    audio_segments = [segment for segment in layout.segments if segment[2] == "audio"]
    if (
        text_segment != (0, int(binding["text_len"]), "text")
        or len(video_segments) != 1
        or len(audio_segments) != 1
    ):
        raise RuntimeError("Prompt Relay runtime layout does not match its authoritative text binding")
    video_start, video_end, _ = video_segments[0]
    audio_start, audio_end, _ = audio_segments[0]
    if video_end != int(layout.seq_len) or audio_end != video_start:
        raise RuntimeError(
            "Prompt Relay requires target audio/video to be the final two H3 packed segments"
        )
    video_query_times = layout.position_ids[video_start:video_end, 0].to(
        device=device,
        dtype=torch.float32,
    )
    video_query_times = video_query_times - video_query_times[0]
    query_route = str(binding.get("query_route", "video_only_paper"))
    if query_route == "video_only_paper":
        query_segments = (
            {
                "kind": "video",
                "start": int(video_start),
                "end": int(video_end),
                "query_times": video_query_times,
            },
        )
    elif query_route == "joint_av_exp":
        audio_query_times = layout.position_ids[audio_start:audio_end, 0].to(
            device=device,
            dtype=torch.float32,
        )
        audio_query_times = audio_query_times - audio_query_times[0]
        query_segments = (
            {
                "kind": "audio",
                "start": int(audio_start),
                "end": int(audio_end),
                "query_times": audio_query_times,
            },
            {
                "kind": "video",
                "start": int(video_start),
                "end": int(video_end),
                "query_times": video_query_times,
            },
        )
    else:
        raise RuntimeError(f"Unknown Prompt Relay runtime query route {query_route!r}")
    return {
        "binding_hash": binding["binding_hash"],
        "query_route": query_route,
        "seq_len": int(layout.seq_len),
        "audio_start": int(audio_start),
        "audio_end": int(audio_end),
        "video_start": int(video_start),
        "video_end": int(video_end),
        "query_segments": query_segments,
        "events": tuple(dict(event) for event in binding["events"]),
    }


def prompt_relay_penalty(query_times: torch.Tensor, event: Mapping) -> torch.Tensor:
    distance = (query_times.float() - float(event["midpoint"])).abs()
    outside = (distance - float(event["window"])).clamp_min(0.0)
    sigma = float(event["sigma"])
    return outside.square().div_(2.0 * sigma * sigma)


def make_prompt_relay_bias(
    query_times: torch.Tensor,
    seq_len: int,
    events,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    bias = torch.zeros(
        (int(query_times.numel()), int(seq_len)),
        device=query_times.device,
        dtype=dtype,
    )
    for event in events:
        start, end = int(event["text_key_start"]), int(event["text_key_end"])
        if not 0 <= start < end <= seq_len:
            raise RuntimeError("Prompt Relay text-key span is outside the packed sequence")
        penalty = prompt_relay_penalty(query_times, event).to(dtype=dtype)
        bias[:, start:end] = -penalty[:, None]
    return bias


def route_prompt_relay_attention(
    q,
    k,
    v,
    heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    skip_output_reshape=False,
    transformer_options=None,
    *,
    query_chunk_rows: int,
    **kwargs,
):
    transformer_options = transformer_options or {}
    route = transformer_options.get(PROMPT_RELAY_RUNTIME_KEY)
    delegate_kwargs = dict(kwargs)
    delegate_kwargs["_inside_attn_wrapper"] = True
    if route is None or q.shape[-2] != int(route["seq_len"]):
        return attention_module.optimized_attention(
            q,
            k,
            v,
            heads,
            mask=mask,
            attn_precision=attn_precision,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            transformer_options=transformer_options,
            **delegate_kwargs,
        )
    if mask is not None:
        raise RuntimeError("Prompt Relay does not stack with a pre-existing attention mask")
    if not skip_reshape or skip_output_reshape:
        raise RuntimeError("Prompt Relay received an unsupported H3 attention tensor layout")
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[1] != heads:
        raise RuntimeError("Prompt Relay currently requires H3 batch size 1 packed attention")

    outputs = []
    cursor = 0
    for segment in route["query_segments"]:
        segment_start = int(segment["start"])
        segment_end = int(segment["end"])
        query_times = segment["query_times"]
        if not cursor <= segment_start < segment_end <= int(route["seq_len"]):
            raise RuntimeError("Prompt Relay routed query segments are invalid or overlap")
        if int(query_times.numel()) != segment_end - segment_start:
            raise RuntimeError("Prompt Relay routed query time count does not match its segment")
        if cursor < segment_start:
            outputs.append(
                attention_module.optimized_attention(
                    q[:, :, cursor:segment_start],
                    k,
                    v,
                    heads,
                    mask=None,
                    attn_precision=attn_precision,
                    skip_reshape=True,
                    skip_output_reshape=False,
                    transformer_options=transformer_options,
                    **delegate_kwargs,
                )
            )
        for start in range(0, int(query_times.numel()), int(query_chunk_rows)):
            end = min(start + int(query_chunk_rows), int(query_times.numel()))
            bias = make_prompt_relay_bias(
                query_times[start:end],
                int(route["seq_len"]),
                route["events"],
                dtype=q.dtype,
            )
            outputs.append(
                attention_module.attention_pytorch(
                    q[:, :, segment_start + start:segment_start + end],
                    k,
                    v,
                    heads,
                    mask=bias,
                    attn_precision=attn_precision,
                    skip_reshape=True,
                    skip_output_reshape=False,
                    transformer_options=transformer_options,
                    **delegate_kwargs,
                )
            )
        cursor = segment_end
    if cursor < int(route["seq_len"]):
        outputs.append(
            attention_module.optimized_attention(
                q[:, :, cursor:int(route["seq_len"])],
                k,
                v,
                heads,
                mask=None,
                attn_precision=attn_precision,
                skip_reshape=True,
                skip_output_reshape=False,
                transformer_options=transformer_options,
                **delegate_kwargs,
            )
        )
    return torch.cat(outputs, dim=1)


def _install_prompt_relay_model(
    model,
    binding: Mapping,
    query_chunk_rows: int,
    core_hashes: Mapping,
):
    if not 32 <= int(query_chunk_rows) <= 2048:
        raise ValueError("Prompt Relay query_chunk_rows must be between 32 and 2048")
    expected_hash = str(binding["binding_hash"])
    patched = model.clone()

    def _diffusion_wrapper(
        executor,
        x,
        timestep,
        context,
        transformer_options=None,
        **kwargs,
    ):
        transformer_options = transformer_options if transformer_options is not None else {}
        if len(executor.wrappers) != 1:
            raise RuntimeError(
                "Prompt Relay detected another diffusion-model wrapper added after binding"
            )
        active_override = transformer_options.get("optimized_attention_override")
        if getattr(active_override, "_t8_prompt_relay_binding_hash", None) != expected_hash:
            raise RuntimeError(
                "Prompt Relay attention override was replaced after the MODEL was bound"
            )
        replacements = transformer_options.get("patches_replace", {})
        if isinstance(replacements, Mapping) and any(
            bool(value) for value in replacements.values()
        ):
            raise RuntimeError(
                "Prompt Relay detected a runtime block/attention replacement and refused to stack"
            )
        supplied_hash = kwargs.pop(PROMPT_RELAY_PAYLOAD_KEY, None)
        if supplied_hash != expected_hash:
            raise RuntimeError(
                "Prompt Relay MODEL and CONDITIONING are not the paired outputs of the same node"
            )
        payload = kwargs.get("minimax_payload")
        if not isinstance(payload, Mapping):
            raise RuntimeError("Prompt Relay could not find the native H3 minimax_payload")
        if PROMPT_RELAY_RUNTIME_KEY in transformer_options:
            raise RuntimeError("Nested Prompt Relay runtime state was refused")
        route = _runtime_route(payload.get("layout"), binding, x[0].device)
        transformer_options[PROMPT_RELAY_RUNTIME_KEY] = route
        try:
            return executor(
                x,
                timestep,
                context,
                transformer_options,
                **kwargs,
            )
        finally:
            transformer_options.pop(PROMPT_RELAY_RUNTIME_KEY, None)

    def _attention_router(*args, **kwargs):
        return route_prompt_relay_attention(
            *args,
            query_chunk_rows=int(query_chunk_rows),
            **kwargs,
        )

    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        PROMPT_RELAY_WRAPPER_KEY,
        _diffusion_wrapper,
    )
    patched.set_model_optimized_attention(_attention_router)
    installed_override = patched.model_options["transformer_options"][
        "optimized_attention_override"
    ]
    installed_override._t8_prompt_relay_binding_hash = expected_hash
    if hasattr(patched, "set_attachments"):
        patched.set_attachments(
            PROMPT_RELAY_WRAPPER_KEY,
            {
                "patch_version": PROMPT_RELAY_PATCH_VERSION,
                "binding_hash": expected_hash,
                "query_chunk_rows": int(query_chunk_rows),
                "core_hashes": core_hashes,
            },
        )
    return patched, core_hashes


def patch_prompt_relay_model(model, binding: Mapping, query_chunk_rows: int):
    core_hashes = _assert_core_contract(model)
    return _install_prompt_relay_model(
        model,
        binding,
        query_chunk_rows,
        core_hashes,
    )


def _attach_binding_model_cond(conditioning, binding_hash: str):
    output = []
    for cross_attn, metadata in conditioning:
        updated = dict(metadata)
        model_conds = dict(updated.get("model_conds", {}))
        if PROMPT_RELAY_PAYLOAD_KEY in model_conds:
            raise RuntimeError("Prompt Relay conditioning already contains a runtime binding")
        model_conds[PROMPT_RELAY_PAYLOAD_KEY] = comfy.conds.CONDConstant(
            str(binding_hash)
        )
        updated["model_conds"] = model_conds
        output.append([cross_attn, updated])
    return output


def build_prompt_relay_conditioning(
    model,
    clip,
    video_vae,
    audio_vae,
    prompt_relay_plan,
    width,
    height,
    task_type,
    audio_mode,
    audio_denoise_strength,
    add_source_as_reference,
    prompt_primary_audio_ordinal,
    strict_prompt_tags,
    ref_image_size,
    reference_video_policy,
    execution_mode,
    query_chunk_rows,
    drive_audio=None,
    final_audio=None,
    first_frame=None,
    last_frame=None,
    ref_images=None,
    ref_videos=None,
    ref_video_audios=None,
    ref_audios=None,
):
    plan = _validate_plan(prompt_relay_plan)
    if execution_mode not in EXECUTION_MODES:
        raise ValueError(f"Unknown Prompt Relay execution mode {execution_mode!r}")
    result = build_conditioning(
        clip,
        video_vae,
        audio_vae,
        plan["compiled_prompt"],
        int(width),
        int(height),
        int(plan["frame_count"]),
        task_type,
        audio_mode,
        float(audio_denoise_strength),
        bool(add_source_as_reference),
        int(prompt_primary_audio_ordinal),
        bool(strict_prompt_tags),
        ref_image_size,
        reference_video_policy,
        drive_audio,
        final_audio,
        first_frame,
        last_frame,
        ref_images,
        ref_videos,
        ref_video_audios,
        ref_audios,
        return_details=True,
    )
    conditioning, latent, output_audio, conditioned_prompt, media_map, stable_report, details = result
    if details["audio_mode"] == "reference_only" and not bool(add_source_as_reference):
        raise ValueError(
            "Prompt Relay reference_only requires add_source_as_reference=true; "
            "otherwise drive_audio would not condition the target audio"
        )
    binding = build_prompt_relay_binding(
        clip,
        plan,
        conditioned_prompt,
        conditioning,
        details["tokens"],
    )
    query_route = str(binding["query_route"])
    if (
        execution_mode == "apply_exp"
        and len(binding["events"]) > 1
        and query_route == "joint_av_exp"
        and details["audio_mode"] == "lock_source"
    ):
        raise ValueError(
            "Prompt Relay joint_av_exp cannot be combined with lock_source: the target "
            "audio latent is locked, so use video_only_paper and mux_audio instead"
        )
    video, audio = nested_av_parts(latent)
    layout = build_packed_layout(
        binding["text_len"],
        int(video.shape[2]),
        int(video.shape[3]),
        int(video.shape[4]),
        int(audio.shape[-1]),
        keyframes=details["keyframes"],
        refs=details["refs"],
        frame_count=details["frame_count"],
    )
    binding = _bind_layout_contract(
        binding,
        layout,
        resolved_task=details["resolved_task"],
        keyframes=details["keyframes"],
        refs=details["refs"],
    )
    video_segment = next(segment for segment in layout.segments if segment[2] == "video")
    audio_segment = next(segment for segment in layout.segments if segment[2] == "audio")
    conditioning_rows = sum(
        end - start
        for start, end, kind in layout.segments
        if kind in {"cond", "cond_audio", "ref_img", "ref_audio"}
    )
    max_bias_bytes_bf16 = int(query_chunk_rows) * int(layout.seq_len) * 2

    patched_model = model
    core_hashes = {}
    status = "report_only"
    warnings = [
        "Prompt Relay for H3 is experimental; the paper validated Wan2.2 T2V, not H3 joint AV",
    ]
    if query_route == "joint_av_exp":
        warnings.append(
            "joint_av_exp directly routes both target-audio and target-video queries on "
            "the native packed time grid; this is an H3 extension, not a paper-validated mode"
        )
    else:
        warnings.append(
            "video_only_paper directly biases only target-video queries; audio can still "
            "change indirectly through shared H3 layers"
        )
    if execution_mode == "apply_exp":
        # The stable conditioning path deliberately normalizes task identifiers
        # to lowercase.  Keep the public UI values unchanged, but compare the
        # authoritative resolved value using the same normalization.
        resolved_task = str(details["resolved_task"]).lower()
        if resolved_task not in APPLY_EXP_TASKS:
            raise RuntimeError(
                "Prompt Relay apply_exp does not support task "
                f"{details['resolved_task']!r}"
            )
        if len(binding["events"]) <= 1:
            if binding["events"]:
                status = "passthrough_single_event"
                warnings.append(
                    "A single local event covers no competing event timeline, so Prompt Relay "
                    "left MODEL and stable CONDITIONING unpatched; normally merge that event "
                    "into global_prompt"
                )
            else:
                status = "passthrough_no_events"
                warnings.append(
                    "No local events are active, so Prompt Relay left MODEL and stable "
                    "CONDITIONING unpatched and used only global_prompt"
                )
        else:
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {PROMPT_RELAY_BINDING_KEY: binding},
            )
            conditioning = _attach_binding_model_cond(
                conditioning,
                binding["binding_hash"],
            )
            patched_model, core_hashes = patch_prompt_relay_model(
                model,
                binding,
                int(query_chunk_rows),
            )
            status = "applied_exp"

    if str(details["resolved_task"]).lower() != "t2va":
        warnings.append(
            "This task now has exact packed-layout routing support, but task-specific "
            "perceptual A/B validation is still required before any quality claim"
        )
    if details["keyframes"] or details["refs"]:
        warnings.append(
            "Keyframe/reference rows are bound into the exact layout contract and remain "
            "globally visible; Prompt Relay does not apply a time penalty to those rows"
        )

    audio_mode = details["audio_mode"]
    audio_mask_range = None
    noise_mask = latent.get("noise_mask") if isinstance(latent, Mapping) else None
    if getattr(noise_mask, "is_nested", False):
        mask_parts = tuple(noise_mask.unbind())
        if len(mask_parts) == 2:
            audio_mask = mask_parts[1]
            audio_mask_range = [
                float(audio_mask.detach().amin().cpu()),
                float(audio_mask.detach().amax().cpu()),
            ]

    if audio_mode == "lock_source":
        audio_policy = "source_latent_locked_by_zero_noise_mask"
        warnings.append(
            "lock_source keeps the encoded source audio latent at audio noise_mask=0; "
            "use mux_audio for the original waveform because AV VAE decode is not bit-exact"
        )
    elif audio_mode == "remix_source":
        audio_policy = "source_latent_jointly_remixed_at_requested_strength"
        if query_route == "joint_av_exp":
            warnings.append(
                "remix_source intentionally denoises the target audio latent and joint_av_exp "
                "directly applies local-event timing to its audio queries; listen for seams "
                "and semantic leakage"
            )
        else:
            warnings.append(
                "remix_source intentionally denoises the target audio latent; Prompt Relay does "
                "not directly bias audio queries, but shared H3 layers can still change sound"
            )
    elif audio_mode == "reference_only":
        audio_policy = "source_audio_reference_only_target_audio_regenerated"
        if query_route == "joint_av_exp":
            warnings.append(
                "reference_only keeps drive_audio in the global reference block, regenerates "
                "target audio, and applies local-event timing to target audio queries"
            )
        else:
            warnings.append(
                "reference_only keeps drive_audio in the global reference block and regenerates "
                "target audio; local event timing is applied only to target video queries"
            )
    else:
        audio_policy = "native_target_audio_regenerated"
        if query_route == "joint_av_exp":
            warnings.append(
                "native target audio is regenerated with direct local-event timing bias; "
                "the result requires listening and A/V timing review"
            )

    report = {
        "status": status,
        "experimental": True,
        "route": (
            "target_audio_and_video_queries_to_local_text_keys"
            if query_route == "joint_av_exp"
            else "target_video_queries_to_local_text_keys"
        ),
        "query_route": query_route,
        "task": details["resolved_task"],
        "audio_mode": details["audio_mode"],
        "audio_policy": audio_policy,
        "audio_direct_prompt_relay_bias": query_route == "joint_av_exp",
        "audio_noise_mask_range": audio_mask_range,
        "plan_hash": plan["plan_hash"],
        "binding_hash": binding["binding_hash"],
        "layout_contract_hash": binding["layout_contract"]["contract_hash"],
        "text_len": binding["text_len"],
        "prompt_token_count": binding["prompt_token_count"],
        "packed_seq_len": int(layout.seq_len),
        "target_video_rows": int(video_segment[1] - video_segment[0]),
        "target_audio_rows": int(audio_segment[1] - audio_segment[0]),
        "routed_query_rows": (
            int(audio_segment[1] - audio_segment[0])
            + int(video_segment[1] - video_segment[0])
            if query_route == "joint_av_exp"
            else int(video_segment[1] - video_segment[0])
        ),
        "condition_reference_rows": int(conditioning_rows),
        "keyframe_count": len(details["keyframes"]),
        "reference_block_count": len(details["refs"]),
        "packed_segments": binding["layout_contract"]["segments"],
        "query_chunk_rows": int(query_chunk_rows),
        "max_explicit_bias_bytes_bf16": max_bias_bytes_bf16,
        "dense_s_by_s_mask_created": False,
        "attention_patch_installed": status == "applied_exp",
        "core_hashes": core_hashes,
        "stable_conditioning_report": stable_report,
        "warnings": warnings,
    }
    return (
        patched_model,
        conditioning,
        latent,
        output_audio,
        conditioned_prompt,
        media_map,
        json.dumps(report, ensure_ascii=False, indent=2),
    )
