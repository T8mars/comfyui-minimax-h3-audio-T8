"""Native text re-encoding for explicit Chunk dialogue scopes.

No VAE, sampling, attention patch, global clock change or embedding slicing.
Capture is opt-in; the old Conditioning node's output/schema stays unchanged.
Live recipes are not serialized as portable CLIP/Bridge identities.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import node_helpers

from .prompt_tags import prepare_prompt
from .temporal_dialogue_scope import compile_window_text, _digest


@dataclass(frozen=True)
class NativeTextRecipe:
    clip: object
    tokenize_kwargs: dict
    metadata: dict
    counts: dict
    source_audio_ordinal: int
    prompt_primary_audio_ordinal: int
    strict_prompt_tags: bool
    semantic_bridge: object
    resolved_task: str
    frame_count: int
    width: int
    height: int
    content_identity: dict


def _recipe_content(tokenize_kwargs, metadata, counts, source_audio_ordinal,
                    prompt_primary_audio_ordinal, strict_prompt_tags,
                    resolved_task, frame_count, width, height):
    from .modular_sampling.results import _input_identity
    return _input_identity({
        "tokenize_kwargs": tokenize_kwargs, "metadata": metadata, "counts": counts,
        "source_audio_ordinal": source_audio_ordinal,
        "prompt_primary_audio_ordinal": prompt_primary_audio_ordinal,
        "strict_prompt_tags": strict_prompt_tags, "resolved_task": resolved_task,
        "frame_count": frame_count, "width": width, "height": height,
    })


def capture_native_text_recipe(clip, tokenize_kwargs, metadata, counts,
                               source_audio_ordinal, prompt_primary_audio_ordinal,
                               strict_prompt_tags, semantic_bridge, resolved_task,
                               frame_count, width, height):
    # Native builder has already prepared the exact media/latents once. Retain
    # those values and hash their contents; do not VAE encode them per window.
    kwargs, values, media_counts = dict(tokenize_kwargs), dict(metadata), dict(counts)
    identity = _recipe_content(kwargs, values, media_counts, source_audio_ordinal,
                               prompt_primary_audio_ordinal, strict_prompt_tags,
                               resolved_task, frame_count, width, height)
    return NativeTextRecipe(clip, kwargs, values, media_counts, source_audio_ordinal,
                            prompt_primary_audio_ordinal, strict_prompt_tags,
                            semantic_bridge, resolved_task, frame_count, width, height, identity)


def validate_native_text_recipe(recipe):
    if type(recipe) is not NativeTextRecipe:
        raise ValueError("Capture an explicit native text recipe; CONDITIONING alone cannot recover raw media")
    current = _recipe_content(recipe.tokenize_kwargs, recipe.metadata, recipe.counts,
                             recipe.source_audio_ordinal, recipe.prompt_primary_audio_ordinal,
                             recipe.strict_prompt_tags, recipe.resolved_task,
                             recipe.frame_count, recipe.width, recipe.height)
    if current != recipe.content_identity:
        raise ValueError("Native text recipe media/metadata/content changed")
    return recipe


def encode_scoped_window(recipe, dialogue_plan, window, *, return_tokens=False):
    """Encode fresh native tokens/tags, preserve media, apply Bridge exactly once."""
    validate_native_text_recipe(recipe)
    compiled = compile_window_text(dialogue_plan, window)
    if recipe.frame_count != dialogue_plan.total_frames:
        raise ValueError("Native media recipe and dialogue source timeline differ")
    prepared_prompt, warnings = prepare_prompt(
        compiled.prompt, recipe.counts,
        source_audio_ordinal=recipe.source_audio_ordinal,
        prompt_primary_audio_ordinal=recipe.prompt_primary_audio_ordinal,
        strict=recipe.strict_prompt_tags,
    )
    # Native canonicalization can change label length/Audio ordinal. Bind
    # dialogue spans only after that step, in exact native prompt order.
    native_spans = list(re.finditer(r"<d>(.*?)</d>", prepared_prompt, re.DOTALL))
    if len(native_spans) != len(compiled.dialogue):
        raise ValueError("Native preparation changed explicit dialogue tag ownership")
    prepared_events = [{"event_id": event.event_id, "state": event.state,
                        "prompt_char_start": span.start(), "prompt_char_end": span.end(),
                        "local_start_frame": event.local_start_frame,
                        "local_end_frame": event.local_end_frame}
                       for event, span in zip(compiled.dialogue, native_spans, strict=True)]
    tokens = recipe.clip.tokenize(prepared_prompt, **recipe.tokenize_kwargs)
    conditioning = recipe.clip.encode_from_tokens_scheduled(tokens)
    if recipe.metadata:
        conditioning = node_helpers.conditioning_set_values(conditioning, recipe.metadata)
    bridge_report = None
    if recipe.semantic_bridge is not None and recipe.semantic_bridge.active:
        from .semantic_bridge import apply_bridge
        conditioning, bridge_report = apply_bridge(
            conditioning, recipe.semantic_bridge,
            encoding_source=f"native_h3:{recipe.resolved_task}:clip.encode_from_tokens_scheduled",
        )
    # Metadata is descriptive. An executor must separately validate ownership,
    # accepted audio content and its own MODEL/effect pair before sampling.
    scope = {"policy": compiled.policy, "plan_sha256": compiled.plan_sha256,
             "text_sha256": compiled.sha256,
             "prepared_prompt_sha256": _digest(prepared_prompt),
             "window_index": window.index,
             "global_frames": [window.start_frame, window.end_frame],
             "owned_frames": [window.owned_start_frame, window.end_frame],
             "audio_tokens": [window.audio_start, window.audio_stop],
             "owned_audio_tokens": [window.owned_audio_start, window.audio_stop]}
    conditioning = node_helpers.conditioning_set_values(
        conditioning, {"t8_temporal_dialogue_scope": scope})
    report = {**scope, "native_text_reencoded": True, "embedding_sliced": False,
              "dialogue_event_ids": [item.event_id for item in compiled.dialogue],
              "continuation_event_ids": [item.event_id for item in compiled.dialogue
                                          if item.state == "continuation"],
              "performance_event_ids": list(compiled.performance_event_ids),
              "prepared_dialogue_events": prepared_events,
              "omitted_event_ids": list(compiled.omitted_event_ids),
              "prompt_warnings": warnings, "semantic_bridge": bridge_report,
              "sampling_executed": False, "hard_time_isolation": False,
              "quality_qualified": False, "recipe_provider_portable": False}
    result = conditioning, compiled, prepared_prompt, report
    return (*result, tokens) if return_tokens else result
