from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import node_helpers
from comfy_api.latest import io


CATEGORY = "T8/MiniMax H3/Conditioning/Experimental"
VISUAL_REFERENCE_KINDS = frozenset({"image", "video", "video_audio"})


def apply_visual_reference_strength(positive, reference_strength: float):
    strength = float(reference_strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("reference_strength must be between 0.0 and 1.0")
    if not isinstance(positive, Sequence) or isinstance(positive, (str, bytes)) or not positive:
        raise ValueError("positive must be a non-empty ComfyUI CONDITIONING value")

    visual_reference_count = 0
    keyframe_count = 0
    has_keyframes = False
    for index, entry in enumerate(positive):
        if not isinstance(entry, Sequence) or len(entry) < 2 or not isinstance(entry[1], Mapping):
            raise ValueError(
                f"positive conditioning entry {index} does not contain a metadata mapping"
            )
        metadata = entry[1]
        refs = metadata.get("minimax_refs")
        if refs is None:
            refs = []
        if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
            raise ValueError("minimax_refs must be a sequence when present")
        entry_visual_count = sum(
            1
            for ref in refs
            if isinstance(ref, Mapping) and ref.get("kind") in VISUAL_REFERENCE_KINDS
        )
        visual_reference_count = max(visual_reference_count, entry_visual_count)

        keyframes = metadata.get("minimax_keyframes")
        if keyframes is None:
            keyframes = []
        if not isinstance(keyframes, Sequence) or isinstance(keyframes, (str, bytes)):
            raise ValueError("minimax_keyframes must be a sequence when present")
        if keyframes:
            has_keyframes = True
            keyframe_count = max(keyframe_count, len(keyframes))

    if visual_reference_count == 0 and not has_keyframes:
        raise ValueError(
            "MiniMax H3 visual conditioning is required: connect image/video references "
            "or first/last-frame keyframes before this EXP node"
        )

    warnings = [
        "Experimental global visual-conditioning control; it may reduce waxy or overly "
        "smoothed reference texture, but improvement is not guaranteed."
    ]
    if has_keyframes:
        warnings.append(
            "The global strength also affects first_frame/last_frame visual keyframes; "
            "lower values may weaken exact endpoint, identity, action, and composition retention."
        )
    if strength <= 0.95:
        warnings.append(
            "reference_strength at or below 0.950 is an aggressive experiment and may "
            "substantially weaken identity, action, composition, and endpoint consistency."
        )

    patched = node_helpers.conditioning_set_values(
        positive,
        {"minimax_visual_cond_noise_aug": strength},
    )
    report = json.dumps(
        {
            "status": "experimental",
            "reference_strength": strength,
            "visual_reference_count": visual_reference_count,
            "has_keyframes": has_keyframes,
            "keyframe_count": keyframe_count,
            "scope": "all MiniMax H3 visual reference and keyframe condition rows",
            "audio_conditioning_changed": False,
            "warnings": warnings,
        },
        ensure_ascii=False,
        indent=2,
    )
    return patched, report


class MiniMaxH3VisualReferenceStrengthEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VisualReferenceStrengthEXPT8",
            display_name="MiniMax H3 Visual Reference Strength (EXP/T8)",
            description=(
                "Experimental global MiniMax H3 visual-conditioning strength. Lower values "
                "add seeded noise to visual reference/keyframe latents and may reduce overly "
                "smoothed texture, but can weaken identity and composition."
            ),
            category=CATEGORY,
            inputs=[
                io.Conditioning.Input(
                    "positive",
                    tooltip="Conditioning containing MiniMax H3 image/video refs or keyframes.",
                ),
                io.Float.Input(
                    "reference_strength",
                    default=0.999,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip=(
                        "Maps directly to minimax_visual_cond_noise_aug. Test 0.999, 0.995, "
                        "then 0.990; values <=0.950 are aggressive."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.String.Output("report"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, positive, reference_strength):
        return io.NodeOutput(
            *apply_visual_reference_strength(positive, reference_strength)
        )
