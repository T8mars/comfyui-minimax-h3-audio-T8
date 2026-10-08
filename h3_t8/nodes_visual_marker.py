"""Separately appended marker authoring nodes; no old node/schema mutation."""
from comfy_api.latest import io

from .reference_package import canonical
from .visual_marker import DEFAULT_MARKERS, prepare_marker_prompt, prepare_markers


MARKER_PLAN = io.Custom("T8_H3_VISUAL_MARKER_PLAN")
MARKER_RECIPE = io.Custom("T8_H3_VISUAL_MARKER_PROMPT_RECIPE")
CATEGORY = "T8/MiniMax H3/Visual Marker Experimental"


def schema(cls, title, inputs, outputs, description):
    return io.Schema(node_id=cls.__name__, display_name="H3 Marker · " + title + " (EXP/T8)",
                     category=CATEGORY, is_experimental=True, inputs=inputs,
                     outputs=outputs, description=description)


class MiniMaxH3VisualMarkerPrepareEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Prepare / 标记图准备", [
            io.Image.Input("image"),
            io.Combo.Input("mode", options=["render_rectangles", "provided_marked"], default="render_rectangles"),
            io.String.Input("markers_json", multiline=True, default=DEFAULT_MARKERS),
            io.Int.Input("line_width", default=3, min=1, max=32),
            io.String.Input("expected_source_sha256", default="", advanced=True,
                tooltip="Optional editor preview RGB binding. Input change rejects a stale preview. Empty is explicit manual authoring.")],
            [io.Image.Output("marked_image"), MARKER_PLAN.Output("marker_plan"), io.String.Output("report_json")],
            "Render explicit source rectangles without modifying the original, or retain an existing "
            "hand-drawn IMAGE exactly. Hand-drawn xyxy is optional; same-color actor/target and shared "
            "targets are allowed. No auto-detection, MASK, VAE, sampling, audio, removal or identity lock. "
            "Connect marked_image to ordinary reference conditioning; binding is unverified here.")

    @classmethod
    def execute(cls, image, mode="render_rectangles", markers_json=DEFAULT_MARKERS, line_width=3, expected_source_sha256=""):
        marked, plan, report = prepare_markers(image, markers_json, mode=mode, line_width=line_width,
                                              expected_source_sha256=expected_source_sha256)
        return io.NodeOutput(marked, plan, canonical(report))


class MiniMaxH3VisualMarkerPromptEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Prompt / 空间提示词", [
            MARKER_PLAN.Input("marker_plan"),
            io.String.Input("base_prompt", multiline=True, dynamic_prompts=True, default=""),
            io.Combo.Input("mode", options=["generated", "manual_original"], default="generated"),
            io.Int.Input("picture_ordinal", default=1, min=1, max=15),
            io.Boolean.Input("include_reminder", default=True)],
            [io.String.Output("full_prompt"), io.String.Output("spatial_legend"),
             io.String.Output("action_text"), MARKER_RECIPE.Output("prompt_recipe"),
             io.String.Output("report_json")],
            "Deterministic new marker legend/actions; preserve original prompt/dialogue bytes. "
            "manual_original emits the original prompt unchanged. No LLM, audio/camera/time defaults "
            "or embedding edit. Picture number is a declaration until an actual downstream media bind. "
            "Static legend and one-off actions remain separate outputs for external temporal/Relay plans.")

    @classmethod
    def execute(cls, marker_plan, base_prompt="", mode="generated", picture_ordinal=1, include_reminder=True):
        full, legend, actions, recipe, report = prepare_marker_prompt(marker_plan, base_prompt,
            mode=mode, picture_ordinal=picture_ordinal, include_reminder=include_reminder)
        return io.NodeOutput(full, legend, actions, recipe, canonical(report))


NODES = [MiniMaxH3VisualMarkerPrepareEXPT8, MiniMaxH3VisualMarkerPromptEXPT8]


class MiniMaxH3VisualMarkerConditioningEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        # A fresh schema instance, never mutate the existing class or sockets.
        from .nodes import MiniMaxH3AudioConditioningT8
        from .nodes_reference_package import RECIPE
        from .visual_marker_binding import POLICIES
        original = MiniMaxH3AudioConditioningT8.define_schema()
        original.node_id = cls.__name__
        original.display_name = "H3 Marker · Bind / 原生空间条件 (EXP/T8)"
        original.category = CATEGORY
        original.is_experimental = True
        original.description = ("Insert the actual Marker Prompt image at an explicit raw-reference position; "
            "native first/last pictures count BEFORE it. Bind its actual resized RGB and generated legend "
            "before tokenize. Explicit marked-VAE/Qwen default or experimental clean-VAE/marked-Qwen; "
            "no box erasure/quality promise. Other images/video/audio and native LOW/HIGH remain intact. "
            "Bridge once; EAV/Relay external. Stored ReferenceSet is not silently merged into raw inputs.")
        replacements = {
            "width": io.Int.Input("width", default=512, min=32, max=16384, step=32),
            "height": io.Int.Input("height", default=288, min=32, max=16384, step=32),
            "audio_mode": io.Combo.Input("audio_mode", options=["native", "reference_only", "lock_source", "remix_source"], default="native"),
        }
        original.inputs = [MARKER_RECIPE.Input("prompt_recipe"),
            io.Int.Input("marker_position", default=1, min=1, max=9),
            io.Combo.Input("reference_policy", options=list(POLICIES), default=POLICIES[0]),
            *[replacements.get(item.id, item) for item in original.inputs if item.id != "prompt"],
            io.Image.Input("clean_image", optional=True)]
        original.outputs = [*original.outputs, RECIPE.Output("native_text_recipe"), io.String.Output("binding_json")]
        return original

    @classmethod
    def execute(cls, prompt_recipe, clip, video_vae, audio_vae, width, height, length,
                marker_position=1, reference_policy="marked_vae_and_qwen", clean_image=None, **kwargs):
        from .conditioning import build_conditioning
        from .core import sorted_autogrow_values
        from .visual_marker_binding import insert_marker_reference
        images, binding = insert_marker_reference(prompt_recipe,
            sorted_autogrow_values(kwargs.pop("ref_images", None)), marker_position=marker_position,
            policy=reference_policy, clean_image=clean_image)
        kwargs.setdefault("audio_mode", "native")
        result = build_conditioning(clip, video_vae, audio_vae, prompt_recipe.base_prompt,
            width, height, length, ref_images={f"ref_image_{i}": image for i, image in enumerate(images)},
            visual_marker_binding=binding, return_text_recipe=True, **kwargs)
        recipe = result[-1]["text_recipe"]
        return io.NodeOutput(*result[:6], recipe, canonical(recipe.metadata["t8_visual_marker_binding"]))


NODES.append(MiniMaxH3VisualMarkerConditioningEXPT8)


class MiniMaxH3VisualMarkerRelayConditioningEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        from .nodes_prompt_relay_advanced import MiniMaxH3PromptRelayConditioningT8Advanced
        from .nodes_reference_package import RECIPE
        from .visual_marker_binding import POLICIES
        original = MiniMaxH3PromptRelayConditioningT8Advanced.define_schema()
        original.node_id = cls.__name__
        original.display_name = "H3 Marker · External Relay / 外置分段条件 (EXP/T8)"
        original.category = CATEGORY
        original.description = ("External Relay Plan owns compiled text/local actions/dialogue. "
            "This adapter appends ONLY the actual-bound static marker legend, not recurring actions. "
            "Fresh native token spans/layout and paired MODEL/positive/latent. Explicit split/reference "
            "mode; EAV stays downstream. Does not rewrite old plans or reuse old text bindings.")
        original.inputs = [MARKER_RECIPE.Input("prompt_recipe"),
            io.Int.Input("marker_position", default=1, min=1, max=9),
            io.Combo.Input("reference_policy", options=list(POLICIES), default=POLICIES[0]),
            *original.inputs, io.Image.Input("clean_image", optional=True)]
        original.outputs = [*original.outputs, RECIPE.Output("native_text_recipe"), io.String.Output("binding_json")]
        return original

    @classmethod
    def execute(cls, prompt_recipe, marker_position=1, reference_policy="marked_vae_and_qwen", clean_image=None, **kwargs):
        from .core import sorted_autogrow_values
        from .prompt_relay_advanced import build_prompt_relay_conditioning
        from .visual_marker_binding import insert_marker_reference
        images, binding = insert_marker_reference(prompt_recipe,
            sorted_autogrow_values(kwargs.pop("ref_images", None)), marker_position=marker_position,
            policy=reference_policy, clean_image=clean_image, external_plan_text=True)
        return io.NodeOutput(*build_prompt_relay_conditioning(
            **kwargs, ref_images={f"ref_image_{i}": image for i, image in enumerate(images)},
            visual_marker_binding=binding))


NODES.append(MiniMaxH3VisualMarkerRelayConditioningEXPT8)
