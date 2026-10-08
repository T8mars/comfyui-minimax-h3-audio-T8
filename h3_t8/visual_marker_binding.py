"""Explicit raw-reference insertion and observed pre-tokenization marker binding.

The existing native builder owns media ordering, VAE, text encoding and audio.
Only the selected opt-in image is affected. No CLIP wrapper/global monkeypatch.
"""
from dataclasses import dataclass

from .reference_package import canonical, tensor_record
from .visual_marker import MarkerPromptRecipe, checked_image, compose_marker_text, digest


POLICIES = ("marked_vae_and_qwen", "clean_vae_marked_qwen")


@dataclass(frozen=True)
class MarkerBinding:
    recipe: MarkerPromptRecipe
    policy: str
    raw_image_index: int
    vae_source: object
    identity_json: str
    external_plan_text: bool = False

    def verify(self):
        record = self.recipe.verify()
        identity = {"recipe_sha256": self.recipe.sha256, "plan_sha256": record["sha256"],
                    "policy": self.policy, "raw_image_index": self.raw_image_index,
                    "vae_source": tensor_record(checked_image(self.vae_source)),
                    "external_plan_text": self.external_plan_text}
        if canonical(identity) != self.identity_json:
            raise ValueError("Marker binding source/selection changed")
        return record


def insert_marker_reference(recipe, other_images, *, marker_position=1, policy="marked_vae_and_qwen", clean_image=None,
                            external_plan_text=False):
    if type(recipe) is not MarkerPromptRecipe:
        raise ValueError("Use an actual Marker Prompt recipe")
    record = recipe.verify()
    if type(external_plan_text) is not bool:
        raise ValueError("External plan text ownership must be explicit")
    if policy not in POLICIES:
        raise ValueError("Select a marker reference policy explicitly")
    if type(other_images) is not list or len(other_images) > 8:
        raise ValueError("Use up to eight other raw reference images plus the selected marker")
    if type(marker_position) is not int or not 1 <= marker_position <= len(other_images) + 1:
        raise ValueError("marker_position must select a real insertion slot, not a label-only ordinal")
    if policy == "marked_vae_and_qwen":
        if clean_image is not None:
            raise ValueError("clean_image only belongs to explicit clean_vae_marked_qwen mode")
        vae_source = recipe.plan.marked_image
    else:
        vae_source = clean_image if clean_image is not None else recipe.plan.clean_image
        if vae_source is None:
            raise ValueError("Hand-drawn split mode needs its actual clean image; no synthetic erasure")
        vae_source = checked_image(vae_source).clone()
        if vae_source.shape != recipe.plan.marked_image.shape:
            raise ValueError("Clean/marked source dimensions differ; explicitly align them first")
        if record["mode"] == "render_rectangles" and tensor_record(vae_source) != record["clean_rgb"]:
            raise ValueError("Rendered split mode must use the actual recorded clean source")
    identity = {"recipe_sha256": recipe.sha256, "plan_sha256": record["sha256"],
                "policy": policy, "raw_image_index": marker_position,
                "vae_source": tensor_record(checked_image(vae_source)), "external_plan_text": external_plan_text}
    binding = MarkerBinding(recipe, policy, marker_position, vae_source, canonical(identity), external_plan_text)
    images = list(other_images)
    images.insert(marker_position - 1, vae_source)
    binding.verify()
    return images, binding


def validate_raw_binding(binding, ref_images, prepared_reference_set):
    if type(binding) is not MarkerBinding:
        raise ValueError("Use the explicit native marker conditioning adapter")
    binding.verify()
    if prepared_reference_set is not None:
        raise ValueError("Raw marker insertion cannot silently merge a stored reference set")
    if not 1 <= binding.raw_image_index <= len(ref_images):
        raise ValueError("Selected marker source is missing from actual media")
    selected = ref_images[binding.raw_image_index - 1]
    if tensor_record(checked_image(selected)) != tensor_record(checked_image(binding.vae_source)):
        raise ValueError("Actual selected reference is not the marker binding source")


def marker_image_inputs(binding, original, vae_rgb, width, height, size_mode, resize):
    record = binding.verify()
    if tensor_record(checked_image(original)) != tensor_record(checked_image(binding.vae_source)):
        raise ValueError("Selected marker image changed before native encoding")
    qwen_rgb = vae_rgb
    if binding.policy == "clean_vae_marked_qwen":
        qwen_rgb, _, _ = resize(binding.recipe.plan.marked_image, width, height, size_mode)
    observed = {"policy": binding.policy, "plan_sha256": record["sha256"],
                "recipe_sha256": binding.recipe.sha256,
                "actual_raw_image_index": binding.raw_image_index,
                "source_marked_rgb": record["marked_rgb"],
                "source_vae_rgb": tensor_record(checked_image(original)),
                "actual_vae_rgb": tensor_record(vae_rgb.detach().cpu()),
                "actual_qwen_rgb": tensor_record(qwen_rgb.detach().cpu()),
                "actual_reference_dimensions": [int(vae_rgb.shape[2]), int(vae_rgb.shape[1])],
                "actual_canvas_dimensions": [width, height],
                "geometry_observation": "same_source_rendered_rectangles" if record["mode"] == "render_rectangles"
                    else "user_declared_same_size_not_registered",
                "producer_observation": "actual_native_builder_call_not_portable_encoder_certification",
                "per_image_processor_grid": None, "per_image_processor_tokens": None,
                "generated_coordinate_lock": False, "marker_removal_guaranteed": False}
    return qwen_rgb, observed


def bind_native_prompt(binding, observed, keyframe_count, native_prompt):
    binding.verify()
    if observed is None:
        raise ValueError("Marker binding had no actual image consumption")
    ordinal = keyframe_count + binding.raw_image_index
    full, legend, actions = compose_marker_text(binding.recipe, actual_picture_ordinal=ordinal)
    if binding.external_plan_text:
        # The external plan owns all local events. Append ONLY the static legend;
        # never duplicate recipe actions/dialogue as recurring global text.
        full = native_prompt
        if binding.recipe.mode == "generated":
            full += ("\n\n" if full else "") + legend
    receipt = {**observed, "native_picture_ordinal": ordinal,
               "keyframe_picture_count": keyframe_count, "binding_verified": True,
               "binding_scope": "actual_marker_media_slot_not_freeform_base_prompt_or_semantic_identity",
               "generated_legend_binding_verified": binding.recipe.mode == "generated",
               "manual_original": binding.recipe.mode == "manual_original",
               "text_owner": "external_compiled_plan_plus_static_legend" if binding.external_plan_text else "marker_prompt_recipe",
               "actions_appended_as_global": not binding.external_plan_text and binding.recipe.mode == "generated",
               "pre_tokenize_prompt_sha256": digest({"text": full}),
               "static_legend": legend, "one_off_actions": actions,
               "new_native_text_encoding": True, "old_embedding_modified": False}
    receipt["binding_sha256"] = digest(receipt)
    return full, receipt


def verify_consumed_binding(binding, receipt, real_ref_items):
    binding.verify()
    if digest({k: v for k, v in receipt.items() if k != "binding_sha256"}) != receipt["binding_sha256"]:
        raise ValueError("Marker binding receipt changed during native encoding")
    item = real_ref_items[binding.raw_image_index - 1]
    if item["type"] != "image" or tensor_record(item["data"].detach().cpu()) != receipt["actual_qwen_rgb"]:
        raise ValueError("Actual marker Qwen pixels changed during encoding")


def bind_marker_relay_plan(plan, binding, receipt, conditioned_prompt):
    """Derive a separately hashed plan, keeping every external event span intact.

    The old Relay's exact compiled-text gate remains intact. Only this explicit
    adapter may append its independently verified static legend before binding.
    It cannot bless arbitrary preprocessing or alter action/timing payloads.
    """
    from .prompt_relay_advanced import _sha256_json, _validate_plan
    original = _validate_plan(plan)
    binding.verify()
    if not binding.external_plan_text:
        raise ValueError("Marker Relay needs explicit external plan text ownership")
    if digest({key: value for key, value in receipt.items() if key != "binding_sha256"}) != receipt.get("binding_sha256"):
        raise ValueError("Marker Relay binding receipt changed")
    ordinal = receipt["keyframe_picture_count"] + binding.raw_image_index
    if (ordinal != receipt["native_picture_ordinal"] or receipt["recipe_sha256"] != binding.recipe.sha256
            or receipt["policy"] != binding.policy or receipt["actual_raw_image_index"] != binding.raw_image_index):
        raise ValueError("Marker Relay recipe/media selection changed")
    _, legend, _ = compose_marker_text(binding.recipe, actual_picture_ordinal=ordinal)
    expected = original["compiled_prompt"]
    if binding.recipe.mode == "generated":
        expected += ("\n\n" if expected else "") + legend
    if expected != conditioned_prompt:
        raise ValueError("Marker Relay compiled text changed beyond its exact static legend; use canonical connected media tags")
    derived = dict(original)
    derived.pop("plan_hash")
    derived["compiled_prompt"] = expected
    derived["marker_static_legend_binding"] = {
        "source_plan_hash": original["plan_hash"], "binding_sha256": receipt["binding_sha256"],
        "event_spans_preserved": True, "actions_appended_as_global": False}
    derived["plan_hash"] = _sha256_json(derived)
    return _validate_plan(derived)
