"""Opt-in image-only Qwen views; native reference packages stay immutable.

No VAE encode, new cache, embedding edit, temporal crop or producer claim.
Budgets describe pixels supplied to CLIP, not an opaque processor's tokens.
"""
from dataclasses import dataclass, replace
import hashlib
import math

import torch

from .core import resize_image
from .reference_package import canonical, tensor_record


POLICY = "t8.h3.qwen-image-view/v1"


def _digest(value):
    return hashlib.sha256(canonical(value).encode("utf8")).hexdigest()


@dataclass(frozen=True)
class QwenImageView:
    short_edge: int
    max_long_edge: int
    max_pixels: int
    source_set_sha256: str

    def validate(self):
        if (type(self.short_edge) is not int or not 32 <= self.short_edge <= 2048
                or type(self.max_long_edge) is not int or not self.short_edge <= self.max_long_edge <= 4096
                or type(self.max_pixels) is not int or not 4096 <= self.max_pixels <= 4 * 1024**2
                or not isinstance(self.source_set_sha256, str) or len(self.source_set_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.source_set_sha256)):
            raise ValueError("Qwen view needs bounded integer edges/pixel budget and its actual source identity")
        return self

    def record(self):
        self.validate()
        return {"policy": POLICY, "short_edge": self.short_edge,
                "max_long_edge": self.max_long_edge, "max_input_pixels_per_image": self.max_pixels,
                "no_upscale": True, "crop": "disabled", "resampler": "core_lanczos_RGB24",
                "source_set_sha256": self.source_set_sha256}


def source_identity(reference_set, routed):
    # These manifests were checked against actual tensor bytes, not caller JSON.
    return _digest({"roles": reference_set.roles_json, "mapping": routed["mapping"],
        "packages": [package.verify(allow_missing_grounding=reference_set.allow_missing_grounding)["sha256"]
                     for package in reference_set.packages]})


def view_dimensions(width, height, strategy):
    strategy.validate()
    if type(width) is not int or type(height) is not int or min(width, height) < 32:
        raise ValueError("Qwen image view requires native RGB dimensions of at least 32 pixels")
    scale = min(1., strategy.short_edge / min(width, height),
                strategy.max_long_edge / max(width, height),
                math.sqrt(strategy.max_pixels / (width * height)))
    target = (math.floor(width * scale), math.floor(height * scale))
    if min(target) < 32:
        raise ValueError("Qwen view budget would collapse this aspect ratio below 32 pixels; relax the budget explicitly")
    return target


def derive_inputs(reference_set, routed):
    strategy = reference_set.qwen_view
    if type(strategy) is not QwenImageView:
        raise ValueError("Use the explicit T8 Qwen image-view strategy")
    strategy.validate()
    if source_identity(reference_set, routed) != strategy.source_set_sha256:
        raise ValueError("Qwen view source set changed; select a new view explicitly")
    items, members = [], []
    for index, (item, mapping) in enumerate(zip(routed["qwen_ref_items"], routed["mapping"], strict=True)):
        if item is None:
            raise ValueError("Qwen view needs complete RGB grounding, not a latent-only reference")
        if item["type"] != "image":
            items.append(item)
            continue
        rgb = item["data"]
        source = tensor_record(rgb)
        width, height = int(rgb.shape[2]), int(rgb.shape[1])
        target_width, target_height = view_dimensions(width, height, strategy)
        changed = (target_width, target_height) != (width, height)
        view = resize_image(rgb, target_width, target_height) if changed else rgb
        derived = tensor_record(view)
        items.append({**item, "data": view})
        members.append({"item_index": index, "role_id": mapping["role_id"],
            "member_id": mapping["member_id"], "routed_picture_ordinal": mapping["ordinal"],
            "source_rgb": source, "qwen_input_rgb": derived,
            "source_dimensions": [width, height], "input_dimensions": [target_width, target_height],
            "resized": changed, "aspect_rounding": "floor_integer_pixels_no_crop"})
    receipt = {**strategy.record(), "members": members,
        "VAE_latents_reencoded": False, "original_packages_changed": False,
        "video_voice_first_last_changed": False, "processor_grid_thw": None,
        "per_image_processor_tokens": None, "processor_observation": "not_exposed_by_conditioning_API",
        "budget_scope": "CLIP_image_input_pixels_not_guaranteed_final_processor_tokens",
        "automatic_accept": False}
    receipt["view_identity_sha256"] = _digest(receipt)
    return {**routed, "qwen_ref_items": items, "qwen_view": receipt}


def verify_view_inputs(inputs):
    receipt = inputs["qwen_view"]
    unsigned = {key: value for key, value in receipt.items() if key != "view_identity_sha256"}
    if _digest(unsigned) != receipt["view_identity_sha256"]:
        raise ValueError("Qwen view receipt changed during fresh encoding")
    for member in receipt["members"]:
        item = inputs["qwen_ref_items"][member["item_index"]]
        if item["type"] != "image" or tensor_record(item["data"]) != member["qwen_input_rgb"]:
            raise ValueError("Qwen view pixels changed during fresh encoding")


def observed_encoding(conditioning):
    """Only measurements returned by this encode, never a predicted grid.

    Core's tags include the vision block's flanking tokens. They are a whole
    conditioning observation, NOT per-image patch counts or encoder provenance.
    """
    observations = []
    for entry in conditioning:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2 or not isinstance(entry[1], dict):
            continue
        tags = entry[1].get("minimax_token_tags")
        if type(tags) is torch.Tensor and tags.ndim == 1 and tags.numel() <= 1024**2:
            tags = tags.detach().cpu()
            if bool(((tags == 0) | (tags == 1)).all()):
                observations.append({"tag_positions": tags.numel(),
                    "vision_tag_positions_including_delimiters": int((tags == 0).sum()),
                    "scope": "actual_returned_whole_conditioning_tags_not_per_image_grid"})
    return {"observed": bool(observations), "conditionings": observations,
            "per_image_processor_grid_available": False}


def select_view(reference_set, *, mode="original", short_edge=512, max_long_edge=1024, max_pixels=524288):
    from .reference_runtime import ReferenceSet
    if type(reference_set) is not ReferenceSet:
        raise ValueError("Use an actual routed T8 reference set")
    routed = reference_set.routed()
    if mode == "original":
        result = reference_set if reference_set.qwen_view is None else replace(reference_set, qwen_view=None)
        return result, {"mode": "original", "policy": None, "automatic_accept": False}
    if mode != "image_short_edge":
        raise ValueError("Choose original or image_short_edge explicitly")
    strategy = QwenImageView(short_edge, max_long_edge, max_pixels,
                             source_identity(reference_set, routed)).validate()
    result = replace(reference_set, qwen_view=strategy)
    return result, derive_inputs(result, routed)["qwen_view"]
