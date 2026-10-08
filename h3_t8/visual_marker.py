"""Opt-in visual annotations and deterministic text, not a spatial sampler.

Plans describe source-image rectangles, not generated-video coordinates,
identity recognition, voice ownership, or guaranteed marker removal.
"""
from dataclasses import dataclass
import hashlib
import json
import math

import torch

from .reference_package import IDENTIFIER, canonical, tensor_record


PLAN_SCHEMA = "t8.h3.visual-marker-plan/v1"
RECIPE_SCHEMA = "t8.h3.visual-marker-prompt/v1"
MAX_JSON_BYTES = 65536
MAX_MARKERS = 64
MAX_RELATIONS = 128
MAX_IMAGE_PIXELS = 4 * 1024**2
COLORS = {"red": "#ff3030", "blue": "#3070ff", "yellow": "#ffdf00",
          "cyan": "#00dfff", "green": "#20df50", "magenta": "#ef30ef",
          "orange": "#ff9000", "white": "#ffffff", "black": "#000000"}
DEFAULT_MARKERS = canonical({"markers": [
    {"marker_id": "actor_A", "kind": "actor", "role_id": "A", "color": "cyan",
     "description": "the person", "xyxy": [0.15, 0.12, 0.45, 0.90]},
    {"marker_id": "target_A", "kind": "target", "color": "cyan",
     "description": "the destination on the right", "xyxy": [0.68, 0.60, 0.96, 0.96]}],
    "relations": [{"actor_marker_id": "actor_A", "target_marker_id": "target_A",
                   "action": "walk to the destination and stop there"}]})


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf8")).hexdigest()


def _text(value, field, *, empty=False, maximum=4096):
    if not isinstance(value, str) or len(value.encode("utf8")) > maximum or (not empty and not value.strip()):
        raise ValueError(f"{field} must be explicit bounded text")
    return value


def _identifier(value, field):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} needs a unique short ASCII identifier")
    return value


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate marker JSON field: " + key)
        result[key] = value
    return result


def _color(value):
    _text(value, "color", maximum=32)
    value = value.lower()
    rgb = COLORS.get(value, value)
    if len(rgb) != 7 or rgb[0] != "#" or any(c not in "0123456789abcdef" for c in rgb[1:]):
        raise ValueError("Marker color must be a supported name or #RRGGBB")
    return rgb, value


def parse_markers(raw, mode):
    if mode not in ("render_rectangles", "provided_marked"):
        raise ValueError("Select render_rectangles or provided_marked explicitly")
    _text(raw, "markers_json", maximum=MAX_JSON_BYTES)
    try:
        spec = json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + x)))
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid bounded marker JSON") from exc
    if type(spec) is not dict or set(spec) != {"markers", "relations"}:
        raise ValueError("Marker JSON requires exactly markers and relations")
    markers, relations = spec["markers"], spec["relations"]
    if type(markers) is not list or not 2 <= len(markers) <= MAX_MARKERS:
        raise ValueError("Use 2..64 explicit actor/target markers")
    if type(relations) is not list or not 1 <= len(relations) <= MAX_RELATIONS:
        raise ValueError("Use 1..128 explicit actor-to-target relations")
    normalized, by_id = [], {}
    for marker in markers:
        if type(marker) is not dict or not {"marker_id", "kind", "color", "description"} <= set(marker):
            raise ValueError("Marker requires marker_id, kind, color and description")
        if set(marker) - {"marker_id", "kind", "color", "description", "role_id", "xyxy"}:
            raise ValueError("Unknown marker fields; don't silently discard a control declaration")
        identifier = _identifier(marker["marker_id"], "marker_id")
        if identifier in by_id or marker["kind"] not in ("actor", "target"):
            raise ValueError("Marker IDs must be unique and kinds actor/target")
        role = _identifier(marker.get("role_id"), "actor role_id") if marker["kind"] == "actor" else None
        if marker["kind"] == "target" and "role_id" in marker:
            raise ValueError("Declare visual roles on actors, not destination markers")
        color, label = _color(marker["color"])
        coordinates = marker.get("xyxy")
        if coordinates is not None:
            if (type(coordinates) is not list or len(coordinates) != 4
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in coordinates)
                    or not 0 <= coordinates[0] < coordinates[2] <= 1
                    or not 0 <= coordinates[1] < coordinates[3] <= 1):
                raise ValueError("xyxy must be finite, nonempty normalized source-image coordinates")
            coordinates = [float(x) for x in coordinates]
        elif mode == "render_rectangles":
            raise ValueError("Automatic rendering needs xyxy; hand-drawn provided_marked does not")
        item = {"marker_id": identifier, "kind": marker["kind"], "role_id": role,
                "color_rgb": color, "color_label": label,
                "description": _text(marker["description"], "marker description"), "xyxy": coordinates}
        normalized.append(item)
        by_id[identifier] = item
    edges = []
    for relation in relations:
        if type(relation) is not dict or set(relation) != {"actor_marker_id", "target_marker_id", "action"}:
            raise ValueError("Relation needs actor_marker_id, target_marker_id and action")
        actor, target = relation["actor_marker_id"], relation["target_marker_id"]
        if (not isinstance(actor, str) or not isinstance(target, str)
                or actor not in by_id or target not in by_id
                or by_id[actor]["kind"] != "actor" or by_id[target]["kind"] != "target"):
            raise ValueError("Relation must resolve an actual actor marker and target marker")
        edges.append({"actor_marker_id": actor, "target_marker_id": target,
                      "action": _text(relation["action"], "action")})
    referenced = {x[field] for x in edges for field in ("actor_marker_id", "target_marker_id")}
    if referenced != set(by_id):
        raise ValueError("Every declared marker needs an explicit relation")
    return {"markers": normalized, "relations": edges}


def checked_image(image):
    if (type(image) is not torch.Tensor or image.ndim != 4 or image.shape[0] != 1
            or image.shape[-1] != 3 or image.dtype != torch.float32
            or min(image.shape[1:3]) < 2 or image.shape[1] * image.shape[2] > MAX_IMAGE_PIXELS
            or not bool(torch.isfinite(image).all()) or not bool(((image >= 0) & (image <= 1)).all())):
        raise ValueError("Marker source needs one finite float32 RGB IMAGE, 0..1, at most 4MP")
    return image.detach().cpu().contiguous()


def _pixel_box(coordinates, width, height):
    x0, y0, x1, y1 = coordinates
    return [math.floor(x0 * width), math.floor(y0 * height),
            min(width, math.ceil(x1 * width)), min(height, math.ceil(y1 * height))]


@dataclass(frozen=True)
class MarkerPlan:
    record_json: str
    marked_image: torch.Tensor
    clean_image: object = None

    def verify(self):
        record = json.loads(self.record_json)
        if record.get("schema") != PLAN_SCHEMA or digest({k: v for k, v in record.items() if k != "sha256"}) != record.get("sha256"):
            raise ValueError("Marker plan record changed")
        if tensor_record(checked_image(self.marked_image)) != record["marked_rgb"]:
            raise ValueError("Marker pixels changed; rebuild the plan from the actual image")
        if self.clean_image is not None and tensor_record(checked_image(self.clean_image)) != record["clean_rgb"]:
            raise ValueError("Marker clean source changed")
        if (self.clean_image is None) != (record["clean_rgb"] is None):
            raise ValueError("Marker source provenance changed")
        return record


def prepare_markers(image, markers_json, *, mode="render_rectangles", line_width=3, expected_source_sha256=""):
    spec = parse_markers(markers_json, mode)
    if type(line_width) is not int or not 1 <= line_width <= 32:
        raise ValueError("line_width must be an explicit 1..32 pixel integer")
    source = checked_image(image)
    before = tensor_record(source)
    if (not isinstance(expected_source_sha256, str) or (expected_source_sha256 and
            (len(expected_source_sha256) != 64 or expected_source_sha256 != before["sha256"]))):
        raise ValueError("Editor preview and actual IMAGE differ; reload the correct source or explicitly clear the preview binding")
    marked = source.clone()
    width, height = int(source.shape[2]), int(source.shape[1])
    pixels, warnings = [], []
    for marker in spec["markers"]:
        box = _pixel_box(marker["xyxy"], width, height) if marker["xyxy"] is not None else None
        pixels.append({"marker_id": marker["marker_id"], "source_pixel_xyxy": box,
                       "geometry_observation": "rendered_source_rectangle" if mode == "render_rectangles"
                            else "unobserved" if box is None else "user_declared_not_detected"})
        if mode == "render_rectangles":
            x0, y0, x1, y1 = box
            color = torch.tensor([int(marker["color_rgb"][i:i+2], 16) / 255. for i in (1, 3, 5)])
            stroke = min(line_width, x1 - x0, y1 - y0)
            marked[:, y0:min(y0+stroke, y1), x0:x1] = color
            marked[:, max(y1-stroke, y0):y1, x0:x1] = color
            marked[:, y0:y1, x0:min(x0+stroke, x1)] = color
            marked[:, y0:y1, max(x1-stroke, x0):x1] = color
            if min(x1-x0, y1-y0) < 16:
                warnings.append("Small marker may obscure details: " + marker["marker_id"])
    if mode == "provided_marked":
        warnings.append("Hand-drawn markers and roles are user declarations, not detected or spatially verified")
    actors = [item for item in spec["markers"] if item["kind"] == "actor"]
    if len({item["color_rgb"] for item in actors}) < len(actors):
        warnings.append("Multiple actor markers share a color; rely on explicit descriptions/relations, not color alone")
    for index, left in enumerate(spec["markers"]):
        for right in spec["markers"][index+1:]:
            a, b = left["xyxy"], right["xyxy"]
            if a is not None and b is not None and min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                warnings.append(f"Overlapping markers may be ambiguous: {left['marker_id']}/{right['marker_id']}")
    warnings.append("Soft 2D image guidance only; no hard trajectory, voice identity or marker-removal guarantee")
    if tensor_record(source) != before:
        raise ValueError("Input pixels changed during marker preparation")
    record = {"schema": PLAN_SCHEMA, "mode": mode, "source_rgb": before,
              "clean_rgb": before if mode == "render_rectangles" else None,
              "marked_rgb": tensor_record(marked), "source_dimensions": [width, height],
              "spec": spec, "pixel_rectangles": pixels, "line_width": line_width if mode == "render_rectangles" else None,
              "pixel_rounding": "floor_start_ceil_exclusive_end_clamped" if mode == "render_rectangles" else None,
              "coordinate_space": "source_reference_image_normalized_xyxy_not_generated_video",
              "binding_verified": False, "native_picture_ordinal": None, "warnings": warnings}
    record["editor_preview_source_verified"] = bool(expected_source_sha256)
    record["sha256"] = digest(record)
    plan = MarkerPlan(canonical(record), marked, source.clone() if mode == "render_rectangles" else None)
    plan.verify()
    return marked, plan, record


@dataclass(frozen=True)
class MarkerPromptRecipe:
    plan: MarkerPlan
    base_prompt: str
    mode: str
    picture_ordinal: int
    include_reminder: bool
    sha256: str

    def verify(self):
        record = self.plan.verify()
        values = {"schema": RECIPE_SCHEMA, "plan_sha256": record["sha256"],
                  "base_prompt": self.base_prompt, "mode": self.mode,
                  "picture_ordinal": self.picture_ordinal, "include_reminder": self.include_reminder}
        if digest(values) != self.sha256:
            raise ValueError("Marker prompt recipe changed")
        return record


def compose_marker_text(recipe, *, actual_picture_ordinal=None):
    record = recipe.verify()
    ordinal = recipe.picture_ordinal if actual_picture_ordinal is None else actual_picture_ordinal
    if type(ordinal) is not int or not 1 <= ordinal <= 15:
        raise ValueError("Picture ordinal needs an actual bounded integer")
    by_id = {x["marker_id"]: x for x in record["spec"]["markers"]}
    legend = []
    for marker in by_id.values():
        kind = f"visual role {marker['role_id']}" if marker["kind"] == "actor" else "destination or interaction target"
        legend.append(f"In <Picture {ordinal}>, the {marker['color_label']} rectangle named {marker['marker_id']} identifies {kind}: {marker['description']}.")
    if recipe.include_reminder:
        legend.append("The colored rectangles are annotation guides, not physical objects in the scene.")
    actions = []
    for relation in record["spec"]["relations"]:
        actor, target = by_id[relation["actor_marker_id"]], by_id[relation["target_marker_id"]]
        actions.append(f"Visual role {actor['role_id']} ({actor['description']}, marker {actor['marker_id']}): {relation['action']}. Target: {target['description']} (marker {target['marker_id']}).")
    legend, actions = "\n".join(legend), "\n".join(actions)
    full = recipe.base_prompt
    if recipe.mode == "generated":
        full += ("\n\n" if full else "") + legend + "\n" + actions
    _text(full, "expanded marker prompt", empty=True, maximum=262144)
    return full, legend, actions


def prepare_marker_prompt(plan, base_prompt, *, mode="generated", picture_ordinal=1, include_reminder=True):
    if type(plan) is not MarkerPlan:
        raise ValueError("Use an actual Marker Prepare plan")
    record = plan.verify()
    _text(base_prompt, "base_prompt", empty=True, maximum=MAX_JSON_BYTES)
    if mode not in ("generated", "manual_original") or type(include_reminder) is not bool:
        raise ValueError("Select generated or manual_original explicitly")
    if type(picture_ordinal) is not int or not 1 <= picture_ordinal <= 15:
        raise ValueError("Picture ordinal is an unverified 1..15 declaration until actual conditioning")
    values = {"schema": RECIPE_SCHEMA, "plan_sha256": record["sha256"], "base_prompt": base_prompt,
              "mode": mode, "picture_ordinal": picture_ordinal, "include_reminder": include_reminder}
    recipe = MarkerPromptRecipe(plan, base_prompt, mode, picture_ordinal, include_reminder, digest(values))
    full, legend, actions = compose_marker_text(recipe)
    report = {**values, "recipe_sha256": recipe.sha256, "binding_verified": False,
              "native_picture_ordinal": None, "declared_picture_ordinal": picture_ordinal,
              "original_text_retained_exact": True, "full_prompt_sha256": hashlib.sha256(full.encode()).hexdigest(),
              "manual_original": mode == "manual_original", "automatic_translation": False,
              "automatic_dialogue_or_audio_rules": False, "warnings": record["warnings"]}
    return full, legend, actions, recipe, report
