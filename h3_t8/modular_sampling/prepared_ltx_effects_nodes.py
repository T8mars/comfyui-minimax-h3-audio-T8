"""Append-only external configuration for the isolated Prepared LTX worker."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from comfy_api.latest import io

from ..prepared_backend.ltx_effects_contract import SCHEMA, RELAY_SCHEMA, validate_effects
from ..prepared_backend.resource_guard import file_identity
from ..prepared_generation_contract import validate_bundle
from ..prepared_identity import absolute_path
from .eav import CONFIG_TYPE, EAVConfig
from .ltx_relay_plan import PLAN_TYPE, validate_ltx_relay_plan, build_ltx_relay_plan
from .prepared_ltx_nodes import BUNDLE, CATEGORY


def bind_prepared_ltx_effects(prepared_bundle, *, eav_config=None, ltx_relay_plan=None,
                              relay_cache_manifest_path="", relay_mode="report_only",
                              relay_max_workspace_mib=32):
    validate_bundle(prepared_bundle)
    if prepared_bundle["kind"] != "ltx_refine":
        raise ValueError("Prepared LTX effects cannot be bound to a Tao route")
    if "effects" in prepared_bundle["generation"]:
        raise ValueError("Branch the original bundle before binding another effects configuration")
    if eav_config is not None and type(eav_config) is not EAVConfig:
        raise TypeError("Prepared LTX effects require the actual external Stage EAV config")
    if ltx_relay_plan is None and relay_cache_manifest_path.strip():
        raise ValueError("A Relay cache manifest requires its connected LTX plan")
    bundle = deepcopy(prepared_bundle)
    relay = None
    if ltx_relay_plan is not None:
        plan = validate_ltx_relay_plan(ltx_relay_plan)
        manifest = {"caches": []}
        if plan["events"] or relay_cache_manifest_path.strip():
            path = Path(absolute_path(relay_cache_manifest_path)).resolve(strict=True)
            if not path.is_file() or not 0 < path.stat().st_size <= 1024 ** 2:
                raise ValueError("Prepared Relay needs a bounded local cache manifest")
            manifest = json.loads(path.read_text(encoding="utf8"))
            if (not isinstance(manifest, dict) or set(manifest) != {"schema", "plan_hash", "caches"}
                    or manifest["schema"] != RELAY_SCHEMA or manifest["plan_hash"] != plan["plan_hash"]):
                raise ValueError("Prepared Relay cache manifest differs from the connected plan")
        relay = {"mode": relay_mode, "max_workspace_mib": relay_max_workspace_mib,
                 "plan": deepcopy(plan), "caches": deepcopy(manifest["caches"])}
        validate_effects({"schema": SCHEMA, "eav": None, "relay": relay},
                         bundle["generation"]["geometry"], bundle["generation"]["prompt"])
        identities = {asset["path"]: asset for asset in bundle["assets"]}
        for cache in relay["caches"]:
            current = file_identity(cache["path"])
            if current["sha256"] != cache["sha256"]:
                raise ValueError("Prepared Relay event cache changed")
            if current["path"] in identities and current != identities[current["path"]]:
                raise ValueError("Prepared Relay cache conflicts with an existing bundle asset")
            if current["path"] not in identities:
                bundle["assets"].append(current)
                identities[current["path"]] = current
    bundle["generation"]["effects"] = {"schema": SCHEMA,
        "eav": asdict(eav_config) if eav_config is not None else None, "relay": relay}
    validate_bundle(bundle)
    return bundle, json.dumps({"status": "effects_bound_not_executed",
        "eav_mode": eav_config.mode if eav_config is not None else None,
        "relay_mode": relay["mode"] if relay else None,
        "relay_event_count": len(relay["caches"]) if relay else 0,
        "old_bundle_unchanged": True, "sampling_verified": False,
        "audio_context_unchanged": True,
        "note": "New chain ID required. Prepared post-connector event caches are separate "
                "prerequisites; ordinary Core CLIP conditioning is not interchangeable."}, ensure_ascii=False)


class MiniMaxH3PreparedLTXEffectsBindEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Prepared LTX · External EAV / Relay (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Between Bundle and Generate/Load/Decode. Bind external Stage EAV and/or "
                        "LTX Relay timeline with actual independently encoded post-connector event caches. "
                        "No sampling, text encoding or old workflow change. Use a new chain ID.",
            inputs=[BUNDLE.Input("prepared_bundle"),
                io.String.Input("relay_cache_manifest_path", default=""),
                io.Combo.Input("relay_mode", options=["disabled", "report_only", "apply_exp"], default="report_only"),
                io.Int.Input("relay_max_workspace_mib", default=32, min=4, max=512),
                io.Custom(CONFIG_TYPE).Input("eav_config", optional=True),
                io.Custom(PLAN_TYPE).Input("ltx_relay_plan", optional=True)],
            outputs=[BUNDLE.Output("prepared_bundle"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, prepared_bundle, **kwargs):
        return io.NodeOutput(*bind_prepared_ltx_effects(prepared_bundle, **kwargs))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")


def plan_prepared_ltx_relay(prepared_bundle, local_prompts, timing_mode="auto_equal",
                            time_ranges="", epsilon=.1, allow_gaps=False, allow_overlaps=False):
    """Use the actual selected prepared video tensor, not a synthetic shape stub."""
    validate_bundle(prepared_bundle)
    if prepared_bundle["kind"] != "ltx_refine":
        raise ValueError("Prepared LTX Relay timeline needs an LTX bundle")
    generation = prepared_bundle["generation"]
    expected = next(asset for asset in prepared_bundle["assets"] if asset["path"] == generation["inputs"])
    if file_identity(generation["inputs"]) != expected:
        raise ValueError("Actual prepared AV input changed before timeline planning")
    from safetensors import safe_open
    import torch
    geometry = generation["geometry"]
    with safe_open(generation["inputs"], framework="pt", device="cpu") as reader:
        metadata = reader.metadata() or {}
        if any(metadata.get(field) != str(geometry[key]) for field, key in (
                ("pixel_frames", "frames"), ("width", "width"), ("height", "height"), ("fps", "fps"))):
            raise ValueError("Actual prepared AV metadata differs from the Relay timeline")
        video = reader.get_tensor("video")
    if (video.dtype != torch.bfloat16 or list(video.shape) != [1, 128, (geometry["frames"] - 1) // 8 + 1,
            geometry["height"] // 32, geometry["width"] // 32] or not torch.isfinite(video).all()):
        raise ValueError("Actual prepared video cannot be used for this LTX Relay timeline")
    return build_ltx_relay_plan({"samples": video}, generation["prompt"], local_prompts,
        timing_mode, time_ranges, geometry["fps"], epsilon, allow_gaps, allow_overlaps)


class MiniMaxH3PreparedLTXRelayPlanEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Prepared LTX · External Relay Timeline (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Read only the SHA-bound prepared CPU video and plan its exact 8n+1 output frames. "
                        "Global text comes from the selected bundle/cache. No TE/model/sampler or VAE.",
            inputs=[BUNDLE.Input("prepared_bundle"),
                io.String.Input("local_prompts", multiline=True, dynamic_prompts=True, default=""),
                io.Combo.Input("timing_mode", options=["auto_equal", "frames", "seconds", "percent"], default="auto_equal"),
                io.String.Input("time_ranges", multiline=True, default=""),
                io.Float.Input("epsilon", default=.1, min=.000001, max=.999999),
                io.Boolean.Input("allow_gaps", default=False), io.Boolean.Input("allow_overlaps", default=False)],
            outputs=[io.Custom(PLAN_TYPE).Output("ltx_relay_plan"), io.String.Output("compiled_prompt"),
                io.Int.Output("frame_count"), io.String.Output("timeline_json"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*plan_prepared_ltx_relay(**kwargs))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")


NODES = [MiniMaxH3PreparedLTXEffectsBindEXPT8, MiniMaxH3PreparedLTXRelayPlanEXPT8]
