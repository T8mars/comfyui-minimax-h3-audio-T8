"""Import-free contracts for explicitly selected prepared-LTX video effects.

Cached event contexts are independently encoded post-connector segments, not
character offsets into the original fixed prompt. All files remain assets.
"""
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = "t8_prepared_ltx_external_effects_v1"
RELAY_SCHEMA = "t8_prepared_ltx_relay_caches_v1"
MODES = ("disabled", "report_only", "apply_exp")


def _finite(value, lower, upper):
    return type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper


def validate_eav(config):
    fields = {"mode", "tau", "start_video_progress", "end_video_progress",
              "max_workspace_mib", "g_hard_limit"}
    if not isinstance(config, dict) or set(config) != fields or config["mode"] not in MODES:
        raise ValueError("Prepared LTX EAV needs the exact external Stage EAV config")
    if (not _finite(config["tau"], -32, 32)
            or not _finite(config["g_hard_limit"], 1, 3)
            or not _finite(config["start_video_progress"], 0, 1)
            or not _finite(config["end_video_progress"], 0, 1)
            or config["start_video_progress"] >= config["end_video_progress"]
            or type(config["max_workspace_mib"]) is not int
            or not 4 <= config["max_workspace_mib"] <= 512):
        raise ValueError("Invalid prepared LTX EAV limits")
    return config


def validate_plan(plan, geometry, prompt):
    if not isinstance(plan, dict):
        raise ValueError("Prepared LTX Relay needs its separate LTX plan")
    canonical = {key: value for key, value in plan.items() if key != "plan_hash"}
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                                      ensure_ascii=False).encode()).hexdigest()
    expected_shape = [1, 128, (geometry["frames"] - 1) // 8 + 1,
                      geometry["height"] // 32, geometry["width"] // 32]
    if (plan.get("type") != "T8_LTX_PROMPT_RELAY_PLAN" or plan.get("schema") != 1
            or plan.get("plan_hash") != digest or plan.get("latent_shape") != expected_shape
            or plan.get("temporal_contract") != "ltx_output_frames_8n_plus_1"
            or plan.get("frame_count") != geometry["frames"] or plan.get("fps") != geometry["fps"]
            or plan.get("global_prompt") != prompt or not _finite(plan.get("epsilon"), 0, 1)
            or not 0 < plan["epsilon"] < 1):
        raise ValueError("Prepared LTX Relay plan hash, prompt or geometry differs")
    events = plan.get("events")
    if not isinstance(events, list) or not 0 <= len(events) <= 7:
        raise ValueError("Prepared LTX Relay needs 0..7 actual timed events")
    for index, event in enumerate(events, 1):
        if (not isinstance(event, dict) or event.get("event_index") != index
                or type(event.get("start_frame")) is not int
                or type(event.get("end_frame_exclusive")) is not int
                or not 0 <= event["start_frame"] < event["end_frame_exclusive"] <= geometry["frames"]
                or event["end_frame_exclusive"] - event["start_frame"] < 2
                or not isinstance(event.get("local_prompt"), str) or not event["local_prompt"].strip()):
            raise ValueError("Prepared LTX Relay event needs finite nonempty output-frame timing")
    return plan


def validate_relay(relay, geometry, prompt, asset_paths=None):
    if (not isinstance(relay, dict) or set(relay) != {"mode", "max_workspace_mib", "plan", "caches"}
            or relay["mode"] not in MODES or type(relay["max_workspace_mib"]) is not int
            or not 4 <= relay["max_workspace_mib"] <= 512):
        raise ValueError("Invalid prepared LTX Relay configuration")
    plan = validate_plan(relay["plan"], geometry, prompt)
    events, caches = plan["events"], relay["caches"]
    if not isinstance(caches, list) or len(caches) != len(events):
        raise ValueError("Prepared LTX Relay needs one actual cache per event (0..7)")
    for index, (event, cache) in enumerate(zip(events, caches), 1):
        if (not isinstance(cache, dict) or set(cache) != {"event_index", "prompt", "path", "sha256"}
                or cache["event_index"] != index or cache["prompt"] != event["local_prompt"]
                or not isinstance(cache["path"], str) or not Path(cache["path"]).is_absolute()
                or not isinstance(cache["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", cache["sha256"])
                or (asset_paths is not None and cache["path"] not in asset_paths)):
            raise ValueError("Prepared LTX Relay event cache needs its exact prompt/file asset identity")
    return relay


def validate_effects(effects, geometry, prompt, asset_paths=None):
    if (not isinstance(effects, dict) or set(effects) != {"schema", "eav", "relay"}
            or effects["schema"] != SCHEMA or effects["eav"] is None and effects["relay"] is None):
        raise ValueError("Expected explicit prepared LTX external effects")
    if effects["eav"] is not None:
        validate_eav(effects["eav"])
    if effects["relay"] is not None:
        validate_relay(effects["relay"], geometry, prompt, asset_paths)
    return effects
