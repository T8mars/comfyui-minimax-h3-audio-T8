"""Stage-local control graph wiring, keeping original RGB and native clocks.

Only active compiled rows enter the graph. Configuration is not an execution
receipt: runtime audit nodes must remain on the selected output dependency.
"""
from copy import deepcopy
import json
from pathlib import Path


def control_condition_length(graph, inputs):
    if "length" in inputs:
        return deepcopy(inputs["length"])
    plan = inputs.get("prompt_relay_plan")
    if isinstance(plan, list) and len(plan) == 2 and plan[0] in graph:
        # Relay owns the actual frame grid in its plan, not a separate
        # conditioning widget. Do not infer it from delivery seconds.
        return deepcopy(graph[plan[0]]["inputs"]["length"])
    raise ValueError("控制源缺少实际条件帧网格，不能猜测交付时长代替")


def append_director_controls(graph, shot, *, stage, model, positive,
                             width, height, length, video_vae, add):
    rows = [row for row in shot.get("control_regions", ()) if row["stage"] == stage.upper()]
    states = []
    for row in rows:
        from .h3_fun_control_advanced import _resolve_fun_control_path

        spec = row["spec"]
        name = spec["control_net_name"]
        path, _folder = _resolve_fun_control_path(name)
        if not Path(path).is_file():
            raise ValueError(f"局部控制找不到所选模型：{name}")
        loader = next((key for key, node in graph.items()
                       if node["class_type"] == "MiniMaxH3FunControlLoaderT8Advanced"
                       and node["inputs"]["control_net_name"] == name), None)
        if loader is None:
            loader = add("MiniMaxH3FunControlLoaderT8Advanced", {"control_net_name": name})
        image = row["asset_kind"] == "image"
        kind, field = ("LoadImage", "image") if image else ("LoadVideo", "file")
        loaded = next((key for key, node in graph.items() if node["class_type"] == kind
                       and node["inputs"].get(field) == row["server_path"]), None)
        if loaded is None:
            # Do not prepare_image here: crop coordinates belong to original RGB.
            loaded = add(kind, {field: row["server_path"]})
        control, fps = [loaded, 0], 24.0
        if not image:
            components = next((key for key, node in graph.items()
                               if node["class_type"] == "GetVideoComponents"
                               and node["inputs"]["video"] == [loaded, 0]), None)
            if components is None:
                components = add("GetVideoComponents", {"video": [loaded, 0]})
            control, fps = [components, 0], [components, 2]
        source = add("MiniMaxH3DirectorControlSourceEXPT8", {
            "control_video": control,
            "spec_json": json.dumps(spec, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False),
            "width": deepcopy(width), "height": deepcopy(height), "length": deepcopy(length),
            "source_kind": "image" if image else "video", "source_fps": fps,
            "stage": stage.upper(),
        })
        applied = add("MiniMaxH3FunControlRegionApplyEXPT8", {
            "model": model, "positive": positive, "control_net": [loader, 0],
            "vae": video_vae, "control_video": [source, 0],
            "width": deepcopy(width), "height": deepcopy(height), "length": deepcopy(length),
            **{key: spec[key] for key in ("control_kind", "fit_mode", "strength",
                                          "start_percent", "end_percent")},
            "region_mask": [source, 1], "crop_plan_json": [source, 2], "stage": stage.upper(),
        })
        model, positive = [applied, 0], [applied, 1]
        states.append([applied, 2])
    return {"model": model, "positive": positive, "states": states}


def audit_control_latent(add, latent, states):
    for state in states:
        latent = [add("MiniMaxH3ControlRegionAuditEXPT8", {
            "region_state": state, "after_sampling": latent}), 0]
    return latent


def audit_control_head(add, boundary, states):
    for state in states:
        boundary = [add("MiniMaxH3DirectorControlHeadAuditEXPT8", {
            "region_state": state, "continuous_boundary": boundary}), 0]
    return boundary
