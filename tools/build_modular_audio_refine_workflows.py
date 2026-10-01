"""Copy every existing Audio Refine sampler graph with opt-in stage guards.

The existing Setup -> SamplerCustomAdvanced route, including empty-SIGMAS
abstain, remains numerically unchanged. These are private frontend drafts;
API serialization/Core queue qualification is a separate gate.
"""
import json
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_modular_h16_storage_workflow import Draft  # noqa: E402

SOURCE_DIR = ROOT / "examples/workflows/18-audio-refine"
TARGET = ROOT / "artifacts/development/modular-sampling-m4-audio-refine-20260923/candidate-v1"
FAMILIES = {
    "MiniMaxH3AudioRefineDualClockSetupT8Advanced": (
        "MiniMaxH3AudioRefineDualClockStageBindEXPT8",
        "MiniMaxH3AudioRefineDualClockStageAuditEXPT8"),
    "MiniMaxH3AudioRefineDualModelSetupT8Advanced": (
        "MiniMaxH3AudioRefineDualModelStageBindEXPT8",
        "MiniMaxH3AudioRefineDualModelStageAuditEXPT8"),
    "MiniMaxH3AudioRefineCompatibilitySetupT8Advanced": (
        "MiniMaxH3AudioRefineCompatStageBindEXPT8",
        "MiniMaxH3AudioRefineCompatStageAuditEXPT8"),
}
NON_EXECUTABLE = {"MarkdownNote", "VHS_VideoCombine"}
WIDGET_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN", "COMBO"}
UI_ONLY_TRAILING_WIDGETS = {"RandomNoise": 1, "LoadImage": 1}
KNOWN_NEW_TRAILING_DEFAULTS = {
    "CreateVideo": {("codec",), ("color_space", "codec")},
    "MiniMaxH3AudioConditioningT8": {("allow_above_reference_area",)},
}


def sources():
    found = []
    for path in sorted(SOURCE_DIR.glob("*.json")):
        graph = json.loads(path.read_text(encoding="utf-8"))
        if any(node["type"] in FAMILIES for node in graph["nodes"]):
            found.append(path)
    return found


def _input_source(draft, node, name):
    item = next(item for item in node["inputs"] if item["name"] == name)
    if item["link"] is None:
        raise ValueError(f"{node['type']} missing connected {name}")
    link = draft.links[item["link"]]
    return (link[1], link[2]), link[5]


def _external_refine_relay(draft, setup, *, independent_refine_model=False):
    """Give the compatible Relay tail its own editable Plan and paired MODEL/COND.

    The first-pass Relay branch and its latent are left alone. The second
    conditioning branch is signed by the existing Audio Refine Audit/Route/Plan
    before Setup, so changing only its text cannot silently reuse a prior Plan.
    """
    def one(kind):
        matches = [node for node in draft.nodes.values() if node["type"] == kind]
        if len(matches) != 1:
            raise ValueError(f"Expected one {kind} for separate refine Relay")
        return matches[0]

    old_plan = one("MiniMaxH3PromptRelayPlanT8Advanced")
    old_conds = [node for node in draft.nodes.values() if node["type"] in {
        "MiniMaxH3PromptRelayConditioningT8Advanced",
        "MiniMaxH3PromptRelayLongVideoConditioningT8Advanced"}]
    if len(old_conds) != 1:
        raise ValueError("Expected one regular or long-video Relay Conditioning")
    old_cond = old_conds[0]
    long_video = old_cond["type"] == "MiniMaxH3PromptRelayLongVideoConditioningT8Advanced"
    if independent_refine_model and long_video:
        raise ValueError("Long Video independent MODEL needs a cold-stage route adapter")
    old_segment_plan = one("MiniMaxH3PromptRelayLongVideoPlanT8Advanced") if long_video else None
    audit = one("MiniMaxH3AudioRefineAuditT8Advanced")
    route = one("MiniMaxH3AudioRefineCompatibilityRouteT8Advanced")
    if setup["type"] != "MiniMaxH3AudioRefineCompatibilitySetupT8Advanced":
        raise ValueError("Separate Audio Refine Relay currently needs the compatibility Plan family")
    plan = draft.make(old_plan["type"], "REFINE ONLY · independent Prompt Relay Plan",
        [(item["name"], item["type"]) for item in old_plan["inputs"]],
        [(item["name"], item["type"]) for item in old_plan["outputs"]],
        deepcopy(old_plan["widgets_values"]), (1450, 1500))
    refine_loader = None
    if independent_refine_model:
        old_loader = one("UNETLoader")
        refine_loader = draft.make(
            old_loader["type"], "REFINE ONLY · independent checkpoint",
            [(item["name"], item["type"]) for item in old_loader["inputs"]],
            [(item["name"], item["type"]) for item in old_loader["outputs"]],
            deepcopy(old_loader["widgets_values"]), (1300, 1850))
        if any(item["link"] is not None for item in old_loader["inputs"]):
            raise ValueError("Independent refine checkpoint expected a widget-only UNETLoader")
    segment_plan = None
    if old_segment_plan is not None:
        segment_plan = draft.make(
            old_segment_plan["type"], "REFINE ONLY · independent segment time map",
            [(item["name"], item["type"]) for item in old_segment_plan["inputs"]],
            [(item["name"], item["type"]) for item in old_segment_plan["outputs"]],
            deepcopy(old_segment_plan["widgets_values"]), (1700, 1500))
        for item in old_segment_plan["inputs"]:
            name = item["name"]
            if name == "prompt_relay_plan":
                source, dtype = (plan["id"], 0), item["type"]
            elif item["link"] is not None:
                source, dtype = _input_source(draft, old_segment_plan, name)
            else:
                continue
            draft.connect(source, segment_plan, name, dtype)
    cond = draft.make(old_cond["type"], "REFINE ONLY · paired Relay MODEL + CONDITIONING",
        [(item["name"], item["type"]) for item in old_cond["inputs"]],
        [(item["name"], item["type"]) for item in old_cond["outputs"]],
        deepcopy(old_cond["widgets_values"]), (1900, 1500))
    for item in old_cond["inputs"]:
        name = item["name"]
        if name == "prompt_relay_plan":
            source, dtype = ((segment_plan or plan)["id"], 0), item["type"]
        elif name == "model" and refine_loader is not None:
            source, dtype = (refine_loader["id"], 0), item["type"]
        elif item["link"] is not None:
            source, dtype = _input_source(draft, old_cond, name)
        else:
            continue
        draft.connect(source, cond, name, dtype)
    refine_model = (cond["id"], 0)
    if independent_refine_model:
        old_lora = one("LoraLoaderBypassModelOnly")
        if [item["name"] for item in old_lora["inputs"]] != ["model"]:
            raise ValueError("Independent refine LoRA requires the known model-only branch")
        lora_widgets = deepcopy(old_lora["widgets_values"])
        lora_widgets[1] = 0.0  # Old refine model was unpatched; opt in to a tail LoRA.
        refine_lora = draft.make(
            old_lora["type"], "REFINE ONLY · optional independent LoRA (0 default)",
            [(item["name"], item["type"]) for item in old_lora["inputs"]],
            [(item["name"], item["type"]) for item in old_lora["outputs"]],
            lora_widgets, (2200, 1850))
        draft.connect(refine_model, refine_lora, "model", "MODEL")
        refine_model = (refine_lora["id"], 0)
    for target, fields in ((audit, {"model": 0, "positive": 1,
                                   "conditioned_prompt": 4, "media_map_json": 5,
                                   "conditioning_report": 6}),
                           (route, {"refine_model": 0, "positive": 1}),
                           (setup, {"positive": 1})):
        for field, slot in fields.items():
            dtype = next(item["type"] for item in target["inputs"] if item["name"] == field)
            draft.disconnect(target, field)
            source = refine_model if (target is audit and field == "model") or (
                target is route and field == "refine_model") else (cond["id"], slot)
            draft.connect(source, target, field, dtype)
    return plan, cond


def split_frontend(source, *, abstain_safe=False, external_refine_relay=False,
                   independent_refine_model=False):
    if external_refine_relay and not abstain_safe:
        raise ValueError("Separate refine Relay requires the ABSTAIN-safe tail audit")
    if independent_refine_model and not external_refine_relay:
        raise ValueError("Independent refine MODEL requires a separate Relay branch")
    draft = Draft(source)
    for node in draft.nodes.values():
        for output in node.get("outputs", []):
            if output.get("links") is None:
                output["links"] = []
    setups = [node for node in draft.nodes.values() if node["type"] in FAMILIES]
    if len(setups) != 1:
        raise ValueError("Expected one existing Audio Refine Setup per workflow")
    setup = setups[0]
    bind_kind, audit_kind = FAMILIES[setup["type"]]
    plan_source, plan_type = _input_source(draft, setup, "plan")
    original_source, original_type = _input_source(draft, setup, "av_latent")
    if original_type != "LATENT":
        raise ValueError("Audio Refine original AV has an unexpected type")
    samplers = [node for node in draft.nodes.values() if node["type"] == "SamplerCustomAdvanced"]
    tail = [node for node in samplers if all(
        _input_source(draft, node, field)[0] == (setup["id"], slot)
        for field, slot in (("noise", 1), ("guider", 2), ("sampler", 3),
                            ("sigmas", 4), ("latent_image", 5)))]
    if len(tail) != 1:
        raise ValueError("Expected one existing Audio Refine external sampler")
    sampler = tail[0]
    bind = draft.make(bind_kind, "Bind signed Audio Refine tail; preserve abstain",
        [("plan", plan_type), ("original_av_latent", "LATENT"),
         ("model", "MODEL"), ("noise", "NOISE"), ("guider", "GUIDER"),
         ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
         ("stage_latent", "LATENT"), ("setup_report_json", "STRING")],
        [("model", "MODEL"), ("noise", "NOISE"), ("guider", "GUIDER"),
         ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
         ("stage_latent", "LATENT"),
         ("stage_boundary", "T8_AUDIO_REFINE_STAGE_BOUNDARY"),
         ("report_json", "STRING")], [], (2800, 760))
    audit = draft.make(audit_kind, "Audit tail candidate before existing Quality Gate",
        [("stage_boundary", "T8_AUDIO_REFINE_STAGE_BOUNDARY"),
         ("plan", plan_type), ("original_av_latent", "LATENT"),
         ("stage_latent", "LATENT"), ("setup_report_json", "STRING"),
         ("candidate_av_latent", "LATENT")],
        [("candidate_av_latent", "LATENT"), ("report_json", "STRING")], [], (3850, 740))
    draft.connect(plan_source, bind, "plan", plan_type)
    draft.connect(original_source, bind, "original_av_latent", "LATENT")
    for slot, name, dtype in ((0, "model", "MODEL"), (1, "noise", "NOISE"),
                              (2, "guider", "GUIDER"), (3, "sampler", "SAMPLER"),
                              (4, "sigmas", "SIGMAS"), (5, "stage_latent", "LATENT"),
                              (6, "setup_report_json", "STRING")):
        draft.connect((setup["id"], slot), bind, name if slot != 6 else "setup_report_json", dtype)
    for slot, name, dtype in ((1, "noise", "NOISE"), (2, "guider", "GUIDER"),
                              (3, "sampler", "SAMPLER"), (4, "sigmas", "SIGMAS"),
                              (5, "latent_image", "LATENT")):
        draft.disconnect(sampler, name)
        draft.connect((bind["id"], slot), sampler, name, dtype)
    for source_slot, name, dtype in (((bind["id"], 6), "stage_boundary", "T8_AUDIO_REFINE_STAGE_BOUNDARY"),
                                     (plan_source, "plan", plan_type),
                                     (original_source, "original_av_latent", "LATENT"),
                                     ((bind["id"], 5), "stage_latent", "LATENT"),
                                     ((setup["id"], 6), "setup_report_json", "STRING")):
        draft.connect(source_slot, audit, name, dtype)
    old_targets = [link for link in list(draft.graph["links"])
                   if link[1:3] == [sampler["id"], 0]]
    if not old_targets:
        raise ValueError("Audio Refine sampler has no candidate consumer")
    for link in old_targets:
        target = draft.nodes[link[3]]
        field = target["inputs"][link[4]]["name"]
        draft.disconnect(target, field)
        draft.connect((audit["id"], 0), target, field, "LATENT")
    draft.connect((sampler["id"], 0), audit, "candidate_av_latent", "LATENT")
    if abstain_safe:
        audio_audits = [node for node in draft.nodes.values()
                        if node["type"] == "MiniMaxH3TwoPassAudioAuditT8Advanced"]
        if len(audio_audits) != 1:
            raise ValueError("Expected one existing two-pass audio audit")
        audio_audit = audio_audits[0]
        audio_audit["type"] = "MiniMaxH3AudioRefineTailDeliveryAuditEXPT8"
        audio_audit["title"] = "Audit sampled tail or exact ABSTAIN passthrough"
        audio_audit["properties"]["Node name for S&R"] = audio_audit["type"]
        audio_audit["inputs"].extend([
            {"name": "stage_boundary", "type": "T8_AUDIO_REFINE_STAGE_BOUNDARY", "link": None},
            {"name": "stage_report_json", "type": "STRING", "link": None},
        ])
        draft.connect((bind["id"], 6), audio_audit, "stage_boundary",
                      "T8_AUDIO_REFINE_STAGE_BOUNDARY")
        draft.connect((audit["id"], 1), audio_audit, "stage_report_json", "STRING")
    if external_refine_relay:
        _external_refine_relay(draft, setup,
                               independent_refine_model=independent_refine_model)
    graph = draft.graph
    graph["last_node_id"] = max(draft.nodes)
    graph["last_link_id"] = max(draft.links)
    seen = set()
    ordered = []
    while len(ordered) < len(draft.nodes):
        ready = [node for node in graph["nodes"] if node["id"] not in seen and
                 all(item.get("link") is None or draft.links[item["link"]][1] in seen
                     for item in node.get("inputs", []))]
        if not ready:
            raise ValueError("Audio Refine candidate has a dependency cycle")
        for node in ready:
            seen.add(node["id"])
            ordered.append(node)
    for order, node in enumerate(ordered):
        node["order"] = order
    return graph


def split_api(frontend, node_info):
    """Serialize exact live-schema widgets/links; reject unknown UI values.

    Third-party VHS terminals and Markdown notes are deliberately omitted from
    standalone Core validation. All executable upstream/audio-gate nodes remain.
    """
    graph = {}
    nodes = {node["id"]: node for node in frontend["nodes"]}
    for node in frontend["nodes"]:
        kind = node["type"]
        if kind in NON_EXECUTABLE:
            continue
        if kind not in node_info:
            raise ValueError(f"Missing current Core schema for {kind}")
        schema = node_info[kind].get("input", node_info[kind])
        connected = {item["name"] for item in node.get("inputs", [])
                     if item.get("link") is not None}
        widget_names = []
        fields = {}
        for group in ("required", "optional"):
            for name, spec in schema.get(group, {}).items():
                dtype = spec[0]
                if (dtype in WIDGET_TYPES if isinstance(dtype, str)
                        else isinstance(dtype, (list, tuple))):
                    widget_names.append(name)
                    fields[name] = spec
        values = node.get("widgets_values") or []
        if isinstance(values, dict):
            if not set(values).issubset(widget_names):
                raise ValueError(f"Unexpected widget names for {kind}: {set(values) - set(widget_names)}")
            widgets = dict(values)
        else:
            if not isinstance(values, list):
                values = [values]
            extra = UI_ONLY_TRAILING_WIDGETS.get(kind, 0)
            if kind == "SaveVideo" and 1 <= len(values) <= 3:
                names = ["filename_prefix", "format", "codec"][:len(values)]
                if any(name not in schema.get("required", {}) and
                       name not in schema.get("optional", {}) for name in names):
                    raise ValueError("SaveVideo dynamic format widget no longer exists")
            elif len(values) == len(widget_names):
                names = widget_names
            elif len(values) == len([name for name in widget_names if name not in connected]):
                names = [name for name in widget_names if name not in connected]
            elif (len(values) < len(widget_names) and
                  tuple(widget_names[len(values):]) in KNOWN_NEW_TRAILING_DEFAULTS.get(kind, set())):
                names = widget_names[:len(values)]
            elif len(values) > len(widget_names) and len(values) <= len(widget_names) + extra:
                names = widget_names
            else:
                raise ValueError(f"Widget contract changed for {kind}: {len(values)} values, "
                                 f"{len(widget_names)} schema widgets {widget_names}")
            widgets = dict(zip(names, values[:len(names)], strict=True))
            if kind == "SaveVideo" and len(values) == 1:
                if not any("format" in schema.get(group, {}) for group in ("required", "optional")):
                    raise ValueError("SaveVideo current Core has no format input")
                widgets["format"] = "auto"
            for name in widget_names[len(names):]:
                if name not in connected and any(
                        name in suffix for suffix in KNOWN_NEW_TRAILING_DEFAULTS.get(kind, set())):
                    default = fields[name][1].get("default")
                    if default is None:
                        raise ValueError(f"Missing current Core default for {kind}.{name}")
                    widgets[name] = default
        graph[str(node["id"])] = {"class_type": kind, "inputs": widgets,
                                  "_meta": {"title": node.get("title", kind)}}
    for _, source, source_slot, target, target_slot, _ in frontend["links"]:
        if str(target) in graph:
            if str(source) not in graph:
                raise ValueError(f"Executable {nodes[target]['type']} depends on skipped {nodes[source]['type']}")
            field = nodes[target]["inputs"][target_slot]["name"]
            graph[str(target)]["inputs"][field] = [str(source), source_slot]
    return graph


def main():
    if TARGET.exists():
        raise FileExistsError("Private Audio Refine candidate directory already exists")
    files = sources()
    if len(files) != 10:
        raise ValueError(f"Expected ten existing refine workflows, found {len(files)}")
    graphs = [(path, split_frontend(json.loads(path.read_text(encoding="utf-8")))) for path in files]
    TARGET.mkdir(parents=True)
    for path, graph in graphs:
        target = TARGET / path.name.replace(".json", "_StageBound_EXP.json")
        target.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"source": path.name, "nodes": len(graph["nodes"]),
                          "links": len(graph["links"]), "candidate": target.name}))


if __name__ == "__main__":
    main()
