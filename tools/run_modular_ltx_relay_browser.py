"""Native browser Save As/reopen audit for eight private S27 LTX Relay graphs."""

from copy import deepcopy

from tools import build_modular_ltx_relay_workflows as relay
from tools import run_modular_s09_s29_browser as browser
from tools import serve_modular_ltx_relay_browser as service


browser.SOURCES = service.browser.SOURCES
browser.SOURCE_NAMES = service.browser.SOURCE_NAMES
browser.EDIT = frozenset("S27R_" + name + ".json" for name in relay.generated())
browser.EDIT_MARKER = "S27 LTX Relay canvas QA"
browser.REPORT_SCHEMA = "t8.s27.ltx-relay-native-browser.v1"
browser.REPORT_LIMITS = (
    "Native canvas widget edits and Save As/reload only. No queue, media, human "
    "quality, all-backend, multi-source or public-workflow qualification."
)
_schema_default_appends_original = browser.schema_default_appends


def _schema_default_appends(original, saved, current_nodes):
    """Allow only Core's empty display widgets for two already-linked reports."""
    expanded = deepcopy(original)
    after = {node["id"]: node for node in saved["nodes"]}
    report_types = {"MiniMaxH3LTXRGBStageBindEXPT8", "MiniMaxH3LTXRGBStageAuditEXPT8"}
    fields = ("prep_report_json", "setup_report_json")
    additions = {}
    for old in expanded["nodes"]:
        if old["type"] not in report_types:
            continue
        new = after.get(old["id"])
        if new is None or new["type"] != old["type"]:
            raise ValueError("LTX report node changed on canvas")
        old_values = list(old.get("widgets_values") or [])
        new_values = list(new.get("widgets_values") or [])
        if len(new_values) <= len(old_values):
            continue
        old_inputs = {item["name"]: item for item in old["inputs"]}
        new_inputs = {item["name"]: item for item in new["inputs"]}
        schema = current_nodes[new["type"]]["info"]["input"]["required"]
        if (old_values or new_values != ["", ""]
                or list((new.get("widgets_values_named") or {}).keys()) != list(fields)
                or any(name not in old_inputs or name not in new_inputs
                       or old_inputs[name].get("link") is None
                       or old_inputs[name].get("link") != new_inputs[name].get("link")
                       or schema[name][0] != "STRING" for name in fields)):
            raise ValueError("LTX linked report display widgets changed")
        old["widgets_values"] = ["", ""]
        additions[old["id"]] = ["", ""]
    remaining = _schema_default_appends_original(expanded, saved, current_nodes)
    if additions.keys() & remaining.keys():
        raise ValueError("LTX report widget allowance collided with schema defaults")
    return {**remaining, **additions}


def _edits(graph, copy_name):
    if copy_name not in browser.EDIT:
        return [], {}
    plans = [node for node in graph["nodes"] if node["type"] == relay.PLAN]
    applies = [node for node in graph["nodes"] if node["type"] == relay.APPLY]
    eavs = [node for node in graph["nodes"]
            if node["type"] == "MiniMaxH3StageEAVConfigEXPT8"]
    combined = copy_name.endswith("_external_eav_relay.json")
    if len(plans) != 1 or len(applies) != 1 or len(eavs) != int(combined):
        raise ValueError("Each S27 Relay graph needs one Plan/Apply and optional one EAV config")
    plan, apply = plans[0], applies[0]
    original = plan["widgets_values"][0]
    if not isinstance(original, str) or not original or apply["widgets_values"][0] != "report_only":
        raise ValueError("Saved LTX Relay canvas controls changed")
    marker = f" [{browser.EDIT_MARKER} node {plan['id']}; do not queue]"
    actions = [{"kind": "relay", "node_id": plan["id"],
                "original": original, "marker": marker},
               {"kind": "mode", "node_id": apply["id"]}]
    changes = {(plan["id"], 0): original + marker,
               (apply["id"], 0): "apply_exp"}
    if combined:
        eav = eavs[0]
        if eav["widgets_values"][0] != "report_only":
            raise ValueError("Saved LTX EAV mode changed")
        actions.append({"kind": "mode", "node_id": eav["id"]})
        changes[(eav["id"], 0)] = "apply_exp"
    return actions, changes


def _edit_mode(page, *, node_id):
    """Click the visible COMBO of one selected Relay Apply or EAV Config node."""
    allowed = {relay.APPLY, "MiniMaxH3StageEAVConfigEXPT8"}
    page.evaluate("""([id, allowed]) => {
        const node = app.graph.getNodeById(id);
        if (!node || !allowed.includes(node.type) || node.widgets[0].value !== 'report_only')
            throw Error('Effect mode source changed');
        app.canvas.centerOnNode(node);
    }""", [node_id, list(allowed)])
    page.wait_for_timeout(250)
    point = page.evaluate("""id => {
        const node = app.graph.getNodeById(id);
        const ds = app.canvas.ds;
        return {x: (node.pos[0] + node.size[0] * .75 + ds.offset[0]) * ds.scale,
                y: (node.pos[1] + node.widgets[0].y + 10 + ds.offset[1]) * ds.scale};
    }""", node_id)
    page.mouse.click(point["x"], point["y"])
    option = page.get_by_text("apply_exp", exact=True)
    option.wait_for(state="visible", timeout=3000)
    option.click()
    actual = page.evaluate("id => app.graph.getNodeById(id).widgets[0].value", node_id)
    if actual != "apply_exp":
        raise ValueError(f"Effect COMBO ignored its visible edit: {node_id}")


browser._edits = _edits
browser._edit_eav_mode = _edit_mode
browser.schema_default_appends = _schema_default_appends


if __name__ == "__main__":
    raise SystemExit(browser.main())
