"""Additive opt-in native load-profile copies; never rewrite the old graphs."""
from pathlib import Path

from tools import build_modular_ltx_relay_workflows as relay
from tools.build_modular_h16_storage_workflow import Draft
from tools.build_modular_ltx_rgb_workflows import _source, _unique, AUDIT
from tools.workflow_paths import public_workflow_path

POLICY = "MiniMaxH3LTXLoadPolicyEXPT8"
OBSERVE = "MiniMaxH3LTXLoadPolicyAuditEXPT8"
RUNTIME = "T8_LTX_LOAD_POLICY_RUNTIME"
TARGET = Path("examples/workflows/60-ltx-rgb-stage-split/native-load-policy-exp")


def build(graph):
    draft = Draft(graph)
    if any(node["type"] in {POLICY, OBSERVE} for node in draft.nodes.values()):
        raise ValueError("Only an original graph without this explicit profile may be copied")
    setups = [node for node in draft.nodes.values() if node["type"] in relay.SETUPS]
    if len(setups) != 1 or setups[0]["outputs"][0]["type"] != "MODEL":
        raise ValueError("Expected the exact original Setup MODEL output")
    setup = setups[0]
    consumers = []
    for node in draft.nodes.values():
        if any(item["name"] == "model" and item["link"] is not None for item in node.get("inputs", [])):
            edge, dtype = _source(draft, node, "model")
            if edge == (setup["id"], 0):
                if dtype != "MODEL":
                    raise ValueError("Original Setup consumer no longer takes MODEL")
                consumers.append(node)
    if not consumers:
        raise ValueError("No original Setup MODEL consumers")
    policy = draft.make(POLICY, "Explicit native consistent streaming; legacy default remains available",
        [("model", "MODEL")], [("model", "MODEL"), ("runtime", RUNTIME), ("report_json", "STRING")],
        ["consistent_streaming_exp"], (setup["pos"][0] + 500, setup["pos"][1] - 260))
    draft.connect((setup["id"], 0), policy, "model", "MODEL")
    for node in consumers:
        draft.disconnect(node, "model")
        draft.connect((policy["id"], 0), node, "model", "MODEL")
    stage = _unique(draft, AUDIT)
    decoder = _unique(draft, "MiniMaxH3SolEngineTAEHVDecodeT8Advanced")
    candidate, candidate_type = _source(draft, decoder, "latent")
    model, model_type = _source(draft, stage, "model")
    if candidate_type != "LATENT" or model_type != "MODEL":
        raise ValueError("Original final candidate/model route changed")
    observe = draft.make(OBSERVE, "Observe actual preparation calls; not completion/cache/quality proof",
        [("model", "MODEL"), ("candidate_latent", "LATENT"), ("runtime", RUNTIME)],
        [("candidate_latent", "LATENT"), ("report_json", "STRING")], [],
        (decoder["pos"][0], decoder["pos"][1] + 600))
    draft.connect(model, observe, "model", "MODEL")
    draft.connect(candidate, observe, "candidate_latent", "LATENT")
    draft.connect((policy["id"], 1), observe, "runtime", RUNTIME)
    result = draft.prune([node["id"] for node in draft.nodes.values() if node["type"] in {
        relay.source.ISOLATED_WRITER, relay.RELAY_AUDIT, relay.eav.EFFECT_AUDIT, OBSERVE}])
    return result


def generated():
    return {public_workflow_path(TARGET / (name + "_native_consistent_streaming.json")).stem: build(graph)
            for name, graph in relay.generated().items()
            if name.endswith("_external_eav_relay")}
