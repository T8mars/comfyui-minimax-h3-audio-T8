"""Private source-bound copies of the two existing H3 Super RGB/LTX graphs."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_modular_h16_storage_workflow import Draft  # noqa: E402

SOURCE_DIR = ROOT / "examples/workflows/22-sol-engine-h3-super"
TARGET = ROOT / "artifacts/development/modular-sampling-m4-ltx-rgb-20260923/candidate-v1"
SETUPS = {"MiniMaxH3SolEngineLTXRefinerSetupT8Advanced",
          "MiniMaxH3SolEngineLTXIdentityRefinerSetupT8Advanced"}
BIND = "MiniMaxH3LTXRGBStageBindEXPT8"
AUDIT = "MiniMaxH3LTXRGBStageAuditEXPT8"


def sources():
    paths = []
    for path in sorted(SOURCE_DIR.glob("*.json")):
        graph = json.loads(path.read_text(encoding="utf-8"))
        if any(node["type"] in SETUPS for node in graph["nodes"]):
            paths.append(path)
    return paths


def _unique(draft, kind):
    selected = [node for node in draft.nodes.values() if node["type"] == kind]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one {kind}, found {len(selected)}")
    return selected[0]


def _source(draft, node, field):
    item = next(item for item in node["inputs"] if item["name"] == field)
    link = draft.links.get(item["link"])
    if link is None:
        raise ValueError(f"Expected connected {node['type']}.{field}")
    return (link[1], link[2]), link[5]


def split_frontend(source):
    draft = Draft(source)
    for node in draft.nodes.values():
        for output in node.get("outputs", []):
            if output.get("links") is None:
                output["links"] = []
    setup = [node for node in draft.nodes.values() if node["type"] in SETUPS]
    if len(setup) != 1:
        raise ValueError("Expected one independent LTX Setup")
    setup = setup[0]
    prep = _unique(draft, "MiniMaxH3SolEngineDraftToLTXT8Advanced")
    sampler = _unique(draft, "SamplerCustomAdvanced")
    guider = _unique(draft, "CFGGuider")
    decoder = _unique(draft, "MiniMaxH3SolEngineTAEHVDecodeT8Advanced")
    trim = _unique(draft, "MiniMaxH3OutputTrimT8")
    if (_source(draft, sampler, "sampler")[0] != (setup["id"], 1) or
            _source(draft, sampler, "sigmas")[0] != (setup["id"], 2) or
            _source(draft, sampler, "guider")[0] != (guider["id"], 0) or
            _source(draft, guider, "model")[0] != (setup["id"], 0) or
            _source(draft, decoder, "latent")[0] != (sampler["id"], 0)):
        raise ValueError("Old LTX refiner path is no longer the expected external sampler")
    source_frames, _ = _source(draft, prep, "frames")
    source_audio, audio_type = _source(draft, trim, "audio")
    prepared_frames = (prep["id"], 0)
    ltx_latent, _ = _source(draft, sampler, "latent_image")
    noise, _ = _source(draft, sampler, "noise")
    if audio_type != "AUDIO" or ltx_latent[0] != _unique(draft, "LTXVLatentUpsampler")["id"]:
        raise ValueError("Old LTX source or H3 audio bypass changed")
    bind = draft.make(BIND, "Bind RGB bridge and original H3 audio before separate LTX sampler",
        [("source_frames", "IMAGE"), ("source_audio", "AUDIO"),
         ("prepared_frames", "IMAGE"), ("prep_report_json", "STRING"),
         ("ltx_latent", "LATENT"), ("model", "MODEL"),
         ("noise", "NOISE"), ("guider", "GUIDER"),
         ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
         ("setup_report_json", "STRING")],
        [("noise", "NOISE"), ("guider", "GUIDER"),
         ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
         ("ltx_latent", "LATENT"), ("stage_boundary", "T8_LTX_RGB_STAGE_BOUNDARY"),
         ("report_json", "STRING")], [], (2200, 820))
    audit = draft.make(AUDIT, "Audit LTX candidate; retain original H3 AUDIO wire",
        [("stage_boundary", "T8_LTX_RGB_STAGE_BOUNDARY"),
         ("source_frames", "IMAGE"), ("source_audio", "AUDIO"),
         ("prepared_frames", "IMAGE"), ("prep_report_json", "STRING"),
         ("ltx_latent", "LATENT"), ("model", "MODEL"),
         ("noise", "NOISE"), ("guider", "GUIDER"),
         ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
         ("setup_report_json", "STRING"), ("candidate_latent", "LATENT")],
        [("candidate_latent", "LATENT"), ("source_audio", "AUDIO"),
         ("report_json", "STRING")], [], (2660, 850))
    sources_to_bind = {
        "source_frames": (source_frames, "IMAGE"),
        "source_audio": (source_audio, "AUDIO"),
        "prepared_frames": (prepared_frames, "IMAGE"),
        "prep_report_json": ((prep["id"], 6), "STRING"),
        "ltx_latent": (ltx_latent, "LATENT"),
        "model": ((setup["id"], 0), "MODEL"),
        "noise": (noise, "NOISE"),
        "guider": ((guider["id"], 0), "GUIDER"),
        "sampler": ((setup["id"], 1), "SAMPLER"),
        "sigmas": ((setup["id"], 2), "SIGMAS"),
        "setup_report_json": ((setup["id"], 4), "STRING"),
    }
    for name, (source_slot, dtype) in sources_to_bind.items():
        draft.connect(source_slot, bind, name, dtype)
    for slot, name, dtype in ((0, "noise", "NOISE"), (1, "guider", "GUIDER"),
                              (2, "sampler", "SAMPLER"), (3, "sigmas", "SIGMAS"),
                              (4, "latent_image", "LATENT")):
        draft.disconnect(sampler, name)
        draft.connect((bind["id"], slot), sampler, name, dtype)
    draft.connect((bind["id"], 5), audit, "stage_boundary", "T8_LTX_RGB_STAGE_BOUNDARY")
    for name, (source_slot, dtype) in sources_to_bind.items():
        if name in {"noise", "guider", "sampler", "sigmas", "ltx_latent"}:
            slot = {"noise": 0, "guider": 1, "sampler": 2,
                    "sigmas": 3, "ltx_latent": 4}[name]
            source_slot = (bind["id"], slot)
        draft.connect(source_slot, audit, name, dtype)
    draft.disconnect(decoder, "latent")
    draft.disconnect(trim, "audio")
    draft.connect((sampler["id"], 0), audit, "candidate_latent", "LATENT")
    draft.connect((audit["id"], 0), decoder, "latent", "LATENT")
    draft.connect((audit["id"], 1), trim, "audio", "AUDIO")
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
            raise ValueError("LTX RGB candidate has a dependency cycle")
        for node in ready:
            seen.add(node["id"])
            ordered.append(node)
    for order, node in enumerate(ordered):
        node["order"] = order
    return graph


def main():
    files = sources()
    if len(files) != 2:
        raise ValueError(f"Expected two existing Sol LTX workflows; found {len(files)}")
    if TARGET.exists():
        raise FileExistsError("Use a new private candidate version; do not overwrite evidence")
    graphs = [(path, split_frontend(json.loads(path.read_text(encoding="utf-8"))))
              for path in files]
    TARGET.mkdir(parents=True)
    for path, graph in graphs:
        target = TARGET / (path.stem + "_StageBound_EXP.json")
        target.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"source": path.name, "nodes": len(graph["nodes"]),
                          "links": len(graph["links"]), "candidate": target.name}))


if __name__ == "__main__":
    main()
