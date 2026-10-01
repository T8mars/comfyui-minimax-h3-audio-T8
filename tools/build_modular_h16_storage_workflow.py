"""Private H16 seven-window freeze/resume drafts with no old-workflow edits.

The paired native checkpoint freezes the learned LOW handoff.  The exact H16
window artifact freezes completed windows 0..2, so the resume graph samples
only windows 3..6.  Both paths and SHAs are intentionally blank in the resume
draft; the user must paste independently saved values before queueing it.
"""
from __future__ import annotations

from copy import deepcopy
import json

from tools.build_modular_h16_workflow import ROOT, TEMPLATE, split_api, split_frontend


TARGET = ROOT / "artifacts/development/modular-sampling-m4-h16-storage-20260923/candidate-v1"
TARGET_V2 = ROOT / "artifacts/development/modular-sampling-m4-h16-storage-20260923/candidate-v2"
H16_RESULT = "T8_H16_PASS2_RESULT"


class Draft:
    def __init__(self, graph):
        self.graph = deepcopy(graph)
        self.nodes = {node["id"]: node for node in self.graph["nodes"]}
        self.links = {link[0]: link for link in self.graph["links"]}
        self.next_node = max(self.nodes) + 1
        self.next_link = max(self.links) + 1

    def make(self, kind, title, inputs, outputs, widgets, pos):
        node = {"id": self.next_node, "type": kind, "title": title,
                "pos": list(pos), "size": [470, max(180, 48 + 25 * len(inputs))],
                "flags": {}, "order": self.next_node, "mode": 0,
                "inputs": [{"name": name, "type": dtype, "link": None}
                           for name, dtype in inputs],
                "outputs": [{"name": name, "type": dtype, "links": []}
                            for name, dtype in outputs],
                "properties": {"Node name for S&R": kind,
                               "cnr_id": "minimax-h3-audio-T8"},
                "widgets_values": list(widgets)}
        self.nodes[self.next_node] = node
        self.graph["nodes"].append(node)
        self.next_node += 1
        return node

    def disconnect(self, node, name):
        item = next(item for item in node["inputs"] if item["name"] == name)
        if item["link"] is None:
            return
        link_id = item["link"]
        link = self.links.pop(link_id)
        self.graph["links"] = [value for value in self.graph["links"] if value[0] != link_id]
        self.nodes[link[1]]["outputs"][link[2]]["links"].remove(link_id)
        item["link"] = None

    def connect(self, source, target, name, dtype):
        item = next(item for item in target["inputs"] if item["name"] == name)
        if item["link"] is not None:
            raise ValueError("Storage draft input already connected")
        slot = target["inputs"].index(item)
        link = [self.next_link, source[0], source[1], target["id"], slot, dtype]
        self.graph["links"].append(link)
        self.links[self.next_link] = link
        self.nodes[source[0]]["outputs"][source[1]]["links"].append(self.next_link)
        item["link"] = self.next_link
        self.next_link += 1

    def prune(self, roots):
        keep = set(roots)
        pending = list(roots)
        while pending:
            node = self.nodes[pending.pop()]
            for item in node.get("inputs", []):
                if item.get("link") is not None:
                    source = self.links[item["link"]][1]
                    if source not in keep:
                        keep.add(source)
                        pending.append(source)
        self.graph["nodes"] = [node for node in self.graph["nodes"] if node["id"] in keep]
        self.graph["links"] = [link for link in self.graph["links"]
                               if link[1] in keep and link[3] in keep]
        valid = {link[0] for link in self.graph["links"]}
        for node in self.graph["nodes"]:
            for item in node.get("outputs", []):
                item["links"] = [link for link in item.get("links") or [] if link in valid]
        self.graph["last_node_id"] = max(keep)
        self.graph["last_link_id"] = max(valid)
        return self.graph


def freeze_frontend(base, *, extra_outputs=()):
    draft = Draft(base)
    native = draft.make(
        "MiniMaxH3NativeLatentCheckpointSaveT8Advanced",
        "STEP 1: freeze learned LOW handoff (enable confirm_save)",
        [("av_latent", "LATENT")],
        [("av_latent", "LATENT"), ("status", "STRING"),
         ("checkpoint_path", "STRING"), ("file_sha256", "STRING"),
         ("manifest_json", "STRING"), ("report_json", "STRING")],
        ["h16_124f_learned_handoff", "h16_124f_learned_handoff",
         False, True, 8], (3100, 300),
    )
    draft.connect((13, 0), native, "av_latent", "LATENT")
    frozen = draft.make(
        "MiniMaxH3H16WindowSaveEXPT8", "STEP 2: freeze H16 window 2 (0..2 completed)",
        [("window_result", H16_RESULT), ("source_segment", "LATENT"),
         ("segment_spec", "T8_CHUNKED_SOURCE_SEGMENT"),
         ("pass2_context", "T8_CHUNKED_PASS2_CONTEXT"),
         ("plan", "T8_H3_CHUNKED_TWO_PASS_PLAN")],
        [("cumulative_av_latent", "LATENT"), ("window_result", H16_RESULT),
         ("artifact_path", "STRING"), ("artifact_sha256", "STRING"),
         ("report_json", "STRING")], [False], (3100, 1100),
    )
    for name, source, dtype in (
        ("window_result", (38, 1), H16_RESULT),
        ("source_segment", (36, 0), "LATENT"),
        ("segment_spec", (36, 1), "T8_CHUNKED_SOURCE_SEGMENT"),
        ("pass2_context", (29, 0), "T8_CHUNKED_PASS2_CONTEXT"),
        ("plan", (28, 0), "T8_H3_CHUNKED_TWO_PASS_PLAN"),
    ):
        draft.connect(source, frozen, name, dtype)
    return draft.prune((native["id"], frozen["id"], *extra_outputs))


def resume_frontend(base, *, extra_outputs=()):
    draft = Draft(base)
    native = draft.make(
        "MiniMaxH3NativeLatentCheckpointLoadT8Advanced",
        "PASTE exact learned handoff checkpoint path + file SHA",
        [], [("av_latent", "LATENT"), ("status", "STRING"),
             ("resume_verified", "BOOLEAN"), ("checkpoint_id", "STRING"),
             ("content_sha256", "STRING"), ("file_sha256", "STRING"),
             ("manifest_json", "STRING"), ("report_json", "STRING")],
        ["", "", "", 8], (300, -700),
    )
    # Reconcile and high conditioning are rebuilt from the exact learned
    # handoff.  Fixed high geometry is already present in node 14's widgets.
    draft.disconnect(draft.nodes[15], "learned_latent")
    draft.connect((native["id"], 0), draft.nodes[15], "learned_latent", "LATENT")
    draft.disconnect(draft.nodes[14], "width")
    draft.disconnect(draft.nodes[14], "height")
    # The independent HIGH Relay Conditioning copied the same learned-width
    # links as the original HIGH conditioner.  On cold resume those fixed
    # widgets must be used, or merely constructing Relay re-runs LOW.
    for node in tuple(draft.nodes.values()):
        if node["type"] != "MiniMaxH3PromptRelayConditioningT8Advanced":
            continue
        for name in ("width", "height"):
            item = next(item for item in node["inputs"] if item["name"] == name)
            if item["link"] is not None and draft.links[item["link"]][1] == 13:
                draft.disconnect(node, name)
    frozen = draft.make(
        "MiniMaxH3H16WindowLoadEXPT8",
        "PASTE exact H16 window-2 manifest path + SHA; resume at window 3",
        [("source_segment", "LATENT"),
         ("segment_spec", "T8_CHUNKED_SOURCE_SEGMENT"),
         ("pass2_context", "T8_CHUNKED_PASS2_CONTEXT"),
         ("plan", "T8_H3_CHUNKED_TWO_PASS_PLAN")],
        [("cumulative_av_latent", "LATENT"),
         ("window_result", H16_RESULT), ("report_json", "STRING")],
        ["", ""], (3200, 1050),
    )
    for name, source, dtype in (
        ("source_segment", (36, 0), "LATENT"),
        ("segment_spec", (36, 1), "T8_CHUNKED_SOURCE_SEGMENT"),
        ("pass2_context", (29, 0), "T8_CHUNKED_PASS2_CONTEXT"),
        ("plan", (28, 0), "T8_H3_CHUNKED_TWO_PASS_PLAN"),
    ):
        draft.connect(source, frozen, name, dtype)
    bridge = draft.make(
        "MiniMaxH3H16VerifiedNativeSourceEXPT8",
        "Verify exact external native receipt before H16 source identity",
        [("av_latent", "LATENT"), ("resume_verified", "BOOLEAN"),
         ("checkpoint_id", "STRING"), ("content_sha256", "STRING"),
         ("file_sha256", "STRING"), ("manifest_json", "STRING"),
         ("report_json", "STRING")],
        [("source_av_latent", "LATENT"), ("report_json", "STRING")],
        [], (880, -700),
    )
    for name, slot, dtype in (
        ("av_latent", 0, "LATENT"), ("resume_verified", 2, "BOOLEAN"),
        ("checkpoint_id", 3, "STRING"), ("content_sha256", 4, "STRING"),
        ("file_sha256", 5, "STRING"), ("manifest_json", 6, "STRING"),
        ("report_json", 7, "STRING"),
    ):
        draft.connect((native["id"], slot), bridge, name, dtype)
    draft.disconnect(draft.nodes[15], "learned_latent")
    draft.connect((bridge["id"], 0), draft.nodes[15], "learned_latent", "LATENT")
    draft.disconnect(draft.nodes[41], "previous_result")
    draft.connect((frozen["id"], 1), draft.nodes[41], "previous_result", H16_RESULT)
    # A resumed window's external effects must consume the same frozen result
    # as its sampler.  Otherwise the effect dependency silently pulls the old
    # windows (and the LOW sampler) back into the supposedly HIGH-only graph.
    for node in tuple(draft.nodes.values()):
        if node["type"] not in {"MiniMaxH3H16RelayProjectEXPT8",
                                "MiniMaxH3H16EAVApplyEXPT8"}:
            continue
        prior = next((item for item in node["inputs"]
                      if item["name"] == "previous_result"), None)
        if prior is None or prior["link"] is None:
            continue
        old_link = draft.links[prior["link"]]
        if old_link[1:3] == [38, 1]:
            draft.disconnect(node, "previous_result")
            draft.connect((frozen["id"], 1), node, "previous_result", H16_RESULT)
    return draft.prune((21, *extra_outputs))


def candidate_api(frontend, *, freeze):
    api = split_api(frontend)
    for node in frontend["nodes"]:
        kind = node["type"]
        values = node.get("widgets_values", [])
        if kind == "MiniMaxH3NativeLatentCheckpointSaveT8Advanced":
            names = ("filename_prefix", "checkpoint_id", "confirm_save",
                     "verify_after_write", "hash_chunk_megabytes")
        elif kind == "MiniMaxH3NativeLatentCheckpointLoadT8Advanced":
            names = ("checkpoint_path", "expected_manifest_json",
                     "expected_file_sha256", "hash_chunk_megabytes")
        elif kind == "MiniMaxH3H16WindowLoadEXPT8":
            names = ("artifact_path", "artifact_sha256")
        elif kind == "MiniMaxH3H16WindowSaveEXPT8":
            names = ("confirm_save",)
        else:
            continue
        api[str(node["id"])]["inputs"].update(dict(zip(names, values, strict=True)))
    if freeze:
        api.pop("h16_report", None)
    return api


def main():
    if TARGET_V2.exists():
        raise FileExistsError("Private H16 storage candidate v2 already exists")
    base = split_frontend(json.loads(TEMPLATE.read_text(encoding="utf-8")))
    freeze = freeze_frontend(base)
    resume = resume_frontend(base)
    TARGET_V2.mkdir(parents=True)
    for label, graph in (("01_freeze_after_window_2", freeze),
                         ("02_resume_windows_3_to_6_DRAFT", resume)):
        (TARGET_V2 / f"{label}.json").write_text(
            json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        (TARGET_V2 / f"{label}.api.json").write_text(
            json.dumps(candidate_api(graph, freeze=label.startswith("01")),
                       ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"freeze_nodes": len(freeze["nodes"]),
                      "resume_nodes": len(resume["nodes"]),
                      "resume_pass2_windows": sum(node["type"] ==
                                                  "MiniMaxH3H16Pass2WindowEXPT8"
                                                  for node in resume["nodes"])}))


if __name__ == "__main__":
    main()
