"""Build a private H16 freeze/resume pair with independent Relay and EAV windows.

This composes the existing seven-window effect candidate with explicit native
and H16 window storage.  It never edits the original all-in-one example.
The resume draft intentionally has blank receipt fields and must not be queued
until the matching freeze has produced real paths and SHA256 values.
"""
from __future__ import annotations

import json

from tools.build_modular_h16_storage_workflow import (
    Draft, candidate_api, freeze_frontend, resume_frontend,
)
from tools.build_modular_h16_workflow import (
    ROOT, TEMPLATE, WINDOW_COUNT, add_eav_frontend, add_relay_frontend,
    split_frontend,
)


TARGET = (ROOT / "artifacts/development/"
          "modular-sampling-m4-h16-effect-storage-20260924/candidate-v4")


def candidate_pair():
    base = split_frontend(json.loads(TEMPLATE.read_text(encoding="utf-8")))
    effects = add_eav_frontend(add_relay_frontend(base, with_audit=False))
    draft = Draft(effects)
    audits = sorted((node for node in draft.nodes.values()
                     if node["type"] == "MiniMaxH3H16EAVAuditEXPT8"),
                    key=lambda node: node["id"])
    if len(audits) != WINDOW_COUNT:
        raise ValueError("Expected one actual EAV audit per H16 window")
    outputs = []
    for index, audit in enumerate(audits):
        x, y = audit["pos"]
        output = draft.make(
            "PreviewAny", f"H16 window {index} Relay + EAV actual-call report",
            [("source", "STRING")], [], [], (x + 470, y),
        )
        draft.connect((audit["id"], 1), output, "source", "STRING")
        outputs.append(output["id"])

    freeze = freeze_frontend(draft.graph, extra_outputs=outputs[:3])
    resume = resume_frontend(draft.graph, extra_outputs=outputs[3:])
    return freeze, resume


def main():
    if TARGET.exists():
        raise FileExistsError("Private H16 effect/storage candidate already exists")
    freeze, resume = candidate_pair()
    TARGET.mkdir(parents=True)
    for label, graph in (
        ("01_freeze_after_window_2_relay_eav", freeze),
        ("02_resume_windows_3_to_6_relay_eav_DRAFT", resume),
    ):
        (TARGET / f"{label}.json").write_text(
            json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        (TARGET / f"{label}.api.json").write_text(
            json.dumps(candidate_api(graph, freeze=label.startswith("01")),
                       ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"freeze_nodes": len(freeze["nodes"]),
                      "resume_nodes": len(resume["nodes"])}))


if __name__ == "__main__":
    main()
