"""Private single-segment Audio Refine first-pass freeze and cold tail drafts.

The freeze graph runs only the original first pass and saves its complete native
AV latent. The resume graph has no first-pass sampler/model chain: its original
AV inputs come only from a manually verified native checkpoint. Old examples
and the existing one-piece route are never edited or migrated.
"""
from __future__ import annotations

import json

from tools.build_modular_h16_storage_workflow import Draft
from tools.build_modular_audio_refine_workflows import SOURCE_DIR, split_frontend


SOURCE = SOURCE_DIR / "2026-08-29_H3_Audio_Refine_Prompt_Relay_Turbo8_Advanced_EXP.json"
CHECKPOINT_SAVE = "MiniMaxH3NativeLatentCheckpointSaveT8Advanced"
CHECKPOINT_LOAD = "MiniMaxH3AudioRefineFrozenFirstPassLoadEXPT8"


def _note(draft, title, message, pos):
    return draft.make("MarkdownNote", title, [], [], [message], pos)


def freeze_frontend(source):
    draft = Draft(source)
    first = draft.nodes[10]
    if first["type"] != "SamplerCustomAdvanced":
        raise ValueError("Audio Refine first-pass sampler contract changed")
    save = draft.make(
        CHECKPOINT_SAVE, "STEP 1 · save completed first-pass AV latent",
        [("av_latent", "LATENT")],
        [("av_latent", "LATENT"), ("status", "STRING"),
         ("checkpoint_path", "STRING"), ("file_sha256", "STRING"),
         ("manifest_json", "STRING"), ("report_json", "STRING")],
        ["audio_refine_firstpass", "audio_refine_firstpass", False, True, 8],
        (2300, 50),
    )
    draft.connect((10, 0), save, "av_latent", "LATENT")
    note = _note(
        draft, "Freeze handoff · keep all three values",
        "Run only this graph. Set confirm_save=true after reviewing output. Copy the "
        "returned checkpoint_path, file_sha256 and manifest_json into the resume "
        "graph. NOT_SAVED is not a frozen first pass.", (2250, -360),
    )
    return draft.prune((save["id"], note["id"]))


def resume_frontend(source):
    base = split_frontend(source, abstain_safe=True, external_refine_relay=True,
                          independent_refine_model=True)
    draft = Draft(base)
    if draft.nodes[10]["type"] != "SamplerCustomAdvanced":
        raise ValueError("Audio Refine first-pass sampler contract changed")
    load = draft.make(
        CHECKPOINT_LOAD, "STEP 2 · paste frozen first-pass path, manifest and SHA",
        [], [("av_latent", "LATENT"), ("status", "STRING"),
             ("resume_verified", "BOOLEAN"), ("checkpoint_id", "STRING"),
             ("content_sha256", "STRING"), ("file_sha256", "STRING"),
             ("manifest_json", "STRING"), ("report_json", "STRING")],
        ["", "PASTE_EXACT_SAVE_MANIFEST_JSON", "0" * 64, 8], (300, -360),
    )
    consumers = [link for link in list(draft.graph["links"])
                 if link[1:3] == [10, 0]]
    if len(consumers) < 4:
        raise ValueError("Audio Refine first-pass AV consumers changed")
    for link in consumers:
        target = draft.nodes[link[3]]
        field = target["inputs"][link[4]]["name"]
        if link[5] != "LATENT":
            raise ValueError("First-pass AV consumer expects non-LATENT data")
        draft.disconnect(target, field)
        draft.connect((load["id"], 0), target, field, "LATENT")
    note = _note(
        draft, "Cold resume · only the tail can sample",
        "Paste all three values from the STEP 1 Save result. An empty path or "
        "placeholder manifest/SHA must fail before refinement. This is an "
        "explicit user-selected checkpoint, not automatic cache reuse.", (300, -680),
    )
    graph = draft.prune((27, 31, note["id"]))
    kinds = {node["type"] for node in graph["nodes"]}
    if (10 in {node["id"] for node in graph["nodes"]} or
            CHECKPOINT_LOAD not in kinds or
            sum(node["type"] == "SamplerCustomAdvanced" for node in graph["nodes"]) != 1):
        raise ValueError("Cold Audio Refine resume still reaches first-pass sampling")
    return graph


def pair_from_current_source():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    return freeze_frontend(source), resume_frontend(source)
