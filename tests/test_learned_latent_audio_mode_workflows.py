from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CATEGORY = ROOT / "examples" / "workflows" / "13-latent-upscale"
SOURCE_AUDIO = "h3_twopass_voice_5683_5p152s.flac"
SEED = 2608215001

CASES = (
    (
        "2026-08-21_H3_Learned_Latent_TwoPass_Hybrid_Lock_Source_Advanced_EXP.json",
        "Hybrid",
        "lock_source",
        0.0,
        True,
        1,
        True,
        True,
    ),
    (
        "2026-08-21_H3_Learned_Latent_TwoPass_Hybrid_Remix_Source_020_Advanced_EXP.json",
        "Hybrid",
        "remix_source",
        0.20,
        True,
        1,
        True,
        False,
    ),
    (
        "2026-08-21_H3_Learned_Latent_TwoPass_Hybrid_Reference_Only_Advanced_EXP.json",
        "Hybrid",
        "reference_only",
        1.0,
        True,
        1,
        True,
        False,
    ),
    (
        "2026-08-21_H3_Learned_Latent_TwoPass_I2VA_Native_Speech_Advanced_EXP.json",
        "I2VA",
        "native",
        1.0,
        False,
        0,
        False,
        False,
    ),
)


def _load(filename: str) -> dict:
    return json.loads((CATEGORY / filename).read_text(encoding="utf-8"))


def _node(workflow: dict, node_id: int) -> dict:
    return next(node for node in workflow["nodes"] if node["id"] == node_id)


@pytest.mark.parametrize(
    (
        "filename",
        "task_type",
        "audio_mode",
        "strength",
        "add_source",
        "ordinal",
        "has_audio",
        "save_mux_audio",
    ),
    CASES,
)
def test_saved_two_pass_audio_mode_workflow_contracts(
    filename,
    task_type,
    audio_mode,
    strength,
    add_source,
    ordinal,
    has_audio,
    save_mux_audio,
):
    workflow = _load(filename)
    assert workflow["version"] == 0.4
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}

    assert nodes[9]["widgets_values"] == [8, 4, 4]
    assert nodes[13]["widgets_values"][2] == 2.0
    assert nodes[16]["widgets_values"][:3] == [12.0, 3.0, False]
    assert nodes[16]["widgets_values"][-1] == SEED
    assert nodes[11]["widgets_values"][0] == SEED
    assert nodes[18]["widgets_values"][0] == SEED
    assert nodes[15]["widgets_values"][1:] == ["legacy_policy", 0.0]
    assert nodes[7]["widgets_values"][0] == nodes[14]["widgets_values"][0]
    assert "<d>" in nodes[7]["widgets_values"][0]
    assert nodes[7]["widgets_values"][-1] is False
    assert nodes[14]["widgets_values"][-1] is True
    note_text = "\n".join(
        node["widgets_values"][0]
        for node in workflow["nodes"]
        if node["type"] == "MarkdownNote"
    )
    assert "4+4" in note_text
    assert "this exact mode/seed was not re-rendered" in note_text

    for conditioning_id in (7, 14):
        values = nodes[conditioning_id]["widgets_values"]
        assert values[4] == task_type
        assert values[5] == audio_mode
        assert values[6] == pytest.approx(strength)
        assert values[7] is add_source
        assert values[8] == ordinal

    if has_audio:
        assert nodes[27]["type"] == "LoadAudio"
        assert nodes[27]["widgets_values"] == [SOURCE_AUDIO]
        assert nodes[27]["outputs"][0]["links"] == [41, 42]
        assert nodes[7]["inputs"][15]["link"] == 41
        assert nodes[14]["inputs"][15]["link"] == 42
        assert links[41][1:] == [27, 0, 7, 15, "AUDIO"]
        assert links[42][1:] == [27, 0, 14, 15, "AUDIO"]
    else:
        assert 27 not in nodes
        assert nodes[7]["inputs"][15]["link"] is None
        assert nodes[14]["inputs"][15]["link"] is None

    if save_mux_audio:
        assert links[40][1:] == [14, 2, 21, 7, "AUDIO"]
        assert nodes[14]["outputs"][2]["links"] == [40]
        assert nodes[20]["outputs"][1]["links"] is None
    else:
        assert links[40][1:] == [20, 1, 21, 7, "AUDIO"]
        assert nodes[20]["outputs"][1]["links"] == [40]


def test_saved_two_pass_audio_mode_workflows_have_unique_ids_and_prefixes():
    workflows = [_load(case[0]) for case in CASES]
    assert len({workflow["id"] for workflow in workflows}) == len(CASES)
    prefixes = {_node(workflow, 21)["widgets_values"][2] for workflow in workflows}
    assert len(prefixes) == len(CASES)
