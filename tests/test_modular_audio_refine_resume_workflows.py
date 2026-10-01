"""S26 single-segment freeze/resume must remove the first-pass execution chain."""
import json
from pathlib import Path

from tools.build_modular_audio_refine_resume_workflows import (
    SOURCE, CHECKPOINT_LOAD, CHECKPOINT_SAVE, pair_from_current_source,
)
from h3_audio_t8_pkg.modular_sampling import node_classes
from h3_audio_t8_pkg.modular_sampling.audio_refine_storage_nodes import NODES as frozen_nodes
from h3_audio_t8_pkg.modular_sampling.stage_unet_loader_nodes import NODES as loader_nodes
from h3_audio_t8_pkg.modular_sampling.chunked_v1_relay_nodes import NODES as relay_nodes


ROOT = Path(__file__).resolve().parents[1]
PAIR = (ROOT / "artifacts/development/modular-sampling-m4-audio-refine-independent-model-20260924"
        / "resume-pair-v3")


def _input_source(graph, node, name):
    links = {link[0]: link for link in graph["links"]}
    item = next(item for item in node["inputs"] if item["name"] == name)
    return links[item["link"]][1:3]


def test_strict_frozen_load_is_appended_after_existing_modular_registration():
    actual = node_classes()
    ids = [cls.define_schema().node_id for cls in actual]
    assert [cls.define_schema().node_id for cls in frozen_nodes] == [
        CHECKPOINT_LOAD, 'MiniMaxH3AudioRefineFrozenAVFrameGuardEXPT8',
        'MiniMaxH3AudioRefineContextCommitGateEXPT8']
    start = actual.index(frozen_nodes[0])
    assert actual[start - len(loader_nodes):start] == list(loader_nodes)
    assert actual[start:start + len(frozen_nodes)] == frozen_nodes
    assert actual[start + len(frozen_nodes):start + len(frozen_nodes) + len(relay_nodes)] == relay_nodes
    assert len(ids) == len(set(ids))


def test_saved_freeze_and_cold_resume_have_distinct_sampling_chains():
    source_bytes = SOURCE.read_bytes()
    freeze, resume = pair_from_current_source()
    assert SOURCE.read_bytes() == source_bytes
    for label, graph in (("01_freeze_first_pass", freeze),
                         ("02_resume_refine_only", resume)):
        assert json.loads((PAIR / (label + ".json")).read_text(encoding="utf-8")) == graph
        nodes = {node["id"]: node for node in graph["nodes"]}
        assert len(nodes) == len(graph["nodes"])
        assert len({link[0] for link in graph["links"]}) == len(graph["links"])
        assert all(link[1] in nodes and link[3] in nodes for link in graph["links"])

    frozen_nodes = {node["id"]: node for node in freeze["nodes"]}
    assert 10 in frozen_nodes and frozen_nodes[10]["type"] == "SamplerCustomAdvanced"
    assert 23 not in frozen_nodes and 28 not in frozen_nodes
    frozen_save = next(node for node in freeze["nodes"] if node["type"] == CHECKPOINT_SAVE)
    assert _input_source(freeze, frozen_save, "av_latent") == [10, 0]
    assert frozen_save["widgets_values"][1:4] == ["audio_refine_firstpass", False, True]

    nodes = {node["id"]: node for node in resume["nodes"]}
    assert 10 not in nodes and 1 not in nodes and 6 not in nodes and 7 not in nodes
    assert 17 not in nodes and 13 not in nodes and 18 not in nodes
    assert 23 in nodes and nodes[23]["type"] == "SamplerCustomAdvanced"
    assert sum(node["type"] == "SamplerCustomAdvanced" for node in resume["nodes"]) == 1
    assert sum(node["type"] == "UNETLoader" for node in resume["nodes"]) == 1
    assert sum(node["type"] == "LoraLoaderBypassModelOnly" for node in resume["nodes"]) == 1
    assert sum(node["type"] == "MiniMaxH3PromptRelayPlanT8Advanced"
               for node in resume["nodes"]) == 1
    loaded = next(node for node in resume["nodes"] if node["type"] == CHECKPOINT_LOAD)
    assert loaded["widgets_values"] == ["", "PASTE_EXACT_SAVE_MANIFEST_JSON", "0" * 64, 8]
    assert _input_source(resume, nodes[19], "av_latent") == [loaded["id"], 0]
    assert _input_source(resume, nodes[22], "av_latent") == [loaded["id"], 0]
    assert _input_source(resume, nodes[28], "original_av_latent") == [loaded["id"], 0]
    assert {27, 31}.issubset(nodes)
