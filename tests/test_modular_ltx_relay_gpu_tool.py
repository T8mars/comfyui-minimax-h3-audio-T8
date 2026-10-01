"""The trained-weight probe stays owned and excludes video publication."""
import pytest
import torch

from tools import build_modular_ltx_relay_workflows as builder
from tools import build_modular_ltx_rgb_source_workflows as source
from tools import run_modular_ltx_relay_gpu as probe
from tools import run_modular_ltx_relay_pair_gpu as pair
from tools import run_modular_ltx_relay_cold_gpu as cold


def minimal_api(*, combined=False):
    kinds = ["LoadVideo", "MiniMaxH3SolEngineDraftToLTXT8Advanced", source.SAVE,
             next(iter(builder.SETUPS)), builder.PLAN, builder.ENCODE, builder.APPLY,
             builder.RELAY_AUDIT, "SamplerCustomAdvanced", source.ISOLATED_WRITER]
    if combined:
        kinds.append("MiniMaxH3LTXEAVAuditEXPT8")
    return {str(index): {"class_type": kind, "inputs": {}} for index, kind in enumerate(kinds, 1)}


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("mode", ["report_only", "apply_exp"])
def test_probe_only_samples_owned_latent_and_observation_reports(combined, mode):
    api = minimal_api(combined=combined)
    graph, pins = probe._probe_graph(api, width=128, height=64,
        global_prompt="Same global scene", local_prompts="Event one\nEvent two", mode=mode)
    assert not any(node["class_type"] == source.ISOLATED_WRITER for node in graph.values())
    assert graph[pins["load"]]["inputs"]["file"] == "source.mp4"
    assert graph[pins["store"]]["inputs"]["confirm_save"] is False
    assert graph[pins["setup"]]["inputs"]["attention_backend"] == "dense_reference"
    assert graph[pins["apply"]]["inputs"]["mode"] == mode
    assert graph[pins["plan"]]["inputs"]["local_prompts"] == "Event one\nEvent two"
    assert graph[pins["save"]]["inputs"]["samples"] == [pins["audit"], 0]
    assert set(pins["reports"]) == ({"plan", "encode", "apply", "audit", "eav"} if combined
                                    else {"plan", "encode", "apply", "audit"})


@pytest.mark.parametrize("width,height,events", [(127, 64, "one\ntwo"),
                                                   (128, 64, "one"), (0, 64, "one\ntwo")])
def test_probe_rejects_unqualified_geometry_or_one_event(width, height, events):
    with pytest.raises(ValueError):
        probe._probe_graph(minimal_api(), width=width, height=height,
                           global_prompt="scene", local_prompts=events, mode="apply_exp")


def test_paired_probe_requires_an_exact_same_mode_repeat():
    baseline = torch.zeros(1, 128, 3, 2, 2)
    changed = baseline.clone()
    changed[..., 0, 0] = 1.
    assert pair._delta(baseline, baseline) == {"equal": True, "max_abs": 0., "rmse": 0.}
    effect = pair._delta(baseline, changed)
    assert effect["equal"] is False and effect["max_abs"] == 1.
    assert effect["rmse"] > 0.


def test_eav_pair_changes_only_eav_mode_with_relay_always_applied():
    api = minimal_api(combined=True)
    api["90"] = {"class_type": "MiniMaxH3StageEAVConfigEXPT8", "inputs": {
        "mode": "report_only", "tau": .25}}
    graphs = []
    for mode in ("report_only", "apply_exp", "report_only"):
        graph, pins = pair._effect_graph(api, width=128, height=64,
            global_prompt="Same scene", local_prompts="First\nSecond",
            effect="eav", mode=mode, tau=4.)
        assert graph[pins["apply"]]["inputs"]["mode"] == "apply_exp"
        assert graph["90"]["inputs"] == {"mode": mode, "tau": 4.}
        assert "eav" in pins["reports"]
        graphs.append(graph)
    assert graphs[0] == graphs[2]
    graphs[1]["90"]["inputs"]["mode"] = "report_only"
    assert graphs[0] == graphs[1]
    assert api["90"]["inputs"] == {"mode": "report_only", "tau": .25}


@pytest.mark.parametrize("combined", [False, True])
def test_cold_probe_uses_bound_source_load_and_no_video_export(combined):
    api = minimal_api(combined=combined)
    api["3"]["class_type"] = source.LOAD
    graph, pins = cold._cold_graph(api, {"path": "frozen/manifest.json", "sha": "a" * 64},
        global_prompt="Same scene", local_prompts="First\nSecond")
    assert graph[pins["store"]]["inputs"] == {
        "artifact_path": "frozen/manifest.json", "artifact_sha256": "a" * 64}
    assert graph[pins["apply"]]["inputs"]["mode"] == "apply_exp"
    assert graph[pins["save"]]["inputs"]["samples"] == [pins["audit"], 0]
    assert not any(node["class_type"] == source.ISOLATED_WRITER for node in graph.values())
    assert ("eav_report" in pins) is combined


@pytest.mark.parametrize("cold_phase", [False, True])
def test_opt_in_media_keeps_candidate_and_adds_original_audio_reference(cold_phase):
    api = minimal_api()
    api["90"] = {"class_type": "MiniMaxH3OutputTrimT8", "inputs": {
        "frames": ["3", 0], "audio": ["3", 1], "fps": ["3", 5]}}
    api["91"] = {"class_type": "CreateVideo", "inputs": {
        "images": ["90", 0], "audio": ["90", 1]}}
    if cold_phase:
        api["3"]["class_type"] = source.LOAD
        graph, pins = cold._cold_graph(api, {"path": "frozen/manifest.json", "sha": "a" * 64},
            global_prompt="Scene", local_prompts="First\nSecond", include_media=True)
    else:
        graph, pins = probe._probe_graph(api, width=128, height=64,
            global_prompt="Scene", local_prompts="First\nSecond", mode="apply_exp",
            include_media=True)
    graph[pins["exporter"]]["inputs"]["filename_prefix"] = "candidate"
    cold._attach_source_reference(graph, pins)
    assert graph[pins["exporter"]]["inputs"]["filename_prefix"] == "candidate"
    assert graph[pins["prep_report"]]["inputs"]["source"] == [pins["store"], 3]
    reference = graph[pins["reference_exporter"]]
    assert reference["class_type"] == source.ISOLATED_WRITER
    assert reference["inputs"]["filename_prefix"] == "source_reference"
    create = graph[reference["inputs"]["video"][0]]
    ref_trim_id = create["inputs"]["images"][0]
    trim = graph[ref_trim_id]
    assert create["inputs"]["audio"] == [ref_trim_id, 1]
    assert trim["inputs"]["frames"] == [pins["store"], 0]
    assert trim["inputs"]["audio"] == [pins["store"], 1]
    assert api["90"]["inputs"]["frames"] == ["3", 0]
