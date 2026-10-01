from tools.build_modular_fast_h3_v2_workflow import (split_graph, split_graph_with_effects,
                                                  split_graph_with_results, resume_high_graph)


def ancestors(graph, key):
    result = set()
    for value in graph[key]["inputs"].values():
        if isinstance(value, list) and len(value) == 2 and str(value[0]) in graph:
            parent = str(value[0])
            result.add(parent)
            result.update(ancestors(graph, parent))
    return result


def test_split_graph_has_two_real_native_samplers_and_independent_editable_branches():
    graph = split_graph()
    assert sum(value["class_type"] == "SamplerCustomAdvanced" for value in graph.values()) == 2
    assert not any("Loop" in value["class_type"] or "LongVideo" in value["class_type"] for value in graph.values())
    assert graph["10"]["inputs"]["stage"] == "low_0_4"
    assert graph["26"]["inputs"]["stage"] == "high_4_8"
    assert graph["10"]["inputs"]["model"] != graph["26"]["inputs"]["model"]
    assert graph["23"]["inputs"]["av_latent"] == ["13", 1]
    assert graph["24"]["inputs"]["width"] == ["23", 1]
    assert graph["24"]["inputs"]["height"] == ["23", 2]
    assert graph["25"]["inputs"]["second_pass_audio_source"] == "legacy_policy"
    assert graph["14"]["inputs"]["av_latent"] == ["30", 0]
    assert "29" in ancestors(graph, "16") and "13" in ancestors(graph, "29")


def test_high_only_edits_have_no_reverse_dependency_into_low_stage():
    graph = split_graph()
    low_dependencies = ancestors(graph, "13") | {"13"}
    assert not {"22", "24", "25", "26", "27", "28", "29"} & low_dependencies
    assert "6" in low_dependencies  # Shared CLIP correctly invalidates LOW.
    assert {"22", "24", "27"} <= ancestors(graph, "29")


def test_external_effects_bind_each_real_stage_without_losing_pairing_or_handoff():
    graph = split_graph_with_effects()
    assert graph["10"]["inputs"]["model"] == ["9", 0]
    assert graph["10"]["inputs"]["av_latent"] == ["9", 2]
    assert graph["12"]["inputs"]["conditioning"] == ["9", 1]
    assert graph["26"]["inputs"]["model"] == ["24", 0]
    assert graph["25"]["inputs"]["highres_template"] == ["24", 2]
    assert graph["25"]["inputs"]["positive"] == ["24", 1]
    assert graph["9"]["inputs"]["prompt_relay_plan"] == ["40", 0]
    assert graph["24"]["inputs"]["prompt_relay_plan"] == ["47", 0]
    assert graph["45"]["inputs"]["av_latent"] == ["13", 1]
    assert graph["46"]["inputs"]["av_latent"] == ["29", 0]
    assert graph["23"]["inputs"]["av_latent"] == ["45", 0]
    assert {"41", "43", "40"} <= ancestors(graph, "13")
    assert not {"42", "44"} & ancestors(graph, "13")
    assert {"42", "44", "24", "47"} <= ancestors(graph, "29")
    assert graph["41"]["inputs"]["mode"] == graph["42"]["inputs"]["mode"] == "report_only"


def test_result_graph_has_exactly_two_one_stage_samplers_and_verified_low_x0_handoff():
    graph = split_graph_with_results()
    assert sum(item["class_type"] == "MiniMaxH3StageSamplerEXPT8" for item in graph.values()) == 2
    assert graph["13"]["inputs"]["stage_context"] == ["10", 3]
    assert graph["29"]["inputs"]["stage_context"] == ["26", 3]
    assert graph["23"]["inputs"]["av_latent"] == ["62", 0]
    assert graph["62"]["inputs"]["low_stage_result"] == ["13", 2]
    assert graph["62"]["class_type"] == "MiniMaxH3FastH3V2CompletedLowX0EXPT8"
    assert graph["50"]["inputs"]["stage_result"] == ["13", 2]
    assert graph["22"]["class_type"] == "MiniMaxH3StageUNETLoaderAfterEXPT8"
    assert graph["22"]["inputs"]["completed_stage"] == ["13", 2]
    assert not {"22", "24", "29", "51"} & ancestors(graph, "50")


def test_recovery_graph_cannot_load_or_execute_low_or_unconnected_low_outputs():
    graph = resume_high_graph("saved-low/manifest.json", "a" * 64)
    assert not {"1", "9", "10", "11", "12", "13", "19", "20", "21", "50"} & set(graph)
    assert sum(item["class_type"] == "MiniMaxH3StageSamplerEXPT8" for item in graph.values()) == 1
    assert graph["23"]["inputs"]["av_latent"] == ["62", 0]
    assert graph["62"]["inputs"]["low_stage_result"] == ["60", 3]
    assert graph["60"]["inputs"]["expected_stage"] == "low_0_4"
    assert graph["22"]["class_type"] == "MiniMaxH3StageUNETLoaderAfterEXPT8"
    assert graph["22"]["inputs"]["completed_stage"] == ["60", 3]
    assert {"60", "62", "23", "24", "25", "29", "51"} <= ancestors(graph, "16")
    assert {item["inputs"].get("stage") for item in graph.values()
            if item["class_type"] == "MiniMaxH3FastH3V2StageSetupEXPT8"} == {"high_4_8"}


def test_effect_result_graph_keeps_verified_low_and_actual_audit_before_handoff():
    original = split_graph_with_effects()
    graph = split_graph_with_results(original)
    assert original["13"]["class_type"] == "SamplerCustomAdvanced"
    assert graph["45"]["inputs"]["av_latent"] == ["62", 0]
    assert graph["46"]["inputs"]["av_latent"] == ["51", 0]
    assert {"43", "13", "62", "45"} <= ancestors(graph, "23")
    assert graph["50"]["inputs"]["stage_result"] == ["13", 2]
    assert {"44", "29", "51", "46"} <= ancestors(graph, "30")
    assert not {"42", "44", "22", "24", "51"} & ancestors(graph, "50")


def test_effect_recovery_retains_only_high_effects_and_real_relay_pair():
    graph = resume_high_graph("completed-low/manifest.json", "a" * 64, with_effects=True)
    assert not {"1", "9", "10", "11", "12", "13", "19", "20", "21", "41", "43", "45", "50"} & set(graph)
    assert "40" not in graph
    assert {"47", "42", "44", "46", "51", "60", "62"} <= set(graph)
    assert graph["26"]["inputs"]["model"] == ["24", 0]
    assert graph["25"]["inputs"]["positive"] == ["24", 1]
    assert graph["46"]["inputs"]["av_latent"] == ["51", 0]
    assert sum(item["class_type"] == "MiniMaxH3StageSamplerEXPT8" for item in graph.values()) == 1
