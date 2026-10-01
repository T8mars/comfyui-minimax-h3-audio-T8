"""P7 segment delivery candidates are separately editable and Core-serializable."""
import asyncio
from copy import deepcopy

import pytest

from tools import build_modular_hyperflow_p7_workflow as builder
from h3_audio_t8_pkg.prompt_relay_advanced import build_prompt_relay_plan
from h3_audio_t8_pkg.prompt_relay_long_video_advanced import project_prompt_relay_plan_to_long_video_window


def ancestors(graph, key):
    found = set()

    def visit(node_id):
        for value in graph[node_id]["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and value[0] in graph:
                if value[0] not in found:
                    found.add(value[0])
                    visit(value[0])

    visit(key)
    return found


@pytest.fixture(scope="module")
def info():
    return builder.load_live_info()


@pytest.mark.parametrize("segment,kind,scope,variant", builder.CASES)
def test_p7_segment_graph_is_importable_and_delivery_is_explicit(info, segment, kind, scope, variant):
    import execution

    graph, workflow, audit = builder.build_candidate(segment, kind, scope, variant, info)
    validation = asyncio.run(execution.validate_prompt("p7-workflow", deepcopy(graph), None))
    expected = {key for key, node in graph.items() if info[node["class_type"]].get("output_node")}
    assert validation[0] and not validation[3] and set(validation[2]) == expected
    assert audit["nodes"] == len(graph)
    assert len(workflow["nodes"]) == len(graph) + 1
    assert graph["28"]["class_type"] == "MiniMaxH3HyperFlowP7CandidateSaveEXPT8"
    assert graph["29"]["class_type"] == "MiniMaxH3HyperFlowP7CandidateAcceptEXPT8"
    assert graph["29"]["inputs"] == {"candidate_json_path": ["28", 0],
                                       "job_sha256": ["28", 2], "accept_candidate": False}
    assert graph["28"]["inputs"]["high_result"] == (["26", 0] if variant == "minimal"
                                                       else ["27", 0])
    assert graph["28"]["inputs"]["is_final_segment"] == (segment == "segment1")
    assert graph["28"]["inputs"]["final_frame_count"] == (68 if segment == "segment1" else 0)
    segment_seed = 123456790 if segment == "segment1" else 123456789
    assert graph["28"]["inputs"]["seed"] == segment_seed
    assert graph["23"]["inputs"]["noise_seed"] == segment_seed + 1
    assert graph["20"]["inputs"]["length"] == 124
    assert graph["12"]["inputs"]["length"] == 124
    assert graph["11"]["inputs"]["model"] == ["50", 0]
    if variant == "resume_high":
        assert "10" not in graph
    else:
        assert graph["10"]["inputs"]["model"] == ["1", 0]
    assert graph["35"]["class_type"] == "MiniMaxH3HyperFlowP7LearnedLiftEXPT8"
    assert graph["21"]["class_type"] == "MiniMaxH3HyperFlowP7HighHandoffEXPT8"
    assert graph["22"]["class_type"] == "MiniMaxH3HyperFlowP7HighSetupEXPT8"
    assert graph["25"]["inputs"]["noise"] == ["23", 0]
    assert graph["25"]["inputs"]["stage_context"] == ["22", 4]
    assert graph["26"]["inputs"]["stage_result"] == ["25", 2]
    if segment == "segment0":
        assert graph["2"]["class_type"] == "MiniMaxH3HyperFlowP7InitialSegmentEXPT8"
        assert "3" not in graph
    else:
        assert graph["2"]["class_type"] == "MiniMaxH3HyperFlowP7AcceptedParentEXPT8"
        assert graph["3"]["class_type"] == "MiniMaxH3HyperFlowP7PrepareContextsEXPT8"
        assert graph["2"]["inputs"]["parent_candidate_id"] == builder.PARENT_ID_PLACEHOLDER
        assert graph["2"]["inputs"]["previous_job_sha256"] == builder.PARENT_SHA_PLACEHOLDER
    if variant == "resume_high":
        assert not {"1", "10", "14", "15", "16", "17", "18", "32", "33", "34"} & set(graph)
        low_relay_source = kind in ("relay", "combined") and scope in ("low", "both")
        assert ("30" in graph) == low_relay_source
        assert ("31" in graph) == low_relay_source
        if low_relay_source:
            assert graph["12"]["inputs"]["prompt"] == ["31", 1]
        assert graph["19"]["class_type"] == "MiniMaxH3HyperFlowP7LowLoadEXPT8"
        assert graph["19"]["inputs"]["artifact_path"] == builder.LOW_PATH_PLACEHOLDER
        assert graph["19"]["inputs"]["artifact_sha256"] == builder.PARENT_SHA_PLACEHOLDER
        assert graph["35"]["inputs"]["low_result"] == ["19", 0]
    else:
        assert {"1", "10", "14", "15", "16", "17", "18"} <= set(graph)
        assert graph["18"]["inputs"]["stage_result"] == ["17", 2]
        assert graph["35"]["inputs"]["low_result"] == (["19", 0] if variant == "save"
                                                      else ["18", 0])
        assert not {"50", "11", "20", "21", "22", "23", "24", "25", "26", "27",
                    "40", "41", "42", "43", "44"} & ancestors(graph, "17")
    if variant == "save":
        assert graph["19"]["class_type"] == "MiniMaxH3HyperFlowP7LowSaveEXPT8"
    assert ("27" in graph) == (variant in ("save", "resume_high"))


@pytest.mark.parametrize("segment,kind,scope,variant", builder.CASES)
def test_p7_effect_ports_only_on_selected_live_phase(segment, kind, scope, variant):
    graph = builder.split_graph(segment, kind, scope, variant)
    for phase, setup, guider, plan, project, relay, config, apply in (
        ("low", "14", "16", "30", "31", "32", "33", "34"),
        ("high", "22", "24", "40", "41", "42", "43", "44"),
    ):
        selected = scope in (phase, "both")
        active = selected and (variant != "resume_high" or phase == "high")
        expect_plan = selected and kind in ("relay", "combined")
        expect_relay = active and kind in ("relay", "combined")
        expect_eav = active and kind in ("eav", "combined")
        assert (plan in graph) == expect_plan
        assert (project in graph) == expect_plan
        assert (relay in graph) == expect_relay
        assert (config in graph) == expect_eav
        assert (apply in graph) == expect_eav
        condition = "12" if phase == "low" else "20"
        if phase == "low" and variant == "resume_high":
            if expect_plan:
                assert graph["12"]["inputs"]["prompt"] == [project, 1]
            continue
        assert graph[guider]["inputs"]["model"] == ([apply, 0] if expect_eav else [setup, 0])
        if expect_relay:
            assert graph[condition]["inputs"]["prompt"] == [project, 1]
            assert graph[project]["inputs"]["accepted_end_frame"] == (192 if segment == "segment1" else 124)
            assert graph[relay]["inputs"]["prepared_phase"] == [condition, 0]
            assert graph[relay]["inputs"]["projected_relay"] == [project, 0]
            assert graph[relay]["inputs"]["model"] == (["10", 0] if phase == "low"
                                                       else ["11", 0])
        if expect_eav:
            assert graph[apply]["class_type"] == "MiniMaxH3HyperFlowP7StageEAVApplyEXPT8"
            assert graph[apply]["inputs"]["av_latent"] == [setup, 3]
            assert graph[apply]["inputs"]["stage_context"] == [setup, 4]
            assert graph[apply]["inputs"]["prepared_phase"] == [condition, 0]
            assert graph[config]["inputs"]["mode"] == "report_only"


def test_p7_graph_case_inventory_and_invalid_variants():
    assert len(builder.CASES) == 60
    assert len(set(builder.CASES)) == len(builder.CASES)
    for segment in builder.SEGMENTS:
        for kind, scope in builder.EFFECTS:
            assert (segment, kind, scope, "minimal") in builder.CASES
            assert (segment, kind, scope, "save") in builder.CASES
        for kind, scope in builder.RESUME_EFFECTS:
            assert (segment, kind, scope, "resume_high") in builder.CASES
    with pytest.raises(ValueError, match="Unknown P7"):
        builder.split_graph("segment2", "none", "both", "minimal")
    with pytest.raises(ValueError, match="Unknown P7"):
        builder.split_graph("segment0", "relay", "invalid", "resume_high")


@pytest.mark.parametrize("segment,start,end,context", [
    ("segment0", 0, 124, 0), ("segment1", 124, 192, 22),
])
def test_p7_default_relay_has_two_live_events_in_each_segment(segment, start, end, context):
    graph = builder.split_graph(segment, "combined", "both", "save")
    for key in ("30", "40"):
        plan = build_prompt_relay_plan(**graph[key]["inputs"])[0]
        projected = project_prompt_relay_plan_to_long_video_window(
            plan, int(segment[-1]), 124, context, start / 24, end / 24)[0]
        window = projected["long_video_projection"]
        assert len(plan["events"]) == 4
        assert len(window["render_active_event_indices"]) >= 2
        assert len(window["accepted_active_event_indices"]) >= 2
