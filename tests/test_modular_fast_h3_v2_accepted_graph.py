"""Second-window candidate has real independent stage/effect edges, not a hidden loop."""

import json
from pathlib import Path

import pytest

from tools import build_modular_fast_h3_v2_continuation_workflow as builder
from legacy_v2_frontend import assert_legacy_v2_frontend

continuation_graph = builder.continuation_graph


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "artifacts/development/modular-sampling-m1-v2-accepted-continuation-20260925"
DELIVERY_CANDIDATE = ROOT / "artifacts/development/modular-sampling-m1-v2-accepted-final68-20260925"
RECIPE_CANDIDATE = ROOT / "artifacts/development/modular-sampling-m1-v2-current-recipe-second-20260925"
ATTEST_CANDIDATE = ROOT / "artifacts/development/modular-sampling-m1-v2-stage-attest-second-v2-20260925"
ORIGIN_CANDIDATE = ROOT / "artifacts/development/modular-sampling-m1-v2-condition-origin-second-v1-20260925"
HANDOFF_CANDIDATE = ROOT / "artifacts/development/modular-sampling-m1-v2-handoff-origin-second-v1-20260925"
JOB_CANDIDATE = ROOT / "artifacts/development/modular-sampling-m1-v2-job-bound-second-v1-20260925"
SAVE_CANDIDATE = ROOT / "artifacts/development/modular-sampling-m1-v2-candidate-save-second-v1-20260925"
FROZEN_BUNDLE_CANDIDATE = ROOT / "artifacts/development/modular-sampling-m1-v2-frozen-low-second-v1-20260925"
FIRST_IMAGE = "02_清晰身份参考图_首段.png"


def test_full_second_window_uses_accepted_parent_and_exact_old_boundary_order():
    graph = continuation_graph()
    assert len(graph) == 42
    assert graph["77"]["inputs"]["total_accepted_frames"] == 192
    assert graph["9"]["inputs"]["length"] == graph["24"]["inputs"]["length"] == ["77", 0]
    assert graph["74"]["inputs"]["length"] == graph["75"]["inputs"]["length"] == ["77", 0]
    assert graph["72"]["inputs"]["phase"] == "low"
    assert graph["73"]["inputs"]["phase"] == "high"
    assert graph["9"]["inputs"]["context"] == ["72", 0]
    assert graph["24"]["inputs"]["context"] == ["73", 0]
    assert graph["9"]["class_type"] == graph["24"]["class_type"] == \
        "MiniMaxH3PromptRelayLongVideoConditioningT8Advanced"
    assert graph["25"]["class_type"] == "MiniMaxH3FastH3V2AcceptedReconcileEXPT8"
    assert graph["25"]["inputs"]["learned_latent"] == ["23", 0]
    assert graph["76"]["inputs"]["reconciled_av"] == ["25", 0]
    assert graph["26"]["inputs"]["av_latent"] == graph["29"]["inputs"]["latent_image"] == ["76", 0]
    assert graph["10"]["inputs"]["stage"] == "low_0_4"
    assert graph["26"]["inputs"]["stage"] == "high_4_8"
    assert graph["10"]["inputs"]["profile"] == graph["26"]["inputs"]["profile"] == "dense_compat_exp"
    assert graph["43"]["inputs"]["eav_config"] == ["41", 0]
    assert graph["44"]["inputs"]["eav_config"] == ["42", 0]
    assert graph["41"]["inputs"]["mode"] == graph["42"]["inputs"]["mode"] == "disabled"
    assert graph["11"]["inputs"]["noise_seed"] == graph["27"]["inputs"]["noise_seed"] == 2609152202
    assert all(item["class_type"] != "MiniMaxH3FastH3V2DualModelLongVideoEXPT8" for item in graph.values())


def test_frozen_low_second_window_has_no_low_sampler_or_conditions():
    graph = continuation_graph(resume_high=True)
    assert len(graph) == 30
    assert graph["60"]["class_type"] == "MiniMaxH3StageLoadEXPT8"
    assert graph["62"]["inputs"]["low_stage_result"] == ["60", 3]
    assert graph["22"]["inputs"]["completed_stage"] == ["60", 3]
    assert graph["23"]["inputs"]["av_latent"] == ["62", 0]
    assert all(key not in graph for key in ("9", "10", "11", "12", "13", "40", "41", "43", "45", "72", "74"))
    assert graph["24"]["inputs"]["context"] == ["73", 0]
    assert graph["76"]["inputs"]["reconciled_av"] == ["25", 0]


def test_saved_candidate_matches_builder_and_live_core_static_audit():
    audit = json.loads((CANDIDATE / "audit.json").read_text(encoding="utf-8"))
    assert len(audit["candidates"]) == 2
    info = builder.base.load_live_info()
    for resume, label in ((False, "Full"), (True, "Frozen_LOW")):
        path = CANDIDATE / f"FastH3V2_Accepted_Second_{label}.api.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        canonical, _ = builder.base.selected_frontend_schema(continuation_graph(resume_high=resume), info)
        assert saved == canonical
    assert all(row["all_outputs_valid"] and row["core_validation"][0] is True
               and row["core_validation"][3] == {} for row in audit["candidates"])


def test_final_remainder_variant_uses_external_trim_and_never_accepts_automatically():
    full = continuation_graph(with_delivery=True)
    frozen = continuation_graph(resume_high=True, with_delivery=True)
    assert (len(full), len(frozen)) == (45, 33)
    for graph in (full, frozen):
        assert graph["78"]["class_type"] == "MiniMaxH3FastH3V2AcceptedDeliveryWindowEXPT8"
        assert graph["78"]["inputs"] == {"contexts": ["71", 0], "total_accepted_frames": 192}
        assert graph["79"]["class_type"] == "MiniMaxH3OutputTrimT8"
        assert graph["79"]["inputs"]["frames"] == ["14", 0]
        assert graph["79"]["inputs"]["audio"] == ["14", 1]
        assert graph["79"]["inputs"]["start_seconds"] == ["78", 5]
        assert graph["79"]["inputs"]["duration_seconds"] == ["78", 6]
        assert graph["15"]["inputs"]["images"] == ["79", 0]
        assert graph["15"]["inputs"]["audio"] == ["79", 1]
        assert all(item["class_type"] not in {
            "MiniMaxH3LongVideoCandidateSaveT8", "MiniMaxH3LongVideoAcceptCandidateT8",
            "MiniMaxH3LongVideoComposeAcceptedT8", "MiniMaxH3FastH3V2DualModelLongVideoEXPT8"}
            for item in graph.values())
    assert "13" not in frozen and "9" not in frozen


def test_final_remainder_saved_candidate_matches_live_core_static_audit():
    audit = json.loads((DELIVERY_CANDIDATE / "audit.json").read_text(encoding="utf-8"))
    assert len(audit["candidates"]) == 2
    info = builder.base.load_live_info()
    for resume, label in ((False, "Full"), (True, "Frozen_LOW")):
        path = DELIVERY_CANDIDATE / f"FastH3V2_Accepted_Second_{label}_Final68_Unaccepted.api.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        canonical, _ = builder.base.selected_frontend_schema(
            continuation_graph(resume_high=resume, with_delivery=True), info)
        assert saved == canonical
    assert all(row["all_outputs_valid"] and row["core_validation"][0] is True
               and row["core_validation"][3] == {} for row in audit["candidates"])


def test_opt_in_second_current_recipe_binds_actual_inputs_but_does_not_accept():
    graph = continuation_graph(with_delivery=True, with_current_recipe=True, first_frame=FIRST_IMAGE)
    assert len(graph) == 48
    recipe = graph["83"]
    assert recipe["class_type"] == "MiniMaxH3FastH3V2CurrentRecipeEXPT8"
    assert recipe["inputs"]["model_pass1"] == ["1", 0]
    assert recipe["inputs"]["model_pass2"] == ["22", 0]
    assert recipe["inputs"]["first_frame"] == ["82", 0]
    assert recipe["inputs"]["upscale_report_json"] == ["23", 3]
    assert recipe["inputs"]["chain_id"] == ["78", 0]
    assert recipe["inputs"]["low_width"] == ["72", 3]
    assert recipe["inputs"]["width"] == ["73", 3]
    assert all(node["class_type"] not in {
        "MiniMaxH3LongVideoCandidateSaveT8", "MiniMaxH3LongVideoAcceptCandidateT8"}
        for node in graph.values())
    with pytest.raises(ValueError, match="Frozen LOW"):
        continuation_graph(resume_high=True, with_current_recipe=True, first_frame=FIRST_IMAGE)
    audit = json.loads((RECIPE_CANDIDATE / "audit.json").read_text(encoding="utf-8"))
    assert len(audit["candidates"]) == 1 and audit["candidates"][0]["all_outputs_valid"]
    info = builder.base.load_live_info()
    import nodes as core_nodes
    info["LoadImage"] = builder.base.native_info("LoadImage", core_nodes.NODE_CLASS_MAPPINGS["LoadImage"])
    expected, _ = builder.base.selected_frontend_schema(graph, info)
    saved = json.loads((RECIPE_CANDIDATE / "FastH3V2_Accepted_Second_Full_Final68_Unaccepted_CurrentRecipe.api.json")
                       .read_text(encoding="utf-8"))
    assert saved == expected


def test_opt_in_second_stage_attest_audits_both_current_sampler_edges():
    graph = continuation_graph(with_delivery=True, with_current_recipe=True,
                               with_stage_attestation=True, first_frame=FIRST_IMAGE)
    assert len(graph) == 52
    for node, sampler, raw, stage_model, phase in (
            ("85", "13", "1", "43", "low_0_4"),
            ("87", "29", "22", "44", "high_4_8")):
        inputs = graph[node]["inputs"]
        sample = graph[sampler]["inputs"]
        assert graph[node]["class_type"] == "MiniMaxH3FastH3V2CurrentStageAttestEXPT8"
        assert inputs["current_recipe"] == ["83", 0]
        assert inputs["stage_result"] == [sampler, 2]
        assert inputs["raw_model"] == [raw, 0]
        assert inputs["stage_model"] == [stage_model, 0]
        assert inputs["guider"] == sample["guider"]
        assert inputs["source_latent"] == sample["latent_image"]
        assert inputs["phase"] == phase
        assert inputs["segment_index"] == ["78", 1]
    with pytest.raises(ValueError, match="requires the editable recipe"):
        continuation_graph(with_stage_attestation=True)
    audit = json.loads((ATTEST_CANDIDATE / "audit.json").read_text(encoding="utf-8"))
    assert len(audit["candidates"]) == 1 and audit["candidates"][0]["all_outputs_valid"]
    info = builder.base.load_live_info()
    import nodes as core_nodes
    info["LoadImage"] = builder.base.native_info("LoadImage", core_nodes.NODE_CLASS_MAPPINGS["LoadImage"])
    expected, _ = builder.base.selected_frontend_schema(graph, info)
    saved = json.loads((ATTEST_CANDIDATE / "FastH3V2_Accepted_Second_Full_Final68_Unaccepted_CurrentRecipe_StageAttest.api.json")
                       .read_text(encoding="utf-8"))
    assert saved == expected
    _, frontend, _ = builder.build_candidate(
        graph, info, with_delivery=True, with_current_recipe=True,
        with_stage_attestation=True)
    saved_frontend = json.loads((ATTEST_CANDIDATE / "FastH3V2_Accepted_Second_Full_Final68_Unaccepted_CurrentRecipe_StageAttest.json")
                                .read_text(encoding="utf-8"))
    assert_legacy_v2_frontend(saved_frontend, frontend, info)


def test_second_handoff_origin_candidate_binds_parent_and_is_saved_exactly():
    graph = continuation_graph(with_delivery=True, with_current_recipe=True,
        with_stage_attestation=True, with_condition_provenance=True,
        with_handoff_provenance=True, first_frame=FIRST_IMAGE)
    assert len(graph) == 54 and "23" not in graph
    assert graph["91"]["class_type"] == "MiniMaxH3FastH3V2UpscaleProvenanceEXPT8"
    assert graph["92"]["inputs"]["contexts"] == ["71", 0]
    assert graph["92"]["inputs"]["high_source_latent"] == graph["29"]["inputs"]["latent_image"]
    with pytest.raises(ValueError, match="requires condition provenance"):
        continuation_graph(with_handoff_provenance=True)
    saved_audit = json.loads((HANDOFF_CANDIDATE / "audit.json").read_text(encoding="utf-8"))
    assert saved_audit["candidates"][0]["all_outputs_valid"] is True
    assert saved_audit["candidates"][0]["serialization"]["edges"] == 159
    info = builder.base.load_live_info()
    import nodes as core_nodes
    info["LoadImage"] = builder.base.native_info("LoadImage", core_nodes.NODE_CLASS_MAPPINGS["LoadImage"])
    expected, _ = builder.base.selected_frontend_schema(graph, info)
    name = "FastH3V2_Accepted_Second_Full_Final68_Unaccepted_CurrentRecipe_StageAttest_ConditionProvenance_HandoffProvenance"
    assert json.loads((HANDOFF_CANDIDATE / (name + ".api.json")).read_text(encoding="utf-8")) == expected
    _, frontend, _ = builder.build_candidate(graph, info, with_delivery=True,
        with_current_recipe=True, with_stage_attestation=True,
        with_condition_provenance=True, with_handoff_provenance=True)
    saved_frontend = json.loads((HANDOFF_CANDIDATE / (name + ".json")).read_text(encoding="utf-8"))
    assert_legacy_v2_frontend(saved_frontend, frontend, info)


def test_second_current_job_binding_candidate_requires_same_parent_job():
    graph = continuation_graph(with_delivery=True, with_current_recipe=True,
        with_stage_attestation=True, with_condition_provenance=True,
        with_handoff_provenance=True, with_job_binding=True, first_frame=FIRST_IMAGE)
    assert len(graph) == 56
    assert graph["94"]["class_type"] == "MiniMaxH3FastH3V2CurrentJobBindEXPT8"
    assert graph["94"]["inputs"]["contexts"] == ["71", 0]
    assert graph["94"]["inputs"]["handoff_attestation"] == ["92", 0]
    with pytest.raises(ValueError, match="requires handoff provenance"):
        continuation_graph(with_job_binding=True)
    audit = json.loads((JOB_CANDIDATE / "audit.json").read_text(encoding="utf-8"))
    assert audit["candidates"][0]["all_outputs_valid"]
    assert audit["candidates"][0]["serialization"]["edges"] == 164
    info = builder.base.load_live_info()
    import nodes as core_nodes
    info["LoadImage"] = builder.base.native_info("LoadImage", core_nodes.NODE_CLASS_MAPPINGS["LoadImage"])
    expected, _ = builder.base.selected_frontend_schema(graph, info)
    name = "FastH3V2_Accepted_Second_Full_Final68_Unaccepted_CurrentRecipe_StageAttest_ConditionProvenance_HandoffProvenance_CurrentJobBound"
    assert json.loads((JOB_CANDIDATE / (name + ".api.json")).read_text(encoding="utf-8")) == expected
    _, frontend, _ = builder.build_candidate(graph, info, with_delivery=True,
        with_current_recipe=True, with_stage_attestation=True,
        with_condition_provenance=True, with_handoff_provenance=True,
        with_job_binding=True)
    saved_frontend = json.loads((JOB_CANDIDATE / (name + ".json")).read_text(encoding="utf-8"))
    assert_legacy_v2_frontend(saved_frontend, frontend, info)


def test_second_source_bound_candidate_graph_is_opt_in_and_saved_exactly():
    flags = dict(with_delivery=True, with_current_recipe=True, with_stage_attestation=True,
        with_condition_provenance=True, with_handoff_provenance=True,
        with_job_binding=True, with_media_provenance=True, with_candidate_save=True)
    graph = continuation_graph(first_frame=FIRST_IMAGE, **flags)
    assert len(graph) == 58 and "14" not in graph and "79" not in graph
    assert graph["15"]["inputs"]["images"] == ["96", 0]
    assert graph["98"]["inputs"]["media_receipt"] == ["96", 2]
    assert graph["98"]["class_type"] == "MiniMaxH3FastH3V2CurrentCandidateSaveEXPT8"
    with pytest.raises(ValueError, match="requires authenticated current media"):
        continuation_graph(with_candidate_save=True)
    audit = json.loads((SAVE_CANDIDATE / "audit.json").read_text(encoding="utf-8"))
    assert audit["candidates"][0]["all_outputs_valid"]
    assert audit["candidates"][0]["serialization"]["edges"] == 166
    info = builder.base.load_live_info()
    import nodes as core_nodes
    info["LoadImage"] = builder.base.native_info("LoadImage", core_nodes.NODE_CLASS_MAPPINGS["LoadImage"])
    expected, _ = builder.base.selected_frontend_schema(graph, info)
    name = audit["candidates"][0]["name"]
    assert json.loads((SAVE_CANDIDATE / (name + ".api.json")).read_text(encoding="utf-8")) == expected
    _, frontend, _ = builder.build_candidate(graph, info, **flags)
    saved_frontend = json.loads((SAVE_CANDIDATE / (name + ".json")).read_text(encoding="utf-8"))
    assert_legacy_v2_frontend(saved_frontend, frontend, info)


def test_second_frozen_low_bundle_saved_graphs_match_live_builder():
    flags = dict(with_delivery=True, with_current_recipe=True, with_stage_attestation=True,
        with_condition_provenance=True, with_handoff_provenance=True,
        with_job_binding=True, with_media_provenance=True, with_candidate_save=True,
        with_frozen_low_bundle=True)
    audit = json.loads((FROZEN_BUNDLE_CANDIDATE / "audit.json").read_text(encoding="utf-8"))
    assert [(row["serialization"]["nodes"], row["serialization"]["edges"])
            for row in audit["candidates"]] == [(60, 173), (44, 103)]
    assert all(row["all_outputs_valid"] and row["core_validation"][3] == {}
               for row in audit["candidates"])
    info = builder.base.load_live_info()
    import nodes as core_nodes
    info["LoadImage"] = builder.base.native_info("LoadImage", core_nodes.NODE_CLASS_MAPPINGS["LoadImage"])
    for resume, row in zip((False, True), audit["candidates"], strict=True):
        graph = continuation_graph(resume_high=resume, first_frame=FIRST_IMAGE, **flags)
        assert graph["101" if not resume else "102"]["class_type"] == (
            "MiniMaxH3FastH3V2FrozenLOWBundleLoadEXPT8" if resume else
            "MiniMaxH3FastH3V2FrozenLOWBundleSaveEXPT8")
        if resume:
            assert all(key not in graph for key in ("1", "13", "83", "85", "89"))
        expected, _ = builder.base.selected_frontend_schema(graph, info)
        assert json.loads((FROZEN_BUNDLE_CANDIDATE / (row["name"] + ".api.json"))
                          .read_text(encoding="utf-8")) == expected
        _, frontend, _ = builder.build_candidate(graph, info, resume_high=resume, **flags)
        saved_frontend = json.loads((FROZEN_BUNDLE_CANDIDATE / (row["name"] + ".json"))
                                    .read_text(encoding="utf-8"))
        assert_legacy_v2_frontend(saved_frontend, frontend, info)


def test_second_condition_origin_graph_is_append_only_and_saved_exactly():
    graph = continuation_graph(with_delivery=True, with_current_recipe=True,
        with_stage_attestation=True, with_condition_provenance=True, first_frame=FIRST_IMAGE)
    assert len(graph) == 52 and "9" not in graph and "24" not in graph
    for key in ("89", "90"):
        assert graph[key]["class_type"] == "MiniMaxH3FastH3V2ConditionProvenanceEXPT8"
    for audit, conditioner in (("85", "89"), ("87", "90")):
        assert graph[audit]["class_type"] == "MiniMaxH3FastH3V2OriginStageAttestEXPT8"
        assert graph[audit]["inputs"]["condition_receipt"] == [conditioner, 7]
        assert graph[audit]["inputs"]["conditioned_latent"] == [conditioner, 2]
    with pytest.raises(ValueError, match="requires current stage attestation"):
        continuation_graph(with_condition_provenance=True)
    saved_audit = json.loads((ORIGIN_CANDIDATE / "audit.json").read_text(encoding="utf-8"))
    assert saved_audit["candidates"][0]["all_outputs_valid"] is True
    assert saved_audit["candidates"][0]["serialization"]["edges"] == 143
    info = builder.base.load_live_info()
    import nodes as core_nodes
    info["LoadImage"] = builder.base.native_info("LoadImage", core_nodes.NODE_CLASS_MAPPINGS["LoadImage"])
    expected, _ = builder.base.selected_frontend_schema(graph, info)
    name = "FastH3V2_Accepted_Second_Full_Final68_Unaccepted_CurrentRecipe_StageAttest_ConditionProvenance"
    assert json.loads((ORIGIN_CANDIDATE / (name + ".api.json")).read_text(encoding="utf-8")) == expected
    _, frontend, _ = builder.build_candidate(graph, info, with_delivery=True,
        with_current_recipe=True, with_stage_attestation=True, with_condition_provenance=True)
    saved_frontend = json.loads((ORIGIN_CANDIDATE / (name + ".json")).read_text(encoding="utf-8"))
    assert_legacy_v2_frontend(saved_frontend, frontend, info)
