"""Full scope/source binding checks, never sampling or qualification by labels."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from h3_audio_t8_pkg.modular_sampling import catalogue
from h3_audio_t8_pkg.modular_sampling.audio_refine_nodes import NODES as AUDIO_REFINE_NODES
from tools.audit_modular_route_catalogue import audit, resolve_sources, attach_progress, symbols

ROOT = Path(__file__).resolve().parents[1]


def test_all_29_research_routes_are_retained_with_eight_independent_dimensions():
    research = json.loads((ROOT / "artifacts/research/two-pass-modular-audit-20260922/INVENTORY.json").read_text(encoding="utf-8"))
    assert [route.id for route in catalogue.validate_catalogue()][:29] == [row["id"] for row in research["coverage_routes"]]
    assert [route.id for route in catalogue.ROUTES][29:] == ["S30"]
    report = audit(ROOT, catalogue=catalogue)
    assert report["status"] == "pass" and report["route_count"] == 30
    # The catalogue is append-only while routes gain public stages; an exact
    # global count would make the next valid route extension look like damage.
    assert report["source_binding_count"] >= 330
    for row in report["routes"]:
        assert set(row["rollout"]) == set(catalogue.DIMENSIONS)
        assert all(value["status"] == "pending" for value in row["rollout"].values())
        assert row["sources"]
        for source in row["sources"]:
            assert source["line"] <= source["end_line"]
            assert len(source["source_sha256"]) == len(source["symbol_ast_sha256"]) == 64
    assert report["live_node_ids_checked"] is False


def test_handoffs_and_differently_hidden_second_passes_remain_distinct():
    routes = {row.id: row for row in catalogue.ROUTES}
    assert {source.symbol for source in routes["S09"].sources} == {
        "_sample_one_segment", "_sample_prepared_segment", "run_long_video_in_node_loop_effects",
        "resolve_long_video_sample_schedules", "build_stage", "build_noise"}
    # Both runners, including the effects-loop second-pass call site, and real public adapters.
    assert routes["S09"].public_nodes == ("MiniMaxH3ManualPassStageSetupEXPT8", "MiniMaxH3StageNoiseEXPT8")
    assert {source.symbol for source in routes["S29"].sources} == {
        "setup_rectified_flow_restart_sampling", "setup_detail_mixer_sampling", "setup_two_pass_detail_mixer_sampling",
        "prepare_handoff", "build_restart_stage", "RFRestartSampler.sample", "build_base_stage", "handoff", "forward_plan",
        "JointClockInitializedModel.__call__", "MiniMaxH3RFRestartJointClockSetupEXPT8.execute"}
    assert routes["S29"].public_nodes == (
        "MiniMaxH3RFBaseStageSetupEXPT8", "MiniMaxH3RFHandoffEXPT8", "MiniMaxH3RFRestartStageSetupEXPT8",
        "MiniMaxH3RFRestartJointClockSetupEXPT8")
    assert routes["S29"].variants[-1] == "explicit_joint_clock_start"
    assert len(routes["S29"].stages) == 2  # Re-noise is a handoff, not a third diffusion stage.
    assert routes["S17"].repeated_stages and "DCT" in routes["S17"].handoff
    assert routes["S17"].public_nodes == catalogue.SPEED_STAGE_NODES
    assert {"prepare_speed_stage", "sample_speed_stage", "handoff_speed_stage",
            "transition_speed_stage", "save_speed_stage", "load_speed_stage"} <= {
                source.symbol for source in routes["S17"].sources}
    assert "no new HIGH noise" in routes["S13"].handoff
    assert "fresh" in routes["S14"].stages[1] and "prediction_x0" in routes["S15"].handoff
    assert {f"S{i}" for i in range(18, 23)} <= set(routes)
    assert {"MiniMaxH3ChunkedV5WindowSaveEXPT8",
            "MiniMaxH3ChunkedV5WindowLoadEXPT8"} <= set(routes["S22"].public_nodes)
    assert {"verify_window", "save_window", "load_window"} <= {
        source.symbol for source in routes["S22"].sources}
    assert set(catalogue.CHUNKED_V1_LOCAL_RELAY_NODES) <= set(routes["S18"].public_nodes)
    assert set(catalogue.CHUNKED_V1_STORAGE_NODES) <= set(routes["S18"].public_nodes)
    assert set(catalogue.CHUNKED_RELAY_NODES).isdisjoint(routes["S18"].public_nodes)
    assert {"project_v1_local_relay", "assert_v1_local_relay_binding",
            "audit_v1_local_relay", "add_v1_relay_frontend", "add_v1_relay_api",
            "verify_segment", "save_segment", "load_segment", "freeze_frontend",
            "resume_frontend", "freeze_api", "resume_api", "build_storage_pair"} <= {
                source.symbol for source in routes["S18"].sources}
    assert len(routes["S24"].variants) == 7 and len(routes["S24"].sources) == 26
    assert {"bind_standard_face_relay", "bind_local_face_relay",
            "MiniMaxH3FaceRelayBindEXPT8.execute", "MiniMaxH3FaceLocalRelayBindEXPT8.execute"} <= {
                source.symbol for source in routes["S24"].sources}
    assert routes["S24"].public_nodes[-10:-2] == (
        "MiniMaxH3FaceStageBindEXPT8", "MiniMaxH3FaceStageAuditEXPT8",
        "MiniMaxH3FaceParityStageBindEXPT8", "MiniMaxH3FaceParityStageAuditEXPT8",
        "MiniMaxH3MultiFaceStageBindEXPT8", "MiniMaxH3MultiFaceStageAuditEXPT8",
        "MiniMaxH3FaceWindowStageBindEXPT8", "MiniMaxH3FaceWindowStageAuditEXPT8")
    assert routes["S24"].public_nodes[-2:] == (
        "MiniMaxH3FaceRelayBindEXPT8", "MiniMaxH3FaceLocalRelayBindEXPT8")
    assert routes["S25"].public_nodes[-2:] == (
        "MiniMaxH3MotionStageBindEXPT8", "MiniMaxH3MotionStageAuditEXPT8")
    assert "third_stage" in routes["S26"].variants and "audio" in routes["S26"].handoff
    assert routes["S26"].public_nodes[-len(AUDIO_REFINE_NODES)-2:-2] == tuple(
        node.NODE_ID for node in AUDIO_REFINE_NODES)
    assert routes["S26"].public_nodes[-2:] == (
        "MiniMaxH3AudioRefineEffectsBindEXPT8", "MiniMaxH3AudioRefineEffectsGuiderEXPT8")
    assert {"bind_audio_refine_stage", "audit_audio_refine_stage", "split_frontend"} <= {
        source.symbol for source in routes["S26"].sources}
    assert "learned_latent_adapter" in routes["S27"].variants
    assert {"bind_ltx_rgb_stage", "audit_ltx_rgb_stage", "convert_video_latent"} <= {
        source.symbol for source in routes["S27"].sources}
    assert routes["S27"].public_nodes[-15:-8] == (
        "MiniMaxH3LTXRGBStageBindEXPT8", "MiniMaxH3LTXRGBStageAuditEXPT8",
        "MiniMaxH3LTXLearnedStageBindEXPT8", "MiniMaxH3LTXLearnedStageAuditEXPT8",
        "MiniMaxH3LTXOriginalAudioDecodeEXPT8", "MiniMaxH3LTXLearnedStageSampleEXPT8",
        "MiniMaxH3LTXLearnedStageCertifiedAuditEXPT8")
    assert routes["S27"].public_nodes[-8:-6] == (
        "MiniMaxH3LTXRGBSourceSaveEXPT8", "MiniMaxH3LTXRGBSourceLoadEXPT8")
    assert routes["S27"].public_nodes[-6:-4] == (
        "MiniMaxH3LTXEAVApplyEXPT8", "MiniMaxH3LTXEAVAuditEXPT8")
    assert routes["S27"].public_nodes[-4:] == (
        "MiniMaxH3LTXPromptRelayPlanEXPT8", "MiniMaxH3LTXPromptRelayEncodeEXPT8",
        "MiniMaxH3LTXPromptRelayApplyEXPT8", "MiniMaxH3LTXPromptRelayAuditEXPT8")
    assert {"build_ltx_relay_plan", "encode_ltx_relay_conditioning",
            "apply_ltx_relay", "audit_ltx_relay"} <= {
        source.symbol for source in routes["S27"].sources}
    assert {"apply_ltx_eav", "audit_ltx_eav"} <= {source.symbol for source in routes["S27"].sources}
    assert {"save_source", "load_source", "build"} <= {source.symbol for source in routes["S27"].sources}
    assert routes["S28"].public_nodes == (
        "MiniMaxH3PreparedLTXGenerateEXPT8", "MiniMaxH3PreparedLTXLoadGenerationEXPT8",
        "MiniMaxH3PreparedLTXDecodeEXPT8", "MiniMaxH3PreparedLTXEffectsBindEXPT8",
        "MiniMaxH3PreparedLTXRelayPlanEXPT8", "MiniMaxH3PreparedLTXRelayEncodeEXPT8")
    assert {"bind_prepared_ltx_effects", "plan_prepared_ltx_relay", "run_cache_encoding",
            "PreparedLTXEffects.installed", "MiniMaxH3PreparedLTXRelayEncodeEXPT8.execute"} <= {
                source.symbol for source in routes["S28"].sources}
    assert "placeholder" in routes["S23"].handoff
    assert {"h16_window_piece", "bind_h16_eav", "audit_h16_eav", "add_eav_frontend",
            "project_h16_relay", "assert_h16_relay_binding", "audit_h16_relay",
            "add_relay_frontend"} <= {
        source.symbol for source in routes["S23"].sources
    }
    assert {"MiniMaxH3H16EAVApplyEXPT8", "MiniMaxH3H16EAVAuditEXPT8",
            "MiniMaxH3H16RelayProjectEXPT8", "MiniMaxH3H16RelayAuditEXPT8"} <= set(
        routes["S23"].public_nodes
    )


def test_progressive_avatar_public_stages_do_not_claim_continuation():
    routes = {row.id: row for row in catalogue.ROUTES}
    assert routes["S10"].public_nodes == catalogue.PROGRESSIVE_NODES + catalogue.PROGRESSIVE_EFFECT_NODES
    assert len(routes["S10"].public_nodes) == 15
    assert routes["S11"].public_nodes == catalogue.CONTINUATION_NODES
    assert len(routes["S11"].public_nodes) == 12
    assert {"capture_source", "prepare_contexts", "prepare_phase", "sample_low", "prepare_high", "sample_high",
            "deliver", "project_relay", "apply_relay", "apply_eav", "split_graph"} <= {
        source.symbol for source in routes["S11"].sources}
    assert routes["S12"].public_nodes == catalogue.AVATAR_NODES
    assert {"bind_source", "sample_low", "prepare_high", "deliver", "split_graph"} <= {
        source.symbol for source in routes["S12"].sources}
    symbols = {source.symbol for source in routes["S10"].sources}
    assert {"sample_high_result", "save_result", "load_result", "split_graph", "apply_relay", "apply_eav", "execution", "audit"} <= symbols


@pytest.mark.parametrize("change", ["drop", "duplicate", "handoff", "stages", "path", "symbol"])
def test_scope_and_contract_damage_fail_instead_of_being_silently_omitted(change):
    rows = list(catalogue.ROUTES)
    if change == "drop":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = rows[0]
    elif change == "handoff":
        rows[0] = replace(rows[0], handoff="")
    elif change == "stages":
        rows[0] = replace(rows[0], stages=("only_one",))
    else:
        source = replace(rows[0].sources[0], **({"path": "../outside.py"} if change == "path" else {"symbol": "bad()"}))
        rows[0] = replace(rows[0], sources=(source,))
    with pytest.raises(ValueError):
        catalogue.validate_catalogue(tuple(rows))


def test_future_routes_are_append_only_without_rewriting_original_scope():
    next_route = catalogue.Route(
        id="S31", phase="M5", name="Future split route", stages=("low", "high"),
        handoff="typed frozen handoff", variants=("default",),
        sources=(catalogue.Source("h3_t8/modular_sampling/future.py", "sample_high"),),
        public_nodes=("FutureLowStage", "FutureHighStage"),
    )
    assert catalogue.validate_catalogue((*catalogue.ROUTES, next_route))[-1] == next_route
    with pytest.raises(ValueError):
        catalogue.validate_catalogue((*catalogue.ROUTES, replace(next_route, id="S32")))
    with pytest.raises(ValueError):
        catalogue.validate_catalogue((*catalogue.ROUTES[:-1], next_route))


def test_deleted_or_renamed_actual_entry_is_an_explicit_issue_not_an_old_line_number():
    route = catalogue.ROUTES[5]
    missing = replace(route.sources[0], symbol="DualModelSegmentRunner.no_such_stage")
    rows, issues = resolve_sources(ROOT, (replace(route, sources=(missing,)),))
    assert rows[0]["sources"] == []
    assert issues[0]["route"] == "S06" and issues[0]["kind"] == "unresolved_source_symbol"


def test_missing_live_public_node_fails_resolution_without_downgrading_scope():
    report = audit(ROOT, catalogue=catalogue, live_ids=set())
    assert report["status"] == "fail" and report["route_count"] == 30
    assert any(item.get("node") == "MiniMaxH3FastH3V2StageSetupEXPT8" for item in report["issues"])


def test_nested_symbols_are_qualified_and_duplicate_definitions_fail():
    import ast
    tree = ast.parse("class Stage:\n def sample(self):\n  def hidden(): pass\n")
    assert set(symbols(tree)) == {"Stage", "Stage.sample", "Stage.sample.hidden"}
    with pytest.raises(ValueError, match="duplicate"):
        symbols(ast.parse("def sample(): pass\ndef sample(): pass\n"))


def test_partial_evidence_bytes_are_bound_without_promoting_gpu_or_any_other_route(tmp_path):
    proof = tmp_path / "proof.json"
    proof.write_text('{"actual_cpu_check":true}', encoding="utf-8")
    rows = [{"id": route.id} for route in catalogue.ROUTES]
    progress = {"schema": "t8.modular-sampling.progress.v1", "entries": [{"route": "S08",
        "dimension": "public_stages", "status": "partial", "scope": "single segment CPU only", "files": ["proof.json"]}]}
    attach_progress(tmp_path, rows, progress, catalogue.DIMENSIONS)
    before = rows[7]["rollout"]["public_stages"]["evidence"][0]["sha256"]
    assert rows[7]["rollout"]["public_stages"]["status"] == "partial"
    assert all(row["rollout"]["gpu_mechanical"]["status"] == "pending" for row in rows)
    assert rows[0]["rollout"]["public_stages"]["status"] == "pending"
    proof.write_text('{"actual_cpu_check":false}', encoding="utf-8")
    attach_progress(tmp_path, rows, progress, catalogue.DIMENSIONS)
    assert before != rows[7]["rollout"]["public_stages"]["evidence"][0]["sha256"]
    progress["entries"][0]["status"] = "complete"
    with pytest.raises(ValueError, match="self-certify"):
        attach_progress(tmp_path, rows, progress, catalogue.DIMENSIONS)


@pytest.mark.parametrize("bad", ["route", "dimension", "duplicate", "missing_file", "escape"])
def test_invalid_progress_or_missing_evidence_is_not_accepted(tmp_path, bad):
    (tmp_path / "proof.json").write_text("{}", encoding="utf-8")
    item = {"route": "S08", "dimension": "public_stages", "status": "partial", "scope": "CPU", "files": ["proof.json"]}
    if bad == "route":
        item["route"] = "omitted_future"
    elif bad == "dimension":
        item["dimension"] = "green"
    elif bad == "missing_file":
        item["files"] = ["missing.json"]
    elif bad == "escape":
        item["files"] = ["../outside.json"]
    manifest = {"schema": "t8.modular-sampling.progress.v1", "entries": [item] * (2 if bad == "duplicate" else 1)}
    with pytest.raises(ValueError):
        attach_progress(tmp_path, [{"id": "S08"}], manifest, catalogue.DIMENSIONS)
