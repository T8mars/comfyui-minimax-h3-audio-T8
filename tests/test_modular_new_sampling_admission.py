"""Future multi-stage sampler callsites cannot silently bypass split-route review."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from h3_audio_t8_pkg.modular_sampling import catalogue
from tools.audit_modular_new_sampling import SCHEMA, audit, discover, guard, reviewed_layout_sites

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests/fixtures/modular_new_sampling_baseline_v10.json"
PREVIOUS_RADAR_BASELINE = ROOT / "tests/fixtures/modular_new_sampling_baseline_v9.json"
PREVIOUS_ENCODER_BASELINE = ROOT / "tests/fixtures/modular_new_sampling_baseline_v8.json"
PREVIOUS_SAFETY_BASELINE = ROOT / "tests/fixtures/modular_new_sampling_baseline_v7.json"
PRIVATE_BASELINE = ROOT / "artifacts/development/modular-sampling-m5-admission-20260924/baseline-v7-director-stage-relay.json"
PREVIOUS_RELAY_BASELINE = ROOT / "tests/fixtures/modular_new_sampling_baseline_v6.json"
PREVIOUS_DIRECTOR_BASELINE = ROOT / "tests/fixtures/modular_new_sampling_baseline_v4.json"
PREVIOUS_BASELINE = ROOT / "artifacts/development/modular-sampling-m5-admission-20260924/baseline-v2.json"


def _baseline(root):
    return {"schema": SCHEMA, "scope": "synthetic test", "sites": discover(root)}


def _admissions(*entries):
    return {"schema": SCHEMA, "entries": list(entries)}


def test_freevideo_finite_reviewed_sites_and_both_independent_resume_topologies():
    from tools.audit_modular_new_sampling import _admission_issues
    review = json.loads((ROOT / "tests/fixtures/freevideo_reviewed_sampling_admission_1_91.json").read_text(encoding="utf8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf8"))
    assert len(review["entries"]) == 3
    sites = [entry["site"] for entry in review["entries"]]
    assert len({tuple(site) for site in sites}) == 3
    assert [list(site.values()) for site in baseline["sites"][-3:]] == sites
    assert not any("freevideo_exp/" in site["path"] for site in baseline["sites"][:-3])
    live_ids = set(json.loads((ROOT / "features.json").read_text(encoding="utf8"))["nodes"])
    entries = [review["entries"][0], review["additional_qualified_route"]]
    routes = {entry["route"]: SimpleNamespace(public_nodes=tuple(live_ids)) for entry in entries}
    for entry in entries:
        assert _admission_issues({}, entry, ROOT, routes, live_ids) == []
        broken = {**entry, "external_effect_nodes": {}}
        assert any(issue["kind"] == "missing_independent_effect_nodes" for issue in
                   _admission_issues({}, broken, ROOT, routes, live_ids))


def test_current_production_sampler_sites_match_frozen_review_baseline():
    baseline = json.loads(BASELINE.read_text(encoding="utf8"))
    before_encoder = json.loads(PREVIOUS_ENCODER_BASELINE.read_text(encoding="utf8"))
    before_safety = json.loads(PREVIOUS_SAFETY_BASELINE.read_text(encoding="utf8"))
    if PRIVATE_BASELINE.is_file():
        assert before_safety == json.loads(PRIVATE_BASELINE.read_text(encoding="utf8"))
    def changed_safety(site):
        return (site["path"], site["symbol"]) == (
            "h3_t8/director_generation.py", "build_director_generation_prompt")
    assert [site for site in before_safety["sites"] if not changed_safety(site)] == [
        site for site in before_encoder["sites"] if not changed_safety(site)]
    assert sum(changed_safety(site) for site in before_safety["sites"]) == 1
    assert sum(changed_safety(site) for site in baseline["sites"]) == 1
    report = audit(ROOT, baseline, _admissions(), {route.id: route for route in catalogue.ROUTES})
    assert report["status"] == "pass"
    effective, omitted = reviewed_layout_sites(ROOT, baseline)
    assert len(baseline["sites"]) == report["declared_baseline_sites"] == 121
    assert report["baseline_sites"] == report["current_sites"] == len(effective)
    assert report["reviewed_absent_root_alias_sites"] == len(omitted)
    assert report["new_sites"] == report["removed_sites"] == 0
    before_stage_relay = json.loads(PREVIOUS_RELAY_BASELINE.read_text(encoding="utf8"))
    def unchanged(site):
        return (site["path"], site["symbol"]) != (
            "h3_t8/director_hyperflow.py", "apply_hyperflow_graph")
    assert [site for site in before_stage_relay["sites"] if unchanged(site)] == [
        site for site in before_safety["sites"] if unchanged(site)]
    assert sum(not unchanged(site) for site in before_stage_relay["sites"]) == 1
    assert sum(not unchanged(site) for site in before_safety["sites"]) == 1
    if PREVIOUS_BASELINE.is_file():
        previous = json.loads(PREVIOUS_BASELINE.read_text(encoding="utf8"))
        assert all(site in before_safety["sites"] for site in previous["sites"])
    before_director = json.loads(PREVIOUS_DIRECTOR_BASELINE.read_text(encoding="utf8"))
    def changed(site):
        return (site["path"], site["symbol"]) in {
            ("h3_t8/director_generation.py", "build_director_generation_prompt"),
            ("h3_t8/director_hyperflow.py", "apply_hyperflow_graph"),
        }
    assert [site for site in before_director["sites"] if not changed(site)] == [
        site for site in before_safety["sites"] if not changed(site)]
    assert sum(changed(site) for site in before_director["sites"]) == 2
    assert sum(changed(site) for site in before_safety["sites"]) == 2
    assert {(site["path"], site["symbol"]) for site in baseline["sites"]
            if site["callee"] == "<director_graph_builder>"} == {
                ("h3_t8/director_generation.py", "build_director_generation_prompt"),
                ("h3_t8/director_two_pass.py", "apply_two_pass_graph"),
                ("h3_t8/director_hyperflow.py", "apply_hyperflow_graph"),
            }


def test_v9_preserves_every_v8_site_and_only_reviews_the_two_explicit_encoder_calls():
    previous = json.loads(PREVIOUS_ENCODER_BASELINE.read_text(encoding="utf8"))
    current = json.loads(PREVIOUS_RADAR_BASELINE.read_text(encoding="utf8"))
    added = [site for site in current["sites"] if site not in previous["sites"]]
    assert len(previous["sites"]) == 103
    assert [site for site in current["sites"] if site not in added] == previous["sites"]
    assert added == [
        {"path": "h3_t8/modular_sampling/prepared_ltx_relay_cache.py", "symbol": "run_cache_encoding",
         "callee": "reader.sample", "call_sha256": "d8694c5a6d380d3314004292ccd5a0064726c070fbb9fac377ed0c4f664b7a62"},
        {"path": "h3_t8/modular_sampling/prepared_ltx_relay_cache.py", "symbol": "run_cache_encoding",
         "callee": "run_worker", "call_sha256": "04d4403995ab9b2da9b3b084dc2cb51a5dd56e76b52513e6f1f837006c61be22"}]
    from h3_audio_t8_pkg.modular_sampling import prepared_ltx_relay_cache as encoder
    assert encoder.SPEC == ("ltx_relay_cache_worker.py", "native_INT8_event_caches_prepared_not_sampling",
                            "encoded-contexts.safetensors", 3600)
    # Neither an NVML resource sample nor the fixed text-only Job is diffusion.
    # Do not whitelist all sample/run_worker calls or change the AST scanner.
    worker = (ROOT / "h3_t8/prepared_backend/ltx_relay_cache_worker.py").read_text(encoding="utf8")
    import ast
    calls = [ast.unparse(node.func) for node in ast.walk(ast.parse(worker)) if isinstance(node, ast.Call)]
    assert "gemma.encode" in calls and "connector.process_hidden_states" in calls
    assert not any(name.rsplit(".", 1)[-1] in {"sample", "sample_custom", "sample_stage"} for name in calls)


def test_v10_preserves104_sites_and_only_explicit_reviewed_delta_and_real_curve_admission():
    from tools.audit_modular_sampling_compat import capture
    previous = json.loads(PREVIOUS_RADAR_BASELINE.read_text(encoding="utf8"))
    current = json.loads(BASELINE.read_text(encoding="utf8"))
    review = json.loads((ROOT / "tests/fixtures/modular_new_sampling_review_v10.json").read_text(encoding="utf8"))
    def key(site):
        return [site[name] for name in ("path", "symbol", "callee", "call_sha256")]
    # Preserve the historical v10 delta; the finite appended FreeVideo delta
    # is independently checked above, not treated as unreviewed curve work.
    historical = current["sites"][:118]
    added = [key(site) for site in historical if site not in previous["sites"]]
    removed = [key(site) for site in previous["sites"] if site not in current["sites"]]
    assert added == review["reviewed_added_sites"] and len(added) == 14
    assert removed == [review["removed_site"]] and removed[0][:3] == [
        "h3_t8/mv_lipsync_advanced.py", "run_local_mv_in_node_loop", "_sample_one_segment"]
    assert len([site for site in previous["sites"] if site in current["sites"]]) == 104
    admissions = review["curve_admissions"]
    assert len(admissions["entries"]) == 3
    curve_sites = {tuple(entry["site"]) for entry in admissions["entries"]}
    assert {site[0] for site in curve_sites} == {
        "h3_t8/modular_sampling/hyperflow_curve.py", "h3_t8/nodes_hyperflow_curve_exp.py"}
    before_curve = {**current, "sites": [site for site in current["sites"] if tuple(key(site)) not in curve_sites]}
    live = {node["id"] for node in capture()["nodes"]}
    assert len(live) == 632
    report = audit(ROOT, before_curve, admissions, {route.id: route for route in catalogue.ROUTES}, live)
    assert report["status"] == "pass" and report["new_sites"] == 3
    effective, _ = reviewed_layout_sites(ROOT, before_curve)
    assert report["removed_sites"] == 0 and report["baseline_sites"] == len(effective)
    assert report["declared_baseline_sites"] == 118
    assert audit(ROOT, before_curve, _admissions(), {})["status"] == "fail"


def test_future_head_tail_and_native_calls_are_not_hidden_outside_multistage_names(tmp_path):
    source = tmp_path / "h3_t8" / "ordinary_name.py"
    source.parent.mkdir()
    previous = _baseline(tmp_path)
    source.write_text("def ordinary(x):\n x.sample_head(1)\n x.sample_tail(2)\n x._native_stage(3)\n", encoding="utf8")
    assert {site["callee"] for site in discover(tmp_path)} == {"x.sample_head", "x.sample_tail", "x._native_stage"}
    report = guard(tmp_path, previous)
    assert report["status"] == "fail" and report["new_sites"] == 3


def test_ci_guard_needs_only_python_stdlib_and_rejects_new_site(tmp_path):
    baseline = json.loads(BASELINE.read_text(encoding="utf8"))
    result = subprocess.run([sys.executable, "-S", str(ROOT / "tools/audit_modular_new_sampling.py"),
                             "guard"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    effective, _ = reviewed_layout_sites(ROOT, baseline)
    assert json.loads(result.stdout)["current_sites"] == len(effective)
    source = tmp_path / "h3_t8" / "new_two_pass.py"
    source.parent.mkdir()
    synthetic = _baseline(tmp_path)
    source.write_text("def run_two_pass(sampler):\n return sampler.sample(1)\n", encoding="utf8")
    report = guard(tmp_path, synthetic)
    assert report["status"] == "fail" and report["new_sites"] == 1


def test_new_two_pass_sampler_requires_registered_split_graph_effects_and_resume(tmp_path):
    source = tmp_path / "h3_t8" / "new_two_pass.py"
    source.parent.mkdir()
    baseline = _baseline(tmp_path)
    source.write_text('def run_two_pass(sampler):\n for stage in ("low", "high"):\n  sampler.sample(stage)\n', encoding="utf8")
    public = {"LowStage", "HighStage", "LowEAV", "HighEAV",
              "LowRelay", "HighRelay"}
    routes = {"S30": SimpleNamespace(public_nodes=tuple(sorted(public)))}
    missing = audit(tmp_path, baseline, _admissions(), routes)
    assert missing["status"] == "fail" and missing["new_sites"] == 1
    assert missing["issues"][0]["kind"] == "new_sampling_site_requires_admission"
    site = discover(tmp_path)[0]
    entry = {"site": [site[key] for key in ("path", "symbol", "callee", "call_sha256")],
             "route": "S30", "variant": "native_low_high", "stage_nodes": ["LowStage", "HighStage"],
             "workflow": "full.json", "resume_workflow": "resume.json",
             "workflow_stage_ids": [1, 2], "resume_source_id": 3,
             "resume_source_type": "StageLoad", "resume_stage_id": 2,
             "external_effects": {"eav": "supported", "prompt_relay": "supported"},
             "external_effect_nodes": {
                 "workflow": {"eav": [10, 11], "prompt_relay": [12, 13]},
                 "resume_workflow": {"eav": 11, "prompt_relay": 13},
             }}
    unproven = audit(tmp_path, baseline, _admissions(entry), routes, live_ids={"LowStage"})
    assert {issue["kind"] for issue in unproven["issues"]} == {
        "stage_not_registered", "missing_or_invalid_workflow", "missing_or_invalid_resume_workflow"}
    full = {"nodes": [
        {"id": 1, "type": "LowStage"}, {"id": 2, "type": "HighStage"},
        {"id": 10, "type": "LowEAV"}, {"id": 11, "type": "HighEAV"},
        {"id": 12, "type": "LowRelay"}, {"id": 13, "type": "HighRelay"}],
        "links": [[1, 1, 0, 2, 0, "STAGE"], [2, 10, 0, 1, 0, "MODEL"],
                  [3, 11, 0, 2, 1, "MODEL"], [4, 12, 0, 1, 1, "CONDITIONING"],
                  [5, 13, 0, 2, 2, "CONDITIONING"]]}
    resume = {"nodes": [
        {"id": 3, "type": "StageLoad"}, {"id": 2, "type": "HighStage"},
        {"id": 11, "type": "HighEAV"}, {"id": 13, "type": "HighRelay"}],
        "links": [[1, 3, 0, 2, 0, "STAGE"], [2, 11, 0, 2, 1, "MODEL"],
                  [3, 13, 0, 2, 2, "CONDITIONING"]]}
    (tmp_path / "full.json").write_text(json.dumps(full), encoding="utf8")
    (tmp_path / "resume.json").write_text(json.dumps(resume), encoding="utf8")
    accepted = audit(tmp_path, baseline, _admissions(entry), routes,
                     live_ids=public | {"StageLoad"})
    assert accepted["status"] == "pass" and not accepted["issues"]
    entry["external_effect_nodes"]["workflow"]["eav"] = [10, 10]
    shared_effect = audit(tmp_path, baseline, _admissions(entry), routes,
                          live_ids=public | {"StageLoad"})
    assert "missing_independent_effect_nodes" in {
        issue["kind"] for issue in shared_effect["issues"]}
    entry["external_effect_nodes"]["workflow"]["eav"] = [10, 11]
    full["nodes"][2]["type"] = "UnregisteredEffect"
    (tmp_path / "full.json").write_text(json.dumps(full), encoding="utf8")
    wrong_node = audit(tmp_path, baseline, _admissions(entry), routes,
                       live_ids=public | {"StageLoad"})
    assert "invalid_eav_topology_workflow" in {
        issue["kind"] for issue in wrong_node["issues"]}
    full["nodes"][2]["type"] = "LowEAV"
    full["links"] = [link for link in full["links"] if link[1] != 11]
    (tmp_path / "full.json").write_text(json.dumps(full), encoding="utf8")
    disconnected = audit(tmp_path, baseline, _admissions(entry), routes,
                         live_ids=public | {"StageLoad"})
    assert "invalid_eav_topology_workflow" in {issue["kind"] for issue in disconnected["issues"]}
    (tmp_path / "full.json").write_text(json.dumps({**full, "links": [
        *full["links"], [3, 11, 0, 2, 1, "MODEL"]]}), encoding="utf8")
    resume["links"] = [link for link in resume["links"] if link[1] != 13]
    (tmp_path / "resume.json").write_text(json.dumps(resume), encoding="utf8")
    disconnected = audit(tmp_path, baseline, _admissions(entry), routes,
                         live_ids=public | {"StageLoad"})
    assert "invalid_prompt_relay_topology_resume_workflow" in {
        issue["kind"] for issue in disconnected["issues"]}
    (tmp_path / "resume.json").write_text(json.dumps({**resume, "links": [
        *resume["links"], [3, 13, 0, 2, 2, "CONDITIONING"]]}), encoding="utf8")
    entry.pop("external_effect_nodes")
    missing_effect_graph = audit(tmp_path, baseline, _admissions(entry), routes,
                                 live_ids=public | {"StageLoad"})
    assert "missing_independent_effect_nodes" in {
        issue["kind"] for issue in missing_effect_graph["issues"]}
    entry["external_effect_nodes"] = {
        "workflow": {"eav": [10, 11], "prompt_relay": [12, 13]},
        "resume_workflow": {"eav": 11, "prompt_relay": 13},
    }
    entry["external_effects"]["prompt_relay"] = "pending"
    assert "external_effects_not_qualified" in {issue["kind"] for issue in
        audit(tmp_path, baseline, _admissions(entry), routes)["issues"]}


def test_same_public_stage_type_needs_two_connected_instances_and_one_resume_instance(tmp_path):
    source = tmp_path / "h3_t8" / "new_two_pass.py"
    source.parent.mkdir()
    baseline = _baseline(tmp_path)
    source.write_text("def run_two_pass(sampler):\n return sampler.sample(1)\n", encoding="utf8")
    site = discover(tmp_path)[0]
    entry = {"site": [site[key] for key in ("path", "symbol", "callee", "call_sha256")],
             "route": "S30", "variant": "two_stage", "stage_nodes": ["StageSampler", "StageSampler"],
             "workflow": "full.json", "resume_workflow": "resume.json",
             "workflow_stage_ids": [9, 23], "resume_source_id": 4,
             "resume_source_type": "StageLoad", "resume_stage_id": 23,
             "external_effects": {"eav": "supported", "prompt_relay": "supported"},
             "external_effect_nodes": {
                 "workflow": {"eav": [10, 11], "prompt_relay": [12, 13]},
                 "resume_workflow": {"eav": 11, "prompt_relay": 13},
             }}
    routes = {"S30": SimpleNamespace(public_nodes=(
        "StageSampler", "LowEAV", "HighEAV", "LowRelay", "HighRelay"))}
    (tmp_path / "full.json").write_text(json.dumps({"nodes": [
        {"id": 9, "type": "StageSampler"}, {"id": 23, "type": "StageSampler"},
        {"id": 10, "type": "LowEAV"}, {"id": 11, "type": "HighEAV"},
        {"id": 12, "type": "LowRelay"}, {"id": 13, "type": "HighRelay"}],
        "links": []}), encoding="utf8")
    (tmp_path / "resume.json").write_text(json.dumps({"nodes": [
        {"id": 4, "type": "StageLoad"}, {"id": 23, "type": "StageSampler"},
        {"id": 11, "type": "HighEAV"}, {"id": 13, "type": "HighRelay"}],
        "links": [[1, 4, 0, 23, 0, "STAGE"], [2, 11, 0, 23, 1, "MODEL"],
                  [3, 13, 0, 23, 2, "CONDITIONING"]]}), encoding="utf8")
    disconnected = audit(tmp_path, baseline, _admissions(entry), routes,
                         live_ids={"StageSampler", "StageLoad", "LowEAV", "HighEAV",
                                   "LowRelay", "HighRelay"})
    assert "invalid_stage_topology_workflow" in {issue["kind"] for issue in disconnected["issues"]}
    (tmp_path / "full.json").write_text(json.dumps({"nodes": [
        {"id": 9, "type": "StageSampler"}, {"id": 23, "type": "StageSampler"},
        {"id": 10, "type": "LowEAV"}, {"id": 11, "type": "HighEAV"},
        {"id": 12, "type": "LowRelay"}, {"id": 13, "type": "HighRelay"}],
        "links": [[1, 9, 0, 23, 0, "STAGE"], [2, 10, 0, 9, 1, "MODEL"],
                  [3, 11, 0, 23, 1, "MODEL"], [4, 12, 0, 9, 2, "CONDITIONING"],
                  [5, 13, 0, 23, 2, "CONDITIONING"]]}), encoding="utf8")
    assert audit(tmp_path, baseline, _admissions(entry), routes,
                 live_ids={"StageSampler", "StageLoad", "LowEAV", "HighEAV",
                           "LowRelay", "HighRelay"})["status"] == "pass"
    (tmp_path / "resume.json").write_text(json.dumps({"nodes": [
        {"id": 4, "type": "StageLoad"}, {"id": 23, "type": "StageSampler"},
        {"id": 99, "type": "StageSampler"}, {"id": 11, "type": "HighEAV"},
        {"id": 13, "type": "HighRelay"}],
        "links": [[1, 4, 0, 23, 0, "STAGE"], [2, 11, 0, 23, 1, "MODEL"],
                  [3, 13, 0, 23, 2, "CONDITIONING"]]}), encoding="utf8")
    assert "invalid_stage_topology_resume_workflow" in {issue["kind"] for issue in
        audit(tmp_path, baseline, _admissions(entry), routes,
              live_ids={"StageSampler", "StageLoad", "LowEAV", "HighEAV",
                        "LowRelay", "HighRelay"})["issues"]}
    (tmp_path / "resume.json").write_text(json.dumps({"nodes": [
        {"id": 4, "type": "StageLoad"}, {"id": 23, "type": "StageSampler"}],
        "links": [[1, 4, 0, 23, 0, "STAGE"]]}), encoding="utf8")
    entry["resume_source_id"] = 23
    assert "invalid_stage_topology_resume_workflow" in {issue["kind"] for issue in
        audit(tmp_path, baseline, _admissions(entry), routes,
              live_ids={"StageSampler", "StageLoad", "LowEAV", "HighEAV",
                        "LowRelay", "HighRelay"})["issues"]}


def test_real_fast_v2_candidate_uses_two_instances_of_one_public_stage_type():
    from tools.audit_modular_new_sampling import _reaches, _workflow_graph

    # Preserve the published graph fixture: a checkout must not need private
    # candidate artifacts for this public topology assertion.
    base = "examples/workflows/34-fasth3-v2/"
    full, full_edges = _workflow_graph(ROOT, base +
        "FastH3_V2_Split_01_First_LOW_HIGH_Relay_EAV_EXP.json")
    resume, resume_edges = _workflow_graph(ROOT, base +
        "FastH3_V2_Split_02_First_Cold_HIGH_EXP.json")
    stage = "MiniMaxH3FastH3V2StageSetupEXPT8"
    assert full[5] == full[13] == resume[8] == stage
    assert _reaches(full_edges, 5, 13)
    assert resume[40] == "MiniMaxH3StageLoadEXPT8"
    assert _reaches(resume_edges, 40, 8)
    assert sum(kind == stage for kind in resume.values()) == 1


def test_rewriting_existing_sampling_site_and_stale_admission_fail(tmp_path):
    source = tmp_path / "h3_t8" / "two_pass.py"
    source.parent.mkdir()
    source.write_text("def run_two_pass(sampler):\n return sampler.sample(1)\n", encoding="utf8")
    baseline = _baseline(tmp_path)
    source.write_text("def run_two_pass(sampler):\n return sampler.sample(2)\n", encoding="utf8")
    report = audit(tmp_path, baseline, _admissions(), {})
    kinds = {issue["kind"] for issue in report["issues"]}
    assert report["status"] == "fail" and report["new_sites"] == report["removed_sites"] == 1
    assert {"existing_sampling_site_changed_or_removed", "new_sampling_site_requires_admission"} <= kinds
    old_site = baseline["sites"][0]
    stale = {"site": [old_site[key] for key in ("path", "symbol", "callee", "call_sha256")],
             "route": "S30"}
    assert "stale_or_unneeded_admission" in {issue["kind"] for issue in
        audit(tmp_path, baseline, _admissions(stale), {})["issues"]}


def test_new_director_dynamic_sampler_requires_admission(tmp_path):
    path = tmp_path / "h3_t8" / "director_new.py"
    path.parent.mkdir()
    baseline = _baseline(tmp_path)
    path.write_text('def apply_new_graph(graph):\n graph["9"] = '
                    '{"class_type": "SamplerCustomAdvanced", "inputs": {}}\n', encoding="utf8")
    sites = discover(tmp_path)
    assert len(sites) == 1
    assert sites[0]["callee"] == "<director_graph_builder>"
    report = audit(tmp_path, baseline, _admissions(), {})
    assert report["status"] == "fail" and report["new_sites"] == 1
    assert report["issues"][0]["kind"] == "new_sampling_site_requires_admission"


def test_rewriting_director_graph_builder_fails_closed(tmp_path):
    path = tmp_path / "h3_t8" / "director_new.py"
    path.parent.mkdir()
    path.write_text('def apply_new_graph(graph):\n graph["9"] = '
                    '{"class_type": "SamplerCustomAdvanced", "inputs": {"seed": 1}}\n', encoding="utf8")
    baseline = _baseline(tmp_path)
    path.write_text('def apply_new_graph(graph):\n graph["9"] = '
                    '{"class_type": "SamplerCustomAdvanced", "inputs": {"seed": 2}}\n', encoding="utf8")
    report = audit(tmp_path, baseline, _admissions(), {})
    assert report["status"] == "fail" and report["new_sites"] == report["removed_sites"] == 1
    assert {issue["kind"] for issue in report["issues"]} == {
        "existing_sampling_site_changed_or_removed", "new_sampling_site_requires_admission"}


def test_director_graph_builder_detects_delegated_graph_call(tmp_path):
    path = tmp_path / "h3_t8" / "director_new.py"
    path.parent.mkdir()
    path.write_text('def compile_graph(graph):\n return apply_future_graph(graph)\n', encoding="utf8")
    assert [(site["symbol"], site["callee"]) for site in discover(tmp_path)] == [
        ("compile_graph", "<director_graph_builder>")]


def test_director_direct_sampler_call_is_also_scanned(tmp_path):
    path = tmp_path / "h3_t8" / "director_new.py"
    path.parent.mkdir()
    path.write_text('def run_two_pass(sampler):\n return sampler.sample(1)\n', encoding="utf8")
    assert [(site["symbol"], site["callee"]) for site in discover(tmp_path)] == [
        ("run_two_pass", "sampler.sample")]
