"""CPU-only contract checks for the private S18 real-asset execution probe."""
from copy import deepcopy

from tools import run_modular_s18_chunked_gpu as probe


def test_s18_probe_keeps_disk_candidates_and_only_changes_test_copy():
    before = {path: probe._sha(path) for path in probe.HASHES}
    freeze = probe.build_graph("freeze", width=128, height=128)
    resume = probe.build_graph("resume", width=128, height=128)
    assert before == {path: probe._sha(path) for path in probe.HASHES}
    assert before == probe.HASHES
    assert freeze["7"]["inputs"]["length"] == freeze["32"]["inputs"]["length"] == 56
    assert freeze["7"]["inputs"]["width"] == freeze["7"]["inputs"]["height"] == 128
    assert freeze["14"]["inputs"]["target_width"] == 256
    assert freeze["14"]["inputs"]["temporal_chunk_frames"] == 34
    assert freeze["14"]["inputs"]["temporal_overlap_frames"] == 17
    assert freeze["40"]["inputs"]["confirm_save"] is True
    assert freeze["41"]["inputs"]["confirm_save"] is True
    assert "12" not in resume and "25" not in resume
    assert resume["14"]["inputs"]["target_width"] == 256
    assert resume["40"]["inputs"]["checkpoint_path"] == ""
    assert resume["41"]["inputs"]["artifact_path"] == ""


def test_s18_probe_uses_two_distinct_save_receipts():
    resume = probe.build_graph("resume", width=128, height=128)
    receipts = {
        "native": {"path": "first.h3latent.safetensors", "sha256": "A" * 64,
                   "manifest_json": '{"schema":"native"}'},
        "segment": {"path": "segment/manifest.json", "sha256": "b" * 64,
                    "manifest_json": '{"schema":"segment"}'},
    }
    result = probe._fill_resume(resume, receipts)
    assert result["40"]["inputs"]["checkpoint_path"] == receipts["native"]["path"]
    assert result["40"]["inputs"]["expected_manifest_json"] == receipts["native"]["manifest_json"]
    assert result["40"]["inputs"]["expected_file_sha256"] == receipts["native"]["sha256"]
    assert result["41"]["inputs"]["artifact_path"] == receipts["segment"]["path"]
    assert result["41"]["inputs"]["artifact_sha256"] == receipts["segment"]["sha256"]


def test_s18_sampler_progress_excludes_vhs_encoding_progress():
    def phase(nodes, terminal="execution_success"):
        return {"terminal": {"type": terminal},
                "events": [{"type": "progress", "node": node} for node in nodes]}

    freeze = phase(["12"] * 4 + ["25"] * 4)
    resume = phase(["28"] * 4 + ["31"] * 4 + ["19"] * 3)
    assert all(probe._stage_checks(freeze, resume).values())
    wrong = deepcopy(resume)
    wrong["events"].insert(0, {"type": "progress", "node": "12"})
    assert probe._stage_checks(freeze, wrong)["resume_only_segments1_and2"] is False


def test_s18_bad_receipt_variants_mutate_only_one_explicit_field():
    base = probe.build_graph("resume", width=128, height=128)
    receipts = {
        "native": {"path": "first.h3latent.safetensors", "sha256": "A" * 64,
                   "manifest_json": '{"schema":"native"}'},
        "segment": {"path": "segment/manifest.json", "sha256": "b" * 64,
                    "manifest_json": '{"schema":"segment"}'},
    }
    probe._fill_resume(base, receipts)
    variants = probe._negative_variants(base)
    assert set(variants) == {
        "bad_native_file_sha", "bad_native_manifest", "bad_segment_sha"}
    for name, graph in variants.items():
        assert "12" not in graph and "25" not in graph
        assert graph["19"]["inputs"]["filename_prefix"].endswith(name)
        assert graph["40"]["inputs"]["checkpoint_path"] == receipts["native"]["path"]
        assert graph["41"]["inputs"]["artifact_path"] == receipts["segment"]["path"]
    assert variants["bad_native_file_sha"]["40"]["inputs"]["expected_file_sha256"] == "0" * 64
    assert variants["bad_native_manifest"]["40"]["inputs"]["expected_manifest_json"] == "{}"
    assert variants["bad_segment_sha"]["41"]["inputs"]["artifact_sha256"] == "0" * 64
    assert base["40"]["inputs"]["expected_file_sha256"] == "A" * 64
    assert base["41"]["inputs"]["artifact_sha256"] == "b" * 64


def test_s18_negative_gate_rejects_any_later_sampler_event():
    rejected = {"terminal": {"type": "execution_error", "data": {
        "node_id": "40", "exception_type": "ValueError",
        "exception_message": "native H3 checkpoint file SHA-256 mismatch: expected 0"}},
                "events": [{"type": "executing", "node": "40"}]}
    assert all(probe._negative_phase_checks(rejected, "bad_native_file_sha").values())
    for event in ({"type": "executing", "node": "28"},
                  {"type": "progress", "node": "31"}):
        bad = deepcopy(rejected)
        bad["events"].append(event)
        assert not probe._negative_phase_checks(
            bad, "bad_native_file_sha")["no_later_segment_sampler_started"]
    successful = deepcopy(rejected)
    successful["terminal"]["type"] = "execution_success"
    assert not probe._negative_phase_checks(successful, "bad_native_file_sha")["rejected_by_core"]
    wrong_error = deepcopy(rejected)
    wrong_error["terminal"]["data"]["exception_message"] = "unrelated model error"
    assert not probe._negative_phase_checks(
        wrong_error, "bad_native_file_sha")["expected_guard_node_and_reason"]


def test_s18_cancel_gate_requires_real_interrupt_and_only_later_retry():
    interrupted = {
        "terminal": {"type": "execution_interrupted"},
        "interrupt_response": {"status": "ok"},
        "progress_at_interrupt": {"value": 1, "max": 4},
        "events": [{"type": "executing", "node": "28"},
                   {"type": "progress", "node": "28"}],
    }
    retry = {
        "terminal": {"type": "execution_success"},
        "events": ([{"type": "progress", "node": "28"}] * 4 +
                   [{"type": "progress", "node": "31"}] * 4 +
                   [{"type": "progress", "node": "19"}] * 3),
    }
    assert all(probe._cancel_phase_checks(interrupted, retry).values())
    later = deepcopy(interrupted)
    later["events"].append({"type": "executing", "node": "31"})
    assert not probe._cancel_phase_checks(later, retry)["interrupted_before_segment2"]
    wrong_terminal = deepcopy(interrupted)
    wrong_terminal["terminal"]["type"] = "execution_success"
    assert not probe._cancel_phase_checks(
        wrong_terminal, retry)["interrupt_acknowledged_during_segment1"]
    reran_low = deepcopy(retry)
    reran_low["events"].insert(0, {"type": "progress", "node": "12"})
    assert not probe._cancel_phase_checks(
        interrupted, reran_low)["no_firstpass_or_segment0_rerun"]
    assert not probe._cancel_phase_checks(
        interrupted, reran_low)["fresh_core_retry_only_later_segments"]


def test_s18_cache_variants_isolate_per_segment_noise_edits():
    original = probe.build_graph("resume", width=128, height=128)
    before = deepcopy(original)
    variants = probe._cache_variants(original)
    assert original == before
    assert list(variants) == ["base", "repeat", "late", "early"]
    base, repeat, late, early = (variants[name] for name in variants)
    for graph in variants.values():
        assert "12" not in graph and "25" not in graph
        assert graph["28"]["inputs"]["noise"] == ["71", 0]
        assert graph["31"]["inputs"]["noise"] == ["70", 0]
        assert graph["40"]["inputs"] == original["40"]["inputs"]
        assert graph["41"]["inputs"] == original["41"]["inputs"]
    seed = original["11"]["inputs"]["noise_seed"]
    assert base["70"]["inputs"]["noise_seed"] == seed
    assert base["71"]["inputs"]["noise_seed"] == seed
    assert repeat["70"]["inputs"]["noise_seed"] == seed
    assert late["70"]["inputs"]["noise_seed"] == seed + 1
    assert late["71"]["inputs"]["noise_seed"] == seed
    assert early["70"]["inputs"]["noise_seed"] == seed + 1
    assert early["71"]["inputs"]["noise_seed"] == seed + 1


def test_s18_cache_matrix_requires_exact_stage_progress():
    def phase(nodes):
        return {"terminal": {"type": "execution_success"},
                "events": [{"type": "progress", "node": node} for node in nodes]}

    phases = {"base": phase(["28"] * 4 + ["31"] * 4),
              "repeat": phase(["19"] * 3),
              "late": phase(["31"] * 4 + ["19"] * 3),
              "early": phase(["28"] * 4 + ["31"] * 4)}
    assert all(probe._cache_phase_checks(phases).values())
    unexpected = deepcopy(phases)
    unexpected["late"]["events"].insert(0, {"type": "progress", "node": "28"})
    assert not probe._cache_phase_checks(unexpected)["late_only_expected_sampling"]
    failed = deepcopy(phases)
    failed["early"]["terminal"]["type"] = "execution_error"
    assert not probe._cache_phase_checks(failed)["early_only_expected_sampling"]
