from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


import h3_audio_t8_pkg.tools.run_h3_motion_quality_matrix as matrix_tool
from h3_audio_t8_pkg.motion_quality_advanced import (
    build_av_sigma_same_nfe_schedule,
)
from h3_audio_t8_pkg.sampling import native_flow_sigmas
ROOT = Path(__file__).resolve().parents[1]


def _template():
    return json.loads(
        (ROOT / "examples" / "motion_quality_same_nfe_repair_8step_api.json").read_text(
            encoding="utf-8"
        )
    )


def _spec():
    return json.loads(
        (ROOT / "examples" / "motion_quality_matrix_spec.json").read_text(
            encoding="utf-8"
        )
    )


def test_strict_matrix_builds_three_cases_three_seeds_four_profiles_and_two_arms():
    template = _template()
    spec = matrix_tool.validate_spec(_spec())
    contract = matrix_tool.validate_template(template)
    records = matrix_tool.build_quality_records(template, spec, "T8/test-motion")

    assert len(records) == 72
    assert {item["case_id"] for item in records.values()} == {
        "rapid_displacement_dance",
        "fast_head_turn",
        "occlusion_whip_pan",
    }
    assert {item["profile"] for item in records.values()} == set(
        matrix_tool.PROFILE_ORDER
    )
    assert {item["arm"] for item in records.values()} == set(matrix_tool.ARM_ORDER)
    assert all(item["schedule"]["same_nfe"] for item in records.values())
    assert {item["expected_dialogue"] for item in records.values()} == {"none"}

    groups = {}
    for item in records.values():
        key = (item["case_id"], item["seed"], item["profile"])
        groups.setdefault(key, []).append(item)
    assert len(groups) == 36
    assert all(len({item["pair_control_fingerprint"] for item in group}) == 1 for group in groups.values())

    stock = next(item for item in records.values() if item["profile"] == "stock20")
    standard = next(
        item for item in records.values() if item["profile"] == "turbo_standard8"
    )
    assert stock["api_prompt"][contract["sampler"]]["inputs"]["steps"] == 20
    assert stock["api_prompt"][contract["sampler"]]["inputs"]["model"] == [
        contract["unet"],
        0,
    ]
    assert standard["api_prompt"][contract["sampler"]]["inputs"]["steps"] == 8
    assert standard["api_prompt"][contract["sampler"]]["inputs"]["model"] == [
        contract["lora"],
        0,
    ]


@pytest.mark.parametrize(
    "profile_name,steps,accept_ood",
    [
        ("stock20", 20, False),
        ("turbo_standard8", 8, True),
        ("turbo_ema8", 8, True),
        ("turbo_fl2v8", 8, True),
    ],
)
def test_runner_schedule_hash_matches_product_schedule(profile_name, steps, accept_ood):
    spec = _spec()
    descriptor = matrix_tool.schedule_descriptor(
        spec["profiles"][profile_name],
        "same_nfe_tail",
        spec["same_nfe"],
    )
    _output, actual_nfe, report_json = build_av_sigma_same_nfe_schedule(
        native_flow_sigmas(steps, 12.0),
        "apply_exp",
        0.5,
        1.6,
        12.0,
        3.0,
        profile_name,
        "dual_clock_euler",
        accept_ood,
    )
    report = json.loads(report_json)
    assert descriptor["nfe"] == actual_nfe
    assert descriptor["video_schedule_sha256"] == report["output_schedule_sha256"]
    assert descriptor["video_sigmas"] == report["video_sigmas"]
    assert descriptor["audio_sigmas"] == report["audio_sigmas"]


def test_template_requires_audit_to_repair_link_and_forbids_cache_treatment():
    template = _template()
    contract = matrix_tool.validate_template(template)
    assert template[contract["repair"]]["inputs"]["audit_report_json"] == [
        contract["audit"],
        3,
    ]

    cached = copy.deepcopy(template)
    cached["99"] = {
        "class_type": "MiniMaxH3BlockCacheT8",
        "inputs": {},
    }
    with pytest.raises(matrix_tool.ValidationError, match="forbids"):
        matrix_tool.validate_template(cached)

    broken = copy.deepcopy(template)
    broken[contract["repair"]]["inputs"]["audit_report_json"] = "{}"
    with pytest.raises(matrix_tool.ValidationError, match="must consume the audit"):
        matrix_tool.validate_template(broken)


def test_plan_command_writes_resumable_manifest_without_selecting_a_winner(tmp_path):
    output = tmp_path / "matrix"
    result = matrix_tool.main(
        [
            "plan",
            str(ROOT / "examples" / "motion_quality_same_nfe_repair_8step_api.json"),
            str(ROOT / "examples" / "motion_quality_matrix_spec.json"),
            str(output),
            "--output-prefix",
            "T8/pytest-motion",
        ]
    )
    assert result == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == matrix_tool.SCHEMA
    assert len(manifest["records"]) == 72
    assert manifest["quality_decision"] == "not_evaluated"
    assert manifest["memory_safe_claim"] is False
    assert manifest["requirements"]["cold_repeat_gate"] == 3
    assert manifest["requirements"]["warm_repeat_gate"] == 3
    assert manifest["asset_contract"] == _spec()["assets"]
    assert len(list((output / "prompts").glob("*.json"))) == 72


def test_asset_preflight_hashes_first_observation_reuses_stable_cache_and_rejects_tamper(
    tmp_path,
):
    spec = _spec()
    comfy_root = tmp_path / "ComfyUI"
    for index, name in enumerate(matrix_tool.ASSET_ORDER):
        relative = Path(spec["assets"][name]["path"])
        path = comfy_root.joinpath(*relative.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"fixture-{index}".encode()
        path.write_bytes(payload)
        spec["assets"][name]["size_bytes"] = len(payload)
        spec["assets"][name]["sha256"] = hashlib.sha256(payload).hexdigest()
    for index, case in enumerate(spec["cases"]):
        path = comfy_root / "input" / case["image"]
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"reference-{index}".encode()
        path.write_bytes(payload)
        case["image_size_bytes"] = len(payload)
        case["image_sha256"] = hashlib.sha256(payload).hexdigest()

    first = matrix_tool.verify_assets(spec, comfy_root)
    assert first["all_verified"] is True
    assert all(not item["cache_reused"] for item in first["files"].values())
    assert all(not item["cache_reused"] for item in first["reference_inputs"].values())
    second = matrix_tool.verify_assets(spec, comfy_root, cached=first)
    assert all(item["cache_reused"] for item in second["files"].values())
    assert all(item["cache_reused"] for item in second["reference_inputs"].values())

    target = comfy_root / spec["assets"]["audio_vae"]["path"]
    target.write_bytes(b"X" * target.stat().st_size)
    with pytest.raises(matrix_tool.ValidationError, match="SHA-256 mismatch"):
        matrix_tool.verify_assets(spec, comfy_root, cached=second)


def test_asset_contract_rejects_profile_or_template_drift():
    spec = _spec()
    broken_profile = copy.deepcopy(spec)
    broken_profile["profiles"]["turbo_ema8"]["lora_name"] = "other.safetensors"
    with pytest.raises(matrix_tool.ValidationError, match="declared lora_ema"):
        matrix_tool.validate_spec(broken_profile)

    template = _template()
    contract = matrix_tool.validate_template(template)
    template["3"]["inputs"]["clip_name"] = "other.safetensors"
    with pytest.raises(matrix_tool.ValidationError, match="declared asset"):
        matrix_tool.validate_template_assets(template, contract, spec)


def test_strict_spec_rejects_a_smaller_convenience_matrix():
    spec = _spec()
    spec["cases"] = spec["cases"][:1]
    spec["seeds"] = spec["seeds"][:1]
    with pytest.raises(matrix_tool.ValidationError, match="exactly three"):
        matrix_tool.validate_spec(spec)
    assert matrix_tool.validate_spec(spec, strict_matrix=False) is spec


def _completed_quality_manifest():
    records = matrix_tool.build_quality_records(_template(), _spec(), "T8/test-motion")
    manifest = matrix_tool.build_manifest(
        ROOT / "examples" / "motion_quality_same_nfe_repair_8step_api.json",
        ROOT / "examples" / "motion_quality_matrix_spec.json",
        records,
    )
    for record in manifest["records"].values():
        record["status"] = "success"
    return manifest


def _repeat_selection(**updates):
    selection = {
        "schema": "t8.minimax_h3.motion_quality_repeat_selection.v1",
        "review_completed": True,
        "review_rationale": (
            "Reviewed the complete blind matrix and selected this exact cell for memory testing."
        ),
        "profiles": ["turbo_standard8"],
        "arms": ["same_nfe_tail"],
        "cases": ["rapid_displacement_dance"],
        "seeds": [2608141001],
    }
    selection.update(updates)
    return selection


def test_repeat_plan_requires_completed_human_review_and_adds_three_cold_three_warm():
    manifest = _completed_quality_manifest()
    with pytest.raises(matrix_tool.ValidationError, match="review_completed"):
        matrix_tool.append_repeat_records(
            manifest,
            _repeat_selection(review_completed=False),
            "T8/repeat",
        )
    with pytest.raises(matrix_tool.ValidationError, match="review_rationale"):
        matrix_tool.append_repeat_records(
            manifest,
            _repeat_selection(review_rationale="too short"),
            "T8/repeat",
        )

    run_ids = matrix_tool.append_repeat_records(
        manifest, _repeat_selection(), "T8/repeat"
    )
    assert len(run_ids) == 7
    protocols = [manifest["records"][run_id]["protocol"] for run_id in run_ids]
    assert protocols.count("cold") == 3
    assert protocols.count("warm_primer") == 1
    assert protocols.count("warm") == 3
    assert len({manifest["records"][run_id]["repeat_group"] for run_id in run_ids}) == 1
    assert matrix_tool.summarize_repeat_gate(manifest)["status"] == "incomplete"


def _write_provenance_sources(root, relative_paths, *, mtime=100.0):
    for relative in relative_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source:{relative}\n", encoding="utf-8")
        path.touch()
        import os

        os.utime(path, (mtime, mtime))


def test_runtime_source_provenance_hashes_sources_and_binds_fresh_process(
    tmp_path, monkeypatch
):
    comfy_root = tmp_path / "ComfyUI"
    plugin_root = comfy_root / "custom_nodes" / "minimax-h3-audio-T8"
    _write_provenance_sources(
        plugin_root,
        matrix_tool.PLUGIN_RUNTIME_SOURCE_FILES
        + matrix_tool.PLUGIN_CLIENT_SOURCE_FILES,
    )
    _write_provenance_sources(comfy_root, matrix_tool.COMFY_RUNTIME_SOURCE_FILES)
    monkeypatch.setattr(
        matrix_tool,
        "_git_snapshot",
        lambda root: {
            "commit": f"commit-{root.name}",
            "worktree_clean": True,
            "status_line_count": 0,
            "status_sha256": "0" * 64,
        },
    )
    monkeypatch.setattr(matrix_tool, "_process_create_time", lambda _pid: 200.0)

    first = matrix_tool.runtime_source_provenance(
        comfy_root, 4242, plugin_root=plugin_root
    )
    second = matrix_tool.runtime_source_provenance(
        comfy_root, 4243, plugin_root=plugin_root
    )
    assert first["schema"] == matrix_tool.PROVENANCE_SCHEMA
    assert first["server_sources_predate_process"] is True
    assert first["server_pid"] == 4242
    assert first["fingerprint"] == second["fingerprint"]
    assert first["plugin"]["runtime_source_files"]["sampling.py"]["sha256"] == hashlib.sha256(
        (plugin_root / "sampling.py").read_bytes()
    ).hexdigest()

    changed = plugin_root / matrix_tool.PLUGIN_RUNTIME_SOURCE_FILES[0]
    import os

    os.utime(changed, (201.0, 201.0))
    stale = matrix_tool.runtime_source_provenance(
        comfy_root, 4244, plugin_root=plugin_root
    )
    assert stale["server_sources_predate_process"] is False
    assert stale["fingerprint"] != first["fingerprint"]


def test_repeat_provenance_rejects_missing_or_changed_successful_runtime():
    record = {
        "protocol": "warm",
        "repeat_group": "candidate-a",
        "status": "pending",
    }
    current = {"fingerprint": "same"}
    manifest = {
        "records": {
            "current": record,
            "prior": {
                "protocol": "warm_primer",
                "repeat_group": "candidate-a",
                "status": "success",
                "runtime_provenance": {"fingerprint": "same"},
            },
        }
    }
    assert matrix_tool.repeat_provenance_error(manifest, record, current) is None
    manifest["records"]["prior"]["runtime_provenance"]["fingerprint"] = "changed"
    assert "changed" in matrix_tool.repeat_provenance_error(
        manifest, record, current
    )
    manifest["records"]["prior"].pop("runtime_provenance")
    assert "lacks" in matrix_tool.repeat_provenance_error(manifest, record, current)
    assert (
        matrix_tool.repeat_provenance_error(
            manifest, {"protocol": "quality_once"}, current
        )
        is None
    )


def _complete_repeat_records(manifest):
    cold_pid = iter((101, 102, 103))
    warm_index = 0
    for record in manifest["records"].values():
        protocol = record.get("protocol", "quality_once")
        if protocol == "quality_once":
            continue
        record["status"] = "success"
        record["minimum_headroom_mib"] = 640.0
        if protocol == "cold":
            record["server_pid"] = next(cold_pid)
            record["baseline_vram_used_bytes"] = 512 * 1024**2
            record["baseline_process_private_bytes"] = 2 * 1024**3
        else:
            record["server_pid"] = 200
            record["baseline_vram_used_bytes"] = (1000 + 10 * warm_index) * 1024**2
            record["baseline_process_private_bytes"] = (4096 + 16 * warm_index) * 1024**2
            if protocol == "warm":
                warm_index += 1
    return manifest


def test_repeat_gate_requires_distinct_cold_pid_warm_pid_headroom_and_no_staircase():
    manifest = _completed_quality_manifest()
    matrix_tool.append_repeat_records(manifest, _repeat_selection(), "T8/repeat")
    _complete_repeat_records(manifest)
    gate = matrix_tool.summarize_repeat_gate(manifest)
    assert gate["status"] == "pass"
    assert gate["groups"][0]["cold_distinct_pid_pass"] is True
    assert gate["groups"][0]["warm_single_pid_pass"] is True
    assert gate["groups"][0]["headroom_512mib_pass"] is True
    assert gate["groups"][0]["warm_staircase_pass"] is True

    broken_pid = copy.deepcopy(manifest)
    cold = [
        item
        for item in broken_pid["records"].values()
        if item.get("protocol") == "cold"
    ]
    cold[1]["server_pid"] = cold[0]["server_pid"]
    assert matrix_tool.summarize_repeat_gate(broken_pid)["status"] == "fail"

    low_headroom = copy.deepcopy(manifest)
    measured = next(
        item
        for item in low_headroom["records"].values()
        if item.get("protocol") in {"cold", "warm"}
    )
    measured["minimum_headroom_mib"] = 511.99
    assert matrix_tool.summarize_repeat_gate(low_headroom)["status"] == "fail"


def test_plan_repeats_command_writes_seven_prompts_only_after_quality_completion(tmp_path):
    output = tmp_path / "matrix"
    output.mkdir()
    manifest = _completed_quality_manifest()
    manifest_path = output / "manifest.json"
    selection_path = output / "selection.json"
    matrix_tool.write_json_atomic(manifest_path, manifest)
    matrix_tool.write_json_atomic(selection_path, _repeat_selection())

    assert matrix_tool.main(
        [
            "plan-repeats",
            str(manifest_path),
            str(selection_path),
            "--output-prefix",
            "T8/repeat",
        ]
    ) == 0
    planned = json.loads(manifest_path.read_text(encoding="utf-8"))
    repeats = [
        item
        for item in planned["records"].values()
        if item.get("protocol") != "quality_once"
    ]
    assert len(repeats) == 7
    assert len(list((output / "prompts").glob("repeat-*.json"))) == 7


def test_audio_pair_metrics_reports_exact_sample_contract_and_correlation(monkeypatch):
    import numpy as np

    control = np.asarray([[0.1, -0.1], [0.2, -0.2], [0.3, -0.3]], dtype=np.float32)
    treatment = control.copy()
    decoded = iter((control, treatment))
    monkeypatch.setattr(
        matrix_tool,
        "_decode_audio_stereo_32k",
        lambda _path, _ffmpeg: next(decoded),
    )
    record_a = {"run_id": "a", "outputs": [{"path": "a.mp4"}]}
    record_b = {"run_id": "b", "outputs": [{"path": "b.mp4"}]}
    result = matrix_tool.audio_pair_metrics(record_a, record_b, "ffmpeg")
    assert result["sample_count_equal"] is True
    assert result["compared_sample_count"] == 3
    assert result["zero_lag_correlation"] == pytest.approx(1.0)
    assert result["control_referenced_snr_db"] == float("inf")


def test_optional_metric_availability_rejects_not_run_placeholders():
    assert matrix_tool._face_metrics_available(
        {"identity_metric_valid": False, "status": "not_run"}
    ) is False
    assert matrix_tool._face_metrics_available(
        {"identity_metric_valid": True, "detected_frames": 1}
    ) is True
    assert matrix_tool._asr_metrics_available(
        {"status": "not_run", "extra_speech_verified": False}
    ) is False
    assert matrix_tool._asr_metrics_available(
        {"nonempty_segment_count": 0, "segments": []}
    ) is True


def test_optional_metric_enrichment_uses_local_case_references_and_persists_no_embeddings(
    tmp_path, monkeypatch
):
    manifest = _completed_quality_manifest()
    for record in manifest["records"].values():
        record["outputs"] = [{"path": "unused.mp4", "sha256": "unused"}]
        record["metrics"] = {}
    comfy_root = tmp_path / "ComfyUI"
    input_root = comfy_root / "input"
    input_root.mkdir(parents=True)
    for name in {record["reference_image"] for record in manifest["records"].values()}:
        (input_root / name).write_bytes(b"review fixture")
    calls = {"asr": 0, "face": []}

    def fake_asr(proxy, _model, *, language, beam_size):
        calls["asr"] += 1
        assert language == "auto"
        assert beam_size == 5
        for record in proxy["runs"].values():
            record.setdefault("metrics", {})["asr"] = {
                "nonempty_segment_count": 0,
                "segments": [],
            }

    def fake_face(
        proxy,
        reference,
        *,
        model_root,
        model_name,
        sample_count,
        detector_threshold,
    ):
        calls["face"].append(reference.name)
        assert model_root == tmp_path / "insightface"
        assert model_name == "buffalo_l"
        assert sample_count == 31
        assert detector_threshold == 0.15
        for record in proxy["runs"].values():
            record.setdefault("metrics", {})["face_identity"] = {
                "requested_frames": 31,
                "detected_frames": 30,
                "detection_coverage": 30 / 31,
                "cosine_median": 0.7,
                "cosine_min": 0.5,
            }

    monkeypatch.setattr(matrix_tool, "add_asr_metrics", fake_asr)
    monkeypatch.setattr(matrix_tool, "add_face_identity_metrics", fake_face)
    matrix_tool.enrich_optional_metrics(
        manifest,
        comfy_root=comfy_root,
        asr_model=tmp_path / "asr",
        asr_language="auto",
        asr_beam_size=5,
        face_model_root=tmp_path / "insightface",
        face_model_name="buffalo_l",
        face_sample_count=31,
        face_detector_threshold=0.15,
    )
    assert calls["asr"] == 1
    assert sorted(calls["face"]) == ["10A.jpg", "10Db.jpg", "10b.jpg"]
    assert all(
        record["metrics"]["asr"]["unexpected_speech_screen_pass"] is True
        and record["metrics"]["face_identity"]["hard_missing_face_count"] == 1
        for record in manifest["records"].values()
    )
    assert manifest["optional_metrics"]["downloads_models"] is False
    assert manifest["optional_metrics"]["biometric_embeddings_persisted"] is False


def test_strict_decode_accepts_one_transient_failure_but_rejects_two(
    tmp_path, monkeypatch
):
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"stable")
    monkeypatch.setattr(
        matrix_tool,
        "wait_for_stable_output",
        lambda _path: {"validated": True, "size_bytes": 6},
    )
    results = iter(
        [
            SimpleNamespace(returncode=1, stderr=b"transient"),
            SimpleNamespace(returncode=0, stderr=b""),
            SimpleNamespace(returncode=0, stderr=b""),
        ]
    )
    monkeypatch.setattr(matrix_tool.subprocess, "run", lambda *_a, **_k: next(results))
    evidence = matrix_tool.strict_decode_metrics(media, "ffmpeg")
    assert evidence["validated"] is True
    assert evidence["success_count"] == 2
    assert evidence["transient_failure_count"] == 1

    failures = iter(
        [
            SimpleNamespace(returncode=1, stderr=b"bad-1"),
            SimpleNamespace(returncode=1, stderr=b"bad-2"),
            SimpleNamespace(returncode=0, stderr=b""),
        ]
    )
    monkeypatch.setattr(matrix_tool.subprocess, "run", lambda *_a, **_k: next(failures))
    with pytest.raises(matrix_tool.ValidationError, match="repeatable strict decode"):
        matrix_tool.strict_decode_metrics(media, "ffmpeg")


def test_record_metrics_retries_partial_opencv_decode(monkeypatch):
    record = {"run_id": "retry", "outputs": [{"path": "retry.mp4"}]}
    monkeypatch.setattr(
        matrix_tool,
        "strict_decode_metrics",
        lambda _path, _ffmpeg: {"validated": True},
    )
    monkeypatch.setattr(
        matrix_tool,
        "video_metrics",
        lambda _path: {"frame_count": 124, "duration_seconds": 124 / 24, "fps": 24},
    )
    motions = iter(
        [
            {"transition_count": 100},
            {"transition_count": 123},
        ]
    )
    monkeypatch.setattr(matrix_tool, "motion_metrics", lambda _path: next(motions))
    monkeypatch.setattr(
        matrix_tool,
        "audio_metrics",
        lambda _path, _ffmpeg: {"duration_seconds": 5.152},
    )
    monkeypatch.setattr(matrix_tool.time, "sleep", lambda _seconds: None)
    metrics = matrix_tool.record_metrics(record, "ffmpeg")
    assert metrics["strict_decode"]["opencv_contract_validated"] is True
    assert len(metrics["strict_decode"]["opencv_measurement_attempts"]) == 2
    assert metrics["motion"]["transition_count"] == 123


def test_optional_metric_enrichment_only_missing_preserves_existing(tmp_path, monkeypatch):
    manifest = _completed_quality_manifest()
    for record in manifest["records"].values():
        record["outputs"] = [{"path": "unused.mp4", "sha256": "unused"}]
        record["metrics"] = {}
    existing_id, existing = next(iter(manifest["records"].items()))
    existing["metrics"]["asr"] = {
        "nonempty_segment_count": 0,
        "segments": [],
        "marker": "preserved",
    }
    existing["metrics"]["face_identity"] = {
        "identity_metric_valid": True,
        "requested_frames": 31,
        "detected_frames": 31,
        "marker": "preserved",
    }
    comfy_root = tmp_path / "ComfyUI"
    input_root = comfy_root / "input"
    input_root.mkdir(parents=True)
    for name in {record["reference_image"] for record in manifest["records"].values()}:
        (input_root / name).write_bytes(b"review fixture")
    processed = {"asr": set(), "face": set()}

    def fake_asr(proxy, _model, *, language, beam_size):
        processed["asr"].update(proxy["runs"])
        for record in proxy["runs"].values():
            record.setdefault("metrics", {})["asr"] = {
                "nonempty_segment_count": 0,
                "segments": [],
            }

    def fake_face(proxy, _reference, **_kwargs):
        processed["face"].update(proxy["runs"])
        for record in proxy["runs"].values():
            record.setdefault("metrics", {})["face_identity"] = {
                "requested_frames": 31,
                "detected_frames": 30,
            }

    monkeypatch.setattr(matrix_tool, "add_asr_metrics", fake_asr)
    monkeypatch.setattr(matrix_tool, "add_face_identity_metrics", fake_face)
    matrix_tool.enrich_optional_metrics(
        manifest,
        comfy_root=comfy_root,
        asr_model=tmp_path / "asr",
        asr_language="auto",
        asr_beam_size=5,
        face_model_root=tmp_path / "insightface",
        face_model_name="buffalo_l",
        face_sample_count=31,
        face_detector_threshold=0.15,
        only_missing=True,
    )
    expected = set(manifest["records"]) - {existing_id}
    assert processed["asr"] == expected
    assert processed["face"] == expected
    assert existing["metrics"]["asr"]["marker"] == "preserved"
    assert existing["metrics"]["face_identity"]["marker"] == "preserved"
    assert manifest["optional_metrics"]["only_missing"] is True
    assert manifest["optional_metrics"]["asr_records_processed"] == len(expected)
    assert manifest["optional_metrics"]["face_records_processed"] == len(expected)


def test_blind_package_atomically_refreshes_replaced_sources_and_hides_key(tmp_path):
    manifest = _completed_quality_manifest()
    records = list(manifest["records"].values())
    first = records[0]
    pair_key = (first["case_id"], first["seed"], first["profile"])
    pair = [
        record
        for record in records
        if (record["case_id"], record["seed"], record["profile"]) == pair_key
    ]
    assert {record["arm"] for record in pair} == set(matrix_tool.ARM_ORDER)
    for record in records:
        record["status"] = "failed"
    sources = tmp_path / "sources"
    sources.mkdir()
    for record in pair:
        record["status"] = "success"
        media = sources / f"{record['arm']}-v1.mp4"
        payload = f"{record['arm']}-first".encode()
        media.write_bytes(payload)
        record["outputs"] = [
            {
                "path": str(media),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ]

    output = tmp_path / "result"
    matrix_tool.build_blind_package(manifest, output, 2608147999)
    key_path = output / "blind" / "blind_key.json"
    key = json.loads(key_path.read_text(encoding="utf-8"))
    assert len(key["rows"]) == 2
    replaced = pair[0]
    original_key_row = next(
        row for row in key["rows"] if row["arm"] == replaced["arm"]
    )
    blind_target = Path(original_key_row["media"])
    assert blind_target.read_bytes() == f"{replaced['arm']}-first".encode()

    new_media = sources / f"{replaced['arm']}-v2.mp4"
    new_payload = f"{replaced['arm']}-replacement".encode()
    new_media.write_bytes(new_payload)
    replaced["outputs"] = [
        {
            "path": str(new_media),
            "sha256": hashlib.sha256(new_payload).hexdigest(),
        }
    ]
    matrix_tool.build_blind_package(manifest, output, 2608147999)

    refreshed_key = json.loads(key_path.read_text(encoding="utf-8"))
    refreshed_row = next(
        row for row in refreshed_key["rows"] if row["arm"] == replaced["arm"]
    )
    assert Path(refreshed_row["media"]) == blind_target
    assert blind_target.read_bytes() == new_payload
    assert refreshed_row["source_sha256"] == hashlib.sha256(new_payload).hexdigest()
    html = (output / "blind" / "blind_review.html").read_text(encoding="utf-8")
    assert "same_nfe_tail" not in html
    assert '"arm"' not in html
    assert "blind_key.json" not in html
    assert "匿名 A/B 评审" in html


def test_final_strict_decode_report_requires_three_full_trials_per_mode(tmp_path):
    summaries = []
    for mode in ("default", "threads1"):
        for trial in range(1, 4):
            summaries.append(
                {
                    "mode": mode,
                    "trial": trial,
                    "checked": 72,
                    "bad_count": 0,
                    "bad_run_ids": [],
                }
            )
    trials = {
        "schema": "t8.minimax_h3.strict_decode_trials.v1",
        "checked_at": "2026-08-15T00:00:00Z",
        "ffmpeg": "ffmpeg",
        "file_count": 72,
        "summaries": summaries,
    }
    trials_path = tmp_path / "strict_decode_trials.json"
    matrix_tool.write_json_atomic(trials_path, trials)
    result = matrix_tool.final_strict_decode_report(tmp_path)
    assert result is not None
    assert result["all_pass"] is True
    assert result["trial_count"] == 6
    assert result["decode_invocation_count"] == 432
    assert result["mode_trial_counts"] == {"default": 3, "threads1": 3}
    assert result["source_trials_sha256"] == hashlib.sha256(
        trials_path.read_bytes()
    ).hexdigest()

    trials["summaries"][0]["bad_count"] = 1
    trials["summaries"][0]["bad_run_ids"] = ["transient"]
    matrix_tool.write_json_atomic(trials_path, trials)
    assert matrix_tool.final_strict_decode_report(tmp_path)["all_pass"] is False

    trials["summaries"] = trials["summaries"][:-1]
    matrix_tool.write_json_atomic(trials_path, trials)
    incomplete = matrix_tool.final_strict_decode_report(tmp_path)
    assert incomplete["required_modes_pass"] is False
    assert incomplete["all_pass"] is False
