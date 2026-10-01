"""Keep the private second-window cold-HIGH probe explicit and source-bound."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path

from tools import run_modular_s08_frozen_second_gpu as probe


def test_frozen_second_full_and_cold_graphs_are_separate_and_core_valid(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    from tools import build_modular_fast_h3_v2_workflow as base

    base.load_live_info()

    parent = {"candidate_id": "parent", "revision": 1, "job_sha256": "a" * 64}
    full = probe._graph(parent)
    assert full["83"]["inputs"]["continuation_render_frames"] == 124
    assert full["77"]["inputs"]["render_policy"] == "old_fixed_124"
    assert full["78"]["inputs"]["render_policy"] == "old_fixed_124"
    assert full["74"]["inputs"]["accepted_end_frame"] == 192
    assert full["75"]["inputs"]["accepted_end_frame"] == 192
    assert "101" in full and "103" in full and "98" not in full
    full_validation = asyncio.run(execution.validate_prompt("frozen-second-full", deepcopy(full), None))
    assert full_validation[0], full_validation

    stage = tmp_path / "output/MiniMaxH3/stage_artifacts/FastH3V2/LOW-test"
    stage.mkdir(parents=True)
    manifest = stage / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    cold = probe._graph(parent, cold=True, manifest=manifest, run_root=tmp_path)
    assert cold["60"]["inputs"]["artifact_path"] == "FastH3V2/LOW-test/manifest.json"
    assert cold["102"]["inputs"]["artifact_path"] == cold["60"]["inputs"]["artifact_path"]
    assert all(key not in cold for key in ("1", "13", "83", "85", "89", "105", "106"))
    assert "107" in cold and "108" in cold and "104" in cold
    assert cold["77"]["inputs"]["render_policy"] == "old_fixed_124"
    assert cold["75"]["inputs"]["accepted_end_frame"] == 192
    cold_validation = asyncio.run(execution.validate_prompt("frozen-second-cold", deepcopy(cold), None))
    assert cold_validation[0], cold_validation


def test_first_candidate_copy_rechecks_original_bytes(tmp_path):
    source_root = tmp_path / "source"
    folder = (source_root / "output/minimax_h3_t8_long_video" / probe.pair.CHAIN /
              "candidates/segment_00000/split_gpu_first_20260925")
    folder.mkdir(parents=True)
    (source_root / "terminal.json").write_text(json.dumps({
        "status": "split_8s_colored_mechanical_chain_pass_same_source_parity_and_human_review_pending",
    }), encoding="utf-8")
    (folder / "candidate.mp4").write_bytes(b"test-video")
    (folder / "candidate.context.safetensors").write_bytes(b"test-context")
    (folder / "source-provenance.json").write_text("{}", encoding="utf-8")
    candidate = {"chain_id": probe.pair.CHAIN, "index": 0, "frame_count": 124,
                 "status": "candidate",
                 "video_sha256": probe.shared._sha256_file(folder / "candidate.mp4").lower(),
                 "context_sha256": probe.shared._sha256_file(folder / "candidate.context.safetensors").lower()}
    (folder / "candidate.json").write_text(json.dumps(candidate), encoding="utf-8")
    checked = probe._first_candidate(source_root)
    copied = probe._copy_first_candidate(checked, tmp_path / "new")
    assert copied != checked and copied.read_bytes() == checked.read_bytes()
    assert (copied.parent / "candidate.mp4").read_bytes() == b"test-video"
