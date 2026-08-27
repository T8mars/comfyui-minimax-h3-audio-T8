from __future__ import annotations

from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import h3_audio_t8_pkg.long_video_in_node_loop_advanced as loop
from h3_audio_t8_pkg.nodes_long_video_in_node_loop_advanced import (
    LONG_VIDEO_IN_NODE_LOOP_ADVANCED_NODE_CLASSES,
    MiniMaxH3LongVideoInNodeLoopT8Advanced,
)
from helpers import plugin_widget_map


WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "workflows"
    / "04-long-video"
    / "2026-08-27_H3_In_Node_Long_Video_Loop_Turbo4_Advanced_EXP.json"
)


def _run_kwargs():
    return {
        "chain_id": "loop-test",
        "total_duration_seconds": 226 / 24,
        "render_window_frames": 124,
        "context_frames": 22,
        "global_prompt": "same subject and continuous motion",
        "segment_prompts_json": "",
        "base_seed": 17,
        "seed_policy": "increment",
        "steps": 4,
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "sampler_name": "dual_clock_euler",
        "scheduler": "native_flow",
        "width": 736,
        "height": 416,
        "task_type": "auto",
        "context_audio": "video_and_audio",
        "audio_mode": "native",
        "audio_denoise_strength": 0.35,
        "add_source_as_reference": False,
        "prompt_primary_audio_ordinal": 0,
        "strict_prompt_tags": True,
        "ref_image_size": "match",
        "reference_video_policy": "official_2_to_15s",
        "first_frame_reuse": "segment0_only",
        "persistent_identity_strategy": "single_reference",
        "persistent_identity_interval": 1,
        "resume_existing": True,
        "filename_prefix": "Loop_Test",
        "audio_seam_policy": "cosine_bridge",
        "bridge_ms": 5.0,
        "bit_depth": 8,
        "crf": 18,
        "model_id": "test-model",
    }


def _install_fake_runtime(monkeypatch, tmp_path: Path):
    manifest = {"revision": 0, "segments": []}
    candidates = {}
    sample_indices = []
    fail_once = {"segment": 1}

    plans = (
        SimpleNamespace(
            context_frames=0,
            render_frames=124,
            final_frame_count=124,
            timeline_start_seconds=0.0,
            timeline_end_seconds=124 / 24,
            final_duration_seconds=124 / 24,
            trim_start_seconds=0.0,
            save_context=True,
            is_final_segment=False,
        ),
        SimpleNamespace(
            context_frames=22,
            render_frames=124,
            final_frame_count=102,
            timeline_start_seconds=124 / 24,
            timeline_end_seconds=226 / 24,
            final_duration_seconds=102 / 24,
            trim_start_seconds=22 / 24,
            save_context=False,
            is_final_segment=True,
        ),
    )
    segments = tuple(
        SimpleNamespace(index=index, prompt=f"prompt-{index}", seed=17 + index, plan=plan)
        for index, plan in enumerate(plans)
    )

    def resolve(*_args, **_kwargs):
        accepted_count = len(manifest["segments"])
        orchestration = SimpleNamespace(
            chain_id="loop-test",
            segments=segments,
            accepted_count=accepted_count,
            manifest_revision=manifest["revision"],
            complete=accepted_count == len(segments),
            sampling_summary="4-step dual_clock_euler/native_flow shift12/3",
            steps=4,
            shift_video=12.0,
            shift_audio=3.0,
            sampler_name="dual_clock_euler",
            scheduler="native_flow",
        )
        return orchestration, manifest if manifest["segments"] else None

    def load_manifest(_chain_id, allow_new=False):
        assert allow_new or manifest["segments"]
        return manifest, "primary" if manifest["segments"] else "new"

    def load_context(_chain_id, index):
        if index == 0:
            return {"empty": True}, False, "", manifest["revision"], "{}"
        parent = manifest["segments"][index - 1]
        return (
            {"empty": False, "source": index - 1},
            True,
            parent["candidate_id"],
            manifest["revision"],
            "{}",
        )

    def build_conditioning(_clip, _vv, _av, _context, index, *_args, **_kwargs):
        latent = {"samples": torch.zeros((1, 1, 1, 1))}
        return index, latent, None, f"prompt-{index}", "{}", "{}"

    def sample(_model, positive, latent, **_kwargs):
        sample_indices.append(int(positive))
        if fail_once["segment"] == int(positive):
            fail_once["segment"] = None
            raise RuntimeError("simulated segment interruption")
        return dict(latent)

    def decode(_latent, _video_vae, _audio_vae):
        frames = torch.zeros((124, 4, 4, 3))
        audio = {"waveform": torch.zeros((1, 2, 170000)), "sample_rate": 32000}
        return frames, audio, {}, {}

    def trim(frames, start_seconds, duration_seconds, audio, _fps):
        frame_count = round(duration_seconds * 24)
        start = round(start_seconds * 24)
        sample_count = round(duration_seconds * 32000)
        return (
            frames[start : start + frame_count],
            {"waveform": audio["waveform"][..., :sample_count], "sample_rate": 32000},
            "{}",
        )

    def save(*args):
        index = int(args[4])
        path = str(tmp_path / f"candidate-{index}.json")
        candidates[path] = {
            "index": index,
            "candidate_id": args[9],
            "parent_candidate_id": args[7],
            "parent_manifest_revision": int(args[8]),
            "frame_count": int(args[0].shape[0]),
            "timeline_start_frame": round(float(args[5]) * 24),
            "timeline_end_frame": round(float(args[5]) * 24) + int(args[0].shape[0]),
            "is_final_segment": not bool(args[6]),
            "model_id": args[10],
            "sampling_summary": args[11],
            "prompt": args[12],
            "seed": int(args[13]),
            "width": 736,
            "height": 416,
        }
        return path, str(tmp_path / f"candidate-{index}.mp4"), "{}"

    def accept(path, accept_candidate, _replace_policy, _strict):
        assert accept_candidate is True
        candidate = candidates[path]
        assert candidate["index"] == len(manifest["segments"])
        manifest["segments"].append(dict(candidate))
        manifest["revision"] += 1
        return "accepted.mp4", True, str(tmp_path / "manifest.json"), "{}"

    def compose(_chain_id, *_args):
        output = tmp_path / "final.mp4"
        output.write_bytes(b"file-backed-final")
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        return str(output), json.dumps({"output_sha256": digest})

    monkeypatch.setattr(loop, "resolve_long_video_orchestration", resolve)
    monkeypatch.setattr(loop, "long_video_chain_root", lambda _chain_id: tmp_path)
    monkeypatch.setattr(loop, "_exclusive_loop_lock", lambda _root: nullcontext())
    monkeypatch.setattr(loop, "load_delivery_manifest", load_manifest)
    monkeypatch.setattr(loop, "load_accepted_context", load_context)
    monkeypatch.setattr(loop, "build_long_video_conditioning", build_conditioning)
    monkeypatch.setattr(loop, "patch_long_video_model", lambda model: model)
    monkeypatch.setattr(loop, "_sample_one_segment", sample)
    monkeypatch.setattr(loop, "decode_av_latent", decode)
    monkeypatch.setattr(loop, "trim_av_output", trim)
    monkeypatch.setattr(loop, "save_long_video_candidate", save)
    monkeypatch.setattr(loop, "accept_long_video_candidate", accept)
    monkeypatch.setattr(loop, "compose_accepted_long_video", compose)
    monkeypatch.setattr(loop, "_check_interrupted", lambda: None)
    monkeypatch.setattr(loop, "_release_segment_memory", lambda: None)
    return manifest, sample_indices


def test_node_is_one_append_only_experimental_file_output():
    assert LONG_VIDEO_IN_NODE_LOOP_ADVANCED_NODE_CLASSES == [
        MiniMaxH3LongVideoInNodeLoopT8Advanced
    ]
    schema = MiniMaxH3LongVideoInNodeLoopT8Advanced.define_schema()
    assert schema.node_id == "MiniMaxH3LongVideoInNodeLoopT8Advanced"
    assert schema.is_output_node is True
    assert schema.is_experimental is True
    assert schema.category == "T8/MiniMax H3/Long Video/Experimental"
    assert [item.id for item in schema.outputs] == [
        "video",
        "video_path",
        "manifest_path",
        "completed_segments",
        "status",
        "report_json",
    ]
    inputs = {item.id: item for item in schema.inputs}
    assert inputs["resume_existing"].default is True
    assert inputs["render_window_frames"].default == 124
    assert inputs["context_frames"].default == 22


def test_media_signature_is_bounded_and_contains_no_tensor_reference():
    value = torch.arange(10000, dtype=torch.float32).reshape(100, 100)
    signature = loop._media_signature(value)
    assert signature["shape"] == [100, 100]
    assert signature["sample_count"] == 64
    assert len(signature["sampled_sha256"]) == 64
    assert "tensor" not in json.dumps(signature).lower()


def test_segment_audio_window_uses_global_timeline_and_includes_context():
    waveform = torch.arange(300, dtype=torch.float32).reshape(1, 1, 300)
    plan = SimpleNamespace(
        render_frames=124,
        context_frames=22,
        timeline_start_seconds=124 / 24,
    )
    window = loop._window_segment_audio(
        {"waveform": waveform, "sample_rate": 24}, plan, name="drive_audio"
    )
    assert window["waveform"].shape == (1, 1, 124)
    assert torch.equal(window["waveform"], waveform[..., 102:226])


def test_retry_candidate_saved_before_acceptance_is_reused(monkeypatch, tmp_path):
    expected = {
        "chain_id": "loop-test",
        "index": 1,
        "parent_candidate_id": "parent",
        "parent_manifest_revision": 1,
        "frame_count": 102,
        "timeline_start_frame": 124,
        "timeline_end_frame": 226,
        "is_final_segment": True,
        "model_id": "test-model",
        "sampling_summary": "4-step dual_clock_euler/native_flow shift12/3",
        "prompt": "raw prompt",
        "seed": 18,
        "width": 736,
        "height": 416,
    }
    base = loop._candidate_base_id(expected, "a" * 64)
    retry_descriptor = (
        tmp_path
        / "candidates"
        / "segment_00001"
        / f"{base}_retry0001"
        / "candidate.json"
    )
    retry_descriptor.parent.mkdir(parents=True)
    retry_descriptor.write_text("{}", encoding="utf-8")
    saved = dict(expected)
    saved["prompt"] = "conditioned prompt with media tags"
    monkeypatch.setattr(
        loop,
        "load_long_video_candidate_descriptor",
        lambda path: (saved, str(Path(path).with_suffix(".mp4"))),
    )

    assert loop._reusable_candidate(tmp_path, expected, base) == str(retry_descriptor)


def test_saved_candidate_detection_does_not_treat_lock_file_as_job_state(tmp_path):
    (tmp_path / loop.LOOP_LOCK_NAME).write_text("0", encoding="utf-8")
    assert loop._has_saved_candidate(tmp_path) is False
    descriptor = tmp_path / "candidates" / "segment_00000" / "candidate" / "candidate.json"
    descriptor.parent.mkdir(parents=True)
    descriptor.write_text("{}", encoding="utf-8")
    assert loop._has_saved_candidate(tmp_path) is True


def test_interrupted_second_segment_resumes_without_resampling_first(monkeypatch, tmp_path):
    manifest, sample_indices = _install_fake_runtime(monkeypatch, tmp_path)
    kwargs = _run_kwargs()

    with pytest.raises(RuntimeError, match="simulated segment interruption"):
        loop.run_long_video_in_node_loop(object(), object(), object(), object(), **kwargs)

    assert [item["index"] for item in manifest["segments"]] == [0]
    failed_state = json.loads((tmp_path / loop.LOOP_STATE_NAME).read_text(encoding="utf-8"))
    assert failed_state["status"] == "failed"
    assert failed_state["accepted_count"] == 1
    assert failed_state["current_segment_index"] == 1

    video_path, manifest_path, completed, status, report_json = (
        loop.run_long_video_in_node_loop(object(), object(), object(), object(), **kwargs)
    )
    assert Path(video_path).read_bytes() == b"file-backed-final"
    assert manifest_path == str(tmp_path / "manifest.json")
    assert completed == 2
    assert status == "complete"
    assert sample_indices == [0, 1, 1]
    assert [item["index"] for item in manifest["segments"]] == [0, 1]
    report = json.loads(report_json)
    assert report["accepted_count"] == 2
    assert report["status"] == "complete"

    sample_indices.clear()
    repeated = loop.run_long_video_in_node_loop(
        object(), object(), object(), object(), **kwargs
    )
    assert repeated[0] == video_path
    assert repeated[3] == "complete"
    assert sample_indices == []


def test_final_compose_failure_preserves_both_segments_and_retry_does_not_resample(
    monkeypatch, tmp_path
):
    manifest, sample_indices = _install_fake_runtime(monkeypatch, tmp_path)
    kwargs = _run_kwargs()

    with pytest.raises(RuntimeError, match="simulated segment interruption"):
        loop.run_long_video_in_node_loop(object(), object(), object(), object(), **kwargs)
    successful_compose = loop.compose_accepted_long_video
    fail_once = {"active": True}

    def compose_with_one_failure(*args, **kwargs):
        if fail_once["active"]:
            fail_once["active"] = False
            raise RuntimeError("simulated final compose failure")
        return successful_compose(*args, **kwargs)

    monkeypatch.setattr(loop, "compose_accepted_long_video", compose_with_one_failure)
    with pytest.raises(RuntimeError, match="simulated final compose failure"):
        loop.run_long_video_in_node_loop(object(), object(), object(), object(), **kwargs)

    failed_state = json.loads((tmp_path / loop.LOOP_STATE_NAME).read_text(encoding="utf-8"))
    assert failed_state["status"] == "failed"
    assert failed_state["accepted_count"] == 2
    assert failed_state["current_segment_index"] is None
    assert [item["index"] for item in manifest["segments"]] == [0, 1]
    assert sample_indices == [0, 1, 1]

    resumed = loop.run_long_video_in_node_loop(
        object(), object(), object(), object(), **kwargs
    )
    assert resumed[3] == "complete"
    assert sample_indices == [0, 1, 1]


def test_comfy_interrupt_is_persisted_as_interrupted_without_acceptance(
    monkeypatch, tmp_path
):
    manifest, sample_indices = _install_fake_runtime(monkeypatch, tmp_path)

    def interrupt_now():
        raise loop.comfy.model_management.InterruptProcessingException()

    monkeypatch.setattr(loop, "_check_interrupted", interrupt_now)
    with pytest.raises(loop.comfy.model_management.InterruptProcessingException):
        loop.run_long_video_in_node_loop(
            object(), object(), object(), object(), **_run_kwargs()
        )

    state = json.loads((tmp_path / loop.LOOP_STATE_NAME).read_text(encoding="utf-8"))
    assert state["status"] == "interrupted"
    assert state["accepted_count"] == 0
    assert state["current_segment_index"] == 0
    assert manifest["segments"] == []
    assert sample_indices == []


def test_second_segment_cuda_oom_preserves_first_acceptance_and_runs_cleanup(
    monkeypatch, tmp_path
):
    manifest, sample_indices = _install_fake_runtime(monkeypatch, tmp_path)
    original_sample = loop._sample_one_segment
    cleanup_calls = []

    def oom_on_second(model, positive, latent, **kwargs):
        if int(positive) == 1:
            raise torch.OutOfMemoryError("simulated CUDA out of memory")
        return original_sample(model, positive, latent, **kwargs)

    monkeypatch.setattr(loop, "_sample_one_segment", oom_on_second)
    monkeypatch.setattr(loop, "_release_segment_memory", lambda: cleanup_calls.append(True))
    with pytest.raises(torch.OutOfMemoryError, match="simulated CUDA out of memory"):
        loop.run_long_video_in_node_loop(
            object(), object(), object(), object(), **_run_kwargs()
        )

    state = json.loads((tmp_path / loop.LOOP_STATE_NAME).read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["accepted_count"] == 1
    assert state["current_segment_index"] == 1
    assert [item["index"] for item in manifest["segments"]] == [0]
    assert sample_indices == [0]
    assert len(cleanup_calls) >= 3


def test_implementation_has_no_global_unload_or_random_seed_source():
    source = inspect.getsource(loop)
    assert "unload_all_models" not in source
    assert "os.urandom" not in source
    assert "for segment in orchestration.segments" in source


def test_frontend_workflow_is_native_importable_documented_and_safe_by_default():
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])

    loop_nodes = [
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoInNodeLoopT8Advanced"
    ]
    assert len(loop_nodes) == 1
    frontend = loop_nodes[0]
    values = plugin_widget_map(frontend, MiniMaxH3LongVideoInNodeLoopT8Advanced)
    assert values["render_window_frames"] == 124
    assert values["context_frames"] == 22
    assert values["steps"] == 4
    assert values["sampler_name"] == "dual_clock_euler"
    assert values["scheduler"] == "native_flow"
    assert values["resume_existing"] is True
    assert values["audio_seam_policy"] == "cosine_bridge"

    note_text = "\n".join(
        node["widgets_values"][0]
        for node in nodes.values()
        if node["type"] == "MarkdownNote"
    )
    assert sum(node["type"] == "MarkdownNote" for node in nodes.values()) == 3
    assert "自动接受" in note_text
    assert "chain_id" in note_text
    assert "不调用全局模型卸载" in note_text
