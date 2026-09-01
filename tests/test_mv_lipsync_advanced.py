from __future__ import annotations

import asyncio
from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

import h3_audio_t8_pkg.mv_lipsync_advanced as mv
from h3_audio_t8_pkg import comfy_entrypoint
from h3_audio_t8_pkg.nodes_mv_lipsync_advanced import (
    MV_LIPSYNC_ADVANCED_NODE_CLASSES,
    MiniMaxH3LocalMVInNodeRendererT8Advanced,
    MiniMaxH3MVRef2VAPromptCompilerT8Advanced,
    MiniMaxH3MVVocalScenePlannerT8Advanced,
)
from helpers import plugin_widget_map


WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/workflows/24-mv-lipsync/"
    "2026-09-01_H3_Local_MV_LipSync_Ref2VA_Turbo4_Advanced_EXP.json"
)


def _audio(seconds: float, sample_rate: int = 8000) -> dict:
    samples = round(seconds * sample_rate)
    timeline = torch.arange(samples, dtype=torch.float32) / sample_rate
    waveform = 0.2 * torch.sin(2 * torch.pi * 220 * timeline)
    for center in (6.0, 13.0):
        start = max(0, round((center - 0.25) * sample_rate))
        end = min(samples, round((center + 0.25) * sample_rate))
        waveform[start:end] *= 0.01
    return {"waveform": waveform.reshape(1, 1, -1), "sample_rate": sample_rate}


def _plans(seconds: float = 18.0):
    scene_plan = mv.build_mv_scene_plan(
        _audio(seconds),
        min_scene_seconds=5.0,
        target_scene_seconds=7.0,
        max_scene_seconds=10.0,
        analysis_hop_ms=100,
        vocal_policy="assume_vocal",
        manual_boundaries_json="",
    )[0]
    prompt_plan = mv.build_mv_prompt_plan(
        scene_plan,
        "A singer performs at night.",
        "the same woman in the reference picture",
        "cinematic and realistic",
        "stable medium shot\nsmooth tracking shot",
        "keeps her mouth closed",
        "",
    )[0]
    return scene_plan, prompt_plan


def test_local_scene_planner_is_deterministic_contiguous_and_h3_aligned():
    first = mv.build_mv_scene_plan(
        _audio(20.0), 5.0, 7.0, 10.0, 100, "assume_vocal", ""
    )[0]
    second = mv.build_mv_scene_plan(
        _audio(20.0), 5.0, 7.0, 10.0, 100, "assume_vocal", ""
    )[0]
    assert first == second
    assert first["external_api_used"] is False
    assert first["total_frames"] == 480
    assert first["scene_count"] >= 2
    cursor = 0
    for scene in first["scenes"]:
        assert scene["start_frame"] == cursor
        assert scene["render_frame_count"] >= scene["frame_count"]
        assert scene["render_frame_count"] >= 124
        assert (scene["render_frame_count"] - 5) % 17 == 0
        cursor = scene["end_frame"]
    assert cursor == 480
    assert mv.validate_mv_scene_plan(first) == first


def test_manual_boundaries_are_exact_24fps_and_oversized_scenes_fail():
    plan = mv.build_mv_scene_plan(
        _audio(12.0), 3.0, 6.0, 10.0, 100, "assume_vocal", "[5.25]"
    )[0]
    assert [scene["start_frame"] for scene in plan["scenes"]] == [0, 126]
    assert plan["boundary_source"] == "manual"


def test_prompt_compiler_never_guesses_lyrics_and_emits_typed_relay_events():
    scene_plan, prompt_plan = _plans()
    result = mv.build_mv_prompt_plan(
        scene_plan,
        "A singer performs at night.",
        "the same woman in the reference picture",
        "cinematic and realistic",
        "stable shot",
        "keeps her mouth closed",
        "",
    )
    compiled, segment_json, events, preview, report = result
    assert compiled["scene_plan"] == prompt_plan["scene_plan"]
    assert compiled["external_api_used"] is False
    assert events["type"] == "H3_T8_PROMPT_RELAY_EVENTS"
    assert len(events["events"]) == scene_plan["scene_count"]
    assert all("<Picture 1>" in item["prompt"] for item in compiled["segments"])
    assert all("<Audio 1>" in item["prompt"] for item in compiled["segments"])
    assert "Do not invent" in preview
    assert len(json.loads(segment_json)) == scene_plan["scene_count"]
    assert json.loads(report)["external_api_used"] is False
    assert mv.validate_mv_prompt_plan(compiled) == compiled


def test_prompt_compiler_rejects_non_string_exact_lyrics():
    scene_plan, _prompt_plan = _plans()
    with pytest.raises(ValueError, match="scene lyric strings"):
        mv.build_mv_prompt_plan(
            scene_plan,
            "A singer performs.",
            "the same singer",
            "cinematic",
            "stable shot",
            "keeps the mouth closed",
            '["valid lyric", {"text": "not a lyric string"}]',
        )


def test_long_mv_keeps_renderer_plan_but_returns_valid_empty_relay_collection():
    scene_plan, _prompt_plan = _plans(240.0)
    assert scene_plan["scene_count"] > 32
    prompt_plan, _segments, events, _preview, report_json = mv.build_mv_prompt_plan(
        scene_plan,
        "A singer performs.",
        "the same singer",
        "cinematic",
        "stable shot",
        "keeps the mouth closed",
        "",
    )
    assert len(prompt_plan["segments"]) == scene_plan["scene_count"]
    assert prompt_plan["prompt_relay_events_available"] is False
    assert events["events"] == []
    assert json.loads(report_json)["prompt_relay_events_available"] is False


def test_nodes_are_append_only_local_and_have_no_external_api_inputs():
    assert MV_LIPSYNC_ADVANCED_NODE_CLASSES == [
        MiniMaxH3MVVocalScenePlannerT8Advanced,
        MiniMaxH3MVRef2VAPromptCompilerT8Advanced,
        MiniMaxH3LocalMVInNodeRendererT8Advanced,
    ]
    registered = asyncio.run(comfy_entrypoint().get_node_list())
    assert registered[-3:] == MV_LIPSYNC_ADVANCED_NODE_CLASSES
    renderer = MiniMaxH3LocalMVInNodeRendererT8Advanced.define_schema()
    assert renderer.is_output_node is True
    assert renderer.is_experimental is True
    ids = {item.id for item in renderer.inputs}
    assert {"model", "clip", "video_vae", "audio_vae", "full_song"} <= ids
    assert not ({"api_url", "api_key", "endpoint", "server_url"} & ids)


def test_frontend_workflow_is_importable_local_and_safe_by_default():
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert len([node for node in nodes.values() if node["type"] == "LoadAudio"]) == 1
    assert len([node for node in nodes.values() if node["type"] == "LoadImage"]) == 1
    planner = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3MVVocalScenePlannerT8Advanced"
    )
    renderer = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3LocalMVInNodeRendererT8Advanced"
    )
    planner_values = plugin_widget_map(planner, MiniMaxH3MVVocalScenePlannerT8Advanced)
    renderer_values = plugin_widget_map(renderer, MiniMaxH3LocalMVInNodeRendererT8Advanced)
    assert planner_values["vocal_policy"] == "assume_vocal"
    assert renderer_values["steps"] == 4
    assert renderer_values["resume_existing"] is True
    assert renderer_values["model_id"] == "minimax_h3_ref2va_int8_convrot+turbo4"
    note_text = "\n".join(
        str(node["widgets_values"][0])
        for node in nodes.values()
        if node["type"] == "MarkdownNote"
    )
    assert "不会提交 HTTP" in note_text
    assert "完整原曲只混入一次" in note_text


def test_renderer_runs_scenes_serially_resumes_and_muxes_master_song_once(
    monkeypatch, tmp_path: Path
):
    _scene_plan, prompt_plan = _plans(12.0)
    manifest = {"revision": 0, "segments": []}
    descriptors = {}
    sampled_indices = []
    mux_calls = []

    monkeypatch.setattr(mv, "long_video_chain_root", lambda _chain: tmp_path)
    monkeypatch.setattr(mv, "_exclusive_loop_lock", lambda _root: nullcontext())
    monkeypatch.setattr(
        mv,
        "load_delivery_manifest",
        lambda _chain, allow_new=False: (manifest, "new" if not manifest["segments"] else "primary"),
    )
    monkeypatch.setattr(mv, "_reusable_candidate", lambda *_args: None)
    monkeypatch.setattr(
        mv,
        "_available_candidate_id",
        lambda _root, index, base: f"{base}-scene-{index}",
    )
    monkeypatch.setattr(mv, "_release_segment_memory", lambda: None)
    monkeypatch.setattr(
        mv,
        "build_conditioning",
        lambda _clip, _vv, _av, prompt, *_args, **_kwargs: (
            prompt,
            {"samples": torch.zeros((1, 1))},
            None,
            prompt,
            "{}",
            "ok",
        ),
    )

    def sample(_model, positive, latent, **kwargs):
        sampled_indices.append(int(kwargs["segment_index"]))
        return dict(latent)

    monkeypatch.setattr(mv, "_sample_one_segment", sample)
    monkeypatch.setattr(
        mv,
        "decode_av_latent",
        lambda _latent, _vv, _av: (
            torch.zeros((362, 32, 32, 3)),
            _audio(16.0),
            {},
            {},
        ),
    )

    def save(frames, audio, _latent, chain, index, start, save_context, parent, revision,
             candidate_id, model_id, summary, prompt, seed, _fps, _bit_depth, _crf):
        descriptor = tmp_path / f"candidate-{index}.json"
        descriptor.write_text("{}", encoding="utf-8")
        scene = prompt_plan["scene_plan"]["scenes"][index]
        descriptors[str(descriptor)] = {
            "chain_id": chain,
            "index": index,
            "candidate_id": candidate_id,
            "parent_candidate_id": parent,
            "parent_manifest_revision": revision,
            "frame_count": int(frames.shape[0]),
            "timeline_start_frame": round(start * 24),
            "timeline_end_frame": round(start * 24) + int(frames.shape[0]),
            "is_final_segment": not save_context,
            "model_id": model_id,
            "sampling_summary": summary,
            "prompt": prompt,
            "seed": seed,
            "width": 736,
            "height": 416,
        }
        assert int(frames.shape[0]) == scene["frame_count"]
        return str(descriptor), str(tmp_path / f"candidate-{index}.mp4"), "{}"

    def accept(path, accepted, _policy, _strict):
        assert accepted is True
        item = descriptors[path]
        manifest["segments"].append(dict(item))
        manifest["revision"] += 1
        return "preview.mp4", True, str(tmp_path / "manifest.json"), "{}"

    monkeypatch.setattr(mv, "save_long_video_candidate", save)
    monkeypatch.setattr(mv, "accept_long_video_candidate", accept)

    def compose(*_args):
        path = tmp_path / "assembled.mp4"
        path.write_bytes(b"assembled")
        return str(path), json.dumps({"output_sha256": hashlib.sha256(b"assembled").hexdigest()})

    monkeypatch.setattr(mv, "compose_accepted_long_video", compose)

    def mux(_path, _audio_value, total_frames, _prefix):
        mux_calls.append(total_frames)
        path = tmp_path / "final.mp4"
        path.write_bytes(b"master")
        return str(path), {
            "output_sha256": hashlib.sha256(b"master").hexdigest(),
            "full_song_muxed_once": True,
        }

    monkeypatch.setattr(mv, "_mux_master_audio", mux)

    kwargs = {
        "chain_id": "local-mv-test",
        "width": 736,
        "height": 416,
        "base_seed": 10,
        "steps": 8,
        "shift_video": 6.0,
        "shift_audio": 3.0,
        "sampler_name": "dual_clock_euler",
        "scheduler": "native_flow",
        "resume_existing": True,
        "filename_prefix": "test",
        "bit_depth": 8,
        "crf": 28,
        "model_id": "test-model",
    }
    result = mv.run_local_mv_in_node_loop(
        object(), object(), object(), object(), torch.zeros((1, 32, 32, 3)),
        _audio(12.0), prompt_plan, **kwargs
    )
    assert result[3] == "complete"
    assert sampled_indices == list(range(prompt_plan["scene_plan"]["scene_count"]))
    assert mux_calls == [prompt_plan["scene_plan"]["total_frames"]]
    report = json.loads(result[4])
    assert report["external_api_used"] is False
    assert report["source_audio_policy"] == "full_original_song_muxed_once"

    sampled_indices.clear()
    repeated = mv.run_local_mv_in_node_loop(
        object(), object(), object(), object(), torch.zeros((1, 32, 32, 3)),
        _audio(12.0), prompt_plan, **kwargs
    )
    assert repeated[0] == result[0]
    assert sampled_indices == []


def test_accepted_manifest_checks_model_seed_and_geometry_contract():
    scene_plan, prompt_plan = _plans(12.0)
    scene = scene_plan["scenes"][0]
    summary = "local_mv_v1 8-step dual_clock_euler/native_flow shift6/3"
    item = {
        "index": 0,
        "timeline_start_frame": scene["start_frame"],
        "timeline_end_frame": scene["end_frame"],
        "frame_count": scene["frame_count"],
        "is_final_segment": len(scene_plan["scenes"]) == 1,
        "model_id": "expected-model",
        "sampling_summary": summary,
        "prompt": prompt_plan["segments"][0]["prompt"],
        "seed": 10,
        "width": 736,
        "height": 416,
    }
    mv._accepted_matches_plan(
        {"segments": [item]},
        prompt_plan,
        summary,
        base_seed=10,
        model_id="expected-model",
        width=736,
        height=416,
    )
    item["model_id"] = "different-model"
    with pytest.raises(ValueError, match="model_id"):
        mv._accepted_matches_plan(
            {"segments": [item]},
            prompt_plan,
            summary,
            base_seed=10,
            model_id="expected-model",
            width=736,
            height=416,
        )


def test_renderer_rejects_accepted_manifest_when_contract_state_is_missing(
    monkeypatch, tmp_path: Path
):
    _scene_plan, prompt_plan = _plans(12.0)
    monkeypatch.setattr(mv, "long_video_chain_root", lambda _chain: tmp_path)
    monkeypatch.setattr(mv, "_exclusive_loop_lock", lambda _root: nullcontext())
    monkeypatch.setattr(
        mv,
        "load_delivery_manifest",
        lambda _chain, allow_new=False: ({"revision": 1, "segments": [{}]}, "primary"),
    )
    with pytest.raises(ValueError, match="contract state is missing"):
        mv.run_local_mv_in_node_loop(
            object(),
            object(),
            object(),
            object(),
            torch.zeros((1, 32, 32, 3)),
            _audio(12.0),
            prompt_plan,
            chain_id="missing-state",
            width=736,
            height=416,
            base_seed=10,
            steps=8,
            shift_video=6.0,
            shift_audio=3.0,
            sampler_name="dual_clock_euler",
            scheduler="native_flow",
            resume_existing=True,
            filename_prefix="test",
            bit_depth=8,
            crf=18,
            model_id="test-model",
        )


def test_implementation_has_no_network_client_or_submitted_prompt_queue():
    source = inspect.getsource(mv).lower()
    assert "requests." not in source
    assert "httpx." not in source
    assert "urllib.request" not in source
    assert '"/prompt"' not in source
    assert "unload_all_models" not in source
