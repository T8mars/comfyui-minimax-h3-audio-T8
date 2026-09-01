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
    MV_LIPSYNC_V2_ADVANCED_NODE_CLASSES,
    MV_LIPSYNC_V3_ADVANCED_NODE_CLASSES,
    MiniMaxH3LocalMVInNodeRendererT8Advanced,
    MiniMaxH3LocalMVVocalLockRendererV2T8Advanced,
    MiniMaxH3LocalMVVocalLockVisualRendererV3T8Advanced,
    MiniMaxH3MVRef2VAPromptCompilerT8Advanced,
    MiniMaxH3MVVocalLockPromptCompilerV2T8Advanced,
    MiniMaxH3MVVocalLockScenePlannerV2T8Advanced,
    MiniMaxH3MVVocalLockVisualDirectorV3T8Advanced,
    MiniMaxH3MVVocalScenePlannerT8Advanced,
)
from h3_audio_t8_pkg.nodes_qwen_prefix_cache_advanced import (
    MiniMaxH3QwenReferencePrefixCacheT8Advanced,
)
from helpers import plugin_widget_map


WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/workflows/24-mv-lipsync/"
    "2026-09-01_H3_Local_MV_LipSync_Ref2VA_Turbo4_Advanced_EXP.json"
)
VOCAL_LOCK_V2_WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/workflows/24-mv-lipsync/"
    "2026-09-01_H3_Local_MV_VocalLock_V2_Ref2VA_8Step_Advanced_EXP.json"
)
VOCAL_LOCK_V3_WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/workflows/24-mv-lipsync/"
    "2026-09-01_H3_Local_MV_VocalLock_V3_Official_Ref2V_Turbo4_Advanced_EXP.json"
)
VOCAL_LOCK_V3_USER_WORKFLOW_PATH = (
    Path(__file__).resolve().parents[3]
    / "user/default/workflows/MiniMax H3 T8/24-mv-lipsync/"
    "2026-09-01_H3_Local_MV_VocalLock_V3_Official_Ref2V_Turbo4_Advanced_EXP.json"
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


def _vocal_lock_plans(seconds: float = 5.152, *, exact_text: str = ""):
    scene_plan = mv.build_mv_vocal_lock_scene_plan(
        _audio(seconds),
        _audio(seconds),
        min_scene_seconds=5.0,
        target_scene_seconds=7.0,
        max_scene_seconds=10.0,
        analysis_hop_ms=50,
        vocal_active_ratio=0.12,
        manual_boundaries_json="",
    )[0]
    prompt_plan = mv.build_mv_vocal_lock_prompt_plan(
        scene_plan,
        "A woman delivers a clear test sentence directly to camera.",
        "the same woman shown in the reference picture",
        "cinematic realism and natural skin texture",
        "extreme wide shot from behind with a gentle push-in",
        "spoken_dialogue",
        "English",
        json.dumps([exact_text]) if exact_text else "",
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


def test_vocal_lock_v2_prompt_uses_official_sections_and_visible_mouth_contract():
    transcript = "All the time he was talking to me."
    _scene_plan, prompt_plan = _vocal_lock_plans(exact_text=transcript)
    prompt = prompt_plan["segments"][0]["prompt"]
    sections = [
        "subject_definitions:",
        "summary:",
        "retention_analysis:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    ]
    positions = [prompt.index(section) for section in sections]
    assert positions == sorted(positions)
    assert all(prompt.count(section) == 1 for section in sections)
    assert "<Subject 1>" in prompt and "<Picture 1>" in prompt
    assert "<Audio 1>" in prompt and "fully_copy" in prompt
    assert "fully_preserved" in prompt
    assert "Medium close-up" in prompt
    assert "unobstructed mouth" in prompt
    assert "every audible phoneme" in prompt
    assert "subject silhouette remain sharply resolved" in prompt
    assert "without haloing, edge smearing, double contours" in prompt
    assert "Natural depth of field may soften only the distant background" in prompt
    assert "extreme wide shot" not in prompt.lower()
    assert "from behind" not in prompt.lower()
    assert f"<d>[English] {transcript}</d>" in prompt
    assert prompt_plan["audio_contract"] == {
        "drive": "vocal_lock_audio",
        "conditioning_mode": "lock_source",
        "delivery": "full_song_muxed_once",
    }
    assert mv.validate_mv_vocal_lock_prompt_plan(prompt_plan) == prompt_plan


def test_vocal_lock_v2_prompt_never_guesses_missing_words():
    _scene_plan, prompt_plan = _vocal_lock_plans()
    prompt = prompt_plan["segments"][0]["prompt"]
    assert "<d>" not in prompt
    assert "No words or lyrics are added" in prompt


def test_vocal_lock_v3_visual_director_enforces_one_person_one_face_per_scene():
    scene_plan, _prompt_plan = _vocal_lock_plans(12.0)
    directions = [
        {
            "camera": "locked frontal medium close-up",
            "lighting": "soft blue-gray studio light",
            "performance": "keeps direct eye-line and restrained movement",
            "emotion": "calm",
        }
        for _scene in scene_plan["scenes"]
    ]
    prompt_plan, _segments, _events, preview, report_json = (
        mv.build_mv_vocal_lock_visual_prompt_plan(
            scene_plan,
            "A coherent studio performance.",
            "the same woman shown in the reference picture",
            "cinematic realism and natural skin texture",
            json.dumps(directions),
            "spoken_dialogue",
            "English",
            "",
        )
    )
    assert prompt_plan["schema"] == mv.MV_VOCAL_LOCK_VISUAL_PROMPT_SCHEMA
    assert prompt_plan["scene_direction_source"] == "user_supplied_exact_scene_directions"
    assert prompt_plan["visual_contract"] == (
        "one_person_one_face_no_reflective_or_figurative_background"
    )
    assert "Exactly one visible person and exactly one visible human face" in preview
    assert "No mirrors, reflections, projections, screens, posters" in preview
    assert "background people, or visible props" in preview
    assert all("scene_direction" in item for item in prompt_plan["segments"])
    assert json.loads(report_json)["visual_contract"] == prompt_plan["visual_contract"]
    assert mv.validate_mv_vocal_lock_visual_prompt_plan(prompt_plan) == prompt_plan


def test_vocal_lock_v3_visual_director_rejects_incomplete_or_conflicting_directions():
    scene_plan, _prompt_plan = _vocal_lock_plans(12.0)
    with pytest.raises(ValueError, match="exactly one object per planned scene"):
        mv.build_mv_vocal_lock_visual_prompt_plan(
            scene_plan,
            "performance",
            "same performer",
            "cinematic",
            '[{"camera":"front"}]',
        )
    unsafe = [
        {
            "camera": "front medium close-up",
            "lighting": "a giant portrait projection behind the performer",
        }
        for _scene in scene_plan["scenes"]
    ]
    with pytest.raises(ValueError, match="single-subject visual contract"):
        mv.build_mv_vocal_lock_visual_prompt_plan(
            scene_plan,
            "performance",
            "same performer",
            "cinematic",
            json.dumps(unsafe),
        )


def test_vocal_lock_v3_visual_director_has_safe_deterministic_fallback_arc():
    scene_plan, _prompt_plan = _vocal_lock_plans(12.0)
    first = mv.build_mv_vocal_lock_visual_prompt_plan(
        scene_plan, "performance", "same performer", "cinematic"
    )[0]
    second = mv.build_mv_vocal_lock_visual_prompt_plan(
        scene_plan, "performance", "same performer", "cinematic"
    )[0]
    assert first == second
    assert first["scene_direction_source"] == "deterministic_safe_studio_arc"
    assert first["segments"][0]["scene_direction"]["camera"].startswith("locked frontal")


def test_vocal_lock_audio_contract_rejects_timeline_mismatch_and_silence():
    scene_plan, _prompt_plan = _vocal_lock_plans()
    total_frames = scene_plan["total_frames"]
    with pytest.raises(ValueError, match="align both audio inputs"):
        mv._validate_vocal_lock_audio_contract(
            _audio(5.152), _audio(4.0), total_frames
        )
    silent = _audio(5.152)
    silent["waveform"].zero_()
    with pytest.raises(ValueError, match="effectively silent"):
        mv._validate_vocal_lock_audio_contract(
            _audio(5.152), silent, total_frames
        )


def test_vocal_lock_audio_sources_keep_full_song_out_of_h3_and_candidate_segments():
    full_song = _audio(5.152)
    vocal_lock_audio = _audio(5.152)
    sources = mv._mv_audio_sources(full_song, vocal_lock_audio)
    assert sources["conditioning"] is vocal_lock_audio
    assert sources["candidate_preview"] is vocal_lock_audio
    assert sources["final_delivery"] is full_song
    assert sources["policy"] == "v2_isolated_vocal_conditioning_full_song_final_mux"


def test_vocal_lock_wrapper_binds_isolated_drive_and_v2_contract(monkeypatch):
    _scene_plan, prompt_plan = _vocal_lock_plans()
    full_song = _audio(5.152)
    vocal_lock_audio = _audio(5.152)
    vocal_lock_audio["waveform"] *= 0.5
    captured = {}

    def run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "video.mp4", "manifest.json", 1, "complete", json.dumps({"status": "complete"})

    monkeypatch.setattr(mv, "run_local_mv_in_node_loop", run)
    result = mv.run_local_mv_vocal_lock_in_node_loop(
        object(),
        object(),
        object(),
        object(),
        torch.zeros((1, 32, 32, 3)),
        full_song,
        vocal_lock_audio,
        prompt_plan,
        chain_id="v2-contract",
        width=736,
        height=416,
        base_seed=1,
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
    assert captured["args"][5] is full_song
    assert captured["kwargs"]["vocal_lock_audio"] is vocal_lock_audio
    assert captured["kwargs"]["route_revision"] == "vocal_lock_v2"
    assert captured["kwargs"]["loop_state_name"] == mv.MV_VOCAL_LOCK_LOOP_STATE_NAME
    report = json.loads(result[4])
    assert report["conditioning_audio"] == "vocal_lock_audio"
    assert report["delivery_audio"] == "full_song_muxed_once"


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
    assert MV_LIPSYNC_V2_ADVANCED_NODE_CLASSES == [
        MiniMaxH3MVVocalLockScenePlannerV2T8Advanced,
        MiniMaxH3MVVocalLockPromptCompilerV2T8Advanced,
        MiniMaxH3LocalMVVocalLockRendererV2T8Advanced,
    ]
    assert MV_LIPSYNC_V3_ADVANCED_NODE_CLASSES == [
        MiniMaxH3MVVocalLockVisualDirectorV3T8Advanced,
        MiniMaxH3LocalMVVocalLockVisualRendererV3T8Advanced,
    ]
    registered = asyncio.run(comfy_entrypoint().get_node_list())
    assert registered[268:271] == MV_LIPSYNC_ADVANCED_NODE_CLASSES
    assert registered[271:274] == MV_LIPSYNC_V2_ADVANCED_NODE_CLASSES
    assert registered[274:276] == MV_LIPSYNC_V3_ADVANCED_NODE_CLASSES
    renderer = MiniMaxH3LocalMVInNodeRendererT8Advanced.define_schema()
    assert renderer.is_output_node is True
    assert renderer.is_experimental is True
    ids = {item.id for item in renderer.inputs}
    assert {"model", "clip", "video_vae", "audio_vae", "full_song"} <= ids
    assert not ({"api_url", "api_key", "endpoint", "server_url"} & ids)

    vocal_lock_renderer = MiniMaxH3LocalMVVocalLockRendererV2T8Advanced.define_schema()
    vocal_lock_ids = {item.id for item in vocal_lock_renderer.inputs}
    assert {"full_song", "vocal_lock_audio", "mv_vocal_lock_prompt_plan"} <= vocal_lock_ids
    assert not ({"api_url", "api_key", "endpoint", "server_url"} & vocal_lock_ids)
    visual_renderer = MiniMaxH3LocalMVVocalLockVisualRendererV3T8Advanced.define_schema()
    visual_ids = {item.id for item in visual_renderer.inputs}
    assert {"full_song", "vocal_lock_audio", "mv_vocal_lock_prompt_plan"} <= visual_ids
    assert not ({"api_url", "api_key", "endpoint", "server_url"} & visual_ids)


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


def test_vocal_lock_v2_frontend_workflow_has_two_audio_roles_and_eight_steps():
    workflow = json.loads(VOCAL_LOCK_V2_WORKFLOW_PATH.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert len([node for node in nodes.values() if node["type"] == "LoadAudio"]) == 2
    planner = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3MVVocalLockScenePlannerV2T8Advanced"
    )
    renderer = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3LocalMVVocalLockRendererV2T8Advanced"
    )
    compiler = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3MVVocalLockPromptCompilerV2T8Advanced"
    )
    planner_values = plugin_widget_map(
        planner, MiniMaxH3MVVocalLockScenePlannerV2T8Advanced
    )
    renderer_values = plugin_widget_map(
        renderer, MiniMaxH3LocalMVVocalLockRendererV2T8Advanced
    )
    compiler_values = plugin_widget_map(
        compiler, MiniMaxH3MVVocalLockPromptCompilerV2T8Advanced
    )
    assert planner_values["vocal_active_ratio"] == 0.12
    assert renderer_values["steps"] == 8
    assert renderer_values["width"] == 736
    assert renderer_values["height"] == 416
    assert renderer_values["resume_existing"] is True
    assert compiler_values["camera_pattern"].startswith("locked-off static camera")
    assert "slow push-in" not in compiler_values["camera_pattern"]
    assert "handheld" not in compiler_values["camera_pattern"]
    note_text = "\n".join(
        str(node["widgets_values"][0])
        for node in nodes.values()
        if node["type"] == "MarkdownNote"
    )
    assert "vocal_lock_audio" in note_text
    assert "full_song" in note_text
    assert "不进入 H3" in note_text
    assert "不会提交HTTP `/prompt`" in note_text


def test_vocal_lock_v3_workflow_adds_visual_contract_and_same_reference_prefix_cache():
    workflow = json.loads(VOCAL_LOCK_V3_WORKFLOW_PATH.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    director = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3MVVocalLockVisualDirectorV3T8Advanced"
    )
    renderer = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3LocalMVVocalLockVisualRendererV3T8Advanced"
    )
    cache = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3QwenReferencePrefixCacheT8Advanced"
    )
    director_values = plugin_widget_map(
        director, MiniMaxH3MVVocalLockVisualDirectorV3T8Advanced
    )
    renderer_values = plugin_widget_map(
        renderer, MiniMaxH3LocalMVVocalLockVisualRendererV3T8Advanced
    )
    cache_values = plugin_widget_map(
        cache, MiniMaxH3QwenReferencePrefixCacheT8Advanced
    )
    assert director_values["scene_directions_json"] == ""
    assert renderer_values["width"] == 1024
    assert renderer_values["height"] == 768
    assert renderer_values["steps"] == 4
    assert renderer_values["shift_video"] == 12.0
    assert renderer_values["shift_audio"] == 3.0
    assert renderer_values["sampler_name"] == "euler"
    assert renderer_values["scheduler"] == "simple"
    assert renderer_values["chain_id"].endswith("v3_visual")
    lora = nodes[2]
    assert lora["type"] == "LoraLoaderBypassModelOnly"
    assert lora["widgets_values"] == [
        "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        1.0,
    ]
    assert cache_values == {
        "mode": "memory_lru_exp",
        "max_entries": 1,
        "maximum_cache_mib": 1024.0,
        "cache_epoch": 0,
    }
    clip_link = next(link for link in workflow["links"] if link[3] == renderer["id"] and link[4] == 1)
    assert nodes[clip_link[1]]["type"] == "MiniMaxH3QwenReferencePrefixCacheT8Advanced"
    note_text = "\n".join(
        str(node["widgets_values"][0])
        for node in nodes.values()
        if node["type"] == "MarkdownNote"
    )
    assert "恰好一名人物和一张人脸" in note_text
    assert "accepted只表示文件和合同已落盘" in note_text
    assert "32秒/5镜通过后才允许约90秒终验" in note_text
    assert VOCAL_LOCK_V3_USER_WORKFLOW_PATH.read_bytes() == VOCAL_LOCK_V3_WORKFLOW_PATH.read_bytes()


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
