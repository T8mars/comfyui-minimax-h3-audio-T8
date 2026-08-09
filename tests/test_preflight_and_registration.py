from __future__ import annotations

import asyncio
import json
from pathlib import Path

import torch

import h3_audio_t8_pkg
from h3_audio_t8_pkg.preflight import run_preflight
from helpers import FakeAudioVAE, FakeVideoVAE, make_audio


def test_all_nodes_register_with_unique_ids_and_valid_schemas():
    extension = h3_audio_t8_pkg.comfy_entrypoint()
    node_classes = asyncio.run(extension.get_node_list())
    schemas = [node.define_schema() for node in node_classes]
    ids = [schema.node_id for schema in schemas]
    assert len(ids) == 25
    assert len(ids) == len(set(ids))
    assert "MiniMaxH3AudioConditioningT8" in ids
    assert "MiniMaxH3DualClockSamplerT8" in ids
    assert "MiniMaxH3MultiRateSamplerEXPT8" in ids
    assert "MiniMaxH3StillConditioningT8" in ids
    assert "MiniMaxH3StillPreflightT8" in ids
    assert "MiniMaxH3StillDecodeT8" in ids
    assert ids[:14] == [
        "MiniMaxH3AudioConditioningT8",
        "MiniMaxH3AudioLatentControlT8",
        "MiniMaxH3DurationPlannerT8",
        "MiniMaxH3AudioWindowT8",
        "MiniMaxH3PromptTagsT8",
        "MiniMaxH3AVDecodeT8",
        "MiniMaxH3AudioMixT8",
        "MiniMaxH3OutputTrimT8",
        "MiniMaxH3PreflightT8",
        "MiniMaxH3DualClockSamplerT8",
        "MiniMaxH3MultiRateSamplerEXPT8",
        "MiniMaxH3StillConditioningT8",
        "MiniMaxH3StillPreflightT8",
        "MiniMaxH3StillDecodeT8",
    ]
    long_video_ids = {
        "MiniMaxH3LongVideoPlannerT8",
        "MiniMaxH3LongVideoContextLoadT8",
        "MiniMaxH3LongVideoConditioningT8",
        "MiniMaxH3LongVideoContextSaveT8",
        "MiniMaxH3LongVideoCandidateSaveT8",
        "MiniMaxH3LongVideoAcceptCandidateT8",
        "MiniMaxH3LongVideoAcceptedContextLoadT8",
        "MiniMaxH3LongVideoComposeAcceptedT8",
        "MiniMaxH3LongVideoOrchestratorT8",
        "MiniMaxH3LongVideoBackgroundStartT8",
        "MiniMaxH3LongVideoAutoQueueT8",
    }
    assert long_video_ids <= set(ids)
    exp_schema = schemas[ids.index("MiniMaxH3MultiRateSamplerEXPT8")]
    assert exp_schema.is_experimental is True
    assert exp_schema.category == "T8/MiniMax H3/Audio/Experimental"
    for still_id in {
        "MiniMaxH3StillConditioningT8",
        "MiniMaxH3StillPreflightT8",
        "MiniMaxH3StillDecodeT8",
    }:
        still_schema = schemas[ids.index(still_id)]
        assert still_schema.is_experimental is True
        assert still_schema.category == "T8/MiniMax H3/Still/Experimental"
    for long_video_id in long_video_ids:
        long_video_schema = schemas[ids.index(long_video_id)]
        assert long_video_schema.is_experimental is True
        assert long_video_schema.category == "T8/MiniMax H3/Long Video/Experimental"
    assert ids[-2:] == [
        "MiniMaxH3LongVideoBackgroundStartT8",
        "MiniMaxH3LongVideoAutoQueueT8",
    ]

    background_start = schemas[ids.index("MiniMaxH3LongVideoBackgroundStartT8")]
    mode = next(item for item in background_start.inputs if item.id == "execution_mode")
    release = next(item for item in background_start.inputs if item.id == "release_policy")
    assert mode.default == "review_only"
    assert mode.options == ["review_only", "auto_accept_and_continue"]
    assert release.default == "clear_execution_cache"
    assert release.options == [
        "keep_loaded",
        "clear_execution_cache",
        "unload_all_models",
    ]

    long_conditioning = schemas[ids.index("MiniMaxH3LongVideoConditioningT8")]
    first_frame_reuse = next(
        item for item in long_conditioning.inputs if item.id == "first_frame_reuse"
    )
    assert long_conditioning.inputs[-3].id == "first_frame_reuse"
    assert long_conditioning.inputs[-2].id == "persistent_identity_image"
    assert long_conditioning.inputs[-2].optional is True
    strategy = long_conditioning.inputs[-1]
    assert strategy.id == "persistent_identity_strategy"
    assert strategy.default == "single_reference"
    assert strategy.options == ["single_reference", "scene_plus_identity"]
    assert first_frame_reuse.default == "segment0_only"
    assert first_frame_reuse.options == [
        "segment0_only",
        "persistent_identity_reference",
    ]


def test_task_type_frontend_labels_preserve_canonical_backend_values():
    extension = h3_audio_t8_pkg.comfy_entrypoint()
    node_classes = asyncio.run(extension.get_node_list())
    conditioning = next(
        node for node in node_classes
        if node.define_schema().node_id == "MiniMaxH3AudioConditioningT8"
    )
    task_type = next(
        item for item in conditioning.define_schema().inputs
        if item.id == "task_type"
    )
    assert task_type.options == [
        "auto", "T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA", "Hybrid",
    ]

    package_root = Path(__file__).resolve().parents[1]
    assert h3_audio_t8_pkg.WEB_DIRECTORY == "./web"
    frontend = (package_root / "web" / "task_type_labels.js").read_text(encoding="utf-8")
    for label in {
        "auto — 自动判断",
        "T2VA — 文生音视频",
        "I2VA — 图生音视频（首帧）",
        "FL2VA — 首尾帧生音视频",
        "L2VA — 尾帧生音视频",
        "Ref2VA — 参考生音视频",
        "Hybrid — 关键帧+参考混合生成",
    }:
        assert label in frontend
    assert '"MiniMaxH3LongVideoConditioningT8"' in frontend
    assert "toBackendValue(widget.value)" in frontend

    background_frontend = (
        package_root / "web" / "long_video_background.js"
    ).read_text(encoding="utf-8")
    assert '"MiniMaxH3LongVideoBackgroundStartT8"' in background_frontend
    for action in {"pause", "resume", "cancel"}:
        assert f'"{action}"' in background_frontend
    assert "/minimax_h3_t8/long_video/background" in background_frontend


def test_dual_clock_sampler_appends_optional_choices_without_reordering_legacy_widgets():
    extension = h3_audio_t8_pkg.comfy_entrypoint()
    node_classes = asyncio.run(extension.get_node_list())
    sampler_node = next(
        node for node in node_classes
        if node.define_schema().node_id == "MiniMaxH3DualClockSamplerT8"
    )
    inputs = sampler_node.define_schema().inputs

    assert [item.id for item in inputs] == [
        "model",
        "av_latent",
        "steps",
        "shift_video",
        "shift_audio",
        "sampler_name",
        "scheduler",
    ]
    sampler_name = inputs[-2]
    scheduler = inputs[-1]
    assert sampler_name.optional is True
    assert sampler_name.default == "dual_clock_euler"
    assert sampler_name.options[0] == "dual_clock_euler"
    assert "euler" in sampler_name.options
    assert scheduler.optional is True
    assert scheduler.default == "native_flow"
    assert scheduler.options[0] == "native_flow"
    assert "normal" in scheduler.options


def test_preflight_reports_alignment_audio_and_reference_guidance():
    ready, warning_count, report = run_preflight(
        1344, 768, 123, "lock_source", video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        drive_audio=make_audio(1, value=0),
        ref_videos={"ref_video_1": torch.zeros((20, 32, 32, 3))},
    )
    data = json.loads(report)
    assert ready is True
    assert warning_count >= 3
    assert data["facts"]["aligned_frames"] == 124
    assert data["facts"]["video_vae_kind"] == "video"
    assert not any("swapped" in warning for warning in data["warnings"])


def test_preflight_allows_1080p_area_and_blocks_only_above_it():
    ready, warning_count, report = run_preflight(1920, 1088, 124, "native")
    data = json.loads(report)
    assert ready is True
    assert warning_count >= 1
    assert data["facts"]["pixels"] == 1920 * 1088
    assert any("VRAM" in warning for warning in data["warnings"])

    ready, _, report = run_preflight(1952, 1088, 124, "lock_source")
    assert ready is False
    assert len(json.loads(report)["errors"]) == 2


def test_preflight_distinguishes_h3_video_and_audio_vaes_by_latent_contract():
    ready, _, report = run_preflight(
        1344,
        768,
        124,
        "native",
        video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(),
    )
    data = json.loads(report)
    assert ready is True
    assert data["facts"]["video_vae_kind"] == "video"
    assert data["facts"]["audio_vae_kind"] == "audio"

    ready, _, report = run_preflight(
        1344,
        768,
        124,
        "native",
        video_vae=FakeAudioVAE(),
        audio_vae=FakeVideoVAE(),
    )
    data = json.loads(report)
    assert ready is False
    assert any("video_vae is an H3 audio VAE" in error for error in data["errors"])
    assert any("audio_vae is an H3 video VAE" in error for error in data["errors"])


def test_example_api_workflow_is_valid_and_references_existing_nodes():
    path = Path(__file__).resolve().parents[1] / "examples" / "audio_lock_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    custom_types = {value["class_type"] for value in workflow.values() if value["class_type"].endswith("T8")}
    assert custom_types == {
        "MiniMaxH3AudioWindowT8", "MiniMaxH3AudioConditioningT8",
        "MiniMaxH3AVDecodeT8", "MiniMaxH3OutputTrimT8",
    }
    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_dual_clock_example_uses_one_coherent_sampling_setup():
    path = Path(__file__).resolve().parents[1] / "examples" / "dual_clock_4step_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    dual_nodes = [value for value in workflow.values() if value["class_type"] == "MiniMaxH3DualClockSamplerT8"]
    assert len(dual_nodes) == 1
    assert dual_nodes[0]["inputs"]["steps"] == 4
    assert not any(value["class_type"] in {"MiniMaxH3SigmaShift", "KSamplerSelect", "BasicScheduler"}
                   for value in workflow.values())
    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_long_video_api_example_is_isolated_retry_safe_and_trimmed():
    path = Path(__file__).resolve().parents[1] / "examples" / "long_video_segment_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    custom_types = {
        value["class_type"] for value in workflow.values()
        if value["class_type"].startswith("MiniMaxH3")
    }
    assert {
        "MiniMaxH3LongVideoPlannerT8",
        "MiniMaxH3LongVideoContextLoadT8",
        "MiniMaxH3LongVideoConditioningT8",
        "MiniMaxH3LongVideoContextSaveT8",
        "MiniMaxH3DualClockSamplerT8",
        "MiniMaxH3AVDecodeT8",
        "MiniMaxH3OutputTrimT8",
    } <= custom_types
    planner_id = next(key for key, value in workflow.items()
                      if value["class_type"] == "MiniMaxH3LongVideoPlannerT8")
    save = next(value for value in workflow.values()
                if value["class_type"] == "MiniMaxH3LongVideoContextSaveT8")
    assert save["inputs"]["save_context"] == [planner_id, 8]
    class_types = {value["class_type"] for value in workflow.values()}
    assert {"CreateVideo", "SaveVideo"} <= class_types
    assert "VHS_VideoCombine" not in class_types
    assert not any(value["class_type"].startswith("MiniMaxH3Motion") for value in workflow.values())
    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_long_video_candidate_api_separates_preview_from_accepted_state():
    path = Path(__file__).resolve().parents[1] / "examples" / "long_video_candidate_accept_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    types = {value["class_type"] for value in workflow.values()}
    assert {
        "MiniMaxH3LongVideoPlannerT8",
        "MiniMaxH3LongVideoAcceptedContextLoadT8",
        "MiniMaxH3LongVideoConditioningT8",
        "MiniMaxH3LongVideoCandidateSaveT8",
        "MiniMaxH3LongVideoAcceptCandidateT8",
    } <= types
    assert "MiniMaxH3LongVideoContextSaveT8" not in types
    accepted_loader_id = next(
        key for key, value in workflow.items()
        if value["class_type"] == "MiniMaxH3LongVideoAcceptedContextLoadT8"
    )
    candidate = next(
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3LongVideoCandidateSaveT8"
    )
    assert candidate["inputs"]["parent_candidate_id"] == [accepted_loader_id, 2]
    assert candidate["inputs"]["parent_manifest_revision"] == [accepted_loader_id, 3]
    seed_id = next(
        key for key, value in workflow.items() if value["class_type"] == "PrimitiveInt"
    )
    noise = next(value for value in workflow.values() if value["class_type"] == "RandomNoise")
    assert noise["inputs"]["noise_seed"] == [seed_id, 0]
    assert candidate["inputs"]["seed"] == [seed_id, 0]
    conditioning_id = next(
        key for key, value in workflow.items()
        if value["class_type"] == "MiniMaxH3LongVideoConditioningT8"
    )
    assert candidate["inputs"]["prompt"] == [conditioning_id, 4]
    review = next(
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3LongVideoAcceptCandidateT8"
    )
    assert review["inputs"]["accept_candidate"] is False
    assert review["inputs"]["replace_policy"] == "reject_existing"
    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_long_video_compose_api_requires_an_explicit_final_segment():
    path = Path(__file__).resolve().parents[1] / "examples" / "long_video_compose_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    assert list(workflow.values())[0]["class_type"] == "MiniMaxH3LongVideoComposeAcceptedT8"
    assert list(workflow.values())[0]["inputs"]["require_final_segment"] is True
    assert list(workflow.values())[0]["inputs"]["audio_seam_policy"] == "cosine_bridge"


def test_long_video_auto_resume_api_drives_segment_prompt_and_seed_from_one_plan():
    path = Path(__file__).resolve().parents[1] / "examples" / "long_video_auto_resume_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    orchestrator_id = next(
        key for key, value in workflow.items()
        if value["class_type"] == "MiniMaxH3LongVideoOrchestratorT8"
    )
    orchestrator = workflow[orchestrator_id]
    assert orchestrator["inputs"]["total_duration_seconds"] == 60.0
    assert orchestrator["inputs"]["render_window_frames"] == 124
    assert orchestrator["inputs"]["context_frames"] == 22
    assert not any(value["class_type"] == "PrimitiveInt" for value in workflow.values())
    conditioning = next(
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3LongVideoConditioningT8"
    )
    noise = next(value for value in workflow.values() if value["class_type"] == "RandomNoise")
    candidate = next(
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3LongVideoCandidateSaveT8"
    )
    assert conditioning["inputs"]["prompt"] == [orchestrator_id, 10]
    assert noise["inputs"]["noise_seed"] == [orchestrator_id, 11]
    assert candidate["inputs"]["seed"] == [orchestrator_id, 11]
    sampler = next(
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3DualClockSamplerT8"
    )
    assert sampler["inputs"]["steps"] == [orchestrator_id, 16]
    assert sampler["inputs"]["shift_video"] == [orchestrator_id, 17]
    assert sampler["inputs"]["shift_audio"] == [orchestrator_id, 18]
    assert sampler["inputs"]["sampler_name"] == [orchestrator_id, 19]
    assert sampler["inputs"]["scheduler"] == [orchestrator_id, 20]
    assert candidate["inputs"]["sampling_summary"] == [orchestrator_id, 21]
    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_long_video_background_api_is_explicit_and_queues_through_one_terminal():
    path = Path(__file__).resolve().parents[1] / "examples" / "long_video_background_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    start_id = next(
        key for key, value in workflow.items()
        if value["class_type"] == "MiniMaxH3LongVideoBackgroundStartT8"
    )
    start = workflow[start_id]
    assert start["inputs"] == {
        "chain_id": "h3_background_demo",
        "execution_mode": "auto_accept_and_continue",
        "max_retries": 1,
        "retry_delay_seconds": 2.0,
        "release_policy": "clear_execution_cache",
    }
    orchestrator = next(
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3LongVideoOrchestratorT8"
    )
    assert orchestrator["inputs"]["chain_id"] == [start_id, 0]
    terminal = next(
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3LongVideoAutoQueueT8"
    )
    assert terminal["inputs"]["job_id"] == [start_id, 2]
    assert terminal["inputs"]["auto_accept"] == [start_id, 1]
    assert terminal["inputs"]["compose_when_complete"] is True
    assert not any(
        value["class_type"] == "MiniMaxH3LongVideoAcceptCandidateT8"
        for value in workflow.values()
    )
    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_long_video_frontend_workflow_has_consistent_links_and_no_global_motion_node():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples" / "workflows" / "H3_Long_Video_22F_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert len(nodes) == len(workflow["nodes"])
    assert workflow["last_link_id"] == len(workflow["links"])
    assert not any(node["type"].startswith("MiniMaxH3Motion") for node in nodes.values())
    node_types = {node["type"] for node in nodes.values()}
    assert {"CreateVideo", "SaveVideo"} <= node_types
    assert "VHS_VideoCombine" not in node_types
    for link_id, source, output_slot, target, input_slot, _ in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])


def test_long_video_accepted_frontend_workflow_is_review_first_and_consistent():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples" / "workflows" / "H3_Long_Video_Accepted_22F_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert len(nodes) == len(workflow["nodes"])
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    types = {node["type"] for node in nodes.values()}
    assert {
        "MiniMaxH3LongVideoAcceptedContextLoadT8",
        "MiniMaxH3LongVideoCandidateSaveT8",
        "MiniMaxH3LongVideoAcceptCandidateT8",
        "PrimitiveInt",
    } <= types
    assert "MiniMaxH3LongVideoContextLoadT8" not in types
    assert "MiniMaxH3LongVideoContextSaveT8" not in types
    assert "CreateVideo" not in types and "SaveVideo" not in types
    review = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoAcceptCandidateT8"
    )
    assert review["widgets_values"] == [False, "reject_existing", True]
    candidate = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoCandidateSaveT8"
    )
    assert candidate["inputs"][7]["name"] == "parent_candidate_id"
    assert candidate["inputs"][7]["link"] is not None
    assert candidate["inputs"][8]["name"] == "parent_manifest_revision"
    assert candidate["inputs"][8]["link"] is not None
    seed_node = next(node for node in nodes.values() if node["type"] == "PrimitiveInt")
    noise = next(node for node in nodes.values() if node["type"] == "RandomNoise")
    assert noise["inputs"][0]["link"] in seed_node["outputs"][0]["links"]
    assert candidate["inputs"][13]["link"] in seed_node["outputs"][0]["links"]
    conditioning = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoConditioningT8"
    )
    assert candidate["inputs"][12]["link"] in conditioning["outputs"][4]["links"]
    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_long_video_auto_resume_frontend_workflow_has_one_timeline_source():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples" / "workflows" / "H3_Long_Video_Auto_Resume_22F_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    types = {node["type"] for node in nodes.values()}
    assert "MiniMaxH3LongVideoOrchestratorT8" in types
    assert "MiniMaxH3LongVideoPlannerT8" not in types
    assert "PrimitiveInt" not in types
    orchestrator = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoOrchestratorT8"
    )
    assert orchestrator["widgets_values"][1:4] == [60.0, 124, 22]
    assert orchestrator["widgets_values"][7] == "increment"
    assert orchestrator["widgets_values"][8:] == [
        4, 12.0, 3.0, "dual_clock_euler", "native_flow",
    ]
    conditioning = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoConditioningT8"
    )
    noise = next(node for node in nodes.values() if node["type"] == "RandomNoise")
    candidate = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoCandidateSaveT8"
    )
    assert conditioning["inputs"][8]["link"] in orchestrator["outputs"][10]["links"]
    assert noise["inputs"][0]["link"] in orchestrator["outputs"][11]["links"]
    assert candidate["inputs"][13]["link"] in orchestrator["outputs"][11]["links"]
    sampler = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3DualClockSamplerT8"
    )
    for input_slot, output_slot in zip(range(2, 7), range(16, 21), strict=True):
        assert sampler["inputs"][input_slot]["link"] in (
            orchestrator["outputs"][output_slot]["links"]
        )
    assert candidate["inputs"][11]["link"] in orchestrator["outputs"][21]["links"]
    review = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoAcceptCandidateT8"
    )
    assert review["widgets_values"][0] is False
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_long_video_background_frontend_workflow_has_explicit_controller_links():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples" / "workflows" / "H3_Long_Video_Background_22F_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    start = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoBackgroundStartT8"
    )
    terminal = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoAutoQueueT8"
    )
    orchestrator = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoOrchestratorT8"
    )
    assert start["widgets_values"] == [
        "h3_background_demo",
        "auto_accept_and_continue",
        1,
        2.0,
        "clear_execution_cache",
    ]
    assert "MiniMaxH3LongVideoAcceptCandidateT8" not in {
        node["type"] for node in nodes.values()
    }
    links = {link[0]: link for link in workflow["links"]}
    chain_link = links[orchestrator["inputs"][0]["link"]]
    job_link = links[terminal["inputs"][1]["link"]]
    auto_link = links[terminal["inputs"][2]["link"]]
    assert chain_link[1:5] == [start["id"], 0, orchestrator["id"], 0]
    assert job_link[1:5] == [start["id"], 2, terminal["id"], 1]
    assert auto_link[1:5] == [start["id"], 1, terminal["id"], 2]
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(links)
    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_scene_plus_identity_background_workflow_wires_two_images_and_exp_policy():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "H3_Long_Video_Background_22F_ScenePlusIdentity_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}

    conditioning = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoConditioningT8"
    )
    full_scene = next(
        node for node in nodes.values()
        if node.get("title", "").startswith("0a. Full scene")
    )
    identity_crop = next(
        node for node in nodes.values()
        if node.get("title", "").startswith("0b. Same-subject")
    )
    inputs = {value["name"]: value for value in conditioning["inputs"]}

    assert conditioning["widgets_values"][-2:] == [
        "persistent_identity_reference",
        "scene_plus_identity",
    ]
    assert links[inputs["first_frame"]["link"]][1:5] == [
        full_scene["id"], 0, conditioning["id"], 22,
    ]
    assert links[inputs["persistent_identity_image"]["link"]][1:5] == [
        identity_crop["id"], 0, conditioning["id"], 24,
    ]

    start = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoBackgroundStartT8"
    )
    orchestrator = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoOrchestratorT8"
    )
    assert start["widgets_values"][0] == orchestrator["widgets_values"][0]
    assert start["widgets_values"][0] == "h3_background_scene_identity_demo"
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(links)

    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_background_control_routes_offload_blocking_manager_calls():
    source = (
        Path(__file__).resolve().parents[1] / "long_video_routes.py"
    ).read_text(encoding="utf-8")
    assert "await asyncio.to_thread(BACKGROUND_JOBS.pause, chain_id)" in source
    assert "await asyncio.to_thread(BACKGROUND_JOBS.resume, chain_id)" in source
    assert "await asyncio.to_thread(BACKGROUND_JOBS.cancel, chain_id)" in source


def test_multirate_exp_example_is_independent_and_uses_eight_joint_calls():
    path = Path(__file__).resolve().parents[1] / "examples" / "multirate_exp_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    exp_nodes = [
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3MultiRateSamplerEXPT8"
    ]
    assert len(exp_nodes) == 1
    assert exp_nodes[0]["inputs"]["video_steps"] == 4
    assert exp_nodes[0]["inputs"]["audio_steps"] == 8
    assert not any(
        value["class_type"] in {
            "MiniMaxH3DualClockSamplerT8",
            "MiniMaxH3SigmaShift",
            "KSamplerSelect",
            "BasicScheduler",
        }
        for value in workflow.values()
    )
    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_still_image_edit_example_uses_ref2va_without_incompatible_lora():
    path = Path(__file__).resolve().parents[1] / "examples" / "still_image_edit_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    types = {value["class_type"] for value in workflow.values()}
    assert "MiniMaxH3StillConditioningT8" in types
    assert "MiniMaxH3StillPreflightT8" in types
    assert "MiniMaxH3StillDecodeT8" in types
    assert "MiniMaxH3DualClockSamplerT8" in types
    assert not any("Lora" in node_type for node_type in types)

    unet = next(value for value in workflow.values() if value["class_type"] == "UNETLoader")
    assert "ref2va" in unet["inputs"]["unet_name"]
    conditioning = next(
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3StillConditioningT8"
    )
    assert conditioning["inputs"]["target_mode"] == "short_video_22_frames"
    assert conditioning["inputs"]["canvas_mode"] == "custom"
    assert conditioning["inputs"]["width"] == 512
    assert conditioning["inputs"]["height"] == 512
    sampler = next(
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3DualClockSamplerT8"
    )
    assert sampler["inputs"]["steps"] == 20

    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_frontend_workflows_cover_stable_and_both_exp_step_counts():
    workflow_dir = Path(__file__).resolve().parents[1] / "examples" / "workflows"
    expected = {
        "H3_Turbo_Stable_4V4A.json": ("MiniMaxH3DualClockSamplerT8", [4, 12.0, 3.0]),
        "H3_Turbo_EXP_4V8A.json": ("MiniMaxH3MultiRateSamplerEXPT8", [4, 8, 12.0, 3.0]),
        "H3_Turbo_EXP_4V10A.json": ("MiniMaxH3MultiRateSamplerEXPT8", [4, 10, 12.0, 3.0]),
    }

    for filename, (sampler_type, sampler_widgets) in expected.items():
        workflow = json.loads((workflow_dir / filename).read_text(encoding="utf-8"))
        assert workflow["version"] == 0.4
        assert workflow["last_node_id"] == max(node["id"] for node in workflow["nodes"])
        assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])

        nodes = {node["id"]: node for node in workflow["nodes"]}
        types = {node["type"] for node in nodes.values()}
        assert "LoraLoaderBypassModelOnly" in types
        assert "MiniMaxH3SigmaShift" not in types
        assert "BasicScheduler" not in types
        assert "KSamplerSelect" not in types

        sampler_nodes = [node for node in nodes.values() if node["type"] == sampler_type]
        assert len(sampler_nodes) == 1
        assert sampler_nodes[0]["widgets_values"] == sampler_widgets

        unet = next(node for node in nodes.values() if node["type"] == "UNETLoader")
        assert unet["widgets_values"][0] == "minimax_h3_fl2va_int8_convrot.safetensors"
        lora = next(
            node for node in nodes.values() if node["type"] == "LoraLoaderBypassModelOnly"
        )
        assert lora["widgets_values"][0] == (
            "minimax_h3_turbo_4步加速ema_comfyui.safetensors"
        )

        for link_id, source_id, source_slot, target_id, target_slot, link_type in workflow["links"]:
            source = nodes[source_id]
            target = nodes[target_id]
            assert link_id in source["outputs"][source_slot]["links"]
            assert target["inputs"][target_slot]["link"] == link_id
            assert source["outputs"][source_slot]["type"] == link_type
            assert target["inputs"][target_slot]["type"] == link_type


def test_frontend_audio_input_workflows_cover_three_source_modes_and_output_routing():
    workflow_dir = Path(__file__).resolve().parents[1] / "examples" / "workflows"
    expected = {
        "H3_Audio_Lock_Source_Stable_4V4A.json": ("lock_source", 6),
        "H3_Audio_Remix_Source_Stable_4V4A.json": ("remix_source", 11),
        "H3_Audio_Reference_Only_Stable_4V4A.json": ("reference_only", 11),
    }

    for filename, (audio_mode, final_audio_source_id) in expected.items():
        workflow = json.loads((workflow_dir / filename).read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}
        conditioning = next(
            node for node in nodes.values()
            if node["type"] == "MiniMaxH3AudioConditioningT8"
        )
        audio_window = next(
            node for node in nodes.values()
            if node["type"] == "MiniMaxH3AudioWindowT8"
        )
        output_trim = next(
            node for node in nodes.values()
            if node["type"] == "MiniMaxH3OutputTrimT8"
        )
        sampler = next(
            node for node in nodes.values()
            if node["type"] == "MiniMaxH3DualClockSamplerT8"
        )
        conditioning_inputs = {
            value["name"]: value for value in conditioning["inputs"]
        }

        assert workflow["version"] == 0.4
        assert workflow["last_node_id"] == max(nodes)
        assert workflow["last_link_id"] == max(links)
        assert conditioning["widgets_values"][1:3] == [736, 416]
        assert conditioning["widgets_values"][5] == audio_mode
        assert "<Audio 1>" in conditioning["widgets_values"][0]
        assert sampler["widgets_values"] == [
            4, 12.0, 3.0, "dual_clock_euler", "native_flow",
        ]
        assert links[conditioning_inputs["drive_audio"]["link"]][1:5] == [
            audio_window["id"], 0, conditioning["id"], 15,
        ]
        assert links[conditioning_inputs["length"]["link"]][1:5] == [
            audio_window["id"], 1, conditioning["id"], 6,
        ]
        final_audio_link = links[
            next(value for value in output_trim["inputs"] if value["name"] == "audio")["link"]
        ]
        assert final_audio_link[1] == final_audio_source_id
        assert final_audio_link[2] == (2 if audio_mode == "lock_source" else 1)

        for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
            assert nodes[target]["inputs"][input_slot]["link"] == link_id
            assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
            assert nodes[source]["outputs"][output_slot]["type"] == link_type
            assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_frontend_still_edit_workflow_uses_native_22_frame_ref2va_target():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "H3_Still_Edit_22Frames_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(node["id"] for node in workflow["nodes"])
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])

    nodes = {node["id"]: node for node in workflow["nodes"]}
    types = {node["type"] for node in nodes.values()}
    assert {
        "MiniMaxH3StillConditioningT8",
        "MiniMaxH3StillPreflightT8",
        "MiniMaxH3StillDecodeT8",
        "MiniMaxH3DualClockSamplerT8",
        "SaveImage",
    } <= types
    assert not any("Lora" in node_type for node_type in types)

    unet = next(node for node in nodes.values() if node["type"] == "UNETLoader")
    assert unet["widgets_values"][0] == (
        "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    )
    conditioning = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3StillConditioningT8"
    )
    assert conditioning["widgets_values"][1:7] == [
        "custom",
        512,
        512,
        "short_video_22_frames",
        0.999,
        "generate_and_discard",
    ]
    sampler = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3DualClockSamplerT8"
    )
    assert sampler["widgets_values"] == [20, 12.0, 3.0]

    for link_id, source_id, source_slot, target_id, target_slot, link_type in workflow["links"]:
        source = nodes[source_id]
        target = nodes[target_id]
        assert link_id in source["outputs"][source_slot]["links"]
        assert target["inputs"][target_slot]["link"] == link_id
        assert (
            source["outputs"][source_slot]["type"] == link_type
            or link_type == "*"
        )
        assert target["inputs"][target_slot]["type"] == link_type
