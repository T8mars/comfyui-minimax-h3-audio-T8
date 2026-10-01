"""The first V2 I2VA window uses separate real Core lanes and never auto-accepts."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import av
import folder_paths
import pytest
import torch
from comfy_api.latest import io

from h3_audio_t8_pkg.modular_sampling import nodes as stage_nodes, stage_unet_loader
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_job as current_job
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_job_nodes as current_job_nodes
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_conditioning as current_origin
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_stage_attest as current_attest
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_upscale as current_upscale
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_current_handoff as current_handoff
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_job_binding as current_binding
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_media as current_media
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_candidate as current_candidate
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_color_candidate as current_color_candidate
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_review as current_review
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_frozen_low as current_frozen_low
from h3_audio_t8_pkg.modular_sampling import continuation_nodes, fast_h3_v2_continuation_nodes
from h3_audio_t8_pkg.modular_sampling import storage as stage_storage
from h3_audio_t8_pkg import long_video_delivery as delivery
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2_handoff_nodes import NODES as HANDOFF_NODES
from h3_audio_t8_pkg.modular_sampling.stage_unet_loader_nodes import NODES as LOADER_NODES
from h3_audio_t8_pkg.modular_sampling.storage import file_sha
from h3_audio_t8_pkg.modular_sampling.results import selected_model_identity
from h3_audio_t8_pkg.nodes import MiniMaxH3AVDecodeT8, MiniMaxH3OutputTrimT8
from h3_audio_t8_pkg.nodes_fast_h3_v2_advanced import MiniMaxH3FastH3V2RuntimeAuditEXPT8
from h3_audio_t8_pkg.nodes_learned_latent_upscale_advanced import MiniMaxH3TwoPassLatentReconcileT8Advanced
from h3_audio_t8_pkg.nodes_long_video_exp import MiniMaxH3LongVideoContextLoadT8, MiniMaxH3LongVideoPlannerT8
from h3_audio_t8_pkg.nodes_prompt_relay_advanced import MiniMaxH3PromptRelayPlanT8Advanced
from h3_audio_t8_pkg.nodes_prompt_relay_long_video_advanced import (
    MiniMaxH3PromptRelayLongVideoConditioningT8Advanced,
    MiniMaxH3PromptRelayLongVideoPlanT8Advanced,
)
from test_modular_fast_h3_v2_accepted_core import TinyAcceptedUpscale
from test_modular_fast_h3_v2_saved_candidate_core import (
    TinyCandidateCLIP, TinyCandidateUNET, TinyCandidateVAE,
)
from tools import build_modular_fast_h3_v2_first_segment_workflow as builder
from tools import build_modular_fast_h3_v2_continuation_workflow as continuation_builder


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "artifacts/development/modular-sampling-m1-v2-first-segment-20260925"
RECIPE_CANDIDATE = ROOT / "artifacts/development/modular-sampling-m1-v2-current-recipe-first-20260925"
ATTEST_CANDIDATE = ROOT / "artifacts/development/modular-sampling-m1-v2-stage-attest-first-v2-20260925"
ORIGIN_CANDIDATE = ROOT / "artifacts/development/modular-sampling-m1-v2-condition-origin-first-v1-20260925"
HANDOFF_CANDIDATE = ROOT / "artifacts/development/modular-sampling-m1-v2-handoff-origin-first-v1-20260925"
JOB_CANDIDATE = ROOT / "artifacts/development/modular-sampling-m1-v2-job-bound-first-v1-20260925"
SAVE_CANDIDATE = ROOT / "artifacts/development/modular-sampling-m1-v2-candidate-save-first-v1-20260925"
FROZEN_BUNDLE_CANDIDATE = ROOT / "artifacts/development/modular-sampling-m1-v2-frozen-low-first-v1-20260925"
FIRST_IMAGE = "02_清晰身份参考图_首段.png"


def assert_historical_frontend_matches_current(saved_frontend, current_frontend):
    """Keep old snapshots immutable across the opt-in 124-frame widget addition."""
    current = deepcopy(current_frontend)
    saved_nodes = {node["id"]: node for node in saved_frontend["nodes"]}
    for node in current["nodes"]:
        if node["type"] != "MiniMaxH3FastH3V2CurrentRecipeEXPT8":
            continue
        historical = saved_nodes[node["id"]]["widgets_values"]
        assert node["widgets_values"] == [*historical, 90]
        node["widgets_values"] = historical
    assert {key: value for key, value in saved_frontend.items() if key != "id"} == {
        key: value for key, value in current.items() if key != "id"}


class TinyFirstImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="LoadImage", inputs=[io.String.Input("image")],
                         outputs=[io.Image.Output("image")])

    @classmethod
    def execute(cls, image):
        assert image == FIRST_IMAGE
        pixels = torch.linspace(0., 1., 128 * 64 * 3).reshape(1, 64, 128, 3)
        return io.NodeOutput(pixels)


def _tiny(graph):
    graph = deepcopy(graph)
    graph["70"]["inputs"]["chain_id"] = "v2_first_segment_cpu"
    graph["91" if "91" in graph else "23"]["inputs"].update(target_width=128, target_height=64)
    for key in ("9", "89"):
        if key in graph:
            graph[key]["inputs"].update(width=64, height=32, query_chunk_rows=64)
    for key in ("24", "90"):
        if key in graph:
            graph[key]["inputs"].update(width=128, height=64, query_chunk_rows=64)
    for key in ("10", "26"):
        if key in graph:
            graph[key]["inputs"]["min_tokens"] = 0
    for key in ("40", "47"):
        if key in graph:
            graph[key]["inputs"].update(global_prompt="One stable scene.",
                local_prompts="She walks.\nShe stops.\nShe turns.")
    return graph


def _media_hashes(path):
    with av.open(str(path)) as container:
        assert len(container.streams.video) == len(container.streams.audio) == 1
        assert container.streams.video[0].codec_context.name == "h264"
        assert container.streams.audio[0].codec_context.name == "aac"
        frames = list(container.decode(video=0))
        assert len(frames) == 124
        assert all((frame.width, frame.height) == (128, 64) for frame in frames)
        assert [float(frame.pts * frame.time_base) for frame in frames] == pytest.approx(
            [index / 24 for index in range(124)])
        rgb = hashlib.sha256()
        for frame in frames:
            rgb.update(frame.to_ndarray(format="rgb24").tobytes())
        container.seek(0)
        sound = list(container.decode(audio=0))
        assert sound and container.streams.audio[0].codec_context.sample_rate == 32000
        assert sum(frame.samples for frame in sound) >= 124 / 24 * 32000
        assert any(frame.to_ndarray().any() for frame in sound)
        pcm = hashlib.sha256()
        for frame in sound:
            pcm.update(frame.to_ndarray().tobytes())
    return rgb.hexdigest(), pcm.hexdigest()


def test_first_window_graph_has_two_i2va_branches_and_no_acceptance():
    full = builder.first_segment_graph(FIRST_IMAGE)
    frozen = builder.first_segment_graph(FIRST_IMAGE, resume_high=True)
    assert (len(full), len(frozen)) == (40, 29)
    for graph in (full, frozen):
        assert graph["70"]["class_type"] == "MiniMaxH3LongVideoPlannerT8"
        assert graph["70"]["inputs"]["segment_index"] == 0
        assert graph["70"]["inputs"]["new_duration_seconds"] == 124 / 24
        assert graph["71"]["inputs"]["segment_index"] == ["70", 1]
        assert graph["24"]["class_type"] == "MiniMaxH3PromptRelayLongVideoConditioningT8Advanced"
        assert graph["24"]["inputs"]["first_frame"] == ["72", 0]
        assert graph["24"]["inputs"]["model"] == ["22", 0]
        assert graph["24"]["inputs"]["context_frames"] == ["70", 3]
        assert graph["25"]["inputs"]["second_pass_audio_source"] == "legacy_policy"
        assert graph["79"]["inputs"]["start_seconds"] == ["70", 4]
        assert graph["79"]["inputs"]["duration_seconds"] == ["70", 5]
        assert graph["15"]["inputs"]["images"] == ["79", 0]
        assert all(node["class_type"] not in {
            "MiniMaxH3FastH3V2DualModelLongVideoEXPT8", "MiniMaxH3LongVideoCandidateSaveT8",
            "MiniMaxH3LongVideoAcceptCandidateT8", "MiniMaxH3LongVideoComposeAcceptedT8"}
            for node in graph.values())
    assert "9" not in frozen and "10" not in frozen and "13" not in frozen
    assert full["9"]["class_type"] == "MiniMaxH3PromptRelayLongVideoConditioningT8Advanced"
    assert full["9"]["inputs"]["first_frame"] == ["72", 0]
    assert full["9"]["inputs"]["model"] == ["1", 0]
    assert full["9"]["inputs"]["context_frames"] == ["70", 3]
    assert frozen["22"]["inputs"]["completed_stage"] == ["60", 3]


def test_private_first_window_files_match_current_builder_and_static_core():
    audit = json.loads((CANDIDATE / "audit.json").read_text(encoding="utf-8"))
    assert len(audit["candidates"]) == 2
    info = builder.base.load_live_info()
    import nodes as core_nodes
    info["LoadImage"] = builder.base.native_info("LoadImage", core_nodes.NODE_CLASS_MAPPINGS["LoadImage"])
    for resume, label in ((False, "Full"), (True, "Frozen_LOW")):
        expected, _ = builder.base.selected_frontend_schema(
            builder.first_segment_graph(FIRST_IMAGE, resume_high=resume), info)
        path = CANDIDATE / f"FastH3V2_First_{label}_Unaccepted.api.json"
        assert json.loads(path.read_text(encoding="utf-8")) == expected
    assert all(row["all_outputs_valid"] and row["core_validation"][0] is True
               and row["core_validation"][3] == {} for row in audit["candidates"])


def test_opt_in_first_current_recipe_is_separate_and_cannot_fake_frozen_low():
    graph = builder.first_segment_graph(FIRST_IMAGE, with_current_recipe=True)
    assert len(graph) == 42
    recipe = graph["80"]
    assert recipe["class_type"] == "MiniMaxH3FastH3V2CurrentRecipeEXPT8"
    assert recipe["inputs"]["model_pass1"] == ["1", 0]
    assert recipe["inputs"]["model_pass2"] == ["22", 0]
    assert recipe["inputs"]["first_frame"] == ["72", 0]
    assert recipe["inputs"]["upscale_report_json"] == ["23", 3]
    assert recipe["inputs"]["chain_id"] == ["70", 0]
    assert all(node["class_type"] not in {
        "MiniMaxH3LongVideoCandidateSaveT8", "MiniMaxH3LongVideoAcceptCandidateT8"}
        for node in graph.values())
    with pytest.raises(ValueError, match="Frozen LOW"):
        builder.first_segment_graph(FIRST_IMAGE, resume_high=True, with_current_recipe=True)
    audit = json.loads((RECIPE_CANDIDATE / "audit.json").read_text(encoding="utf-8"))
    assert len(audit["candidates"]) == 1 and audit["candidates"][0]["all_outputs_valid"]
    info = builder.base.load_live_info()
    import nodes as core_nodes
    info["LoadImage"] = builder.base.native_info("LoadImage", core_nodes.NODE_CLASS_MAPPINGS["LoadImage"])
    expected, _ = builder.base.selected_frontend_schema(graph, info)
    saved = json.loads((RECIPE_CANDIDATE / "FastH3V2_First_Full_Unaccepted_CurrentRecipe.api.json")
                       .read_text(encoding="utf-8"))
    assert saved == expected


def test_opt_in_first_stage_attest_audits_both_real_sampler_edges():
    graph = builder.first_segment_graph(
        FIRST_IMAGE, with_current_recipe=True, with_stage_attestation=True)
    assert len(graph) == 46
    for node, sampler, raw, stage_model, phase in (
            ("85", "13", "1", "43", "low_0_4"),
            ("87", "29", "22", "44", "high_4_8")):
        inputs = graph[node]["inputs"]
        sample = graph[sampler]["inputs"]
        assert graph[node]["class_type"] == "MiniMaxH3FastH3V2CurrentStageAttestEXPT8"
        assert inputs["current_recipe"] == ["80", 0]
        assert inputs["stage_result"] == [sampler, 2]
        assert inputs["raw_model"] == [raw, 0]
        assert inputs["stage_model"] == [stage_model, 0]
        assert inputs["guider"] == sample["guider"]
        assert inputs["source_latent"] == sample["latent_image"]
        assert inputs["phase"] == phase
    with pytest.raises(ValueError, match="requires the editable recipe"):
        builder.first_segment_graph(FIRST_IMAGE, with_stage_attestation=True)
    audit = json.loads((ATTEST_CANDIDATE / "audit.json").read_text(encoding="utf-8"))
    assert len(audit["candidates"]) == 1 and audit["candidates"][0]["all_outputs_valid"]
    info = builder.base.load_live_info()
    import nodes as core_nodes
    info["LoadImage"] = builder.base.native_info("LoadImage", core_nodes.NODE_CLASS_MAPPINGS["LoadImage"])
    expected, _ = builder.base.selected_frontend_schema(graph, info)
    saved = json.loads((ATTEST_CANDIDATE / "FastH3V2_First_Full_Unaccepted_CurrentRecipe_StageAttest.api.json")
                       .read_text(encoding="utf-8"))
    assert saved == expected
    _, frontend, _ = builder.build_candidate(
        graph, info, with_current_recipe=True, with_stage_attestation=True)
    saved_frontend = json.loads((ATTEST_CANDIDATE / "FastH3V2_First_Full_Unaccepted_CurrentRecipe_StageAttest.json")
                                .read_text(encoding="utf-8"))
    assert_historical_frontend_matches_current(saved_frontend, frontend)


def test_first_condition_origin_graph_is_append_only_and_saved_exactly():
    graph = builder.first_segment_graph(FIRST_IMAGE, with_current_recipe=True,
        with_stage_attestation=True, with_condition_provenance=True)
    assert len(graph) == 46 and "9" not in graph and "24" not in graph
    for key in ("89", "90"):
        assert graph[key]["class_type"] == "MiniMaxH3FastH3V2ConditionProvenanceEXPT8"
    for audit, conditioner in (("85", "89"), ("87", "90")):
        assert graph[audit]["class_type"] == "MiniMaxH3FastH3V2OriginStageAttestEXPT8"
        assert graph[audit]["inputs"]["condition_receipt"] == [conditioner, 7]
        assert graph[audit]["inputs"]["conditioned_latent"] == [conditioner, 2]
    with pytest.raises(ValueError, match="requires current stage attestation"):
        builder.first_segment_graph(FIRST_IMAGE, with_condition_provenance=True)
    saved_audit = json.loads((ORIGIN_CANDIDATE / "audit.json").read_text(encoding="utf-8"))
    assert saved_audit["candidates"][0]["all_outputs_valid"] is True
    assert saved_audit["candidates"][0]["serialization"]["edges"] == 135
    info = builder.base.load_live_info()
    import nodes as core_nodes
    info["LoadImage"] = builder.base.native_info("LoadImage", core_nodes.NODE_CLASS_MAPPINGS["LoadImage"])
    expected, _ = builder.base.selected_frontend_schema(graph, info)
    name = "FastH3V2_First_Full_Unaccepted_CurrentRecipe_StageAttest_ConditionProvenance"
    assert json.loads((ORIGIN_CANDIDATE / (name + ".api.json")).read_text(encoding="utf-8")) == expected
    _, frontend, _ = builder.build_candidate(graph, info, with_current_recipe=True,
        with_stage_attestation=True, with_condition_provenance=True)
    saved_frontend = json.loads((ORIGIN_CANDIDATE / (name + ".json")).read_text(encoding="utf-8"))
    assert_historical_frontend_matches_current(saved_frontend, frontend)


def test_first_handoff_origin_candidate_is_opt_in_and_saved_exactly():
    graph = builder.first_segment_graph(FIRST_IMAGE, with_current_recipe=True,
        with_stage_attestation=True, with_condition_provenance=True,
        with_handoff_provenance=True)
    assert len(graph) == 48 and "23" not in graph
    assert graph["91"]["class_type"] == "MiniMaxH3FastH3V2UpscaleProvenanceEXPT8"
    assert graph["92"]["inputs"]["upscale_receipt"] == ["91", 4]
    assert graph["92"]["inputs"]["high_source_latent"] == graph["29"]["inputs"]["latent_image"]
    with pytest.raises(ValueError, match="requires condition provenance"):
        builder.first_segment_graph(FIRST_IMAGE, with_handoff_provenance=True)
    saved_audit = json.loads((HANDOFF_CANDIDATE / "audit.json").read_text(encoding="utf-8"))
    assert saved_audit["candidates"][0]["all_outputs_valid"] is True
    assert saved_audit["candidates"][0]["serialization"]["edges"] == 150
    info = builder.base.load_live_info()
    import nodes as core_nodes
    info["LoadImage"] = builder.base.native_info("LoadImage", core_nodes.NODE_CLASS_MAPPINGS["LoadImage"])
    expected, _ = builder.base.selected_frontend_schema(graph, info)
    name = "FastH3V2_First_Full_Unaccepted_CurrentRecipe_StageAttest_ConditionProvenance_HandoffProvenance"
    assert json.loads((HANDOFF_CANDIDATE / (name + ".api.json")).read_text(encoding="utf-8")) == expected
    _, frontend, _ = builder.build_candidate(graph, info, with_current_recipe=True,
        with_stage_attestation=True, with_condition_provenance=True, with_handoff_provenance=True)
    saved_frontend = json.loads((HANDOFF_CANDIDATE / (name + ".json")).read_text(encoding="utf-8"))
    assert_historical_frontend_matches_current(saved_frontend, frontend)


def test_first_current_job_binding_candidate_preserves_old_graphs():
    graph = builder.first_segment_graph(FIRST_IMAGE, with_current_recipe=True,
        with_stage_attestation=True, with_condition_provenance=True,
        with_handoff_provenance=True, with_job_binding=True)
    assert len(graph) == 50
    assert graph["94"]["class_type"] == "MiniMaxH3FastH3V2CurrentJobBindEXPT8"
    assert graph["94"]["inputs"]["handoff_attestation"] == ["92", 0]
    assert "contexts" not in graph["94"]["inputs"]
    with pytest.raises(ValueError, match="requires handoff provenance"):
        builder.first_segment_graph(FIRST_IMAGE, with_job_binding=True)
    audit = json.loads((JOB_CANDIDATE / "audit.json").read_text(encoding="utf-8"))
    assert audit["candidates"][0]["all_outputs_valid"]
    assert audit["candidates"][0]["serialization"]["edges"] == 154
    info = builder.base.load_live_info()
    import nodes as core_nodes
    info["LoadImage"] = builder.base.native_info("LoadImage", core_nodes.NODE_CLASS_MAPPINGS["LoadImage"])
    expected, _ = builder.base.selected_frontend_schema(graph, info)
    name = "FastH3V2_First_Full_Unaccepted_CurrentRecipe_StageAttest_ConditionProvenance_HandoffProvenance_CurrentJobBound"
    assert json.loads((JOB_CANDIDATE / (name + ".api.json")).read_text(encoding="utf-8")) == expected
    _, frontend, _ = builder.build_candidate(graph, info, with_current_recipe=True,
        with_stage_attestation=True, with_condition_provenance=True,
        with_handoff_provenance=True, with_job_binding=True)
    saved_frontend = json.loads((JOB_CANDIDATE / (name + ".json")).read_text(encoding="utf-8"))
    assert_historical_frontend_matches_current(saved_frontend, frontend)


def test_first_source_bound_candidate_graph_is_opt_in_and_saved_exactly():
    flags = dict(with_current_recipe=True, with_stage_attestation=True,
        with_condition_provenance=True, with_handoff_provenance=True,
        with_job_binding=True, with_media_provenance=True, with_candidate_save=True)
    graph = builder.first_segment_graph(FIRST_IMAGE, **flags)
    assert len(graph) == 52 and "14" not in graph and "79" not in graph
    assert graph["15"]["inputs"]["images"] == ["96", 0]
    assert graph["98"]["inputs"]["media_receipt"] == ["96", 2]
    assert graph["98"]["class_type"] == "MiniMaxH3FastH3V2CurrentCandidateSaveEXPT8"
    with pytest.raises(ValueError, match="requires authenticated current media"):
        builder.first_segment_graph(FIRST_IMAGE, with_candidate_save=True)
    audit = json.loads((SAVE_CANDIDATE / "audit.json").read_text(encoding="utf-8"))
    assert audit["candidates"][0]["all_outputs_valid"]
    assert audit["candidates"][0]["serialization"]["edges"] == 156
    info = builder.base.load_live_info()
    import nodes as core_nodes
    info["LoadImage"] = builder.base.native_info("LoadImage", core_nodes.NODE_CLASS_MAPPINGS["LoadImage"])
    expected, _ = builder.base.selected_frontend_schema(graph, info)
    name = audit["candidates"][0]["name"]
    assert json.loads((SAVE_CANDIDATE / (name + ".api.json")).read_text(encoding="utf-8")) == expected
    _, frontend, _ = builder.build_candidate(graph, info, **flags)
    saved_frontend = json.loads((SAVE_CANDIDATE / (name + ".json")).read_text(encoding="utf-8"))
    assert_historical_frontend_matches_current(saved_frontend, frontend)


def test_first_frozen_low_bundle_saved_graphs_match_live_builder():
    flags = dict(with_current_recipe=True, with_stage_attestation=True,
        with_condition_provenance=True, with_handoff_provenance=True,
        with_job_binding=True, with_media_provenance=True, with_candidate_save=True,
        with_frozen_low_bundle=True)
    audit = json.loads((FROZEN_BUNDLE_CANDIDATE / "audit.json").read_text(encoding="utf-8"))
    assert [(row["serialization"]["nodes"], row["serialization"]["edges"])
            for row in audit["candidates"]] == [(54, 163), (40, 96)]
    assert all(row["all_outputs_valid"] and row["core_validation"][3] == {}
               for row in audit["candidates"])
    info = builder.base.load_live_info()
    import nodes as core_nodes
    info["LoadImage"] = builder.base.native_info("LoadImage", core_nodes.NODE_CLASS_MAPPINGS["LoadImage"])
    for resume, row in zip((False, True), audit["candidates"], strict=True):
        graph = builder.first_segment_graph(FIRST_IMAGE, resume_high=resume, **flags)
        assert graph["101" if not resume else "102"]["class_type"] == (
            "MiniMaxH3FastH3V2FrozenLOWBundleLoadEXPT8" if resume else
            "MiniMaxH3FastH3V2FrozenLOWBundleSaveEXPT8")
        if resume:
            assert all(key not in graph for key in ("1", "13", "80", "85", "89"))
        expected, _ = builder.base.selected_frontend_schema(graph, info)
        assert json.loads((FROZEN_BUNDLE_CANDIDATE / (row["name"] + ".api.json"))
                          .read_text(encoding="utf-8")) == expected
        _, frontend, _ = builder.build_candidate(graph, info, resume_high=resume, **flags)
        saved_frontend = json.loads((FROZEN_BUNDLE_CANDIDATE / (row["name"] + ".json"))
                                    .read_text(encoding="utf-8"))
        assert_historical_frontend_matches_current(saved_frontend, frontend)


def test_first_window_core_full_and_fresh_high_only_export_equal(monkeypatch, tmp_path, request):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    import execution
    import nodes
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
    from comfy_extras.nodes_preview_any import PreviewAny
    from comfy_extras.nodes_video import CreateVideo, SaveVideo

    registered = [TinyCandidateUNET, TinyCandidateCLIP, TinyCandidateVAE,
                  TinyAcceptedUpscale, TinyFirstImage, BasicGuider, RandomNoise,
                  PreviewAny, CreateVideo, SaveVideo, MiniMaxH3AVDecodeT8,
                  MiniMaxH3OutputTrimT8, MiniMaxH3FastH3V2RuntimeAuditEXPT8,
                  MiniMaxH3TwoPassLatentReconcileT8Advanced,
                  MiniMaxH3LongVideoPlannerT8, MiniMaxH3LongVideoContextLoadT8,
                  MiniMaxH3PromptRelayPlanT8Advanced,
                  MiniMaxH3PromptRelayLongVideoPlanT8Advanced,
                  MiniMaxH3PromptRelayLongVideoConditioningT8Advanced,
                  *HANDOFF_NODES, *LOADER_NODES, *current_job_nodes.NODES,
                  *continuation_nodes.NODES, *fast_h3_v2_continuation_nodes.NODES]
    registered += [getattr(stage_nodes, name) for name in (
        "MiniMaxH3FastH3V2StageSetupEXPT8", "MiniMaxH3StageSamplerEXPT8",
        "MiniMaxH3StageEAVConfigEXPT8", "MiniMaxH3StageEAVApplyEXPT8",
        "MiniMaxH3StageEAVAuditEXPT8", "MiniMaxH3StageSaveEXPT8", "MiniMaxH3StageLoadEXPT8")]
    for cls in registered:
        name = cls.define_schema().node_id if hasattr(cls, "define_schema") else cls.__name__
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, cls)
    monkeypatch.setattr(stage_nodes, "_stage_store_root", lambda: tmp_path)
    monkeypatch.setattr(stage_unet_loader, "_load_unet", lambda _name, _dtype: TinyCandidateUNET.execute(
        "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors", "default").result[0])
    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output_root))
    original_threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(original_threads))
    torch.set_num_threads(1)
    stages, receipts = [], []
    original_sample = stage_nodes.sample_stage

    def observed(*args):
        stages.append(args[-1].stage)
        actual_model = args[1].model_patcher
        result = original_sample(*args)
        receipts.append(result[2].verify())
        if receipts[-1]["portable_identity"]:
            assert selected_model_identity(actual_model) == receipts[-1]["request"]["model"]
        return result

    monkeypatch.setattr(stage_nodes, "sample_stage", observed)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *args, **kwargs: None)

    def executor():
        return execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))

    full = _tiny(builder.first_segment_graph(FIRST_IMAGE))
    run = executor()
    run.execute(deepcopy(full), "v2-first-full-cpu", execute_outputs=["16", "50", "51"])
    assert run.success, run.status_messages
    assert stages == ["low_0_4", "high_4_8"]
    files = list(output_root.rglob("FastH3V2_First124_Unaccepted_EXP_*.mp4"))
    assert len(files) == 1
    expected = _media_hashes(files[0])
    full_high = receipts[-1]
    low_manifest = next(path for path in tmp_path.rglob("manifest.json")
                        if json.loads(path.read_text(encoding="utf-8")).get("stage_context", {}).get("stage") == "low_0_4")
    frozen = _tiny(builder.first_segment_graph(FIRST_IMAGE, resume_high=True))
    frozen["60"]["inputs"].update(artifact_path=low_manifest.relative_to(tmp_path).as_posix(),
                                  artifact_sha256=file_sha(low_manifest))
    stages.clear()
    fresh = executor()
    fresh.execute(deepcopy(frozen), "v2-first-frozen-cpu", execute_outputs=["16", "51"])
    assert fresh.success, fresh.status_messages
    assert stages == ["high_4_8"]
    files = sorted(output_root.rglob("FastH3V2_First124_Unaccepted_EXP_*.mp4"))
    assert len(files) == 2 and _media_hashes(files[-1]) == expected
    assert receipts[-1]["request_sha256"] == full_high["request_sha256"]
    assert receipts[-1]["outputs"] == full_high["outputs"]
    bad = deepcopy(frozen)
    bad["60"]["inputs"]["artifact_sha256"] = "0" * 64
    stages.clear()
    rejected = executor()
    rejected.execute(bad, "v2-first-bad-sha-cpu", execute_outputs=["51"])
    assert not rejected.success and stages == []

    def fake_identity(_, **_kwargs):
        return {"sha256": "1" * 64}

    monkeypatch.setattr(current_job, "stage_model_identity", fake_identity)
    monkeypatch.setattr(current_job, "_component_identity", fake_identity)
    monkeypatch.setattr(current_job, "_upscale_contract", lambda _report, _geometry: {"test_upscale": "cpu_fake"})
    monkeypatch.setattr(current_attest, "stage_model_identity", fake_identity)
    monkeypatch.setattr(current_attest, "_raw_model_from_stage", fake_identity)
    actual_attest = current_attest.attest_current_stage
    captured = []

    def observed_attest(**kwargs):
        captured.append(kwargs)
        return actual_attest(**kwargs)

    monkeypatch.setattr(current_job_nodes, "attest_current_stage", observed_attest)
    attested = _tiny(builder.first_segment_graph(
        FIRST_IMAGE, with_current_recipe=True, with_stage_attestation=True))
    attested["80"]["inputs"].update(low_width=64, low_height=32, width=128, height=64)
    certified = executor()
    certified.execute(deepcopy(attested), "v2-first-stage-audit-cpu", execute_outputs=["86", "88"])
    assert certified.success, certified.status_messages
    assert [item["phase"] for item in captured] == ["low_0_4", "high_4_8"]
    monkeypatch.setattr(current_origin, "_component_identity", fake_identity)
    origin_calls = []
    original_origin = current_job_nodes.condition_with_provenance

    def observed_origin(**kwargs):
        result = original_origin(**kwargs)
        origin_calls.append(len(result))
        return result

    monkeypatch.setattr(current_job_nodes, "condition_with_provenance", observed_origin)
    proven = _tiny(builder.first_segment_graph(
        FIRST_IMAGE, with_current_recipe=True, with_stage_attestation=True,
        with_condition_provenance=True))
    proven["80"]["inputs"].update(low_width=64, low_height=32, width=128, height=64)
    for key in ("89", "90"):
        proven[key]["inputs"]["query_chunk_rows"] = 256
    proof_run = executor()
    proof_run.execute(deepcopy(proven), "v2-first-condition-origin-cpu", execute_outputs=["86", "88"])
    assert proof_run.success, (origin_calls, [(name, details.get("node_id"), details.get("exception_message"))
                                              for name, details in proof_run.status_messages])
    assert origin_calls == [8, 8]
    assert [item["phase"] for item in captured[-2:]] == ["low_0_4", "high_4_8"]
    assert all(item["condition_receipt"].verify()["portable_identity"] for item in captured[-2:])
    def fake_contract(_report, geometry):
        return {"model_name": "minimax_h3_latent_upscaler_3d_fp16.safetensors",
                "geometry": geometry, "test_upscale": "cpu_fake"}
    monkeypatch.setattr(current_job, "_upscale_contract", fake_contract)
    monkeypatch.setattr(current_handoff, "_upscale_contract", fake_contract)
    monkeypatch.setattr(current_upscale, "learned_upscale_h3_av_latent",
                        lambda latent, **kwargs: TinyAcceptedUpscale.execute(latent, **kwargs).result)
    handoff_calls = []
    actual_handoff = current_handoff.attest_current_handoff

    def observed_handoff(**kwargs):
        handoff_calls.append(kwargs)
        return actual_handoff(**kwargs)

    monkeypatch.setattr(current_job_nodes, "attest_current_handoff", observed_handoff)
    handoff_graph = _tiny(builder.first_segment_graph(
        FIRST_IMAGE, with_current_recipe=True, with_stage_attestation=True,
        with_condition_provenance=True, with_handoff_provenance=True))
    handoff_graph["80"]["inputs"].update(low_width=64, low_height=32, width=128, height=64)
    for key in ("89", "90"):
        handoff_graph[key]["inputs"]["query_chunk_rows"] = 256
    handoff_run = executor()
    handoff_run.execute(deepcopy(handoff_graph), "v2-first-handoff-origin-cpu", execute_outputs=["93"])
    assert handoff_run.success, handoff_run.status_messages
    assert len(handoff_calls) == 1
    proof = actual_handoff(**handoff_calls[0])[0].verify()
    assert proof["segment_index"] == 0 and proof["accepted_source_sha256"] is None
    bound = current_binding.bind_current_job(
        handoff_calls[0]["current_recipe"], actual_handoff(**handoff_calls[0])[0], 0)[0]
    assert bound.verify()["sampling_summary"] == handoff_calls[0]["current_recipe"].sha256
    frozen_path, frozen_sha, _ = stage_storage.save_stage(
        handoff_calls[0]["low_result"], tmp_path, prefix="v2-current-low-bundle")
    bundle_path, bundle_sha, _ = current_frozen_low.save_current_low_bundle(
        handoff_calls[0]["current_recipe"], handoff_calls[0]["low_result"],
        handoff_calls[0]["low_attestation"], handoff_calls[0]["low_condition_receipt"],
        frozen_path, frozen_sha, tmp_path)
    loaded_bundle = current_frozen_low.load_current_low_bundle(
        frozen_path, frozen_sha, handoff_calls[0]["low_result"], tmp_path)
    assert loaded_bundle[0].sha256 == bound.recipe.sha256
    assert loaded_bundle[4].sha256 == handoff_calls[0]["low_attestation"].sha256
    assert loaded_bundle[5].sha256 == handoff_calls[0]["low_condition_receipt"].sha256
    assert loaded_bundle[6] == bundle_sha and (tmp_path / bundle_path).is_file()
    selected_bundle = tmp_path / bundle_path
    original_bundle = selected_bundle.read_bytes()
    damaged_bundle = json.loads(original_bundle)
    damaged_bundle["low_condition_receipt_sha256"] = "0" * 64
    selected_bundle.write_text(json.dumps(damaged_bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint changed"):
        current_frozen_low.load_current_low_bundle(
            frozen_path, frozen_sha, handoff_calls[0]["low_result"], tmp_path)
    selected_bundle.write_bytes(original_bundle)
    with pytest.raises(ValueError):
        current_frozen_low.load_current_low_bundle(
            frozen_path, "0" * 64, handoff_calls[0]["low_result"], tmp_path)
    before_bundles = set(tmp_path.rglob(current_frozen_low.SIDECAR))
    full_bundle = _tiny(builder.first_segment_graph(
        FIRST_IMAGE, with_current_recipe=True, with_stage_attestation=True,
        with_condition_provenance=True, with_handoff_provenance=True,
        with_job_binding=True, with_media_provenance=True, with_candidate_save=True,
        with_frozen_low_bundle=True))
    full_bundle["80"]["inputs"].update(low_width=64, low_height=32, width=128, height=64)
    for key in ("89", "90"):
        full_bundle[key]["inputs"]["query_chunk_rows"] = 256
    bundle_run = executor()
    bundle_run.execute(deepcopy(full_bundle), "v2-first-save-current-low-bundle-cpu",
                       execute_outputs=["103"])
    assert bundle_run.success, bundle_run.status_messages
    new_bundles = set(tmp_path.rglob(current_frozen_low.SIDECAR)) - before_bundles
    assert len(new_bundles) == 1
    selected_manifest = next(iter(new_bundles)).parent / "manifest.json"
    cold_bundle = _tiny(builder.first_segment_graph(
        FIRST_IMAGE, resume_high=True, with_current_recipe=True,
        with_stage_attestation=True, with_condition_provenance=True,
        with_handoff_provenance=True, with_job_binding=True,
        with_media_provenance=True, with_candidate_save=True,
        with_frozen_low_bundle=True))
    assert all(key not in cold_bundle for key in ("1", "13", "80", "85", "89"))
    assert cold_bundle["92"]["inputs"]["low_result"] == ["60", 3]
    assert cold_bundle["92"]["inputs"]["low_attestation"] == ["102", 4]
    for key in ("60", "102"):
        cold_bundle[key]["inputs"].update(
            artifact_path=selected_manifest.relative_to(tmp_path).as_posix(),
            artifact_sha256=file_sha(selected_manifest))
    cold_bundle["90"]["inputs"]["query_chunk_rows"] = 256
    stages.clear()
    cold_run = executor()
    cold_run.execute(deepcopy(cold_bundle), "v2-first-current-frozen-high-cpu",
                     execute_outputs=["95"])
    assert cold_run.success, cold_run.status_messages
    assert stages == ["high_4_8"]
    assert receipts[-1]["request_sha256"] == handoff_calls[0]["high_result"].verify()["request_sha256"]
    assert receipts[-1]["outputs"] == handoff_calls[0]["high_result"].verify()["outputs"]
    sidecar = next(iter(new_bundles))
    original_sidecar = sidecar.read_bytes()
    modified_sidecar = json.loads(original_sidecar)
    modified_sidecar["low_attestation_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(modified_sidecar), encoding="utf-8")
    stages.clear()
    rejected_bundle = executor()
    rejected_bundle.execute(deepcopy(cold_bundle), "v2-first-bad-current-low-bundle-cpu",
                            execute_outputs=["95"])
    assert not rejected_bundle.success and stages == []
    sidecar.write_bytes(original_sidecar)
    bound_graph = _tiny(builder.first_segment_graph(
        FIRST_IMAGE, with_current_recipe=True, with_stage_attestation=True,
        with_condition_provenance=True, with_handoff_provenance=True,
        with_job_binding=True))
    bound_graph["80"]["inputs"].update(low_width=64, low_height=32, width=128, height=64)
    for key in ("89", "90"):
        bound_graph[key]["inputs"]["query_chunk_rows"] = 256
    bound_run = executor()
    bound_run.execute(deepcopy(bound_graph), "v2-first-current-job-bound-cpu", execute_outputs=["95"])
    assert bound_run.success, bound_run.status_messages
    monkeypatch.setattr(current_media, "_component_identity", fake_identity)
    media_receipts = []
    original_media = current_job_nodes.prepare_current_media

    def observed_media(**kwargs):
        result = original_media(**kwargs)
        media_receipts.append(result[2])
        return result

    monkeypatch.setattr(current_job_nodes, "prepare_current_media", observed_media)
    media_graph = _tiny(builder.first_segment_graph(
        FIRST_IMAGE, with_current_recipe=True, with_stage_attestation=True,
        with_condition_provenance=True, with_handoff_provenance=True,
        with_job_binding=True, with_media_provenance=True))
    media_graph["80"]["inputs"].update(low_width=64, low_height=32, width=128, height=64)
    for key in ("89", "90"):
        media_graph[key]["inputs"]["query_chunk_rows"] = 256
    media_run = executor()
    media_run.execute(deepcopy(media_graph), "v2-first-current-media-cpu",
                      execute_outputs=["16", "97"])
    assert media_run.success, media_run.status_messages
    assert len(media_receipts) == 1
    assert media_receipts[0].verify()["trim"]["frame_count"] == 124
    media_files = sorted(output_root.rglob("FastH3V2_First124_Unaccepted_EXP_*.mp4"))
    assert len(media_files) == 3 and _media_hashes(media_files[-1]) == expected
    with pytest.raises(ValueError):
        current_candidate.save_current_candidate(replace(
            media_receipts[0], frames=media_receipts[0].frames[:123]), "bad-source")
    path, video, provenance, save_report = current_candidate.save_current_candidate(
        media_receipts[0], "first-source-bound")
    candidate = json.loads(Path(path).read_text(encoding="utf-8"))
    source = json.loads(Path(provenance).read_text(encoding="utf-8"))
    assert candidate["status"] == "candidate" and candidate["frame_count"] == 124
    assert candidate["context_sha256"] and candidate["timeline_start_frame"] == 0
    assert candidate["sampling_summary"] == media_receipts[0].binding.verify()["sampling_summary"]
    assert source["candidate_json_sha256"] == file_sha(Path(path))
    assert source["media_receipt_sha256"] == media_receipts[0].sha256
    assert source["accepted"] is False and json.loads(save_report)["accepted"] is False
    candidate_rgb, candidate_pcm = _media_hashes(Path(video))
    assert candidate_rgb == expected[0] and candidate_pcm
    preview = current_review.review_accept_current_candidate(path, False)
    assert preview[0] == video and preview[1] is False and preview[2] == ""
    assert preview[4:7] == ("", 0, "")
    manifest_path = Path(path).parents[3] / "manifest.json"
    assert not manifest_path.exists()
    original_sidecar = Path(provenance).read_text(encoding="utf-8")
    altered_sidecar = json.loads(original_sidecar)
    altered_sidecar["media_receipt_sha256"] = "0" * 64
    Path(provenance).write_text(json.dumps(altered_sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="media fingerprint"):
        current_review.review_accept_current_candidate(path, True)
    assert not manifest_path.exists()
    Path(provenance).write_text(original_sidecar, encoding="utf-8")
    accepted_candidate = current_job_nodes.MiniMaxH3FastH3V2CurrentCandidateReviewAcceptEXPT8.execute(
        path, True).result
    assert accepted_candidate[1] is True and Path(accepted_candidate[2]).is_file()
    assert accepted_candidate[4] == "first-source-bound" and accepted_candidate[5] == 1
    assert accepted_candidate[6] == candidate["sampling_summary"]
    accepted_manifest = json.loads(Path(accepted_candidate[2]).read_text(encoding="utf-8"))
    assert len(accepted_manifest["segments"]) == 1
    assert accepted_manifest["segments"][0]["candidate_id"] == "first-source-bound"
    second = continuation_builder.continuation_graph(
        with_delivery=True, with_current_recipe=True, with_stage_attestation=True,
        with_condition_provenance=True, with_handoff_provenance=True,
        with_job_binding=True, with_media_provenance=True, with_candidate_save=True,
        first_frame=FIRST_IMAGE)
    second["70"]["inputs"].update(
        chain_id="v2_first_segment_cpu", parent_candidate_id="first-source-bound",
        parent_revision=1, previous_job_sha256=candidate["sampling_summary"],
        low_width=64, low_height=32, width=128, height=64)
    second["91"]["inputs"].update(target_width=128, target_height=64)
    second["83"]["inputs"].update(low_width=64, low_height=32, width=128, height=64)
    second["98"]["inputs"]["candidate_id"] = "second-real-parent"
    for key in ("89", "90"):
        second[key]["inputs"]["query_chunk_rows"] = 256
    for key in ("10", "26"):
        second[key]["inputs"]["min_tokens"] = 0
    for key in ("40", "47"):
        second[key]["inputs"].update(global_prompt="One stable scene.",
            local_prompts="She walks.\nShe stops.\nShe turns.")
    colored_second = continuation_builder.continuation_graph(
        with_delivery=True, with_current_recipe=True, with_stage_attestation=True,
        with_condition_provenance=True, with_handoff_provenance=True,
        with_job_binding=True, with_media_provenance=True, with_candidate_save=True,
        with_color_match=True, first_frame=FIRST_IMAGE)
    for key in ("70", "91", "83", "89", "90", "10", "26", "40", "47"):
        colored_second[key]["inputs"].update(second[key]["inputs"])
    colored_second["98"]["inputs"]["candidate_id"] = "second-colored-source-bound"
    colored_run = executor()
    colored_run.execute(deepcopy(colored_second), "v2-accepted-real-first-colored-second-cpu",
                        execute_outputs=["99"])
    assert colored_run.success, colored_run.status_messages
    colored_path = Path(path).parents[3] / "candidates/segment_00001/second-colored-source-bound/candidate.json"
    colored_candidate, colored_movie, colored_sidecar = current_review.verify_saved_current_candidate(colored_path)
    colored_source = json.loads(Path(colored_sidecar).read_text(encoding="utf-8"))
    assert colored_source["schema"] == current_color_candidate.SCHEMA
    assert colored_source["color_receipt"]["mode"] == "bounded_motion_color_exp"
    assert colored_source["color_receipt"]["color_report"]["predecessor_candidate_id"] == "first-source-bound"
    assert colored_candidate["frame_count"] == 68 and colored_candidate["parent_candidate_id"] == "first-source-bound"
    with av.open(colored_movie) as colored_container:
        assert len(list(colored_container.decode(video=0))) == 68
    assert current_review.review_accept_current_candidate(colored_path, False)[1] is False
    second_run = executor()
    second_run.execute(deepcopy(second), "v2-accepted-real-first-then-second-cpu",
                       execute_outputs=["99"])
    assert second_run.success, second_run.status_messages
    second_path = Path(path).parents[3] / "candidates/segment_00001/second-real-parent/candidate.json"
    second_candidate, _movie, second_sidecar = current_review.verify_saved_current_candidate(second_path)
    assert second_candidate["frame_count"] == 68
    assert second_candidate["parent_candidate_id"] == "first-source-bound"
    assert second_candidate["sampling_summary"] == candidate["sampling_summary"]
    accepted_second = current_review.review_accept_current_candidate(second_path, True)
    assert accepted_second[1] is True and accepted_second[4] == "second-real-parent"
    chain_proof = current_job_nodes.MiniMaxH3FastH3V2CurrentAcceptedChainVerifyEXPT8.execute(
        "v2_first_segment_cpu").result
    assert json.loads(chain_proof[1])["frame_count"] == 192
    original_second_sidecar = Path(second_sidecar).read_text(encoding="utf-8")
    altered_second_sidecar = json.loads(original_second_sidecar)
    altered_second_sidecar["candidate_json_sha256"] = "0" * 64
    Path(second_sidecar).write_text(json.dumps(altered_second_sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar identity"):
        current_review.verify_current_accepted_chain("v2_first_segment_cpu")
    Path(second_sidecar).write_text(original_second_sidecar, encoding="utf-8")
    composed_path, composed_report = delivery.compose_accepted_long_video(
        current_review.verify_current_accepted_chain("v2_first_segment_cpu")[0],
        "v2_source_bound_cpu", True, "cosine_bridge", 5.0, 18)
    composed_info = json.loads(composed_report)
    assert composed_info["frame_count"] == 192 and composed_info["audio_samples"] == 256000
    assert composed_info["segment_count"] == 2 and composed_info["strict_decode_validated"] is True
    with av.open(composed_path) as composed:
        assert len(list(composed.decode(video=0))) == 192
        composed.seek(0)
        assert sum(frame.samples for frame in composed.decode(audio=0)) >= 256000
    with pytest.raises(ValueError, match="completed LOW x0"):
        actual_handoff(**{**handoff_calls[0], "upscaled_latent": handoff_calls[0]["low_result"].denoised_output})
    report_only = deepcopy(attested)
    report_only["41"]["inputs"]["mode"] = "report_only"
    report_only["42"]["inputs"]["mode"] = "report_only"
    effect_run = executor()
    effect_run.execute(report_only, "v2-first-stage-effect-audit-cpu", execute_outputs=["86", "88"])
    assert effect_run.success, effect_run.status_messages
    assert len(receipts) >= 2 and all(item["portable_identity"] for item in receipts[-2:])
    assert all(item["request"]["model"]["stage_effects"]["effects"]["relay"]["long_video_projection"]
               for item in receipts[-2:])
    with pytest.raises(ValueError, match="exact dense FastH3 V2"):
        actual_attest(**{**captured[0], "phase": "high_4_8"})
    monkeypatch.setattr(current_attest, "_raw_model_from_stage", lambda _, **_kwargs: {"sha256": "2" * 64})
    with pytest.raises(ValueError, match="not derived"):
        actual_attest(**captured[0])
