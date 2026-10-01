"""Execute the saved V2 candidate graphs with explicit CPU asset doubles.

The private candidate JSON remains unchanged. This tests its actual wiring and
fresh-Core resume and media export, not pretrained weights or learned-upscaler math.
"""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import av
import folder_paths
import pytest
import torch
from comfy_api.latest import io

from h3_audio_t8_pkg.modular_sampling import nodes as stage_nodes, stage_unet_loader
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2_handoff_nodes import NODES as HANDOFF_NODES
from h3_audio_t8_pkg.modular_sampling.stage_unet_loader_nodes import NODES as STAGE_LOADER_NODES
from h3_audio_t8_pkg.modular_sampling.storage import file_sha
from h3_audio_t8_pkg.nodes import MiniMaxH3AVDecodeT8
from h3_audio_t8_pkg.nodes_fast_h3_v2_advanced import MiniMaxH3FastH3V2RuntimeAuditEXPT8
from h3_audio_t8_pkg.nodes_learned_latent_upscale_advanced import MiniMaxH3TwoPassLatentReconcileT8Advanced
from h3_audio_t8_pkg.nodes_prompt_relay_advanced import (
    MiniMaxH3PromptRelayConditioningT8Advanced,
    MiniMaxH3PromptRelayPlanT8Advanced,
)
from helpers import FakeAudioVAE, FakeVideoVAE
from test_fast_h3_v2_core_sampler import model
from test_modular_progressive_core_cache import InterpolationDouble
from test_prompt_relay_advanced import NativeLikeFakeClip
from tools.build_modular_fast_h3_v2_workflow import (
    resume_high_graph,
    split_graph_with_effects,
    split_graph_with_results,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "artifacts/development/modular-sampling-m1-v2-independent-loader-20260924"
FULL = CANDIDATES / "candidate-full-v2/FastH3_V2_Separate_LOW_HIGH_Save_Stages_EAV_Relay_EXP.api.json"
RESUME = CANDIDATES / "candidate-resume-v2/FastH3_V2_Frozen_LOW_Resume_HIGH_EAV_Relay_EXP.api.json"


class TinyCandidateUNET(io.ComfyNode):
    calls = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="UNETLoader", inputs=[
            io.String.Input("unet_name"), io.String.Input("weight_dtype")],
            outputs=[io.Model.Output("model")])

    @classmethod
    def execute(cls, unet_name, weight_dtype):
        assert "fasth3_8step_v2" in unet_name and weight_dtype == "default"
        cls.calls.append((unet_name, weight_dtype))
        return io.NodeOutput(model())


class TinyCandidateCLIP(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="CLIPLoader", inputs=[
            io.String.Input("clip_name"), io.String.Input("type"), io.String.Input("device")],
            outputs=[io.Clip.Output("clip")])

    @classmethod
    def execute(cls, clip_name, type, device):
        assert "qwen3vl" in clip_name and type == "minimax" and device == "default"
        return io.NodeOutput(NativeLikeFakeClip())


class TinyCandidateVAE(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="VAELoader", inputs=[io.String.Input("vae_name")],
                         outputs=[io.Vae.Output("vae")])

    @classmethod
    def execute(cls, vae_name):
        assert "minimax_h3" in vae_name
        return io.NodeOutput(FakeAudioVAE() if "audio" in vae_name else FakeVideoVAE())


class TinyCandidateUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3LearnedLatentUpscaleT8Advanced", inputs=[
            io.Latent.Input("av_latent"), io.String.Input("model_name"),
            io.String.Input("size_mode"), io.Float.Input("scale_by"),
            io.Float.Input("target_megapixels"), io.Int.Input("target_width"),
            io.Int.Input("target_height"), io.String.Input("aspect_policy"),
            io.Float.Input("max_anisotropy"), io.String.Input("precision"),
            io.String.Input("release_policy")], outputs=[
                io.Latent.Output("av_latent"), io.Int.Output("width"),
                io.Int.Output("height"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, av_latent, model_name, size_mode, scale_by, target_megapixels,
                target_width, target_height, aspect_policy, max_anisotropy, precision,
                release_policy):
        assert model_name.endswith("latent_upscaler_3d_fp16.safetensors")
        assert size_mode == "scale_by" and scale_by == 2.0
        assert aspect_policy == "preserve_source" and precision == "fp16"
        assert release_policy == "offload_after"
        assert target_megapixels > 0 and target_width > 0 and target_height > 0
        assert max_anisotropy >= 1
        video = av_latent["samples"].unbind()[0]
        width = int(video.shape[-1]) * 16 * 2
        height = int(video.shape[-2]) * 16 * 2
        lifted = InterpolationDouble.execute(av_latent, width, height).result
        return io.NodeOutput(*lifted, json.dumps({"network": "explicit_cpu_interpolation_double"}))


def _saved_graph(path, expected):
    if not path.is_file():
        pytest.skip("private saved candidate graph is not present in this workspace")
    graph = json.loads(path.read_text(encoding="utf-8"))
    assert set(graph) == set(expected)
    for key, node in graph.items():
        assert node["class_type"] == expected[key]["class_type"]
        actual_edges = {name: value for name, value in node["inputs"].items()
                        if isinstance(value, list) and len(value) == 2}
        planned_edges = {name: value for name, value in expected[key]["inputs"].items()
                         if isinstance(value, list) and len(value) == 2}
        assert actual_edges == planned_edges, key
    return graph


def _tiny_parameters(graph):
    graph = deepcopy(graph)
    if "9" in graph:
        graph["9"]["inputs"].update(width=128, height=64)
        graph["10"]["inputs"]["min_tokens"] = 0
        graph["43"]["inputs"]["eav_config"] = ["41", 0]
    graph["26"]["inputs"]["min_tokens"] = 0
    for key in ("40", "47"):
        if key in graph:
            graph[key]["inputs"].update(global_prompt="One stable scene",
                                         local_prompts="A woman waves.\nShe walks away.", length=22)
    for key in ("9", "24"):
        if key in graph:
            graph[key]["inputs"]["query_chunk_rows"] = 64
    for key in ("41", "42"):
        if key in graph:
            graph[key]["inputs"].update(tau=0.2, start_video_progress=0.,
                                         end_video_progress=1., g_hard_limit=3.)
    return graph


def _assert_exported_media(path):
    with av.open(str(path)) as container:
        assert len(container.streams.video) == len(container.streams.audio) == 1
        video_stream = container.streams.video[0]
        audio_stream = container.streams.audio[0]
        assert video_stream.codec_context.name == "h264"
        assert audio_stream.codec_context.name == "aac"
        video_frames = list(container.decode(video=0))
        assert len(video_frames) == 22
        assert all(frame.width == 256 and frame.height == 128 for frame in video_frames)
        assert [float(frame.pts * frame.time_base) for frame in video_frames] == pytest.approx(
            [index / 24 for index in range(22)])
        container.seek(0)
        audio_frames = list(container.decode(audio=0))
        assert audio_frames
        assert audio_stream.codec_context.sample_rate == 32000
        assert sum(frame.samples for frame in audio_frames) >= 22 / 24 * 32000
        assert any(frame.to_ndarray().any() for frame in audio_frames)


def test_saved_full_and_resume_candidate_execute_real_high_only_relay_eav(monkeypatch, tmp_path, request):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    import execution
    import nodes
    from comfy_extras.nodes_custom_sampler import RandomNoise, BasicGuider
    from comfy_extras.nodes_preview_any import PreviewAny
    from comfy_extras.nodes_video import CreateVideo, SaveVideo

    expected_full = split_graph_with_results(split_graph_with_effects())
    expected_resume = resume_high_graph(with_effects=True)
    full = _tiny_parameters(_saved_graph(FULL, expected_full))
    resume = _tiny_parameters(_saved_graph(RESUME, expected_resume))
    for cls in (TinyCandidateUNET, TinyCandidateCLIP, TinyCandidateVAE,
                TinyCandidateUpscale, RandomNoise, BasicGuider,
                PreviewAny, CreateVideo, SaveVideo,
                MiniMaxH3AVDecodeT8, MiniMaxH3FastH3V2RuntimeAuditEXPT8,
                MiniMaxH3PromptRelayPlanT8Advanced,
                MiniMaxH3PromptRelayConditioningT8Advanced,
                MiniMaxH3TwoPassLatentReconcileT8Advanced,
                stage_nodes.MiniMaxH3FastH3V2StageSetupEXPT8,
                stage_nodes.MiniMaxH3StageSamplerEXPT8,
                stage_nodes.MiniMaxH3StageSaveEXPT8,
                stage_nodes.MiniMaxH3StageLoadEXPT8,
                stage_nodes.MiniMaxH3StageEAVConfigEXPT8,
                stage_nodes.MiniMaxH3StageEAVApplyEXPT8,
                stage_nodes.MiniMaxH3StageEAVAuditEXPT8,
                *HANDOFF_NODES, *STAGE_LOADER_NODES):
        name = cls.define_schema().node_id if hasattr(cls, "define_schema") else cls.__name__
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, cls)
    monkeypatch.setattr(stage_nodes, "_stage_store_root", lambda: tmp_path)
    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output_root))
    TinyCandidateUNET.calls.clear()
    high_model_loads = []

    def load_high_model(name, dtype):
        high_model_loads.append((name, dtype))
        return model()

    monkeypatch.setattr(stage_unet_loader, "_load_unet", load_high_model)
    original_threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(original_threads))
    torch.set_num_threads(1)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *args, **kwargs: None)

    def fresh_executor():
        return execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))

    stages = []
    receipts = []
    original_sample = stage_nodes.sample_stage

    def observed_sample(*args):
        stages.append(args[-1].stage)
        result = original_sample(*args)
        receipts.append(result[2].verify())
        return result

    monkeypatch.setattr(stage_nodes, "sample_stage", observed_sample)
    audits = []
    original_audit = stage_nodes.audit_stage_eav

    def observed_audit(*args):
        value, report = original_audit(*args)
        audits.append(json.loads(report))
        return value, report

    monkeypatch.setattr(stage_nodes, "audit_stage_eav", observed_audit)

    executor = fresh_executor()
    full_executor = executor
    executor.execute(deepcopy(full), "saved-v2-full-cpu", execute_outputs=["16", "45", "46", "50", "51"])
    assert executor.success, executor.status_messages
    saved_media = list(output_root.rglob("Modular_FastH3_V2_EXP_*.mp4"))
    assert len(saved_media) == 1
    _assert_exported_media(saved_media[0])
    assert stages == ["low_0_4", "high_4_8"]
    assert len(TinyCandidateUNET.calls) == len(high_model_loads) == 1
    assert len(audits) == 2
    assert all(report["relay_attention_calls"] == 4 for report in audits)
    assert all(report["status"] == "observed_report_only" for report in audits)
    full_high = receipts[-1]
    assert full_high["portable_identity"] is True, {
        "low_portable": receipts[0]["portable_identity"],
        "low_model_schema": receipts[0]["request"]["model"]["schema"],
        "high_model_schema": full_high["request"]["model"]["schema"],
        "high_model_reason": full_high["request"]["model"].get("reason"),
    }

    high_prompt_edit = deepcopy(full)
    high_prompt_edit["47"]["inputs"]["local_prompts"] = "A woman looks up.\nShe turns away."
    stages.clear()
    audits.clear()
    executor.execute(high_prompt_edit, "saved-v2-high-relay-edit-cpu", execute_outputs=["46", "51"])
    assert executor.success, executor.status_messages
    assert stages == ["high_4_8"]
    assert len(TinyCandidateUNET.calls) == len(high_model_loads) == 1
    assert len(audits) == 1 and audits[0]["relay_attention_calls"] == 4
    assert receipts[-1]["request_sha256"] != full_high["request_sha256"]

    low_manifest = next(path for path in tmp_path.rglob("manifest.json")
                        if json.loads(path.read_text(encoding="utf-8"))["stage_context"]["stage"] == "low_0_4")
    resume["60"]["inputs"].update(artifact_path=low_manifest.relative_to(tmp_path).as_posix(),
                                  artifact_sha256=file_sha(low_manifest))
    assert not {"1", "9", "10", "11", "12", "13", "41", "43", "45", "50"} & set(resume)
    stages.clear()
    audits.clear()
    executor = fresh_executor()
    executor.execute(resume, "saved-v2-resume-cpu", execute_outputs=["16", "46", "51"])
    assert executor.success, executor.status_messages
    saved_media = list(output_root.rglob("Modular_FastH3_V2_EXP_*.mp4"))
    assert len(saved_media) == 2
    _assert_exported_media(sorted(saved_media)[-1])
    assert stages == ["high_4_8"]
    assert len(TinyCandidateUNET.calls) == 1 and len(high_model_loads) == 2
    assert len(audits) == 1 and audits[0]["relay_attention_calls"] == 4
    assert audits[0]["status"] == "observed_report_only"
    assert receipts[-1]["outputs"] == full_high["outputs"]
    assert receipts[-1]["request_sha256"] == full_high["request_sha256"]

    bad = deepcopy(resume)
    bad["60"]["inputs"]["artifact_sha256"] = "0" * 64
    stages.clear()
    executor = fresh_executor()
    executor.execute(bad, "saved-v2-bad-low-sha-cpu", execute_outputs=["46"])
    assert executor.success is False
    assert stages == []
    assert len(TinyCandidateUNET.calls) == 1 and len(high_model_loads) == 2

    low_prompt_edit = deepcopy(full)
    low_prompt_edit["40"]["inputs"]["local_prompts"] = "A woman starts moving.\nShe sits down."
    stages.clear()
    full_executor.execute(low_prompt_edit, "saved-v2-low-relay-edit-cpu",
                          execute_outputs=["45", "46", "50", "51"])
    assert full_executor.success, full_executor.status_messages
    assert stages == ["low_0_4", "high_4_8"]
    assert len(TinyCandidateUNET.calls) == 1 and len(high_model_loads) == 3
