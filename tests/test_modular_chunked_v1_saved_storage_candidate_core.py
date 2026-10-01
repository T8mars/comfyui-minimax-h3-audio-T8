"""Execute saved S18 candidate branches with explicit tiny CPU inputs.

The disk JSON bytes remain unchanged. An in-memory copy replaces large upstream
assets and trained VAEs; the installed VHS media terminal runs unchanged.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import av
import folder_paths
from comfy_api.latest import io
import pytest
import torch
import torch.nn.functional as F

from h3_audio_t8_pkg.modular_sampling import chunked_stage_nodes
from h3_audio_t8_pkg.modular_sampling.chunked_source_nodes import NODES as SOURCE_NODES
from h3_audio_t8_pkg.modular_sampling.chunked_stage_nodes import NODES as STAGE_NODES
from h3_audio_t8_pkg.modular_sampling.chunked_v1_relay_nodes import NODES as RELAY_NODES
from h3_audio_t8_pkg.modular_sampling.chunked_v1_storage import file_sha
from h3_audio_t8_pkg.modular_sampling.chunked_v1_storage_nodes import NODES as STORAGE_NODES
from h3_audio_t8_pkg.nodes_chunked_two_pass_upscale_advanced import (
    MiniMaxH3ChunkedTwoPassPlanT8Advanced,
)
from h3_audio_t8_pkg.nodes_native_latent_checkpoint_advanced import (
    MiniMaxH3NativeLatentCheckpointLoadT8Advanced,
    MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
)
from h3_audio_t8_pkg.prompt_relay_advanced import PROMPT_RELAY_PLAN_TYPE
from test_modular_chunked_v1_relay import _case
from tools.build_modular_chunked_v1_relay_workflow import build_pair
from tools.build_modular_chunked_v1_storage_workflow import build_storage_pair, _prune_api


ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT / "artifacts/development/modular-sampling-s18-v1-storage-20260924/candidate-v1"
RELAY = ROOT / "artifacts/development/modular-sampling-s18-v1-relay-20260924/candidate-v1"
FREEZE = STORAGE / "01_freeze_after_segment_0.api.json"
RESUME = STORAGE / "02_resume_segments_1_and_2_DRAFT.api.json"
FULL = RELAY / "Chunked_v1_56F_Three_Segments_External_Relay_EXP.api.json"
HASHES = {
    FREEZE: "f941c3c5cbfb6616000ff4d92d9e280302bda79d8f38ecad89822344a9cfd91e",
    RESUME: "7a959b24176e65dd12bd3f8e969d4abbc04bc38c65410a03c2ba44238c765191",
    FULL: "2083d27f87f968068bb4e905ddec79e9ae95e190be2956e3f2840d518b5a7d34",
}


class TinySavedChunkedInputs(io.ComfyNode):
    case = None

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[], outputs=[
            io.Latent.Output("first_pass_latent"), io.Model.Output("raw_model"),
            io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas"),
            io.Model.Output("relay_model"), io.Conditioning.Output("relay_positive"),
            io.Latent.Output("relay_full_av_latent"),
            io.Custom(PROMPT_RELAY_PLAN_TYPE).Output("prompt_relay_plan"),
            io.Noise.Output("noise"),
        ])

    @classmethod
    def execute(cls):
        raw, sampler, sigmas, relay_model, positive, full_av, relay_plan, source, _plan, noise, _context = cls.case
        return io.NodeOutput(source, raw, sampler, sigmas, relay_model, positive,
                             full_av, relay_plan, noise)


class TinySavedChunkedSink(io.ComfyNode):
    seen = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True,
                         inputs=[io.Latent.Input("av_latent")],
                         outputs=[io.String.Output("report")])

    @classmethod
    def execute(cls, av_latent):
        video, audio = av_latent["samples"].unbind()
        cls.seen.append((video.clone(), audio.clone()))
        return io.NodeOutput(str(float(video.sum())))


class TinySavedAVDecode(io.ComfyNode):
    """Replace only trained VAEs while exercising the saved VHS terminal."""

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
                         inputs=[io.Latent.Input("av_latent")],
                         outputs=[io.Image.Output("images"), io.Audio.Output("audio")])

    @classmethod
    def execute(cls, av_latent):
        video, _audio = av_latent["samples"].unbind()
        image = F.interpolate(video[0, 0].unsqueeze(1), size=(64, 64), mode="nearest")
        images = image.permute(0, 2, 3, 1).repeat(1, 1, 1, 3).sigmoid()
        audio = {"waveform": torch.zeros(1, 1, 24000), "sample_rate": 24000}
        return io.NodeOutput(images, audio)


class TinySavedFreezeReceipt(io.ComfyNode):
    seen = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True,
                         inputs=[io.String.Input("native_status"),
                                 io.String.Input("native_path"),
                                 io.String.Input("native_sha"),
                                 io.String.Input("native_manifest"),
                                 io.String.Input("segment_path"),
                                 io.String.Input("segment_sha"),
                                 io.String.Input("segment_manifest")],
                         outputs=[io.String.Output("report")])

    @classmethod
    def execute(cls, native_status, native_path, native_sha, native_manifest,
                segment_path, segment_sha, segment_manifest):
        cls.seen.append((native_status, native_path, native_sha, native_manifest,
                         segment_path, segment_sha, segment_manifest))
        return io.NodeOutput(native_status)


def _saved(path, expected):
    if not path.is_file():
        pytest.skip("Private S18 saved candidate API is not present")
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == HASHES[path]
    graph = json.loads(raw)
    assert graph == expected
    return graph


def _tiny_graph(saved, *, kind, native=None, segment=None):
    graph = deepcopy(saved)
    graph["50"] = {"class_type": "TinySavedChunkedInputs", "inputs": {}}
    graph["14"]["inputs"].update(target_width=64, target_height=64,
                                  tile_width=64, tile_height=64,
                                  minimum_tile_size=32)
    for key in ("20", "21", "26", "29"):
        if key in graph:
            graph[key]["inputs"]["first_pass_latent"] = ["40", 0] if kind == "resume" else ["50", 0]
    for key in ("21", "25", "28", "31"):
        if key in graph:
            graph[key]["inputs"]["noise"] = ["50", 8]
    for key in ("25", "28", "31"):
        if key in graph:
            graph[key]["inputs"]["sampler"] = ["50", 2]
            graph[key]["inputs"]["sigmas"] = ["50", 3]
    for key in ("34", "36", "38"):
        if key in graph:
            graph[key]["inputs"].update(
                raw_model=["50", 1], relay_model=["50", 4],
                relay_positive=["50", 5], relay_full_av_latent=["50", 6],
                prompt_relay_plan=["50", 7], sigmas=["50", 3],
            )
    if kind == "freeze":
        graph["40"]["inputs"].update(av_latent=["50", 0], confirm_save=True)
        graph["41"]["inputs"]["confirm_save"] = True
        roots = ("35", "40", "41")
    else:
        graph["51"] = {"class_type": "TinySavedChunkedSink",
                       "inputs": {"av_latent": ["31", 0]}}
        if kind == "resume":
            graph["40"]["inputs"].update(
                checkpoint_path=native[0], expected_manifest_json=native[2],
                expected_file_sha256=native[1],
            )
            graph["41"]["inputs"].update(artifact_path=segment[0], artifact_sha256=segment[1])
        roots = ("35", "37", "39", "51") if kind == "full" else ("37", "39", "51")
    graph = _prune_api(graph, roots)
    allowed = {"TinySavedChunkedInputs", "TinySavedChunkedSink",
               "MiniMaxH3ChunkedTwoPassPlanT8Advanced",
               "MiniMaxH3ChunkedSourceSegmentEXPT8", "MiniMaxH3ChunkedPass2PrepareEXPT8",
               "MiniMaxH3ChunkedLearnedLiftEXPT8", "MiniMaxH3ChunkedPass2SegmentEXPT8",
               "MiniMaxH3ChunkedV1RelayProjectEXPT8", "MiniMaxH3ChunkedV1RelayAuditEXPT8",
               "MiniMaxH3NativeLatentCheckpointSaveT8Advanced",
               "MiniMaxH3NativeLatentCheckpointLoadT8Advanced",
               "MiniMaxH3ChunkedV1SegmentSaveEXPT8", "MiniMaxH3ChunkedV1SegmentLoadEXPT8"}
    assert {item["class_type"] for item in graph.values()} <= allowed
    if kind == "resume":
        assert "12" not in graph and "25" not in graph
    return graph, roots


def _executor():
    import execution

    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *args, **kwargs: None)
    return execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                    asset_manager=SimpleNamespace(enabled=False))


def _run(graph, outputs):
    executor = _executor()
    executor.execute(deepcopy(graph), "saved-s18-tiny-candidate", execute_outputs=list(outputs))
    assert executor.success, executor.status_messages


def _av_sha(pair):
    return [hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()
            for tensor in pair]


def _cold_resume(output_root, native, segment):
    """Run the exact saved resume API in a new Python/Core interpreter."""
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    torch.set_num_threads(1)
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(ROOT.parents[1]))
        import nodes

        patch.setattr(folder_paths, "get_output_directory", lambda: output_root)
        for cls in (*SOURCE_NODES, *STAGE_NODES, *RELAY_NODES, *STORAGE_NODES,
                    MiniMaxH3ChunkedTwoPassPlanT8Advanced,
                    MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
                    MiniMaxH3NativeLatentCheckpointLoadT8Advanced,
                    TinySavedChunkedInputs, TinySavedChunkedSink, TinySavedFreezeReceipt):
            patch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)
        TinySavedChunkedInputs.case = _case(patch, with_frames=True)
        TinySavedChunkedSink.seen.clear()
        calls = []
        sample = chunked_stage_nodes.sample_chunked_pass2

        def observed_sample(*args):
            calls.append(args[4].index)
            return sample(*args)

        patch.setattr(chunked_stage_nodes, "sample_chunked_pass2", observed_sample)
        resume = _saved(RESUME, build_storage_pair()[1][1])
        graph, outputs = _tiny_graph(resume, kind="resume", native=native, segment=segment)
        _run(graph, outputs)
        assert calls == [1, 2]
        assert not torch.cuda.is_initialized()
        return {"sampled_segments": calls, "av_sha256": _av_sha(TinySavedChunkedSink.seen[-1]),
                "cuda_initialized": False}


def test_saved_relay_candidate_freeze_and_resume_sampling_branches(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    import nodes

    (expected_freeze, expected_resume) = build_storage_pair()
    expected_full = build_pair()[1]
    freeze = _saved(FREEZE, expected_freeze[1])
    resume = _saved(RESUME, expected_resume[1])
    full = _saved(FULL, expected_full)
    for cls in (*SOURCE_NODES, *STAGE_NODES, *RELAY_NODES, *STORAGE_NODES,
                MiniMaxH3ChunkedTwoPassPlanT8Advanced,
                MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
                MiniMaxH3NativeLatentCheckpointLoadT8Advanced,
                TinySavedChunkedInputs, TinySavedChunkedSink, TinySavedFreezeReceipt):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.define_schema().node_id, cls)
    torch.set_num_threads(1)
    TinySavedChunkedInputs.case = _case(monkeypatch, with_frames=True)
    TinySavedChunkedSink.seen.clear()
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output))

    calls = []
    sample = chunked_stage_nodes.sample_chunked_pass2

    def observed_sample(*args):
        calls.append(args[4].index)
        return sample(*args)

    monkeypatch.setattr(chunked_stage_nodes, "sample_chunked_pass2", observed_sample)
    full, full_outputs = _tiny_graph(full, kind="full")
    _run(full, full_outputs)
    assert calls == [0, 1, 2]
    expected_av = TinySavedChunkedSink.seen[-1]

    freeze, freeze_outputs = _tiny_graph(freeze, kind="freeze")
    freeze["52"] = {"class_type": "TinySavedFreezeReceipt", "inputs": {
        "native_status": ["40", 1], "native_path": ["40", 2],
        "native_sha": ["40", 3], "native_manifest": ["40", 4],
        "segment_path": ["41", 2], "segment_sha": ["41", 3],
        "segment_manifest": ["41", 4],
    }}
    TinySavedFreezeReceipt.seen.clear()
    calls.clear()
    _run(freeze, (*freeze_outputs, "52"))
    assert calls == [0]
    native_root = output / "MiniMaxH3" / "latent_checkpoints"
    segment_root = output / "MiniMaxH3" / "chunked_v1_segment_artifacts"
    native_files = list(native_root.glob("*.h3latent.safetensors"))
    manifests = list(segment_root.glob("*/manifest.json"))
    assert len(native_files) == len(manifests) == 1
    assert len(TinySavedFreezeReceipt.seen) == 1
    (status, native_path, native_sha, native_manifest,
     segment_path, segment_sha, segment_manifest) = TinySavedFreezeReceipt.seen[0]
    assert status == "SAVED_VERIFIED"
    assert native_path == native_files[0].name
    assert native_sha == file_sha(native_files[0]).upper()
    assert segment_path == manifests[0].relative_to(segment_root).as_posix()
    assert segment_sha == file_sha(manifests[0])
    assert isinstance(json.loads(native_manifest), dict)
    assert json.loads(segment_manifest) == json.loads(manifests[0].read_text(encoding="utf-8"))
    native = (native_path, native_sha, native_manifest)
    segment = (segment_path, segment_sha)

    resume, resume_outputs = _tiny_graph(resume, kind="resume", native=native, segment=segment)
    calls.clear()
    _run(resume, resume_outputs)
    assert calls == [1, 2]
    actual_av = TinySavedChunkedSink.seen[-1]
    assert all(torch.equal(actual, expected) for actual, expected in
               zip(actual_av, expected_av, strict=True))

    child_code = """
import json,runpy,sys
import comfy.options
comfy.options.enable_args_parsing()
args=sys.argv[1:]
sys.argv=['s18-cold-resume','--cpu','--use-pytorch-cross-attention']
import comfy.cli_args
sys.argv=['s18-cold-resume']
runpy.run_path('tests/conftest.py')
from test_modular_chunked_v1_saved_storage_candidate_core import _cold_resume
print('RESULT='+json.dumps(_cold_resume(args[0],(args[1],args[2],args[3]),(args[4],args[5])),sort_keys=True))
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(
        [str(ROOT.parents[1]), *env.get("PYTHONPATH", "").split(os.pathsep)]))
    child = subprocess.run(
        [sys.executable, "-c", child_code, str(output), *native, *segment],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=90,
    )
    assert child.returncode == 0, child.stderr
    cold = json.loads(next(line[7:] for line in child.stdout.splitlines()
                           if line.startswith("RESULT=")))
    assert cold == {"sampled_segments": [1, 2], "av_sha256": _av_sha(expected_av),
                    "cuda_initialized": False}

    bad_resume = deepcopy(resume)
    bad_resume["41"]["inputs"]["artifact_sha256"] = "0" * 64
    calls.clear()
    failed = _executor()
    failed.execute(bad_resume, "saved-s18-bad-segment-sha", execute_outputs=list(resume_outputs))
    assert not failed.success and calls == []

    bad_native = deepcopy(resume)
    bad_native["40"]["inputs"]["expected_file_sha256"] = "0" * 64
    calls.clear()
    failed = _executor()
    failed.execute(bad_native, "saved-s18-bad-native-sha", execute_outputs=list(resume_outputs))
    assert not failed.success and calls == []

    bad_manifest = deepcopy(resume)
    bad_manifest["40"]["inputs"]["expected_manifest_json"] = "{}"
    calls.clear()
    failed = _executor()
    failed.execute(bad_manifest, "saved-s18-bad-native-manifest",
                   execute_outputs=list(resume_outputs))
    assert not failed.success and calls == []

    plugin_root = ROOT.parent / "ComfyUI-VideoHelperSuite"
    assert (plugin_root / "videohelpersuite/nodes.py").is_file()
    monkeypatch.syspath_prepend(str(plugin_root))
    import server

    monkeypatch.setattr(server.PromptServer, "instance",
                        SimpleNamespace(prompt_queue=SimpleNamespace()), raising=False)
    from videohelpersuite.nodes import VideoCombine

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "VHS_VideoCombine", VideoCombine)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "TinySavedAVDecode", TinySavedAVDecode)
    media_graph = deepcopy(resume)
    media_graph["18"] = {"class_type": "TinySavedAVDecode",
                         "inputs": {"av_latent": ["31", 0]}}
    media_graph["19"] = deepcopy(_saved(RESUME, expected_resume[1])["19"])
    media_graph["19"]["inputs"].update(images=["18", 0], audio=["18", 1],
                                        filename_prefix="s18_tiny_resume_vhs")
    calls.clear()
    _run(media_graph, ("19",))
    assert calls == [1, 2]
    media_files = list(output.rglob("s18_tiny_resume_vhs*-audio.mp4"))
    assert len(media_files) == 1
    with av.open(str(media_files[0])) as container:
        assert len(container.streams.video) == len(container.streams.audio) == 1
        assert len(list(container.decode(video=0))) == expected_av[0].shape[2]
        container.seek(0)
        assert list(container.decode(audio=0))
