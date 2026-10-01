"""Actual CPU Core freeze and fresh-executor only-later-segments behavior."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import folder_paths
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io
import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as old
from h3_audio_t8_pkg.modular_sampling.chunked_source_nodes import NODES as SOURCE_NODES
from h3_audio_t8_pkg.modular_sampling.chunked_stage_nodes import NODES as STAGE_NODES
from h3_audio_t8_pkg.modular_sampling.chunked_v1_storage import file_sha
from h3_audio_t8_pkg.modular_sampling.chunked_v1_storage_nodes import NODES as STORAGE_NODES
from h3_audio_t8_pkg.nodes_native_latent_checkpoint_advanced import (
    MiniMaxH3NativeLatentCheckpointLoadT8Advanced,
    MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
)
from h3_audio_t8_pkg.native_latent_checkpoint_advanced import load_native_h3_av_checkpoint
from h3_audio_t8_pkg.modular_sampling.chunked_source import slice_chunked_source
from test_chunked_two_pass_global_noise_advanced import _CountingCoordinateNoise, _plan
from test_modular_chunked_core_cache import (
    TinyChunkedConditioning, TinyChunkedModel, TinyChunkedSource, _graph,
)
from test_modular_chunked_v1_storage import _three_segment_source


class TinyThreeSegmentSource(TinyChunkedSource):
    @classmethod
    def execute(cls):
        return io.NodeOutput(_three_segment_source(),
                             _plan(old.build_chunked_two_pass_plan, strategy="full_frame_safe"),
                             _CountingCoordinateNoise(), object(), torch.tensor([1., 0.]))


class TinyThreeSegmentSink(io.ComfyNode):
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


def _full_graph():
    graph = _graph()
    graph["1"]["class_type"] = "TinyThreeSegmentSource"
    graph["6"] = {"class_type": "TinyChunkedModel", "inputs": {"offset": 0.3}}
    graph["7"] = {"class_type": "TinyChunkedConditioning", "inputs": {"prompt": "third"}}
    graph["31"] = {"class_type": "MiniMaxH3ChunkedSourceSegmentEXPT8", "inputs": {
        "first_pass_latent": ["1", 0], "plan": ["1", 1], "segment_index": 2}}
    graph["32"] = {"class_type": "MiniMaxH3ChunkedLearnedLiftEXPT8", "inputs": {
        "source_segment": ["31", 0], "segment_spec": ["31", 1],
        "pass2_context": ["10", 0], "plan": ["1", 1]}}
    graph["33"] = {"class_type": "MiniMaxH3ChunkedPass2SegmentEXPT8", "inputs": {
        "model": ["6", 0], "positive": ["7", 0],
        "source_segment": ["31", 0], "lifted_segment": ["32", 0],
        "segment_spec": ["31", 1], "pass2_context": ["10", 0],
        "plan": ["1", 1], "noise": ["1", 2], "sampler": ["1", 3],
        "sigmas": ["1", 4], "previous_result": ["23", 1], "cfg": 1.0}}
    graph["30"] = {"class_type": "TinyThreeSegmentSink", "inputs": {
        "av_latent": ["33", 0]}}
    return graph


def _executor():
    import execution

    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *args, **kwargs: None)
    return execution.PromptExecutor(
        server, cache_args={"ram": 0., "ram_inactive": 0.},
        asset_manager=SimpleNamespace(enabled=False),
    )


def _run(graph, *outputs):
    executor = _executor()
    executor.execute(deepcopy(graph), "chunked-v1-frozen-segment", execute_outputs=list(outputs))
    assert executor.success, executor.status_messages


def test_real_core_freeze_then_fresh_executor_only_samples_segments_one_and_two(
        monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import nodes

    for cls in (*SOURCE_NODES, *STAGE_NODES, *STORAGE_NODES,
                MiniMaxH3NativeLatentCheckpointSaveT8Advanced,
                MiniMaxH3NativeLatentCheckpointLoadT8Advanced,
                TinyThreeSegmentSource, TinyChunkedModel, TinyChunkedConditioning,
                TinyThreeSegmentSink):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output_root))

    def fake_lift(chunk, *_args):
        video, audio = chunk["samples"].unbind()
        return ({"samples": NestedTensor((
            video.repeat_interleave(2, -1).repeat_interleave(2, -2), audio))},
            64, 64, "{}")

    calls = []

    def fake_spatial(video, _audio, _positive, _plan, model, *_args, **_kwargs):
        calls.append(model.offset)
        return video + model.offset, {"mock": True}

    monkeypatch.setattr(old, "learned_upscale_h3_av_latent", fake_lift)
    monkeypatch.setattr(old, "_spatial_resample", fake_spatial)
    full = _full_graph()
    TinyThreeSegmentSink.seen.clear()
    _run(full, "30")
    expected = TinyThreeSegmentSink.seen[-1]
    assert calls == [0.1, 0.2, 0.3]

    freeze = {key: deepcopy(full[key]) for key in
              ("1", "2", "4", "10", "11", "12", "13")}
    freeze["40"] = {"class_type": "MiniMaxH3NativeLatentCheckpointSaveT8Advanced",
                    "inputs": {"av_latent": ["1", 0], "filename_prefix": "chunked_v1_test",
                               "checkpoint_id": "chunked_v1_test", "confirm_save": True,
                               "verify_after_write": True, "hash_chunk_megabytes": 8}}
    freeze["41"] = {"class_type": "MiniMaxH3ChunkedV1SegmentSaveEXPT8",
                    "inputs": {"segment_result": ["13", 1],
                               "source_segment": ["11", 0], "segment_spec": ["11", 1],
                               "pass2_context": ["10", 0], "plan": ["1", 1],
                               "confirm_save": True}}
    calls.clear()
    _run(freeze, "40", "41")
    assert calls == [0.1]
    native_root = output_root / "MiniMaxH3" / "latent_checkpoints"
    segment_root = output_root / "MiniMaxH3" / "chunked_v1_segment_artifacts"
    native_files = list(native_root.glob("*.h3latent.safetensors"))
    manifests = list(segment_root.glob("*/manifest.json"))
    assert len(native_files) == len(manifests) == 1
    loaded_av = load_native_h3_av_checkpoint(
        native_root, native_files[0].name,
        expected_file_sha256=file_sha(native_files[0]),
    )[0]
    loaded_spec = slice_chunked_source(
        loaded_av, _plan(old.build_chunked_two_pass_plan, strategy="full_frame_safe"), 0,
    )[1]
    saved_binding = json.loads(manifests[0].read_text(encoding="utf-8"))["binding"]
    assert loaded_spec.source_identity["samples"] == saved_binding["source_identity"]["samples"]
    assert set(loaded_spec.source_identity) - set(saved_binding["source_identity"]) == {
        "t8_native_latent_checkpoint"}

    resume = {key: deepcopy(full[key]) for key in
              ("1", "3", "5", "6", "7", "10", "11", "21", "22", "23",
               "30", "31", "32", "33")}
    resume["40"] = {"class_type": "MiniMaxH3NativeLatentCheckpointLoadT8Advanced",
                    "inputs": {"checkpoint_path": native_files[0].name,
                               "expected_manifest_json": "",
                               "expected_file_sha256": file_sha(native_files[0]),
                               "hash_chunk_megabytes": 8}}
    for key in ("10", "11", "21", "31"):
        resume[key]["inputs"]["first_pass_latent"] = ["40", 0]
    resume["41"] = {"class_type": "MiniMaxH3ChunkedV1SegmentLoadEXPT8",
                    "inputs": {"source_segment": ["11", 0],
                               "segment_spec": ["11", 1], "pass2_context": ["10", 0],
                               "plan": ["1", 1],
                               "artifact_path": manifests[0].relative_to(segment_root).as_posix(),
                               "artifact_sha256": file_sha(manifests[0])}}
    resume["23"]["inputs"]["previous_result"] = ["41", 1]
    assert "13" not in resume and "12" not in resume
    calls.clear()
    _run(resume, "30")
    assert calls == [0.2, 0.3]
    actual = TinyThreeSegmentSink.seen[-1]
    assert all(torch.equal(got, want) for got, want in zip(actual, expected, strict=True))

    bad_segment = deepcopy(resume)
    bad_segment["41"]["inputs"]["artifact_sha256"] = "0" * 64
    calls.clear()
    failed = _executor()
    failed.execute(bad_segment, "chunked-v1-bad-segment-sha", execute_outputs=["30"])
    assert not failed.success and calls == []

    bad_first_pass = deepcopy(resume)
    bad_first_pass["40"]["inputs"]["expected_file_sha256"] = "0" * 64
    calls.clear()
    failed = _executor()
    failed.execute(bad_first_pass, "chunked-v1-bad-av-sha", execute_outputs=["30"])
    assert not failed.success and calls == []
