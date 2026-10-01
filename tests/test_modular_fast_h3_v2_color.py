"""Opt-in V2 external Color Match keeps the accepted source and old route intact."""

import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from legacy_v2_frontend import assert_legacy_v2_frontend
import torch
import av
import folder_paths
import numpy as np

from h3_audio_t8_pkg import long_video_delivery as delivery
from h3_audio_t8_pkg import long_video_dual_color
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_color as color
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_color_candidate as colored_writer
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_review as review
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2_job_binding import (
    FastH3V2CurrentJobBinding, SCHEMA as BINDING_SCHEMA,
    implementation_sha256 as binding_implementation,
)
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2_media import (
    FastH3V2MediaReceipt, SCHEMA as MEDIA_SCHEMA,
    implementation_sha256 as media_implementation,
)
from h3_audio_t8_pkg.modular_sampling.results import _input_identity, canonical
from tools import build_modular_fast_h3_v2_first_segment_workflow as first_builder
from tools import build_modular_fast_h3_v2_continuation_workflow as second_builder
from tools import run_modular_s08_old_new_pair_gpu as gpu_probe


def _signed(payload):
    encoded = canonical(payload)
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@pytest.fixture
def accepted_parent(tmp_path):
    path = tmp_path / "previous.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        stream.codec_context.thread_count = 1
        for _ in range(8):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), 102, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    record = {"index": 0, "candidate_id": "parent", "video_path": path.name,
              "video_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "frame_count": 8}
    manifest = {"chain_id": "chain", "segments": [record]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, manifest


def _source(accepted_root, manifest, monkeypatch):
    parent_sha = manifest["segments"][0]["video_sha256"]
    parent = {"parent_candidate_id": "parent", "parent_revision": 1,
              "timeline_start_frame": 124, "accepted_video_sha256": parent_sha}
    binding_payload = {"schema": BINDING_SCHEMA, "implementation_sha256": binding_implementation(),
                       "chain_id": "chain", "segment_index": 1, "parent": parent,
                       "current_recipe_sha256": "a" * 64, "model_id": "model",
                       "sampling_summary": "a" * 64, "seed": 7}
    binding_json, binding_sha = _signed(binding_payload)
    binding = FastH3V2CurrentJobBinding(binding_json, binding_sha, None, None, None)
    binding_state = deepcopy(binding_payload)
    monkeypatch.setattr(FastH3V2CurrentJobBinding, "verify", lambda self: binding_state)
    frames = torch.full((68, 32, 32, 3), .41)
    frames[1, 8:24, :16, 0] += .02
    audio = {"waveform": torch.zeros((1, 2, 90752)), "sample_rate": 32000}
    media_payload = {"schema": MEDIA_SCHEMA, "implementation_sha256": media_implementation(),
                     "portable_identity": True, "frames": _input_identity(frames),
                     "audio": _input_identity(audio), "trim": {"fps": 24, "start_frame": 22,
                                                                   "frame_count": 68},
                     "current_recipe_sha256": "a" * 64,
                     "current_job_binding_sha256": binding_sha,
                     "high_stage_receipt_sha256": "b" * 64}
    media_json, media_sha = _signed(media_payload)
    source = FastH3V2MediaReceipt(media_json, media_sha, binding,
                                  SimpleNamespace(output={"samples": "unmodified"}), frames, audio)
    monkeypatch.setattr(FastH3V2MediaReceipt, "verify", lambda self: media_payload)
    monkeypatch.setattr(delivery, "long_video_chain_root", lambda _chain_id: accepted_root)
    return source, binding_state


def test_external_color_matches_old_algorithm_and_rejects_changed_parent(accepted_parent, monkeypatch):
    root, manifest = accepted_parent
    source, binding = _source(root, manifest, monkeypatch)
    expected, old_report = long_video_dual_color.correct_dual_segment_color(
        source.frames, root, "chain", 1, "parent", True, "bounded_motion_color_exp")
    frames, audio, receipt, _ = color.color_match_current_media(source)
    assert torch.equal(frames, expected)
    assert receipt.verify()["color_report"] == old_report
    assert audio is source.audio and receipt.audio is source.audio
    assert not torch.equal(frames, source.frames)
    binding["parent"] = {**binding["parent"], "accepted_video_sha256": "0" * 64}
    with pytest.raises(ValueError, match="accepted parent"):
        receipt.verify()


def test_external_color_first_segment_is_rgb_identity_and_audio_passthrough(accepted_parent, monkeypatch):
    root, manifest = accepted_parent
    source, binding = _source(root, manifest, monkeypatch)
    binding["segment_index"] = 0
    binding["parent"] = None
    frames, audio, receipt, _ = color.color_match_current_media(source)
    assert frames is source.frames and audio is source.audio
    assert receipt.verify()["color_report"]["status"] == "first_segment_identity"
    damaged = color.FastH3V2ColoredMediaReceipt(receipt.payload_json, receipt.sha256,
                                                source, frames[:1], audio)
    with pytest.raises(ValueError, match="changed audio or lost"):
        damaged.verify()


def test_colored_candidate_has_versioned_source_and_old_review_accepts_it(accepted_parent, monkeypatch):
    root, manifest = accepted_parent
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(root))
    source, _binding = _source(root, manifest, monkeypatch)
    frames, audio, receipt, _ = color.color_match_current_media(source)
    candidate_dir = root / "candidates" / "segment_00001" / "colored"
    candidate_path = candidate_dir / "candidate.json"
    movie_path = candidate_dir / "candidate.mp4"
    descriptor = {"chain_id": "chain", "index": 1, "candidate_id": "colored",
                  "parent_candidate_id": "parent", "parent_manifest_revision": 1,
                  "timeline_start_frame": 124, "timeline_end_frame": 192,
                  "frame_count": 68, "model_id": "model", "sampling_summary": "a" * 64,
                  "seed": 7, "is_final_segment": True,
                  "video_sha256": "c" * 64, "context_sha256": ""}

    def save(**kwargs):
        assert kwargs["frames"] is frames and kwargs["audio"] is audio
        assert kwargs["av_latent"] is source.high_result.output
        candidate_dir.mkdir(parents=True)
        candidate_path.write_text(json.dumps(descriptor), encoding="utf-8")
        movie_path.write_bytes(b"mock movie: encoder is covered by existing candidate tests")
        return str(candidate_path), str(movie_path), ""

    monkeypatch.setattr(delivery, "save_long_video_candidate", save)
    monkeypatch.setattr(delivery, "load_long_video_candidate_descriptor",
                        lambda _path: (descriptor, str(movie_path)))
    monkeypatch.setattr(delivery, "_load_candidate",
                        lambda _path: (descriptor, root, movie_path))
    candidate, movie, sidecar = colored_writer.save_colored_candidate(receipt, "colored")[:3]
    assert candidate == str(candidate_path) and movie == str(movie_path)
    assert json.loads(Path(sidecar).read_text(encoding="utf-8"))["schema"] == colored_writer.SCHEMA
    checked, checked_movie, _ = review.verify_saved_current_candidate(candidate)
    assert checked == descriptor and checked_movie == movie
    corrupted = json.loads(Path(sidecar).read_text(encoding="utf-8"))
    corrupted["color_receipt"]["parent_video_sha256"] = "0" * 64
    Path(sidecar).write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        review.verify_saved_current_candidate(candidate)


def test_opt_in_color_graphs_have_separate_node_and_keep_default_graphs():
    flags = {"with_current_recipe": True, "with_stage_attestation": True,
             "with_condition_provenance": True, "with_handoff_provenance": True,
             "with_job_binding": True, "with_media_provenance": True,
             "with_candidate_save": True}
    for build, extra in ((first_builder.first_segment_graph, {"first_frame": "first.png"}),
                         (second_builder.continuation_graph,
                          {"with_delivery": True, "first_frame": "first.png"})):
        plain = build(**extra, **flags)
        colored = build(**extra, **flags, with_color_match=True)
        assert "105" not in plain and "106" not in plain
        assert plain["98"]["class_type"] == "MiniMaxH3FastH3V2CurrentCandidateSaveEXPT8"
        assert colored["105"]["inputs"]["media_receipt"] == ["96", 2]
        assert colored["105"]["inputs"]["mode"] == "bounded_motion_color_exp"
        assert colored["98"]["class_type"] == "MiniMaxH3FastH3V2ColoredCandidateSaveEXPT8"
        assert colored["98"]["inputs"]["color_receipt"] == ["105", 2]
        assert colored["15"]["inputs"]["images"] == ["105", 0]
        with pytest.raises(ValueError, match="External Color Match requires"):
            build(**extra, with_color_match=True)


def test_external_color_graphs_pass_current_core_static_validation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes as core_nodes

    first_frame = "02_清晰身份参考图_首段.png"
    flags = {"with_current_recipe": True, "with_stage_attestation": True,
             "with_condition_provenance": True, "with_handoff_provenance": True,
             "with_job_binding": True, "with_media_provenance": True,
             "with_candidate_save": True, "with_color_match": True}
    info = first_builder.base.load_live_info()
    info["LoadImage"] = first_builder.base.native_info(
        "LoadImage", core_nodes.NODE_CLASS_MAPPINGS["LoadImage"])
    graphs = (first_builder.first_segment_graph(first_frame, **flags),
              second_builder.continuation_graph(with_delivery=True, first_frame=first_frame, **flags))
    for graph in graphs:
        front_api, selected = first_builder.base.selected_frontend_schema(graph, info)
        assert front_api["105"] == graph["105"] and selected
        validation = asyncio.run(execution.validate_prompt("v2-external-color", deepcopy(graph), None))
        assert validation[0] and validation[3] == {}


def test_private_colored_workflow_files_match_current_builders():
    root = Path(__file__).resolve().parents[1]
    first_frame = "02_清晰身份参考图_首段.png"
    flags = {"with_current_recipe": True, "with_stage_attestation": True,
             "with_condition_provenance": True, "with_handoff_provenance": True,
             "with_job_binding": True, "with_media_provenance": True,
             "with_candidate_save": True, "with_color_match": True,
             "with_frozen_low_bundle": True}
    info = first_builder.base.load_live_info()
    for label, folder, build, build_candidate, extra in (
            ("first", root / "artifacts/development/modular-sampling-s08-colored-first-v1-20260926",
             first_builder.first_segment_graph, first_builder.build_candidate, {}),
            ("second", root / "artifacts/development/modular-sampling-s08-colored-second-v1-20260926",
             second_builder.continuation_graph, second_builder.build_candidate, {"with_delivery": True})):
        saved = json.loads((folder / "audit.json").read_text(encoding="utf-8"))["candidates"]
        assert len(saved) == 2 and all(item["all_outputs_valid"] for item in saved)
        for resume, item in zip((False, True), saved):
            graph = build(resume_high=resume, first_frame=first_frame, **extra, **flags)
            api, frontend, _ = build_candidate(graph, info, resume_high=resume, **extra, **flags)
            assert json.loads((folder / (item["name"] + ".api.json")).read_text(encoding="utf-8")) == api
            current = json.loads((folder / (item["name"] + ".json")).read_text(encoding="utf-8"))
            assert_legacy_v2_frontend(current, frontend, info)


def test_old_fixed_second_probe_is_current_core_static_valid(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution

    graph = gpu_probe.second_graph({"candidate_id": "parent", "revision": 1,
                                    "job_sha256": "a" * 64}, with_color_match=True)
    validation = asyncio.run(execution.validate_prompt("v2-fixed-second", deepcopy(graph), None))
    assert validation[0] and validation[3] == {}


def test_gpu_probe_memory_wrappers_do_not_replace_external_color_nodes():
    first = gpu_probe.first_graph(with_color_match=True)
    second = gpu_probe.second_graph({"candidate_id": "parent", "revision": 1,
                                     "job_sha256": "a" * 64}, with_color_match=True)
    assert second["77"]["inputs"]["render_policy"] == "old_fixed_124"
    assert second["78"]["inputs"]["render_policy"] == "old_fixed_124"
    assert second["74"]["inputs"]["accepted_end_frame"] == 192
    assert second["75"]["inputs"]["accepted_end_frame"] == 192
    for graph in (first, second):
        recipe_nodes = [node for node in graph.values()
                        if node["class_type"] == "MiniMaxH3FastH3V2CurrentRecipeEXPT8"]
        assert len(recipe_nodes) == 1
        assert recipe_nodes[0]["inputs"]["continuation_render_frames"] == 124
    for graph in (first, second):
        assert graph["105"]["class_type"] == "MiniMaxH3FastH3V2ExternalColorMatchEXPT8"
        assert graph["106"]["class_type"] == "PreviewAny"
        memory = [key for key, node in graph.items() if node["class_type"] == gpu_probe.old.LOWVRAM]
        assert len(memory) == 2 and not {"105", "106"}.intersection(memory)
        assert graph["98"]["inputs"]["color_receipt"] == ["105", 2]


def test_private_pair_controller_selects_old_pytorch_attention_only_when_requested(tmp_path):
    args = SimpleNamespace(python=Path(sys.executable), host="127.0.0.1", port=8895,
                           comfy_root=tmp_path, use_pytorch_cross_attention=True)
    command = gpu_probe.shared._server_command(args, tmp_path)
    assert command.count("--use-pytorch-cross-attention") == 1
    del args.use_pytorch_cross_attention
    assert "--use-pytorch-cross-attention" not in gpu_probe.shared._server_command(args, tmp_path)
