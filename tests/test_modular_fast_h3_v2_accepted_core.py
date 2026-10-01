"""Run the accepted-parent V2 second window in Core with explicit CPU asset doubles."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import av
import folder_paths
import pytest
import torch
from comfy_api.latest import io

from h3_audio_t8_pkg.core import empty_av_latent
from h3_audio_t8_pkg import long_video_delivery as delivery
from h3_audio_t8_pkg.modular_sampling import continuation as accepted_source
from h3_audio_t8_pkg.modular_sampling import continuation_nodes
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_continuation_nodes
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_job as current_job
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_job_nodes as current_job_nodes
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_conditioning as current_origin
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_stage_attest as current_attest
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_upscale as current_upscale
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_current_handoff as current_handoff
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_media as current_media
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_candidate as current_candidate
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_review as current_review
from h3_audio_t8_pkg.modular_sampling import nodes as stage_nodes
from h3_audio_t8_pkg.modular_sampling import stage_unet_loader
from h3_audio_t8_pkg.modular_sampling.storage import file_sha
from h3_audio_t8_pkg.modular_sampling.results import selected_model_identity
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2_handoff_nodes import NODES as HANDOFF_NODES
from h3_audio_t8_pkg.modular_sampling.stage_unet_loader_nodes import NODES as LOADER_NODES
from h3_audio_t8_pkg.nodes import MiniMaxH3AVDecodeT8, MiniMaxH3OutputTrimT8
from h3_audio_t8_pkg.nodes_fast_h3_v2_advanced import MiniMaxH3FastH3V2RuntimeAuditEXPT8
from h3_audio_t8_pkg.nodes_prompt_relay_long_video_advanced import (
    MiniMaxH3PromptRelayLongVideoConditioningT8Advanced,
)
from h3_audio_t8_pkg.nodes_prompt_relay_advanced import MiniMaxH3PromptRelayPlanT8Advanced
from h3_audio_t8_pkg.prompt_relay_long_video_advanced import PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY
from test_modular_fast_h3_v2_saved_candidate_core import (
    TinyCandidateCLIP, TinyCandidateUNET, TinyCandidateVAE,
)
from test_modular_progressive_core_cache import InterpolationDouble
from test_progressive_continuation import accepted  # noqa: F401
from tools.build_modular_fast_h3_v2_continuation_workflow import continuation_graph


class TinyAcceptedUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        from h3_audio_t8_pkg.nodes_learned_latent_upscale_advanced import MiniMaxH3LearnedLatentUpscaleT8Advanced
        return MiniMaxH3LearnedLatentUpscaleT8Advanced.define_schema()

    @classmethod
    def execute(cls, av_latent, model_name, size_mode, scale_by, target_megapixels,
                target_width, target_height, aspect_policy, max_anisotropy, precision, release_policy):
        assert model_name.endswith("latent_upscaler_3d_fp16.safetensors")
        assert (size_mode, target_width, target_height, aspect_policy) == (
            "target_dimensions", 128, 64, "honor_dimensions_exp")
        assert (scale_by, precision, release_policy) == (2., "fp16", "offload_after")
        assert max_anisotropy == 1.05 and target_megapixels > 0
        return io.NodeOutput(*InterpolationDouble.execute(av_latent, 128, 64).result, "explicit_cpu_double")


def tiny_graph(case, *, resume_high=False, with_stage_attestation=False,
               with_condition_provenance=False, with_handoff_provenance=False,
               with_job_binding=False, with_media_provenance=False,
               with_frozen_low_bundle=False, with_old_fixed_window=False):
    graph = continuation_graph(resume_high=resume_high, with_delivery=True,
        with_current_recipe=with_stage_attestation,
        with_stage_attestation=with_stage_attestation,
        with_condition_provenance=with_condition_provenance,
        with_handoff_provenance=with_handoff_provenance,
        with_job_binding=with_job_binding,
        with_media_provenance=with_media_provenance,
        with_frozen_low_bundle=with_frozen_low_bundle,
        first_frame="02_清晰身份参考图_首段.png" if with_stage_attestation else "")
    graph["70"]["inputs"] = {**case.request, "previous_job_sha256": case.request["job_sha256"]}
    graph["70"]["inputs"].pop("job_sha256")
    graph["91" if "91" in graph else "23"]["inputs"].update(target_width=128, target_height=64)
    for key in (("89", "90") if with_condition_provenance else ("9", "24")):
        if key in graph:
            graph[key]["inputs"]["query_chunk_rows"] = 256 if with_condition_provenance else 64
    for key in ("10", "26"):
        if key in graph:
            graph[key]["inputs"]["min_tokens"] = 0
    for key in ("40", "47"):
        if key in graph:
            graph[key]["inputs"].update(global_prompt="One stable scene.",
                local_prompts="She walks.\nShe stops.\nShe turns.")
    if with_stage_attestation and "83" in graph:
        graph["83"]["inputs"].update(
            low_width=case.request["low_width"], low_height=case.request["low_height"],
            width=case.request["width"], height=case.request["height"])
    if with_old_fixed_window:
        if not with_stage_attestation:
            raise ValueError("The fixed-window Core test requires a current V2 recipe")
        graph["83"]["inputs"]["continuation_render_frames"] = 124
        for key in ("77", "78"):
            graph[key]["inputs"]["render_policy"] = "old_fixed_124"
        for key in ("74", "75"):
            graph[key]["inputs"]["accepted_end_frame"] = 192
    return graph


def _decoded_delivery_hashes(path):
    with av.open(str(path)) as container:
        assert len(container.streams.video) == len(container.streams.audio) == 1
        video = container.streams.video[0]
        audio = container.streams.audio[0]
        assert video.codec_context.name == "h264"
        assert audio.codec_context.name == "aac"
        frames = list(container.decode(video=0))
        assert len(frames) == 68
        assert all((frame.width, frame.height) == (128, 64) for frame in frames)
        assert [float(frame.pts * frame.time_base) for frame in frames] == pytest.approx(
            [index / 24 for index in range(68)])
        rgb = hashlib.sha256()
        for frame in frames:
            rgb.update(frame.to_ndarray(format="rgb24").tobytes())
        container.seek(0)
        pcm = hashlib.sha256()
        sound_frames = list(container.decode(audio=0))
        assert sound_frames and audio.codec_context.sample_rate == 32000
        assert sum(frame.samples for frame in sound_frames) >= 68 / 24 * 32000
        assert any(frame.to_ndarray().any() for frame in sound_frames)
        for frame in sound_frames:
            pcm.update(frame.to_ndarray().tobytes())
    return rgb.hexdigest(), pcm.hexdigest()


def test_real_core_executes_accepted_second_low_then_high(accepted, monkeypatch, tmp_path, request):  # noqa: F811
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes
    from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
    from comfy_extras.nodes_preview_any import PreviewAny
    from comfy_extras.nodes_video import CreateVideo, SaveVideo

    registered = [TinyCandidateUNET, TinyCandidateCLIP, TinyCandidateVAE, TinyAcceptedUpscale,
                  BasicGuider, RandomNoise, PreviewAny, CreateVideo, SaveVideo,
                  MiniMaxH3AVDecodeT8, MiniMaxH3OutputTrimT8, MiniMaxH3FastH3V2RuntimeAuditEXPT8,
                  MiniMaxH3PromptRelayPlanT8Advanced,
                  MiniMaxH3PromptRelayLongVideoConditioningT8Advanced,
                  *continuation_nodes.NODES, *fast_h3_v2_continuation_nodes.NODES,
                  *HANDOFF_NODES, *LOADER_NODES, *current_job_nodes.NODES]
    registered += [getattr(stage_nodes, name) for name in (
        "MiniMaxH3FastH3V2StageSetupEXPT8", "MiniMaxH3StageSamplerEXPT8",
        "MiniMaxH3StageEAVConfigEXPT8", "MiniMaxH3StageEAVApplyEXPT8",
        "MiniMaxH3StageEAVAuditEXPT8", "MiniMaxH3StageSaveEXPT8", "MiniMaxH3StageLoadEXPT8")]
    for cls in registered:
        name = cls.define_schema().node_id if hasattr(cls, "define_schema") else cls.__name__
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, name, cls)
    monkeypatch.setattr(accepted_source.delivery, "long_video_chain_root", lambda _chain: accepted.root)
    monkeypatch.setattr(stage_nodes, "_stage_store_root", lambda: tmp_path)
    output_root = tmp_path / "output"
    output_root.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output_root))
    monkeypatch.setattr(stage_unet_loader, "_load_unet", lambda _name, _dtype: TinyCandidateUNET.execute(
        "fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors", "default").result[0])
    stages, receipts = [], []
    original = stage_nodes.sample_stage

    def observed(*args):
        stages.append(args[-1].stage)
        if stages == ["low_0_4"]:
            model = args[1].model_patcher
            assert selected_model_identity(model).get("portable_cache_reuse", True)
            broken_binding = model.clone()
            attachment = dict(model.get_attachment(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY))
            attachment["binding_hash"] = "0" * 64
            broken_binding.set_attachments(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY, attachment)
            assert not selected_model_identity(broken_binding).get("portable_cache_reuse", True)
            broken_plan = model.clone()
            attachment = dict(model.get_attachment(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY))
            attachment["projected_plan_hash"] = "0" * 64
            broken_plan.set_attachments(PROMPT_RELAY_LONG_VIDEO_ATTACHMENT_KEY, attachment)
            assert not selected_model_identity(broken_plan).get("portable_cache_reuse", True)
            unknown_owner = model.clone()
            unknown_owner.set_attachments("unverified_third_party_owner", {"enabled": True})
            assert not selected_model_identity(unknown_owner).get("portable_cache_reuse", True)
            unknown_payload = model.clone()
            unknown_payload.add_object_patch("extra_conds", lambda **kwargs: kwargs)
            assert not selected_model_identity(unknown_payload).get("portable_cache_reuse", True)
        result = original(*args)
        receipts.append(result[2].verify())
        return result

    monkeypatch.setattr(stage_nodes, "sample_stage", observed)
    original_threads = torch.get_num_threads()
    request.addfinalizer(lambda: torch.set_num_threads(original_threads))
    torch.set_num_threads(1)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *args, **kwargs: None)
    executor = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))
    graph = tiny_graph(accepted)
    executor.execute(deepcopy(graph), "accepted-v2-second-cpu", execute_outputs=["16", "45", "46", "50", "51", "80"])
    assert executor.success, executor.status_messages
    media = list(output_root.rglob("FastH3V2_Accepted_Second_New68_Unaccepted_EXP_*.mp4"))
    assert len(media) == 1
    full_media_hashes = _decoded_delivery_hashes(media[0])
    assert stages == ["low_0_4", "high_4_8"]
    full_high = receipts[-1]
    low_manifest = next(path for path in tmp_path.rglob("manifest.json")
                        if json.loads(path.read_text(encoding="utf-8")).get("stage_context", {}).get("stage") == "low_0_4")
    frozen = tiny_graph(accepted, resume_high=True)
    frozen["60"]["inputs"].update(artifact_path=low_manifest.relative_to(tmp_path).as_posix(),
                                  artifact_sha256=file_sha(low_manifest))
    stages.clear()
    fresh = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                     asset_manager=SimpleNamespace(enabled=False))
    fresh.execute(deepcopy(frozen), "accepted-v2-second-frozen-cpu", execute_outputs=["16", "46", "51", "80"])
    assert fresh.success, fresh.status_messages
    media = sorted(output_root.rglob("FastH3V2_Accepted_Second_New68_Unaccepted_EXP_*.mp4"))
    assert len(media) == 2
    assert _decoded_delivery_hashes(media[-1]) == full_media_hashes
    assert stages == ["high_4_8"]
    assert receipts[-1]["request_sha256"] == full_high["request_sha256"]
    assert receipts[-1]["outputs"] == full_high["outputs"]
    damaged = deepcopy(frozen)
    damaged["60"]["inputs"]["artifact_sha256"] = "0" * 64
    stages.clear()
    rejected = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))
    rejected.execute(damaged, "accepted-v2-second-wrong-sha-cpu", execute_outputs=["46"])
    assert not rejected.success and stages == []

    def fake_identity(_, **_kwargs):
        return {"sha256": "1" * 64}

    monkeypatch.setattr(current_job, "stage_model_identity", fake_identity)
    monkeypatch.setattr(current_job, "_component_identity", fake_identity)
    monkeypatch.setattr(current_job, "_upscale_contract", lambda _report, _geometry: {"test_upscale": "cpu_fake"})
    monkeypatch.setattr(current_attest, "stage_model_identity", fake_identity)
    monkeypatch.setattr(current_attest, "_raw_model_from_stage", fake_identity)
    auditable = tiny_graph(accepted, with_stage_attestation=True)
    stage_run = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                         asset_manager=SimpleNamespace(enabled=False))
    stage_run.execute(deepcopy(auditable), "accepted-v2-second-stage-audit-cpu",
                      execute_outputs=["86", "88"])
    assert stage_run.success, stage_run.status_messages
    assert stages == ["low_0_4", "high_4_8"]
    monkeypatch.setattr(current_origin, "_component_identity", fake_identity)
    provenance = tiny_graph(accepted, with_stage_attestation=True,
                            with_condition_provenance=True)
    proven = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                      asset_manager=SimpleNamespace(enabled=False))
    proven.execute(deepcopy(provenance), "accepted-v2-second-condition-origin-cpu",
                   execute_outputs=["86", "88"])
    assert proven.success, proven.status_messages
    fixed = tiny_graph(accepted, with_stage_attestation=True,
                       with_condition_provenance=True, with_old_fixed_window=True)
    fixed_run = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                         asset_manager=SimpleNamespace(enabled=False))
    fixed_run.execute(deepcopy(fixed), "accepted-v2-second-old-fixed-124-cpu",
                      execute_outputs=["16", "86", "88"])
    assert fixed_run.success, fixed_run.status_messages
    fixed_media = sorted(output_root.rglob("FastH3V2_Accepted_Second_New68_Unaccepted_EXP_*.mp4"))
    assert len(fixed_media) == 3
    assert all(_decoded_delivery_hashes(path) for path in fixed_media)
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
    provenance = tiny_graph(accepted, with_stage_attestation=True,
                            with_condition_provenance=True, with_handoff_provenance=True)
    handoff_run = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                           asset_manager=SimpleNamespace(enabled=False))
    handoff_run.execute(deepcopy(provenance), "accepted-v2-second-handoff-origin-cpu",
                        execute_outputs=["93"])
    assert handoff_run.success, handoff_run.status_messages
    assert len(handoff_calls) == 1
    proof = actual_handoff(**handoff_calls[0])[0].verify()
    assert proof["segment_index"] == 1
    assert proof["accepted_source_sha256"] == handoff_calls[0]["contexts"].source.sha256
    with pytest.raises(ValueError, match="accepted-parent contexts"):
        actual_handoff(**{**handoff_calls[0], "contexts": None})
    old_parent_job = tiny_graph(accepted, with_stage_attestation=True,
        with_condition_provenance=True, with_handoff_provenance=True,
        with_job_binding=True)
    rejected_job = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                            asset_manager=SimpleNamespace(enabled=False))
    rejected_job.execute(deepcopy(old_parent_job), "accepted-v2-different-current-job-cpu",
                         execute_outputs=["95"])
    assert not rejected_job.success
    assert any("another current V2 execution recipe" in str(item)
               for item in rejected_job.status_messages)
    recipe = handoff_calls[0]["current_recipe"]
    recipe_payload = recipe.verify()
    model_id = (recipe_payload["model_pass1"]["sha256"][:16] + ":" +
                recipe_payload["model_pass2"]["sha256"][:16])
    replacement, _ = empty_av_latent(128, 64, 124)
    candidate_context = accepted.descriptor.parent / "context.safetensors"
    candidate_report = delivery._write_context_candidate(
        replacement, candidate_context, "chain", 0, model_id, recipe.sha256)
    accepted.accepted_context = accepted.root / "accepted/context-current.safetensors"
    accepted.accepted_context.write_bytes(candidate_context.read_bytes())
    assert candidate_report["sha256"] == delivery._sha256_file(accepted.accepted_context)
    accepted.info.update(context_sha256=candidate_report["sha256"],
                         model_id=model_id, sampling_summary=recipe.sha256)
    accepted.manifest["segments"][0].update(
        context_sha256=candidate_report["sha256"], model_id=model_id,
        context_path="accepted/context-current.safetensors",
        sampling_summary=recipe.sha256)
    accepted.request["job_sha256"] = recipe.sha256
    accepted.descriptor.write_text(json.dumps(accepted.info), encoding="utf-8")
    accepted.manifest_path.write_text(json.dumps(accepted.manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="another execution contract"):
        handoff_calls[0]["contexts"].verify()
    bound_receipts = []
    original_bind = current_job_nodes.bind_current_job

    def observed_bind(**kwargs):
        result = original_bind(**kwargs)
        bound_receipts.append(result[0])
        return result

    monkeypatch.setattr(current_job_nodes, "bind_current_job", observed_bind)
    matched_parent_job = tiny_graph(accepted, with_stage_attestation=True,
        with_condition_provenance=True, with_handoff_provenance=True,
        with_job_binding=True)
    matched_run = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                           asset_manager=SimpleNamespace(enabled=False))
    matched_run.execute(deepcopy(matched_parent_job), "accepted-v2-matching-current-job-cpu",
                        execute_outputs=["95"])
    assert matched_run.success, matched_run.status_messages
    assert len(bound_receipts) == 1
    binding = bound_receipts[0].verify()
    assert binding["current_recipe_sha256"] == recipe.sha256
    assert binding["parent"]["parent_candidate_id"] == "parent"
    matched_high = receipts[-1]
    before_bundles = set(tmp_path.rglob("current-v2-low-bundle.json"))
    bundled_full = tiny_graph(accepted, with_stage_attestation=True,
        with_condition_provenance=True, with_handoff_provenance=True,
        with_job_binding=True, with_media_provenance=True,
        with_frozen_low_bundle=True)
    bundle_run = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                          asset_manager=SimpleNamespace(enabled=False))
    bundle_run.execute(deepcopy(bundled_full), "accepted-v2-second-current-low-bundle-cpu",
                       execute_outputs=["103"])
    assert bundle_run.success, bundle_run.status_messages
    new_bundles = set(tmp_path.rglob("current-v2-low-bundle.json")) - before_bundles
    assert len(new_bundles) == 1
    selected_manifest = next(iter(new_bundles)).parent / "manifest.json"
    bundled_cold = tiny_graph(accepted, resume_high=True, with_stage_attestation=True,
        with_condition_provenance=True, with_handoff_provenance=True,
        with_job_binding=True, with_media_provenance=True,
        with_frozen_low_bundle=True)
    assert all(key not in bundled_cold for key in ("1", "13", "83", "85", "89"))
    assert bundled_cold["92"]["inputs"]["low_result"] == ["60", 3]
    assert bundled_cold["92"]["inputs"]["low_attestation"] == ["102", 4]
    for key in ("60", "102"):
        bundled_cold[key]["inputs"].update(
            artifact_path=selected_manifest.relative_to(tmp_path).as_posix(),
            artifact_sha256=file_sha(selected_manifest))
    stages.clear()
    cold_run = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                        asset_manager=SimpleNamespace(enabled=False))
    cold_run.execute(deepcopy(bundled_cold), "accepted-v2-second-current-frozen-high-cpu",
                     execute_outputs=["95"])
    assert cold_run.success, cold_run.status_messages
    assert stages == ["high_4_8"]
    assert receipts[-1]["request_sha256"] == matched_high["request_sha256"]
    assert receipts[-1]["outputs"] == matched_high["outputs"]
    monkeypatch.setattr(current_media, "_component_identity", fake_identity)
    media_receipts = []
    original_media = current_job_nodes.prepare_current_media

    def observed_media(**kwargs):
        result = original_media(**kwargs)
        media_receipts.append(result[2])
        return result

    monkeypatch.setattr(current_job_nodes, "prepare_current_media", observed_media)
    media_graph = tiny_graph(accepted, with_stage_attestation=True,
        with_condition_provenance=True, with_handoff_provenance=True,
        with_job_binding=True, with_media_provenance=True)
    media_run = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                         asset_manager=SimpleNamespace(enabled=False))
    media_run.execute(deepcopy(media_graph), "accepted-v2-matching-media-cpu",
                      execute_outputs=["16", "97"])
    assert media_run.success, media_run.status_messages
    assert len(media_receipts) == 1
    assert media_receipts[0].verify()["trim"]["frame_count"] == 68
    native_graph = tiny_graph(accepted, with_stage_attestation=True,
        with_condition_provenance=True, with_handoff_provenance=True,
        with_job_binding=True)
    native_run = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
                                          asset_manager=SimpleNamespace(enabled=False))
    native_run.execute(deepcopy(native_graph), "accepted-v2-matching-native-trim-cpu",
                       execute_outputs=["16"])
    assert native_run.success, native_run.status_messages
    media_files = sorted(output_root.rglob("FastH3V2_Accepted_Second_New68_Unaccepted_EXP_*.mp4"))
    assert len(media_files) == 5
    assert _decoded_delivery_hashes(media_files[-2]) == _decoded_delivery_hashes(media_files[-1])
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(delivery, "_chain_root", lambda _chain: accepted.root)
    path, video, provenance, save_report = current_candidate.save_current_candidate(
        media_receipts[0], "second-source-bound")
    candidate = json.loads(Path(path).read_text(encoding="utf-8"))
    source = json.loads(Path(provenance).read_text(encoding="utf-8"))
    assert candidate["status"] == "candidate" and candidate["frame_count"] == 68
    assert candidate["timeline_start_frame"] == 124
    assert candidate["context_sha256"] == "" and candidate["is_final_segment"] is True
    assert candidate["sampling_summary"] == media_receipts[0].binding.verify()["sampling_summary"]
    assert source["candidate_json_sha256"] == file_sha(Path(path))
    assert source["media_receipt_sha256"] == media_receipts[0].sha256
    assert source["accepted"] is False and json.loads(save_report)["accepted"] is False
    candidate_rgb, candidate_pcm = _decoded_delivery_hashes(Path(video))
    assert candidate_rgb == _decoded_delivery_hashes(media_files[-1])[0] and candidate_pcm
    preview = current_review.review_accept_current_candidate(path, False)
    assert preview[0] == video and preview[1] is False
    assert preview[4:7] == ("", 0, "")
    accepted.manifest["revision"] += 1
    accepted.manifest_path.write_text(json.dumps(accepted.manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="stale manifest revision"):
        current_review.review_accept_current_candidate(path, True)
    with pytest.raises(ValueError):
        bound_receipts[0].verify()
    assert not torch.cuda.is_initialized()
