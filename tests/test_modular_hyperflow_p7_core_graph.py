"""Execute saved P7 split API graphs with tiny inputs in the real Core executor.

Only weight/encoder providers and the learned-3D implementation are doubled.
The public phase, Relay, EAV, sampler, LOW store and HIGH store nodes remain live.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from comfy_api.latest import io
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise
import torch

from h3_audio_t8_pkg import long_video_delivery as delivery
from h3_audio_t8_pkg.modular_sampling import hyperflow_p7 as p7
from h3_audio_t8_pkg.modular_sampling import hyperflow_p7_delivery as candidate
from h3_audio_t8_pkg.modular_sampling import hyperflow_p7_storage as frozen
from h3_audio_t8_pkg.modular_sampling import node_classes
from h3_audio_t8_pkg.modular_sampling import nodes as stage_nodes
from h3_audio_t8_pkg.nodes_prompt_relay_advanced import MiniMaxH3PromptRelayPlanT8Advanced
from helpers import FakeAudioVAE, FakeVideoVAE
from test_modular_hyperflow import inputs as hyperflow_inputs
from test_modular_hyperflow_p7_storage import fake_lift_for
from test_prompt_relay_long_video_advanced import NativeLikeFakeClip
from tools import build_modular_hyperflow_p7_workflow as builder


ROOT = Path(__file__).resolve().parents[1]


class TinyP7Models(io.ComfyNode):
    models = {}

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[io.Combo.Input(
            "phase", options=["low", "high"])], outputs=[io.Model.Output()])

    @classmethod
    def execute(cls, phase):
        return io.NodeOutput(cls.models[phase])


class TinyP7Media(io.ComfyNode):
    clip = NativeLikeFakeClip()
    video_vae = FakeVideoVAE()
    audio_vae = FakeAudioVAE()

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[], outputs=[
            io.Clip.Output(), io.Vae.Output(), io.Vae.Output()])

    @classmethod
    def execute(cls):
        return io.NodeOutput(cls.clip, cls.video_vae, cls.audio_vae)


class P7HighSink(io.ComfyNode):
    seen = []

    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True,
            inputs=[io.Custom("T8_HYPERFLOW_P7_HIGH_RESULT").Input("high_result")],
            outputs=[io.String.Output()])

    @classmethod
    def execute(cls, high_result):
        cls.seen.append(high_result)
        return io.NodeOutput(high_result.verify()["sha256"])


def _candidate(name):
    case = next(case for case in builder.CASES if "_".join(case) == name)
    # Round-trip the actual API payload without depending on private artifacts
    # that are deliberately omitted from the public source package.
    return json.loads(json.dumps(builder.split_graph(*case)))


def _tiny_graph(name, *, chain, resume_path=None, resume_sha=None, parent=None, exact=False):
    graph = _candidate(name)
    continuation = name.startswith("segment1_")
    length = 124 if exact else (22 if continuation else 39)
    end = (192 if continuation else 124) if exact else (56 if continuation else 39)
    if "1" in graph:
        graph["1"] = {"class_type": "TinyP7Models", "inputs": {"phase": "low"}}
    if "50" in graph:
        graph["50"] = {"class_type": "TinyP7Models", "inputs": {"phase": "high"}}
    if "10" in graph:
        graph["10"] = {"class_type": "TinyP7Models", "inputs": {"phase": "low"}}
    graph["11"] = {"class_type": "TinyP7Models", "inputs": {"phase": "high"}}
    graph["6"] = {"class_type": "TinyP7Media", "inputs": {}}
    graph["7"] = {"class_type": "TinyP7Media", "inputs": {}}
    graph["8"] = {"class_type": "TinyP7Media", "inputs": {}}
    graph["12"]["inputs"].update(clip=["6", 0], video_vae=["7", 1],
        audio_vae=["8", 2], prompt="LOW scene.", length=length)
    graph["20"]["inputs"].update(clip=["6", 0], video_vae=["7", 1],
        audio_vae=["8", 2], prompt="HIGH scene.", length=length)
    graph["28"]["inputs"].update(video_vae=["7", 1], audio_vae=["8", 2],
        candidate_id="tiny_second" if continuation else "tiny_source",
        final_frame_count=(68 if exact else 17) if continuation else 0, color_match=False)
    graph["2"]["inputs"].update(chain_id=chain, context_frames=22 if exact else 5,
        width=128, height=64, low_width=64, low_height=32)
    if continuation:
        if parent is None:
            raise ValueError("The tiny continuation needs a real accepted parent receipt")
        graph["2"]["inputs"].update(parent_candidate_id=parent[0],
            parent_revision=parent[1], previous_job_sha256=parent[2])
        graph["3"]["inputs"]["video_vae"] = ["7", 1]
    for phase, plan, project, condition in (("LOW", "30", "31", "12"),
                                           ("HIGH", "40", "41", "20")):
        if plan in graph:
            graph[plan]["inputs"].update(global_prompt=f"{phase} scene.",
                local_prompts="Walk.\nTurn.\nLook.\nPause.", length=193 if exact else 57)
            graph[project]["inputs"].update(length=length, accepted_end_frame=end)
            graph[condition]["inputs"]["prompt"] = [project, 1]
    for relay in ("32", "42"):
        if relay in graph:
            graph[relay]["inputs"]["clip"] = ["6", 0]
    if continuation or exact:
        for config in ("33", "43"):
            if config in graph:
                graph[config]["inputs"]["g_hard_limit"] = 3.0
    if resume_path is not None:
        graph["19"]["inputs"].update(artifact_path=resume_path,
                                     artifact_sha256=resume_sha)
    graph["100"] = {"class_type": "P7HighSink", "inputs": {
        "high_result": ["27", 0]}}
    return graph


def test_saved_p7_api_low_receipt_high_only_core_cache_cancel_and_bad_receipt(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT.parent.parent))
    import execution
    import nodes
    import comfy.model_management

    output = tmp_path / "output"
    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(output))
    upscaler = tmp_path / "upscaler.safetensors"
    upscaler.write_bytes(b"P7 saved-graph tiny learned lift")
    monkeypatch.setattr(p7.folder_paths, "get_full_path_or_raise",
                        lambda _kind, _name: str(upscaler))
    monkeypatch.setattr(p7, "learned_upscale_h3_av_latent", fake_lift_for(upscaler))
    low, high, _, _ = hyperflow_inputs(monkeypatch, distinct=True)
    TinyP7Models.models = {"low": low, "high": high}
    P7HighSink.seen.clear()
    for cls in (*node_classes(), MiniMaxH3PromptRelayPlanT8Advanced,
                BasicGuider, RandomNoise,
                TinyP7Models, TinyP7Media, P7HighSink):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)

    original_sample = stage_nodes.sample_stage
    calls = []
    cancel = {"high": False}

    def counted_sample(noise, guider, sampler, sigmas, source, context):
        phase = "high" if context.start == 4 else "low"
        calls.append(phase)
        if phase == "high" and cancel["high"]:
            raise comfy.model_management.InterruptProcessingException()
        return original_sample(noise, guider, sampler, sigmas, source, context)

    monkeypatch.setattr(stage_nodes, "sample_stage", counted_sample)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *_args, **_kwargs: None)

    def executor():
        return execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
            asset_manager=SimpleNamespace(enabled=False))

    def run(current_executor, graph, name, *, success=True):
        current_executor.execute(deepcopy(graph), name, execute_outputs=["100"])
        assert current_executor.success is success, current_executor.status_messages

    chain = "p7_core_graph_" + tmp_path.name.replace("-", "_")
    save = _tiny_graph("segment0_combined_both_save", chain=chain)
    first_executor = executor()
    run(first_executor, save, "low_high_save")
    assert calls == ["low", "high"]
    initial = P7HighSink.seen[-1]
    assert initial.sampled.verify()["execution"]["hyperflow_fresh"]["absolute_apply_intervals"] == [4, 5, 6, 7]
    root = frozen.stage_root(initial.handoff.lift.low.phase)
    low_manifest = next(path for path in root.rglob("manifest.json")
                        if json.loads(path.read_text(encoding="utf-8"))["stage_context"]["stage"]
                        == "hyperflow_low_partial4")
    path = low_manifest.relative_to(root).as_posix()
    digest = hashlib.sha256(low_manifest.read_bytes()).hexdigest()

    resumed = _tiny_graph("segment0_combined_both_resume_high", chain=chain,
                          resume_path=path, resume_sha=digest)
    assert not {"1", "10", "14", "15", "16", "17", "18", "32", "33", "34"} & set(resumed)
    assert {"30", "31"} <= set(resumed)
    calls.clear()
    second_executor = executor()
    run(second_executor, resumed, "high_only")
    assert calls == ["high"]
    restored = P7HighSink.seen[-1]
    for left, right in zip(initial.sampled.output["samples"].unbind(),
                           restored.sampled.output["samples"].unbind()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    calls.clear()
    run(second_executor, resumed, "high_only_unchanged")
    assert calls == [], set(resumed) - set(second_executor.status_messages[1][1]["nodes"])

    calls.clear()
    resumed["43"]["inputs"]["mode"] = "apply_exp"
    cancel["high"] = True
    run(second_executor, resumed, "high_only_cancel", success=False)
    assert calls == ["high"]
    assert any(event == "execution_interrupted" for event, _ in second_executor.status_messages)
    calls.clear()
    cancel["high"] = False
    run(second_executor, resumed, "high_only_retry")
    assert calls == ["high"]
    observed_eav = P7HighSink.seen[-1].sampled.verify()["execution"]["hyperflow_fresh"]["eav"]
    assert observed_eav["config"]["mode"] == "apply_exp"
    assert observed_eav["status"] == "observed_apply_exp"
    calls.clear()
    run(second_executor, resumed, "high_only_retry_unchanged")
    assert calls == []

    invalid = deepcopy(resumed)
    invalid["19"]["inputs"]["artifact_sha256"] = "0" * 64
    calls.clear()
    run(executor(), invalid, "high_only_bad_low_sha", success=False)
    assert calls == []

    invalid = deepcopy(resumed)
    invalid["30"]["inputs"]["global_prompt"] = "Different frozen LOW scene."
    run(executor(), invalid, "high_only_changed_low_relay", success=False)
    assert calls == []

    invalid = deepcopy(resumed)
    invalid["19"]["inputs"]["artifact_path"] = "../" + path
    run(executor(), invalid, "high_only_escape_path", success=False)
    assert calls == []

    first_executor.execute(deepcopy(save), "candidate_preview", execute_outputs=["28"])
    assert first_executor.success, first_executor.status_messages
    candidate_path = delivery.long_video_chain_root(chain) / "candidates/segment_00000/tiny_source/candidate.json"
    assert candidate_path.is_file()
    checked, audit, movie = candidate.verify_candidate(candidate_path, initial.verify()["sha256"])
    assert checked["frame_count"] == 39 and Path(movie).is_file()
    assert audit["sampling_plan"]["hyperflow"]["intervals"] == [[0, 4], [4, 8]]
    assert not (delivery.long_video_chain_root(chain) / delivery.MANIFEST_NAME).exists()

    # A changed state file must invalidate an otherwise cached LOW Load and
    # reject before any HIGH forward, even with identical graph inputs.
    state = low_manifest.with_name("state.safetensors")
    with state.open("ab") as handle:
        handle.write(b"corrupt")
    calls.clear()
    run(second_executor, resumed, "high_only_changed_frozen_file", success=False)
    assert calls == []
    assert not torch.cuda.is_initialized()


def test_p7_two_saved_api_segments_accept_parent_and_resume_only_second_high(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT.parent.parent))
    import execution
    import nodes

    output = tmp_path / "output"
    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(output))
    upscaler = tmp_path / "upscaler.safetensors"
    upscaler.write_bytes(b"P7 two saved API graphs tiny learned lift")
    monkeypatch.setattr(p7.folder_paths, "get_full_path_or_raise",
                        lambda _kind, _name: str(upscaler))
    monkeypatch.setattr(p7, "learned_upscale_h3_av_latent", fake_lift_for(upscaler))
    low, high, _, _ = hyperflow_inputs(monkeypatch, distinct=True)
    TinyP7Models.models = {"low": low, "high": high}
    P7HighSink.seen.clear()
    for cls in (*node_classes(), MiniMaxH3PromptRelayPlanT8Advanced,
                BasicGuider, RandomNoise, TinyP7Models, TinyP7Media, P7HighSink):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
    original_sample = stage_nodes.sample_stage
    calls = []

    def counted_sample(noise, guider, sampler, sigmas, source, context):
        calls.append("high" if context.start == 4 else "low")
        return original_sample(noise, guider, sampler, sigmas, source, context)

    monkeypatch.setattr(stage_nodes, "sample_stage", counted_sample)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *_args, **_kwargs: None)

    def new_executor():
        return execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
            asset_manager=SimpleNamespace(enabled=False))

    def run(current, graph, name, outputs=("100",), *, success=True):
        current.execute(deepcopy(graph), name, execute_outputs=list(outputs))
        assert current.success is success, current.status_messages

    chain = "p7_saved_api_two_segments_" + tmp_path.name.replace("-", "_")
    chain_root = delivery.long_video_chain_root(chain)
    first = _tiny_graph("segment0_combined_both_save", chain=chain)
    first_executor = new_executor()
    run(first_executor, first, "segment0_stages")
    assert calls == ["low", "high"]
    first_job = P7HighSink.seen[-1].verify()["sha256"]
    calls.clear()
    run(first_executor, first, "segment0_candidate", outputs=("28",))
    assert calls == []
    first_candidate = chain_root / "candidates/segment_00000/tiny_source/candidate.json"
    first_descriptor, _, _ = candidate.verify_candidate(first_candidate, first_job)
    assert first_descriptor["timeline_end_frame"] == 39
    _, accepted, _, parent_id, revision, _ = candidate.accept_candidate(
        first_candidate, first_job, accept=True)
    assert accepted and parent_id == "tiny_source" and revision == 1
    parent = (parent_id, revision, first_job)

    second = _tiny_graph("segment1_combined_both_save", chain=chain, parent=parent)
    second_executor = new_executor()
    run(second_executor, second, "segment1_stages")
    assert calls == ["low", "high"]
    second_result = P7HighSink.seen[-1]
    second_job = second_result.verify()["sha256"]
    assert second_result.sampled.verify()["execution"]["hyperflow_fresh"]["absolute_apply_intervals"] == [4, 5, 6, 7]
    stage_root = frozen.stage_root(second_result.handoff.lift.low.phase)
    low_manifest = next(path for path in stage_root.rglob("manifest.json")
                        if json.loads(path.read_text(encoding="utf-8"))["stage_context"]["stage"]
                        == "hyperflow_low_partial4")
    relative = low_manifest.relative_to(stage_root).as_posix()
    digest = hashlib.sha256(low_manifest.read_bytes()).hexdigest()
    resumed = _tiny_graph("segment1_combined_both_resume_high", chain=chain,
        parent=parent, resume_path=relative, resume_sha=digest)
    assert not {"1", "10", "14", "15", "16", "17", "18", "32", "33", "34"} & set(resumed)
    calls.clear()
    run(new_executor(), resumed, "segment1_high_only")
    assert calls == ["high"]
    restored = P7HighSink.seen[-1]
    for left, right in zip(second_result.sampled.output["samples"].unbind(),
                           restored.sampled.output["samples"].unbind()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)

    invalid = deepcopy(resumed)
    invalid["2"]["inputs"]["previous_job_sha256"] = "0" * 64
    calls.clear()
    run(new_executor(), invalid, "segment1_wrong_parent", success=False)
    assert calls == []

    calls.clear()
    run(second_executor, second, "segment1_candidate", outputs=("28",))
    assert calls == []
    second_candidate = chain_root / "candidates/segment_00001/tiny_second/candidate.json"
    descriptor, audit, _ = candidate.verify_candidate(second_candidate, second_job)
    assert descriptor["timeline_start_frame"] == 39
    assert descriptor["frame_count"] == 17 and descriptor["timeline_end_frame"] == 56
    assert audit["trim"]["frame_count"] == 17
    assert candidate.accept_candidate(second_candidate, second_job, accept=True)[1]
    movie, compose_json = delivery.compose_accepted_long_video(
        chain, "p7_saved_api_tiny", True, "cosine_bridge", 5.0, 28)
    assert Path(movie).is_file() and json.loads(compose_json)["frame_count"] == 56
    assert not torch.cuda.is_initialized()


def test_p7_saved_api_exact_124_plus_68_window_with_external_effects(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT.parent.parent))
    import execution
    import nodes

    output = tmp_path / "output"
    monkeypatch.setattr(delivery.folder_paths, "get_output_directory", lambda: str(output))
    upscaler = tmp_path / "upscaler.safetensors"
    upscaler.write_bytes(b"P7 exact saved API tiny learned lift")
    monkeypatch.setattr(p7.folder_paths, "get_full_path_or_raise",
                        lambda _kind, _name: str(upscaler))
    monkeypatch.setattr(p7, "learned_upscale_h3_av_latent", fake_lift_for(upscaler))
    low, high, _, _ = hyperflow_inputs(monkeypatch, distinct=True)
    TinyP7Models.models = {"low": low, "high": high}
    P7HighSink.seen.clear()
    for cls in (*node_classes(), MiniMaxH3PromptRelayPlanT8Advanced,
                BasicGuider, RandomNoise, TinyP7Models, TinyP7Media, P7HighSink):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={},
                             send_sync=lambda *_args, **_kwargs: None)

    def executor():
        return execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.},
            asset_manager=SimpleNamespace(enabled=False))

    def run(current, graph, name, outputs=("100",)):
        current.execute(deepcopy(graph), name, execute_outputs=list(outputs))
        assert current.success, current.status_messages

    chain = "p7_saved_api_exact_" + tmp_path.name.replace("-", "_")
    chain_root = delivery.long_video_chain_root(chain)
    first = _tiny_graph("segment0_combined_both_save", chain=chain, exact=True)
    first_executor = executor()
    run(first_executor, first, "exact_segment0_stages")
    first_result = P7HighSink.seen[-1]
    first_job = first_result.verify()["sha256"]
    for sampled in (first_result.handoff.lift.low.sampled, first_result.sampled):
        effects = sampled.verify()["execution"]["hyperflow_fresh"]
        assert effects["eav"]["selector_calls"] == 200
        assert effects["relay"]["routed_attention_calls"] == 200
    run(first_executor, first, "exact_segment0_candidate", outputs=("28",))
    first_candidate = chain_root / "candidates/segment_00000/tiny_source/candidate.json"
    first_descriptor, _, _ = candidate.verify_candidate(first_candidate, first_job)
    assert first_descriptor["frame_count"] == 124
    assert first_descriptor["timeline_end_frame"] == 124
    _, accepted, _, parent_id, revision, _ = candidate.accept_candidate(
        first_candidate, first_job, accept=True)
    assert accepted

    second = _tiny_graph("segment1_combined_both_save", chain=chain,
        parent=(parent_id, revision, first_job), exact=True)
    second_executor = executor()
    run(second_executor, second, "exact_segment1_stages")
    second_result = P7HighSink.seen[-1]
    second_job = second_result.verify()["sha256"]
    for sampled in (second_result.handoff.lift.low.sampled, second_result.sampled):
        effects = sampled.verify()["execution"]["hyperflow_fresh"]
        assert effects["eav"]["selector_calls"] == 200
        assert effects["relay"]["routed_attention_calls"] == 200
    stage_root = frozen.stage_root(second_result.handoff.lift.low.phase)
    low_manifest = next(path for path in stage_root.rglob("manifest.json")
                        if json.loads(path.read_text(encoding="utf-8"))["stage_context"]["stage"]
                        == "hyperflow_low_partial4")
    relative = low_manifest.relative_to(stage_root).as_posix()
    digest = hashlib.sha256(low_manifest.read_bytes()).hexdigest()
    resumed = _tiny_graph("segment1_combined_both_resume_high", chain=chain,
        parent=(parent_id, revision, first_job), resume_path=relative,
        resume_sha=digest, exact=True)
    assert not {"1", "10", "14", "15", "16", "17", "18", "32", "33", "34"} & set(resumed)
    run(executor(), resumed, "exact_segment1_high_only")
    restored = P7HighSink.seen[-1]
    for original, recovered in zip(second_result.sampled.output["samples"].unbind(),
                                   restored.sampled.output["samples"].unbind()):
        torch.testing.assert_close(original, recovered, rtol=0, atol=0)
    restored_effects = restored.sampled.verify()["execution"]["hyperflow_fresh"]
    assert restored_effects["eav"]["selector_calls"] == 200
    assert restored_effects["relay"]["routed_attention_calls"] == 200
    run(second_executor, second, "exact_segment1_candidate", outputs=("28",))
    second_candidate = chain_root / "candidates/segment_00001/tiny_second/candidate.json"
    second_descriptor, audit, _ = candidate.verify_candidate(second_candidate, second_job)
    assert second_descriptor["timeline_start_frame"] == 124
    assert second_descriptor["frame_count"] == 68
    assert second_descriptor["timeline_end_frame"] == 192
    assert audit["trim"]["frame_count"] == 68
    assert candidate.accept_candidate(second_candidate, second_job, accept=True)[1]
    movie, report = delivery.compose_accepted_long_video(
        chain, "p7_saved_api_exact_tiny", True, "cosine_bridge", 5.0, 28)
    assert Path(movie).is_file() and json.loads(report)["frame_count"] == 192
    assert not torch.cuda.is_initialized()
