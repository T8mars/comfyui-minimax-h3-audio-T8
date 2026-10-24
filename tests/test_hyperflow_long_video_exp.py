"""CPU contracts for the isolated, unqualified HyperFlow long-film route."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from comfy.nested_tensor import NestedTensor

import h3_audio_t8_pkg
from h3_audio_t8_pkg.hyperflow_long_video_exp import nodes, runner
from h3_audio_t8_pkg.hyperflow_long_video_exp import single8, single8_node
from h3_audio_t8_pkg.long_video_dual_model_runner import DualModelSegmentRunner
from h3_audio_t8_pkg.long_video_in_node_loop_effects_advanced import _write_effects_audit
from tools.build_hyperflow_long_video_workflow import build_prompt
from tools.build_hyperflow_single8_long_video_workflow import build_prompt as build_single8_prompt
from h3_audio_t8_pkg.long_video_orchestration import build_long_video_chain_plan


ROOT = Path(__file__).resolve().parents[1]
def _legacy_source_digest():
    entries = [f"{path.name}:{hashlib.sha256(path.read_bytes()).hexdigest()}"
               for path in sorted((ROOT / "h3_t8").glob("*.py"), key=lambda item: item.name)]
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def test_isolated_registration_tracks_current_legacy_source_scan_and_preserves_schema_prefix():
    # HR1 media-contract hardening changed director_batch.py, which is in the
    # legacy glob identity. Old chain receipts must fail closed rather than
    # silently reuse a new source snapshot under the same chain_id.
    source_before = _legacy_source_digest()
    assert not list((ROOT / "h3_t8").glob("*hyperflow_long_video*.py"))
    from h3_audio_t8_pkg import nodes as legacy_nodes
    old = asyncio.run(legacy_nodes.comfy_entrypoint().get_node_list())
    new = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    from h3_audio_t8_pkg.modular_sampling import node_classes
    assert new[:len(old)] == old
    assert new[len(old)] is nodes.MiniMaxH3HyperFlowLongVideoEXPT8
    assert new[len(old) + 1] is single8_node.MiniMaxH3HyperFlowSingle8LongVideoEXPT8
    assert new[len(old) + 2:] == node_classes()
    assert [cls.define_schema().node_id for cls in new[:len(old)]] == [
        cls.define_schema().node_id for cls in old]
    assert _legacy_source_digest() == source_before


def test_schema_has_only_supported_mode_and_append_only_output_contract():
    schema = nodes.MiniMaxH3HyperFlowLongVideoEXPT8.define_schema()
    ids = [item.id for item in schema.inputs]
    assert schema.node_id == "MiniMaxH3HyperFlowLongVideoEXPT8"
    assert {"model_pass1", "model_pass2", "hyperflow_file", "upscaler_model"} <= set(ids)
    assert not nodes._OMIT.intersection(ids)
    assert [item.id for item in schema.outputs] == [item.id for item in
        nodes.MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced.define_schema().outputs]
    values = {item.id: item.default for item in schema.inputs if item.id in {
        "chain_id", "total_duration_seconds", "width", "height", "context_frames", "render_window_frames"}}
    assert values == {"chain_id": "h3_hyperflow_long_video_exp", "total_duration_seconds": 8.0,
                      "width": 896, "height": 448, "context_frames": 22, "render_window_frames": 124}


def test_core_validates_own_hyperflow_long_video_api_graph(monkeypatch):
    previous_paths = list(sys.path)
    try:
        sys.path.insert(0, str(ROOT.parents[1]))
        import execution
        import nodes as core_nodes
    finally:
        sys.path[:] = previous_paths

    info = {nodes.MiniMaxH3HyperFlowLongVideoEXPT8.define_schema().node_id:
            nodes.MiniMaxH3HyperFlowLongVideoEXPT8.GET_NODE_INFO_V1()}
    graph = build_prompt(info, chain_id="hf_cpu_validation_not_queued")
    from h3_audio_t8_pkg.nodes_h3_lora_compat_advanced import MiniMaxH3LoRACompatibilityLoaderT8Advanced
    monkeypatch.setitem(core_nodes.NODE_CLASS_MAPPINGS,
                        "MiniMaxH3LoRACompatibilityLoaderT8Advanced",
                        MiniMaxH3LoRACompatibilityLoaderT8Advanced)
    monkeypatch.setitem(core_nodes.NODE_CLASS_MAPPINGS,
                        nodes.MiniMaxH3HyperFlowLongVideoEXPT8.define_schema().node_id,
                        nodes.MiniMaxH3HyperFlowLongVideoEXPT8)
    result = asyncio.run(execution.validate_prompt("hyperflow-long-video-cpu", graph, None))
    assert result[0], result
    assert result[2] == ["8"]
    assert graph["8"]["inputs"]["hyperflow_file"].startswith("hyperflow/")
    assert graph["8"]["inputs"]["total_duration_seconds"] == 8.0


def test_builder_keeps_default_geometry_and_allows_exact_0_6mp_grid():
    info = {nodes.MiniMaxH3HyperFlowLongVideoEXPT8.define_schema().node_id:
            nodes.MiniMaxH3HyperFlowLongVideoEXPT8.GET_NODE_INFO_V1()}
    old = build_prompt(info, chain_id="hf_default_geometry")["8"]["inputs"]
    assert (old["width"], old["height"], old["low_width"], old["low_height"]) == (
        896, 448, 448, 224)
    larger = build_prompt(
        info, chain_id="hf_0_6mp_geometry", width=1024, height=576,
        low_width=512, low_height=288)["8"]["inputs"]
    assert (larger["width"], larger["height"], larger["low_width"], larger["low_height"]) == (
        1024, 576, 512, 288)
    assert larger["width"] * larger["height"] == 589824


def test_saved_frontend_examples_are_editable_and_route_specific():
    root = ROOT / "examples" / "workflows" / "04-long-video"
    cases = (
        ("2026-09-22_H3_HyperFlow_P7_Dual_0p6MP_8s_EXP.json",
         "MiniMaxH3HyperFlowLongVideoEXPT8", 2),
        ("2026-09-22_H3_HyperFlow_Native_Single8_0p6MP_8s_EXP.json",
         "MiniMaxH3HyperFlowSingle8LongVideoEXPT8", 1),
    )
    for filename, output_type, lora_count in cases:
        workflow = json.loads((root / filename).read_text(encoding="utf-8"))
        nodes_by_id = {node["id"]: node for node in workflow["nodes"]}
        assert len(nodes_by_id) == len(workflow["nodes"])
        assert sum(node["type"] == output_type for node in workflow["nodes"]) == 1
        assert sum(node["type"] == "MiniMaxH3LoRACompatibilityLoaderT8Advanced"
                   for node in workflow["nodes"]) == lora_count
        assert workflow["extra"]["t8_hyperflow_status"].startswith("experimental_")
        for link_id, source_id, source_slot, target_id, target_slot, *_ in workflow["links"]:
            assert nodes_by_id[source_id]["outputs"][source_slot]["links"] is not None
            assert link_id in nodes_by_id[source_id]["outputs"][source_slot]["links"]
            assert nodes_by_id[target_id]["inputs"][target_slot]["link"] == link_id


def test_single8_schema_and_core_graph_are_native_full_size(monkeypatch):
    cls = single8_node.MiniMaxH3HyperFlowSingle8LongVideoEXPT8
    schema = cls.define_schema()
    ids = {item.id for item in schema.inputs}
    assert {"model", "hyperflow_file", "color_match"} <= ids
    assert {"model_pass1", "model_pass2", "low_width", "upscaler_model"}.isdisjoint(ids)
    info = {schema.node_id: cls.GET_NODE_INFO_V1()}
    graph = build_single8_prompt(info, chain_id="hf_single8_cpu_graph", prompt="Bright living room")
    assert graph["8"]["inputs"]["width"] * graph["8"]["inputs"]["height"] == 589824
    assert graph["8"]["inputs"]["global_prompt"] == "Bright living room"
    previous_paths = list(sys.path)
    try:
        sys.path.insert(0, str(ROOT.parents[1]))
        import execution
        import nodes as core_nodes
    finally:
        sys.path[:] = previous_paths
    from h3_audio_t8_pkg.nodes_h3_lora_compat_advanced import MiniMaxH3LoRACompatibilityLoaderT8Advanced
    monkeypatch.setitem(core_nodes.NODE_CLASS_MAPPINGS,
                        "MiniMaxH3LoRACompatibilityLoaderT8Advanced",
                        MiniMaxH3LoRACompatibilityLoaderT8Advanced)
    monkeypatch.setitem(core_nodes.NODE_CLASS_MAPPINGS, schema.node_id, cls)
    result = asyncio.run(execution.validate_prompt("hf-single8-cpu", graph, None))
    assert result[0], result
    assert result[2] == ["8"]


def test_single8_runner_uses_full_typed_interval_not_upscaler(monkeypatch, tmp_path):
    observed = []
    class Model:
        def clone(self):
            return self
    model = Model()
    contract = {"hyperflow": {"source_sha256": "h" * 64}}
    runner8 = single8.HyperFlowSingle8SegmentRunner(
        model, contract=contract, width=1024, height=576)
    monkeypatch.setattr(runner8, "_conditions", lambda _model, _context, _inputs, _plan:
        (model, "positive", "latent", None, "prompt", "{}", {"status": "disabled"}))
    monkeypatch.setattr(single8, "release_stage_residency", lambda *_args: {})
    monkeypatch.setattr(single8, "build_hyperflow_plan", lambda _model, start, stop:
        (observed.append((start, stop)) or SimpleNamespace(as_report=lambda: {"interval": [start, stop]})))
    monkeypatch.setattr(single8, "setup_hyperflow_sampler", lambda *_args:
        (model, "sampler", torch.linspace(1, 0, 9)))
    monkeypatch.setattr(single8, "sample_model_stage", lambda *_args, **kwargs:
        (observed.append(kwargs["output_kind"]) or ({"samples": "av"}, {"completed_forwards": 8})))
    def cache(_root):
        def load(stage, _contract):
            observed.append(("load", stage))
            return None
        def save(stage, _contract, _output, _report):
            observed.append(("save", stage))
        return SimpleNamespace(load=load, save=save)
    monkeypatch.setattr(single8, "AVStageCache", cache)
    segment = SimpleNamespace(index=0, seed=123)
    result = runner8.run(
        root=tmp_path, chain_id="hf-single8", job_sha256="j" * 64,
        segment=segment, candidate_id="candidate", base_candidate_id="base",
        high_context=None, parent_candidate_id=None, parent_revision=0,
        projected_plan=None, inputs={"task_type": "T2VA", "audio_mode": "native",
            "clip": object(), "video_vae": object(), "audio_vae": object(),
            "width": 1024, "height": 576})
    assert observed == [("load", "high_output"), (0, 8),
                        "zero_sigma_output", ("save", "high_output")]
    assert result["sampling_report"]["hyperflow"]["learned_upscaler"] == "not_used"
    assert result["sampling_report"]["hyperflow"]["nfe"] == 8


def test_qualification_film_is_total_eight_seconds_two_serial_segments():
    segments = build_long_video_chain_plan("hf_new_chain", 8.0, 124, 22)
    assert len(segments) == 2
    assert sum(segment.plan.final_frame_count for segment in segments) == 192
    assert segments[0].plan.final_frame_count == 124
    assert segments[1].plan.timeline_start_seconds == pytest.approx(124 / 24)
    assert segments[1].plan.context_frames == 22


def test_node_execute_freezes_hf_identity_and_submits_only_fixed_exp_loop(monkeypatch, tmp_path):
    shared = object()
    class Bare:
        model = shared
        def get_attachment(self, _key):
            return None
    low, high = Bare(), Bare()
    tiny = tmp_path / "weight.safetensors"
    tiny.write_bytes(b"identity-only")
    monkeypatch.setattr(nodes, "require_safe_dual_branch_core", lambda: None)
    monkeypatch.setattr(nodes, "learned_upscale_geometry", lambda *_args: {
        "output_width": 896, "output_height": 448})
    monkeypatch.setattr(nodes, "_resolve", lambda _name: tiny)
    monkeypatch.setattr(nodes.folder_paths, "get_full_path_or_raise", lambda _category, _name: str(tiny))
    monkeypatch.setattr(nodes, "stage_model_identity", lambda model: {
        "sha256": "1" * 64 if model is low else "2" * 64})
    monkeypatch.setattr(nodes, "load_hyperflow_original", lambda _path: SimpleNamespace(
        source_sha256="h" * 64, metadata=SimpleNamespace(raw={"hyperflow": "true"})))
    def install(model, weights):
        binding = SimpleNamespace(sha256=weights.source_sha256,
                                  raw_sigmas=(1.0, 0.0), model_identity=id(shared))
        return model, binding, {"status": "installed", "source_path": str(tiny)}
    monkeypatch.setattr(nodes, "install_hyperflow", install)
    monkeypatch.setattr(nodes, "_sha256_file", lambda _path: "f" * 64)
    monkeypatch.setattr(nodes, "_component_identity", lambda _component: {"sha256": "c" * 64})
    monkeypatch.setattr(nodes, "_preview_video", lambda _path: ("preview", {}))
    submitted = {}
    def loop(*args, **kwargs):
        submitted.update(kwargs)
        return "video.mp4", "manifest.json", 2, "complete", "{}"
    monkeypatch.setattr(nodes, "run_long_video_in_node_loop_effects", loop)
    info = nodes.MiniMaxH3HyperFlowLongVideoEXPT8.GET_NODE_INFO_V1()
    graph = build_prompt({nodes.MiniMaxH3HyperFlowLongVideoEXPT8.define_schema().node_id: info},
                         chain_id="hf_cpu_execute_contract")
    values = dict(graph["8"]["inputs"])
    for key in ("model_pass1", "model_pass2", "clip", "video_vae", "audio_vae"):
        values[key] = {"model_pass1": low, "model_pass2": high}.get(key, object())
    result = nodes.MiniMaxH3HyperFlowLongVideoEXPT8.execute(**values).result
    assert result[1:5] == ("video.mp4", "manifest.json", 2, "complete")
    assert submitted["steps"] == 8 and submitted["task_type"] == "T2VA"
    assert submitted["audio_mode"] == "native" and submitted["prompt_relay_mode"] == "disabled"
    assert submitted["eav_mode"] == "disabled" and submitted["render_window_frames"] == 124
    assert submitted["_stage_runner"].contract["hyperflow"]["source_sha256"] == "h" * 64
    assert submitted["_stage_runner"].contract["stage_cache_namespace"] == "hyperflow_stages"
    assert submitted["_stage_runner"].models == (low, high)
    original_contract = submitted["_stage_runner"].contract
    assert original_contract["settings"]["color_match"] is True
    assert original_contract["settings"]["low_width"] == 448
    for field, changed in (("color_match", False), ("audio_seam_policy", "none"),
                           ("bridge_ms", 11), ("bit_depth", 10),
                           ("crf", 23), ("filename_prefix", "other-name")):
        changed_values = {**values, field: changed}
        nodes.MiniMaxH3HyperFlowLongVideoEXPT8.execute(**changed_values)
        next_contract = submitted["_stage_runner"].contract
        assert next_contract != original_contract
        assert next_contract["settings"][field] != original_contract["settings"][field]


def test_node_direct_call_rejects_hidden_external_media_before_identity(monkeypatch):
    monkeypatch.setattr(nodes, "require_safe_dual_branch_core", lambda: None)
    with pytest.raises(ValueError, match="unsupported inputs.*drive_audio"):
        nodes.MiniMaxH3HyperFlowLongVideoEXPT8.execute(
            object(), object(), "unused", 448, 224, "unused",
            drive_audio=object())


def test_typed_head_tail_setup_never_uses_legacy_sampling(monkeypatch):
    seen = []
    def setup(model, latent, plan, **flags):
        seen.append((plan, flags))
        return model, "sampler", torch.linspace(1, 0, 5)
    monkeypatch.setattr(runner, "setup_hyperflow_sampler", setup)
    monkeypatch.setattr(runner, "build_hyperflow_plan", lambda model, start, stop:
        SimpleNamespace(start_interval=start, stop_interval=stop, as_report=lambda: {"interval": [start, stop]}))
    assert runner.HyperFlowLongVideoSegmentRunner._setup("low", {}, True)[3] == {"interval": [0, 4]}
    assert runner.HyperFlowLongVideoSegmentRunner._setup("high", {}, False)[3] == {"interval": [4, 8]}
    assert seen[0][1] == {"internal_continuation": True, "new_noise_restart": False}
    assert seen[1][1] == {"internal_continuation": True, "new_noise_restart": True}


def _av(width, height, value=0.0):
    video = torch.full((1, 24, 12, height // 16, width // 16), value)
    audio = torch.full((1, 32, 2, 100), value)
    return {"samples": NestedTensor((video, audio))}


@pytest.fixture
def rig(monkeypatch, tmp_path):
    calls = []
    low, high = object(), object()
    failure = {"high": False}
    def condition(self, model, context, inputs, projected):
        calls.append(("condition", model, context, inputs["width"]))
        return model, [], _av(inputs["width"], inputs["height"]), None, "prompt", "{}", {"status": "disabled"}
    monkeypatch.setattr(DualModelSegmentRunner, "_conditions", condition)
    monkeypatch.setattr(DualModelSegmentRunner, "_reconcile", lambda self, enlarged, template, positive:
        (enlarged, positive, "{}"))
    monkeypatch.setattr(runner, "release_stage_residency", lambda *parts: {"released": True})
    monkeypatch.setattr(runner.HyperFlowLongVideoSegmentRunner, "_setup", staticmethod(
        lambda model, latent, first: (model, object(), torch.linspace(1, 0, 5),
                                     {"absolute_interval": [0, 4] if first else [4, 8]})))
    def sample(model, positive, latent, **kwargs):
        is_low = model is low
        calls.append(("sample", model, kwargs["output_kind"], kwargs["seed"]))
        if not is_low and failure["high"]:
            raise RuntimeError("forced high failure")
        video, audio = latent["samples"].unbind()
        result = {**latent, "samples": NestedTensor((video + 2, audio + (1 if is_low else 7)))}
        return result, {"nfe": 4}
    monkeypatch.setattr(runner, "sample_model_stage", sample)
    def upscale(latent, _name, _mode, _scale, _mp, width, height, *_rest):
        calls.append(("upscale", width, height))
        output = _av(width, height, 2.0)
        output["samples"] = NestedTensor((output["samples"].unbind()[0], latent["samples"].unbind()[1]))
        return output, width, height, "{}"
    monkeypatch.setattr(runner, "learned_upscale_h3_av_latent", upscale)
    contract = {"first_model": {"sha256": "low-model"}, "hyperflow": {"source_sha256": "h" * 64}}
    engine = runner.HyperFlowLongVideoSegmentRunner(
        low, high, contract=contract, low_width=64, low_height=32, upscaler_model="upscaler")
    inputs = {"task_type": "T2VA", "audio_mode": "native", "width": 128, "height": 64,
              "clip": object(), "video_vae": object(), "audio_vae": object()}
    def run(index=0, parent="", context=None, candidate=None):
        candidate = candidate or f"candidate{index}"
        return engine.run(
            root=tmp_path, chain_id="hf-test", job_sha256="hf-job",
            segment=SimpleNamespace(index=index, seed=7 + index,
                                    plan=SimpleNamespace(save_context=True, context_frames=22)),
            candidate_id=candidate, base_candidate_id=candidate,
            high_context=context or {"empty": True}, parent_candidate_id=parent,
            parent_revision=index, projected_plan=None, inputs=inputs)
    return engine, run, calls, failure, tmp_path, inputs


def test_serial_two_stage_joint_audio_and_separate_cache_namespace(rig):
    engine, run, calls, _, root, _ = rig
    old_cache = root / "dual_stages" / "sentinel"
    old_cache.parent.mkdir(parents=True)
    old_cache.write_text("unchanged", encoding="utf-8")
    result = run()
    assert [item[0] for item in calls] == ["condition", "sample", "condition", "upscale", "sample"]
    assert [item[2] for item in calls if item[0] == "sample"] == ["denoised_x0", "zero_sigma_output"]
    assert [item[3] for item in calls if item[0] == "sample"] == [7, 8]
    assert engine.audio == ("legacy_policy", 0.0)
    assert torch.all(result["sampled"]["samples"].unbind()[1] == 8)
    assert old_cache.read_text(encoding="utf-8") == "unchanged"
    assert list((root / "hyperflow_stages").rglob("*low_x0-*.json"))
    assert not list((root / "dual_stages").rglob("*.json"))
    before = len(calls)
    cached = run()
    assert cached["sampling_report"]["dual_model"]["low_reused"]
    assert cached["sampling_report"]["dual_model"]["high_reused"]
    assert not any(row[0] == "sample" for row in calls[before:])


def test_low_context_uses_completed_high_audio_and_rechecks_accepted_movie_on_hit(rig, monkeypatch):
    _, run, calls, _, root, _ = rig
    first = run()
    _write_effects_audit(root / "candidates/segment_00000/candidate0/candidate.json", {
        "contract_sha256": "hf-job", "segment_index": 0, "candidate_id": "candidate0",
        "sampling_plan": first["sampling_report"],
    })
    from h3_audio_t8_pkg import long_video_dual_picture_context as picture
    source = {"source_media_sha256": "a" * 64}
    checks, reencodes = [], []
    def accepted(*args):
        checks.append(args)
        return root / "movie.mp4", dict(source)
    def prepare(context, *_args):
        reencodes.append(context)
        return {**context, "video_tail": context["video_tail"] + 7}, {
            "audio_tensor_preserved": True, "additional_sampling_nfe": 0}
    monkeypatch.setattr(picture, "accepted_source", accepted)
    monkeypatch.setattr(picture, "prepare_context", prepare)
    high_context = {"empty": False, "video_tail": torch.full((1, 24, 12, 2, 4), 19.0)}
    start = len(calls)
    second = run(1, "candidate0", high_context)
    low_conditions = [item for item in calls[start:] if item[0] == "condition" and item[3] == 64]
    high_conditions = [item for item in calls[start:] if item[0] == "condition" and item[3] == 128]
    assert len(low_conditions) == len(high_conditions) == 1
    assert torch.all(low_conditions[0][2]["audio_tail"] == 8)
    assert high_conditions[0][2] is high_context
    assert second["sampling_report"]["dual_model"]["first_pass"]["accepted_picture_context"]["additional_sampling_nfe"] == 0
    assert len(reencodes) == 1
    cached = run(1, "candidate0", high_context)
    assert cached["sampling_report"]["dual_model"]["high_reused"]
    assert len(checks) == 2 and len(reencodes) == 1
    source["source_media_sha256"] = "b" * 64
    changed = run(1, "candidate0", high_context)
    assert not changed["sampling_report"]["dual_model"]["low_reused"]


def test_failure_resumes_low_only_and_corrupt_receipt_fails_closed(rig):
    _, run, calls, failure, root, _ = rig
    failure["high"] = True
    with pytest.raises(RuntimeError, match="forced high failure"):
        run()
    assert list((root / "hyperflow_stages").rglob("*low_x0-*.json"))
    assert not list((root / "hyperflow_stages").rglob("*high_output-*.json"))
    failure["high"] = False
    before = len(calls)
    recovered = run()
    assert recovered["sampling_report"]["dual_model"]["low_reused"]
    # The already persisted HIGH input is also safe to reuse; only HIGH
    # sampling must be retried after its failed attempt.
    assert [item[0] for item in calls[before:]] == ["condition", "sample"]
    receipt = next((root / "hyperflow_stages").rglob("*low_x0-*.json"))
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["tensor_sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity failed"):
        run()


@pytest.mark.parametrize("fault", ["task", "audio", "relay", "media"])
def test_unsupported_stages_are_rejected_before_cache_or_sampling(rig, fault):
    engine, run, calls, _, root, inputs = rig
    if fault == "task":
        inputs["task_type"] = "Ref2VA"
    elif fault == "audio":
        inputs["audio_mode"] = "lock_source"
    elif fault == "media":
        inputs["first_frame"] = torch.zeros(1)
    with pytest.raises(ValueError):
        if fault == "relay":
            engine.run(root=root, chain_id="hf-test", job_sha256="hf-job",
                       segment=SimpleNamespace(index=0, seed=7,
                                               plan=SimpleNamespace(save_context=True, context_frames=22)),
                       candidate_id="candidate0", base_candidate_id="candidate0",
                       high_context={"empty": True}, parent_candidate_id="",
                       parent_revision=0, projected_plan={"plan_hash": "x"}, inputs=inputs)
        else:
            run()
    assert calls == []
    assert not (root / "hyperflow_stages").exists()
