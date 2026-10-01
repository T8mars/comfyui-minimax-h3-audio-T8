"""Public graphs and real tiny Core; learned network/encoders are named doubles."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import comfy.nested_tensor
import pytest
from legacy_schema_review import assert_reviewed_legacy_schemas
import torch
from comfy_api.latest import io
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise

from h3_audio_t8_pkg.modular_sampling import hyperflow_fresh as fresh, hyperflow_fresh_nodes as public, nodes as common
from h3_audio_t8_pkg.modular_sampling import results, eav, storage
from test_modular_hyperflow import inputs, equal
from test_modular_hyperflow_core_cache import TinyHyperBase, SyntheticHyperLoader
from test_modular_progressive_core_cache import TinyCondition, InterpolationDouble
from test_modular_workflows import ancestors
from test_progressive_relay import paired
from tools import build_modular_hyperflow_fresh_workflow as builder


def paired_geometry(model, source, route="joint_av_exp", prompt=0.):
    _, _, positive = paired(model)
    positive[0][0].add_(prompt)
    video, audio = source["samples"].unbind()
    layout = fresh.relay.build_packed_layout(2, video.shape[2], video.shape[3], video.shape[4],
        audio.shape[-1], keyframes=[], refs=[], frame_count=5)
    binding = fresh.relay._bind_layout_contract({"schema": fresh.relay.PROMPT_RELAY_PATCH_VERSION,
        "plan_hash": "tiny-fresh-events", "text_len": 2, "query_route": route,
        "events": [{"text_key_start": i, "text_key_end": i + 1, "midpoint": float(i * 4),
                    "window": .1, "sigma": .2} for i in range(2)]}, layout,
        resolved_task="t2va", keyframes=[], refs=[])
    marked = [[value, {**meta, fresh.relay.PROMPT_RELAY_BINDING_KEY: binding}] for value, meta in positive]
    marked = fresh.relay._attach_binding_model_cond(marked, binding["binding_hash"])
    selected, _ = fresh.relay.patch_prompt_relay_model(model, binding, 32)
    return selected, marked


def args_for(model, source, stage, kind, seed):
    from test_progressive_sampling_runtime import conditioning
    positive = conditioning()
    if kind in ("relay", "combined"):
        model, positive = paired_geometry(model, source, "video_only_paper" if "high" in stage else "joint_av_exp")
    prepared, sampler, sigmas, context, _ = public.MiniMaxH3HyperFlowFreshStageSetupEXPT8.execute(model, source, stage).result
    if kind in ("eav", "combined"):
        prepared, _, _ = eav.apply_stage_eav(prepared, sigmas, source, context,
                                           eav.EAVConfig("apply_exp", .2, 0., 1., 32, 3.))
    return RandomNoise.execute(seed).result[0], BasicGuider.execute(prepared, positive).result[0], sampler, sigmas, source, context


def learned_facade_with_network_double(source):
    from h3_audio_t8_pkg import learned_latent_upscale_advanced as learned
    from test_learned_latent_upscale_advanced import _patch_model_manager, _ResizeNetwork
    with pytest.MonkeyPatch.context() as patch:
        _patch_model_manager(patch, _ResizeNetwork())
        result = learned.learned_upscale_h3_av_latent(source, "fixture.safetensors", "scale_by", 2., .1,
            256, 128, "preserve_source", 1.05, "fp32", "offload_after")
    assert result[0]["samples"].unbind()[1] is source["samples"].unbind()[1]
    assert result[1:3] == (256, 128)
    return result[0]


@pytest.fixture(scope="module")
def info():
    return builder.base.load_live_info()


@pytest.mark.parametrize("route,kind,scope,variant", builder.CASES)
def test_all_fresh_candidates_outputs_ports_and_dependencies(info, route, kind, scope, variant):
    import execution
    graph, workflow, audit = builder.build_candidate(route, kind, scope, variant, info)
    validation = asyncio.run(execution.validate_prompt("fresh", deepcopy(graph), None))
    expected = {key for key, node in graph.items() if info[node["class_type"]].get("output_node")}
    assert validation[0] and not validation[3] and set(validation[2]) == expected
    assert audit["nodes"] == len(graph) and len(workflow["nodes"]) == len(graph) + 1
    if variant == "load_high":
        assert not {"1", "22", "6", "9", "24", "13", "29", "10", "26", "23", "11", "27"} & set(graph)
        return
    if variant == "resume_high":
        assert not {"1", "9", "10", "11", "12", "13", "50", "101", "202", "203", "204", "205"} & set(graph)
        assert graph["20"]["inputs"]["stage_result"] == ["60", 3]
    else:
        assert not {"22", "24", "26", "27", "28", "29", "102", "212", "213", "214", "215"} & ancestors(graph, "13")
    assert graph["29"]["inputs"]["noise"] == ["27", 0]
    assert graph["29"]["inputs"]["latent_image"] == ["23", 0]
    assert graph["23"]["class_type"] == "MiniMaxH3LearnedLatentUpscaleT8Advanced"
    for phase, group in (("low", 200), ("high", 210)):
        if phase == "low" and variant == "resume_high":
            continue
        active = scope in (phase, "both")
        assert (str(group + 2) in graph) == (active and kind in ("eav", "combined"))
        assert (str(group + 5) in graph) == (active and kind in ("relay", "combined"))


def test_all_430_old_schemas_unchanged_except_reviewed_named_deltas(info):
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "artifacts/development/modular-sampling-m3-hyperflow-fresh-20260923/before-registration.json").read_text(encoding="utf-8"))
    assert len(baseline["nodes"]) == 430
    assert_reviewed_legacy_schemas(baseline["nodes"], info)


@pytest.mark.parametrize("route", [0, 1])
@pytest.mark.parametrize("kind", ["eav", "relay", "combined"])
def test_actual_two_stage_resize_and_new_process_high_only(monkeypatch, tmp_path, route, kind):
    low, high, _, _ = inputs(monkeypatch, distinct=True)
    source = {"samples": comfy.nested_tensor.NestedTensor((torch.full((1, 24, 2, 4, 8), .1), torch.full((1, 32, 2, 8), -.2)))}
    result = results.sample_stage(*args_for(low, source, fresh.STAGES[route], kind, 31))[2]
    audited = public.MiniMaxH3HyperFlowFreshStageAuditEXPT8.execute(result).result
    assert audited[0] is result and json.loads(audited[3])["portable_identity"]
    monkeypatch.setattr(common, "_stage_store_root", lambda: tmp_path)
    saved = common.MiniMaxH3StageSaveEXPT8.execute(result, "LOW").result
    loaded = public.MiniMaxH3HyperFlowFreshStageLoadEXPT8.execute(saved[2], saved[3], fresh.STAGES[route]).result
    lifted = public.MiniMaxH3HyperFlowFreshLiftInputEXPT8.execute(loaded[3]).result[0]
    equal(lifted, result.output if route == 0 else result.denoised_output)
    enlarged = learned_facade_with_network_double(lifted)
    expected = results.sample_stage(*args_for(high, enlarged, fresh.STAGES[route+2], kind, 32))[2]
    assert expected.verify()["portable_identity"]
    code = r'''
import json, runpy, sys
runpy.run_path('tools/check_director_d1_cpu.py', run_name='config')
runpy.run_path('tests/conftest.py')
import pytest, torch
from test_modular_hyperflow import inputs
from test_modular_hyperflow_fresh_public import args_for, learned_facade_with_network_double
from h3_audio_t8_pkg.modular_sampling import hyperflow_fresh as fresh, results, storage
from h3_audio_t8_pkg.modular_sampling.hyperflow import snapshot
from h3_audio_t8_pkg import hyperflow_two_pass_advanced as legacy
def forbidden(*a, **kw):
    raise AssertionError('Hidden full two-pass executed')
with pytest.MonkeyPatch.context() as patch:
    _, high, _, _ = inputs(patch, distinct=True)
    patch.setattr(legacy, 'sample_hyperflow_split', forbidden)
    source = storage.load_stage(sys.argv[1], sys.argv[2], sys.argv[3], fresh.STAGES[int(sys.argv[4])])[3]
    lifted = learned_facade_with_network_double(fresh.lift_input(source)[0])
    args = args_for(high, lifted, fresh.STAGES[int(sys.argv[4])+2], sys.argv[5], 32)
    assert args[5].start == 4
    actual = results.sample_stage(*args)[2]
    receipt = actual.verify()
    assert receipt['execution']['hyperflow_fresh']['absolute_apply_intervals'] == [4,5,6,7]
    assert receipt['portable_identity'] is True
    assert receipt['execution']['denoiser_evaluations'] == 4
assert not torch.cuda.is_initialized()
print('RESULT=' + json.dumps(snapshot(actual.output)))
'''
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), saved[2], saved[3], str(route), kind],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stdout + child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    from h3_audio_t8_pkg.modular_sampling.hyperflow import snapshot
    assert actual == snapshot(expected.output)
    path, digest, _ = storage.save_stage(expected, tmp_path, "HIGH")
    def forbidden(*a, **kw):
        raise AssertionError("Completed HIGH restoration cannot sample")
    monkeypatch.setattr(results, "sample_stage", forbidden)
    restored = public.MiniMaxH3HyperFlowFreshStageLoadEXPT8.execute(path, digest, fresh.STAGES[route+2]).result
    equal(restored[0], expected.output)
    with pytest.raises(ValueError, match="expected stage"):
        public.MiniMaxH3HyperFlowFreshStageLoadEXPT8.execute(path, digest, fresh.STAGES[route])


class FreshCondition(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[io.Model.Input("model"), io.Int.Input("width"),
            io.Int.Input("height"), io.Float.Input("prompt", default=0.),
            io.Combo.Input("route", options=["joint_av_exp", "video_only_paper"])],
            outputs=[io.Model.Output(), io.Conditioning.Output(), io.Latent.Output()])

    @classmethod
    def execute(cls, model, width, height, prompt, route):
        _, source = TinyCondition.execute(width, height, prompt).result
        model, positive = paired_geometry(model, source, route, prompt)
        return io.NodeOutput(model, positive, source)


class FreshSink(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, is_output_node=True,
            inputs=[io.Custom("T8_STAGE_RESULT").Input("stage_result")], outputs=[io.String.Output()])

    @classmethod
    def execute(cls, stage_result):
        stage_result.verify()
        return io.NodeOutput(stage_result.receipt_json)


class SyntheticFreshLoader(SyntheticHyperLoader):
    @classmethod
    def execute(cls, model, patch):
        from test_hyperflow_advanced import _tiny_weights
        branch = model.clone()
        if patch:
            key, weight = next((k, w) for k, w in branch.model.named_parameters() if w.ndim == 2)
            branch.add_patches({key: ("diff", (torch.full_like(weight, patch),))})
        selected, _, _ = fresh.install(branch, _tiny_weights(branch.get_model_object("diffusion_model")))
        return io.NodeOutput(selected)


@pytest.mark.parametrize("route", builder.ROUTES)
@pytest.mark.parametrize("change", ["high_prompt", "high_model", "high_noise", "high_eav", "high_relay", "low_eav", "high_cancel"])
def test_real_core_cache_independent_phases_and_cancel(monkeypatch, route, change):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import execution
    import nodes
    import comfy.model_management
    from test_hyperflow_advanced import _tiny_native_model
    from test_progressive_sampling_runtime import tiny_model
    _, diffusion = _tiny_native_model(monkeypatch)
    base = tiny_model()
    base.model.diffusion_model = diffusion
    monkeypatch.setattr(TinyHyperBase, "execute", classmethod(lambda cls: io.NodeOutput(base)))
    for cls in (*public.NODES, TinyHyperBase, SyntheticFreshLoader, FreshCondition, InterpolationDouble, FreshSink,
                common.MiniMaxH3StageSamplerEXPT8, common.MiniMaxH3StageEAVConfigEXPT8,
                common.MiniMaxH3StageEAVApplyEXPT8, RandomNoise, BasicGuider):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
    graph = builder.split_graph(route)
    graph["1"] = {"class_type": "TinyHyperBase", "inputs": {}}
    for key in ("101", "102"):
        graph[key] = {"class_type": "SyntheticFreshLoader", "inputs": {"model": ["1", 0], "patch": .01 if key == "101" else -.02}}
    for key, loader in (("9", "101"), ("24", "102")):
        graph[key] = {"class_type": "FreshCondition", "inputs": {"model": [loader, 0],
            "width": 128 if key == "9" else ["23", 1], "height": 64 if key == "9" else ["23", 2],
            "prompt": 0., "route": "joint_av_exp"}}
    graph["23"] = {"class_type": "InterpolationDouble", "inputs": {"av_latent": ["20", 0], "target_width": 256, "target_height": 128}}
    for key in ("202", "212"):
        graph[key]["inputs"].update(mode="apply_exp", tau=.2, g_hard_limit=3.)
    graph["100"] = {"class_type": "FreshSink", "inputs": {"stage_result": ["214", 0]}}
    graph = builder.base.common.prune(graph, ["100"])
    server = SimpleNamespace(client_id=None, last_node_id=None, sockets_metadata={}, send_sync=lambda *a, **kw: None)
    executor = execution.PromptExecutor(server, cache_args={"ram": 0., "ram_inactive": 0.}, asset_manager=SimpleNamespace(enabled=False))
    original, calls, cancel = common.sample_stage, [], {"enabled": False}
    def observe(*args, **kwargs):
        context = args[-1]
        calls.append("high" if context.start else "low")
        if context.start and cancel["enabled"]:
            original_sample = args[1].sample
            def interrupting(*a, **kw):
                def interrupt(*a, **kw):
                    raise comfy.model_management.InterruptProcessingException()
                return original_sample(*a, **{**kw, "callback": interrupt})
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(args[1], "sample", interrupting)
                return original(*args, **kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(common, "sample_stage", observe)
    def run(label):
        executor.execute(deepcopy(graph), label, execute_outputs=["100"])
        assert executor.success, executor.status_messages
    run("initial")
    assert calls == ["low", "high"]
    calls.clear()
    run("unchanged")
    assert not calls
    if change == "high_prompt":
        graph["24"]["inputs"]["prompt"] = .3
    elif change == "high_model":
        graph["102"]["inputs"]["patch"] = .03
    elif change == "high_noise":
        graph["27"]["inputs"]["noise_seed"] += 1
    elif change == "high_relay":
        graph["24"]["inputs"]["route"] = "video_only_paper"
    else:
        graph["202" if change == "low_eav" else "212"]["inputs"]["tau"] = .4
    if change == "high_cancel":
        cancel["enabled"] = True
        executor.execute(deepcopy(graph), "cancel", execute_outputs=["100"])
        assert not executor.success and calls == ["high"]
        calls.clear()
        cancel["enabled"] = False
    run("changed")
    assert calls == (["low", "high"] if change == "low_eav" else ["high"])
    assert not torch.cuda.is_initialized()
