"""Actual Core cache and retained backend/patch execution for external effects."""
from copy import deepcopy

import pytest
import torch
from comfy_api.latest import io

from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg.modular_sampling import progressive_effect_nodes as nodes
from h3_audio_t8_pkg.modular_sampling import progressive_effects as effects
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig
from h3_audio_t8_pkg.modular_sampling.nodes import MiniMaxH3StageEAVConfigEXPT8
from test_modular_progressive_effects import selected, sample
from test_modular_progressive_stages import inputs, low_only, restart
from test_progressive_relay import paired
import test_modular_progressive_core_cache as core_cache
import test_progressive_sampling_runtime as fixtures
import test_relay_kj_memory as memory_fixtures
import test_relay_sol_backend as sol_fixtures
from test_relay_kj_backend import kj as kj

stub_lifter = fixtures.stub_lifter
memory_nodes = memory_fixtures.memory_nodes
installed_sol = sol_fixtures.installed_sol


def distinct_pair(model, marker):
    _, positive, plain = paired(model)
    binding = deepcopy(positive[0][1][relay.PROMPT_RELAY_BINDING_KEY])
    binding["plan_hash"] = f"independent-event-plan-{marker}"
    binding["events"][0]["window"] = .1 + marker
    binding.pop("binding_hash")
    binding["binding_hash"] = relay._sha256_json(binding)
    marked = [[embedding, {**meta, relay.PROMPT_RELAY_BINDING_KEY: binding}] for embedding, meta in plain]
    marked = relay._attach_binding_model_cond(marked, binding["binding_hash"])
    bound, _ = relay.patch_prompt_relay_model(model, binding, 32)
    return bound, marked


class TinyRelayEncodingDouble(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, inputs=[io.Model.Input("model"),
            io.Int.Input("width"), io.Int.Input("height"), io.Float.Input("prompt", default=0.)],
            outputs=[io.Model.Output(), io.Conditioning.Output(), io.Latent.Output()])

    @classmethod
    def execute(cls, model, width, height, prompt):
        _, latent = core_cache.TinyCondition.execute(width, height, prompt).result
        model, positive = distinct_pair(model, prompt)
        return io.NodeOutput(model, positive, latent)


@pytest.mark.parametrize("kind", ["eav", "relay", "combined"])
@pytest.mark.parametrize("change", ["high_prompt", "high_model_patch", "high_video_noise", "low_prompt", "high_cancel"])
def test_real_core_cached_effect_apply_nodes_invalidate_only_changed_stage(monkeypatch, kind, change):
    original_graph = core_cache.graph
    def graph():
        value = original_graph()
        for phase, encoder, stage, sampler, model, group, plan in (
            ("low", "9", "12", "13", "10", 220, ["26", 0]),
            ("high", "24", "28", "29", "92", 230, ["25", 1])):
            if kind != "eav":
                value[encoder]["class_type"] = "TinyRelayEncodingDouble"
                value[encoder]["inputs"]["model"] = [model, 0]
                value[stage] = {"class_type": "MiniMaxH3ProgressiveRelayStageApplyEXPT8", "inputs": {
                    "model": [encoder, 0], "positive": [encoder, 1], "negative": [encoder, 1],
                    "plan": plan, "phase": phase, "mode": "apply_exp", "guide_resize": "legacy_bilinear"}}
                value[sampler]["inputs"].update(model=[stage, 0], positive=[stage, 1], negative=[stage, 2])
                if phase == "high":
                    value["25"]["inputs"]["high_source"] = [encoder, 2]
            if kind != "relay":
                config, apply = str(group), str(group + 1)
                value[config] = {"class_type": "MiniMaxH3StageEAVConfigEXPT8", "inputs": {
                    "mode": "apply_exp", "tau": .2, "start_video_progress": 0., "end_video_progress": 1.,
                    "max_workspace_mib": 32, "g_hard_limit": 1.5}}
                value[apply] = {"class_type": "MiniMaxH3Progressive" + phase.title() + "EAVApplyEXPT8", "inputs": {
                    "model": value[sampler]["inputs"]["model"], "eav_config": [config, 0]}}
                if phase == "low":
                    value[apply]["inputs"].update(plan=plan, low_source=["26", 1])
                else:
                    value[apply]["inputs"]["high_restart"] = ["25", 0]
                value[sampler]["inputs"]["model"] = [apply, 0]
        return value
    monkeypatch.setattr(core_cache, "graph", graph)
    monkeypatch.setattr(core_cache.public, "NODES", [*core_cache.public.NODES, *nodes.NODES,
        MiniMaxH3StageEAVConfigEXPT8, TinyRelayEncodingDouble])
    original_sink = core_cache.ResultSink.execute
    observed = []
    def sink(cls, result):
        report = effects.audit(result, "high")
        assert report["status"] == "verified_stage_execution_quality_unverified"
        if kind != "relay":
            assert report["eav"]["model_forward_count"] == 2
        if kind != "eav":
            assert report["relay"]["completed_calls"] == {"forward": 2, "routed_attention": 2}
        observed.append(report)
        return original_sink(result)
    monkeypatch.setattr(core_cache.ResultSink, "execute", classmethod(sink))
    core_cache.test_actual_core_cache_only_invalidates_the_changed_phase(monkeypatch, change)
    assert len(observed) == 2
    if kind != "eav" and change in ("high_prompt", "high_cancel"):
        assert observed[0]["relay"]["plan_hash"] != observed[1]["relay"]["plan_hash"]


@pytest.mark.parametrize("kind", ["relay", "eav", "combined"])
@pytest.mark.parametrize("backend", ["kj_memory", "sol"])
def test_selected_installed_backend_and_ordered_weight_patches_are_retained(stub_lifter, memory_nodes, installed_sol, kind, backend):
    case = inputs()
    state = restart(case, low_only(case))
    base = case["model"]
    if backend == "kj_memory":
        lowmem, sage, _ = memory_nodes
        model = sage.MiniMaxH3MemoryEfficientSageAttentionPatch.execute(base).result[0]
        model = lowmem.MiniMaxLowVRAMAttention.execute(model, 2).result[0]
    else:
        model = installed_sol.SolAttentionPatch().patch(base, True, .5, min_tokens=256)[0]
    key, weight = next(iter(dict(model.model.named_parameters()).items()))
    model.add_patches({key: ("diff", (torch.ones_like(weight) * .1,))}, strength_patch=.3)
    model.add_patches({key: ("diff", (torch.ones_like(weight) * .2,))}, strength_patch=.2)
    before = list(model.patches[key])
    selected_methods = dict(model.object_patches)
    selected_override = model.model_options['transformer_options'].get('optimized_attention_override')
    original_sage_delegate = memory_nodes[1]._sageattn_int8_fp8_nhd
    args = selected(case, "high", source=state, model=model, with_relay=kind != "eav",
                    config=EAVConfig("apply_exp", .2, 0., 1.) if kind != "relay" else None)
    result = sample(case, "high", *args, state=state)
    report = effects.audit(result, "high")
    assert report['status'] == 'verified_stage_execution_quality_unverified'
    assert report['quality_accepted'] is False
    if kind != 'relay':
        assert report['eav']['model_forward_count'] == 2
    if kind != 'eav':
        assert report['relay']['completed_calls'] == {'forward': 2, 'routed_attention': 2}
    assert set(model.object_patches) == set(selected_methods)
    assert all(model.object_patches[path] is method for path, method in selected_methods.items())
    assert model.model_options['transformer_options'].get('optimized_attention_override') is selected_override
    if backend == 'kj_memory':
        # The original projection forward is scoped to a routed delegate (the
        # real dense-equation tests verify it), not a global Sage replacement.
        assert memory_nodes[1]._sageattn_int8_fp8_nhd is original_sage_delegate
    assert len(args[0].patches[key]) == len(model.patches[key]) == 2
    assert all(a is b for a, b in zip(before, model.patches[key]))
    assert result.verify()["portable_identity"] is False
    # The CPU Sol backend intentionally delegates; this does not certify a
    # CUDA Sol kernel. Native actual effect counts above are independently checked.


@pytest.mark.parametrize("kind", ["relay", "eav", "combined"])
def test_foreign_attention_delegate_runs_and_is_not_certified_portable(stub_lifter, kind):
    from h3_audio_t8_pkg.h3_core_compat import set_h3_attention_backend
    from comfy.ldm.modules import attention
    case = inputs()
    state = restart(case, low_only(case))
    calls = []
    def delegate(q, k, v, heads, **kwargs):
        calls.append(kwargs.get("mask") is not None)
        return attention.attention_pytorch(q, k, v, heads, **kwargs)
    patched = case["model"].clone()
    set_h3_attention_backend(patched, delegate)
    args = selected(case, "high", source=state, model=patched, with_relay=kind != "eav",
                    config=EAVConfig("apply_exp", .2, 0., 1.) if kind != "relay" else None)
    result = sample(case, "high", *args, state=state)
    assert len(calls) >= 2
    assert effects.audit(result, "high")["status"] == "verified_stage_execution_quality_unverified"
    assert not result.verify()["portable_identity"]


@pytest.mark.parametrize("kind", ["relay", "eav", "combined"])
def test_same_owner_early_resource_failure_clears_old_audit(stub_lifter, monkeypatch, kind):
    case = inputs()
    state = restart(case, low_only(case))
    args = selected(case, "high", source=state, with_relay=kind != "eav",
                    config=EAVConfig("apply_exp", .2, 0., 1.) if kind != "relay" else None)
    sample(case, "high", *args, state=state)
    runtime = effects.owner(args[0])
    from h3_audio_t8_pkg.modular_sampling import progressive as stages
    def fail(*a, **k):
        raise RuntimeError("synthetic reserve failure")
    monkeypatch.setattr(stages.legacy, "_resource_snapshot", fail)
    with pytest.raises(RuntimeError, match="reserve failure"):
        sample(case, "high", *args, state=state)
    assert runtime.last_report["status"] == "aborted" and not runtime.lock.locked()
    if runtime.eav_runtime is not None:
        assert runtime.eav_runtime.snapshot(consume=False)["model_forward_count"] == 0
    if runtime.relay_report is not None:
        assert runtime.relay_report["completed_calls"] == {"forward": 0, "routed_attention": 0}


@pytest.mark.parametrize("change", ["foreign_instance_forward", "class_forward", "hook", "weight_patch"])
@pytest.mark.parametrize("unknown_stack", [False, True])
def test_cleanup_identity_normalization_does_not_hide_real_model_mutations(stub_lifter, monkeypatch, change, unknown_stack):
    from types import MethodType
    from h3_audio_t8_pkg.modular_sampling import progressive as stages
    case = inputs()
    state = restart(case, low_only(case))
    args = selected(case, "high", source=state, config=EAVConfig("report_only"))
    if unknown_stack:
        args[0].add_wrapper_with_key("apply_model", "unverified", lambda executor, *a, **k: executor(*a, **k))
    module = case["model"].model.diffusion_model.blocks[0]
    before = stages.native_model_identity(args[0], case["sampler"])
    assert stages.portable(before) is not unknown_stack
    if change == "foreign_instance_forward":
        original = module.forward
        monkeypatch.setattr(module, "forward", MethodType(lambda self, *a, **k: original(*a, **k), module))
    elif change == "class_forward":
        original = type(module).forward
        monkeypatch.setattr(type(module), "forward", lambda self, *a, **k: original(self, *a, **k))
    elif change == "hook":
        hook = module.register_forward_pre_hook(lambda *a: None)
    else:
        key, weight = next(iter(dict(args[0].model.named_parameters()).items()))
        args[0].add_patches({key: ("diff", (torch.ones_like(weight) * .1,))})
    after = stages.native_model_identity(args[0], case["sampler"])
    assert not stages.model_identity_matches(before, after)
    if change == "hook":
        hook.remove()


@pytest.mark.parametrize("with_hook", [False, True])
def test_only_exact_native_bound_method_alias_is_equal(with_hook, monkeypatch):
    from types import MethodType
    from h3_audio_t8_pkg.modular_sampling import progressive as stages
    case = inputs()
    model, _, _ = selected(case, "low", config=EAVConfig("report_only"))
    module = model.model.diffusion_model.blocks[0]
    handle = module.register_forward_pre_hook(lambda *a: None) if with_hook else None
    before = stages.native_model_identity(model, case["sampler"])
    monkeypatch.setattr(module, "forward", MethodType(type(module).forward, module))
    after = stages.native_model_identity(model, case["sampler"])
    assert stages.model_identity_matches(before, after)
    assert stages.portable(after) is not with_hook
    if handle:
        handle.remove()
