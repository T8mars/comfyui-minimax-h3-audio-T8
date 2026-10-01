"""Actual Core side-model low-memory loading; CPU is not GPU residency proof."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import comfy.model_patcher as core
from comfy.weight_adapter.lora import LoRAAdapter

from h3_audio_t8_pkg import vdn_h3_advanced as vdn
from h3_audio_t8_pkg.modular_sampling import vdn_identity, vdn_stages
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import save_stage, load_stage
from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from test_modular_vdn_baseline import tiny_vdn, source
import test_modular_vdn_relay as relay_fixture

CPU = torch.device("cpu")
WEIGHT = "blocks.0.softmax_gate.up.weight"
BIAS = "blocks.0.softmax_gate.up.bias"


def patched_model(training="stage_dmd_8nfe", kind="diff", dtype=torch.float32):
    model, branch = tiny_vdn(training)
    branch.model.to(dtype)
    weights = branch.model_state_dict()
    if kind != "none":
        value = weights[WEIGHT]
        patch = ("diff", (torch.ones_like(value) * .15,)) if kind == "diff" else LoRAAdapter(
            {"up", "down"}, (torch.ones(value.shape[0], 1) * .2,
                torch.ones(1, value.shape[1]) * .2, 1., None, None, None))
        branch.add_patches({WEIGHT: patch, BIAS: ("diff", (torch.ones_like(weights[BIAS]) * .05,))}, .7)
    return model, branch


def inputs(training="stage_dmd_8nfe", stage="vdn_complete", effects=True, latent=None, dtype=torch.float32, kind="diff"):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relay_fixture, "tiny_vdn", lambda training: patched_model(training, kind=kind, dtype=dtype))
        return relay_fixture.inputs(stage=stage, frames=16, with_eav=effects, enabled=effects,
                                    training=training, latent=latent)


def high_inputs(training, first, dtype=torch.float32):
    from h3_audio_t8_pkg import learned_latent_upscale_advanced as learned
    from test_progressive_sampling_runtime import conditioning
    high, _, _ = learned.reconcile_two_pass_h3_latent(first, source(16), conditioning(), "first_pass",
        second_pass_audio_source="first_pass", second_pass_audio_strength=0.)
    return inputs(training, "vdn_refine", latent=high, dtype=dtype)


def force_side_low_memory(monkeypatch):
    """Override only the memory choice, not Core load/patch/cast/forward math."""
    original = core.ModelPatcher.load
    calls = []

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        if type(self.model) is vdn.VDNBranchModel:
            calls.append(self)
            lowvram_model_memory, full_load = 1, False
        return original(self, device_to, lowvram_model_memory, force_patch_weights, full_load)

    monkeypatch.setattr(core.ModelPatcher, "load", load)
    return calls


@pytest.mark.parametrize("kind", ["none", "diff", "lora"])
@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_original_side_identity_cold_low_full_unload_clone_and_prepared(kind, force, dtype):
    model, branch = patched_model(kind=kind, dtype=dtype)
    expected = vdn_identity.inspect(model)
    try:
        branch.patch_model(CPU, lowvram_model_memory=1, force_patch_weights=force)
        assert branch.model.model_lowvram
        functions = [(module, name, fn) for module in branch.model.modules()
                     for name in ("weight_function", "bias_function")
                     for fn in getattr(module, name, [])]
        assert len(functions) == (2 if kind != "none" and not force else 0)
        assert vdn_identity.inspect(model) == vdn_identity.inspect(model.clone()) == expected
        for _, _, fn in functions:
            fn.prepare(None, None)
        assert vdn_identity.inspect(model) == expected
        assert all(fn.prepared_patches is not None for _, _, fn in functions)
        for _, _, fn in functions:
            fn.clear_prepared()
        branch.unpatch_model(CPU)
        assert vdn_identity.inspect(model) == expected
        branch.patch_model(CPU)
        assert not branch.model.model_lowvram
        assert vdn_identity.inspect(model) == expected
    finally:
        branch.unpatch_model(CPU)
    assert vdn_identity.inspect(model) == expected


@pytest.mark.parametrize("training", ["stage_dmd_8nfe", "stage_b_50nfe"])
@pytest.mark.parametrize("stage", vdn_stages.STAGES)
@pytest.mark.parametrize("effects", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kind", ["diff", "lora"])
def test_actual_side_low_memory_stage_matches_full_load_and_is_portable(monkeypatch, tmp_path, training, stage, effects, dtype, kind):
    normal_args, _, _, _ = inputs(training, stage, effects, dtype=dtype, kind=kind)
    full = sample_stage(*normal_args)[2]
    calls = force_side_low_memory(monkeypatch)
    args, relay, eav, _ = inputs(training, stage, effects, dtype=dtype, kind=kind)
    result = sample_stage(*args)[2]
    assert calls and all(item.model.model_lowvram for item in calls)
    same = all(torch.equal(a, b) for x, y in ((full.output, result.output), (full.denoised_output, result.denoised_output))
               for a, b in zip(x["samples"].unbind(), y["samples"].unbind()))
    assert same is (dtype is torch.float32)
    receipt = result.verify()
    assert receipt["portable_identity"] and receipt["request_sha256"] != full.verify()["request_sha256"]
    assert receipt["request"]["model"] == full.verify()["request"]["model"]
    execution = receipt["request"]["vdn_weight_execution"]
    assert execution["verified"] and execution["call_indices"] == [0] * args[-1].steps
    assert all(item["application"] == "core_delayed_compute_dtype" for item in execution["contracts"][0]["patches"])
    assert sample_stage(*args)[2].verify()["request_sha256"] == receipt["request_sha256"]
    rebuilt, _, _, _ = inputs(training, stage, effects, dtype=dtype, kind=kind)
    assert sample_stage(*rebuilt)[2].verify()["request_sha256"] == receipt["request_sha256"]
    if effects:
        assert relay.snapshot()["status"] == eav.snapshot()["status"] == "observed_apply_exp"
        assert relay.snapshot()["stats"]["linear_frames"] == 14 * args[-1].steps
    path, digest, _ = save_stage(result, tmp_path)
    assert load_stage(tmp_path, path, digest, stage)[3].verify()["request_sha256"] == receipt["request_sha256"]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("damage", ["owner", "key", "patch", "prepared", "extra", "subclass",
    "callback", "convert", "set", "missing", "double", "cast", "previous", "root_flag", "uuid"])
def test_unknown_or_corrupt_low_memory_state_cannot_claim_portable_identity(damage):
    model, branch = patched_model()
    branch.patch_model(CPU, lowvram_model_memory=1)
    module = branch.model.blocks[0].softmax_gate.up
    fn = module.weight_function[0]
    if damage == "owner":
        fn.patches = {**branch.patches, WEIGHT: []}
    elif damage == "key":
        fn.key = BIAS
    elif damage == "patch":
        fn.patches = {WEIGHT: []}
    elif damage == "prepared":
        fn.prepared_patches = []
    elif damage == "extra":
        fn.user_forward = lambda x: x
    elif damage == "subclass":
        class Foreign(core.LowVramPatch):
            pass
        module.weight_function = [Foreign(WEIGHT, branch.patches)]
    elif damage == "callback":
        module.weight_function.append(lambda x: x)
    elif damage in ("convert", "set"):
        setattr(fn, damage + "_func", lambda x: x)
    elif damage == "missing":
        module.weight_function.clear()
    elif damage == "double":
        branch.backup[WEIGHT] = type("Backup", (), {"weight": module.weight.detach().clone(), "inplace_update": False})()
    elif damage == "cast":
        module.comfy_cast_weights = False
    elif damage == "previous":
        module.prev_comfy_cast_weights = False
    elif damage == "root_flag":
        branch.model.model_lowvram = False
    else:
        branch.model.current_weight_patches_uuid = "foreign"
    before = tuple(module.weight_function)
    try:
        with pytest.raises(UnverifiedModelStack):
            vdn_identity.inspect(model)
        assert tuple(module.weight_function) == before
    finally:
        branch.model.model_lowvram = True  # allow native cleanup of the deliberately damaged fixture
        branch.unpatch_model(CPU)


@pytest.mark.parametrize("training", ["stage_dmd_8nfe", "stage_b_50nfe"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cpu_threads", [1, 2])
def test_new_process_loads_low_only_runs_low_memory_vdn_high_with_effects(tmp_path, monkeypatch, training, dtype, cpu_threads):
    torch.set_num_threads(cpu_threads)
    from h3_audio_t8_pkg import learned_latent_upscale_advanced as learned
    force_side_low_memory(monkeypatch)
    low = sample_stage(*inputs(training, dtype=dtype)[0])[2]
    path, digest, _ = save_stage(low, tmp_path)
    args, _, _, _ = high_inputs(training, low.output, dtype)
    high = sample_stage(*args)[2]
    expected = high.verify()
    audited, _ = learned.audit_two_pass_h3_audio(args[4], high.output, 0., True, 1e-5)
    assert torch.equal(audited["samples"].unbind()[1], low.output["samples"].unbind()[1])
    code = """
import json,runpy,sys
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
runpy.run_path('tests/conftest.py')
import pytest,torch
torch.set_num_threads(int(sys.argv[6]))
from test_modular_vdn_lowvram import high_inputs,force_side_low_memory
from h3_audio_t8_pkg.modular_sampling.storage import load_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg import learned_latent_upscale_advanced as learned
with pytest.MonkeyPatch.context() as patch:
    calls=force_side_low_memory(patch)
    first=load_stage(sys.argv[1],sys.argv[2],sys.argv[3],'vdn_complete')[0]
    args,relay,eav,_=high_inputs(sys.argv[4],first,getattr(torch,sys.argv[5]))
    result=sample_stage(*args)[2]
    receipt=result.verify()
    assert calls and all(item.model.model_lowvram for item in calls)
    assert receipt['portable_identity'] and receipt['execution']['denoiser_evaluations']==4
    assert relay.snapshot()['status']==eav.snapshot()['status']=='observed_apply_exp'
    audited,_=learned.audit_two_pass_h3_audio(args[4],result.output,0.,True,1e-5)
    assert torch.equal(audited['samples'].unbind()[1],first['samples'].unbind()[1])
    assert not torch.cuda.is_initialized()
    print('RESULT='+json.dumps({'request':receipt['request_sha256'],'outputs':receipt['outputs']}))
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest, training, str(dtype).split('.')[-1],
        str(torch.get_num_threads())],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60)
    assert child.returncode == 0, child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == {"request": expected["request_sha256"], "outputs": expected["outputs"]}
