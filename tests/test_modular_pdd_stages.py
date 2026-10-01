"""Real tiny Core PDD heads, not trained-file/2688-wide loader qualification.

Only the full pretrained artifact gate is outside this fixture. Native padded
patch application, head selection, AV sampler, masks, Relay and EAV execute.
"""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import pdd_advanced as pdd, sampling
from h3_audio_t8_pkg.modular_sampling import pdd_stages as split, eav
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import save_stage, load_stage
from test_fast_h3_v2_core_sampler import model
from test_modular_rf_restart import source_latent
from test_progressive_sampling_runtime import conditioning
from test_progressive_relay import paired


def prepared_inputs(variant="FL2VA", effects="none", source=None):
    bare = model()
    generator = torch.Generator().manual_seed(783)
    final = bare.get_model_object("diffusion_model.final_layer")
    banks = {}
    for stream in ("video", "audio"):
        for field in ("weight", "bias"):
            value = getattr(getattr(final, stream + "_out"), field)
            banks[f"pdd.final_layer.{stream}_out.{field}"] = torch.randn(
                (32, *value.shape), generator=generator) * .03
    prepared, _, _ = pdd._apply_native_pdd_lora(bare, banks, 1.)
    # Explicit tiny-artifact boundary, never call this a pretrained loader test.
    prepared.set_attachments(pdd.PDD_ATTACHMENT_KEY, {
        "schema": "t8_minimax_h3_pdd_8step_setup_v2",
        "lora": {"application_mode": split.NATIVE}, "base": {"variant_declared_by_user": variant},
        "test_boundary": "tiny_random_banks_not_full_pretrained_artifact_validation"})
    positive = conditioning()
    if "relay" in effects:
        prepared, positive, _ = paired(prepared)
    source = source_latent(True) if source is None else source
    prepared, sampler, full = sampling.setup_dual_clock_sampling(prepared, source, 8, 12., 3., "euler", "simple")
    return prepared, sampler, full, source, positive


def stage_inputs(stage=split.STAGES[0], variant="FL2VA", effects="none", source=None):
    prepared, old_sampler, full, source, positive = prepared_inputs(variant, effects, source)
    bound, sampler, sigmas, context, report = split.build_stage(prepared, source, full, stage)
    begin = 0 if stage == split.STAGES[0] else 4
    if begin == 4:
        prepared, old_sampler, _ = sampling.setup_dual_clock_sampling(prepared, source, 8, 12., 3.,
                                                                     "dual_clock_euler", "native_flow")
    noise = RandomNoise.execute(73).result[0]
    original = (noise, BasicGuider.execute(prepared, positive).result[0], old_sampler, full[begin:begin + 5], source)
    runtime = None
    if "eav" in effects:
        mode = "apply_exp" if "apply" in effects else "report_only"
        bound, runtime, _ = eav.apply_stage_eav(bound, sigmas, source, context,
            eav.EAVConfig(mode, tau=.2, start_video_progress=0., end_video_progress=1., g_hard_limit=3.))
    return (noise, BasicGuider.execute(bound, positive).result[0], sampler, sigmas, source, context), original, runtime, report


@pytest.mark.parametrize("stage", split.STAGES)
@pytest.mark.parametrize("variant", pdd.PDD_VARIANTS)
@pytest.mark.parametrize("effects", ["none", "relay_eav"])
def test_original_pdd_window_parity_cold_hot_rebuilt_and_saved(stage, variant, effects, tmp_path):
    args, original, runtime, report = stage_inputs(stage, variant, effects)
    expected = SamplerCustomAdvanced.execute(*original).result
    result = sample_stage(*args)[2]
    for actual, reference in zip((result.output, result.denoised_output), expected):
        assert all(torch.equal(a, b) for a, b in zip(actual["samples"].unbind(), reference["samples"].unbind()))
    receipt = result.verify()
    assert receipt["portable_identity"] is receipt["verified_recipe_completion"] is True
    assert receipt["execution"]["denoiser_evaluations"] == 4
    assert sample_stage(*args)[2].verify()["request_sha256"] == receipt["request_sha256"]
    assert sample_stage(*stage_inputs(stage, variant, effects)[0])[2].verify()["request_sha256"] == receipt["request_sha256"]
    if runtime is not None:
        assert runtime.snapshot()["status"] == "observed_report_only"
        assert runtime.snapshot()["relay_attention_calls"] == 4
    assert json.loads(report)["absolute_head_groups"] == list(range(args[-1].start, args[-1].end))
    path, digest, _ = save_stage(result, tmp_path)
    loaded = load_stage(tmp_path, path, digest, stage)[0]
    assert all(torch.equal(a, b) for a, b in zip(loaded["samples"].unbind(), result.output["samples"].unbind()))


@pytest.mark.parametrize("stage", split.STAGES)
def test_real_native_final_layer_consumes_absolute_32head_rows(stage, monkeypatch):
    from comfy.ldm.minimax import model as native
    original = native._pdd_head
    seen = []
    def observed(head, hidden, count, start, stop, shift):
        seen.append((count, start, stop, shift))
        return original(head, hidden, count, start, stop, shift)
    monkeypatch.setattr(native, "_pdd_head", observed)
    args, _, runtime, _ = stage_inputs(stage, effects="relay_eav_apply")
    result = sample_stage(*args)[2]
    expected = [(32, i * 4, (i + 1) * 4, shift) for i in range(args[-1].start, args[-1].end) for shift in (12., 3.)]
    assert seen == expected
    assert result.verify()["portable_identity"] is True
    assert runtime.snapshot()["status"] == "observed_apply_exp"
    plain = sample_stage(*stage_inputs(stage, effects="none")[0])[2]
    assert any(not torch.equal(a, b) for a, b in zip(result.output["samples"].unbind(), plain.output["samples"].unbind()))


@pytest.mark.parametrize("damage", ["head", "dtype", "window", "geometry", "receipt"])
def test_own_contract_changes_cannot_be_certified(damage):
    args, _, _, _ = stage_inputs()
    _, guider, _, sigmas, source, context = args
    selected = guider.model_patcher
    if damage == "head":
        selected.patches["diffusion_model.final_layer.video_out.weight"][-1][1][1][0][0, 0] += .1
    elif damage == "dtype":
        sigmas = sigmas.double()
    elif damage == "window":
        sigmas = sigmas.clone()
        sigmas[2] -= .001
    elif damage == "geometry":
        import comfy.nested_tensor
        video, audio = source["samples"].unbind()
        source = {"samples": comfy.nested_tensor.NestedTensor([video.repeat(1, 1, 1, 2, 1), audio])}
    else:
        selected.get_attachment(pdd.PDD_ATTACHMENT_KEY)["base"]["variant_declared_by_user"] = "changed"
    with pytest.raises(ValueError):
        split.validate_stage(selected, sigmas, source, context)


def test_plain_model_or_truncated_input_is_not_mislabeled_pdd():
    with pytest.raises(ValueError, match="existing PDD"):
        split.build_stage(model(), source_latent(), pdd.pdd_runtime_sigmas())
    prepared, _, full, source, _ = prepared_inputs()
    with pytest.raises(ValueError, match="exactly 8"):
        split.build_stage(prepared, source, full[4:])


def test_setup_preserves_original_branch_patches_and_actual_full_table():
    prepared, _, full, source, _ = prepared_inputs()
    marker = object()
    prepared.set_attachments("user_marker", marker)
    bound, _, sigmas, context, _ = split.build_stage(prepared, source, full.double(), split.STAGES[1])
    assert prepared.get_attachment(split.KEY) is None
    assert bound.get_attachment("user_marker") is marker
    assert bound.patches == prepared.patches
    assert context.trajectory_sigmas == tuple(full.tolist())
    assert sigmas.dtype is torch.float64 and sigmas.tolist() == full[4:].tolist()


@pytest.mark.parametrize("variant", pdd.PDD_VARIANTS)
def test_new_process_reads_completed_low_and_samples_only_high(tmp_path, variant):
    from h3_audio_t8_pkg import learned_latent_upscale_advanced as learned
    low = sample_stage(*stage_inputs(variant=variant, effects="relay_eav_apply")[0])[2]
    path, digest, _ = save_stage(low, tmp_path)
    high, _, _ = learned.reconcile_two_pass_h3_latent(low.denoised_output, source_latent(True), conditioning(), "auto")
    args, _, runtime, _ = stage_inputs(split.STAGES[1], variant, "relay_eav_apply", high)
    expected = sample_stage(*args)[2].verify()["outputs"]
    assert runtime.snapshot()["status"] == "observed_apply_exp"
    code = """
import json,runpy,sys
runpy.run_path('tools/check_director_d1_cpu.py',run_name='config')
runpy.run_path('tests/conftest.py')
from test_modular_pdd_stages import stage_inputs,source_latent,conditioning
from h3_audio_t8_pkg.modular_sampling.storage import load_stage
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg import learned_latent_upscale_advanced as learned
import torch
low=load_stage(sys.argv[1],sys.argv[2],sys.argv[3],'pdd_low_0_4')[1]
high,_,_=learned.reconcile_two_pass_h3_latent(low,source_latent(True),conditioning(),'auto')
args,_,runtime,_=stage_inputs('pdd_high_4_8',sys.argv[4],'relay_eav_apply',high)
receipt=sample_stage(*args)[2].verify()
assert receipt['execution']['denoiser_evaluations']==4
assert receipt['portable_identity'] and runtime.snapshot()['status']=='observed_apply_exp'
assert not torch.cuda.is_initialized()
print('RESULT='+json.dumps(receipt['outputs']))
"""
    child = subprocess.run([sys.executable, "-c", code, str(tmp_path), path, digest, variant],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=45)
    assert child.returncode == 0, child.stderr
    actual = json.loads(next(line[7:] for line in child.stdout.splitlines() if line.startswith("RESULT=")))
    assert actual == expected


@pytest.mark.parametrize("stage", split.STAGES)
def test_nonzero_backbone_lora_remains_active_and_portable(stage):
    from comfy.weight_adapter.lora import LoRAAdapter
    prepared, _, full, source, positive = prepared_inputs()
    key, value = next((key, value) for key, value in prepared.model_state_dict().items()
        if ".blocks.0.attn." in key and key.endswith("weight") and value.ndim == 2)
    adapter = LoRAAdapter({"up", "down"}, (torch.ones(value.shape[0], 1) * .2,
        torch.ones(1, value.shape[1]) * .2, 1., None, None, None))
    assert prepared.add_patches({key: adapter}, .7) == [key]
    bound, sampler, sigmas, context, _ = split.build_stage(prepared, source, full, stage)
    result = sample_stage(RandomNoise.execute(73).result[0], BasicGuider.execute(bound, positive).result[0],
                          sampler, sigmas, source, context)[2]
    assert result.verify()["portable_identity"] is True
    assert bound.patches[key][0][1] is adapter
    plain = sample_stage(*stage_inputs(stage)[0])[2]
    assert any(not torch.equal(a, b) for a, b in zip(result.output["samples"].unbind(), plain.output["samples"].unbind()))


def test_unknown_wrapper_runs_is_preserved_and_not_given_portable_identity():
    args, _, _, _ = stage_inputs()
    selected = args[1].model_patcher
    calls = []
    def wrapper(executor, *a, **kw):
        calls.append(True)
        return executor(*a, **kw)
    selected.add_wrapper_with_key("diffusion_model", "user_wrapper", wrapper)
    result = sample_stage(*args)[2]
    assert len(calls) == 4
    assert selected.get_wrappers("diffusion_model", "user_wrapper") == [wrapper]
    assert result.verify()["portable_identity"] is False


@pytest.mark.parametrize("factor", [2, 3])
def test_high_rebuilds_actual_larger_packed_geometry_and_keeps_mask_audio(factor):
    import comfy.nested_tensor
    source = source_latent(True)
    # Explicit synthetic spatial expansion, not a trained learned-upscale claim.
    video, audio = source["samples"].unbind()
    mask_video, mask_audio = source["noise_mask"].unbind()
    source["samples"] = comfy.nested_tensor.NestedTensor([video.repeat_interleave(factor, -1), audio])
    source["noise_mask"] = comfy.nested_tensor.NestedTensor([mask_video.repeat_interleave(factor, -1), mask_audio])
    args, original, runtime, _ = stage_inputs(split.STAGES[1], effects="eav", source=source)
    expected = SamplerCustomAdvanced.execute(*original).result
    result = sample_stage(*args)[2]
    assert result.verify()["portable_identity"] is True
    assert runtime.snapshot()["status"] == "observed_report_only"
    for actual, reference in zip((result.output, result.denoised_output), expected):
        assert all(torch.equal(a, b) for a, b in zip(actual["samples"].unbind(), reference["samples"].unbind()))
    assert args[-1].video_shape[-1] == 8 * factor
    assert source["samples"].unbind()[1] is audio
    assert source["noise_mask"].unbind()[1] is mask_audio


@pytest.mark.parametrize("variant", pdd.PDD_VARIANTS)
def test_legacy_dynamic_setup_retains_authenticated_real_injection_and_heads(monkeypatch, variant):
    bare = model()
    final = bare.get_model_object("diffusion_model.final_layer")
    banks = {}
    for stream in ("video", "audio"):
        for field in ("weight", "bias"):
            value = getattr(getattr(final, stream + "_out"), field)
            banks[f"pdd.final_layer.{stream}_out.{field}"] = value.detach().expand(32, *value.shape).clone()
    monkeypatch.setattr(pdd, "PDD_HEAD_SPECS", {key: (tuple(value.shape), value.dtype) for key, value in banks.items()})
    patched, _, _ = pdd._apply_dynamic_lora(bare, banks, .5)
    head = pdd.PDDHeadFinalLayer(final, *banks.values(), strength=.5, variant=variant)
    injection = pdd._create_pdd_runtime_injection(patched.get_injections(pdd.PDD_INJECTION_KEY)[0], final, head)
    patched.set_injections(pdd.PDD_INJECTION_KEY, [injection])
    patched.model_options["transformer_options"]["minimax_h3_pdd_final"] = head
    patched.add_wrapper_with_key("diffusion_model", pdd.PDD_WRAPPER_KEY, pdd.pdd_forward_wrapper)
    patched.set_attachments(pdd.PDD_ATTACHMENT_KEY, {"schema": "t8_minimax_h3_pdd_8step_setup_v2",
        "lora": {"application_mode": split.DYNAMIC}})
    for stage in split.STAGES:
        bound, _, sigmas, context, _ = split.build_stage(patched, source_latent(), pdd.pdd_runtime_sigmas(), stage)
        assert bound.get_injections(pdd.PDD_INJECTION_KEY) == [injection]
        assert bound.model_options["transformer_options"]["minimax_h3_pdd_final"] is head
        split.validate_stage(bound, sigmas, source_latent(), context)
        view, contract = split.project_identity(bound)
        assert contract["dynamic"]["schema"] == "t8.modular-sampling.pdd-dynamic.v1"
        assert not view.get_injections(pdd.PDD_INJECTION_KEY)
        assert bound.get_injections(pdd.PDD_INJECTION_KEY) == [injection]
    assert patched.get_attachment(split.KEY) is None
