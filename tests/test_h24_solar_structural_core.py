"""Actual tiny Core VAE, PackedLayout, LoRA patches, LCM and request lifecycle.

Only Qwen's text embedding producer is a labelled recording test double. Tiny
random Core networks and synthetic adapters are not trained media qualification.
"""
from copy import deepcopy
import json

import pytest
import torch
from safetensors.torch import save_file
import comfy.ops
from comfy.nested_tensor import NestedTensor
from comfy.ldm.minimax import model as native
from comfy.ldm.minimax import vae as video_core
from comfy.ldm.modules import attention
from comfy.model_management import InterruptProcessingException
from comfy.patcher_extension import WrappersMP
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import solar_structural as structural
from h3_audio_t8_pkg import solar_structural_sampling as sampling
from test_fast_h3_v2_core_sampler import model
from test_h21_control_regions import RealTinyEncoderOwner, initialized


class RecordingUntrainedText:
    def __init__(self):
        self.items = None
        self.metadata = {"minimax_token_tags": torch.ones(2, dtype=torch.long), "user_marker": "retained"}

    def tokenize(self, prompt, minimax_ref_items):
        self.items = minimax_ref_items
        return prompt

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 2, 8), self.metadata]]


@pytest.fixture(autouse=True)
def actual_cpu(monkeypatch):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    monkeypatch.setattr(video_core, "optimized_attention", attention.attention_pytorch)
    with torch.no_grad():
        yield
    assert not torch.cuda.is_initialized()


@pytest.fixture(scope="module")
def actual_encoder():
    return RealTinyEncoderOwner()


@pytest.fixture
def prepared_condition(actual_encoder):
    source = torch.linspace(.1, .9, 40).reshape(-1, 1, 1, 1).expand(-1, 32, 32, 3).contiguous()
    text = RecordingUntrainedText()
    positive, latent, report = structural.encode_structure(
        text, actual_encoder, source, "synthetic short structure", 32, 32, 22,
        source_start=3, first_frame=source[3:4], first_frame_source_index=3)
    return positive, latent, report, source, text


def sample(selected, positive, latent):
    sampler, sigmas, _ = sampling.sampling_contract()
    guider = BasicGuider.execute(selected, positive).result[0]
    return SamplerCustomAdvanced.execute(RandomNoise.execute(241027).result[0], guider,
                                         sampler, sigmas, latent).result[0]


def equal_av(left, right):
    for actual, expected in zip(left["samples"].unbind(), right["samples"].unbind()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_actual_vae_full_structure_and_same_source_first_use_target_time_layout(prepared_condition, actual_encoder):
    positive, latent, report, source, text = prepared_condition
    metadata = positive[0][1]
    keys = metadata["minimax_keyframes"]
    assert [key["resolved_frame_index"] for key in keys] == [0, 0]
    assert tuple(keys[0]["latent"].shape) == (1, 24, 7, 2, 2)
    assert tuple(keys[1]["latent"].shape) == (1, 24, 1, 2, 2)
    assert torch.equal(actual_encoder.calls[-2], source[3:25])
    assert torch.equal(actual_encoder.calls[-1], source[3:4])
    assert torch.equal(text.items[0]["data"], source[[3, 15]])
    assert text.items[0]["timestamps"] == [0., .5]
    assert torch.equal(text.items[1]["data"], source[3:4].repeat(2, 1, 1, 1))
    assert text.items[1]["timestamps"] == [0., .5]
    assert metadata["user_marker"] == "retained" and "minimax_keyframes" not in text.metadata
    assert report["source_range"] == [3, 25] and not report["first_frame_content_verified"]
    # Real Core consumes dense structure as target-aligned cond rows, not refs.
    base = model()
    video, audio = latent["samples"].unbind()
    conds = base.model.extra_conds(cross_attn=positive[0][0], device=torch.device("cpu"),
        latent_shapes=[video.shape, audio.shape], **metadata)
    payload = conds["minimax_payload"].cond
    assert isinstance(payload["layout"], native.PackedLayout)
    assert payload["keyframes"] is keys and "refs" not in payload
    assert all(actual is key["latent"] for actual, key in zip(payload["cond_video_latents"], keys))
    layout = payload["layout"]
    structure_segment, first_segment = [row for row in layout.segments if row[2] == "cond"]
    target_segment = next(row for row in layout.segments if row[2] == "video")
    positions = layout.position_ids
    assert torch.equal(positions[structure_segment[0]:structure_segment[1]],
                       positions[target_segment[0]:target_segment[1]])
    assert torch.equal(positions[first_segment[0]:first_segment[1]],
                       positions[target_segment[0]:target_segment[0]+1])


def test_wrong_first_frame_coordinate_and_actual_encoder_shape_are_not_silently_repaired(actual_encoder):
    source = torch.zeros(22, 32, 32, 3)
    with pytest.raises(ValueError, match="coordinate"):
        structural.encode_structure(RecordingUntrainedText(), actual_encoder, source, "", 32, 32, 22,
                                    first_frame=source[:1], first_frame_source_index=1)

    class WrongShape:
        def encode(self, frames):
            return actual_encoder.encode(frames)[:, :, :-1]

    with pytest.raises(ValueError, match="target latent shape"):
        structural.encode_structure(RecordingUntrainedText(), actual_encoder, source, "", 32, 32, 22,
                                    structure_vae=WrongShape())


def test_actual_lcm_holds_seeded_silence_each_hop_and_restores_plain_model(prepared_condition):
    positive, latent, _, _, _ = prepared_condition
    bare = model()
    baseline = sample(bare, positive, latent)
    clean = torch.full((1, 32, 2, 37), .15)
    selected, _ = sampling.configure_silence(bare, clean, seed=332)
    observed = []

    def after(executor, x, timestep, context, opts, **kwargs):
        observed.append((x[1].clone(), timestep.clone()))
        return executor(x, timestep, context, opts, **kwargs)

    selected.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "user_observer", after)
    first = sample(selected, positive, latent)
    assert len(observed) == 4
    generator = torch.Generator("cpu").manual_seed(332)
    draw = torch.randn(clean.shape, generator=generator)
    for held, time in observed:
        video_sigma = float(time.flatten()[0]) / 1000
        # Independent inverse-shift/re-shift calculation, not the implementation helper.
        raw = video_sigma / (12 - 11 * video_sigma)
        audio_sigma = 3 * raw / (1 + 2 * raw)
        torch.testing.assert_close(held, (1-audio_sigma)*clean+audio_sigma*draw, rtol=1e-6, atol=1e-6)
    report = sampling.audit_silence(selected)
    assert report["last_request"]["diffusion_calls"] == 4 and not report["request_active"]
    assert not report["audio_driving"] and not report["trained_consumer_qualified"]
    assert all(torch.isfinite(value).all() for value in first["samples"].unbind())
    observed.clear()
    equal_av(first, sample(selected, positive, latent))
    assert len(observed) == 4
    assert selected.get_attachment(sampling.KEY).request.get() is None
    assert not bare.wrappers and not bare.object_patches
    equal_av(baseline, sample(bare, positive, latent))
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("mixed", [False, True])
def test_actual_core_audio_masks_keep_zero_and_partial_rows_and_restore_requests(prepared_condition, mixed):
    positive, original, _, _, _ = prepared_condition
    video, audio = original["samples"].unbind()
    mask = torch.zeros(1, 1, 2, audio.shape[-1])
    mask[..., 12:24] = .5
    if mixed:
        mask[..., 24:] = 1.
    latent = {**original, "samples": NestedTensor([video.clone(), torch.full_like(audio, .35)]),
              "noise_mask": NestedTensor([torch.ones_like(video), mask.expand_as(audio).clone()])}
    bare = model()
    baseline = sample(bare, positive, latent)
    before, observed, fail = [], [], {"enabled": False}

    def incoming(executor, x, timestep, context, opts, **kwargs):
        before.append(x[1].clone())
        return executor(x, timestep, context, opts, **kwargs)

    def consumer(executor, x, timestep, context, opts, **kwargs):
        observed.append((x[1].clone(), timestep.clone(), kwargs["audio_denoise_mask"].clone()))
        result = executor(x, timestep, context, opts, **kwargs)
        if fail["enabled"]:
            raise RuntimeError("masked request failed after actual Core consumption")
        return result

    bare.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "incoming_audio", incoming)
    clean = torch.full_like(audio, .15)
    selected, _ = sampling.configure_silence(bare, clean, seed=332)
    selected.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "masked_consumer", consumer)
    first = sample(selected, positive, latent)
    assert len(before) == len(observed) == 4
    draw = torch.randn(clean.shape, generator=torch.Generator("cpu").manual_seed(332))
    changed = 2 * (audio.shape[-1] - 24) if mixed else 0
    total = 2 * audio.shape[-1]
    for incoming_audio, (actual, timestep, actual_mask) in zip(before, observed):
        torch.testing.assert_close(actual_mask, mask, rtol=0, atol=0)
        video_sigma = float(timestep.flatten()[0]) / 1000
        raw = video_sigma / (12 - 11 * video_sigma)
        audio_sigma = 3 * raw / (1 + 2 * raw)
        full = actual_mask.expand_as(actual) == 1
        expected = torch.where(full, (1-audio_sigma)*clean+audio_sigma*draw, incoming_audio)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(actual[~full], incoming_audio[~full], rtol=0, atol=0)
    report = sampling.audit_silence(selected)["last_request"]
    assert report["silence_audio_row_evaluations"] == 4 * changed
    assert report["preserved_audio_row_evaluations"] == 4 * (total - changed)
    assert all(row["silence_audio_rows"] == changed and row["preserved_audio_rows"] == total - changed
               for row in report["clock"])
    assert report["status"] == ("silence_input_partially_delegated_unqualified" if mixed
                                else "delegated_preserving_masked_audio")
    if not mixed:
        equal_av(first, baseline)
    fail["enabled"] = True
    with pytest.raises(RuntimeError, match="after actual Core"):
        sample(selected, positive, latent)
    assert not sampling.audit_silence(selected)["request_active"]
    fail["enabled"] = False
    equal_av(first, sample(selected, positive, latent))
    equal_av(baseline, sample(bare, positive, latent))
    torch.testing.assert_close(latent["samples"].unbind()[1], torch.full_like(audio, .35), rtol=0, atol=0)


@pytest.mark.parametrize("override", [False, True])
def test_actual_core_clock_uses_diffusion_owner_with_none_sampling_audio_shift(prepared_condition, override):
    positive, latent, _, _, _ = prepared_condition
    bare = model()
    # This is an actual Core-supported unscaled audio carry, not an absent API.
    bare.model.model_sampling.set_parameters(shift=9., audio_shift=None)
    assert bare.model.model_sampling.audio_scale == 1.
    network = bare.model.diffusion_model
    network.sigma_shift_video, network.sigma_shift_audio = 6., 2.
    shifts = (8., 4.) if override else (6., 2.)
    if override:
        bare.model_options["transformer_options"].update(
            minimax_h3_sigma_shift_video=shifts[0], minimax_h3_sigma_shift_audio=shifts[1])
    baseline = sample(bare, positive, latent)
    clean = torch.full((1, 32, 2, 37), .15)
    selected, _ = sampling.configure_silence(bare, clean, seed=332)
    observed = []

    def consumer(executor, x, timestep, context, opts, **kwargs):
        assert executor.class_obj is network
        observed.append((x[1].clone(), timestep.clone()))
        return executor(x, timestep, context, opts, **kwargs)

    selected.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "actual_clock", consumer)
    first = sample(selected, positive, latent)
    assert len(observed) == 4
    draw = torch.randn(clean.shape, generator=torch.Generator("cpu").manual_seed(332))
    sv, sa = shifts
    for actual, timestep in observed:
        video_sigma = float(timestep.flatten()[0]) / 1000
        raw = video_sigma / (sv - (sv-1)*video_sigma)
        audio_sigma = sa * raw / (1 + (sa-1)*raw)
        torch.testing.assert_close(actual, (1-audio_sigma)*clean+audio_sigma*draw, rtol=1e-6, atol=1e-6)
    report = sampling.audit_silence(selected)
    assert all(row["shifts"] == list(shifts) for row in report["last_request"]["clock"])
    assert not report["request_active"]
    assert bare.model.model_sampling.audio_shift is None
    equal_av(first, sample(selected, positive, latent))
    equal_av(baseline, sample(bare, positive, latent))


def write_pair(path, base, multiplier):
    net = base.model.diffusion_model
    tensors = {}
    parts = (("attn.to_q", "attn.qkv_proj", True),
                               ("attn.to_k", "attn.qkv_proj", True),
                               ("attn.to_v", "attn.qkv_proj", True),
                               ("attn.to_out.0", "attn.out_proj", False),
                               ("ff.net.0.proj", "mlp.fc1", False),
                               ("ff.net.2", "mlp.fc2", False))
    scopes = [("transformer_blocks.0.", "blocks.0.")]
    scopes.extend((f"token_refiner.refiner_blocks.{index}.", f"token_refiner.blocks.{index}.")
                  for index in range(len(net.token_refiner.blocks)))
    for source, destination in scopes:
        for suffix, target, qkv in parts:
            module = net.get_submodule(destination + target)
            key = source + suffix
            out = module.out_features // (3 if qkv else 1)
            tensors[key + ".lora_A.weight"] = torch.full((2, module.in_features), multiplier)
            tensors[key + ".lora_B.weight"] = torch.linspace(-.2, .3, out * 2).reshape(out, 2)
    save_file(tensors, str(path))


def test_actual_lora_pair_maps_fused_targets_keeps_user_order_and_unknown_class(tmp_path, prepared_condition):
    positive, latent, _, _, _ = prepared_condition
    bare = model()
    net = bare.model.diffusion_model
    net.token_refiner.blocks.append(initialized(native.RefinerBlock(24, 3, 128, 32, 1e-6, 1e-6,
        dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init), 241028))
    net.__class__ = type("UserChosenSolarCompatibleNetwork", (type(net),), {})
    key = "diffusion_model.blocks.0.attn.qkv_proj.weight"
    bare.add_patches({key: ("diff", (torch.full_like(bare.model_state_dict()[key], .001),))}, -.25)
    paths = [tmp_path / "user-one.safetensors", tmp_path / "user-two.safetensors"]
    write_pair(paths[0], bare, .02)
    write_pair(paths[1], bare, .03)
    patches = deepcopy(bare.patches)
    baseline = sample(bare, positive, latent)
    selected, receipt = structural.load_lora_pair(bare, *paths, fewstep_strength=.5, control_strength=-.75)
    assert [patch[0] for patch in selected.patches[key]] == [-.25, .5, -.75]
    assert len(bare.patches[key]) == len(patches[key]) == 1
    assert all(row["loader"]["applied_patch_count"] == 8 for row in receipt["adapters"])
    assert "diffusion_model.token_refiner.blocks.0.mlp.fc2.weight" in selected.patches
    output = sample(selected, positive, latent)
    assert not torch.equal(output["samples"].unbind()[0], baseline["samples"].unbind()[0])
    equal_av(baseline, sample(bare, positive, latent))


@pytest.mark.parametrize("failure", ["exception", "interrupt", "nan", "shape"])
def test_actual_downstream_failure_releases_request_and_keeps_user_wrappers(failure, prepared_condition):
    positive, latent, _, _, _ = prepared_condition
    bare = model()
    before_calls = []

    def before(executor, *args, **kwargs):
        before_calls.append(1)
        return executor(*args, **kwargs)

    bare.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "user_before", before)
    baseline = sample(bare, positive, latent)
    selected, _ = sampling.configure_silence(bare, torch.zeros(1, 32, 2, 37), seed=332)

    def broken(executor, *args, **kwargs):
        if failure == "exception":
            raise RuntimeError("actual downstream failure")
        if failure == "interrupt":
            raise InterruptProcessingException()
        result = executor(*args, **kwargs)
        result[1] = result[1] * float("nan") if failure == "nan" else result[1][..., :-1]
        return result

    selected.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "user_after", broken)
    with pytest.raises((RuntimeError, ValueError, InterruptProcessingException)):
        sample(selected, positive, latent)
    report = sampling.audit_silence(selected)
    assert report["last_request"]["status"] == "failed" and not report["request_active"]
    assert selected.wrappers[WrappersMP.DIFFUSION_MODEL]["user_before"] == [before]
    assert selected.wrappers[WrappersMP.DIFFUSION_MODEL]["user_after"] == [broken]
    assert bare.wrappers[WrappersMP.DIFFUSION_MODEL]["user_before"] == [before]
    equal_av(baseline, sample(bare, positive, latent))


def test_reselection_replaces_only_owned_wrappers_and_short_silence_is_real_error(prepared_condition):
    positive, latent, _, _, _ = prepared_condition
    first, _ = sampling.configure_silence(model(), torch.zeros(1, 32, 2, 37), seed=1)
    second, _ = sampling.configure_silence(first, torch.zeros(1, 32, 2, 36), seed=2)
    assert first.get_attachment(sampling.KEY).seed == 1
    for kind in (WrappersMP.SAMPLER_SAMPLE, WrappersMP.DIFFUSION_MODEL):
        assert len(second.wrappers[kind][sampling.KEY]) == 1
        assert second.wrappers[kind][sampling.KEY][0].__self__ is second.get_attachment(sampling.KEY)
    with pytest.raises(ValueError, match="actual clip needs 37"):
        sample(second, positive, latent)
    assert not sampling.audit_silence(second)["request_active"]
