"""Experimental node wrappers reach real tiny Core; no trained-media claim.

Only Qwen text embeddings are a labelled recording double. Encoder, sampler,
attention, patch consumption and both decoders use existing actual Core helpers.
"""
import json
import math

import pytest
import torch
from safetensors.torch import save_file
from comfy.ldm.minimax import model as native
from comfy.ldm.minimax import vae as video_core
from comfy.ldm.modules import attention
from comfy.nested_tensor import NestedTensor
from comfy.patcher_extension import WrappersMP
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg import nodes_solar_structural as nodes
from h3_audio_t8_pkg import solar_structural_sampling as runtime
from h3_audio_t8_pkg.audio_ops import decode_av_latent
from h3_audio_t8_pkg.core import empty_av_latent
from h3_audio_t8_pkg.sampling import setup_dual_clock_sampling
from test_fast_h3_v2_core_sampler import model
from test_h21_control_regions import RealTinyEncoderOwner
from test_h21_core_vae_contracts import real_tiny_vae_owners
from test_h24_solar_structural_core import RecordingUntrainedText, equal_av


Encode = nodes.MiniMaxH3SolarStructureEncodeEXPT8
Setup = nodes.MiniMaxH3SolarSilenceSetupEXPT8
Audit = nodes.MiniMaxH3SolarAuditEXPT8


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    monkeypatch.setattr(video_core, "optimized_attention", attention.attention_pytorch)
    with torch.no_grad():
        yield
    assert not torch.cuda.is_initialized()


@pytest.fixture(scope="module")
def encoder():
    return RealTinyEncoderOwner()


@pytest.fixture
def source():
    return torch.linspace(.1, .9, 40).reshape(-1, 1, 1, 1).expand(-1, 32, 32, 3).contiguous()


@pytest.fixture
def encoded(encoder, source):
    return Encode.execute(RecordingUntrainedText(), encoder, source, "synthetic structural test",
                          32, 32, 23, source_start=3, first_frame=source[3:4],
                          first_frame_source_index=3).result


@pytest.fixture
def silence_file(tmp_path):
    path = tmp_path / "user-silence.safetensors"
    save_file({"silence_170f": torch.full((2, 32, 37), .15)}, path)
    return str(path)


def sample(selected, positive, latent, sampler, sigmas):
    guider = BasicGuider.execute(selected, positive).result[0]
    noise = RandomNoise.execute(241027).result[0]
    return SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent).result[0]


def test_candidate_schemas_are_explicit_default_off_and_do_not_embed_loaders():
    assert len(nodes.NODES) == 3
    assert len({cls.define_schema().node_id for cls in nodes.NODES}) == 3
    assert all(cls.define_schema().is_experimental for cls in nodes.NODES)
    fields = {item.id: item for item in Setup.define_schema().inputs}
    assert fields["enabled"].default is False
    assert fields["av_latent"].optional
    assert not {"lora_name", "steps", "sampler", "sigmas"} & fields.keys()
    assert math.isnan(Setup.fingerprint_inputs(enabled=True))
    assert math.isnan(Audit.fingerprint_inputs())
    assert Audit.define_schema().is_output_node


def test_off_returns_same_object_without_reading_file_model_or_latent(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("disabled Solar touched an inactive input")

    monkeypatch.setattr(nodes.structural, "load_silence", forbidden)
    monkeypatch.setattr(nodes.runtime, "configure_silence", forbidden)
    monkeypatch.setattr(nodes, "silence_path", forbidden)
    monkeypatch.setattr(nodes, "nested_av_parts", forbidden)
    original = object()
    selected, text = Setup.execute(original, False, object(), object(), -1, object()).result
    assert selected is original
    assert Setup.fingerprint_inputs(enabled=False, silence_file=object(), av_latent=object()) == "disabled_identity"
    assert json.loads(text)["status"] == "disabled_identity"
    assert json.loads(text)["existing_effects_removed"] is False


def test_encode_delegates_actual_vae_preserves_metadata_and_exact_global_original_audio(encoder, source):
    text = RecordingUntrainedText()
    rate = 44100
    waveform = torch.linspace(-.5, .5, 73500).repeat(1, 2, 1)
    source_audio = {"waveform": waveform, "sample_rate": rate, "user_marker": "original"}
    before = waveform.clone()
    positive, latent, frames, final, raw = Encode.execute(
        text, encoder, source, "recording text double", 32, 32, 23,
        source_start=3, first_frame=source[3:4], first_frame_source_index=3,
        source_audio=source_audio).result
    report = json.loads(raw)
    assert frames == 22 and report["source_range"] == [3, 25]
    assert torch.equal(encoder.calls[-2], source[3:25])
    assert torch.equal(encoder.calls[-1], source[3:4])
    assert torch.equal(text.items[0]["data"], source[[3, 15]])
    assert positive[0][1]["user_marker"] == "retained"
    assert "minimax_keyframes" not in text.metadata
    assert [part.shape[-1] for part in latent["samples"].unbind()][-1] == 37
    # Independent exact endpoints: 3*44100/24=5512.5; 25*44100/24=45937.5.
    # Ties-to-even makes 40426 samples, not round(22*44100/24)=40425.
    assert final["waveform"].shape == (1, 2, 40426)
    assert torch.equal(final["waveform"], waveform[..., 5512:45938])
    assert final["sample_rate"] == rate and final["user_marker"] == "original"
    assert report["source_soundtrack"]["source_samples"] == [5512, 45938]
    assert not report["source_soundtrack"]["audio_driving"]
    assert not report["source_soundtrack"]["padded"] and not report["quality_accepted"]
    final["waveform"].zero_()
    assert torch.equal(waveform, before)


def test_no_original_audio_remains_absent_and_real_short_or_invalid_audio_errors(encoded):
    assert encoded[3] is None
    assert json.loads(encoded[4])["source_soundtrack"]["status"] == "not_supplied"
    with pytest.raises(ValueError, match="no implicit padding"):
        nodes._source_track({"waveform": torch.zeros(1, 2, 10), "sample_rate": 48000}, [3, 25])
    with pytest.raises(ValueError, match="positive integer"):
        nodes._source_track({"waveform": torch.zeros(1, 2, 100), "sample_rate": True}, [0, 5])
    with pytest.raises(ValueError, match="mono-or-stereo"):
        nodes._source_track({"waveform": torch.zeros(1, 3, 100), "sample_rate": 48000}, [0, 5])


def test_true_structure_coordinate_error_remains_visible(encoder, source):
    with pytest.raises(ValueError, match="source coordinate"):
        Encode.execute(RecordingUntrainedText(), encoder, source, "", 32, 32, 22,
                       source_start=3, first_frame=source[3:4], first_frame_source_index=4)


def test_setup_actual_lcm_decode_preserves_user_patches_and_audits_real_consumption(
        encoded, silence_file, monkeypatch):
    positive, latent, frames, _, _ = encoded
    bare = model()
    network = bare.model.diffusion_model
    network.__class__ = type("UserChosenSolarArchitecture", (type(network),), {})
    key = "diffusion_model.blocks.0.attn.out_proj.weight"
    delta = torch.full_like(bare.model_state_dict()[key], .001)
    bare.add_patches({key: ("diff", (delta,))}, strength_patch=.5)
    bare.add_patches({key: ("diff", (delta,))}, strength_patch=-.25)
    calls, hooks = [], []

    def user_wrapper(executor, *args, **kwargs):
        calls.append(1)
        return executor(*args, **kwargs)

    bare.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "user", user_wrapper)
    handle = network.blocks[0].register_forward_hook(lambda *_: hooks.append(1))
    try:
        timed, sampler, sigmas = setup_dual_clock_sampling(bare, latent, 4, 12., 3., "lcm", "native_flow")
        torch.testing.assert_close(sigmas, runtime.sampling_contract()[1], rtol=0, atol=0)
        baseline = sample(timed, positive, latent, sampler, sigmas)
        selected, setup_text = Setup.execute(timed, True, silence_file, seed=241027, av_latent=latent).result
        assert [patch[0] for patch in selected.patches[key]] == [.5, -.25]
        assert selected.wrappers[WrappersMP.DIFFUSION_MODEL]["user"] == [user_wrapper]
        assert json.loads(setup_text)["compatibility_warning"]
        before = json.loads(Audit.execute(selected, object()).result[0])
        assert before["last_request"]["status"] == "selected_not_executed"
        calls.clear()
        hooks.clear()
        sampled = sample(selected, positive, latent, sampler, sigmas)
        assert len(calls) == len(hooks) == 4
        video_vae, audio_vae = real_tiny_vae_owners.__wrapped__(monkeypatch)
        images, generated, *_ = decode_av_latent(sampled, video_vae, audio_vae)
        assert images.shape == (frames, 32, 32, 3)
        assert generated["waveform"].shape == (1, 2, 29600)
        report = json.loads(Audit.execute(selected, sampled).result[0])
        assert report["last_request"]["diffusion_calls"] == 4
        assert report["last_request"]["silence_audio_row_evaluations"] == 4 * 2 * 37
        assert report["last_request"]["preserved_audio_row_evaluations"] == 0
        assert all(row["shifts"] == [12., 3.] for row in report["last_request"]["clock"])
        assert not report["request_active"] and not report["sample_result_binding_proven"]
        assert not report["quality_accepted"] and not report["generated_audio_delivery_qualified"]
        equal_av(baseline, sample(timed, positive, latent, sampler, sigmas))
        assert timed.get_attachment(runtime.KEY) is None
        assert [patch[0] for patch in bare.patches[key]] == [.5, -.25]
    finally:
        handle.remove()


def test_node_reports_actual_partial_mask_and_recovers_after_downstream_failure(encoded, silence_file):
    positive, latent, _, _, _ = encoded
    video, audio = latent["samples"].unbind()
    mask = torch.ones_like(audio)
    mask[..., :12] = 0
    mask[..., 12:24] = .5
    masked = {**latent, "noise_mask": NestedTensor([torch.ones_like(video), mask])}
    timed, sampler, sigmas = setup_dual_clock_sampling(model(), masked, 1, 12., 3., "lcm", "native_flow")
    selected, _ = Setup.execute(timed, True, silence_file, av_latent=masked).result
    fail = [True]

    def downstream(executor, *args, **kwargs):
        result = executor(*args, **kwargs)
        if fail[0]:
            raise RuntimeError("node-chain downstream failure")
        return result

    selected.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "user_after", downstream)
    with pytest.raises(RuntimeError, match="node-chain downstream failure"):
        sample(selected, positive, masked, sampler, sigmas)
    failed = json.loads(Audit.execute(selected, object()).result[0])
    assert failed["last_request"]["status"] == "failed" and not failed["request_active"]
    fail[0] = False
    sampled = sample(selected, positive, masked, sampler, sigmas)
    report = json.loads(Audit.execute(selected, sampled).result[0])
    assert report["last_request"]["status"] == "silence_input_partially_delegated_unqualified"
    assert report["last_request"]["silence_audio_row_evaluations"] == 2 * 13
    assert report["last_request"]["preserved_audio_row_evaluations"] == 2 * 24
    assert selected.wrappers[WrappersMP.DIFFUSION_MODEL]["user_after"] == [downstream]
    assert not report["request_active"]


def test_enabled_file_shape_extent_and_path_errors_remain_explicit(silence_file, tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        Setup.execute(model(), True, str(tmp_path / "missing.safetensors"))
    with pytest.raises(ValueError, match="no tensor"):
        Setup.execute(model(), True, silence_file, tensor_key="missing")
    with pytest.raises(ValueError, match="joint AV"):
        Setup.execute(model(), True, silence_file, av_latent={})
    with pytest.raises(ValueError, match="actual clip needs"):
        Setup.execute(model(), True, silence_file, av_latent=empty_av_latent(32, 32, 56)[0])
    monkeypatch.setattr(nodes.folder_paths, "models_dir", str(tmp_path))
    with pytest.raises(ValueError, match="stay inside"):
        nodes.silence_path("../escaped.safetensors")


def test_foreign_attachment_is_preserved_and_same_path_reselection_reads_new_bytes(silence_file):
    bare, foreign = model(), object()
    bare.set_attachments(runtime.KEY, foreign)
    selected, raw = Setup.execute(bare, True, silence_file).result
    assert selected is bare and bare.get_attachment(runtime.KEY) is foreign
    assert json.loads(raw)["status"] == "not_installed_foreign_owner"
    first, _ = Setup.execute(model(), True, silence_file).result
    original_sha = first.get_attachment(runtime.KEY).identity["sha256"]
    save_file({"silence_170f": torch.full((2, 32, 37), -.2)}, silence_file)
    second, _ = Setup.execute(first, True, silence_file).result
    assert second.get_attachment(runtime.KEY).identity["sha256"] != original_sha
    assert torch.all(first.get_attachment(runtime.KEY).silence == .15)
    assert torch.all(second.get_attachment(runtime.KEY).silence == -.2)
    assert math.isnan(Setup.fingerprint_inputs(enabled=True, silence_file=silence_file))
    for kind in (WrappersMP.SAMPLER_SAMPLE, WrappersMP.DIFFUSION_MODEL):
        assert len(second.wrappers[kind][runtime.KEY]) == 1
