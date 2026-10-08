"""New AV importer boundaries only. Tiny encoders do not certify real weights."""
import json
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch

from h3_audio_t8_pkg import external_reference_av_import as runtime
from h3_audio_t8_pkg.reference_package import file_sha, load_package, route_packages, save_package


class TinyVideo:
    def __init__(self):
        self.calls = 0

    def encode(self, frames):
        self.calls += 1
        t = (len(frames) - 5) // 17 * 5 + 2
        return torch.arange(24 * t * 4, dtype=torch.float32).reshape(1, 24, t, 2, 2) / 197 + frames.mean()


class TinyAudio:
    audio_sample_rate = 32000

    def __init__(self):
        self.spans = []

    def encode(self, pcm):
        # The native wrapper takes B,L,C rather than the codec's B,C,L.
        assert pcm.ndim == 3 and pcm.shape[-1] == 2
        self.spans.append(pcm.shape[1])
        t = (pcm.shape[1] + 799) // 800
        return torch.arange(t, dtype=torch.float32).view(1, 1, 1, t).repeat(1, 32, 2, 1) / 997 + pcm.mean()


@pytest.fixture
def identity(monkeypatch):
    monkeypatch.setattr(runtime, "portable_producer", lambda component, role: {
        "role": role, "sha256": "1" * 64,
        "scope": "actual_native_weights_tokenizer_configuration_and_implementation"})


def asset(path, kind, latent, *, hybrid=False, mode="encode"):
    member = {"_format_version": 4, "kind": kind, "mode": mode,
              "refmod_config": '{"execute":"must never run"}', "source": "Z:/not-followed.mp4"}
    if kind == "audio":
        member["sample_rate"] = 32000
    if hybrid:
        # Deliberately nonfinite unselected reference/LoRA payload must never be materialized.
        other = {"_format_version": 4, "kind": "image", "mode": "encode"}
        tensors = {"ref_0": torch.full((1, 24, 1, 2, 2), float("nan")), "ref_1": latent,
                   "lora.test.weight": torch.full((2, 2), float("nan"))}
        metadata = {"refmod_meta": json.dumps({"_format_version": 5, "kind": "bundle", "members": [other, member]}),
                    "h3_hybrid": json.dumps({"version": 1, "lora": {"keys": 1}, "refmod_count": 2})}
    else:
        tensors, metadata = {"latent": latent}, {"refmod_meta": json.dumps(member)}
    save_file(tensors, str(path), metadata=metadata)
    return file_sha(path)


def video(path, frames, vae, **kwargs):
    return runtime.import_video(path, frames, vae, expected_sha256=file_sha(path), confirm_import=True, **kwargs)


def audio(path, pcm, vae, **kwargs):
    return runtime.import_audio(path, {"waveform": pcm, "sample_rate": 32000}, vae,
                                expected_sha256=file_sha(path), confirm_import=True, **kwargs)


def test_video_exact_preserves_all_frames_routes_and_create_only_roundtrip(tmp_path, identity):
    frames = torch.linspace(0, 1, 5 * 32 * 32 * 3).reshape(5, 32, 32, 3)
    vae = TinyVideo()
    native = vae.encode(frames)
    vae.calls = 0
    path = tmp_path / "video.safetensors"
    digest = asset(path, "video", native)
    original = frames.clone()
    package, report = video(path, frames, vae)
    assert vae.calls == 1 and torch.equal(frames, original) and file_sha(path) == digest
    assert report["selected_native_encoding_bytes_exact"] and report["prepared_frames"] == 5
    assert not report["AV_synchronization_certified"] and not report["source_time_origin_inferred"]
    routed = route_packages([package], [{"role_id": "A", "visual": True, "voice": False}])
    assert routed["refs"][0]["kind"] == "video" and routed["refs"][0]["ref_audio_t"] == 0
    assert torch.equal(routed["qwen_ref_items"][0]["data"], frames)
    out = tmp_path / "native.safetensors"
    saved = save_package(package, out, confirmed=True)
    assert load_package(out, expected_sha256=saved["sha256"]).verify() == package.verify()
    with pytest.raises(FileExistsError):
        save_package(package, out, confirmed=True)


def test_video_fp16_is_explicit_and_selected_only_in_hybrid(tmp_path, identity, monkeypatch):
    frames, vae = torch.full((5, 32, 32, 3), 0.25), TinyVideo()
    path = tmp_path / "hybrid.safetensors"
    digest = asset(path, "video", vae.encode(frames).half(), hybrid=True)
    with pytest.raises(ValueError, match="exactly match"):
        video(path, frames, vae, member_ordinal=2)
    actual = runtime.safe_open
    loads = []

    class SelectedReader:
        def __enter__(self):
            self.context = actual(path, framework="pt", device="cpu")
            self.stream = self.context.__enter__()
            return self

        def get_tensor(self, key):
            loads.append(key)
            return self.stream.get_tensor(key)

        def __exit__(self, *args):
            return self.context.__exit__(*args)

    monkeypatch.setattr(runtime, "safe_open", lambda *a, **k: SelectedReader())
    package, report = video(path, frames, vae, member_ordinal=2, precision_profile="author_encode_fp16")
    assert loads == ["ref_1"] and report["external_latent_bytes_preserved"]
    assert not report["whole_clip_native_encode_bit_parity_claimed"] and file_sha(path) == digest
    assert package.tensors["visual_latent"].dtype == torch.float16


def test_video_temporal_geometry_clock_and_training_reject_before_encode(tmp_path, identity):
    frames, vae = torch.zeros((5, 32, 32, 3)), TinyVideo()
    path = tmp_path / "video.safetensors"
    asset(path, "video", vae.encode(frames))
    vae.calls = 0
    for wrong_frames, kwargs, message in [(frames[:4], {}, "geometry"),
                                         (frames[:, :16], {}, "geometry"),
                                         (frames, {"source_fps": 30}, "24fps"),
                                         (frames, {"member_ordinal": True}, "integer"),
                                         (frames, {"member_ordinal": 2}, "ordinal")]:
        with pytest.raises(ValueError, match=message):
            video(path, wrong_frames, vae, **kwargs)
    asset(path, "video", torch.zeros((1, 24, 3, 2, 2)))
    with pytest.raises(ValueError, match="temporal grid"):
        video(path, frames, vae)
    asset(path, "video", torch.zeros((1, 24, 2, 2, 2)), mode="pooled")
    with pytest.raises(ValueError, match="Training/pooled"):
        video(path, frames, vae)
    assert vae.calls == 0


def test_audio_exact_stereo_unaligned_tail_routes_without_final_track(tmp_path, identity):
    pcm, vae = torch.full((1, 2, 1601), 0.1), TinyAudio()
    native = vae.encode(pcm.movedim(1, -1))
    vae.spans.clear()
    path = tmp_path / "audio.safetensors"
    digest = asset(path, "audio", native)
    original = pcm.clone()
    package, report = audio(path, pcm, vae, role_id="speaker_B")
    assert vae.spans == [1601] and torch.equal(pcm, original) and file_sha(path) == digest
    assert report["native_encode_chunks"] == [{"sample_start": 0, "sample_stop": 1601, "right_zero_padding_samples": 799}]
    assert report["whole_clip_native_encode_bit_parity_claimed"] and not report["PCM_stored_in_package"]
    routed = route_packages([package], [{"role_id": "speaker_B", "visual": False, "voice": True}])
    assert routed["qwen_ref_items"] == [{"type": "audio"}] and routed["mapping"][0]["ordinal"] == 1
    assert torch.equal(routed["refs"][0]["audio_latent"], native)
    out = tmp_path / "voice.safetensors"
    saved = save_package(package, out, confirmed=True)
    assert load_package(out, expected_sha256=saved["sha256"]).verify() == package.verify()


def test_audio_author_chunk_profile_keeps_boundary_and_no_fp16_conversion(tmp_path, identity):
    pcm, vae = torch.zeros((1, 2, 320801)), TinyAudio()
    pcm[..., 320000:] = 0.4
    native = torch.cat([vae.encode(pcm[..., :320000].movedim(1, -1)),
                        vae.encode(pcm[..., 320000:].movedim(1, -1))], dim=-1)
    path = tmp_path / "chunked.safetensors"
    asset(path, "audio", native, hybrid=True)
    with pytest.raises(ValueError, match="exactly match"):
        audio(path, pcm, vae, member_ordinal=2)
    vae.spans.clear()
    package, report = audio(path, pcm, vae, member_ordinal=2, encoding_profile="author_10s_native_exact")
    assert vae.spans == [320000, 801] and report["native_encode_calls"] == 2
    assert report["native_encode_chunks"][1] == {"sample_start": 320000, "sample_stop": 320801, "right_zero_padding_samples": 799}
    assert package.tensors["voice_latent"].dtype == torch.float32
    assert report["selected_native_encoding_bytes_exact"] and not report["whole_clip_native_encode_bit_parity_claimed"]
    assert not report["whole_clip_vs_chunked_boundary_equivalence_claimed"]


def test_audio_source_preparation_shape_length_and_precision_are_not_guessed(tmp_path, identity):
    pcm, vae = torch.zeros((1, 2, 1601)), TinyAudio()
    path = tmp_path / "audio.safetensors"
    asset(path, "audio", vae.encode(pcm.movedim(1, -1)))
    vae.spans.clear()
    for value, rate in [(pcm[:, :1], 32000), (pcm.double(), 32000), (pcm, 44100), (pcm, True),
                        (pcm[..., :800], 32000), (torch.zeros((1, 2, 960001)), 32000)]:
        with pytest.raises(ValueError):
            runtime.import_audio(path, {"waveform": value, "sample_rate": rate}, vae,
                                 expected_sha256=file_sha(path), confirm_import=True)
    with pytest.raises(ValueError, match="encoding profile"):
        audio(path, pcm, vae, encoding_profile="author_encode_fp16")
    asset(path, "audio", torch.zeros((1, 32, 2, 3), dtype=torch.float16))
    with pytest.raises(ValueError, match="exactly match"):
        audio(path, pcm, vae)
    assert vae.spans == [1601]  # Only the final explicit dtype mismatch reaches a real encode.


def test_bad_confirmation_hash_kind_and_nonfinite_payload_never_encode(tmp_path, identity):
    pcm, vae = torch.zeros((1, 2, 1601)), TinyAudio()
    path = tmp_path / "audio.safetensors"
    digest = asset(path, "audio", torch.zeros((1, 32, 2, 3)))
    for digest_value, confirmed in [(digest, False), ("0" * 64, True)]:
        with pytest.raises(ValueError):
            runtime.import_audio(path, {"waveform": pcm, "sample_rate": 32000}, vae,
                                 expected_sha256=digest_value, confirm_import=confirmed)
    asset(path, "audio", torch.full((1, 32, 2, 3), float("nan")))
    with pytest.raises(ValueError, match="finite"):
        audio(path, pcm, vae)
    asset(path, "video", torch.zeros((1, 24, 2, 2, 2)))
    with pytest.raises(ValueError, match="kind"):
        audio(path, pcm, vae)
    assert vae.spans == []


def test_source_mutation_producer_change_file_change_and_encode_errors_do_not_publish(tmp_path, identity, monkeypatch):
    frames, vae = torch.full((5, 32, 32, 3), 0.25), TinyVideo()
    path = tmp_path / "video.safetensors"
    asset(path, "video", vae.encode(frames))
    actual = vae.encode

    def changing_source(rgb):
        out = actual(rgb)
        rgb.add_(0.01)
        return out

    monkeypatch.setattr(vae, "encode", changing_source)
    with pytest.raises(ValueError, match="changed during validation"):
        video(path, frames, vae)
    checks = []
    monkeypatch.setattr(vae, "encode", actual)
    monkeypatch.setattr(runtime, "portable_producer", lambda component, role: checks.append(role) or {
        "role": role, "sha256": str(len(checks)) * 64,
        "scope": "actual_native_weights_tokenizer_configuration_and_implementation"})
    with pytest.raises(ValueError, match="changed during validation"):
        video(path, frames, vae)
    monkeypatch.setattr(runtime, "portable_producer", lambda component, role: {
        "role": role, "sha256": "1" * 64,
        "scope": "actual_native_weights_tokenizer_configuration_and_implementation"})

    def changing_file(rgb):
        out = actual(rgb)
        with path.open("ab") as stream:
            stream.write(b"\0")
        return out

    monkeypatch.setattr(vae, "encode", changing_file)
    with pytest.raises(ValueError, match="changed during import"):
        video(path, frames, vae)
    asset(path, "video", actual(frames))

    def broken(rgb):
        raise RuntimeError("native encode failed")

    monkeypatch.setattr(vae, "encode", broken)
    with pytest.raises(RuntimeError, match="native encode failed"):
        video(path, frames, vae)
    assert list(tmp_path.iterdir()) == [path]


def test_actual_core_audio_normalization_and_tail_wrapper_match_direct_codec(tmp_path, identity, monkeypatch):
    # Execute the actual Core wrapper and actual codec encode method, replacing
    # only the learned sublayers with a tiny deterministic unit fixture. This
    # proves axis/padding/normalization plumbing, NOT trained-weight qualification.
    from comfy import model_management
    from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
    from comfy.sd import VAE

    class TinySublayer(torch.nn.Module):
        def forward(self, value):
            return value.reshape(value.shape[0], 1, -1, 800).mean(-1).repeat(1, 32, 1)

    model = MiniMaxH3AudioVAE.__new__(MiniMaxH3AudioVAE)
    torch.nn.Module.__init__(model)
    model.hop_length = 800
    model.encoder = TinySublayer()
    model.pre_block = torch.nn.Identity()
    model.mean_proj = torch.nn.Identity()
    model.register_buffer("latents_mean", torch.arange(32, dtype=torch.float32) / 100)
    model.register_buffer("latents_std", torch.linspace(0.5, 1.5, 32))
    vae = VAE.__new__(VAE)
    vae.first_stage_model = model
    vae.crop_input = False
    vae.output_channels = 2
    vae.latent_dim = 2
    vae.process_input = lambda value: value
    vae.vae_dtype = torch.float32
    vae.device = vae.output_device = torch.device("cpu")
    vae.audio_sample_rate = 32000
    vae.memory_used_encode = lambda shape, dtype: 1
    vae.patcher = SimpleNamespace(get_free_memory=lambda device: 1)
    vae.disable_offload = False
    vae.vae_output_dtype = lambda: torch.float32
    vae.format_encoded = None
    monkeypatch.setattr(model_management, "load_models_gpu", lambda *a, **k: None)
    pcm = torch.linspace(-0.5, 0.5, 320801).view(1, 1, -1).repeat(1, 2, 1)
    direct = torch.cat([model.encode(pcm[..., :320000]), model.encode(pcm[..., 320000:])], dim=-1)
    path = tmp_path / "core_audio.safetensors"
    asset(path, "audio", direct)
    package, report = audio(path, pcm, vae, encoding_profile="author_10s_native_exact")
    assert torch.equal(package.tensors["voice_latent"], direct)
    assert report["native_encode_chunks"][1]["right_zero_padding_samples"] == 799
    assert not torch.cuda.is_initialized()
