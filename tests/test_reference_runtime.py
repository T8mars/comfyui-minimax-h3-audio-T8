"""Synthetic adapters + actual Core layouts; not pretrained generation evidence."""
import hashlib
import json

import pytest
import torch

from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE, make_audio
from h3_audio_t8_pkg import reference_runtime as runtime
from h3_audio_t8_pkg.conditioning import build_conditioning
from h3_audio_t8_pkg.nodes_reference_package import (
    MiniMaxH3ReferenceSaveEXPT8, MiniMaxH3ReferenceLoadEXPT8,
    MiniMaxH3ReferenceConditioningEXPT8, NODES,
)
from h3_audio_t8_pkg.reference_package import capture_package, tensor_record, NORMALIZATION


@pytest.fixture
def encoders(monkeypatch):
    video, audio = FakeVideoVAE(), FakeAudioVAE()

    def synthetic_identity(value, role):
        assert value in (video, audio)
        return {"role": role, "scope": runtime.PRODUCER_SCOPE,
                "sha256": hashlib.sha256((role+str(getattr(value, "revision", 0))).encode()).hexdigest()}

    monkeypatch.setattr(runtime, "native_producer_identity", synthetic_identity)
    return video, audio


def image(role, video, audio, *, voice=False):
    rgb = torch.full((1, 32, 32, 3), .25 if role == "A" else .75)
    return runtime.create_package(role, frames=rgb, width=32, height=32,
        audio=make_audio(.1) if voice else None, video_vae=video, audio_vae=audio)


class CountingClip(FakeClip):
    def __init__(self):
        super().__init__()
        self.encode_count = 0

    def encode_from_tokens_scheduled(self, tokens):
        self.encode_count += 1
        return super().encode_from_tokens_scheduled(tokens)


def test_capture_actual_encode_once_keeps_source_and_reports_effective_resample(encoders):
    video, audio = encoders
    rgb = torch.full((1, 32, 32, 3), .25)
    source = make_audio(.1, sample_rate=48000)
    original = source["waveform"].clone()
    package, report = runtime.create_package("A", frames=rgb, audio=source,
        width=32, height=32, video_vae=video, audio_vae=audio)
    manifest = package.verify()
    assert len(video.encode_calls) == len(audio.encode_calls) == 1
    assert torch.equal(source["waveform"], original) and torch.all(rgb == .25)
    assert report["voice_samples"] == 3200 and report["voice_input_sample_rate"] == 48000
    assert report["encoded_sample_rate"] == 32000 and report["sampling_executed"] is False
    assert all(member["normalization"] == NORMALIZATION for member in manifest["members"])
    assert manifest["members"][0]["source"]["sha256"] == tensor_record(package.tensors["visual_rgb"])["sha256"]
    assert torch.equal(package.tensors["voice_latent"], torch.full((1, 32, 2, 4), .25))


def test_fresh_apply_removes_both_absent_visual_paths_retains_voice_and_encodes_qwen_once(encoders):
    video, audio = encoders
    a, _ = image("A", video, audio, voice=True)
    b, _ = image("B", video, audio)
    refs, mapping = runtime.capture_set([a, b], json.dumps([
        {"role_id": "A", "visual": False, "voice": True},
        {"role_id": "B", "visual": True, "voice": False}]))
    clip = CountingClip()
    output = MiniMaxH3ReferenceConditioningEXPT8.execute(refs, clip, video, audio,
        "<Picture 1> listens to <Audio 1>.", 32, 32, 124)
    positive, latent, mux_audio, prompt, media, report, recipe = output.result
    assert len(video.encode_calls) == 2 and len(audio.encode_calls) == 1
    assert len(clip.tokenize_calls) == clip.encode_count == 1 and mux_audio is None
    blocks = positive[0][1]["minimax_refs"]
    assert [block["kind"] for block in blocks] == ["audio", "image"]
    assert blocks[0]["audio_latent"] is a.tensors["voice_latent"]
    assert blocks[1]["latent"] is b.tensors["visual_latent"]
    items = clip.tokenize_calls[0][1]["minimax_ref_items"]
    assert [item["type"] for item in items] == ["audio", "image"]
    assert items[1]["data"] is b.tensors["visual_rgb"]
    assert all(item.get("data") is not a.tensors["visual_rgb"] for item in items)
    assert json.loads(media)["pictures"] == {"1": "role:B/visual"}
    assert recipe.metadata["minimax_refs"] == blocks and recipe.counts == {"pictures": 1, "videos": 0, "audios": 1}
    assert "fresh_Qwen_encoding" in report and prompt == "<Picture 1> listens to <Audio 1>."
    assert mapping["packed_reference_rows"] == 9 and latent["samples"].is_nested
    from comfy.ldm.minimax.model import PackedLayout
    layout = PackedLayout(4, 2, 2, 2, 1, refs=blocks)
    assert [(stop-start, kind) for start, stop, kind in layout.segments][1:3] == [(8, "ref_audio"), (1, "ref_img")]


def test_changed_producer_or_asset_rejects_before_qwen_no_ordinary_loader_ban(encoders):
    video, audio = encoders
    a, _ = image("A", video, audio)
    refs, _ = runtime.capture_set([a], '[{"role_id":"A","visual":true,"voice":false}]')
    clip = FakeClip()
    args = dict(clip=clip, video_vae=video, audio_vae=audio, prompt="<Picture 1>",
        width=32, height=32, length=124, audio_mode="native", prepared_reference_set=refs)
    video.revision = 1
    with pytest.raises(ValueError, match="encoder differs"):
        build_conditioning(**args)
    assert not clip.tokenize_calls
    del video.revision
    a.tensors["visual_latent"][0, 0, 0, 0, 0] = .2
    with pytest.raises(ValueError, match="content changed"):
        build_conditioning(**args)
    assert not clip.tokenize_calls
    # Unknown ordinary FakeVAE remains allowed on the old, uncached path.
    args.pop("prepared_reference_set")
    build_conditioning(**args, ref_images={"ref_image_1": torch.zeros((1, 32, 32, 3))})
    assert len(clip.tokenize_calls) == 1


def test_explicit_video_alignment_and_native_two_fps_qwen_samples(encoders):
    video, audio = encoders
    frames = torch.full((24, 32, 32, 3), .5)
    package, report = runtime.create_package("V", kind="video", frames=frames,
        width=32, height=32, frame_limit=22, video_vae=video)
    assert report["used_frames"] == 22 and report["trimmed_frames"] == 2
    assert package.tensors["visual_latent"].shape == (1, 24, 7, 2, 2)
    refs, _ = runtime.capture_set([package], '[{"role_id":"V","visual":true,"voice":false}]')
    inputs = refs.conditioning_inputs(video, audio)
    item = inputs["qwen_ref_items"][0]
    assert item["timestamps"] == [0., .5]
    assert torch.equal(item["data"], package.tensors["visual_rgb"][[0, 12]])
    with pytest.raises(ValueError, match="short ASCII"):
        runtime.create_package("../bad", frames=frames, video_vae=video)
    with pytest.raises(ValueError, match="exactly one"):
        runtime.create_package("A", frames=frames, video_vae=video)


def test_actual_native_producer_hash_used_and_wrong_or_opaque_owners_not_certified():
    from test_progressive_producers import component
    video = component("video_vae")
    producer = runtime.portable_producer(video, "video_vae")
    rgb = torch.zeros((1, 32, 32, 3))
    latent = torch.zeros((1, 24, 1, 2, 2))
    package = capture_package([{"id": "visual", "role_id": "A", "kind": "image",
        "latent": "latent", "grounding": "rgb", "normalization": NORMALIZATION,
        "producer": producer, "source": {"sha256": tensor_record(rgb)["sha256"],
                                            "scope": "actual_decoded_rgb_tensor"}}], {"latent": latent, "rgb": rgb})
    refs, _ = runtime.capture_set([package], '[{"role_id":"A","visual":true,"voice":false}]')
    assert refs.conditioning_inputs(video, None)["actual_producers"]["video_vae"] == producer
    with torch.no_grad():
        video.first_stage_model.probe.add_(1)
    with pytest.raises(ValueError, match="encoder differs"):
        refs.conditioning_inputs(video, None)
    video.first_stage_model.forward = lambda *args: None
    with pytest.raises(ValueError, match="ordinary native refs"):
        runtime.portable_producer(video, "video_vae")
    with pytest.raises(ValueError, match="native CLIP"):
        runtime.portable_producer(FakeVideoVAE(), "video_vae")


def test_new_node_schema_save_confirmation_and_actual_models_directory_load(tmp_path, monkeypatch, encoders):
    import folder_paths
    video, audio = encoders
    package, _ = image("A", video, audio)
    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path))
    monkeypatch.setattr(folder_paths, "folder_names_and_paths", {})
    monkeypatch.setattr(folder_paths, "filename_list_cache", {})
    schemas = [node.define_schema() for node in NODES]
    assert len(schemas) == 6 and len({item.node_id for item in schemas}) == 6
    assert all(item.is_experimental for item in schemas)
    for node, item in zip(NODES, schemas, strict=True):
        item.get_v1_info(node)
    output = MiniMaxH3ReferenceSaveEXPT8.execute(package)
    assert output.result[1:3] == ("", "") and not (tmp_path / "refmods").exists()
    saved = MiniMaxH3ReferenceSaveEXPT8.execute(package, "A.safetensors", True).result
    restored = MiniMaxH3ReferenceLoadEXPT8.execute(saved[1], saved[2]).result[0]
    assert restored.manifest_json == package.manifest_json
    assert MiniMaxH3ReferenceLoadEXPT8.fingerprint_inputs(saved[1]) == saved[2]
    with pytest.raises(FileExistsError):
        MiniMaxH3ReferenceSaveEXPT8.execute(package, "A.safetensors", True)
    with pytest.raises(ValueError, match="short ASCII"):
        MiniMaxH3ReferenceSaveEXPT8.execute(package, "../A.safetensors", True)


def test_mid_encode_producer_change_never_publishes_asset(encoders, monkeypatch):
    video, audio = encoders
    original = video.encode

    def changed(values):
        output = original(values)
        video.revision = 1
        return output

    monkeypatch.setattr(video, "encode", changed)
    with pytest.raises(ValueError, match="changed during encoding"):
        image("A", video, audio)


def test_fresh_conditioning_detects_mid_qwen_encoder_change_and_keeps_master_audio_separate(encoders):
    video, audio = encoders
    package, _ = image("A", video, audio, voice=True)
    refs, _ = runtime.capture_set([package], '[{"role_id":"A","visual":false,"voice":true}]')
    drive, final = make_audio(.2, value=.2), make_audio(.2, value=.3)
    clip = CountingClip()
    args = dict(clip=clip, video_vae=video, audio_vae=audio, prompt="<Audio 1> performs.",
        width=32, height=32, length=124, audio_mode="reference_only",
        drive_audio=drive, final_audio=final, prepared_reference_set=refs,
        prompt_primary_audio_ordinal=1)
    output = build_conditioning(**args)
    assert output[2] is final and output[3] == "<Audio 2> performs."
    assert json.loads(output[4])["source_audio_ordinal"] == 2
    assert "noise_mask" not in output[1]  # Voice anchor never locks target or replaces master.
    assert len(audio.encode_calls) == 2 and clip.encode_count == 1

    class ChangedClip(CountingClip):
        def encode_from_tokens_scheduled(self, tokens):
            audio.revision = 1
            return super().encode_from_tokens_scheduled(tokens)

    with pytest.raises(ValueError, match="encoder differs"):
        build_conditioning(**{**args, "clip": ChangedClip()})
