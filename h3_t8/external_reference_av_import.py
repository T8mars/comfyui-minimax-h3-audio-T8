"""Explicit source-validated AV reference imports; no legacy loader changes.

Prepared RGB/PCM is supplied by the caller. Only a fresh native encode that
matches the selected external bytes is accepted; historical writer identity,
AV synchronization and source-use rights are not inferred from metadata.
"""
from collections.abc import Mapping
from pathlib import Path
import struct

from safetensors import safe_open
import torch

from .core import ensure_h3_audio_vae_non_aligned_crop_compat
from .external_reference_image_import import _signature
from .reference_package import (IDENTIFIER, MAX_BYTES, NORMALIZATION, SHA, _unlinked,
                                capture_package, file_sha, tensor_record)
from .reference_runtime import portable_producer
from .tools.inspect_h3_external_reference import MAX_HEADER, inspect_header, plain_json


VIDEO_PROFILES = ("native_exact", "author_encode_fp16")
AUDIO_PROFILES = ("native_exact", "author_10s_native_exact")
AUDIO_CHUNK_SAMPLES = 320000
MAX_AUDIO_SAMPLES = 30 * 32000


def _select(path, kind, expected_sha256, member_ordinal, role_id, confirm_import):
    if confirm_import is not True:
        raise ValueError("Explicit reference import/source-use confirmation is required")
    if not isinstance(expected_sha256, str) or not SHA.fullmatch(expected_sha256):
        raise ValueError("Pin the entire external file with its actual SHA256")
    if (not isinstance(role_id, str) or not IDENTIFIER.fullmatch(role_id)
            or type(member_ordinal) is not int):
        raise ValueError("Use an explicit role and integer member ordinal")
    path = Path(path)
    _unlinked(path)
    signature = _signature(path)
    header = inspect_header(path)
    if not 1 <= member_ordinal <= len(header["members"]):
        raise ValueError("Selected external reference ordinal does not exist")
    row = header["members"][member_ordinal - 1]
    if row["kind"] != kind or row["bytes"] > MAX_BYTES:
        raise ValueError("Selected reference kind or bounded latent size does not match this importer")
    if file_sha(path) != expected_sha256 or _signature(path) != signature:
        raise ValueError("External reference file changed or differs from the selected SHA256")
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= length <= MAX_HEADER:
            raise ValueError("External reference header changed")
        metadata = plain_json(stream.read(length))["__metadata__"]
    bundle = plain_json(metadata["refmod_meta"])
    member = bundle if header["format_version"] == 4 else bundle["members"][member_ordinal - 1]
    if member.get("mode") != "encode":
        raise ValueError("Training/pooled/unknown references are not an unmodified Encode asset")
    return path, signature, header, row


def _load_selected(path, row):
    with safe_open(path, framework="pt", device="cpu") as stream:
        latent = stream.get_tensor(row["tensor_key"]).clone()
    record = tensor_record(latent)
    dtype = {"F16": "torch.float16", "BF16": "torch.bfloat16", "F32": "torch.float32"}[row["dtype"]]
    if record["shape"] != row["shape"] or record["dtype"] != dtype:
        raise ValueError("External latent differs from the inspected descriptor")
    return latent, record


def _finish(path, signature, header, row, expected_sha256, role_id, profile,
            kind, latent, encoded, source, source_record, producer, component, extra):
    native = tensor_record(encoded)
    expected = encoded.to(torch.float16) if profile == "author_encode_fp16" else encoded
    external = tensor_record(latent)
    if external != tensor_record(expected):
        raise ValueError("External reference does not exactly match this source/VAE/encoding profile; no guessed conversion")
    role = "audio_vae" if kind == "audio" else "video_vae"
    if portable_producer(component, role) != producer or tensor_record(source) != source_record:
        raise ValueError("Actual VAE or supplied source changed during validation")
    _unlinked(path)
    if _signature(path) != signature or file_sha(path) != expected_sha256:
        raise ValueError("External file changed during import")
    audio = kind == "audio"
    tensors = {"voice_latent" if audio else "visual_latent": latent}
    if not audio:
        tensors["visual_rgb"] = source
    package = capture_package([{
        "id": "voice" if audio else "visual", "role_id": role_id, "kind": kind,
        "latent": "voice_latent" if audio else "visual_latent",
        "grounding": None if audio else "visual_rgb",
        "source": {"sha256": source_record["sha256"],
                   "scope": "actual_decoded_pcm_tensor" if audio else "actual_decoded_rgb_tensor"},
        "producer": producer, "normalization": NORMALIZATION,
    }], tensors)
    report = {
        "schema": "t8.community-reference.av-import.v1", "kind": kind,
        "conversion_performed": True, "format_version": header["format_version"],
        "external_file_sha256": expected_sha256, "external_header_sha256": header["header_sha256"],
        "selected_member_ordinal": row["ordinal"], "selected_tensor_key": row["tensor_key"],
        "encoding_profile": profile, "external_latent_bytes_preserved": True,
        "selected_native_encoding_bytes_exact": external == native,
        "whole_clip_native_encode_bit_parity_claimed": profile == "native_exact",
        "fresh_source_reencoded_for_validation": True, "validation_producer": producer,
        "source_tensor_sha256": source_record["sha256"], "source_tensor_shape": source_record["shape"],
        "external_latent_sha256": external["sha256"], "native_encode_sha256": native["sha256"],
        "manifest_sha256": package.verify()["sha256"], "tensor_loads": 1,
        "nonselected_members_loaded": False, "LoRA_weights_loaded_or_applied": False,
        "source_payload_modified": False, "configuration_executed": False,
        "embedded_paths_followed": False, "external_writer_or_encoder_authenticated": False,
        "source_use_confirmed_by_caller": True, "source_rights_independently_certified": False,
        "AV_synchronization_certified": False, "sampling_executed": False,
        "package_saved": False, "automatic_accept": False, **extra,
    }
    return package, report


def import_video(path, source_frames, video_vae, *, expected_sha256, member_ordinal=1,
                 role_id="A", precision_profile="native_exact", source_fps=24.0, confirm_import=False):
    if precision_profile not in VIDEO_PROFILES:
        raise ValueError("Choose an explicit video precision profile")
    if type(source_fps) not in (int, float) or source_fps != 24:
        raise ValueError("Supply explicitly prepared 24fps frames; no implicit resampling or clock conversion")
    path, signature, header, row = _select(path, "video", expected_sha256, member_ordinal, role_id, confirm_import)
    t, h, w = row["shape"][2:]
    if t < 2 or (t - 2) % 5:
        raise ValueError("External video does not use the native 17n+5 grounding temporal grid")
    count = (t - 2) // 5 * 17 + 5
    if (type(source_frames) is not torch.Tensor or source_frames.dtype != torch.float32
            or list(source_frames.shape) != [count, h * 16, w * 16, 3] or count > 360
            or source_frames.numel() * source_frames.element_size() + row["bytes"] > MAX_BYTES):
        raise ValueError("Supply actual prepared RGB at exactly the encoded geometry/frame count within the byte budget")
    rgb = source_frames.detach().cpu().contiguous().clone()
    source = tensor_record(rgb)
    if not bool(((rgb >= 0) & (rgb <= 1)).all()):
        raise ValueError("Source RGB must be finite float32 in 0..1")
    latent, _ = _load_selected(path, row)
    producer = portable_producer(video_vae, "video_vae")
    encoded = video_vae.encode(rgb).detach().cpu().contiguous()
    return _finish(path, signature, header, row, expected_sha256, role_id, precision_profile,
        "video", latent, encoded, rgb, source, producer, video_vae, {
            "prepared_frames": count, "prepared_fps": 24,
            "prepared_clock_evidence": "caller_declared_not_source_PTS_authenticated",
            "source_time_origin_inferred": False, "temporal_grid": "17n+5",
            "resized_or_cropped_or_trimmed_or_resampled": False,
            "video_audio_track_imported": False,
        })


def import_audio(path, source_audio, audio_vae, *, expected_sha256, member_ordinal=1,
                 role_id="A", encoding_profile="native_exact", confirm_import=False):
    if encoding_profile not in AUDIO_PROFILES:
        raise ValueError("Choose an explicit audio encoding profile")
    path, signature, header, row = _select(path, "audio", expected_sha256, member_ordinal, role_id, confirm_import)
    if not isinstance(source_audio, Mapping):
        raise ValueError("Supply actual prepared stereo 32000Hz AUDIO")
    waveform, rate = source_audio.get("waveform"), source_audio.get("sample_rate")
    if (type(waveform) is not torch.Tensor or waveform.dtype != torch.float32 or waveform.ndim != 3
            or tuple(waveform.shape[:2]) != (1, 2) or type(rate) is not int or rate != 32000
            or not 1 <= waveform.shape[-1] <= MAX_AUDIO_SAMPLES
            or waveform.numel() * waveform.element_size() + row["bytes"] > MAX_BYTES):
        raise ValueError("Supply actual prepared float32 stereo 32000Hz PCM, at most 30s; no implicit resample/trim/mono duplication")
    length = waveform.shape[-1]
    if row["shape"][-1] != (length + 799) // 800:
        raise ValueError("PCM length and external audio latent disagree; truncated/pooled references are not inferred")
    pcm = waveform.detach().cpu().contiguous().clone()
    source = tensor_record(pcm)
    if not bool(((pcm >= -1) & (pcm <= 1)).all()):
        raise ValueError("Source PCM must be finite float32 in -1..1")
    latent, _ = _load_selected(path, row)
    # Establish the existing H3-specific non-aligned-tail compatibility before
    # producer capture, not halfway through an encode. Other VAE types stay untouched.
    ensure_h3_audio_vae_non_aligned_crop_compat(audio_vae)
    producer = portable_producer(audio_vae, "audio_vae")
    if getattr(audio_vae, "audio_sample_rate", None) != 32000:
        raise ValueError("Connected audio VAE must use the actual 32000Hz native codec")
    chunk = length if encoding_profile == "native_exact" else AUDIO_CHUNK_SAMPLES
    pieces, spans = [], []
    for start in range(0, length, chunk):
        stop = min(length, start + chunk)
        encoded = audio_vae.encode(pcm[..., start:stop].movedim(1, -1)).detach().cpu().contiguous()
        if list(encoded.shape) != [1, 32, 2, (stop - start + 799) // 800]:
            raise ValueError("Native audio encoder changed the selected PCM chunk geometry")
        tensor_record(encoded)
        pieces.append(encoded)
        spans.append({"sample_start": start, "sample_stop": stop,
                      "right_zero_padding_samples": (-(stop - start)) % 800})
    encoded = torch.cat(pieces, dim=-1)
    return _finish(path, signature, header, row, expected_sha256, role_id, encoding_profile,
        "audio", latent, encoded, pcm, source, producer, audio_vae, {
            "sample_rate": 32000, "stereo_channels": 2, "source_samples": length,
            "native_encode_chunks": spans, "native_encode_calls": len(spans),
            "resampled_or_trimmed_or_mono_duplicated": False,
            "PCM_stored_in_package": False, "audio_usage": "voice_reference_not_final_track",
            "whole_clip_vs_chunked_boundary_equivalence_claimed": False,
        })
