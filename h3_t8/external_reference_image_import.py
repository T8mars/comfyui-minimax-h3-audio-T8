"""Explicit one-image import validated against an actual fresh native RGB encode.

Only selected reference bytes are loaded. Foreign configuration, source paths,
LoRA weights and writer/encoder claims are never trusted or executed.
"""
from pathlib import Path
import struct

from safetensors import safe_open
import torch

from .reference_package import (IDENTIFIER, MAX_BYTES, NORMALIZATION, SHA, _unlinked,
                                capture_package, file_sha, tensor_record)
from .reference_runtime import portable_producer
from .tools.inspect_h3_external_reference import MAX_HEADER, inspect_header, plain_json


PROFILES = ("native_exact", "author_encode_fp16")


def _signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def import_image(path, source_image, video_vae, *, expected_sha256, member_ordinal=1,
                 role_id="A", precision_profile="native_exact", confirm_import=False):
    if confirm_import is not True:
        raise ValueError("Explicit reference import/source-use confirmation is required")
    if not isinstance(expected_sha256, str) or not SHA.fullmatch(expected_sha256):
        raise ValueError("Pin the entire external file with its actual SHA256")
    if (precision_profile not in PROFILES or not isinstance(role_id, str)
            or not IDENTIFIER.fullmatch(role_id) or type(member_ordinal) is not int):
        raise ValueError("Choose an explicit precision profile, role and integer member ordinal")
    path = Path(path)
    _unlinked(path)
    signature = _signature(path)
    header = inspect_header(path)
    if not 1 <= member_ordinal <= len(header["members"]):
        raise ValueError("Selected external reference ordinal does not exist")
    row = header["members"][member_ordinal - 1]
    if row["kind"] != "image":
        raise ValueError("Only one image is qualified; video/audio need their own source/clock contracts")
    if row["bytes"] > MAX_BYTES:
        raise ValueError("Selected image latent exceeds the bounded CPU asset budget")
    if file_sha(path) != expected_sha256 or _signature(path) != signature:
        raise ValueError("External reference file changed or differs from the selected SHA256")
    # Read only the already bounded header, not embedded sidecar/source paths.
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= length <= MAX_HEADER:
            raise ValueError("External reference header changed")
        metadata = plain_json(stream.read(length))["__metadata__"]
    bundle = plain_json(metadata["refmod_meta"])
    member = bundle if header["format_version"] == 4 else bundle["members"][member_ordinal - 1]
    if member.get("mode") != "encode":
        raise ValueError("Training/pooled/unknown references are not an unmodified Encode image")
    h, w = row["shape"][-2] * 16, row["shape"][-1] * 16
    if (type(source_image) is not torch.Tensor or source_image.dtype != torch.float32
            or list(source_image.shape) != [1, h, w, 3]
            or source_image.numel() * source_image.element_size() + row["bytes"] > MAX_BYTES):
        raise ValueError("Supply the actual single RGB frame at the encoded geometry; no implicit resize/crop")
    rgb = source_image.detach().cpu().contiguous().clone()
    source = tensor_record(rgb)
    if not bool(((rgb >= 0) & (rgb <= 1)).all()):
        raise ValueError("Source RGB must be finite float32 in 0..1")
    with safe_open(path, framework="pt", device="cpu") as stream:
        # Never materialize another member or the ordinary LoRA half.
        latent = stream.get_tensor(row["tensor_key"]).clone()
    external = tensor_record(latent)
    if (external["shape"] != row["shape"]
            or external["dtype"] != {"F16": "torch.float16", "BF16": "torch.bfloat16", "F32": "torch.float32"}[row["dtype"]]):
        raise ValueError("External latent differs from the inspected descriptor")
    before = portable_producer(video_vae, "video_vae")
    # Validate normalization/source using the real connected native VAE, not a
    # caller label or a pre-existing package's self-declared producer field.
    encoded = video_vae.encode(rgb).detach().cpu().contiguous()
    native = tensor_record(encoded)
    expected = encoded if precision_profile == "native_exact" else encoded.to(torch.float16)
    expected_record = tensor_record(expected)
    if native["shape"] != row["shape"] or external != expected_record:
        raise ValueError("External image does not exactly match this RGB/VAE/precision profile; no guessed conversion")
    if portable_producer(video_vae, "video_vae") != before or tensor_record(rgb) != source:
        raise ValueError("Actual VAE or source RGB changed during validation")
    _unlinked(path)
    if _signature(path) != signature or file_sha(path) != expected_sha256:
        raise ValueError("External file changed during import")
    package = capture_package([{
        "id": "visual", "role_id": role_id, "kind": "image", "latent": "visual_latent",
        "grounding": "visual_rgb", "source": {"sha256": source["sha256"], "scope": "actual_decoded_rgb_tensor"},
        "producer": before, "normalization": NORMALIZATION,
    }], {"visual_latent": latent, "visual_rgb": rgb})
    report = {
        "schema": "t8.community-reference.image-import.v1", "conversion_performed": True,
        "format_version": header["format_version"], "external_file_sha256": expected_sha256,
        "external_header_sha256": header["header_sha256"], "selected_member_ordinal": member_ordinal,
        "selected_tensor_key": row["tensor_key"], "precision_profile": precision_profile,
        "external_latent_bytes_preserved": True, "native_encode_bytes_exact": external == native,
        "fresh_source_reencoded_for_validation": True, "validation_producer": before,
        "original_RGB_sha256": source["sha256"], "external_latent_sha256": external["sha256"],
        "native_encode_sha256": native["sha256"], "manifest_sha256": package.verify()["sha256"],
        "tensor_loads": 1, "nonselected_members_loaded": False, "LoRA_weights_loaded_or_applied": False,
        "source_payload_modified": False, "configuration_executed": False, "embedded_paths_followed": False,
        "external_writer_or_encoder_authenticated": False, "source_use_confirmed_by_caller": True,
        "source_rights_independently_certified": False, "sampling_executed": False,
        "package_saved": False, "automatic_accept": False,
    }
    return package, report
