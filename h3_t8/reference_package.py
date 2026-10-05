"""Bounded T8 reference assets, not LoRA, a trained identity, or old RefMod import.

The first format stores unmodified native VAE.encode latents and actual RGB
grounding. Producer/source fields are content declarations, not signatures:
the future create/apply adapter must bind actual loaded native producers. No
pickle, external path, dynamic import, pooling, strength curve or global patch.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from .prepared_identity import linked


SCHEMA = "t8.h3.reference-package/v1"
META_KEY = "t8_reference_package"
NORMALIZATION = "native_comfy_H3_VAE_encode_output_not_denoiser_scaled/v1"
MAX_BYTES = 256 * 1024**2
MAX_HEADER = 1024**2
MAX_MEMBERS = 15
SHA = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def tensor_record(value):
    if (type(value) is not torch.Tensor or value.device.type != "cpu"
            or value.dtype not in (torch.float32, torch.float16, torch.bfloat16)
            or value.numel() * value.element_size() > MAX_BYTES
            or not bool(torch.isfinite(value).all())):
        raise ValueError("Reference tensors must be bounded finite CPU floating values")
    raw = value.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def _plain_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate reference metadata field")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))


def _sha(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError("Reference source/producer requires an explicit content SHA256")


def _manifest_structure(manifest):
    if (type(manifest) is not dict or set(manifest) != {"schema", "members", "tensors", "sha256"}
            or manifest["schema"] != SCHEMA or type(manifest["members"]) is not list
            or not 0 < len(manifest["members"]) <= MAX_MEMBERS or type(manifest["tensors"]) is not dict):
        raise ValueError("Unsupported reference package format; old/community RefMods require a separate qualified importer")
    if len(canonical(manifest).encode()) > MAX_HEADER:
        raise ValueError("Reference manifest exceeds its metadata budget")
    _sha(manifest["sha256"])
    unsigned = {key: value for key, value in manifest.items() if key != "sha256"}
    if hashlib.sha256(canonical(unsigned).encode()).hexdigest() != manifest["sha256"]:
        raise ValueError("Reference manifest content binding changed")
    names, required = set(), set()
    total = 0
    for member in manifest["members"]:
        if (type(member) is not dict or set(member) != {"id", "role_id", "kind", "latent", "grounding", "source", "producer", "normalization"}
                or not isinstance(member["id"], str) or not IDENTIFIER.fullmatch(member["id"])
                or member["id"] in names or not isinstance(member["role_id"], str)
                or not IDENTIFIER.fullmatch(member["role_id"]) or member["kind"] not in ("image", "video", "audio")
                or member["normalization"] != NORMALIZATION):
            raise ValueError("Invalid reference member or normalization contract")
        names.add(member["id"])
        source, producer = member["source"], member["producer"]
        if (type(source) is not dict or set(source) != {"sha256", "scope"}
                or source["scope"] not in ("actual_decoded_rgb_tensor", "actual_decoded_pcm_tensor")
                or type(producer) is not dict or set(producer) != {"role", "sha256", "scope"}
                or producer["role"] != ("audio_vae" if member["kind"] == "audio" else "video_vae")
                or producer["scope"] != "actual_native_weights_tokenizer_configuration_and_implementation"):
            raise ValueError("Missing or unsupported actual source/producer declaration")
        if source["scope"] != ("actual_decoded_pcm_tensor" if member["kind"] == "audio" else "actual_decoded_rgb_tensor"):
            raise ValueError("Source media type differs from reference latent")
        _sha(source["sha256"])
        _sha(producer["sha256"])
        if not isinstance(member["latent"], str) or not IDENTIFIER.fullmatch(member["latent"]):
            raise ValueError("Invalid reference tensor name")
        if member["latent"] in required:
            raise ValueError("Reference members cannot secretly repeat one tensor as extra strength")
        required.add(member["latent"])
        grounding = member["grounding"]
        if member["kind"] == "audio" and grounding is not None:
            raise ValueError("Standalone audio uses an audio tag, not visual grounding")
        if grounding is not None:
            if not isinstance(grounding, str) or not IDENTIFIER.fullmatch(grounding):
                raise ValueError("Grounding must name one internal tensor, never a path")
            if grounding in required:
                raise ValueError("Reference members cannot share ambiguous grounding ordinals")
            required.add(grounding)
    if set(manifest["tensors"]) != required:
        raise ValueError("Reference tensor membership differs from declared members")
    for record in manifest["tensors"].values():
        if (type(record) is not dict or set(record) != {"shape", "dtype", "bytes", "sha256"}
                or type(record["shape"]) is not list or not record["shape"] or len(record["shape"]) > 5
                or any(type(n) is not int or not 0 < n <= 16384 for n in record["shape"])
                or record["dtype"] not in ("torch.float16", "torch.bfloat16", "torch.float32")
                or type(record["bytes"]) is not int or not 0 < record["bytes"] <= MAX_BYTES):
            raise ValueError("Invalid bounded reference tensor descriptor")
        elements = 1
        for n in record["shape"]:
            elements *= n
        if elements * (4 if record["dtype"] == "torch.float32" else 2) != record["bytes"]:
            raise ValueError("Reference shape/dtype/byte count disagree")
        _sha(record["sha256"])
        total += record["bytes"]
    if total > MAX_BYTES:
        raise ValueError("Reference package exceeds its CPU/RAM byte budget")


@dataclass(frozen=True)
class ReferencePackage:
    manifest_json: str
    tensors: dict
    file_sha256: str | None = None

    def verify(self, *, allow_missing_grounding=False):
        if type(allow_missing_grounding) is not bool:
            raise ValueError("Latent-only EXP needs an explicit boolean choice")
        if not isinstance(self.manifest_json, str) or len(self.manifest_json.encode()) > MAX_HEADER:
            raise ValueError("Reference manifest exceeds its metadata budget")
        manifest = _plain_json(self.manifest_json)
        _manifest_structure(manifest)
        if set(self.tensors) != set(manifest["tensors"]):
            raise ValueError("Reference tensors changed after capture")
        for name, record in manifest["tensors"].items():
            if tensor_record(self.tensors[name]) != record:
                raise ValueError("Reference tensor content changed: " + name)
        for member in manifest["members"]:
            latent = self.tensors[member["latent"]]
            if member["kind"] == "audio":
                if latent.ndim != 4 or tuple(latent.shape[:3]) != (1, 32, 2):
                    raise ValueError("Audio reference requires [1,32,2,T]")
                continue
            if (latent.ndim != 5 or tuple(latent.shape[:2]) != (1, 24)
                    or latent.shape[-2] % 2 or latent.shape[-1] % 2
                    or (member["kind"] == "image" and latent.shape[2] != 1)
                    or (member["kind"] == "video" and (latent.shape[2] < 2 or (latent.shape[2] - 2) % 5))):
                raise ValueError("Visual reference requires native [1,24,T,even-H,even-W]")
            if member["grounding"] is None:
                if not allow_missing_grounding:
                    raise ValueError("Missing Qwen RGB grounding; only separately explicit latent-only EXP may bypass this field")
                continue
            frames = self.tensors[member["grounding"]]
            if (frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != torch.float32
                    or not bool(((frames >= 0) & (frames <= 1)).all())
                    or tuple(frames.shape[1:3]) != (latent.shape[-2] * 16, latent.shape[-1] * 16)
                    or frames.shape[0] > 360
                    or (member["kind"] == "image" and frames.shape[0] != 1)
                    or (member["kind"] == "video" and (frames.shape[0] < 5
                        or (frames.shape[0] - 5) % 17 or latent.shape[2] != (frames.shape[0] - 5) // 17 * 5 + 2))):
                raise ValueError("Actual grounding dimensions/temporal grid disagree with native reference")
            if tensor_record(frames)["sha256"] != member["source"]["sha256"]:
                raise ValueError("RGB grounding differs from declared actual decoded source")
        return manifest


def capture_package(members, tensors, *, allow_missing_grounding=False):
    # Snapshot caller-owned CPU assets; no input mutation or hidden compression.
    if type(tensors) is not dict or not 0 < len(tensors) <= 2 * MAX_MEMBERS:
        raise ValueError("Use a bounded CPU reference tensor dictionary")
    records = {name: tensor_record(value) for name, value in tensors.items()}
    if sum(record["bytes"] for record in records.values()) > MAX_BYTES:
        raise ValueError("Reference package exceeds its CPU/RAM byte budget")
    values = {name: value.detach().cpu().contiguous().clone() for name, value in tensors.items()}
    manifest = {"schema": SCHEMA, "members": deepcopy(members),
                "tensors": records}
    manifest["sha256"] = hashlib.sha256(canonical(manifest).encode()).hexdigest()
    package = ReferencePackage(canonical(manifest), values)
    package.verify(allow_missing_grounding=allow_missing_grounding)
    return package


def register_reference_directory():
    """Add the normal models/refmods directory without replacing configured extras."""
    import folder_paths
    folder_paths.add_model_folder_path("refmods", str(Path(folder_paths.models_dir) / "refmods"))
    paths, extensions = folder_paths.folder_names_and_paths["refmods"]
    folder_paths.folder_names_and_paths["refmods"] = (paths, set(extensions) | {".safetensors"})


def installed_package_names():
    import folder_paths
    register_reference_directory()
    return folder_paths.get_filename_list("refmods")


def resolve_installed_package(selection):
    import folder_paths
    if (not isinstance(selection, str) or not selection or len(selection) > 4096
            or Path(selection).is_absolute() or ":" in selection
            or any(part in ("", ".", "..") for part in selection.replace("\\", "/").split("/"))):
        raise ValueError("Choose one exact installed refmods file, not an absolute/traversal path")
    if selection not in installed_package_names():
        raise FileNotFoundError("Selected reference package is no longer in configured refmods directories")
    path = Path(folder_paths.get_full_path_or_raise("refmods", selection))
    _unlinked(path)
    if not any(path.resolve().is_relative_to(Path(root).resolve()) for root in folder_paths.get_folder_paths("refmods")):
        raise ValueError("Reference file escaped configured directories")
    return path


def _unlinked(path):
    current = Path(path)
    while True:
        if current.exists() and linked(current):
            raise ValueError("Reference package path contains a symlink/reparse point")
        if current == current.parent:
            return
        current = current.parent


def save_package(package, path, *, confirmed=False, allow_missing_grounding=False):
    if confirmed is not True:
        raise ValueError("Explicit new reference file save confirmation is required")
    if type(package) is not ReferencePackage:
        raise ValueError("Use a verified T8 reference package")
    manifest = package.verify(allow_missing_grounding=allow_missing_grounding)
    # Do not let another consumer change a mutable tensor between verification
    # and publication. This invocation owns the exact snapshot it writes.
    values = {name: value.detach().contiguous().clone() for name, value in package.tensors.items()}
    ReferencePackage(canonical(manifest), values).verify(allow_missing_grounding=allow_missing_grounding)
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or path.suffix != ".safetensors" or len(path.name) > 64:
        raise ValueError("Use an explicit absolute short .safetensors destination")
    _unlinked(path)
    if path.exists():
        raise FileExistsError("Reference Save never replaces an existing user package")
    path.parent.mkdir(parents=True, exist_ok=True)
    _unlinked(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=".t8-ref-", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        save_file(values, temporary, metadata={META_KEY: canonical(manifest)})
        with open(temporary, "r+b") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic new-file only; racing same name fails.
    finally:
        Path(temporary).unlink(missing_ok=True)  # Only this invocation's mkstemp.
    return {"path": str(path), "sha256": file_sha(path), "manifest_sha256": manifest["sha256"],
            "saved": True, "automatic_accept": False}


def load_package(path, *, expected_sha256=None, allow_missing_grounding=False):
    path = Path(path)
    _unlinked(path)
    if not path.is_file() or not 10 <= path.stat().st_size <= MAX_BYTES + MAX_HEADER + 8:
        raise ValueError("Reference package missing or exceeds its file budget")
    actual_sha = file_sha(path)
    if expected_sha256 is not None:
        _sha(expected_sha256)
        if actual_sha != expected_sha256:
            raise ValueError("Selected reference package SHA changed")
    # Reject allocations and metadata before safe_open materializes tensors.
    with path.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= length <= MAX_HEADER or length + 8 > path.stat().st_size:
            raise ValueError("Invalid bounded reference safetensors header")
        header = _plain_json(stream.read(length))
    if type(header) is not dict or type(header.get("__metadata__")) is not dict:
        raise ValueError("Missing T8 reference package metadata")
    raw = header["__metadata__"].get(META_KEY)
    if not isinstance(raw, str) or len(raw.encode()) > MAX_HEADER:
        raise ValueError("Unsupported reference metadata; community format is not implicitly adopted")
    manifest = _plain_json(raw)
    _manifest_structure(manifest)
    if set(header) - {"__metadata__"} != set(manifest["tensors"]):
        raise ValueError("Unexpected tensor names in reference package")
    dtype_names = {"torch.float32": "F32", "torch.float16": "F16", "torch.bfloat16": "BF16"}
    for name, record in manifest["tensors"].items():
        item = header[name]
        if (type(item) is not dict or item.get("shape") != record["shape"]
                or item.get("dtype") != dtype_names[record["dtype"]]
                or type(item.get("data_offsets")) is not list or len(item["data_offsets"]) != 2
                or any(type(v) is not int for v in item["data_offsets"])
                or not 0 <= item["data_offsets"][0] <= item["data_offsets"][1] <= path.stat().st_size - 8 - length
                or item["data_offsets"][1] - item["data_offsets"][0] != record["bytes"]):
            raise ValueError("Reference safetensors shape/dtype/offset differs from manifest")
    with safe_open(path, framework="pt", device="cpu") as stream:
        values = {name: stream.get_tensor(name).clone() for name in manifest["tensors"]}
    if file_sha(path) != actual_sha:
        raise ValueError("Reference file changed while loading")
    package = ReferencePackage(canonical(manifest), values, actual_sha)
    package.verify(allow_missing_grounding=allow_missing_grounding)
    return package


def route_packages(packages, roles, *, allow_missing_grounding=False):
    """Select both visual paths before fresh Qwen encode, preserving voice-only.

    This returns inputs for a future conditioner, not a patch to previously
    encoded CONDITIONING. It cannot remove pictures already baked into Qwen.
    """
    if (type(roles) is not list or not roles or len(roles) > MAX_MEMBERS
            or any(type(role) is not dict or set(role) != {"role_id", "visual", "voice"}
                   or type(role["visual"]) is not bool or type(role["voice"]) is not bool for role in roles)):
        raise ValueError("Use an explicit ordered presence/voice list")
    ids = [role["role_id"] for role in roles]
    if any(not isinstance(value, str) or not IDENTIFIER.fullmatch(value) for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Role IDs must be unique and explicit")
    if type(packages) is not list or not 0 < len(packages) <= MAX_MEMBERS:
        raise ValueError("Use a bounded ordered package list")
    by_role, seen = {}, set()
    for package in packages:
        if type(package) is not ReferencePackage:
            raise ValueError("Unknown reference package provider")
        manifest = package.verify(allow_missing_grounding=allow_missing_grounding)
        package_roles = {member["role_id"] for member in manifest["members"]}
        if package_roles & seen:
            raise ValueError("Role belongs to multiple packages; no guessed merge/duplicate-strength")
        seen.update(package_roles)
        for member in manifest["members"]:
            by_role.setdefault(member["role_id"], []).append((member, package))
    if set(ids) != set(by_role):
        raise ValueError("Declare presence/voice for every package role, including absent roles")
    refs, items, mapping = [], [], []
    counts = {"image": 0, "video": 0, "audio": 0}
    for role in roles:
        for member, package in by_role[role["role_id"]]:
            kind = member["kind"]
            if not role["voice" if kind == "audio" else "visual"]:
                continue  # Drops VAE block AND Qwen grounding together, not strength=0.
            latent = package.tensors[member["latent"]]
            counts[kind] += 1
            if counts[kind] > {"image": 9, "video": 3, "audio": 3}[kind]:
                raise ValueError("Selected reference set exceeds native per-type media limits")
            if kind == "audio":
                block = {"kind": "audio", "ref_audio_t": int(latent.shape[-1]), "audio_latent": latent}
                item = {"type": "audio"}
                cost = 2 * int(latent.shape[-1])
            else:
                block = {"kind": kind, "latent": latent, "latent_h": int(latent.shape[-2]), "latent_w": int(latent.shape[-1])}
                if kind == "video":
                    block.update(latent_t=int(latent.shape[2]), ref_audio_t=0, audio_latent=None)
                frames = package.tensors.get(member["grounding"])
                item = {"type": kind, "data": frames} if frames is not None else None
                cost = int(latent.shape[2] * (latent.shape[-2] // 2) * (latent.shape[-1] // 2))
            refs.append(block)
            items.append(item)
            mapping.append({"role_id": role["role_id"], "member_id": member["id"], "kind": kind,
                            "ordinal": counts[kind], "packed_reference_rows": cost})
    return {"refs": refs, "qwen_ref_items": items, "mapping": mapping,
            "packed_reference_rows": sum(row["packed_reference_rows"] for row in mapping),
            "latent_only_exp": any(item is None for item in items),
            "applied_to_conditioning": False, "automatic_accept": False}
