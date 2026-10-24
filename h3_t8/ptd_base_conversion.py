"""CPU streaming plain H3 layout conversion, not an expert/DMD/INT8 merge.

Mapping adapted from EPFL VITA Lab PTD-inference43ae22c3 (Apache-2.0).
See THIRD_PARTY_NOTICES/H21-PTD-BASE-SOURCES.md. No remote code is imported.
Import and metadata preparation do not import Torch or read tensor payloads.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import uuid

from .meridian_checkpoint_io import atomic_json, canonical_identity, conversion_lock, file_sha
from .meridian_conversion import build_rules

DTYPES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}
CONFIG = "transformer/config.json"
INDEX = "transformer/diffusion_pytorch_model.safetensors.index.json"
DEFAULTS = dict(hidden_size=5376, num_layers=50, num_refiner_layers=2,
                num_attention_heads=56, attention_head_dim=128, ffn_dim=14336,
                in_channels=24, audio_in_channels=32, text_dim=5120, freq_dim=256,
                time_embed_hidden_dim=5376, time_embed_dim=2688, rope_freq_dim=16,
                rope_theta=10000., patch_size=[1, 2, 2], norm_eps=1e-5,
                qk_norm_eps=1e-5, final_norm_eps=1e-5)


class SourcePending(FileNotFoundError):
    """A declared download is incomplete; no completed model is published."""


def _cancel(callback):
    if callback is not None and callback():
        raise InterruptedError("PTD base conversion cancelled")


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(raw):
    return json.loads(raw, object_pairs_hook=_pairs)


def _relative(name):
    if (type(name) is not str or not name or "\\" in name or ":" in name or "\0" in name
            or PurePosixPath(name).is_absolute() or str(PurePosixPath(name)) != name
            or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError(f"Expected canonical relative source path: {name!r}")
    return name


def _path(root, name):
    path = root / _relative(name)
    if not path.is_file():
        raise SourcePending(name)
    path = path.resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"Source path escapes its explicit root: {name}")
    return path


def manifest_files(manifest):
    if type(manifest) is not dict or type(manifest.get("files")) is not list:
        raise ValueError("An explicit complete manifest.files list is required")
    result = {}
    for row in manifest["files"]:
        if type(row) is not dict:
            raise ValueError("Invalid manifest file record")
        name = _relative(row.get("file", row.get("path")))
        size, sha = row.get("bytes"), row.get("sha256")
        if name in result or type(size) is not int or size <= 0 or type(sha) is not str or not re.fullmatch("[0-9a-f]{64}", sha):
            raise ValueError(f"Invalid manifest size/SHA or duplicate file: {name}")
        result[name] = {"bytes": size, "sha256": sha}
    if not result:
        raise ValueError("Manifest has no declared files")
    return result


def _header(path, read=None):
    size = path.stat().st_size
    def get(offset, count):
        if read is not None:
            return read(offset, count)
        with path.open("rb") as stream:
            stream.seek(offset)
            return stream.read(count)
    prefix = get(0, 8)
    if len(prefix) != 8:
        raise ValueError("Truncated safetensors prefix")
    length = struct.unpack("<Q", prefix)[0]
    if not 2 <= length <= size - 8:
        raise ValueError("Safetensors header exceeds the actual file")
    raw = get(8, length)
    if len(raw) != length:
        raise ValueError("Truncated safetensors header")
    header = _json(raw)
    if type(header) is not dict:
        raise ValueError("Safetensors header is not an object")
    metadata = header.pop("__metadata__", {})
    if type(metadata) is not dict or any(type(k) is not str or type(v) is not str for k, v in metadata.items()):
        raise ValueError("Safetensors metadata must be string pairs")
    for key, spec in header.items():
        if not key or type(spec) is not dict or set(spec) != {"dtype", "shape", "data_offsets"}:
            raise ValueError("Invalid safetensors tensor entry")
        dims, span, dtype = spec["shape"], spec["data_offsets"], spec["dtype"]
        if (type(dims) is not list or any(type(x) is not int or x <= 0 for x in dims)
                or dtype not in DTYPES or type(span) is not list or len(span) != 2
                or any(type(x) is not int or x < 0 for x in span)
                or span[1] - span[0] != math.prod(dims) * DTYPES[dtype]):
            raise ValueError(f"Invalid actual floating dtype/shape/span: {key}")
    cursor = 0
    for spec in sorted(header.values(), key=lambda x: x["data_offsets"]):
        if spec["data_offsets"][0] != cursor:
            raise ValueError("Tensor ranges overlap or leave holes")
        cursor = spec["data_offsets"][1]
    if not header or size != 8 + length + cursor:
        raise ValueError("Safetensors payload is truncated or has unclaimed bytes")
    return {"tensors": header, "metadata": metadata, "start": 8 + length, "prefix": prefix + raw}


def _rules(config):
    cfg = {**DEFAULTS, **config}
    for key, default in DEFAULTS.items():
        if type(default) is int and (type(cfg[key]) is not int or cfg[key] < (0 if key in ("num_layers", "num_refiner_layers") else 1)):
            raise ValueError(f"Invalid actual architecture dimension: {key}")
    for key in ("rope_theta", "norm_eps", "qk_norm_eps", "final_norm_eps"):
        if type(cfg[key]) not in (int, float) or not math.isfinite(cfg[key]) or (key == "rope_theta" and cfg[key] <= 0):
            raise ValueError(f"Invalid finite model configuration: {key}")
    if 6 * cfg["rope_freq_dim"] > cfg["attention_head_dim"]:
        raise ValueError("Rotary dimensions do not fit the actual attention head")
    # The shared explicit mapping supports positive counts; a zero-sized family
    # has no weights, exactly as in the official original shape contract.
    mapping_cfg = {**cfg, "num_layers": max(1, cfg["num_layers"]),
                   "num_refiner_layers": max(1, cfg["num_refiner_layers"])}
    rules = tuple(r for r in build_rules(mapping_cfg)
                  if not (r.target.startswith("blocks.") and cfg["num_layers"] == 0)
                  and not (r.target.startswith("token_refiner.blocks.") and cfg["num_refiner_layers"] == 0))
    return cfg, rules


def _read_verified(path, offset, length, ledger, chunk_bytes, cancelled):
    if offset < 0 or length < 0 or offset + length > ledger["bytes"]:
        raise ValueError("Source byte range exceeds the verified file")
    with path.open("rb") as stream:
        stop = offset + length
        while offset < stop:
            _cancel(cancelled)
            index = offset // chunk_bytes
            start = index * chunk_bytes
            stream.seek(start)
            block = stream.read(min(chunk_bytes, ledger["bytes"] - start))
            if hashlib.sha256(block).hexdigest() != ledger["blocks"][index]:
                raise ValueError(f"Source block changed during conversion: {path.name}:{index}")
            end = min(stop, start + len(block))
            yield block[offset - start:end - start]
            offset = end


def prepare(source, manifest, *, ledgers=None, chunk_bytes=8 * 1024**2, cancelled=None):
    """Metadata-only without ledgers; no tensor payload or Torch is read/imported."""
    source = Path(source).resolve(strict=True)
    files = manifest_files(manifest)
    if CONFIG not in files or INDEX not in files:
        raise ValueError("Manifest must declare transformer config and index")
    def read(name):
        path = _path(source, name)
        if path.stat().st_size != files[name]["bytes"]:
            raise ValueError(f"Source file size differs from manifest: {name}")
        raw = (path.read_bytes() if ledgers is None else
               b"".join(_read_verified(path, 0, files[name]["bytes"], ledgers[name], chunk_bytes, cancelled)))
        if hashlib.sha256(raw).hexdigest() != files[name]["sha256"]:
            raise ValueError(f"Source config/index SHA mismatch: {name}")
        return _json(raw)
    config, index = read(CONFIG), read(INDEX)
    if type(config) is not dict or type(index) is not dict or type(index.get("weight_map")) is not dict:
        raise ValueError("Invalid transformer config/index")
    cfg, rules = _rules(config)
    weight_map = index["weight_map"]
    expected = {key for rule in rules for key in rule.sources}
    if set(weight_map) != expected:
        raise ValueError("Source index has missing/unknown actual tensor keys")
    shard_for = {key: "transformer/" + _relative(value) for key, value in weight_map.items()}
    shards = set(shard_for.values())
    if set(files) != shards | {CONFIG, INDEX}:
        raise ValueError("Manifest must match every indexed shard exactly")
    headers = {}
    for name in sorted(shards):
        _cancel(cancelled)
        path = _path(source, name)
        if path.stat().st_size != files[name]["bytes"]:
            raise ValueError(f"Source shard size differs from manifest: {name}")
        reader = None if ledgers is None else lambda a, b, n=name, p=path: b"".join(
            _read_verified(p, a, b, ledgers[n], chunk_bytes, cancelled))
        headers[name] = _header(path, reader)
        if set(headers[name]["tensors"]) != {k for k, v in shard_for.items() if v == name}:
            raise ValueError(f"Actual shard keys differ from the source index: {name}")
    targets = []
    for rule in rules:
        pieces, dtype, shape = [], None, None
        for key, wanted in zip(rule.sources, rule.shapes, strict=True):
            name = shard_for[key]
            header = headers[name]
            spec = header["tensors"][key]
            if tuple(spec["shape"]) != wanted:
                raise ValueError(f"Wrong actual source tensor shape: {key}")
            if dtype is not None and dtype != spec["dtype"]:
                raise ValueError(f"Q/K/V dtypes differ; no implicit cast: {rule.target}")
            dtype, shape = spec["dtype"], list(wanted)
            a, b = spec["data_offsets"]
            pieces.append({"file": name, "offset": header["start"] + a, "length": b - a})
        if rule.operation == "concat_qkv":
            shape[0] *= 3
        elif rule.operation == "swap_swiglu":
            part = pieces[0]
            half = part["length"] // 2
            pieces = [{**part, "offset": part["offset"] + half, "length": half}, {**part, "length": half}]
        targets.append({"name": rule.target, "dtype": dtype, "shape": shape, "pieces": pieces})
    targets.append({"name": "rope.inv_freq", "dtype": "F32", "shape": [cfg["rope_freq_dim"]], "pieces": []})
    return {"config": config, "normalized_config": cfg, "files": files, "targets": targets,
            "source_root": str(source), "source_tensors": len(expected), "native_tensors": len(targets),
            "shards": len(shards), "source_complete_SHA_verified": ledgers is not None,
            "Torch_imported_by_prepare": False, "weights_merged_or_quantized": False}


def _capture(path, wanted, chunk_bytes, cancelled):
    digest, blocks, size = hashlib.sha256(), [], 0
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            _cancel(cancelled)
            size += len(block)
            digest.update(block)
            blocks.append(hashlib.sha256(block).hexdigest())
    if size != wanted["bytes"] or digest.hexdigest() != wanted["sha256"]:
        raise ValueError(f"Actual complete source SHA/size mismatch: {path.name}")
    return {"bytes": size, "blocks": blocks}


def _encoded_header(tensors, metadata):
    data = {"__metadata__": metadata, **tensors}
    raw = json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw


def _target_chunks(source, target, ledgers, chunk_bytes, cancelled):
    """The same authoritative byte order for fresh writes and resumed proofs."""
    if target["pieces"]:
        for piece in target["pieces"]:
            yield from _read_verified(_path(source, piece["file"]), piece["offset"], piece["length"],
                                      ledgers[piece["file"]], chunk_bytes, cancelled)
    else:
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError("PTD base conversion is CPU-only")
        cfg = target["rope_config"]
        value = 1. / (cfg["rope_theta"] ** (torch.arange(0, 2 * cfg["rope_freq_dim"], 2,
                        dtype=torch.float32, device="cpu") / (2 * cfg["rope_freq_dim"])))
        yield value.numpy().tobytes()


def _finite_chunk(carry, block, target):
    """Identical bounded floating validation for fresh and untrusted saved parts."""
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("PTD base conversion is CPU-only")
    dtype = {"F32": torch.float32, "F64": torch.float64,
             "F16": torch.float16, "BF16": torch.bfloat16}[target["dtype"]]
    combined = carry + block
    n = len(combined) // DTYPES[target["dtype"]] * DTYPES[target["dtype"]]
    if n and not bool(torch.isfinite(torch.frombuffer(bytearray(combined[:n]), dtype=dtype)).all()):
        raise ValueError(f"Nonfinite actual source tensor: {target['name']}")
    return combined[n:]


def _part(source, job, target, identity, ledgers, chunk_bytes, cancelled):
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("PTD base conversion is CPU-only")
    size = math.prod(target["shape"]) * DTYPES[target["dtype"]]
    spec = {k: target[k] for k in ("dtype", "shape")}
    prefix = _encoded_header({target["name"]: {**spec, "data_offsets": [0, size]}},
                             {"t8_identity": identity, "target": target["name"]})
    part = job / (uuid.uuid4().hex + ".safetensors")
    temporary = part.with_suffix(".partial")
    digest, count, carry = hashlib.sha256(prefix), 0, b""
    try:
        with temporary.open("xb") as output:
            output.write(prefix)
            for block in _target_chunks(source, target, ledgers, chunk_bytes, cancelled):
                _cancel(cancelled)
                carry = _finite_chunk(carry, block, target)
                output.write(block)
                digest.update(block)
                count += len(block)
            if carry or count != size:
                raise ValueError("Copied tensor byte count differs from actual shape/dtype")
            output.flush()
            os.fsync(output.fileno())
        _cancel(cancelled)
        os.link(temporary, part)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"file": part.name, "sha256": digest.hexdigest()}


def _verify_part(job, item, target, identity, cancelled):
    if type(item) is not dict or set(item) != {"file", "sha256"}:
        raise ValueError("Malformed saved part identity")
    path = _path(job, item["file"])
    if file_sha(path, cancelled=cancelled) != item["sha256"]:
        raise ValueError("Saved tensor part SHA changed")
    header = _header(path)
    if header["metadata"] != {"t8_identity": identity, "target": target["name"]}:
        raise ValueError("Saved part belongs to a different job/target")
    spec = header["tensors"].get(target["name"])
    if set(header["tensors"]) != {target["name"]} or {k: spec[k] for k in ("dtype", "shape")} != {k: target[k] for k in ("dtype", "shape")}:
        raise ValueError("Saved part actual tensor metadata changed")
    return path, header


def _verify_resumed_payload(source, path, header, target, ledgers, chunk_bytes, cancelled, expected_sha):
    """A mutable part and state SHA cannot authenticate each other's contents.

    Re-read the actual verified source ranges only for reused targets, preserving
    QKV/half-swap order and the exact fresh-write FP32 RoPE formula. No full
    tensor is materialized. This is real additional resume I/O, not a signature.
    """
    count, digest, carry = 0, hashlib.sha256(header["prefix"]), b""
    with path.open("rb") as stream:
        stream.seek(header["start"])
        for block in _target_chunks(source, target, ledgers, chunk_bytes, cancelled):
            _cancel(cancelled)
            if stream.read(len(block)) != block:
                raise ValueError(f"Saved part payload differs from verified source or derived RoPE: {target['name']}")
            carry = _finite_chunk(carry, block, target)
            digest.update(block)
            count += len(block)
        if carry or count != math.prod(target["shape"]) * DTYPES[target["dtype"]] or stream.read(1):
            raise ValueError(f"Saved part payload length differs from verified source: {target['name']}")
    if digest.hexdigest() != expected_sha:
        raise ValueError(f"Saved part content changed during verified source payload proof: {target['name']}")


def _assembly_header(parts, metadata):
    offset, tensors = 0, {}
    for _, header, _ in parts:
        for name, spec in header["tensors"].items():
            if name in tensors:
                raise ValueError("Duplicate assembled native tensor")
            tensors[name] = {**spec, "data_offsets": [offset + x for x in spec["data_offsets"]]}
        offset += sum(s["data_offsets"][1] - s["data_offsets"][0] for s in header["tensors"].values())
    return _encoded_header(tensors, metadata), len(tensors)


def _verify_published_payload(parts, output, metadata, cancelled):
    """A publication journal cannot authenticate a co-modified output file.

    All reused parts have already been bound to actual verified source bytes.
    Compare their real stream to the real output while checking these same
    part bytes against their proven SHA. No second assembled model is written.
    """
    prefix, count = _assembly_header(parts, metadata)
    digest = hashlib.sha256(prefix)
    with output.open("rb") as published:
        if published.read(len(prefix)) != prefix:
            raise ValueError("Published header differs from verified source layout")
        for path, header, expected_sha in parts:
            _cancel(cancelled)
            part_digest = hashlib.sha256()
            with path.open("rb") as source:
                raw = source.read(header["start"])
                if raw != header["prefix"]:
                    raise ValueError("Verified source part header changed during published payload proof")
                part_digest.update(raw)
                while block := source.read(8 * 1024**2):
                    _cancel(cancelled)
                    if published.read(len(block)) != block:
                        raise ValueError("Published payload differs from verified source parts")
                    digest.update(block)
                    part_digest.update(block)
            if part_digest.hexdigest() != expected_sha:
                raise ValueError("Verified source part changed during published payload proof")
        if published.read(1):
            raise ValueError("Published payload length differs from verified source parts")
        size = published.tell()
    return {"path": str(output), "sha256": digest.hexdigest(), "bytes": size, "tensor_count": count}


def _assemble(parts, output, metadata, cancelled, before_publish):
    prefix, count = _assembly_header(parts, metadata)
    temporary = output.with_name(output.name + ".partial-" + uuid.uuid4().hex)
    digest = hashlib.sha256(prefix)
    try:
        with temporary.open("xb") as stream:
            stream.write(prefix)
            for path, header, expected in parts:
                _cancel(cancelled)
                part_digest = hashlib.sha256()
                with path.open("rb") as source:
                    raw = source.read(header["start"])
                    if raw != header["prefix"]:
                        raise ValueError("Part header changed during assembly")
                    part_digest.update(raw)
                    while block := source.read(8 * 1024**2):
                        _cancel(cancelled)
                        stream.write(block)
                        digest.update(block)
                        part_digest.update(block)
                if part_digest.hexdigest() != expected:
                    raise ValueError("Actual part bytes changed during assembly")
            stream.flush()
            os.fsync(stream.fileno())
        _header(temporary)
        receipt = {"path": str(output), "sha256": digest.hexdigest(),
                   "bytes": temporary.stat().st_size, "tensor_count": count}
        before_publish(receipt)
        _cancel(cancelled)
        os.link(temporary, output)
        return receipt
    finally:
        if temporary.exists():
            temporary.unlink()


def run_conversion(source, job, output, manifest, *, chunk_bytes=8 * 1024**2, cancelled=None, progress=None):
    """Create a plain native file; never load a full tensor/model or change sources."""
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    source, job, output = Path(source).resolve(strict=True), Path(job).resolve(), Path(output).resolve()
    if job.is_relative_to(source) or source.is_relative_to(job) or output.is_relative_to(source) or output.is_relative_to(job):
        raise ValueError("Keep source, resumable job and output in separate explicit paths")
    files = manifest_files(manifest)
    def notify(stage, **values):
        if progress is not None:
            progress({"stage": stage, **values})
    implementation = {p.name: file_sha(p) for p in (Path(__file__), Path(__file__).with_name("meridian_conversion.py"),
                                                     Path(__file__).with_name("meridian_checkpoint_io.py"))}
    identity_data = {"schema": "t8.ptd_plain_base_job.v1", "source": str(source), "output": str(output),
                     "files": files, "implementation": implementation, "chunk_bytes": chunk_bytes,
                     "repository": manifest.get("repository", manifest.get("repo_id", "unknown")),
                     "revision": manifest.get("revision", "unknown")}
    identity = canonical_identity(identity_data)
    job.mkdir(parents=True, exist_ok=True)
    with conversion_lock(job / "conversion.lock"):
        marker, state_file, publication = job / "job.json", job / "state.json", job / "publication.json"
        if marker.exists():
            if _json(marker.read_bytes()) != identity_data:
                raise ValueError("Conversion identity changed; use a new job directory")
        else:
            if any(p.name != "conversion.lock" for p in job.iterdir()):
                raise ValueError("Unknown nonempty conversion directory")
            atomic_json(marker, identity_data)
        if output.exists() and not publication.exists():
            raise FileExistsError("Unrelated output exists; never overwritten or adopted")
        ledgers = {}
        for name, wanted in files.items():
            notify("verifying_source", file=name)
            ledgers[name] = _capture(_path(source, name), wanted, chunk_bytes, cancelled)
        plan = prepare(source, manifest, ledgers=ledgers, chunk_bytes=chunk_bytes, cancelled=cancelled)
        targets = plan["targets"]
        targets[-1]["rope_config"] = plan["normalized_config"]
        state = _json(state_file.read_bytes()) if state_file.exists() else {"identity": identity, "completed": {}}
        names = {t["name"] for t in targets}
        if type(state) is not dict or state.get("identity") != identity or type(state.get("completed")) is not dict or set(state["completed"]) - names:
            raise ValueError("Malformed or foreign conversion state")
        for index, target in enumerate(targets):
            _cancel(cancelled)
            name = target["name"]
            if name in state["completed"]:
                path, header = _verify_part(job, state["completed"][name], target, identity, cancelled)
                _verify_resumed_payload(source, path, header, target, ledgers, chunk_bytes, cancelled,
                                        state["completed"][name]["sha256"])
                notify("reused_tensor", index=index, target=name)
                continue
            if output.exists():
                raise ValueError("Published output has incomplete resumable state")
            notify("converting_tensor", index=index, target=name)
            state["completed"][name] = _part(source, job, target, identity, ledgers, chunk_bytes, cancelled)
            atomic_json(state_file, state, replace_existing=state_file.exists())
            notify("completed_tensor", index=index, target=name)
        cfg = plan["normalized_config"]
        core_config = {"hidden_size": cfg["hidden_size"], "num_layers": cfg["num_layers"],
                       "token_refiner_num_layers": cfg["num_refiner_layers"], "num_attention_heads": cfg["num_attention_heads"],
                       "attention_head_dim": cfg["attention_head_dim"], "ffn_hidden_size": cfg["ffn_dim"],
                       "latents_dim": cfg["in_channels"], "audio_latents_dim": cfg["audio_in_channels"],
                       "patch_size": cfg["patch_size"], "text_dim": cfg["text_dim"], "timestep_input_dim": cfg["freq_dim"],
                       "time_embed_hidden_size": cfg["time_embed_hidden_dim"], "time_embed_dim": cfg["time_embed_dim"],
                       "rope_inv_freq_len": cfg["rope_freq_dim"], **{k: cfg[k] for k in ("norm_eps", "qk_norm_eps", "final_norm_eps")}}
        metadata = {"format": "pt", "ptd_base_format": "minimax-h3-unpruned-v1", "t8_conversion_identity": identity,
                    "source_config": json.dumps(plan["config"], sort_keys=True),
                    "config": json.dumps({"transformer": core_config}, sort_keys=True),
                    "source_repo": identity_data["repository"], "source_revision": identity_data["revision"],
                    "source_manifest_identity": canonical_identity(files)}
        def postflight():
            for name, wanted in files.items():
                notify("postflight_source", file=name)
                if file_sha(_path(source, name), cancelled=cancelled) != wanted["sha256"]:
                    raise ValueError(f"Source SHA changed before publication: {name}")
        def before_publish(receipt):
            postflight()
            value = {"identity": identity, "output": receipt}
            if publication.exists() and _json(publication.read_bytes()) != value:
                raise ValueError("Existing publication journal differs")
            if not publication.exists():
                atomic_json(publication, value)
            notify("before_publish", output=receipt)
        parts = [(*_verify_part(job, state["completed"][t["name"]], t, identity, cancelled),
                  state["completed"][t["name"]]["sha256"]) for t in targets]
        if output.exists():
            value = _json(publication.read_bytes())
            actual = _verify_published_payload(parts, output, metadata, cancelled)
            if value != {"identity": identity, "output": actual}:
                raise ValueError("Published model/journal content changed")
            postflight()
            result = actual
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            result = _assemble(parts, output, metadata, cancelled, before_publish)
        notify("published", output=result)
        result = {**result, "identity": identity, "source_tensors": plan["source_tensors"],
                  "source_shards": plan["shards"], "plain_dtype_preserved": True, "experts_or_DMD_merged": False,
                  "quantized": False, "GPU_validated": False, "quality_accepted": False}
        result_file = job / "result.json"
        if result_file.exists():
            if _json(result_file.read_bytes()) != result:
                raise ValueError("Saved result receipt changed")
        else:
            atomic_json(result_file, result)
        return result
