"""Explicit CPU-only, tensor-only PTD DCP -> safetensors conversion.

Default: prepare a plan without importing Torch or reading any shard tensors.
--convert requires an explicit manifest and output. Metadata is decoded by our
closed data VM, never by pickle.load / FileSystemReader.read_metadata. Each
bounded tensor slice is loaded using Torch's weights_only=True default allowlist.
No GPU, service, download, environment installation, or original-file overwrite.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "h3_t8"))
import ptd_safe_metadata as safe  # noqa: E402

GRIDS = {
    "h3_4step": (0.999, 0.947, 0.857, 0.667, 0.0),
    "h3_6step": (0.999, 0.968, 0.923, 0.857, 0.750, 0.545, 0.0),
}
_FACTOR = re.compile(r"^(transformer\..+)\.lora_([AB])\.(0|[1-9][0-9]*)\.weight$")
_BLOCK = re.compile(r"^transformer\.(transformer_blocks|token_refiner\.refiner_blocks)\.(0|[1-9][0-9]*)\.(.+)$")
_PARTS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
_BUFFERS = {
    "transformer.rope.inv_freq", "transformer.proj_out.orig_weight",
    "transformer.proj_out.orig_bias", "transformer.audio_proj_out.orig_weight",
    "transformer.audio_proj_out.orig_bias",
}
_SAFEDTYPE = {"float64": "F64", "float32": "F32", "float16": "F16", "bfloat16": "BF16"}


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _peak_working_set():
    """Current process OS peak; not a whole-machine RAM requirement estimate."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return counters.PeakWorkingSetSize
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def _inside(source, relative):
    path = (source / relative).resolve(strict=True)
    if not path.is_relative_to(source) or not path.is_file():
        raise ValueError(f"Source reference escapes the checkpoint: {relative}")
    return path


def _training_keys(recipe):
    return {
        f"transformer.{head}.fan_chead.heads.{hop}.{part}"
        for head in ("proj_out", "audio_proj_out")
        for hop in range(len(GRIDS[recipe]) - 1) for part in ("weight", "bias")
    }


def select_inference(metadata, recipe):
    """Check actual pair/topology shapes without loading or importing Torch."""
    if recipe not in GRIDS:
        raise ValueError("Select h3_4step or h3_6step")
    hops = len(GRIDS[recipe]) - 1
    selected, groups, indices, warnings = {}, {}, {"transformer_blocks": set(), "token_refiner.refiner_blocks": set()}, []
    training = _training_keys(recipe)
    excluded = training & metadata.tensors.keys()
    if excluded and excluded != training:
        raise ValueError("Incomplete published training curvature scaffold")
    for key, entry in metadata.tensors.items():
        if key in excluded:
            continue
        if key in _BUFFERS:
            expected_ndim = 2 if key.endswith("orig_weight") else 1
            if len(entry.shape) != expected_ndim:
                raise ValueError(f"Wrong reference-buffer dimensions: {key}")
            selected[key] = entry
            continue
        match = _FACTOR.fullmatch(key)
        if match is None:
            raise ValueError(f"Unrecognized DCP tensor not discarded: {key}")
        source, factor, hop_text = match.groups()
        hop = int(hop_text)
        if hop >= hops or len(entry.shape) != 2:
            raise ValueError(f"Wrong PTD expert hop/matrix: {key}")
        if source not in ("transformer.proj_out.main", "transformer.audio_proj_out.main"):
            block = _BLOCK.fullmatch(source)
            if block is None or block[3] not in _PARTS:
                raise ValueError(f"Unknown inference projection: {key}")
            indices[block[1]].add(int(block[2]))
        groups.setdefault((source, hop), {})[factor] = entry
        selected[key] = entry
    for family, found in indices.items():
        # Count is bounded by actual records; never allocate up to an untrusted
        # numeric block label just to establish contiguity.
        if sorted(found) != list(range(len(found))):
            raise ValueError(f"Noncontiguous PTD block indices: {family}")
    expected_sources = {"transformer.proj_out.main", "transformer.audio_proj_out.main"}
    for family, found in indices.items():
        expected_sources.update(f"transformer.{family}.{i}.{part}" for i in found for part in _PARTS)
    expected = {(source, hop) for source in expected_sources for hop in range(hops)}
    if set(groups) != expected:
        raise ValueError(f"Incomplete PTD expert topology: {sorted(expected - set(groups))[:5]}")
    dimensions = {}
    for (source, hop), pair in groups.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"Missing PTD A/B pair: {source} hop {hop}")
        a, b = pair["A"], pair["B"]
        if a.shape[0] != b.shape[1] or a.dtype != b.dtype:
            raise ValueError(f"PTD A/B rank/dtype mismatch: {source} hop {hop}")
        if source.endswith("ff.net.0.proj") and b.shape[0] % 2:
            raise ValueError(f"SwiGLU output rows must be even: {source}")
        expected_rank = b.shape[0] if source.endswith(".main") else 128
        if a.shape[0] != expected_rank:
            warnings.append(f"Non-published independent rank retained: {source}, hop={hop}, rank={a.shape[0]}")
        shape = (a.shape, b.shape)
        if dimensions.setdefault(source, shape) != shape:
            warnings.append(f"Varying hop factors retained: {source}; MODEL setup checks actual target dimensions")
    present_buffers = (set(_BUFFERS) - {"transformer.rope.inv_freq"}) & selected.keys()
    if present_buffers and len(present_buffers) != 4:
        raise ValueError("Incomplete original video/audio head buffers")
    for head in ("proj_out", "audio_proj_out"):
        weight, bias = f"transformer.{head}.orig_weight", f"transformer.{head}.orig_bias"
        if weight in selected:
            for hop in range(hops):
                pair = groups[(f"transformer.{head}.main", hop)]
                if selected[weight].shape != (pair["B"].shape[0], pair["A"].shape[1]) or selected[bias].shape != (pair["B"].shape[0],):
                    raise ValueError(f"Head buffers disagree with actual expert shape: {head}")
    return selected, sorted(excluded), sorted(set(warnings)), {
        "blocks": len(indices["transformer_blocks"]), "refiners": len(indices["token_refiner.refiner_blocks"]), "pairs": len(groups),
    }


def prepare(source, recipe):
    source = Path(source).resolve(strict=True)
    metadata_path = _inside(source, ".metadata")
    metadata = safe.read_metadata(metadata_path)
    selected, excluded, warnings, topology = select_inference(metadata, recipe)
    shard_sizes = {}
    for entry in metadata.tensors.values():
        for chunk in entry.chunks:
            if chunk.relative_path not in shard_sizes:
                shard_sizes[chunk.relative_path] = _inside(source, chunk.relative_path).stat().st_size
            size = shard_sizes[chunk.relative_path]
            if chunk.offset + chunk.length > size:
                raise ValueError(f"Tensor storage exceeds actual shard: {entry.name}")
    declaration = {"base_model": "Unknown", "base_model_revision": "Unknown"}
    info_path = source / "ptd_info.json"
    source_info_sha256 = None
    if info_path.exists():
        info_bytes = _inside(source, "ptd_info.json").read_bytes()
        source_info_sha256 = hashlib.sha256(info_bytes).hexdigest()
        info = json.loads(info_bytes)
        if type(info) is not dict or info.get("recipe") != recipe or info.get("hop_grid") != list(GRIDS[recipe]):
            raise ValueError("PTD source declaration disagrees with selected recipe/grid")
        for name in declaration:
            if type(info.get(name)) is str:
                declaration[name] = info[name]
    if declaration["base_model"] != "MiniMaxAI/MiniMax-H3" or declaration["base_model_revision"] != "42ed227ee7df40d41602854ae760620d6eb651fe":
        warnings.append("Source base declaration is not the published pinned base; retained, not falsely attributed")
    plan = {
        "status": "prepared_tensor_reads_not_executed", "source": str(source), "recipe": recipe,
        "metadata": metadata.summary(), "topology": topology,
        "selected_tensors": len(selected), "selected_tensor_bytes": sum(t.nbytes for t in selected.values()),
        "excluded_training_keys": excluded, "warnings": warnings,
        "source_declared_base": declaration,
        "source_info_sha256": source_info_sha256,
        "shards": dict(sorted(shard_sizes.items())),
        "weights_only_tensor_reads": 0, "GPU_actions_requested": False,
        "Torch_imported": "torch" in sys.modules,
        "GPU_initialized": None if "torch" in sys.modules else False,
        "process_peak_working_set_bytes": _peak_working_set(),
        "memory_note": "Expanded metadata costs CPU RAM. The fixed verified input is converted chunkwise without an explicit full expert bank allocation. A hostile serialized tensor can declare storage much larger than its record: weights_only is not DoS protection or a RAM quota. Use SHA-pinned, independently owned immutable source files. Contiguous copies and output pages have real costs. OS peak is observational, not a machine minimum.",
        "trained_GPU_or_quality_qualified": False,
    }
    return metadata, selected, plan


def verify_manifest(source, metadata, manifest_path, manifest_root):
    """Check every declared file, canonical path, bytes and actual SHA."""
    source, manifest_root = Path(source).resolve(strict=True), Path(manifest_root).resolve(strict=True)
    raw = Path(manifest_path).read_bytes()
    manifest = json.loads(raw)
    files = manifest.get("files")
    if type(files) is not list or not files:
        raise ValueError("An explicit manifest.files list is required")
    records, seen = [], set()
    for item in files:
        if type(item) is not dict or type(item.get("file")) is not str:
            raise ValueError("Invalid manifest file record")
        relative = item["file"]
        posix = PurePosixPath(relative)
        if "\\" in relative or ":" in relative or posix.is_absolute() or str(posix) != relative or any(p in (".", "..", "") for p in relative.split("/")):
            raise ValueError("Manifest file path must be canonical and relative")
        path = (manifest_root / relative).resolve(strict=True)
        if not path.is_relative_to(source) or not path.is_file() or path in seen:
            raise ValueError("Manifest files must be unique and inside the selected source")
        seen.add(path)
        size, expected = item.get("bytes"), item.get("sha256")
        if type(size) is not int or size < 0 or type(expected) is not str or re.fullmatch("[0-9a-fA-F]{64}", expected) is None:
            raise ValueError("Manifest needs exact nonnegative bytes and SHA256")
        actual = _sha(path)
        if path.stat().st_size != size or actual != expected.lower():
            raise ValueError(f"Manifest content mismatch: {relative}")
        records.append({"path": str(path), "bytes": size, "sha256": actual})
    required = {source / ".metadata"} | {source / c.relative_path for t in metadata.tensors.values() for c in t.chunks}
    if not required.issubset(seen):
        raise ValueError("Manifest omits metadata or referenced tensor shards")
    if _sha(source / ".metadata") != metadata.sha256:
        raise ValueError("Metadata changed after prepare")
    return {"manifest_sha256": hashlib.sha256(raw).hexdigest(), "source_revision": manifest.get("source_revision"), "files": records}


class TensorSlice(io.IOBase):
    """A seekable view that cannot read beyond a verified serialized record."""

    def __init__(self, stream, offset, length):
        super().__init__()
        self.stream, self.offset, self.length, self.position = stream, offset, length, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        position = offset if whence == os.SEEK_SET else self.position + offset if whence == os.SEEK_CUR else self.length + offset if whence == os.SEEK_END else -1
        if not 0 <= position <= self.length:
            raise ValueError("Tensor slice seek outside verified storage")
        self.position = position
        return position

    def read(self, size=-1):
        remaining = self.length - self.position
        size = remaining if size is None or size < 0 else min(size, remaining)
        self.stream.seek(self.offset + self.position)
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError("Truncated tensor storage slice")
        self.position += size
        return data

    def readinto(self, buffer):
        value = self.read(len(buffer))
        buffer[:len(value)] = value
        return len(value)


def _output_metadata(recipe, metadata, excluded, declaration):
    return {
        "ptd_format": "comfyui-native-h3", "ptd_schema": "1", "ptd_recipe": recipe,
        "ptd_hops": str(len(GRIDS[recipe]) - 1), "ptd_mapping": "diffusers-h3-to-native-h3-v1", "ptd_lora_scale": "1.0",
        "ptd_base_model": declaration["base_model"], "ptd_base_revision": declaration["base_model_revision"],
        "ptd_source_format": "torch.distributed.checkpoint",
        "ptd_source_metadata_sha256": metadata.sha256,
        "ptd_excluded_keys": json.dumps(excluded, separators=(",", ":")),
        "ptd_conversion": "T8-closed-data-metadata-and-weights-only-stream-v1",
    }


def _header(selected, metadata):
    header, offsets, current = {"__metadata__": metadata}, {}, 0
    for name, item in sorted(selected.items()):
        header[name] = {"dtype": _SAFEDTYPE[item.dtype], "shape": list(item.shape), "data_offsets": [current, current + item.nbytes]}
        offsets[name] = current
        current += item.nbytes
    data = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    data += b" " * (-len(data) % 8)
    return struct.pack("<Q", len(data)) + data, offsets, current


def _write_chunk(stream, start, tensor, entry, chunk):
    if not tensor.numel():
        return
    value = tensor.detach().contiguous()
    raw = memoryview(value.view(__import__("torch").uint8).numpy()).cast("B")
    shape, sizes, offsets = entry.shape, chunk.sizes, chunk.offsets
    axis = len(shape) - 1
    while axis > 0 and offsets[axis] == 0 and sizes[axis] == shape[axis]:
        axis -= 1
    strides = [math.prod(shape[i + 1:]) for i in range(len(shape))]
    itemsize = value.element_size()
    run_bytes, cursor = math.prod(sizes[axis:]) * itemsize, 0
    for prefix in itertools.product(*(range(dim) for dim in sizes[:axis])):
        coordinates = [offsets[i] + (prefix[i] if i < axis else 0) for i in range(len(shape))]
        index = sum(coord * stride for coord, stride in zip(coordinates, strides))
        stream.seek(start + index * itemsize)
        stream.write(raw[cursor:cursor + run_bytes])
        cursor += run_bytes
    if cursor != value.numel() * itemsize:
        raise AssertionError("Internal streaming tensor rectangle size mismatch")


def _verify_torch_automatic_globals(torch):
    # Modern Torch cold-imports nested/__init__.py itself and registers these
    # objects plus DeviceMesh's data-only Layout. They are not a user's expanded
    # allowlist. Never clear
    # any registry, never accept a matching name, alias tuple, or foreign object.
    from torch.nested._internal.nested_tensor import NestedTensor, _rebuild_njt
    from torch.distributed._mesh_layout import _MeshLayout

    automatic = (NestedTensor, _rebuild_njt, _MeshLayout)
    live = torch.serialization.get_safe_globals()
    if any(not any(item is known for known in automatic) for item in live):
        raise ValueError("Use a clean conversion process: user-added Torch safe_globals could execute checkpoint callbacks")
    return [f"{item.__module__}.{item.__name__}" for item in live]


def convert(source, recipe, destination, *, manifest_path, manifest_root):
    """Explicit conversion. No unsafe checkpoint callback or whole-bank loading."""
    if sys.byteorder != "little":
        raise ValueError("This byte-preserving safetensors writer requires a little-endian CPU")
    destination = Path(destination).absolute()
    if destination.suffix != ".safetensors" or destination.exists():
        raise FileExistsError("Output must be a new .safetensors path; overwrite is not supported")
    metadata, selected, plan = prepare(source, recipe)
    source = Path(plan["source"])
    if destination.resolve().is_relative_to(source):
        raise ValueError("Converted output must be outside original checkpoint directory")
    verified = verify_manifest(source, metadata, manifest_path, manifest_root)
    info_path = source / "ptd_info.json"
    if (_sha(info_path) if info_path.exists() else None) != plan["source_info_sha256"]:
        raise ValueError("Source base declaration changed after prepare")
    if info_path.exists() and str(info_path) not in {item["path"] for item in verified["files"]}:
        raise ValueError("Manifest omits the consumed PTD source declaration")
    # Explicit CPU process only; never import ComfyUI or alter its model config.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    from safetensors import safe_open

    if torch.cuda.is_initialized():
        raise RuntimeError("Conversion must run in a CPU-only process")
    automatic_globals = _verify_torch_automatic_globals(torch)
    output_metadata = _output_metadata(recipe, metadata, plan["excluded_training_keys"], plan["source_declared_base"])
    header, offsets, nbytes = _header(selected, output_metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent)
    temporary = Path(name)
    reads = 0
    try:
        with os.fdopen(descriptor, "w+b") as out, ExitStack() as opened:
            out.write(header)
            out.truncate(len(header) + nbytes)
            streams = {}
            for key, entry in sorted(selected.items()):
                for chunk in entry.chunks:
                    if chunk.relative_path not in streams:
                        streams[chunk.relative_path] = opened.enter_context(_inside(source, chunk.relative_path).open("rb"))
                    view = TensorSlice(streams[chunk.relative_path], chunk.offset, chunk.length)
                    tensor = torch.load(view, map_location="cpu", weights_only=True)
                    reads += 1
                    if type(tensor) is not torch.Tensor or tensor.layout != torch.strided or tensor.device.type != "cpu" or tensor.dtype != getattr(torch, entry.dtype) or tuple(tensor.shape) != chunk.sizes:
                        raise ValueError(f"Loaded tensor differs from verified storage metadata: {key}")
                    if not bool(torch.isfinite(tensor).all()):
                        raise ValueError(f"Non-finite selected PTD tensor: {key}")
                    _write_chunk(out, len(header) + offsets[key], tensor, entry, chunk)
                    del tensor
            out.flush()
            os.fsync(out.fileno())
        # Rehash actual source bytes again, not just mtime/version numbers.
        if verify_manifest(source, metadata, manifest_path, manifest_root) != verified:
            raise ValueError("Source checkpoint or manifest changed during conversion")
        with safe_open(str(temporary), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(selected) or handle.metadata() != output_metadata:
                raise ValueError("Written safetensors header differs from conversion plan")
            for key, entry in selected.items():
                if tuple(handle.get_slice(key).get_shape()) != entry.shape:
                    raise ValueError("Written tensor shape differs from conversion plan")
        digest = _sha(temporary)
        # Atomic publication, no replace: even a concurrent writer is preserved.
        os.link(temporary, destination)
        plan.update({
            "status": "converted_tensor_bytes_not_GPU_qualified", "output": str(destination.resolve()),
            "output_bytes": temporary.stat().st_size, "output_sha256": digest,
            "weights_only_tensor_reads": reads, "source_integrity": verified,
            "process_peak_working_set_bytes": _peak_working_set(), "GPU_initialized": torch.cuda.is_initialized(),
            "Torch_imported": True, "torch_num_threads": torch.get_num_threads(),
            "torch_exact_automatic_safe_globals": automatic_globals,
        })
        return plan
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="DCP directory containing .metadata")
    parser.add_argument("--recipe", required=True, choices=tuple(GRIDS))
    parser.add_argument("--convert", action="store_true", help="Explicitly read verified tensor slices and create a NEW output")
    parser.add_argument("--output", help="New .safetensors outside the source directory")
    parser.add_argument("--manifest", help="JSON containing files:[{file,bytes,sha256}]")
    parser.add_argument("--manifest-root", help="Explicit directory that manifest relative file names resolve against")
    parser.add_argument("--report", help="Optional new JSON report path, never overwritten")
    args = parser.parse_args(argv)
    if args.report and Path(args.report).exists():
        parser.error("Report path already exists; choose a new path")
    if args.report and Path(args.report).resolve().is_relative_to(Path(args.source).resolve()):
        parser.error("Report must be outside the original source checkpoint directory")
    start = time.monotonic()
    if args.convert:
        if not all((args.output, args.manifest, args.manifest_root)):
            parser.error("--convert requires --output, --manifest and --manifest-root")
        report = convert(args.source, args.recipe, args.output, manifest_path=args.manifest, manifest_root=args.manifest_root)
    else:
        if args.output or args.manifest or args.manifest_root:
            parser.error("Output/manifest options require explicit --convert; default prepare reads metadata and file sizes only")
        _, _, report = prepare(args.source, args.recipe)
    report["seconds"] = time.monotonic() - start
    if args.report:
        with Path(args.report).open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
