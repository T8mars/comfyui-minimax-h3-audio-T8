"""Read DCP metadata as inert data, never as a Python pickle program.

This deliberately small protocol-4/5 VM uses pickletools' opcode decoder but
does not import a checkpoint-named symbol, call REDUCE/BUILD targets, or create
checkpoint-defined Python objects. Only the tensor-only DCP schema is accepted.
It is separate from the frozen PTD runtime and does not import Torch.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import pickletools


class UnsafeMetadata(ValueError):
    """The input is not the supported closed, tensor-only data schema."""


@dataclass(frozen=True, slots=True)
class Symbol:
    module: str
    name: str


@dataclass(eq=False, slots=True)
class Record:
    symbol: Symbol
    state: object = None


_MD = "torch.distributed.checkpoint.metadata"
_FS = "torch.distributed.checkpoint.filesystem"
_RECORDS = frozenset(
    Symbol(_MD, name)
    for name in (
        "Metadata", "TensorStorageMetadata", "TensorProperties",
        "ChunkStorageMetadata", "MetadataIndex", "StorageMeta",
    )
) | {Symbol(_FS, "_StorageInfo")}
_DTYPES = {"float64": 8, "float32": 4, "float16": 2, "bfloat16": 2}
_GLOBALS = _RECORDS | {Symbol("torch", name) for name in _DTYPES} | {
    Symbol("torch", "Size"), Symbol("torch.serialization", "_get_layout"),
    Symbol(_MD, "_MEM_FORMAT_ENCODING"),
}
_MARK = object()


def _fail(message, position=None):
    suffix = "" if position is None else f" at byte {position}"
    raise UnsafeMetadata(message + suffix)


def _tuple(value, context):
    if type(value) is not tuple:
        _fail(f"{context} requires a tuple")
    return value


def _integer(value, context, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        _fail(f"{context} requires a {'positive' if positive else 'nonnegative'} integer")
    return value


def _size(value, context, *, positive=False):
    value = _tuple(value, context)
    for item in value:
        _integer(item, context, positive=positive)
    return value


def decode_inert(data: bytes):
    """Decode to our own inert records; no pickle Unpickler or callback exists.

    Input size itself bounds the number of opcodes and memo entries. No arbitrary
    model/metadata byte cap is imposed. Memory used by the expanded metadata is a
    real CPU cost; callers should inspect it in their own process.
    """
    if type(data) is not bytes:
        raise TypeError("Metadata input must be bytes")
    stack, memo, marks = [], [], []
    globals_seen, histogram = set(), Counter()
    protocol, stopped = None, False

    def pop_mark():
        if not marks:
            _fail("Missing MARK")
        index = marks.pop()
        if index >= len(stack) or stack[index] is not _MARK:
            _fail("Corrupt MARK stack")
        items = stack[index + 1:]
        del stack[index:]
        return items

    def symbol(module, name, pos):
        if type(module) is not str or type(name) is not str:
            _fail("GLOBAL requires literal module/name strings", pos)
        result = Symbol(module, name)
        if result not in _GLOBALS:
            _fail(f"Unsupported symbol {module}.{name}; no import or call was performed", pos)
        globals_seen.add(f"{module}.{name}")
        return result

    def set_items(target, items):
        if type(target) is not dict or len(items) % 2:
            _fail("Invalid dictionary item sequence")
        for index in range(0, len(items), 2):
            key = items[index]
            # Only our primitives or inert identity-hashed records can be keys.
            if not (type(key) in (str, int, Record) or key is None):
                _fail("Unsupported dictionary key")
            if key in target:
                _fail("Duplicate metadata dictionary key")
            target[key] = items[index + 1]

    try:
        for opcode, argument, position in pickletools.genops(io.BytesIO(data)):
            name = opcode.name
            histogram[name] += 1
            if name == "PROTO":
                if position != 0 or protocol is not None or argument not in (4, 5):
                    _fail("Only one explicit protocol 4/5 header is supported", position)
                protocol = argument
            elif protocol is None:
                _fail("Missing protocol header", position)
            elif name == "FRAME":
                if argument < 0 or position + 9 + argument > len(data):
                    _fail("FRAME extends outside metadata", position)
            elif name == "STOP":
                if position != len(data) - 1 or marks or len(stack) != 1:
                    _fail("Trailing bytes, unclosed MARK, or invalid final stack", position)
                stopped = True
                break
            elif name in ("SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8"):
                stack.append(argument)
            elif name in ("BININT", "BININT1", "BININT2", "LONG1", "LONG4"):
                stack.append(argument)
            elif name in ("NEWFALSE", "NEWTRUE", "NONE"):
                stack.append({"NEWFALSE": False, "NEWTRUE": True, "NONE": None}[name])
            elif name == "EMPTY_TUPLE":
                stack.append(())
            elif name == "EMPTY_DICT":
                stack.append({})
            elif name == "EMPTY_LIST":
                stack.append([])
            elif name == "MARK":
                marks.append(len(stack))
                stack.append(_MARK)
            elif name == "MEMOIZE":
                memo.append(stack[-1])
            elif name in ("BINGET", "LONG_BINGET"):
                if argument >= len(memo):
                    _fail("Missing memo entry", position)
                stack.append(memo[argument])
            elif name == "STACK_GLOBAL":
                target_name, module = stack.pop(), stack.pop()
                stack.append(symbol(module, target_name, position))
            elif name == "NEWOBJ":
                args, cls = stack.pop(), stack.pop()
                if type(cls) is not Symbol or cls not in _RECORDS or args != ():
                    _fail("NEWOBJ is restricted to empty inert DCP records", position)
                stack.append(Record(cls))
            elif name == "BUILD":
                state = stack.pop()
                if type(stack[-1]) is not Record or stack[-1].state is not None:
                    _fail("BUILD requires one uninitialized inert record", position)
                stack[-1].state = state
            elif name == "REDUCE":
                args, function = _tuple(stack.pop(), "REDUCE"), stack.pop()
                if function == Symbol("torch", "Size") and len(args) == 1:
                    result = _size(args[0], "torch.Size")
                elif function == Symbol("torch.serialization", "_get_layout") and args == ("torch.strided",):
                    result = "torch.strided"
                elif function == Symbol(_MD, "_MEM_FORMAT_ENCODING") and len(args) == 1 and type(args[0]) is int and args[0] in (0, 1, 2):
                    result = ("memory_format", args[0])
                else:
                    _fail("REDUCE target/arguments outside the closed literal schema", position)
                stack.append(result)
            elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
                count = int(name[-1])
                if len(stack) < count:
                    _fail("Tuple stack underflow", position)
                result = tuple(stack[-count:])
                del stack[-count:]
                stack.append(result)
            elif name == "TUPLE":
                stack.append(tuple(pop_mark()))
            elif name == "SETITEMS":
                items = pop_mark()
                set_items(stack[-1], items)
            elif name == "SETITEM":
                value, key = stack.pop(), stack.pop()
                set_items(stack[-1], [key, value])
            elif name == "APPENDS":
                values = pop_mark()
                if type(stack[-1]) is not list:
                    _fail("APPENDS target must be a list", position)
                stack[-1].extend(values)
            elif name == "APPEND":
                value = stack.pop()
                if type(stack[-1]) is not list:
                    _fail("APPEND target must be a list", position)
                stack[-1].append(value)
            else:
                _fail(f"Unsupported pickle opcode {name}; not executed", position)
    except (IndexError, KeyError, TypeError, UnicodeError) as exc:
        raise UnsafeMetadata(f"Malformed inert metadata stack: {exc}") from exc
    except ValueError as exc:
        if isinstance(exc, UnsafeMetadata):
            raise
        raise UnsafeMetadata(f"Malformed metadata encoding: {exc}") from exc
    if not stopped:
        _fail("Metadata has no final STOP")
    return stack[0], {"protocol": protocol, "opcodes": dict(histogram), "symbols": sorted(globals_seen), "memo_entries": len(memo)}


def _record(value, name, required, optional=()):
    if type(value) is not Record or value.symbol != Symbol(_MD if name != "_StorageInfo" else _FS, name):
        _fail(f"Expected inert {name} record")
    state = value.state
    if type(state) is not dict or set(state) - set(required) - set(optional) or set(required) - set(state):
        _fail(f"Unexpected {name} fields: {tuple(state) if type(state) is dict else type(state).__name__}")
    return state


@dataclass(frozen=True, slots=True)
class Chunk:
    offsets: tuple[int, ...]
    sizes: tuple[int, ...]
    relative_path: str
    offset: int
    length: int


@dataclass(frozen=True, slots=True)
class TensorEntry:
    name: str
    dtype: str
    shape: tuple[int, ...]
    chunks: tuple[Chunk, ...]
    requires_grad: bool = False
    pin_memory: bool = False

    @property
    def nbytes(self):
        return math.prod(self.shape) * _DTYPES[self.dtype]


@dataclass(frozen=True, slots=True)
class SafeMetadata:
    tensors: dict[str, TensorEntry]
    sha256: str
    file_bytes: int
    syntax: dict
    version: str | None

    def summary(self):
        return {
            "metadata_sha256": self.sha256, "metadata_bytes": self.file_bytes,
            "tensor_count": len(self.tensors), "tensor_bytes": sum(item.nbytes for item in self.tensors.values()),
            "chunk_count": sum(len(item.chunks) for item in self.tensors.values()),
            "max_chunk_serialized_bytes": max((c.length for t in self.tensors.values() for c in t.chunks), default=0),
            "version": self.version, "syntax": self.syntax,
            "pickle_callbacks_executed": False, "checkpoint_imports_executed": False,
        }


def _relative_path(value):
    if type(value) is not str or not value or "\\" in value or ":" in value or "\x00" in value:
        _fail("Shard path must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or PureWindowsPath(value).is_absolute() or any(part in (".", "..") for part in value.split("/")) or str(path) != value:
        _fail("Shard path escapes or is not canonical")
    if path.suffix != ".distcp":
        _fail("Tensor storage must reference a .distcp shard")
    return value


def validate_inert(root, *, sha256="", file_bytes=0, syntax=None):
    """Validate every storage and tensor record, including unselected scaffolds."""
    md = _record(root, "Metadata", ("state_dict_metadata",), ("planner_data", "storage_data", "storage_meta", "version"))
    entries, storage = md["state_dict_metadata"], md.get("storage_data")
    if type(entries) is not dict or not entries or type(storage) is not dict:
        _fail("Metadata must contain nonempty tensor entries and storage_data")
    version = md.get("version")
    if version is not None and type(version) is not str:
        _fail("Invalid DCP version")
    sm = md.get("storage_meta")
    if sm is not None:
        state = _record(sm, "StorageMeta", (), ("checkpoint_id", "save_id", "load_id", "modules"))
        for key in ("checkpoint_id", "save_id", "load_id"):
            if state.get(key) is not None and type(state[key]) is not str:
                _fail("StorageMeta IDs must be strings or None")
        if "modules" in state and (type(state["modules"]) is not list or any(type(x) is not str for x in state["modules"])):
            _fail("StorageMeta modules must be literal strings")
    # The flat tensor planner must neither request load_bytes nor carry objects.
    planner = md.get("planner_data")
    if planner is not None:
        if type(planner) is not dict or set(planner) != set(entries):
            _fail("Planner data must be the complete flat tensor-name mapping")
        if any(type(value) not in (tuple, list) or tuple(value) != (key,) for key, value in planner.items()):
            _fail("Only the flat tensor planner schema is supported")
    storage_by_key, intervals = {}, {}
    for index, value in storage.items():
        idx = _record(index, "MetadataIndex", ("fqn",), ("offset", "index"))
        name = idx["fqn"]
        if type(name) is not str or name not in entries:
            _fail("Storage index references an unknown tensor")
        offsets = _size(idx.get("offset"), "MetadataIndex.offset")
        if idx.get("index") is not None:
            _integer(idx["index"], "MetadataIndex.index")
        state = _record(value, "_StorageInfo", ("relative_path", "offset", "length"), ("transform_descriptors",))
        if state.get("transform_descriptors") not in (None, (), []):
            _fail("Storage transforms/extensions are not allowed")
        path = _relative_path(state["relative_path"])
        start = _integer(state["offset"], "Storage offset")
        length = _integer(state["length"], "Storage length", positive=True)
        key = (name, offsets)
        if key in storage_by_key:
            _fail("Duplicate logical storage index")
        storage_by_key[key] = (path, start, length)
        intervals.setdefault(path, []).append((start, start + length))
    for ranges in intervals.values():
        end = 0
        for start, next_end in sorted(ranges):
            if start < end:
                _fail("Overlapping serialized tensor storage slices")
            end = next_end
    result, consumed = {}, set()
    for name, value in entries.items():
        if type(name) is not str or not name or "\x00" in name:
            _fail("Tensor names must be nonempty literal strings")
        entry = _record(value, "TensorStorageMetadata", ("properties", "size", "chunks"))
        props = entry["properties"]
        if type(props) is not Record or props.symbol != Symbol(_MD, "TensorProperties") or type(props.state) is not tuple or len(props.state) != 5:
            _fail("Invalid TensorProperties literal tuple")
        dtype, layout, requires_grad, memory_format, pin = props.state
        if type(dtype) is not Symbol or dtype.module != "torch" or dtype.name not in _DTYPES or layout != "torch.strided" or type(requires_grad) is not bool or type(pin) is not bool or memory_format != ("memory_format", 0):
            _fail("Only dense contiguous floating tensor properties are supported")
        shape = _size(entry["size"], "Tensor size", positive=True)
        if not shape or type(entry["chunks"]) is not list or not entry["chunks"]:
            _fail("A nonempty tensor and complete chunks are required")
        chunks, positions, count = [], set(), 0
        for raw_chunk in entry["chunks"]:
            chunk = _record(raw_chunk, "ChunkStorageMetadata", ("offsets", "sizes"))
            # DCP's distributed writer emits empty tail shards, e.g. [0,5376]
            # at row 96 of a [96,5376] head. They still have real serialized
            # storage records and must be checked, not mistaken for bad shapes.
            offsets, sizes = _size(chunk["offsets"], "Chunk offsets"), _size(chunk["sizes"], "Chunk sizes")
            if len(offsets) != len(shape) or len(sizes) != len(shape) or any(o + n > dim for o, n, dim in zip(offsets, sizes, shape)):
                _fail(f"Chunk shape outside tensor {name}")
            if offsets in positions:
                _fail(f"Duplicate chunk offset {name}")
            positions.add(offsets)
            key = (name, offsets)
            if key not in storage_by_key:
                _fail(f"Chunk missing serialized storage {name}")
            path, start, length = storage_by_key[key]
            consumed.add(key)
            chunks.append(Chunk(offsets, sizes, path, start, length))
            count += math.prod(sizes)
        # Rectangles may not overlap; equal total volume then proves coverage.
        ordered = sorted((c for c in chunks if math.prod(c.sizes)), key=lambda c: c.offsets[0])
        active = []
        for chunk in ordered:
            active = [other for other in active if other.offsets[0] + other.sizes[0] > chunk.offsets[0]]
            for other in active:
                if all(a < b + nb and b < a + na for a, na, b, nb in zip(chunk.offsets, chunk.sizes, other.offsets, other.sizes)):
                    _fail(f"Overlapping logical tensor chunks {name}")
            active.append(chunk)
        if count != math.prod(shape):
            _fail(f"Incomplete tensor chunk coverage {name}")
        result[name] = TensorEntry(name, dtype.name, shape, tuple(chunks), requires_grad, pin)
    if consumed != set(storage_by_key):
        _fail("Unreferenced serialized storage records")
    return SafeMetadata(result, sha256, file_bytes, syntax or {}, version)


def read_metadata(path):
    path = Path(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    root, syntax = decode_inert(data)
    return validate_inert(root, sha256=digest, file_bytes=len(data), syntax=syntax)


def to_torch_metadata(metadata: SafeMetadata):
    """Optional explicit trusted constructors after all inert checks, not pickle.

    The streaming converter need not construct DCP objects or call its reader.
    This bridge is useful to callers with a verified tensor-only DCP planner.
    """
    import torch
    from torch.distributed.checkpoint import metadata as m
    from torch.distributed.checkpoint.filesystem import _StorageInfo

    entries, storage = {}, {}
    for name, tensor in metadata.tensors.items():
        entries[name] = m.TensorStorageMetadata(
            m.TensorProperties(dtype=getattr(torch, tensor.dtype), requires_grad=tensor.requires_grad, pin_memory=tensor.pin_memory), torch.Size(tensor.shape),
            [m.ChunkStorageMetadata(torch.Size(c.offsets), torch.Size(c.sizes)) for c in tensor.chunks],
        )
        for index, chunk in enumerate(tensor.chunks):
            storage[m.MetadataIndex(name, chunk.offsets, index)] = _StorageInfo(chunk.relative_path, chunk.offset, chunk.length)
    return m.Metadata(entries, {name: (name,) for name in entries}, storage, m.StorageMeta(), metadata.version)
