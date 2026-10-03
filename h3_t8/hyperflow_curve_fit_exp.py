"""Explicit per-projection two-time approximation; never a full-base fallback.

Independent numerical implementation informed by Adudeguyman's Apache-2.0
HyperFlow-H3 curve-fit contract at 99778905b0074a622af7ba45901d31b28bca5d53.
No upstream source/weights are vendored. Model terms remain separate.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

from .hyperflow_weights_advanced import parse_metadata, plan_source_keys, _sha256

SCHEMA = "t8.hyperflow.curve_fit.v1"
REFERENCE = "99778905b0074a622af7ba45901d31b28bca5d53"
PROJECTIONS = tuple(f"blocks.{i}.adaln_proj.linear" for i in range(50)) + (
    "final_layer.adaln_proj.linear",
)
PINNED_GRID = 1025


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _tensor_digest(value):
    if (type(value) is not torch.Tensor or value.layout != torch.strided
            or value.is_meta):
        raise ValueError("Curve fit identity requires materialized plain tensors")
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256(canonical({"shape": list(value.shape), "dtype": str(value.dtype)}).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _file(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".safetensors":
        raise ValueError("Curve fit inputs must be existing safetensors files")
    stat = path.stat()
    return {"sha256": _sha256(path), "size": stat.st_size,
            "stat": [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]}


def _finite(tensor, label):
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise ValueError("Non-finite or non-floating curve fit " + label)


def code_identity():
    from comfy.ldm.minimax import model as core_h3
    from comfy_kitchen.tensor import int8 as core_int8
    files = [Path(__file__), Path(core_h3.__file__), Path(core_int8.__file__),
             Path(parse_metadata.__code__.co_filename)]
    return {str(p.resolve()): _sha256(p) for p in files}


def _prefix(handle):
    prefixes = {"model.diffusion_model.", "diffusion_model.", ""}
    matches = [p for p in prefixes if any(k.startswith(p + "blocks.0.") for k in handle.keys())]
    if len(matches) != 1:
        raise ValueError("Ambiguous or non-native H3 checkpoint prefix")
    return matches[0]


def linear(handle, prefix, device):
    """Use actual Core INT8/ConvRot dequantization, not scale-only imitation."""
    weight = handle.get_tensor(prefix + ".weight").to(device)
    if weight.dtype == torch.int8:
        from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout
        raw = handle.get_tensor(prefix + ".comfy_quant")
        if raw.dtype != torch.uint8 or raw.ndim != 1 or raw.numel() > 65536:
            raise ValueError("Invalid bounded native INT8 metadata")
        config = json.loads(bytes(raw.tolist()))
        if config.get("format") != "int8_tensorwise":
            raise ValueError("Unsupported teacher quantization")
        params = TensorWiseINT8Layout.Params(
            scale=handle.get_tensor(prefix + ".weight_scale").to(device),
            orig_dtype=torch.float32, orig_shape=tuple(weight.shape),
            convrot=config.get("convrot", False),
            convrot_groupsize=config.get("convrot_groupsize", 256),
        )
        weight = TensorWiseINT8Layout.dequantize(weight, params)
    elif weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Fit teacher/basis requires FP16/BF16/FP32 or native tensorwise INT8")
    bias = handle.get_tensor(prefix + ".bias").to(device).float()
    weight = weight.float()
    _finite(weight, prefix + " weight")
    _finite(bias, prefix + " bias")
    if weight.ndim != 2 or bias.shape != (weight.shape[0],):
        raise ValueError("Invalid curve projection shape")
    return weight, bias


def basis_identity(handle, prefix):
    keys = set(handle.keys())
    selected = [prefix + "adaln_t_table"]
    for name in PROJECTIONS:
        for suffix in (".weight", ".bias", ".weight_scale", ".comfy_quant"):
            key = prefix + name + suffix
            if key in keys:
                selected.append(key)
    required = {prefix + name + suffix for name in PROJECTIONS for suffix in (".weight", ".bias")}
    if not required.issubset(selected):
        raise ValueError("Fit requires all fifty block bases and the final projection")
    # Relative keys are identical in an actual native model's raw state.
    return {k.removeprefix(prefix): _tensor_digest(handle.get_tensor(k)) for k in selected}


def trained_times(metadata):
    from comfy.ldm.minimax.model import time_shift_sigma
    raw = torch.tensor(metadata.raw_sigmas, dtype=torch.float32)
    grid = metadata.video_shift * raw / (1 + (metadata.video_shift - 1) * raw)
    current = (grid[:-1] * 1000.0 / 1000.0).clamp(min=1e-6)
    t = torch.cat((1 - current, 1 - time_shift_sigma(current, metadata.video_shift, metadata.audio_shift)))
    r = torch.cat((1 - grid[1:], 1 - time_shift_sigma(grid[1:], metadata.video_shift, metadata.audio_shift)))
    return t, r


def interpolate(table, t):
    pos = t.clamp(0, 1) * (table.shape[-2] - 1)
    i = pos.floor().long().clamp(max=table.shape[-2] - 2)
    return torch.lerp(table[..., i, :], table[..., i + 1, :], (pos - i).unsqueeze(-1))


def solve_projection(curve_weight, curve_bias, target, *, chunk=64):
    """Explicit least squares; all generated AND pinned residuals are measured."""
    if curve_weight.ndim != 2 or curve_weight.shape[1] != 8 or target.ndim != 2:
        raise ValueError("Fit needs an eight-coordinate native basis")
    if target.shape[1] != curve_weight.shape[0] or curve_bias.shape != (target.shape[1],):
        raise ValueError("Teacher and pruned projection output dimensions disagree")
    # Tall QR has full-rank precondition; never silently solve a rank-deficient
    # basis with CUDA gels' unspecified result. SVD spectrum is only eight wide.
    w64 = curve_weight.double()
    singular = torch.linalg.svdvals(w64)
    if not bool(torch.isfinite(singular).all()) or float(singular[-1]) <= float(singular[0]) * 1e-10:
        raise ValueError("Rank-deficient or ill-conditioned pruned curve basis")
    q, upper = torch.linalg.qr(w64, mode="reduced")
    coordinates = torch.linalg.solve_triangular(
        upper, q.T @ (target - curve_bias).double().T, upper=True,
    ).T.float()
    _finite(coordinates, "coordinates")
    relative = []
    for start in range(0, target.shape[0], chunk):
        actual = target[start:start + chunk]
        residual = F.linear(coordinates[start:start + chunk], curve_weight, curve_bias) - actual
        denominator = actual.norm(dim=1)
        if bool((denominator <= 0).any()):
            raise ValueError("Zero teacher modulation cannot define relative error")
        relative.extend((residual.norm(dim=1) / denominator).cpu().tolist())
    if not all(math.isfinite(x) for x in relative):
        raise ValueError("Non-finite approximation error")
    return coordinates, relative


@dataclass(frozen=True)
class CurveFit:
    path: Path
    sha256: str
    metadata: dict
    generated: torch.Tensor
    pinned: torch.Tensor
    t: torch.Tensor
    r: torch.Tensor

    def verify(self):
        if _sha256(self.path) != self.sha256:
            raise ValueError("Curve fit file bytes changed")
        if code_identity() != self.metadata["implementation"]:
            raise ValueError("Curve fit implementation/Core identity changed; rebuild explicitly")
        for name in ("generated", "pinned", "t", "r"):
            if _tensor_digest(getattr(self, name)) != self.metadata["tensor_sha256"][name]:
                raise ValueError("Curve fit tensor mutated: " + name)


def load_fit(path, *, base_sha256, adapter_sha256):
    """Exact content matching only, never a same-name/best-effort fallback."""
    path = Path(path).resolve(strict=True)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Curve fit exceeds 16MiB bound")
    before = _file(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if set(metadata) != {"t8_curve_fit"} or len(metadata["t8_curve_fit"]) > 262144:
            raise ValueError("Not a bounded T8 curve-fit contract")
        data = json.loads(metadata["t8_curve_fit"])
        if data.get("schema") != SCHEMA or set(handle.keys()) != {"generated", "pinned", "t", "r"}:
            raise ValueError("Unsupported curve fit schema/keys")
        if data["base"]["sha256"] != base_sha256 or data["adapter"]["sha256"] != adapter_sha256:
            raise ValueError("Curve fit does not match exact base and original adapter bytes")
        values = {name: handle.get_tensor(name) for name in handle.keys()}
    for name, shape in {"generated": (51, 16, 8), "pinned": (51, 1025, 8), "t": (16,), "r": (16,)}.items():
        value = values[name]
        if value.dtype != torch.float32 or tuple(value.shape) != shape:
            raise ValueError("Incorrect per-projection curve fit shape/dtype: " + name)
        _finite(value, name)
    if _file(path) != before:
        raise ValueError("Curve fit changed during loading")
    result = CurveFit(path, before["sha256"], data, **values)
    result.verify()
    return result


def build_fit(base_path, teacher_path, adapter_path, output, *, device="cpu", progress=None, cancel=None):
    """Build one new explicit artifact. Does not load/patch a MODEL or queue."""
    paths = {key: Path(path).resolve(strict=True) for key, path in (
        ("base", base_path), ("teacher", teacher_path), ("adapter", adapter_path))}
    output = Path(output).absolute()
    if output.exists() or output.is_symlink() or output.resolve() in paths.values():
        raise FileExistsError("Curve fit output already exists or aliases an input")
    identities = {name: _file(path) for name, path in paths.items()}
    implementation = code_identity()
    device = torch.device(device)
    with ExitStack() as stack, torch.inference_mode():
        handles = {key: stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
                   for key, path in paths.items()}
        base, teacher, adapter = (handles[k] for k in ("base", "teacher", "adapter"))
        bp, tp = _prefix(base), _prefix(teacher)
        meta = parse_metadata(adapter.metadata())
        plan = plan_source_keys(list(adapter.keys()))
        basis = basis_identity(base, bp)
        table = base.get_tensor(bp + "adaln_t_table").to(device).float()
        if table.shape != (1025, 8):
            raise ValueError("Expected native pruned H3 1025x8 time table")
        _finite(table, "original table")
        if tp + "adaln_t_table" in teacher.keys():
            raise ValueError("Fit teacher must be the full-time structure, not a pruned base")
        time_weights = {p: linear(teacher, tp + "time_embedder." + p, device)
                        for p in ("proj_in", "proj_out")}
        loras = {}
        for branch in ("time_embedder", "endpoint_time_embedder"):
            for p in ("proj_in", "proj_out"):
                entry = plan[branch + "." + p]
                a, b = (adapter.get_tensor(entry[k]).to(device).float() for k in ("A", "B"))
                w = time_weights[p][0]
                if a.shape != (meta.rank, w.shape[1]) or b.shape != (w.shape[0], meta.rank):
                    raise ValueError("HyperFlow time adapter/teacher shape mismatch")
                _finite(a, "time A")
                _finite(b, "time B")
                loras[branch, p] = (a, b)

        def embed(t, branch):
            half = time_weights["proj_in"][0].shape[1] // 2
            if half * 2 != time_weights["proj_in"][0].shape[1] or half <= 0:
                raise ValueError("Time teacher needs even positive Fourier width")
            freq = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=device) / half)
            angles = t[:, None] * freq
            x = torch.cat((angles.cos(), angles.sin()), dim=1)
            for p in ("proj_in", "proj_out"):
                a, b = loras[branch, p]
                x = F.linear(x, *time_weights[p]) + F.linear(F.linear(x, a), b) * (meta.alpha / meta.rank)
                if p == "proj_in":
                    x = F.silu(x)
            return x

        t, r = trained_times(meta)
        pin = torch.linspace(0, 1, PINNED_GRID, device=device, dtype=torch.float32)
        tt, rr = torch.cat((t.to(device), pin)), torch.cat((r.to(device), pin))
        embeddings = F.silu((1 - meta.gate) * embed(tt, "time_embedder")
                            + meta.gate * embed(rr, "endpoint_time_embedder"))
        _finite(embeddings, "teacher embeddings")
        backbone = interpolate(table, tt)
        fits, errors = [], []
        for index, name in enumerate(PROJECTIONS):
            if cancel is not None:
                cancel()
            fw, fb = linear(teacher, tp + name, device)
            cw, cb = linear(base, bp + name, device)
            target = F.linear(embeddings, fw, fb)
            del fw, fb
            _finite(target, name + " teacher modulation")
            fitted, residuals = solve_projection(cw, cb, target)
            prior = F.linear(backbone, cw, cb)
            before = ((prior - target).norm(dim=1) / target.norm(dim=1)).cpu().tolist()
            del target, prior, cw, cb
            fits.append(fitted.cpu())
            errors.append({"projection": name, "generated_before": sum(before[:16]) / 16,
                           "generated_after": sum(residuals[:16]) / 16,
                           "pinned_before": sum(before[16:]) / PINNED_GRID,
                           "pinned_after": sum(residuals[16:]) / PINNED_GRID,
                           "worst_after": max(residuals)})
            if progress is not None:
                progress(index + 1, errors[-1])
        fitted = torch.stack(fits)
        tensors = {"generated": fitted[:, :16].contiguous(), "pinned": fitted[:, 16:].contiguous(),
                   "t": t.contiguous(), "r": r.contiguous()}
    if ({key: _file(path) for key, path in paths.items()} != identities
            or code_identity() != implementation):
        raise ValueError("Actual fit source file or implementation changed during fitting")
    metadata = {"schema": SCHEMA, "reference_revision": REFERENCE, **identities,
                "basis_sha256": basis, "implementation": implementation,
                "gate": meta.gate, "strength": 1.0, "raw_sigmas": list(meta.raw_sigmas),
                "video_shift": meta.video_shift, "audio_shift": meta.audio_shift,
                "errors": errors, "teacher_role": "explicit_selected_full_structure_not_verified_derivation",
                "quality_verified": False, "tensor_sha256": {k: _tensor_digest(v) for k, v in tensors.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".safetensors.part", delete=False) as temporary:
        temp = Path(temporary.name)
    try:
        save_file(tensors, str(temp), metadata={"t8_curve_fit": canonical(metadata)})
        # No replace: even a concurrent creator's new artifact is preserved.
        os.link(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    return load_fit(output, base_sha256=identities["base"]["sha256"],
                    adapter_sha256=identities["adapter"]["sha256"])
