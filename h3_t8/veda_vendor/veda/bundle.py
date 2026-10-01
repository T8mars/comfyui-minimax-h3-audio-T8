# SPDX-License-Identifier: MIT
# Vendored from veda-sparse/Miowtion at 3bc5e5ae53e8c0d4539f2bc872265164947d013a.
"""Self-contained predictor bundle: live weights plus the tile plans.

A training checkpoint is not a deployment artifact. It carries the EMA
shadow next to the live weights (twice the size for something inference
never reads), it is a `torch.load` pickle, and it says nothing about which
tile shapes the predictor was trained against -- those live in a separate
plan directory that has to be shipped and named correctly alongside it.
Getting that pairing wrong is silent: the predictor still produces scores,
they are just scores for a tiling it never saw.

The bundle is one safetensors file holding the live predictor weights, with
every plan of the plan table serialized into its `__metadata__` (which
safetensors defines as str -> str, so the plans go in as one JSON blob).
Geometry, tile shapes and the weights therefore travel together and a
mismatch is impossible by construction.

EMA is deliberately dropped: which of the two the run should deploy is a
decision made once, at export time, and recorded in `source_weights`.

Weights are stored in bf16 by default. `LayerPredictor.embed` upcasts the
projection it selects (`proj.index_select(0, heads).float()`), so the
scoring arithmetic is fp32 whatever the storage dtype is; keeping fp32 on
disk only doubles the file and, more importantly, doubles the resident copy
each inference replica holds on its own card.

`float8_e4m3fn` is offered one step below bf16, with a per-head amax scale
alongside each tensor. It loads back as bf16 parameters (bmm has no e4m3
path), so it halves the file and the load, not the resident copy; its real
use is to measure what three mantissa bits do to the selected block set.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..h3 import geometry as h3_geometry
from . import plan as veda_plan
from . import predictor as veda_predictor
from . import tiling

FORMAT = 'miowtion-veda-predictor-v1'

_PREFIX = 'predictor.'

# Storage dtypes. e4m3 carries 3 mantissa bits, so it is only offered with
# a per-head amax scale (see `_quantize_fp8`): the projections of different
# heads differ in scale by more than e4m3's exponent range leaves room for
# once the mantissa is that short. Whether the rounding reorders the top-k
# is a measurement, not an assumption -- see docs/features/quant_scoring.md.
DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
          'float8_e4m3fn': torch.float8_e4m3fn}

# Largest finite e4m3 value; the per-head amax is mapped onto it.
_E4M3_MAX = 448.0

# Suffix of the per-head scale that accompanies every fp8 tensor. It is not
# a state-dict key, so it is stripped before the weights reach the module.
_SCALE_SUFFIX = '.__scale'

# Bundles written before the dtype was recorded are fp32.
_LEGACY_DTYPE = 'float32'


def _quantize_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Splits `[H, ...]` weights into e4m3 values and a per-head scale.

    The scale maps each head's amax onto the largest finite e4m3 value, so
    the whole exponent budget of the format is spent on that head's own
    range. A head of all zeros gets a tiny scale and stays zero.

    Returns:
        (values `[H, ...]` e4m3, scale `[H]` fp32) with
        `weight ~= values * scale`.
    """
    dims = tuple(range(1, weight.dim()))
    amax = weight.float().abs().amax(dim=dims)
    scale = (amax / _E4M3_MAX).clamp(min=torch.finfo(torch.float32).tiny)
    values = weight.float() / scale.view(-1, *([1] * len(dims)))
    return values.to(torch.float8_e4m3fn).contiguous(), scale.contiguous()


def _dequantize_fp8(values: torch.Tensor, scale: torch.Tensor,
                    dtype: torch.dtype) -> torch.Tensor:
    """Inverse of `_quantize_fp8`, into `dtype`."""
    dims = values.dim() - 1
    return (values.float() * scale.view(-1, *([1] * dims))).to(dtype)


@dataclasses.dataclass
class Bundle:
    """A loaded bundle.

    Attributes:
        predictor: The predictor, on the requested device, in eval mode.
        plans: Tile plans the predictor was trained against.
        keep_ratio: Keep ratio the run trained with; a default for
            inference, not a constraint.
        metadata: The file's raw string metadata (provenance).
    """

    predictor: veda_predictor.TileScorePredictor
    plans: veda_plan.PlanTable
    keep_ratio: float
    metadata: dict[str, str]


def save(path: str, weights: dict[str, torch.Tensor],
         plans: veda_plan.PlanTable, *, num_layers: int, num_heads: int,
         head_dim: int, keep_ratio: float, source: str,
         source_weights: str, step: int,
         dtype: torch.dtype = torch.bfloat16) -> None:
    """Writes a bundle.

    Args:
        path: Destination `.safetensors` file.
        weights: Predictor state dict, keys either bare (`layers.0.proj_q`)
            or checkpoint style (`predictor.layers.0.proj_q`).
        plans: Plans to embed; every plan of the table is written.
        num_layers: Layers the predictor was built with.
        num_heads: Heads per layer.
        head_dim: Head dimension.
        keep_ratio: Keep ratio the run trained with.
        source: Checkpoint directory the weights came from (provenance).
        source_weights: 'live' or 'ema' (which of the two was exported).
        step: Training update the checkpoint was written at.
        dtype: Storage dtype; must be one of `DTYPES`.

    Raises:
        ValueError: If `weights` is empty, mixes prefixed and bare keys, or
            `dtype` is not a supported storage dtype.
    """
    if not weights:
        raise ValueError('no predictor weights to save')
    names = {v: k for k, v in DTYPES.items()}
    if dtype not in names:
        raise ValueError(f'unsupported storage dtype {dtype}; '
                         f'expected one of {sorted(DTYPES)}')
    prefixed = [k for k in weights if k.startswith(_PREFIX)]
    if prefixed and len(prefixed) != len(weights):
        raise ValueError('weights mix prefixed and bare keys')
    bare = {k[len(_PREFIX):] if prefixed else k: v
            for k, v in weights.items()}
    if dtype is torch.float8_e4m3fn:
        tensors = {}
        for key, value in bare.items():
            values, scale = _quantize_fp8(value)
            tensors[key] = values
            tensors[key + _SCALE_SUFFIX] = scale
    else:
        tensors = {k: v.to(dtype).contiguous() for k, v in bare.items()}
    if not plans.plans:
        raise ValueError('bundle without plans; pass the run\'s plan table')
    metadata = {
        'format': FORMAT,
        'num_layers': str(num_layers),
        'num_heads': str(num_heads),
        'head_dim': str(head_dim),
        'dtype': names[dtype],
        'keep_ratio': repr(float(keep_ratio)),
        'source': source,
        'source_weights': source_weights,
        'step': str(step),
        'plans': json.dumps({name: p.to_json()
                             for name, p in sorted(plans.plans.items())}),
    }
    save_file(tensors, path, metadata=metadata)


def read_metadata(path: str) -> dict[str, str]:
    """The file's string metadata, without reading a single tensor.

    Raises:
        ValueError: If the file is not a bundle of this format.
    """
    with safe_open(path, framework='pt', device='cpu') as f:
        metadata = dict(f.metadata() or {})
    if metadata.get('format') != FORMAT:
        raise ValueError(f'{path}: not a {FORMAT} bundle '
                         f'(format {metadata.get("format")!r})')
    return metadata


def load(path: str, device: torch.device | str = 'cpu') -> Bundle:
    """Loads a bundle, strictly: every stored tensor must land on a parameter.

    Args:
        path: A `.safetensors` file written by `save()`.
        device: Where to put the predictor.

    Returns:
        The bundle.

    Raises:
        ValueError: If the file is not a bundle of this format, or its
            tensors do not match the predictor the metadata describes.
    """
    metadata = read_metadata(path)
    with safe_open(path, framework='pt', device='cpu') as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    stored = metadata.get('dtype', _LEGACY_DTYPE)
    if stored not in DTYPES:
        raise ValueError(f'{path}: unknown storage dtype {stored!r}')
    model = veda_predictor.TileScorePredictor(int(metadata['num_layers']),
                                              int(metadata['num_heads']),
                                              int(metadata['head_dim']))
    resident = DTYPES[stored]
    if stored == 'float8_e4m3fn':
        # The parameters come back in bf16: `LayerPredictor.embed` reads
        # them with `.float()` and `torch.bmm`, neither of which takes an
        # e4m3 tensor. fp8 therefore buys a file (and a host-to-device
        # copy) half the size of bf16, not a smaller resident copy; what
        # the bundle keeps is exactly the rounding, so its effect on the
        # top-k is measurable against the bf16 export of the same step.
        resident = torch.bfloat16
        scales = {k[:-len(_SCALE_SUFFIX)]: v for k, v in tensors.items()
                  if k.endswith(_SCALE_SUFFIX)}
        values = {k: v for k, v in tensors.items()
                  if not k.endswith(_SCALE_SUFFIX)}
        missing = sorted(set(values) - set(scales))
        if missing:
            raise ValueError(f'{path}: fp8 tensors without a '
                             f'{_SCALE_SUFFIX} scale: {missing}')
        tensors = {k: _dequantize_fp8(v, scales[k], resident)
                   for k, v in values.items()}
    # Before the load, not after: load_state_dict copies into the existing
    # parameter, so an fp32 module would silently upcast a bf16 file back to
    # fp32 and hold twice the memory the bundle was exported to save.
    model = model.to(resident)
    shape = (f'{metadata["num_layers"]} layers x {metadata["num_heads"]} '
             f'heads x {metadata["head_dim"]}')
    try:
        model.load_state_dict(tensors, strict=True)
    except RuntimeError as error:
        raise ValueError(f'{path}: predictor weights do not match the '
                         f'{shape} the metadata declares: {error}') from error
    plans = veda_plan.PlanTable(
        [veda_plan.TilePlan.from_json(p)
         for p in json.loads(metadata['plans']).values()])
    return Bundle(predictor=model.to(device).eval(), plans=plans,
                  keep_ratio=float(metadata['keep_ratio']),
                  metadata=metadata)


def random_bundle(num_layers: int, num_heads: int, head_dim: int,
                  geometries: Sequence[h3_geometry.Geometry],
                  keep_ratio: float, seed: int = 0,
                  device: torch.device | str = 'cpu') -> Bundle:
    """A bundle with a randomly initialized predictor, for benchmarks.

    What a Veda step costs is fixed by the budget (the kept fraction of
    blocks), the plan's tile shapes and the predictor's shape, not by the
    scores: top-k keeps the same number of blocks whatever it ranks. So a
    random predictor times the same as a trained one; its masks are just
    arbitrary.

    The plan of each geometry is uniform over the least-padding tile shape
    of its video grid. A searched plan mixes up to a few shapes per layer
    (miowtion.veda.search), so its padding, and hence its cost, can differ
    slightly from this bootstrap plan.

    The predictor is resident in bf16, as a loaded fp8 or bf16 bundle is.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        predictor = veda_predictor.TileScorePredictor(num_layers, num_heads,
                                                      head_dim)
    plans = veda_plan.PlanTable([
        veda_plan.TilePlan.uniform(
            g, tiling.least_padding_shape(g.video_grid), num_layers,
            num_heads) for g in geometries])
    return Bundle(predictor=predictor.to(device, torch.bfloat16).eval(),
                  plans=plans, keep_ratio=keep_ratio,
                  metadata={'source': 'random', 'seed': str(seed)})
