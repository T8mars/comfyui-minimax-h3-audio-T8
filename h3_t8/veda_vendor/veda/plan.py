# SPDX-License-Identifier: MIT
# Vendored from veda-sparse/Miowtion at 3bc5e5ae53e8c0d4539f2bc872265164947d013a.
"""Tile plans: which tile shape every (layer, head) uses on one geometry.

A plan is produced by the tile search (search.py) for one target geometry
and is timestep independent. Every geometry has its own plan, either
searched on its own grid or, by explicit choice, derived from the searched
plan of the H<->W mirrored aspect (`TilePlan.mirrored`, e.g. 9:16 from 16:9;
recorded in the provenance). Training and search must pick plans with the
same rule (`PlanTable.select`).

The keep ratio a plan was searched with is recorded for provenance only;
training always uses its own runtime budget.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
from collections.abc import Sequence

import torch

from ..h3 import geometry as h3_geometry
from . import tiling


@dataclasses.dataclass(frozen=True)
class HeadGroup:
    """Heads of one layer that share a tile shape."""

    shape: tiling.TileShape
    heads: torch.Tensor  # [H'] int64


@dataclasses.dataclass
class TilePlan:
    """Per-(layer, head) tile shapes for one geometry.

    Attributes:
        geometry: Geometry name, e.g. '16x9_t37'.
        grid: Target token grid (T, H, W) the plan was searched on.
        shapes: Distinct shapes referenced by `head_shape`.
        head_shape: [num_layers][num_heads] index into `shapes`.
        provenance: Search metadata (keep ratio, entries, MSEs, ...).
    """

    geometry: str
    grid: tuple[int, int, int]
    shapes: list[tiling.TileShape]
    head_shape: list[list[int]]
    provenance: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        for layer, row in enumerate(self.head_shape):
            if len(set(row)) > 2:
                raise ValueError(f'layer {layer} uses {len(set(row))} shapes; '
                                 'at most 2 per layer are allowed')
        self._groups = {}

    @property
    def num_layers(self) -> int:
        return len(self.head_shape)

    def layer_shapes(self, layer: int) -> list[tiling.TileShape]:
        return [self.shapes[i] for i in sorted(set(self.head_shape[layer]))]

    def head_groups(self, layer: int,
                    device: torch.device | str) -> list[HeadGroup]:
        """Groups of heads sharing a shape (cached per layer and device)."""
        key = (layer, str(device))
        if key not in self._groups:
            row = torch.tensor(self.head_shape[layer])
            self._groups[key] = [
                HeadGroup(self.shapes[i],
                          torch.nonzero(row == i).view(-1).to(device))
                for i in sorted(set(self.head_shape[layer]))
            ]
        return self._groups[key]

    def to_json(self) -> dict:
        return {
            'geometry': self.geometry,
            'grid': list(self.grid),
            'shapes': [str(s) for s in self.shapes],
            'head_shape': self.head_shape,
            'provenance': self.provenance,
        }

    @classmethod
    def from_json(cls, data: dict) -> TilePlan:
        return cls(geometry=data['geometry'], grid=tuple(data['grid']),
                   shapes=[tiling.TileShape.parse(s) for s in data['shapes']],
                   head_shape=[list(r) for r in data['head_shape']],
                   provenance=data.get('provenance', {}))

    def save(self, path: str) -> None:
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.to_json(), f, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> TilePlan:
        with open(path) as f:
            return cls.from_json(json.load(f))

    @classmethod
    def uniform(cls, geometry: h3_geometry.Geometry, shape: tiling.TileShape,
                num_layers: int, num_heads: int) -> TilePlan:
        """Every head uses one shape (baseline / bootstrap plan)."""
        return cls(geometry.name, geometry.video_grid, [shape],
                   [[0] * num_heads for _ in range(num_layers)],
                   {'source': 'uniform'})

    def mirrored(self) -> TilePlan:
        """The plan of the H<->W mirrored geometry, e.g. 9x16_t37 from 16x9_t37.

        Shapes and grid are transposed and the plan takes the mirrored
        geometry's name, so `PlanTable.select` matches it exactly.

        Raises:
            ValueError: If the mirrored geometry's grid is not the transpose
                of this plan's grid (canvas rounding differs).
        """
        aspect, latent_t = self.geometry.split('_t')
        w, h = aspect.split('x')
        mirror = h3_geometry.geometry_from_latent_t(f'{h}:{w}', int(latent_t))
        t, grid_h, grid_w = self.grid
        if tuple(mirror.video_grid) != (t, grid_w, grid_h):
            raise ValueError(f'{mirror.name} grid {mirror.video_grid} is not '
                             f'the transpose of {self.geometry} {self.grid}')
        return TilePlan(mirror.name, (t, grid_w, grid_h),
                        [s.transposed() for s in self.shapes],
                        [list(r) for r in self.head_shape],
                        {**self.provenance, 'mirrored_from': self.geometry})

    def transposed(self) -> TilePlan:
        """H<->W mirrored plan (fallback only, see PlanTable.select)."""
        t, h, w = self.grid
        return TilePlan(self.geometry + '_T', (t, w, h),
                        [s.transposed() for s in self.shapes],
                        [list(r) for r in self.head_shape],
                        {**self.provenance, 'transposed_from': self.geometry})


class PlanTable:
    """All searched plans; picks the plan for a geometry."""

    def __init__(self, plans: Sequence[TilePlan]):
        self.plans = {p.geometry: p for p in plans}

    @classmethod
    def load_dir(cls, directory: str) -> PlanTable:
        paths = sorted(glob.glob(os.path.join(directory, '*.json')))
        if not paths:
            raise FileNotFoundError(f'no plans in {directory}')
        return cls([TilePlan.load(p) for p in paths])

    def select(self, geometry: h3_geometry.Geometry) -> TilePlan:
        """Plan for `geometry`.

        Rule: the exact geometry if searched; else the same aspect ratio with
        the nearest latent_t (no interpolation); else, only when no plan of
        this aspect exists, the transposed plan of the mirrored aspect whose
        shapes pad the target grid least.

        Raises:
            KeyError: If no plan of this or the mirrored aspect exists.
        """
        if geometry.name in self.plans:
            return self.plans[geometry.name]
        aspect_key = geometry.aspect.replace(':', 'x')
        same = [p for p in self.plans.values()
                if p.geometry.split('_t')[0] == aspect_key]
        if same:
            return min(same, key=lambda p: (abs(p.grid[0] - geometry.latent_t),
                                            p.grid[0]))
        w, h = geometry.aspect.split(':')
        mirrored = [p for p in self.plans.values()
                    if p.geometry.split('_t')[0] == f'{h}x{w}']
        if not mirrored:
            raise KeyError(f'no plan for {geometry.name}')
        nearest = min(mirrored,
                      key=lambda p: abs(p.grid[0] - geometry.latent_t))
        grid = geometry.video_grid
        pad = lambda plan: sum(s.num_tiles(grid) for s in plan.shapes)
        return min((nearest, nearest.transposed()), key=pad)
