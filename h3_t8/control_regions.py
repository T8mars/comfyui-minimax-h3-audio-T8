"""Explicit source-crop geometry and original-target-canvas region coverage.

No model, sampler, output compositing or uniform latent-time interpolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math

import torch
import torch.nn.functional as F


CROP_SCHEMA = "t8.h3.control_crop.v1"
GEOMETRY_SCHEMA = "t8.h3.control_geometry.v1"
FRAME_BINS = (1, 4, 4, 4, 4)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def tensor_sha(value):
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256(canonical({"shape": list(tensor.shape), "dtype": str(tensor.dtype)}).encode())
    raw = tensor.view(torch.uint8).reshape(-1)
    for start in range(0, raw.numel(), 8 * 1024 * 1024):
        digest.update(raw[start:start + 8 * 1024 * 1024].numpy().tobytes())
    return digest.hexdigest()


def _frame_map(value, length, source_count, name):
    if value is None:
        return tuple(min(index, source_count - 1) for index in range(length))
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must have exactly {length} explicit source frame indices")
    if any(type(index) is not int or not 0 <= index < source_count for index in value):
        raise ValueError(f"{name} indices must be integers in [0, {source_count})")
    return tuple(value)


def _box(value):
    if not isinstance(value, list) or len(value) != 4 or any(
            type(item) not in (int, float) or not math.isfinite(item) for item in value):
        raise ValueError("crop boxes must be finite [left, top, right, bottom] pixel edges")
    left, top, right, bottom = map(float, value)
    if (right <= left or bottom <= top
            or not math.isfinite(right - left) or not math.isfinite(bottom - top)):
        raise ValueError("crop boxes must have finite positive width and height")
    return left, top, right, bottom


def _area_weights(source, target):
    # Exact normalized pixel-cell overlap geometry, evaluated in float32 below.
    edges = torch.arange(target + 1, dtype=torch.float64) * (source / target)
    pixels = torch.arange(source, dtype=torch.float64)
    overlap = (torch.minimum(edges[1:, None], pixels[None] + 1)
               - torch.maximum(edges[:-1, None], pixels[None])).clamp_min(0)
    return (overlap / (source / target)).float()


@dataclass(eq=False)
class RegionGate:
    mask: torch.Tensor
    frame_indices: tuple
    width: int
    height: int
    length: int
    _cache: dict = field(default_factory=dict, repr=False)

    def rows(self, latent_t, latent_h, latent_w):
        key = (int(latent_t), int(latent_h), int(latent_w))
        if (latent_h * 16, latent_w * 16) != (self.height, self.width) or latent_h % 2 or latent_w % 2:
            raise ValueError("Region gate must map the actual stage canvas and 2x2 latent patches")
        bins = [FRAME_BINS[index % 5] for index in range(latent_t)]
        if sum(bins) != self.length:
            raise ValueError("Region gate temporal frame bins do not match this stage length")
        if key not in self._cache:
            masks = self.mask
            wy, wx = _area_weights(masks.shape[1], latent_h // 2), _area_weights(masks.shape[2], latent_w // 2)
            reduced = (wy @ masks) @ wx.T
            # Preserve the exact identity/zero of uniform planes even when
            # float32 overlap weights do not sum to exactly one.
            full = (masks == 1).all(dim=(-2, -1))[:, None, None]
            empty = (masks == 0).all(dim=(-2, -1))[:, None, None]
            reduced = torch.where(full, torch.ones_like(reduced), reduced)
            reduced = torch.where(empty, torch.zeros_like(reduced), reduced)
            selected = reduced.index_select(0, torch.tensor(self.frame_indices, dtype=torch.long))
            blocks, start = [], 0
            for count in bins:
                blocks.append(selected[start:start + count].mean(dim=0))
                start += count
            self._cache[key] = torch.stack(blocks).clamp(0, 1).reshape(-1, 1)
        return self._cache[key]

    def report(self):
        return {"mask_sha256": tensor_sha(self.mask), "original_mask_shape": list(self.mask.shape),
                "frame_indices": list(self.frame_indices),
                "target": {"width": self.width, "height": self.height, "length": self.length},
                "coordinate_space": "original_target_canvas_normalized_independent_of_source_crop",
                "coverage": "pixel_cell_overlap_area_then_native_1_4_4_4_4_frame_mean_float32"}


def prepare_control_geometry(control_video, *, width, height, length, fit_mode, fit_video,
                             region_mask=None, crop_plan_json=""):
    if (not torch.is_tensor(control_video) or control_video.ndim != 4
            or control_video.shape[-1] < 3 or any(size <= 0 for size in control_video.shape)
            or not control_video.is_floating_point() or not bool(torch.isfinite(control_video).all())):
        raise ValueError("control_video must contain finite floating IMAGE frames [T,H,W,C]")
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("Control geometry requires a positive 32-aligned target canvas")
    if type(length) is not int or length < 5 or length % 17 != 5:
        raise ValueError("Control geometry length must follow the H3 17n+5 frame grid")
    if fit_mode not in {"exact", "center_crop", "stretch"}:
        raise ValueError("Unknown original control fit_mode")
    try:
        spec = json.loads(crop_plan_json) if str(crop_plan_json).strip() else {}
    except (ValueError, TypeError) as error:
        raise ValueError("crop_plan_json must be a JSON object") from error
    if not isinstance(spec, dict):
        raise ValueError("crop_plan_json must be a JSON object")
    allowed = {"schema", "source_size", "frame_indices", "boxes", "fit", "flip_x", "flip_y", "region_frame_indices"}
    if set(spec) - allowed:
        raise ValueError("Unknown crop fields: " + ", ".join(sorted(set(spec) - allowed)))
    if spec.get("schema", CROP_SCHEMA) != CROP_SCHEMA:
        raise ValueError("Unsupported control crop schema")
    source_t, source_h, source_w = control_video.shape[:3]
    if "source_size" in spec and spec["source_size"] != [source_w, source_h]:
        raise ValueError("Declared crop source_size does not match actual source pixels")
    indices = _frame_map(spec.get("frame_indices"), length, source_t, "frame_indices")
    for name in ("flip_x", "flip_y"):
        if name in spec and type(spec[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    boxes = spec.get("boxes", [[0, 0, source_w, source_h]])
    if not isinstance(boxes, list) or len(boxes) not in (1, source_t):
        raise ValueError("boxes must contain one box or one box per source frame")
    boxes = tuple(_box(value) for value in boxes)
    explicit_crop = any(name in spec for name in ("boxes", "fit", "flip_x", "flip_y"))
    transforms = []
    if not explicit_crop:
        source = control_video
        if "frame_indices" in spec:
            source = source.index_select(0, torch.tensor(indices, device=source.device))
        frames = fit_video(source, width=width, height=height, length=length, fit_mode=fit_mode).detach()
        # Exact fit may return the caller's view; only our own snapshot is bound.
        if frames.untyped_storage().data_ptr() == control_video.untyped_storage().data_ptr():
            frames = frames.clone()
    else:
        fit = spec.get("fit", {"center_crop": "cover"}.get(fit_mode, fit_mode))
        if fit not in {"exact", "stretch", "contain", "cover"}:
            raise ValueError("Explicit crop fit must be exact, stretch, contain or cover")
        xx = torch.arange(width, device=control_video.device, dtype=torch.float64) + .5
        yy = torch.arange(height, device=control_video.device, dtype=torch.float64) + .5
        output = []
        for index in indices:
            left, top, right, bottom = boxes[0 if len(boxes) == 1 else index]
            cw, ch = right - left, bottom - top
            if fit == "exact" and (cw, ch) != (width, height):
                raise ValueError("Exact crop dimensions must equal the actual target canvas")
            rw, rh = width, height
            if fit in {"contain", "cover"}:
                scale = (min if fit == "contain" else max)(width / cw, height / ch)
                rw, rh = max(1, round(cw * scale)), max(1, round(ch * scale))
            dx, dy = (width - rw) // 2, (height - rh) // 2
            ux, uy = (xx - dx) / rw, (yy - dy) / rh
            if spec.get("flip_x", False):
                ux = 1 - ux
            if spec.get("flip_y", False):
                uy = 1 - uy
            sx, sy = left + ux * cw, top + uy * ch
            gy, gx = torch.meshgrid(sy * (2 / source_h) - 1, sx * (2 / source_w) - 1, indexing="ij")
            # Coordinates beyond the bilinear support already mean zero
            # padding. Clamp only that exterior before float32 conversion so
            # a finite, distant virtual crop cannot become an inf/NaN grid.
            grid = torch.stack((gx, gy), -1).clamp(-3, 3).float()[None]
            frame = F.grid_sample(control_video[index:index + 1, ..., :3].movedim(-1, 1).float(),
                                  grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            if fit == "contain":
                visible = ((xx >= dx) & (xx < dx + rw))[None] & ((yy >= dy) & (yy < dy + rh))[:, None]
                frame = frame * visible
            output.append(frame[0].movedim(0, -1))
            transforms.append({"source_frame": index, "box_xyxy": [left, top, right, bottom],
                               "resized_extent": [rw, rh], "offset_xy": [dx, dy], "fit": fit})
        frames = torch.stack(output).to(control_video.dtype).detach()

    gate = None
    if region_mask is not None:
        if (not torch.is_tensor(region_mask) or region_mask.ndim != 3
                or any(size <= 0 for size in region_mask.shape) or not region_mask.is_floating_point()
                or not bool(torch.isfinite(region_mask).all())
                or not bool(((region_mask >= 0) & (region_mask <= 1)).all())):
            raise ValueError("region_mask must be a nonempty finite MASK [T,H,W] with coverage in [0,1]")
        count = region_mask.shape[0]
        if "region_frame_indices" in spec:
            region_indices = _frame_map(spec["region_frame_indices"], length, count, "region_frame_indices")
        elif count == 1:
            region_indices = (0,) * length
        elif count == source_t:
            region_indices = indices
        else:
            raise ValueError("A moving region MASK must match source frame count or supply region_frame_indices")
        gate = RegionGate(region_mask.detach().to(device="cpu", dtype=torch.float32).clone(),
                          region_indices, width, height, length)
    elif "region_frame_indices" in spec:
        raise ValueError("region_frame_indices requires an actual region_mask")

    report = {"schema": GEOMETRY_SCHEMA, "source": {"width": source_w, "height": source_h, "frames": source_t},
              "source_sha256": tensor_sha(control_video), "prepared_sha256": tensor_sha(frames),
              "target": {"width": width, "height": height, "length": length},
              "frame_indices": list(indices), "crop_requested": explicit_crop,
              "crop_spec": spec, "transforms": transforms, "original_fit_mode": fit_mode,
              "interpolation": "bilinear_pixel_centers_zeros_align_corners_false" if explicit_crop else "unchanged_original_fit_video",
              "region": gate.report() if gate else None, "output_rgb_compositing": False}
    report["geometry_sha256"] = hashlib.sha256(canonical(report).encode()).hexdigest()
    return frames, gate, report
