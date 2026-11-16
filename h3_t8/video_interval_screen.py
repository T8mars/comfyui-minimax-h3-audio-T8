"""Offline review hints. Never rejects, edits or rerenders a user's video."""

from __future__ import annotations

import math
from collections import deque

import numpy as np


def _tiles(values):
    return np.array(
        [
            float(part.mean())
            for row in np.array_split(values, 5, axis=0)
            for part in np.array_split(row, 8, axis=1)
        ]
    )


def frame_observations(frames, region=None):
    """Consume (seconds, small grayscale [H,W] floats) in presentation order.

    Holds three small frames, not a full decoded movie. Region is explicit
    normalized xyxy; no face detection or identity classifier is implied.
    """
    if region is not None:
        if len(region) != 4 or not all(math.isfinite(x) for x in region):
            raise ValueError("region needs four finite normalized coordinates")
        if not 0 <= region[0] < region[2] <= 1 or not 0 <= region[1] < region[3] <= 1:
            raise ValueError(
                "region must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1"
            )
    rows, history = [], deque(maxlen=2)
    shape = None
    for index, (time, frame) in enumerate(frames):
        time = float(time)
        image = np.asarray(frame, dtype=np.float32)
        if not math.isfinite(time) or (rows and time <= rows[-1]["seconds"]):
            raise ValueError(
                "presentation timestamps must be finite and strictly increasing"
            )
        if image.ndim != 2 or min(image.shape) < 8 or not np.isfinite(image).all():
            raise ValueError("expected finite small grayscale frames [H,W]")
        if image.min() < 0 or image.max() > 1:
            raise ValueError("grayscale values must be in [0,1]")
        if shape is None:
            shape = image.shape
        if image.shape != shape:
            raise ValueError("frame dimensions changed within this screening stream")
        crop = image
        if region is not None:
            h, w = shape
            x1, y1, x2, y2 = region
            crop = image[
                int(y1 * h) : max(int(y1 * h) + 1, math.ceil(y2 * h)),
                int(x1 * w) : max(int(x1 * w) + 1, math.ceil(x2 * w)),
            ]
        row = {
            "frame": index,
            "seconds": time,
            "mean_luma": float(image.mean()),
            "delta": 0.0,
            "structural_delta": 0.0,
            "changed_tile_fraction": 0.0,
            "region_delta": 0.0,
            "region_return_delta": None,
            "return_delta": None,
        }
        if history:
            prior, prior_crop = history[-1]
            delta = np.abs(image - prior)
            row.update(
                delta=float(delta.mean()),
                structural_delta=float(
                    np.abs((image - image.mean()) - (prior - prior.mean())).mean()
                ),
                changed_tile_fraction=float((_tiles(delta) > 0.10).mean()),
                region_delta=float(np.abs(crop - prior_crop).mean())
                if region is not None
                else float(_tiles(delta).max()),
            )
        if len(history) == 2:
            row["return_delta"] = float(np.abs(image - history[0][0]).mean())
            row["region_return_delta"] = (
                float(np.abs(crop - history[0][1]).mean())
                if region is not None
                else float(_tiles(np.abs(image - history[0][0])).max())
            )
        rows.append(row)
        history.append((image.copy(), crop.copy()))
    if not rows:
        raise ValueError("video contains no decoded frames")
    return rows


def suspected_intervals(
    rows,
    *,
    cut_threshold=0.22,
    local_threshold=0.08,
    expected_cuts=(),
    has_region=False,
):
    for value in (cut_threshold, local_threshold):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("screening thresholds must be finite and positive")
    expected = [float(x) for x in expected_cuts]
    if any(not math.isfinite(x) or x < 0 for x in expected):
        raise ValueError("expected cut times must be finite and nonnegative")
    intervals, transient_edges = [], set()
    for i in range(1, len(rows) - 1):
        a, b, c = rows[i - 1 : i + 2]
        if (
            b["delta"] > cut_threshold
            and c["delta"] > cut_threshold
            and c["return_delta"] < cut_threshold / 4
        ):
            intervals.append(
                {
                    "kind": "transient_flash_or_occlusion",
                    "start": a["seconds"],
                    "end": c["seconds"],
                    "focus": b["seconds"],
                    "label": "短暂闪光或遮挡，复看是否符合创作意图",
                }
            )
            transient_edges.update((i, i + 1))
        elif (
            b["region_delta"] > local_threshold
            and c["region_delta"] > local_threshold
            and c["region_return_delta"] < local_threshold / 3
            and b["changed_tile_fraction"] < 0.6
        ):
            intervals.append(
                {
                    "kind": "local_transient",
                    "start": a["seconds"],
                    "end": c["seconds"],
                    "focus": b["seconds"],
                    "label": "疑似局部闪动，也可能是快速运动或遮挡",
                }
            )
            transient_edges.update((i, i + 1))
    for i, row in enumerate(rows[1:], 1):
        if i in transient_edges:
            continue
        if row["delta"] > cut_threshold and row["changed_tile_fraction"] >= 0.6:
            dt = row["seconds"] - rows[i - 1]["seconds"]
            intentional = any(abs(t - row["seconds"]) <= dt for t in expected)
            intervals.append(
                {
                    "kind": "expected_cut" if intentional else "possible_cut",
                    "start": rows[i - 1]["seconds"],
                    "end": row["seconds"],
                    "focus": row["seconds"],
                    "label": "已标注的正常剪辑"
                    if intentional
                    else "疑似画面突变，复看是否为正常剪辑",
                }
            )
        elif (
            has_region
            and row["region_delta"] > max(local_threshold * 2, cut_threshold)
            and row["delta"] < cut_threshold
        ):
            intervals.append(
                {
                    "kind": "region_appearance_change",
                    "start": rows[i - 1]["seconds"],
                    "end": row["seconds"],
                    "focus": row["seconds"],
                    "label": "所选区域外观变化；检查身份、动作和遮挡，不是自动身份结论",
                }
            )
    intervals.sort(key=lambda row: (row["focus"], row["kind"]))
    return {
        "schema": "t8.offline.video_intervals.v1",
        "status": "review_hints_only",
        "frame_count": len(rows),
        "first_pts": rows[0]["seconds"],
        "last_pts": rows[-1]["seconds"],
        "intervals": intervals,
        "identity_assessment": "manual_comparison_required",
        "thresholds": {"cut": cut_threshold, "local": local_threshold},
        "source_changed": False,
        "automatic_retry": False,
        "quality_accepted": False,
    }


def screen_video(
    path,
    region=None,
    *,
    cut_threshold=0.22,
    local_threshold=0.08,
    expected_cuts=(),
    cancel_check=None,
):
    import av

    fallback_pts = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate is not None else None

        def frames():
            for index, frame in enumerate(container.decode(stream)):
                if cancel_check is not None:
                    cancel_check()
                time = frame.time
                if time is None:
                    if rate is None or rate <= 0:
                        raise ValueError(
                            "missing PTS and no average frame rate for explicit CFR inference"
                        )
                    time = index / rate
                    fallback_pts.append(index)
                height = max(8, round(frame.height * 160 / frame.width))
                small = (
                    frame.reformat(width=160, height=height, format="gray")
                    .to_ndarray()
                    .astype(np.float32)
                    / 255.0
                )
                yield time, small

        rows = frame_observations(frames(), region)
    report = suspected_intervals(
        rows,
        cut_threshold=cut_threshold,
        local_threshold=local_threshold,
        expected_cuts=expected_cuts,
        has_region=region is not None,
    )
    report.update(
        timestamp_source="decoded_presentation_time"
        if not fallback_pts
        else "mixed_PTS_and_inferred_CFR",
        inferred_frame_timestamps=fallback_pts,
        region=region,
    )
    return report, rows
