from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "analyze_human_face_replacement_reviews.py"


def _load_tool():
    tools_root = str(TOOL_PATH.parent)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    spec = importlib.util.spec_from_file_location(
        "analyze_human_face_replacement_reviews", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_creator_join_metrics_use_exact_video_and_audio_boundary():
    tool = _load_tool()
    frames = np.zeros((243, 8, 8, 3), dtype=np.uint8)
    frames[124:] = 10
    audio = np.zeros((324_000, 2), dtype=np.float32)
    audio[165_600:] = 0.25

    metrics = tool._creator_join_metrics(
        frames, audio, join_frame=124, join_sample=165_600
    )

    assert metrics["join_frame"] == 124
    assert metrics["join_sample"] == 165_600
    assert metrics["video_boundary_absdiff_mean"] == 10.0
    assert metrics["audio_single_sample_jump_max_abs"] == 0.25


def test_public_arm_removes_large_decoded_arrays():
    tool = _load_tool()
    value = {
        "frames": np.zeros((1, 1, 1, 3), dtype=np.uint8),
        "audio": np.zeros((1, 2), dtype=np.float32),
        "sha256": "A" * 64,
        "checks": {"strict_decode": True},
    }

    public = tool._public_arm(value)

    assert set(public) == {"sha256", "checks"}


def test_frame_health_counts_black_white_and_frozen_frames():
    tool = _load_tool()
    frames = np.stack(
        [
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.full((4, 4, 3), 255, dtype=np.uint8),
        ]
    )

    health = tool._frame_health(frames)

    assert health["near_black_frame_count_luma_below_5"] == 2
    assert health["near_white_frame_count_luma_above_250"] == 1
    assert health["near_frozen_transition_count_mad_below_0p01"] == 1
