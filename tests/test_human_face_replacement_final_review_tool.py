from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "build_human_face_replacement_final_review.py"


def _load_tool():
    tools_root = str(TOOL_PATH.parent)
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    spec = importlib.util.spec_from_file_location(
        "build_human_face_replacement_final_review", TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _contract():
    return {
        "prompt": "portrait <d>hello</d>",
        "seed": 2608245001,
        "input_image": "10A.jpg",
        "input_image_sha256": "34E67512265DA29076075030B62BA93EC304210A09171FF68E1F44894D15A36C",
        "width": 512,
        "height": 256,
        "frame_count": 124,
        "fps": 24,
        "steps": 8,
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "task_type": "I2VA",
    }


def _reports():
    creator = {
        "status": "PASS",
        "passed": True,
        "contract": {**_contract(), "output_frames": 243},
        "media": {
            "candidate_checks": {"strict_decode": True},
            "control_checks": {"strict_decode": True},
            "combined_frames": 243,
            "separate_frames": 243,
            "combined_audio_samples": 324_000,
            "separate_audio_samples": 324_000,
            "candidate_path": "candidate.mp4",
            "control_path": "control.mp4",
        },
    }
    clip4 = {
        "status": "PASS",
        "passed": True,
        "contract": _contract(),
        "media": {"path": "4b.mp4", "strict_decode_passed": True},
    }
    clip8 = {
        "status": "PASS",
        "passed": True,
        "contract": _contract(),
        "media": {"path": "8b.mp4", "strict_decode_passed": True},
    }
    return creator, clip4, clip8


def test_manifest_combines_two_independent_assessable_pairs(tmp_path):
    tool = _load_tool()
    creator, clip4, clip8 = _reports()
    manifest = tool.build_manifest(
        creator, clip4, clip8, reference_image=tmp_path / "10A.jpg"
    )

    assert manifest["review_id"] == tool.REVIEW_ID
    assert manifest["export_filename"] == "human_face_replacement_final_blind_review.json"
    assert [row["pair_id"] for row in manifest["pairs"]] == [
        "creator-human-face-long-final",
        "clipproj-human-face-4b-vs-8b-final",
    ]
    assert all(row["reference_metrics"] == ["first_frame", "identity"] for row in manifest["pairs"])


def test_report_validation_rejects_contract_drift():
    tool = _load_tool()
    creator, clip4, clip8 = _reports()
    clip8["contract"]["seed"] += 1

    with pytest.raises(ValueError, match="contracts differ"):
        tool.validate_reports(creator, clip4, clip8)


def test_report_validation_rejects_nonpass_media():
    tool = _load_tool()
    creator, clip4, clip8 = _reports()
    clip4["media"]["strict_decode_passed"] = False

    with pytest.raises(ValueError, match="strict media contract"):
        tool.validate_reports(creator, clip4, clip8)
