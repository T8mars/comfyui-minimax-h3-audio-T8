from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load():
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    path = TOOLS / "build_human_face_0p6mp_4b_vs_native_review.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _media(tool, audio_hash="AUDIO"):
    return {
        "strict_decode_passed": True,
        "probe": {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": tool.high.WIDTH,
                    "height": tool.high.HEIGHT,
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "sample_rate": "32000",
                    "channels": 2,
                },
            ]
        },
        "decoded_video": {
            "bytes": tool.high.WIDTH * tool.high.HEIGHT * tool.high.legacy.FRAME_COUNT * 3
        },
        "decoded_audio": {"bytes": 1, "sha256": audio_hash},
    }


def _reports(tool):
    contract = tool.high._contract()
    clip4 = {
        "status": "PASS",
        "passed": True,
        "contract": contract,
        "media": {
            "strict_decode_passed": True,
            "decoded_audio": {"sha256": "FOUR"},
        },
    }
    native = {
        "status": "FAIL_RUNTIME_OR_MEDIA_CONTRACT",
        "passed": False,
        "contract": contract,
        "phase": {"terminal": {"type": "execution_success"}},
        "runtime_error": None,
        "gpu_monitor": {"minimum_free_mib": 643},
        "checks": {
            "asset_hash_contract": True,
            "execution_success": True,
            "decoded_video_exact_frames": True,
            "decoded_audio_nonempty": True,
            "observed_minimum_free_vram_at_least_512_mib": True,
            "strict_decode": False,
        },
        "media": {"decoded_audio": {"sha256": "NATIVE"}},
    }
    return clip4, native


def test_validation_accepts_only_transport_failure_and_preserved_audio():
    tool = _load()
    clip4, native = _reports(tool)
    assert tool.validate_reports(
        clip4, native, _media(tool, "FOUR"), _media(tool, "NATIVE")
    ) == tool.high._contract()


def test_validation_rejects_native_execution_failure():
    tool = _load()
    clip4, native = _reports(tool)
    native["checks"]["execution_success"] = False
    with pytest.raises(ValueError, match="may fail only"):
        tool.validate_reports(
            clip4, native, _media(tool, "FOUR"), _media(tool, "NATIVE")
        )


def test_manifest_is_one_pair_and_keeps_method_mapping_private(tmp_path):
    tool = _load()
    manifest = tool.build_manifest(
        tool.high._contract(),
        normalized_4b=tmp_path / "four.mp4",
        normalized_native=tmp_path / "native.mp4",
        reference_image=tmp_path / "ref.jpg",
    )
    assert manifest["review_id"] == tool.REVIEW_ID
    assert len(manifest["pairs"]) == 1
    pair = manifest["pairs"][0]
    assert pair["control_method"].startswith("Native")
    assert pair["candidate_method"].startswith("ClipProj 4B")
