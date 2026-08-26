from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from h3_audio_t8_pkg.skin_finish_vretoucher_runtime import VRetoucherRuntimeSession


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "validate_skin_finish_vretoucher_single_window.py"


def _tool():
    name = "h3_t8_vretoucher_single_window_tool_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_png(path: Path, array: np.ndarray, mode: str) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode=mode).save(path, format="PNG")
    return {"path": path.name, "sha256": _sha(path)}


def _manifest(tmp_path: Path, *, frame_count: int = 3) -> Path:
    height, width = 48, 80
    frames = []
    for index in range(frame_count):
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[..., 0] = 30 + index * 20
        rgba[..., 1] = 70
        rgba[..., 2] = 100
        rgba[..., 3] = 187
        frames.append(_write_png(tmp_path / f"frame_{index}.png", rgba, "RGBA"))
    semantic = np.zeros((height, width), dtype=np.uint8)
    semantic[13:37, 27:49] = 255
    person = np.zeros((height, width), dtype=np.uint8)
    person[8:44, 20:60] = 255
    payload = {
        "schema": _tool().MANIFEST_SCHEMA,
        "frames": frames,
        "current_frame": frame_count - 1,
        "shot_start": 0,
        "shot_end": frame_count - 1,
        "track_key": "0:0",
        "frame_track_keys": ["0:0"] * frame_count,
        "face_boxes": [[24.0, 10.0, 52.0, 40.0]] * frame_count,
        "semantic_skin_mask": _write_png(tmp_path / "semantic.png", semantic, "L"),
        "person_mask": _write_png(tmp_path / "person.png", person, "L"),
        "output_current_frame_only": True,
        "context_factor": 1.45,
        "amount": 1.0,
        "feather_px": 4,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class _BrightNewestModel(torch.nn.Module):
    def __init__(self, *, fail: bool = False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(1))
        self.fail = fail
        self.calls = 0
        self.last_inputs = None

    def forward(self, inputs):
        self.calls += 1
        self.last_inputs = [item.detach().clone() for item in inputs]
        if self.fail:
            raise RuntimeError("controlled tool inference failure")
        result = torch.ones_like(inputs[-1])
        masks = [torch.ones((1, 1, 256, 256)) for _ in range(6)]
        flows = torch.zeros((5, 1, 64, 64, 2))
        return result, masks, flows


def test_manifest_is_png_hash_bound_bounded_and_current_frame_only(tmp_path: Path):
    tool = _tool()
    normalized = tool.load_and_verify_manifest(_manifest(tmp_path))
    assert normalized["schema"] == tool.MANIFEST_SCHEMA
    assert len(normalized["frames"]) == 3
    assert normalized["current_frame"] == 2
    assert normalized["output_current_frame_only"] is True
    assert normalized["geometry"] == {"width": 80, "height": 48}
    assert len(normalized["normalized_manifest_sha256"]) == 64


def test_manifest_rejects_path_escape_and_asset_hash_drift(tmp_path: Path):
    tool = _tool()
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["frames"][0]["path"] = "../outside.png"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(tool.ValidationUnavailable) as error:
        tool.load_and_verify_manifest(path)
    assert error.value.status == "ABSTAIN_MANIFEST_PATH_UNSAFE"

    path = _manifest(tmp_path / "hash")
    frame = tmp_path / "hash" / "frame_0.png"
    frame.write_bytes(frame.read_bytes() + b"drift")
    with pytest.raises(tool.ValidationUnavailable) as error:
        tool.load_and_verify_manifest(path)
    assert error.value.status == "ABSTAIN_INPUT_SHA256_MISMATCH"


def test_prepared_window_rechecks_asset_hash_after_preflight(tmp_path: Path):
    tool = _tool()
    normalized = tool.load_and_verify_manifest(_manifest(tmp_path))
    frame = Path(normalized["frames"][0]["path"])
    frame.write_bytes(frame.read_bytes() + b"changed-after-preflight")
    with pytest.raises(tool.ValidationUnavailable) as error:
        tool.prepare_one_window(normalized)
    assert error.value.status == "ABSTAIN_INPUT_CHANGED_AFTER_PREFLIGHT"


def test_missing_checkpoint_preflight_never_imports_or_constructs_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    tool = _tool()
    manifest = _manifest(tmp_path)
    args = tool.parse_args(
        [
            "--manifest",
            str(manifest),
            "--checkpoint",
            str(tmp_path / "missing.pth"),
            "--checkpoint-sha256",
            "0" * 64,
            "--output-root",
            str(tmp_path / "out"),
        ]
    )
    monkeypatch.setattr(tool, "_port_is_listening", lambda _port=8188: False)
    monkeypatch.setattr(
        tool,
        "_gpu_memory_mib",
        lambda: {"available": True, "free_mib": 16_000, "total_mib": 16_384},
    )
    monkeypatch.setattr(
        tool,
        "_verify_bundled_source",
        lambda: (_ for _ in ()).throw(AssertionError("source import must not run")),
    )
    report = tool.preflight(args)
    assert report["status"] == "ABSTAIN_CHECKPOINT_MISSING"
    assert report["real_model_loaded"] is False
    assert report["checkpoint_deserialized"] is False
    assert report["inference_executed"] is False
    assert report["source"]["status"] == "NOT_CHECKED_DUE_TO_EARLIER_PREFLIGHT_FAILURE"


def test_fake_session_exercises_one_causal_window_without_auto_accept(tmp_path: Path):
    tool = _tool()
    manifest = tool.load_and_verify_manifest(_manifest(tmp_path))
    model = _BrightNewestModel()
    session = VRetoucherRuntimeSession(
        model,
        {"status": "TEST_MODEL", "checkpoint_loaded": False},
    )
    source, candidate, effective_mask, report = tool.execute_one_window(manifest, session)
    assert model.calls == 1
    assert len(model.last_inputs) == 6
    assert all(tuple(item.shape) == (1, 3, 512, 512) for item in model.last_inputs)
    assert all(torch.equal(model.last_inputs[0], item) for item in model.last_inputs[1:4])
    assert not torch.equal(model.last_inputs[3], model.last_inputs[4])
    assert not torch.equal(model.last_inputs[4], model.last_inputs[5])
    assert source.shape == candidate.shape == (48, 80, 4)
    assert int(torch.count_nonzero(effective_mask)) > 0
    outside = effective_mask <= 0
    assert torch.equal(source[..., :3][outside], candidate[..., :3][outside])
    assert torch.equal(source[..., 3:], candidate[..., 3:])
    assert report["pipeline"]["automatic_accept"] is False
    assert report["pipeline"]["candidate_selected"] is False
    assert report["pipeline"]["current_frame_only"] is True
    assert report["release"]["owner_reference_cleared"] is True
    assert session.closed is True


def test_fake_inference_exception_still_closes_owner_session(tmp_path: Path):
    tool = _tool()
    manifest = tool.load_and_verify_manifest(_manifest(tmp_path))
    model = _BrightNewestModel(fail=True)
    session = VRetoucherRuntimeSession(model, {"status": "TEST_MODEL"})
    with pytest.raises(RuntimeError, match="controlled tool inference failure"):
        tool.execute_one_window(manifest, session)
    assert model.calls == 1
    assert session.closed is True
    assert session.close_report["owner_reference_cleared"] is True


def test_confirmation_and_provisional_vram_floor_cannot_be_implicit():
    tool = _tool()
    args = tool.parse_args(["--manifest", "missing.json"])
    assert args.confirm_run == ""
    assert args.minimum_free_vram_mib == tool.PROVISIONAL_MINIMUM_FREE_VRAM_MIB
    assert tool.CONFIRMATION_TOKEN not in {"", "true", "yes", "1"}
