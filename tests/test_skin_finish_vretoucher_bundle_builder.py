from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = PROJECT_ROOT / "tools" / "build_skin_finish_vretoucher_single_window_manifest.py"
VALIDATOR_PATH = PROJECT_ROOT / "tools" / "validate_skin_finish_vretoucher_single_window.py"


def _module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _builder():
    return _module("h3_t8_vretoucher_bundle_builder_test", BUILDER_PATH)


def _validator():
    return _module("h3_t8_vretoucher_bundle_validator_test", VALIDATOR_PATH)


def _png(path: Path, array: np.ndarray, mode: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode=mode).save(path, format="PNG")
    return path


def _inputs(tmp_path: Path, *, frame_count: int = 3):
    height, width = 48, 80
    frames = []
    for index in range(frame_count):
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[..., 0] = 30 + index * 20
        rgba[..., 1] = 70
        rgba[..., 2] = 100
        rgba[..., 3] = 187
        frames.append(_png(tmp_path / f"source_{index}.png", rgba, "RGBA"))
    semantic = np.zeros((height, width), dtype=np.uint8)
    semantic[13:37, 27:49] = 255
    person = np.zeros((height, width), dtype=np.uint8)
    person[8:44, 20:60] = 255
    return {
        "frames": frames,
        "semantic": _png(tmp_path / "semantic.png", semantic, "L"),
        "person": _png(tmp_path / "person.png", person, "L"),
        "boxes": ["24,10,52,40"] * frame_count,
    }


def _args(tmp_path: Path, *, write: bool = False, frame_count: int = 3):
    builder = _builder()
    inputs = _inputs(tmp_path / "inputs", frame_count=frame_count)
    values: list[str] = []
    for frame in inputs["frames"]:
        values.extend(("--frame", str(frame)))
    values.extend(("--semantic-mask", str(inputs["semantic"])))
    values.extend(("--person-mask", str(inputs["person"])))
    for box in inputs["boxes"]:
        values.extend(("--face-box", box))
    values.extend(("--track-key", "shot0:person0"))
    values.extend(("--output-dir", str(tmp_path / "bundle")))
    if write:
        values.append("--write-bundle")
    return builder.parse_args(values), inputs


def test_default_builder_is_inspection_only_and_imports_no_runtime(tmp_path: Path):
    builder = _builder()
    args, _ = _args(tmp_path)
    report = builder.preflight(args)
    assert report["status"] == "READY_TO_WRITE_HASH_BOUND_BUNDLE"
    assert report["ready_to_write"] is True
    assert report["files_written"] is False
    assert report["torch_imported"] is False
    assert report["comfyui_imported"] is False
    assert report["model_loaded"] is False
    assert report["inference_executed"] is False
    assert not Path(report["output_directory"]).exists()


def test_written_bundle_is_byte_exact_and_passes_formal_validator(tmp_path: Path):
    builder = _builder()
    validator = _validator()
    assert builder.MANIFEST_SCHEMA == validator.MANIFEST_SCHEMA
    args, inputs = _args(tmp_path, write=True)
    preflight = builder.preflight(args)
    result = builder.write_bundle(args, preflight)
    bundle = Path(result["output_directory"])
    assert result["files_written"] is True
    assert (bundle / "build_report.json").is_file()
    for index, source in enumerate(inputs["frames"]):
        assert (bundle / "frames" / f"{index:03d}.png").read_bytes() == source.read_bytes()
    assert (bundle / "masks" / "semantic_skin.png").read_bytes() == inputs[
        "semantic"
    ].read_bytes()
    assert (bundle / "masks" / "person.png").read_bytes() == inputs[
        "person"
    ].read_bytes()
    normalized = validator.load_and_verify_manifest(bundle / "manifest.json")
    assert normalized["track_key"] == "shot0:person0"
    assert normalized["current_frame"] == 2
    assert normalized["output_current_frame_only"] is True


def test_builder_never_overwrites_an_existing_bundle(tmp_path: Path):
    builder = _builder()
    args, _ = _args(tmp_path)
    first = builder.preflight(args)
    builder.write_bundle(args, first)
    second = builder.preflight(args)
    assert second["status"] == "ABSTAIN_OUTPUT_DIRECTORY_EXISTS"
    with pytest.raises(builder.BundleUnavailable) as error:
        builder.write_bundle(args, first)
    assert error.value.status == "ABSTAIN_OUTPUT_DIRECTORY_EXISTS"


def test_builder_rejects_face_box_count_and_geometry_drift(tmp_path: Path):
    builder = _builder()
    args, _ = _args(tmp_path / "boxes")
    args.face_box = args.face_box[:-1]
    report = builder.preflight(args)
    assert report["status"] == "ABSTAIN_FACE_BOX_COUNT_MISMATCH"

    args, _ = _args(tmp_path / "geometry")
    changed = np.zeros((32, 32, 4), dtype=np.uint8)
    _png(args.frame[-1], changed, "RGBA")
    report = builder.preflight(args)
    assert report["status"] == "ABSTAIN_SOURCE_GEOMETRY_OR_MODE_MISMATCH"


def test_builder_rejects_non_l_mask_and_nonintersecting_box(tmp_path: Path):
    builder = _builder()
    args, _ = _args(tmp_path / "mask")
    rgb_mask = np.zeros((48, 80, 3), dtype=np.uint8)
    _png(args.semantic_mask, rgb_mask, "RGB")
    report = builder.preflight(args)
    assert report["status"] == "ABSTAIN_MASK_CONTRACT_MISMATCH"

    args, _ = _args(tmp_path / "box")
    args.face_box[0] = "200,200,220,220"
    report = builder.preflight(args)
    assert report["status"] == "ABSTAIN_FACE_BOX_OUTSIDE_FRAME"
