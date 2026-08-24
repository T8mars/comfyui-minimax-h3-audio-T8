from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

import comfy.nested_tensor


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "h3_audio_t8_restart_probe_pkg"


def _load_package() -> None:
    if PACKAGE_NAME in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)


_load_package()

from h3_audio_t8_restart_probe_pkg.native_latent_checkpoint_advanced import (  # noqa: E402
    load_native_h3_av_checkpoint,
    save_native_h3_av_checkpoint,
)


def _latent():
    video = torch.arange(1 * 24 * 7 * 2 * 3, dtype=torch.float32).reshape(1, 24, 7, 2, 3)
    audio = torch.linspace(-0.5, 0.5, 1 * 32 * 2 * 37, dtype=torch.float32).reshape(
        1, 32, 2, 37
    )
    return {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": comfy.nested_tensor.NestedTensor(
            (torch.full_like(video, 0.25), torch.full_like(audio, 0.75))
        ),
        "probe_metadata": {"phase": "completed_save_then_new_process_load", "index": 23},
    }


def _emit(value):
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] not in {"save", "load"}:
        raise SystemExit("usage: worker.py save|load STORAGE_ROOT HANDOFF_JSON")
    mode = sys.argv[1]
    storage_root = Path(sys.argv[2]).resolve()
    handoff = Path(sys.argv[3]).resolve()
    if mode == "save":
        saved = save_native_h3_av_checkpoint(
            _latent(),
            storage_root,
            filename_prefix="restart_probe/probe",
            checkpoint_id="restart_probe_23",
            confirm_save=True,
            verify_after_write=True,
            hash_chunk_megabytes=1,
        )
        record = {
            "checkpoint_path": saved[2],
            "file_sha256": saved[3],
            "manifest_json": saved[4],
            "save_pid": os.getpid(),
        }
        handoff.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        _emit({"status": saved[1], **record})
        return 0

    record = json.loads(handoff.read_text(encoding="utf-8"))
    loaded = load_native_h3_av_checkpoint(
        storage_root,
        record["checkpoint_path"],
        expected_manifest_json=record["manifest_json"],
        expected_file_sha256=record["file_sha256"],
        hash_chunk_megabytes=8,
    )
    expected = _latent()
    loaded_parts = tuple(loaded[0]["samples"].unbind())
    expected_parts = tuple(expected["samples"].unbind())
    loaded_masks = tuple(loaded[0]["noise_mask"].unbind())
    expected_masks = tuple(expected["noise_mask"].unbind())
    exact = all(torch.equal(left, right) for left, right in zip(loaded_parts, expected_parts))
    exact = exact and all(
        torch.equal(left, right) for left, right in zip(loaded_masks, expected_masks)
    )
    exact = exact and loaded[0]["probe_metadata"] == expected["probe_metadata"]
    if not exact:
        raise RuntimeError("cross-process native latent checkpoint content changed")
    _emit(
        {
            "status": loaded[1],
            "resume_verified": loaded[2],
            "checkpoint_id": loaded[3],
            "content_sha256": loaded[4],
            "file_sha256": loaded[5],
            "save_pid": record["save_pid"],
            "load_pid": os.getpid(),
            "exact_tensor_metadata_match": exact,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
