#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping


PACKAGE_NAME = "h3_audio_t8_pkg"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
if PACKAGE_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = package
    assert spec.loader is not None
    spec.loader.exec_module(package)

finalize_spectrum_dataset = importlib.import_module(
    f"{PACKAGE_NAME}.speed_advanced"
).finalize_spectrum_dataset
load_spectrum_dataset_file = importlib.import_module(
    f"{PACKAGE_NAME}.speed_spectrum_storage"
).load_spectrum_dataset_file


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def finalize_file(
    *,
    storage_root: Path,
    dataset_name: str,
    profile_name: str,
    minimum_r_squared: float,
    minimum_independent_clips: int,
) -> dict[str, Any]:
    dataset, dataset_path, storage_report_json = load_spectrum_dataset_file(
        root=storage_root,
        dataset_name=dataset_name,
    )
    profile, _ = finalize_spectrum_dataset(
        dataset,
        profile_name=profile_name,
        minimum_r_squared=minimum_r_squared,
        minimum_independent_clips=minimum_independent_clips,
    )
    return {
        "dataset_path": dataset_path,
        "storage": json.loads(storage_report_json),
        "profile": profile,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Finalize one persisted H3 SPEED spectrum dataset into a JSON profile report."
    )
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--profile-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-r-squared", type=float, default=0.80)
    parser.add_argument("--minimum-independent-clips", type=int, default=100)
    args = parser.parse_args()
    result = finalize_file(
        storage_root=args.storage_root,
        dataset_name=args.dataset_name,
        profile_name=args.profile_name,
        minimum_r_squared=args.minimum_r_squared,
        minimum_independent_clips=args.minimum_independent_clips,
    )
    _write_json_atomic(args.output, result)
    profile = result["profile"]
    print(
        json.dumps(
            {
                "status": profile["status"],
                "clips": profile["independent_clip_count"],
                "amplitude": profile["fit"]["amplitude"],
                "beta": profile["fit"]["beta"],
                "r_squared": profile["fit"]["r_squared"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
