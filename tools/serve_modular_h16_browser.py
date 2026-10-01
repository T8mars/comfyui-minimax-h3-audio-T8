"""Bounded isolated CPU ComfyUI for four H16 storage browser candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_progressive_pilot as transport  # noqa: E402
from vdn_probe_environment import probe_resource_config  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]
CORE = PROJECT.parents[1]
PRIVATE = PROJECT / "artifacts/development"
SOURCES = (
    ("modular-sampling-m4-h16-storage-20260923/candidate-v2",
     "01_freeze_after_window_2.json", "H16_plain_freeze.json"),
    ("modular-sampling-m4-h16-storage-20260923/candidate-v2",
     "02_resume_windows_3_to_6_DRAFT.json", "H16_plain_resume_DRAFT.json"),
    ("modular-sampling-m4-h16-effect-storage-20260924/candidate-v4",
     "01_freeze_after_window_2_relay_eav.json", "H16_effect_freeze.json"),
    ("modular-sampling-m4-h16-effect-storage-20260924/candidate-v4",
     "02_resume_windows_3_to_6_relay_eav_DRAFT.json", "H16_effect_resume_DRAFT.json"),
)
GEOMETRY_SOURCES = (
    ("modular-sampling-m4-h16-storage-20260923/candidate-v3",
     "01_freeze_after_window_2.json", "H16_plain_freeze.json"),
    ("modular-sampling-m4-h16-storage-20260923/candidate-v3",
     "02_resume_windows_3_to_6_DRAFT.json", "H16_plain_resume_DRAFT.json"),
    ("modular-sampling-m4-h16-effect-storage-20260924/candidate-v5",
     "01_freeze_after_window_2_relay_eav.json", "H16_effect_freeze.json"),
    ("modular-sampling-m4-h16-effect-storage-20260924/candidate-v5",
     "02_resume_windows_3_to_6_relay_eav_DRAFT.json", "H16_effect_resume_DRAFT.json"),
)
SOURCE_SETS = {"historical": SOURCES, "geometry": GEOMETRY_SOURCES}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8245)
    parser.add_argument("--seconds", type=int, default=1200)
    parser.add_argument("--candidate-set", choices=tuple(SOURCE_SETS),
                        default="historical")
    args = parser.parse_args()
    root = args.root.resolve()
    image = CORE / "input/10A.jpg"
    sources = SOURCE_SETS[args.candidate_set]
    source_paths = [PRIVATE / directory / name for directory, name, _ in sources]
    if (root.exists() or not root.is_relative_to(PRIVATE.resolve())
            or not 1 <= args.seconds <= 3600
            or not 1024 <= args.port <= 65535 or args.port == 8189):
        parser.error("Use a new private profile, isolated non-user port and bounded lifetime")
    if not image.is_file() or any(not source.is_file() for source in source_paths):
        parser.error("Pinned H16 image or candidate is missing")

    root.mkdir(parents=True)
    input_dir = root / "input"
    input_dir.mkdir()
    shutil.copyfile(image, input_dir / image.name)
    transport.CORE, transport.PROJECT = CORE, PROJECT
    transport.write_json(root / "paths.json", probe_resource_config(CORE, PROJECT))
    original_command = transport.server_command

    def command(*values):
        result = original_command(*values)
        result[result.index("progressive_probe_extension")] = "ComfyUI-VideoHelperSuite"
        result[result.index("--input-directory") + 1] = str(input_dir)
        return result

    transport.server_command = command
    server = transport.OwnedServer(root, args.port, True)
    try:
        server.start()
        workflows = root / "user/default/workflows"
        workflows.mkdir(parents=True)
        copies = []
        for source, (_, _, profile_name) in zip(source_paths, sources, strict=True):
            target = workflows / profile_name
            shutil.copyfile(source, target)
            copies.append({"source": str(source), "source_sha256": _sha(source),
                           "profile_copy": str(target), "copy_sha256": _sha(target)})
        transport.wait_ready(server, lambda: None)
        ready = {"url": server.url, "pid": server.process.pid, "cpu_only": True,
                 "candidate_set": args.candidate_set,
                 "copies": copies, "input_sha256": _sha(input_dir / image.name),
                 "deadline_seconds": args.seconds}
        transport.write_json(root / "ready.json", ready)
        print(json.dumps(ready, ensure_ascii=False), flush=True)
        deadline = time.monotonic() + args.seconds
        while (time.monotonic() < deadline and server.process.poll() is None
               and not (root / "STOP").exists()):
            time.sleep(0.5)
    finally:
        server.stop()
        transport.write_json(root / "terminal.json", {
            "server_stop": server.stop_receipt,
            "qualification": "Isolated UI service lifecycle only",
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
