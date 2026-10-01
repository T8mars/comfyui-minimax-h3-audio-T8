"""Bounded isolated CPU ComfyUI for S22 three/four-window browser QA."""
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
BASE = PRIVATE / "modular-sampling-m4-chunked-v5-storage-20260924"
WORKFLOWS = {
    3: ("candidate-v4-three-windows",
        "01_freeze_window_1.json", "02_resume_window_2_DRAFT.json"),
    4: ("candidate-v6-four-windows",
        "01_freeze_window_2.json", "02_resume_window_3_DRAFT.json"),
}
PROFILE_NAMES = {
    3: ("S22_freeze_window_1.json", "S22_resume_window_2_DRAFT.json"),
    4: ("S22_freeze_window_2.json", "S22_resume_window_3_DRAFT.json"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--window-count", type=int, choices=(3, 4), required=True)
    parser.add_argument("--port", type=int, default=8243)
    parser.add_argument("--seconds", type=int, default=1200)
    args = parser.parse_args()
    root = args.root.resolve()
    candidate_name, *source_names = WORKFLOWS[args.window_count]
    candidates = BASE / candidate_name
    image = CORE / "input/10A.jpg"
    if (root.exists() or not root.is_relative_to(PRIVATE.resolve())
            or not 1 <= args.seconds <= 3600
            or not 1024 <= args.port <= 65535 or args.port == 8189):
        parser.error("Use a new private profile, isolated non-user port and bounded lifetime")
    if not image.is_file() or any(not (candidates / name).is_file()
                                  for name in source_names):
        parser.error("Pinned S22 source image or candidate is missing")

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
        for source_name, profile_name in zip(
            source_names, PROFILE_NAMES[args.window_count], strict=True
        ):
            source, target = candidates / source_name, workflows / profile_name
            shutil.copyfile(source, target)
            copies.append({"source": str(source), "source_sha256": _sha(source),
                           "profile_copy": str(target), "copy_sha256": _sha(target)})
        transport.wait_ready(server, lambda: None)
        ready = {"url": server.url, "pid": server.process.pid, "cpu_only": True,
                 "window_count": args.window_count, "copies": copies,
                 "input_sha256": _sha(input_dir / image.name),
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
