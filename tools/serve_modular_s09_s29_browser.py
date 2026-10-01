"""Bounded isolated CPU Core for current S09 and S29 split candidate UI QA."""

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
    ("S09", PRIVATE / "modular-sampling-m2-manual-pass-20260925/candidates-v3",
     "Manual_Pass_*_EXP.json", 8),
    ("S29", PRIVATE / "modular-sampling-m2-rf-restart-20260925/candidates-v3",
     "RF_*_EXP.json", 12),
)
SOURCE_NAMES: dict[str, frozenset[str]] | None = None
SOURCE_AUDIT_REQUIRED = True
INPUT_ALIASES: tuple[str, ...] = ()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--seconds", type=int, default=1200)
    options = parser.parse_args()
    root = options.root.resolve()
    image = CORE / "input/10A.jpg"
    if (root.exists() or not root.is_relative_to(PRIVATE.resolve())
            or not 1024 <= options.port <= 65535 or options.port == 8189
            or not 1 <= options.seconds <= 3600 or not image.is_file()):
        parser.error("Use a new private profile, isolated port and bounded lifetime")
    sources = []
    for route, directory, pattern, expected in SOURCES:
        selected = tuple(sorted(path for path in directory.glob(pattern)
                                if SOURCE_NAMES is None or path.name in SOURCE_NAMES[route]))
        if (len(selected) != expected or any(not path.is_file() for path in selected)
                or (SOURCE_NAMES is not None
                    and {path.name for path in selected} != SOURCE_NAMES[route])):
            parser.error(f"Missing current {route} candidates")
        if SOURCE_AUDIT_REQUIRED:
            audit = directory / "audit.json"
            if not audit.is_file():
                parser.error(f"Missing current {route} candidate audit")
            recorded = json.loads(audit.read_text(encoding="utf-8"))["candidates"]
            if (len(recorded) != expected or any(
                    result[0] is not True or not result[2] or result[3]
                    for entry in recorded for result in (entry["core_validation"],))):
                parser.error(f"{route} candidates have incomplete Core validation")
        sources.extend((route, path) for path in selected)

    root.mkdir(parents=True)
    input_dir = root / "input"
    input_dir.mkdir()
    shutil.copyfile(image, input_dir / image.name)
    for alias in INPUT_ALIASES:
        if Path(alias).name != alias or alias in ("", image.name):
            parser.error("Input alias must be a distinct plain filename")
        shutil.copyfile(image, input_dir / alias)
    transport.CORE, transport.PROJECT = CORE, PROJECT
    transport.write_json(root / "paths.json", probe_resource_config(CORE, PROJECT))
    original_command = transport.server_command

    def command(*values):
        result = original_command(*values)
        result[result.index("progressive_probe_extension")] = "ComfyUI-VideoHelperSuite"
        result[result.index("--input-directory") + 1] = str(input_dir)
        return result

    transport.server_command = command
    server = transport.OwnedServer(root, options.port, True)
    try:
        server.start()
        workflows = root / "user/default/workflows"
        workflows.mkdir(parents=True)
        copies = []
        for route, source in sources:
            target = workflows / f"{route}_{source.name}"
            shutil.copyfile(source, target)
            copies.append({"route": route, "source": str(source),
                           "source_sha256": _sha(source), "profile_copy": str(target),
                           "copy_sha256": _sha(target)})
        transport.wait_ready(server, lambda: None)
        ready = {"url": server.url, "pid": server.process.pid, "cpu_only": True,
                 "copies": copies, "input_sha256": _sha(input_dir / image.name),
                 "deadline_seconds": options.seconds}
        transport.write_json(root / "ready.json", ready)
        print(json.dumps(ready, ensure_ascii=False), flush=True)
        deadline = time.monotonic() + options.seconds
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
