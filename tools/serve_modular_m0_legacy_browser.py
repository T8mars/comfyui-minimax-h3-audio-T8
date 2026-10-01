"""Serve the 51 SHA-frozen legacy graphs within the audited plugin scope.

Only copies are placed in a new private CPU profile. Never queue or edit the
original workflows. The 255 graphs outside this plugin scope are not claimed.
"""

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

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.parents[1]
PRIVATE = (ROOT / "artifacts/development").resolve()
BASELINE = PRIVATE / "modular-sampling-m0-20260922/baseline.json"
SCOPE = PRIVATE / "modular-sampling-m0-followup-20260924/legacy-core-scope-isolated-v4.json"
STATUS = "types_registered_not_browser_validated"
PROFILE_SCHEMA = "t8.modular-sampling.m0-legacy-browser-profile.v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected() -> list[dict]:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    scope = json.loads(SCOPE.read_text(encoding="utf-8"))
    if scope["counts"]["frontend_all_types_registered"] != 51:
        raise ValueError("The pinned 51-graph Core scope changed")
    items = [entry for entry in scope["files"] if entry.get("status") == STATUS]
    if len(items) != 51:
        raise ValueError("Expected exactly 51 SHA-frozen in-scope frontend graphs")
    result = []
    for index, entry in enumerate(items, 1):
        relative = entry["path"]
        source = (ROOT / relative).resolve()
        expected = baseline["files"].get(relative)
        if (not source.is_relative_to(ROOT) or not source.is_file()
                or _sha(source) != expected):
            raise ValueError(f"Frozen M0 source changed: {relative}")
        result.append({"index": index, "relative": relative, "source": str(source),
                       "sha256": expected, "copy_name": f"M0_{index:03}_Legacy.json",
                       "saved_name": f"QA_M0_{index:03}_Saved.json",
                       "nodes": entry["node_count"]})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--seconds", type=int, default=1800)
    options = parser.parse_args()
    profile = options.root.resolve()
    image = CORE / "input/10A.jpg"
    if (profile.exists() or not profile.is_relative_to(PRIVATE)
            or not 1024 <= options.port <= 65535 or options.port == 8189
            or not 1 <= options.seconds <= 3600 or not image.is_file()):
        parser.error("Use a new private CPU profile, isolated port and bounded lifetime")
    cases = selected()
    profile.mkdir(parents=True)
    input_dir = profile / "input"
    input_dir.mkdir()
    shutil.copyfile(image, input_dir / image.name)
    transport.CORE, transport.PROJECT = CORE, ROOT
    transport.write_json(profile / "paths.json", probe_resource_config(CORE, ROOT))
    original_command = transport.server_command

    def command(*values):
        result = original_command(*values)
        result[result.index("progressive_probe_extension")] = "ComfyUI-VideoHelperSuite"
        point = result.index("--extra-model-paths-config")
        result[point:point] = ["ComfyUI-KJNodes", "ComfyUI-ClipProj"]
        result[result.index("--input-directory") + 1] = str(input_dir)
        return result

    transport.server_command = command
    server = transport.OwnedServer(profile, options.port, True)
    try:
        server.start()
        workflows = profile / "user/default/workflows"
        workflows.mkdir(parents=True)
        for case in cases:
            target = workflows / case["copy_name"]
            shutil.copyfile(case["source"], target)
            if _sha(target) != case["sha256"]:
                raise ValueError("Private legacy copy differs from frozen source")
        transport.wait_ready(server, lambda: None)
        ready = {"schema": PROFILE_SCHEMA,
                 "url": server.url, "pid": server.process.pid, "cpu_only": True,
                 "baseline_sha256": _sha(BASELINE), "scope_sha256": _sha(SCOPE),
                 "cases": cases, "input_sha256": _sha(input_dir / image.name)}
        transport.write_json(profile / "ready.json", ready)
        print(json.dumps({"url": server.url, "pid": server.process.pid,
                          "cases": len(cases)}, ensure_ascii=False), flush=True)
        deadline = time.monotonic() + options.seconds
        while (time.monotonic() < deadline and server.process.poll() is None
               and not (profile / "STOP").exists()):
            time.sleep(0.5)
    finally:
        server.stop()
        transport.write_json(profile / "terminal.json", {"server_stop": server.stop_receipt,
            "qualification": "Isolated CPU browser service lifecycle only"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
