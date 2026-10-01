"""Serve eight SHA-frozen legacy workflows with bundled frontend subgraphs."""
from __future__ import annotations

import json

from tools import serve_modular_m0_legacy_browser as legacy

ROOT = legacy.ROOT
BASELINE = legacy.BASELINE
SCOPE = legacy.PRIVATE / "modular-sampling-m0-resume-20260927/legacy-core-embedded-20260928-v2.json"


def selected() -> list[dict]:
    baseline = json.loads(BASELINE.read_bytes())
    scope = json.loads(SCOPE.read_bytes())
    entries = [entry for entry in scope["files"]
               if entry.get("status") == "embedded_types_available_not_browser_validated"]
    if (scope["counts"]["frontend"] != 306 or len(entries) != 8
            or scope["counts"]["frontend_invalid_definitions"] != 0):
        raise ValueError("Frozen embedded frontend scope changed")
    result = []
    for index, entry in enumerate(entries, 1):
        relative = entry["path"]
        source = (ROOT / relative).resolve()
        expected = baseline["files"].get(relative)
        if (not source.is_relative_to(ROOT) or not source.is_file()
                or not isinstance(expected, str) or legacy._sha(source) != expected
                or entry.get("missing_registered_types")):
            raise ValueError(f"Frozen embedded source changed: {relative}")
        result.append({"index": index, "relative": relative, "source": str(source),
                       "sha256": expected, "copy_name": f"M0E_{index:03}_Legacy.json",
                       "saved_name": f"QA_M0E_{index:03}_Saved.json",
                       "nodes": entry["node_count"]})
    return result


def main() -> int:
    legacy.SCOPE = SCOPE
    legacy.selected = selected
    legacy.PROFILE_SCHEMA = "t8.modular-sampling.m0-embedded-browser-profile.v1"
    return legacy.main()


if __name__ == "__main__":
    raise SystemExit(main())
