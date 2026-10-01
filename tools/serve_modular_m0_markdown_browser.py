"""Serve frozen old graphs whose only unavailable Core type is UI MarkdownNote.

This is an isolated CPU serialization test, not a claim that MarkdownNote is a
registered execution node or that the graphs can generate media.
"""

from __future__ import annotations

import json

from tools import serve_modular_m0_legacy_browser as legacy

ROOT = legacy.ROOT
BASELINE = legacy.BASELINE
SCOPE = legacy.PRIVATE / "modular-sampling-m0-resume-20260927/legacy-core-scope-restart-20260928-v3.json"
EXPECTED = 236


def selected() -> list[dict]:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    scope = json.loads(SCOPE.read_text(encoding="utf-8"))
    if scope["counts"]["frontend"] != 306:
        raise ValueError("Frozen frontend scope count changed")
    items = [entry for entry in scope["files"] if
             entry.get("status") == "registry_scope_incomplete"
             and entry.get("missing_registered_types") == []
             and entry.get("ui_only_types") == ["MarkdownNote"]]
    if len(items) != EXPECTED:
        raise ValueError("Markdown-only old-graph scope changed")
    result = []
    for index, entry in enumerate(items, 1):
        relative = entry["path"]
        source = (ROOT / relative).resolve()
        expected = baseline["files"].get(relative)
        if (not source.is_relative_to(ROOT) or not source.is_file()
                or not isinstance(expected, str) or legacy._sha(source) != expected):
            raise ValueError(f"Frozen M0 source changed: {relative}")
        result.append({"index": index, "relative": relative, "source": str(source),
                       "sha256": expected, "copy_name": f"M0M_{index:03}_Legacy.json",
                       "saved_name": f"QA_M0M_{index:03}_Saved.json",
                       "nodes": entry["node_count"]})
    return result


def main() -> int:
    legacy.SCOPE = SCOPE
    legacy.selected = selected
    legacy.PROFILE_SCHEMA = "t8.modular-sampling.m0-markdown-browser-profile.v1"
    return legacy.main()


if __name__ == "__main__":
    raise SystemExit(main())
