"""Serve nine frozen workflows using their already-installed real providers."""
from __future__ import annotations

import json

from tools import serve_modular_m0_legacy_browser as legacy

ROOT = legacy.ROOT
BASELINE = legacy.BASELINE
PRIOR = legacy.PRIVATE / "modular-sampling-m0-resume-20260927/legacy-core-embedded-20260928-v2.json"
SCOPE = legacy.PRIVATE / "modular-sampling-m0-resume-20260927/legacy-core-external-20260928-v2.json"
PROVIDERS = ("LanPaint", "ComfyUI-sol-attn", "comfyui-minimax-h3-blockcache-T8", "ComfyUI-MiniMaxH3")


def selected() -> list[dict]:
    baseline = json.loads(BASELINE.read_bytes())
    prior, scope = json.loads(PRIOR.read_bytes()), json.loads(SCOPE.read_bytes())
    missing = {item["path"] for item in prior["files"] if item.get("missing_registered_types")}
    resolved = [item for item in scope["files"] if item["path"] in missing
                and item.get("missing_registered_types") == []]
    loads = {item["directory"]: item for item in scope["extra_custom_node_loads"]}
    if (len(missing) != 11 or len(resolved) != 9 or scope["counts"]["frozen_error"] != 0
            or scope["cuda_initialized"] is not False
            or any(not loads.get(name, {}).get("loaded") for name in PROVIDERS)):
        raise ValueError("External-provider scope changed")
    result = []
    for index, item in enumerate(resolved, 1):
        relative = item["path"]
        source = (ROOT / relative).resolve()
        expected = baseline["files"].get(relative)
        if (not source.is_relative_to(ROOT) or not source.is_file()
                or legacy._sha(source) != expected or item.get("ui_only_types") != ["MarkdownNote"]):
            raise ValueError(f"Frozen external source changed: {relative}")
        result.append({"index": index, "relative": relative, "source": str(source),
                       "sha256": expected, "copy_name": f"M0X_{index:03}_Legacy.json",
                       "saved_name": f"QA_M0X_{index:03}_Saved.json", "nodes": item["node_count"]})
    return result


def main() -> int:
    legacy.SCOPE, legacy.selected = SCOPE, selected
    legacy.PROFILE_SCHEMA = "t8.modular-sampling.m0-external-browser-profile.v1"
    original = legacy.transport.server_command

    def command(*args):
        result = original(*args)
        point = result.index("--extra-model-paths-config")
        result[point:point] = list(PROVIDERS)
        return result

    legacy.transport.server_command = command
    return legacy.main()


if __name__ == "__main__":
    raise SystemExit(main())
