"""Create additive, source-pinned S27 RGB/LTX stage-bound example graphs.

The original Sol Engine examples are read-only. This does not qualify a
portable cold checkpoint or real-weight quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json

from tools.build_modular_ltx_rgb_workflows import ROOT, sources, split_frontend
from tools.workflow_paths import public_workflow_path


DESTINATION = ROOT / "examples/workflows/60-ltx-rgb-stage-split"
SOURCE_SHA256 = {
    "2026-08-29_H3_Sol_Engine_Super_Acceleration_LTX25_Advanced_EXP.json":
        "01F2109D28FDD5DB6C7725DEAAC831D6F672B4A37AFFE864113D901A5091788E",
    "2026-08-30_H3_Sol_Engine_LTX25_Identity_Preserve_3Step_Advanced_EXP.json":
        "DEC745418DDEC78A079516B343094310FD8DF4594C4EF96A9E814620528F35B1",
}


def generated():
    paths = sources()
    if {path.name for path in paths} != set(SOURCE_SHA256):
        raise ValueError("S27 original LTX source inventory changed")
    result = {}
    for path in paths:
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest().upper() != SOURCE_SHA256[path.name]:
            raise ValueError(f"S27 original LTX source changed: {path}")
        result[public_workflow_path(DESTINATION / f"S27_{path.stem}_StageBound_Separate_EXP.json")] = (
            split_frontend(json.loads(source)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Create only missing examples")
    options = parser.parse_args()
    expected = generated()
    if len(expected) != 2:
        raise ValueError("S27 requires two RGB/LTX source variants")
    pending = {}
    for path, graph in expected.items():
        content = (json.dumps(graph, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf8")
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError(f"Refusing to overwrite modified public S27 graph: {path}")
        else:
            pending[path] = content
    if pending and not options.write:
        print(f"S27 verified {len(expected) - len(pending)} graphs; {len(pending)} missing")
        return 1
    if pending:
        DESTINATION.mkdir(parents=True, exist_ok=True)
        for path, content in pending.items():
            path.write_bytes(content)
    print(f"S27 verified {len(expected)} graphs; created {len(pending)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
