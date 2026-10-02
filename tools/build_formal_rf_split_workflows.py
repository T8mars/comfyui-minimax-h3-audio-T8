"""Create additive S29 RF base/restart stage examples from the current builder.

No existing workflow is edited. Saved examples are EXP and require separate
browser, real-weight and media qualification before production use.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import uuid

from tools import build_modular_rf_workflow as rf


DESTINATION = rf.ROOT / "examples/workflows/62-rf-restart-split"


@lru_cache(maxsize=1)
def generated():
    info = rf.manual.load_info()
    result = {}
    for entry in rf.ENTRIES:
        for variant in rf.native.VARIANTS:
            _api, graph, audit = rf.build_candidate(entry, variant, info)
            if audit["scope"] != "candidate_serialization_only_not_browser_roundtrip":
                raise ValueError("S29 frontend serialization audit contract changed")
            graph["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"t8:formal-rf:{entry}:{variant}"))
            result[DESTINATION / f"S29_RF_{entry}_{variant}_Separate_EXP.json"] = graph
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Create only missing examples")
    options = parser.parse_args()
    expected = generated()
    if len(expected) != 12:
        raise ValueError("S29 requires three RF entry routes × four variants")
    pending = {}
    for path, graph in expected.items():
        content = (json.dumps(graph, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf8")
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError(f"Refusing to overwrite modified public S29 graph: {path}")
        else:
            pending[path] = content
    if pending and not options.write:
        print(f"S29 verified {len(expected) - len(pending)} graphs; {len(pending)} missing")
        return 1
    if pending:
        DESTINATION.mkdir(parents=True, exist_ok=True)
        for path, content in pending.items():
            path.write_bytes(content)
    print(f"S29 verified {len(expected)} graphs; created {len(pending)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
