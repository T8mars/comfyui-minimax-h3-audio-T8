"""Create additive S28 Prepared LTX generation/decode and decode-only graphs.

Only explicit prepared-bundle inputs are accepted; this builder does not
convert an arbitrary H3 latent or overwrite an existing public workflow.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys

from tools.build_modular_prepared_ltx_workflows import ROOT, build_workflow


DESTINATION = ROOT / "examples/workflows/61-prepared-ltx-split"
STEMS = ((False, "S28_Prepared_LTX_Generate_Decode_Separate_EXP"),
         (True, "S28_Prepared_LTX_DecodeOnly_Separate_EXP"))


def _node_info():
    package_name = "h3_audio_t8_pkg"
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name, ROOT / "__init__.py",
            submodule_search_locations=[str(ROOT)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
    from h3_audio_t8_pkg.nodes_prepared_generation import MiniMaxH3PreparedGenerationBundleEXPT8
    from h3_audio_t8_pkg.modular_sampling.prepared_ltx_nodes import NODES
    classes = (MiniMaxH3PreparedGenerationBundleEXPT8, *NODES)
    result = {cls.define_schema().node_id: cls.GET_NODE_INFO_V1() for cls in classes}
    for value in result.values():
        value["cnr_id"] = "minimax-h3-audio-T8"
    return result


def generated():
    info = _node_info()
    return {DESTINATION / (stem + ".json"): build_workflow(info, resume=resume)
            for resume, stem in STEMS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Create only missing examples")
    options = parser.parse_args()
    expected = generated()
    if len(expected) != 2:
        raise ValueError("S28 requires a full and decode-only graph")
    pending = {}
    for path, graph in expected.items():
        content = (json.dumps(graph, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf8")
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError(f"Refusing to overwrite modified public S28 graph: {path}")
        else:
            pending[path] = content
    if pending and not options.write:
        print(f"S28 verified {len(expected) - len(pending)} graphs; {len(pending)} missing")
        return 1
    if pending:
        DESTINATION.mkdir(parents=True, exist_ok=True)
        for path, content in pending.items():
            path.write_bytes(content)
    print(f"S28 verified {len(expected)} graphs; created {len(pending)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
