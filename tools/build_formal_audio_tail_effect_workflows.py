"""Add 26 opt-in S26 effect graphs; never replace a source or existing graph."""
import argparse
import json

from tools.build_modular_audio_tail_effect_workflows import ROOT, generated_matrix
from tools.workflow_paths import public_workflow_path

DESTINATION = ROOT / "examples/workflows/59-audio-refine-split/effects"


def generated():
    return {public_workflow_path(DESTINATION / (path.stem + "_TailEffects.json")): graph
            for path, (_, graph) in generated_matrix().items()}


def preflight():
    expected = generated()
    if len(expected) != 26:
        raise ValueError("S26 requires ten EAV, eight fresh Relay and eight combined graphs")
    pending = {}
    # Check every existing destination before writing even the first missing file.
    for path, graph in expected.items():
        content = (json.dumps(graph, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf8")
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError("Refusing to overwrite modified S26 effects graph: " + str(path))
        else:
            pending[path] = content
    return expected, pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Create only missing EXP examples")
    args = parser.parse_args()
    expected, pending = preflight()
    if pending and not args.write:
        print(f"S26 effects: {len(pending)} missing; use --write to create additive EXP examples")
        return 1
    for path, content in pending.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    print(f"S26 effects: {len(expected)} verified; {len(pending)} created; old pairs untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
