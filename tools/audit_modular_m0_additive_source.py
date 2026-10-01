"""Prove two M0 old-node source drifts contain only guarded new subclasses.

This is a textual preservation check, not the strict M0 gate or numerical parity.
The preimage must have the exact SHA recorded by the frozen M0 baseline.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from tools.audit_modular_sampling_compat import write_new


PROJECT = Path(__file__).resolve().parents[1]
SOURCES = {
    "h3_t8/nodes_motion_recovery_advanced.py": (
        "MiniMaxH3MotionRetimingPrepareT8Advanced",
        "MiniMaxH3MotionOverloadAnalyzePassThroughEXPT8",
        "MiniMaxH3MotionOverloadAnalyzeT8Advanced",
        False,
    ),
    "h3_t8/nodes_native_latent_checkpoint_advanced.py": (
        "MiniMaxH3NativeLatentCheckpointLoadT8Advanced",
        "MiniMaxH3NativeLatentCheckpointPassThroughSaveEXPT8",
        "MiniMaxH3NativeLatentCheckpointSaveT8Advanced",
        True,
    ),
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def additive_insert(old: bytes, current: bytes, *, next_class: str,
                    new_class: str, parent_class: str,
                    added_json_import: bool) -> bytes:
    """Return only the added subclass; reject any change to prior file bytes."""
    newline = b"\r\n" if b"\r\n" in old else b"\n"
    if added_json_import:
        added = b"import json" + newline
        if added in old or current.count(added) != 1:
            raise ValueError("Checkpoint json import is not a unique addition")
        current = current.replace(added, b"", 1)
    anchor = ("class " + next_class).encode("ascii")
    if old.count(anchor) != 1 or current.count(anchor) != 1:
        raise ValueError("Old class anchor changed")
    prefix, suffix = old.split(anchor, 1)
    tail = anchor + suffix
    if not current.startswith(prefix) or not current.endswith(tail):
        raise ValueError("Old source bytes changed outside the new subclass")
    inserted = current[len(prefix):len(current) - len(tail)]
    try:
        tree = ast.parse(inserted.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError("Added subclass is not valid UTF-8 Python") from error
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.ClassDef)
            or tree.body[0].name != new_class
            or len(tree.body[0].bases) != 1
            or not isinstance(tree.body[0].bases[0], ast.Name)
            or tree.body[0].bases[0].id != parent_class):
        raise ValueError("Insertion is not exactly the expected derived node class")
    return inserted


def audit(baseline: dict, live: dict, preimage_root: Path) -> dict:
    baseline_nodes = {node["id"]: node for node in baseline["nodes"]}
    live_nodes = {node["id"]: node for node in live["nodes"]}
    results = {}
    affected = []
    for relative, (anchor, new_class, parent, json_import) in SOURCES.items():
        old_file = preimage_root / relative
        current_file = PROJECT / relative
        old, current = old_file.read_bytes(), current_file.read_bytes()
        old_nodes = [node for node in baseline_nodes.values() if node["source"] == relative]
        if not old_nodes or any(node["source_sha256"] != _sha(old) for node in old_nodes):
            raise ValueError(f"Preimage is not the frozen M0 source: {relative}")
        if any(live_nodes[node["id"]]["source"] != relative
               or live_nodes[node["id"]]["source_sha256"] != _sha(current)
               for node in old_nodes):
            raise ValueError(f"Live capture differs from current source: {relative}")
        inserted = additive_insert(old, current, next_class=anchor,
                                   new_class=new_class, parent_class=parent,
                                   added_json_import=json_import)
        results[relative] = {"frozen_sha256": _sha(old),
                             "current_sha256": _sha(current),
                             "new_subclass": new_class,
                             "new_subclass_sha256": _sha(inserted),
                             "added_json_import": json_import,
                             "old_node_ids": [node["id"] for node in old_nodes]}
        affected.extend(node["id"] for node in old_nodes)
    return {"schema": "t8.modular-m0-additive-old-source.v1",
            "status": "pass_textual_additions_only_not_runtime_parity",
            "affected_old_nodes": sorted(affected), "files": results,
            "qualification": "Frozen old source bytes are exactly preserved after removing the "
                             "specified new subclasses/import. Strict live M0 still fails; "
                             "this is not indirect-helper, execution, numerical/GPU or "
                             "human quality qualification."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--preimage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(
            (PROJECT / "artifacts/development").resolve()):
        parser.error("Use a new private audit output")
    report = audit(json.loads(args.baseline.read_bytes()),
                   json.loads(args.live.read_bytes()), args.preimage_root.resolve())
    write_new(output, report)
    print(json.dumps({"status": report["status"],
                      "old_nodes": len(report["affected_old_nodes"]),
                      "files": list(report["files"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
