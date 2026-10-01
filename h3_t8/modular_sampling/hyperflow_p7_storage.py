"""Explicit P7-only frozen stages, separate from historical HyperFlow caches.

These are selected artifacts, not automatic cache hits. The generic store
commits its manifest last and authenticates every tensor and sampler receipt;
the P7 binder additionally rechecks the current accepted parent/phase before
allowing a stored LOW or HIGH result into a new graph.
"""
from pathlib import Path
import re

import folder_paths

from .. import long_video
from . import hyperflow_p7 as p7
from .results import canonical
from .storage import fingerprint_stage, load_stage, save_stage

LOW_STAGE = "hyperflow_low_partial4"
HIGH_STAGE = "hyperflow_high_after_partial4"
SCHEMA = "t8.modular-sampling.hyperflow-p7-frozen-stage.v1"


def stage_root(phase, *, output_root=None):
    if type(phase) is not p7.P7Phase:
        raise ValueError("Select an authenticated P7 phase for the stage store")
    phase.verify()
    request = (phase.contexts.parent.binding["request"] if type(phase.contexts) is p7.P7Contexts
               else phase.contexts.request)
    chain_id, index = request["chain_id"], request["segment_index"]
    if (type(chain_id) is not str or long_video.sanitize_chain_id(chain_id) != chain_id
            or type(index) is not int or index < 0):
        raise ValueError("P7 stage store has an invalid chain/segment identity")
    base = Path(folder_paths.get_output_directory() if output_root is None else output_root)
    return base / "MiniMaxH3" / "modular_p7_stages" / chain_id / f"segment_{index:05d}"


def save_low(bound, *, output_root=None):
    if type(bound) is not p7.P7LowResult:
        raise ValueError("P7 LOW save needs its authenticated completed result")
    binding = bound.verify()
    receipt = bound.sampled.verify()
    if receipt["portable_identity"] is not True:
        raise ValueError("P7 LOW has an unverified executable stack; do not persist for HIGH-only reuse")
    root = stage_root(bound.phase, output_root=output_root)
    path, digest, manifest = save_stage(bound.sampled, root, "p7-low")
    report = {"schema": SCHEMA, "phase": "low", "binding_sha256": binding["sha256"],
              "stage_receipt_sha256": receipt["receipt_sha256"], "artifact_path": path,
              "artifact_sha256": digest, "manifest": manifest, "automatic_cache_reuse": False,
              "sampling_calls": 0}
    return bound, bound.sampled.denoised_output, path, digest, canonical(report)


def load_low(phase, path, digest, *, output_root=None):
    if type(phase) is not p7.P7Phase or phase.phase != "low":
        raise ValueError("P7 LOW load needs its freshly authenticated LOW phase")
    root = stage_root(phase, output_root=output_root)
    _, _, _, result, storage_report = load_stage(root, path, digest, LOW_STAGE)
    bound = p7.bind_low(phase, result)
    report = {"schema": SCHEMA, "phase": "low", "binding_sha256": bound.verify()["sha256"],
              "artifact_path": path, "artifact_sha256": digest,
              "storage": storage_report, "automatic_cache_reuse": False, "sampling_calls": 0}
    return bound, result.denoised_output, canonical(report)


def save_high(bound, *, output_root=None):
    if type(bound) is not p7.P7HighResult:
        raise ValueError("P7 HIGH save needs its authenticated completed result")
    binding = bound.verify()
    receipt = bound.sampled.verify()
    if receipt["portable_identity"] is not True:
        raise ValueError("P7 HIGH has an unverified executable stack; do not persist as completed")
    root = stage_root(bound.handoff.phase, output_root=output_root)
    path, digest, manifest = save_stage(bound.sampled, root, "p7-high")
    report = {"schema": SCHEMA, "phase": "high", "binding_sha256": binding["sha256"],
              "stage_receipt_sha256": receipt["receipt_sha256"], "artifact_path": path,
              "artifact_sha256": digest, "manifest": manifest, "automatic_cache_reuse": False,
              "sampling_calls": 0}
    return bound, bound.sampled.output, path, digest, canonical(report)


def load_high(handoff, path, digest, *, output_root=None):
    if type(handoff) is not p7.P7HighInput:
        raise ValueError("P7 HIGH load needs its freshly authenticated handoff")
    root = stage_root(handoff.phase, output_root=output_root)
    _, _, _, result, storage_report = load_stage(root, path, digest, HIGH_STAGE)
    bound = p7.bind_high(handoff, result)
    report = {"schema": SCHEMA, "phase": "high", "binding_sha256": bound.verify()["sha256"],
              "artifact_path": path, "artifact_sha256": digest,
              "storage": storage_report, "automatic_cache_reuse": False, "sampling_calls": 0}
    return bound, result.output, canonical(report)


def fingerprint(phase, path, *, output_root=None):
    return fingerprint_stage(stage_root(phase, output_root=output_root), path)


def fingerprint_selected(path, stage, *, output_root=None):
    """Fingerprint an explicit P7 receipt before Core resolves linked phase inputs.

    The path is the Save node's UUID-relative output, never an arbitrary lookup.
    Ambiguous matches fail closed; execution still rechecks the selected phase.
    """
    prefix = {LOW_STAGE: "p7-low", HIGH_STAGE: "p7-high"}.get(stage)
    if (prefix is None or type(path) is not str
            or re.fullmatch(rf"{prefix}-[0-9a-f]{{32}}/manifest\.json", path) is None):
        raise ValueError("Select one exact P7 completed-stage receipt")
    base = Path(folder_paths.get_output_directory() if output_root is None else output_root)
    base = base / "MiniMaxH3" / "modular_p7_stages"
    matches = tuple(base.glob(f"*/segment_*/{path}"))
    if len(matches) != 1:
        raise ValueError("P7 completed-stage receipt is missing or ambiguous")
    root = matches[0].parent.parent
    return str(root.resolve()), *fingerprint_stage(root, path)
