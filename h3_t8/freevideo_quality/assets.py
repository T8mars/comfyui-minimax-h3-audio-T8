"""Explicit preparation and read-only, process-local schedule-table binding."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

from .profiles import table_identity, plan

REVISIONS = {
    "community-sigma3-v1-20261005": "4bbe518a5b24aa596f1e262e5f1a08062790670b",
    "sampling-presets-v1-20261005": "0cd66b30a58021012def78b0340caf131e0df5c2",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def table_path(root, name):
    r"""Resolve containment with consistent Windows DOS/extended-DOS spelling.

    Python can return \\?\ for an existing leaf but DOS for its parent. That
    is not an escape; compare the two fully resolved paths in one namespace.
    Junction/symlink destinations still participate in the containment check.
    """
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
        or PurePosixPath(name).is_absolute() or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError("Invalid quality table relative path")

    def normalized(path):
        value = str(Path(path).resolve())
        if os.name == "nt":
            if value.startswith("\\\\?\\UNC\\"):
                value = "\\\\" + value[8:]
            elif value.startswith("\\\\?\\") and len(value) > 6 and value[5:7] == ":\\":
                value = value[4:]
        return Path(value)

    base = normalized(root)
    path = normalized(base.joinpath(*PurePosixPath(name).parts))
    if not path.is_relative_to(base):
        raise ValueError("Quality table escapes the resolved cache")
    return path


def banks(catalog):
    result = []
    for bank in catalog.get("optional_adaln_sets", []) + [catalog.get("optional_adaln", {})]:
        if not bank:
            continue
        prefix = bank.get("prefix")
        if (prefix not in REVISIONS or bank.get("revision") != REVISIONS[prefix]
            or bank.get("repo") != "OpenVDN/vdn-minimax-h3-edge"):
            raise ValueError("Unpinned quality table bank")
        for table in bank["tables"]:
            result.append(dict(table, download={k: bank[k] for k in ("repo", "revision", "prefix")}))
    return result


def select(manifest, catalog, quality, role, task):
    from freevideo_engine import adaln_assets as official
    expected = table_identity(official.weight_identity(manifest), quality, role, task)
    matches = [(table, "base") for table in manifest.get("adaln_tables", []) if table["identity"] == expected]
    matches += [(table, "sampling") for table in banks(catalog) if table["identity"] == expected]
    # The published eight-step visual/T2VA tables already reside in the base.
    if not matches:
        raise ValueError(f"Missing official table identity for {quality}/{role}/{task}; no projection fallback")
    table, location = matches[0]
    official.validate_catalog(dict(manifest, adaln_tables=[table]), 50)
    return table, location


def required(manifest, catalog, quality, role, task):
    selected = plan(quality, role)
    roles = ["LOW", "HIGH"] if selected["role"] == "HIGH" else [selected["role"]]
    return [select(manifest, catalog, quality, stage, task) for stage in roles]


def verify(table, root, finite=True):
    from freevideo_engine import adaln_assets as official
    from safetensors.torch import load_file
    import torch
    for row in table["files"]:
        path = table_path(root, row["file"])
        official.check_table(path, row, table["identity"])
        if finite:
            state = load_file(str(path), device="cpu")
            if any(not bool(torch.isfinite(t).all()) for t in state.values()):
                raise ValueError("Nonfinite quality table: " + str(path))


@contextmanager
def offline_binding(base, sampling, catalog, selected_tables):
    """Replace only this worker's table I/O, never author source or old manifest.

    LoRA-derived manifests retain the same modulation weight identity; lookup
    stays bound to it rather than the derived cache's source_id or directory.
    """
    from freevideo_engine import adaln, adaln_assets as official, runtime
    from safetensors.torch import load_file
    import torch
    from diffusers import MiniMaxH3Scheduler
    from .profiles import raw_grid
    evidence = []
    allowed = {canonical(table["identity"]): (table, root) for table, root in selected_tables}

    class ReadOnlyTables:
        def __init__(self, root, source_id, steps, task="t2va", *, manifest=None,
                     timesteps=None, channels=5376, device="cuda"):
            times = adaln.schedule_timesteps(steps, task=task, device="cpu") if timesteps is None else timesteps
            self.identity = official.identity(official.weight_identity(manifest),
                [t.detach().float().cpu().tolist() for t in times], channels)
            entry = allowed.get(canonical(self.identity))
            if entry is None:
                raise ValueError("Unprepared exact quality/task clock; run the explicit preparation tool")
            self._table, self.root = entry[0], Path(entry[1])
            self.asset = self._table if self.root.resolve() == Path(base).resolve() else None
            self.optional_asset = None if self.asset is not None else self._table
            # Match the author's observable TableCache report contract. These
            # tables were prepared explicitly; this worker never downloads.
            self.optional_loaded, self.optional_downloaded = set(), set()
            self.optional_download_bytes, self.shared = 0, None
            self.device, self.producer = device, self._table.get("producer")

        def load(self, index, steps):
            if steps != len(self.identity["timesteps"]) or type(index) is not int or not 0 <= index < 50:
                raise ValueError("Quality table call/index mismatch")
            row = next(r for r in self._table["files"] if r["index"] == index)
            path = table_path(self.root, row["file"])
            official.check_table(path, row, self.identity)
            state = load_file(str(path), device=self.device)
            if any(not bool(torch.isfinite(t).all()) for t in state.values()):
                raise ValueError("Nonfinite quality constants")
            if self.optional_asset is not None:
                self.optional_loaded.add(index)
            evidence.append(dict(identity=self.identity, index=index, bytes=row["bytes"], sha256=row["sha256"]))
            return [state[f"step_{i}"].chunk(6, dim=-1) for i in range(steps)]

        def save(self, *args, **kwargs):
            raise ValueError("Quality tables are immutable; implicit projection calculation is disabled")

    def forbidden(*args, **kwargs):
        raise ValueError("Implicit table/projection download disabled; explicitly prepare the exact profile/task")

    original_set = MiniMaxH3Scheduler.set_timesteps

    def exact_schedule(self, num_inference_steps=None, device=None, sigmas=None):
        if num_inference_steps == 20 and sigmas is None:
            raw = raw_grid(20)
            sigmas = self.shift * raw / (1 + (self.shift - 1) * raw)
        return original_set(self, num_inference_steps=num_inference_steps, device=device, sigmas=sigmas)

    original = adaln.TableCache, runtime.TableCache, official.restore_projections, official.download_table
    adaln.TableCache = runtime.TableCache = ReadOnlyTables
    official.restore_projections = official.download_table = forbidden
    MiniMaxH3Scheduler.set_timesteps = exact_schedule
    try:
        yield evidence
    finally:
        adaln.TableCache, runtime.TableCache, official.restore_projections, official.download_table = original
        MiniMaxH3Scheduler.set_timesteps = original_set


def evidence_sha(evidence):
    return hashlib.sha256(canonical(evidence).encode()).hexdigest()
