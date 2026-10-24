"""Schedule-content lookup with read-only tables; no placeholders or downloads."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

from .source import canonical, contained, digest


def model_structure(base):
    value = json.loads((Path(base) / "transformer/config.json").read_text(encoding="utf8"))
    for key in ("hidden_size", "num_layers", "in_channels", "text_dim"):
        if type(value.get(key)) is not int or value[key] < 1:
            raise ValueError("Actual selected Diffusers config needs positive integer " + key)
    return {key: value[key] for key in ("hidden_size", "num_layers", "in_channels", "text_dim")}


def model_metadata(base, checkpoint, cache):
    """Bind actual selected configuration paths, not only older inventory labels."""
    paths = [contained(cache, "manifest.json")]
    for folder in (base, checkpoint):
        root = Path(folder).resolve(strict=True)
        paths.extend(path.resolve(strict=True) for path in root.rglob("*.json") if path.is_file())
    return [dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path), kind="metadata")
            for path in sorted(set(paths))]


def local_projections(cache, manifest, official):
    """Read/hash actual existing projection bytes; never restore/download them."""
    rows = official.projection_groups(manifest)
    if not rows:
        raise FileNotFoundError("Selected model has no actual AdaLN projection groups for dynamic execution")
    missing = []
    for row in rows:
        try:
            path = contained(cache, row["file"])
        except FileNotFoundError:
            missing.append(row)
            continue
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Actual local AdaLN projection content changed: " + str(path))
    if missing:
        raise FileNotFoundError("Actual dynamic AdaLN computation needs " + str(sum(r["bytes"] for r in missing))
            + " bytes in " + str(len(missing)) + " missing projection files; no implicit download: "
            + ", ".join(r["file"] for r in missing))
    return rows


def dynamic_projection_view(cache, manifest, home, official):
    """A complete read-only hardlink view, not a weight conversion/model swap.

    The author identifies slim storage by `format` and forces cached tables for
    that storage even when projection files exist. Only after verifying every
    real projection can a new owned manifest truthfully describe full storage.
    Original model/cache/metadata bytes are never modified.
    """
    projections = local_projections(cache, manifest, official)
    groups = {row["group"]: row for row in manifest["groups"]}
    for row in projections:
        if row["group"] in groups and groups[row["group"]] != row:
            raise ValueError("Conflicting real projection groups in selected manifest")
        groups[row["group"]] = row
    manifest = dict(manifest, groups=list(groups.values()))
    manifest.pop("format", None)
    manifest.pop("adaln_tables", None)
    dependencies = manifest["groups"]
    for spec in manifest.get("online_lora", {}).get("blocks", {}).values():
        if not any(row["file"] == spec["file"] for row in dependencies):
            raise ValueError("Online LoRA factor is missing from its actual content-bound groups")
    key = hashlib.sha256(canonical(dict(cache=str(Path(cache).resolve()), manifest=manifest)).encode()).hexdigest()
    root = Path(home) / "projection-views" / key
    marker = root / "manifest.json"
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf8")) != manifest:
            raise ValueError("Owned full-projection view metadata changed")
    else:
        # An interrupted link is not a completed cache and must not poison a
        # later request. Resume only a directory explicitly claimed by this
        # exact producer/content; never erase or adopt an unknown directory.
        owner_path = root / ".h21-projection-owner.json"
        owner = dict(schema="t8.freevideo.h21.projection-view.v1", key=key,
            cache=str(Path(cache).resolve()), manifest_sha256=hashlib.sha256(canonical(manifest).encode()).hexdigest())
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            if not owner_path.is_file() or owner_path.is_symlink() or json.loads(owner_path.read_text(encoding="utf8")) != owner:
                raise ValueError("Incomplete projection directory has no matching owned content identity; preserved: " + str(root))
        else:
            with owner_path.open("x", encoding="utf8") as stream:
                stream.write(canonical(owner))
                stream.flush()
                os.fsync(stream.fileno())
        for row in dependencies:
            origin = contained(cache, row["file"])
            if origin.stat().st_size != row["bytes"] or digest(origin) != row["sha256"]:
                raise ValueError("Selected source group changed before full-projection view")
            target = root / row["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                target = contained(root, row["file"])
                if target.is_symlink() or target.stat().st_size != row["bytes"] or digest(target) != row["sha256"]:
                    raise ValueError("Incomplete owned projection group content changed: " + str(target))
                continue
            # No unbudgeted copy of an entire model on link failure. This is an
            # actual filesystem requirement, not a model/class admission rule.
            os.link(origin, target)
        with marker.open("x", encoding="utf8") as stream:
            stream.write(canonical(manifest))
            stream.flush()
            os.fsync(stream.fileno())
    for row in dependencies:
        path = contained(root, row["file"])
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Owned full-projection view content changed")
    return root, manifest


def uses_tables(source, manifest, options):
    """Mirror the real Engine's storage branch, not a model-name allowlist."""
    return manifest.get("format") in source.module("adaln_assets").SLIM_FORMATS or bool(options.get("adaln_cache"))


def prepare_loras(cache, adapters, options, output_root):
    """Use a real author path supported by the explicitly selected Engine API.

    The online branch requires bounded native kernels. Do not enable/change
    those kernels behind a user's back, or construct an unusable online cache
    merely to have Engine reject it. The author's fused path remains usable
    with the dense/non-kernel placement and keeps ordered factors/strengths.
    """
    online = bool(options.get("head_chunk")) and bool(options.get("inference_kernels"))
    if online:
        from freevideo_engine.lora_online_cache import prepare
    else:
        from freevideo_engine.lora_cache import prepare
    path, report = prepare(cache, adapters, output_root=output_root)
    return path, dict(report, selected_preparation="online_capable" if online else "author_fused",
        explicit_engine_options_preserved=True)


def modulation_strategy(source, cache, manifest, catalog, selected, options, structure, *, sampling_root, home):
    """Keep actual weights/ordered LoRA; optional constants may bypass, never lie."""
    official = source.module("adaln_assets")
    options = dict(options)
    selections = []
    reason = None
    if uses_tables(source, manifest, options):
        roles = ["LOW", "HIGH"] if selected["sampling"]["enabled"] else ["SINGLE"]
        try:
            selections = [select(source, manifest, catalog, selected, role,
                dict(base=str(cache), sampling=str(sampling_root)), channels=structure["hidden_size"]) for role in roles]
        except FileNotFoundError as error:
            # A missing optional precompute is not model incompatibility. A
            # damaged declared table is ValueError and is not silently bypassed.
            reason = str(error)
    if not uses_tables(source, manifest, options) or reason is not None:
        projections = local_projections(cache, manifest, official)
        if manifest.get("format") in official.SLIM_FORMATS:
            cache, manifest = dynamic_projection_view(cache, manifest, home, official)
        options.update(adaln_cache=False, adaln_disk_cache=False)
        return Path(cache), manifest, options, [], dict(mode="actual_dynamic_projections", portable_optimization=False,
            optimization_enabled=False, reason=reason or "explicit_AdALN_cache_disabled", projection_groups=projections,
            warning="User-selected weights/LoRA preserved; precomputed constants are not claimed or reused.")
    return Path(cache), manifest, options, selections, dict(mode="exact_prepared_constants", optimization_enabled=True,
        portable_optimization=False, weight_identity=official.weight_identity(manifest))


def catalog_tables(catalog):
    result = []
    for bank in catalog.get("optional_adaln_sets", []) + [catalog.get("optional_adaln", {})]:
        result.extend(bank.get("tables", []))
    return result


def select(source, manifest, catalog, selected, role, table_root, *, channels=None):
    from .plans import clock
    official = source.module("adaln_assets")
    times = clock(source, selected, role)["modulation_timesteps"]
    if channels is None:
        declared = manifest.get("adaln_tables") or []
        if not declared:
            raise ValueError("Exact table lookup requires actual model hidden_size, not a guessed default")
        channels = declared[0]["identity"]["channels"]
    wanted = official.identity(official.weight_identity(manifest), times, channels)
    candidates = [(row, source_root, installed) for source_root, rows, installed in
        ((Path(table_root["base"]), manifest.get("adaln_tables", []), True),
         (Path(table_root["sampling"]), catalog_tables(catalog), False)) for row in rows if row["identity"] == wanted]
    if not candidates:
        raise FileNotFoundError("Exact v4 modulation table is not declared; explicit preparation required: " + canonical(wanted))
    table, folder, installed = candidates[0]
    rows = []
    for entry in table["files"]:
        try:
            path = contained(folder, entry["file"])
        except FileNotFoundError as error:
            if not installed:
                raise FileNotFoundError("Published optional v4 table has not been installed: " + entry["file"]) from error
            raise ValueError("Installed exact v4 modulation table is missing: " + entry["file"]) from error
        official.check_table(path, entry, wanted)
        rows.append(dict(path=str(path), bytes=entry["bytes"], sha256=entry["sha256"], kind="weight"))
    return dict(table=table, folder=str(folder), files=rows)


@contextmanager
def offline_tables(source, selections):
    """Only the owned child installs this overlay, restored even on failure."""
    import torch
    from safetensors.torch import load_file
    from freevideo_engine import adaln, adaln_assets as official, runtime
    allowed = {canonical(row["table"]["identity"]): row for row in selections}
    evidence = []
    class Tables:
        def __init__(self, root, source_id, steps, task="t2va", *, manifest=None,
                     timesteps=None, channels=5376, device="cuda", fetch=True):
            times = adaln.schedule_timesteps(steps, task=task, device="cpu") if timesteps is None else timesteps
            self.identity = official.identity(official.weight_identity(manifest), [t.float().cpu().tolist() for t in times], channels)
            self.selection = allowed.get(canonical(self.identity))
            if self.selection is None:
                raise ValueError("Actual v4 Engine requested an unprepared weight/clock table; no zero placeholder")
            self.root = Path(self.selection["folder"])
            self.asset = self.selection["table"]
            self.optional_asset = None
            self.optional_loaded, self.optional_downloaded, self.placeholders = set(), set(), set()
            self.optional_download_bytes, self.shared = 0, None
            self.device, self.producer = device, self.asset.get("producer")

        def load(self, index, steps):
            if steps != len(self.identity["timesteps"]):
                raise ValueError("Actual v4 modulation schedule changed")
            row = next((r for r in self.asset["files"] if r["index"] == index), None)
            if row is None:
                raise ValueError("Actual v4 layer has no prepared modulation tensor")
            path = contained(self.root, row["file"])
            official.check_table(path, row, self.identity)
            state = load_file(str(path), device=self.device)
            if any(not torch.isfinite(tensor).all() for tensor in state.values()):
                raise ValueError("Non-finite v4 modulation constants")
            evidence.append(dict(identity=self.identity, index=index, sha256=row["sha256"], bytes=row["bytes"]))
            return [state[f"step_{i}"].chunk(6, dim=-1) for i in range(steps)]

        def save(self, *args, **kwargs):
            raise ValueError("v4 modulation assets are immutable; implicit recomputation is not prepared")

    def forbidden(*args, **kwargs):
        raise FileNotFoundError("No implicit model/table download in the H21 owned worker; prepare explicit assets first")
    def restore_existing(cache, manifest):
        return local_projections(cache, manifest, official)
    previous = adaln.TableCache, runtime.TableCache, official.download_table, official.restore_projections
    adaln.TableCache = runtime.TableCache = Tables
    official.download_table = forbidden
    official.restore_projections = restore_existing
    try:
        yield evidence
    finally:
        adaln.TableCache, runtime.TableCache, official.download_table, official.restore_projections = previous
