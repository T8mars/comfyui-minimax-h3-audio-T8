"""Prepare an independent v4 config from verified existing assets; never download or execute a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from h3_t8.freevideo_h21.source import Source, REVISION, canonical, digest  # noqa: E402
from h3_t8.freevideo_h21.plans import RUNTIME_SCHEMA, plan  # noqa: E402
from h3_t8.freevideo_exp.runtime import verify_files, write_json_new  # noqa: E402


def prepare(base_config, source_root, home, sampling_root, output, *, tiers, tasks, upscaler=None, engine_options=None):
    from h3_t8.freevideo_h21.assets import select, local_projections, model_structure, model_metadata, uses_tables
    output, home = Path(output).resolve(), Path(home).resolve()
    if output.exists() or home.exists():
        raise FileExistsError("v4 config and owned home must be new; never overwrite an old producer")
    base = json.loads(Path(base_config).read_text(encoding="utf8"))
    source = Source(source_root)
    structure = model_structure(base["base"])
    options = dict(attention="dense", linear_compute="bf16-weight-only", stream_weights=True,
        offload_refiner=True, projection_chunk=1024, adaln_cache=True, adaln_disk_cache=True,
        inference_kernels=False, prefetch=True) if engine_options is None else dict(engine_options)
    vdn_root = Path(base["vdn_root"]).resolve(strict=True)
    # Actual selected files, not the old family's module/code inventory. Existing
    # weights/VDN API sources remain read-only and keep their original SHA.
    borrowed = [row for row in base["files"] if row.get("kind") == "weight"
                or Path(row["path"]).resolve().is_relative_to(vdn_root)]
    if not borrowed or not any(row.get("kind") == "weight" for row in borrowed):
        raise ValueError("The explicit base config contains no actual prepared model assets")
    if not any(Path(row["path"]).resolve().is_relative_to(vdn_root) for row in borrowed):
        raise ValueError("The explicit base config contains no actual selected VDN API content inventory")
    verify_files(borrowed)
    # This is the preparation process only. No imported parent CLI/source
    # defaults are modified and no old runtime config is rewritten.
    original_path = list(sys.path)
    rows = list(source.files) + list(borrowed) + model_metadata(base["base"], base["checkpoint"], base["cache"])
    selected_tables = {}
    preparation_modes = []
    try:
        sys.path[:0] = [str(vdn_root / "diffusers/src"), str(vdn_root)]
        manifest = json.loads((Path(base["cache"]) / "manifest.json").read_text(encoding="utf8"))
        for row in manifest["groups"]:
            from h3_t8.freevideo_h21.source import contained
            path = contained(base["cache"], row["file"])
            rows.append(dict(path=str(path), bytes=row["bytes"], sha256=row["sha256"], kind="weight"))
        catalog = json.loads((source.root / "freevideo_engine/prepared_models.json").read_text(encoding="utf8"))
        for tier in tiers:
            for task in tasks:
                selected = plan(source, tier, 1344, 768, 124, task=task)
                roles = ["LOW", "HIGH"] if selected["sampling"]["enabled"] else ["SINGLE"]
                for role in roles:
                    entry = None
                    reason = "explicit_AdALN_cache_disabled"
                    if uses_tables(source, manifest, options):
                        try:
                            entry = select(source, manifest, catalog, selected, role,
                                dict(base=base["cache"], sampling=str(Path(sampling_root).resolve(strict=True))),
                                channels=structure["hidden_size"])
                        except FileNotFoundError as error:
                            reason = str(error)
                    if entry is not None:
                        selected_tables[canonical(entry["table"]["identity"])] = entry
                        preparation_modes.append(dict(tier=tier, task=task, role=role, mode="exact_prepared_constants"))
                    else:
                        projections = local_projections(base["cache"], manifest, source.module("adaln_assets"))
                        for row in projections:
                            path = Path(base["cache"]) / row["file"]
                            rows.append(dict(path=str(path.resolve()), bytes=row["bytes"], sha256=row["sha256"], kind="weight"))
                        preparation_modes.append(dict(tier=tier, task=task, role=role, mode="actual_dynamic_projections",
                                                      optimization_enabled=False, reason=reason))
    finally:
        sys.path[:] = original_path
    for entry in selected_tables.values():
        rows.extend(entry["files"])
    if upscaler:
        path = Path(upscaler).resolve(strict=True)
        rows.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path), kind="weight"))
    # Bind every owned adapter source plus the exact generic borrowed I/O helper
    # actually used. Do not freeze or import the obsolete quality profile family.
    helpers = list((ROOT / "h3_t8/freevideo_h21").glob("*.py")) + [
        ROOT / "h3_t8/freevideo_exp/runtime.py", ROOT / "h3_t8/freevideo_exp/supervision.py",
        ROOT / "h3_t8/freevideo_exp/effects.py", ROOT / "h3_t8/freevideo_exp/worker_effects.py"]
    for path in helpers:
        rows.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path), kind="source"))
    deduplicated = {}
    for row in rows:
        key = str(Path(row["path"]).resolve())
        if key in deduplicated and (row["bytes"], row["sha256"]) != (deduplicated[key]["bytes"], deduplicated[key]["sha256"]):
            raise ValueError("Conflicting selected asset inventory: " + key)
        deduplicated[key] = dict(row, path=key)
    rows = list(deduplicated.values())
    verify_files(rows)
    config = {key: base[key] for key in ("python", "vdn_root", "vdn_revision", "cache", "base", "checkpoint")}
    config.update(schema=RUNTIME_SCHEMA, freevideo_revision=REVISION, source_root=str(source.root),
        source_proof_sha256=source.proof_sha256, home=str(home), sampling_root=str(Path(sampling_root).resolve()),
        files=rows, inventory_sha256=hashlib.sha256(canonical(rows).encode()).hexdigest(),
        prepared_tiers=list(tiers), prepared_tasks=list(tasks), engine_options=options,
        actual_model_structure=structure, preparation_modes=preparation_modes,
        upscaler=str(Path(upscaler).resolve()) if upscaler else None,
        qualification="prepared_existing_files_only_not_GPU_or_trained_completion", implicit_downloads=False)
    source.verify()
    if model_structure(base["base"]) != structure:
        raise ValueError("Actual selected model structure changed during v4 preparation")
    home.mkdir(parents=True, exist_ok=False)
    temporary = output.with_name(output.name + ".partial-" + uuid.uuid4().hex)
    published = False
    try:
        write_json_new(temporary, config)
        # Atomic create-only publication. A failed write retains its uniquely
        # named evidence, never a half-written apparently valid config.
        os.link(temporary, output)
        published = True
        temporary.unlink()
    except BaseException:
        if not published:
            # Only remove the empty directory this invocation just created.
            # If anything appeared inside it, preserve it and the real error.
            try:
                home.rmdir()
            except OSError:
                pass
        raise
    return dict(path=str(output), sha256=digest(output), tables=len(selected_tables), files=len(rows), GPU_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-config", "source-root", "home", "sampling-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tiers", nargs="+", default=["light", "medium", "high", "max"])
    parser.add_argument("--tasks", nargs="+", default=["t2va"])
    parser.add_argument("--upscaler", type=Path)
    parser.add_argument("--engine-options", type=Path, help="Explicit Engine kwargs JSON, never a model or schedule replacement")
    args = parser.parse_args()
    options = json.loads(args.engine_options.read_text(encoding="utf8")) if args.engine_options else None
    print(canonical(prepare(args.base_config, args.source_root, args.home, args.sampling_root, args.output,
        tiers=args.tiers, tasks=args.tasks, upscaler=args.upscaler, engine_options=options)))


if __name__ == "__main__":
    main()
