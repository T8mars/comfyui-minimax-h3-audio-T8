"""Reuse verified body weights, fetch selected tables, create independent v2 config."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from h3_t8.freevideo_quality.profiles import (PROFILES, binding, AUDIO_RUNTIME, LEGACY_RUNTIME)  # noqa: E402
from h3_t8.freevideo_quality.assets import required, verify, table_path  # noqa: E402
from h3_t8.freevideo_exp.runtime import canonical, digest, verify_files, write_json_new, MODEL_REVISION  # noqa: E402


def fetch(table, row, root):
    from freevideo_engine.adaln_assets import check_table
    path = table_path(root, row["file"])
    if path.exists():
        check_table(path, row, table["identity"])
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > row["bytes"]:
        raise ValueError("Oversized partial table; retain it for inspection: " + str(partial))
    remote = table["download"]
    url = f"https://huggingface.co/{remote['repo']}/resolve/{remote['revision']}/{remote['prefix']}/{row['file']}"
    if offset != row["bytes"]:
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        with urllib.request.urlopen(request, timeout=60) as response:
            status = response.status
            if offset and (status != 206 or response.headers.get("Content-Range", "").split("/")[0] != f"bytes {offset}-{row['bytes']-1}"):
                raise ValueError("Resume response range mismatch; partial retained")
            with partial.open("ab" if offset else "xb") as stream:
                size = offset
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    size += len(block)
                    if size > row["bytes"]:
                        raise ValueError("Oversized table response")
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
    check_table(partial, row, table["identity"])
    partial.rename(path)
    print(f"Prepared {row['file']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("legacy-config", "source-root", "home", "sampling-root", "output-config"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=list(PROFILES), default=list(PROFILES))
    parser.add_argument("--task", default="t2va")
    parser.add_argument("--resources-only", action="store_true", help="Prepare/verify resources without freezing a runtime source epoch")
    parser.add_argument("--audio-reference-v3", action="store_true", help="Independent author v0.3.5 audio t=1 runtime")
    parser.add_argument("--relocate-t8-from", type=Path,
                        help="Explicit former T8 root; unchanged borrowed helpers must still match their old SHA")
    args = parser.parse_args()
    producer = binding(AUDIO_RUNTIME if args.audio_reference_v3 else LEGACY_RUNTIME)
    if args.output_config.exists():
        raise FileExistsError("Create a new v2 config; do not overwrite existing configurations")
    old = json.loads(args.legacy_config.read_text(encoding="utf8"))
    proof = json.loads((args.source_root / "t8-source-proof.json").read_text(encoding="utf8"))
    if proof.get("commit") != producer["revision"]:
        raise ValueError("New source proof revision mismatch")
    sys.path.insert(0, str(args.source_root))
    from freevideo_engine import adaln_assets as official
    rows = []
    for row in proof["files"]:
        path = args.source_root / row["file"]
        actual = path.read_bytes()
        blob = hashlib.sha1(f"blob {len(actual)}\0".encode() + actual).hexdigest()
        if blob != row["git_blob"] or hashlib.sha256(actual).hexdigest() != row["sha256"]:
            raise ValueError("New source differs from verified official Git blob")
        rows.append(dict(path=str(path.resolve()), bytes=len(actual), sha256=row["sha256"], kind="source"))
    # Old inventory is reused only after actual content verification. Old source
    # files are also borrowed helpers, but the old author engine is not imported.
    reference_root = Path(old["audio_reference_cache"]) if old.get("audio_reference_cache") else None
    borrowed = [row for row in old["files"] if not Path(row["path"]).is_relative_to(Path(old["source_root"]))
                and not (reference_root is not None and Path(row["path"]).is_relative_to(reference_root))]
    if args.relocate_t8_from is not None:
        relocated = []
        for row in borrowed:
            path = Path(row["path"])
            if path.is_relative_to(args.relocate_t8_from):
                relative = path.relative_to(args.relocate_t8_from)
                if relative.is_relative_to(Path("h3_t8/freevideo_quality")):
                    continue  # The explicitly updated adapter gets a fresh inventory below.
                row = dict(row, path=str(ROOT / relative))
            relocated.append(row)
        borrowed = relocated
    print(f"Verifying {len(borrowed)} borrowed body/source assets once; old derived audio cache is not needed", flush=True)
    verify_files(borrowed)
    catalog = json.loads((args.source_root / "freevideo_engine/prepared_models.json").read_text(encoding="utf8"))
    variant = catalog["variants"]["rowwise"]
    model_rows = {r.get("catalog_file"): r for r in borrowed if r.get("catalog_file")}
    if variant["revision"] != MODEL_REVISION or len(model_rows) != 278:
        raise ValueError("Wrong body model catalog")
    for row in variant["files"]:
        actual = model_rows[row["file"]]
        if actual["bytes"] != row["bytes"] or actual["sha256"] != row["sha256"]:
            raise ValueError("Borrowed body differs from new official catalog")
    rows += borrowed
    manifest = json.loads((Path(old["cache"]) / "manifest.json").read_text(encoding="utf8"))
    official.validate_catalog(manifest, 50)
    selected = {}
    for quality in args.profiles:
        role = "HIGH" if quality == "light" else "SINGLE"
        for table, location in required(manifest, catalog, quality, role, args.task,
                                       reference_audio_t=producer["reference_audio_t"]):
            selected[table["directory"]] = (table, location)
    print(f"Preparing {len(selected)} exact tables for task {args.task}", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(fetch, table, row, args.sampling_root) for table, location in selected.values()
                if location == "sampling" for row in table["files"]]
        for job in jobs:
            job.result()
    for table, location in selected.values():
        root = Path(old["cache"]) if location == "base" else args.sampling_root
        verify(table, root)
        for row in table["files"]:
            path = table_path(root, row["file"])
            if not any(r["path"] == str(path) for r in rows):
                rows.append(dict(path=str(path), bytes=row["bytes"], sha256=row["sha256"], kind="weight"))
    for path in sorted((ROOT / "h3_t8/freevideo_quality").glob("*.py")):
        rows.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path), kind="source"))
    # A duplicate borrowed helper is forbidden, not silently ignored.
    rows = list({row["path"]: row for row in rows}.values())
    verify_files([row for row in rows if row["kind"] != "weight"])
    args.home.mkdir(parents=True, exist_ok=True)
    config = {key: old[key] for key in ("python", "vdn_revision", "model_revision", "vdn_root", "cache", "base", "checkpoint")}
    config.update(schema=producer["runtime"], freevideo_revision=producer["revision"],
        source_root=str(args.source_root.resolve()), home=str(args.home.resolve()), sampling_root=str(args.sampling_root.resolve()),
        files=rows, inventory_sha256=hashlib.sha256(canonical(rows).encode()).hexdigest(),
        prepared_profiles=args.profiles, prepared_task=args.task,
        boundary="Independent pinned worker, read-only exact table overlay, existing body. No implicit downloads.")
    if not args.resources_only:
        write_json_new(args.output_config, config)
    print(canonical(dict(config=str(args.output_config), sha256=None if args.resources_only else digest(args.output_config),
        tables=len(selected), weight_identity=official.weight_identity(manifest))))


if __name__ == "__main__":
    main()
