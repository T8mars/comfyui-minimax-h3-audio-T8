"""Verify a pinned FreeVideo installation and create its T8 config; no installs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "h3_t8"))
from freevideo_exp.runtime import (FREEVIDEO_REVISION, VDN_REVISION, MODEL_REVISION,
                                 canonical, digest, verify_files, write_json_new)


def source_inventory(checkout, revision, actual, prefixes):
    result = subprocess.check_output(["git", "ls-tree", "-r", "-z", revision], cwd=checkout)
    entries = []
    for item in result.split(b"\0"):
        if not item:
            continue
        head, name = item.split(b"\t", 1)
        mode, kind, blob = head.decode().split()
        name = name.decode("utf-8")
        if kind == "blob" and any(name.startswith(prefix) for prefix in prefixes):
            entries.append((blob, name))
    output = subprocess.check_output(["git", "cat-file", "--batch"], cwd=checkout,
                                     input="".join(blob + "\n" for blob, _ in entries).encode())
    rows, cursor = [], 0
    for blob, name in entries:
        end = output.index(b"\n", cursor)
        got, kind, size = output[cursor:end].decode().split()
        if got != blob or kind != "blob":
            raise ValueError("Pinned source blob lookup failed")
        cursor = end + 1
        expected = output[cursor:cursor + int(size)]
        cursor += int(size) + 1
        path = actual / name
        actual_bytes = path.read_bytes()
        # Git for Windows checkout CRLF is allowed only as a newline transform.
        if actual_bytes != expected and actual_bytes.replace(b"\r\n", b"\n") != expected.replace(b"\r\n", b"\n"):
            raise ValueError("Installed source differs from pinned official blob: " + str(path))
        rows.append(dict(path=str(path.resolve()), bytes=len(actual_bytes), sha256=hashlib.sha256(actual_bytes).hexdigest(), kind="source", git_blob=blob))
    if not rows:
        raise ValueError("No pinned source files verified")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "vdn-verification-root", "diffusers-verification-root", "vdn-root", "edge-root", "metadata-root", "python", "home", "output-config"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--audio-reference-cache", type=Path)
    args = parser.parse_args()
    roots = {k: v.resolve() for k, v in vars(args).items() if v is not None}
    rows = source_inventory(roots["source_root"], FREEVIDEO_REVISION, roots["source_root"], ["freevideo_engine/", "LICENSE", "NOTICE", "THIRD_PARTY"])
    rows += source_inventory(roots["vdn_verification_root"], VDN_REVISION, roots["vdn_root"], ["src/", "LICENSE", "NOTICE"])
    tree = subprocess.check_output(["git", "write-tree"], cwd=roots["diffusers_verification_root"], text=True).strip()
    if tree != "37068ab7331d8b28f4cf718dba7015c742a306d2":
        raise ValueError("Diffusers verification tree differs from pinned patched tree")
    rows += source_inventory(roots["diffusers_verification_root"], tree, roots["vdn_root"] / "diffusers", ["src/", "LICENSE", "NOTICE"])
    catalog = json.loads((roots["source_root"] / "freevideo_engine/prepared_models.json").read_text(encoding="utf-8"))
    variant = catalog["variants"]["rowwise"]
    if variant["revision"] != MODEL_REVISION:
        raise ValueError("Official rowwise revision changed")
    model_rows = []
    for row in variant["files"]:
        name = row["file"]
        candidates = [roots["metadata_root"] / name, roots["edge_root"] / name]
        path = next((p for p in candidates if p.is_file() and digest(p) == row["sha256"]), None)
        if path is None:
            raise ValueError("Missing or changed pinned FreeVideo asset: " + name)
        model_rows.append(dict(path=str(path), bytes=row["bytes"], sha256=row["sha256"],
                               kind="weight" if name.endswith(".safetensors") else "metadata", catalog_file=name))
        if len(model_rows) % 50 == 0 or len(model_rows) == len(variant["files"]):
            print(f"Verified model asset {len(model_rows)}/{len(variant['files'])}", flush=True)
    verify_files(model_rows)
    rows += model_rows
    reference_root = roots.get("audio_reference_cache")
    if reference_root is not None:
        # The owned derived cache is not trusted merely because it hashes its
        # own files: read every new constant row back against the official base.
        sys.path.insert(0, str(roots["source_root"]))
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["FREEVIDEO_VDN_ROOT"] = str(roots["vdn_root"])
        from freevideo_engine.paths import add_vdn
        add_vdn()
        from freevideo_exp.reference_tables import audit
        reference_audit = audit(reference_root, roots["edge_root"] / variant["cache_prefix"])
        if reference_audit["cuda_initialized"]:
            raise ValueError("Reference constant validation must be CPU-only")
        print(canonical(reference_audit), flush=True)
        derived = json.loads((reference_root / "t8-reference-derivation.json").read_text(encoding="utf-8"))
        if derived.get("schema") != "t8-freevideo-reference-tables-v1" or derived.get("model_revision") != MODEL_REVISION:
            raise ValueError("Invalid owned audio-reference cache derivation")
        for path in sorted(reference_root.rglob("*")):
            if path.is_file():
                rows.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path), kind="weight" if path.suffix == ".safetensors" else "metadata"))
    own = Path(__file__).resolve().parents[1] / "h3_t8/freevideo_exp"
    for path in sorted(own.glob("*.py")):
        rows.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path), kind="source"))
    rows.append(dict(path=str(roots["python"]), bytes=roots["python"].stat().st_size, sha256=digest(roots["python"]), kind="source"))
    roots["home"].mkdir(parents=True, exist_ok=True)
    config = dict(schema="t8-freevideo-runtime-v1", freevideo_revision=FREEVIDEO_REVISION,
                  vdn_revision=VDN_REVISION, model_revision=MODEL_REVISION,
                  python=str(roots["python"]), source_root=str(roots["source_root"]), vdn_root=str(roots["vdn_root"]),
                  cache=str(roots["edge_root"] / variant["cache_prefix"]), base=str(roots["metadata_root"] / "config/h3-base"),
                  checkpoint=str(roots["metadata_root"] / "config/stage-dmd-step-250"), home=str(roots["home"]),
                  files=rows, model_asset_count=len(model_rows),
                  audio_reference_cache=str(reference_root) if reference_root is not None else None,
                  inventory_sha256=hashlib.sha256(canonical(rows).encode()).hexdigest(),
                  boundary="Read-only borrowed weights/source, explicit cudnn, isolated Python. Online linear LoRA; separate EAV/Relay adapters; optional explicit derived audio-reference constant tables.")
    write_json_new(roots["output_config"], config)
    print(json.dumps(dict(config=str(roots["output_config"]), sha256=digest(roots["output_config"]), model_assets=len(model_rows), source_files=len(rows)-len(model_rows))))


if __name__ == "__main__":
    main()
