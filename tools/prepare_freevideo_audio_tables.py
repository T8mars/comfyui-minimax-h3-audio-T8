"""Create an owned audio-reference cache by exact official constant-row reuse.

No model download, projection reconstruction, CUDA, or borrowed cache mutation.
Link immutable base assets read-only; create new manifests/tables only. Constant
rows are byte-exact to their specified source, NOT a claim of bitwise equality
to reevaluating original projections with a different GEMM batch dimension.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "h3_t8"))
from freevideo_exp.runtime import (FREEVIDEO_REVISION, MODEL_REVISION, VDN_REVISION,
                                 canonical, digest, tensor_record, verify_files, write_json_new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--output-cache", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    if tuple(config.get(k) for k in ("freevideo_revision", "vdn_revision", "model_revision")) != (FREEVIDEO_REVISION, VDN_REVISION, MODEL_REVISION):
        raise ValueError("Use the pinned T8 FreeVideo runtime")
    upstream = (Path(config["source_root"]), Path(config["vdn_root"]))
    verify_files([r for r in config["files"] if r["kind"] in ("weight", "metadata") or any(Path(r["path"]).is_relative_to(p) for p in upstream)])
    sys.path.insert(0, config["source_root"])
    os.environ["FREEVIDEO_VDN_ROOT"] = config["vdn_root"]
    from freevideo_engine.paths import add_vdn
    add_vdn()
    from freevideo_engine import adaln_assets as assets
    from freevideo_engine.adaln import schedule_timesteps
    from safetensors.torch import load_file, save_file
    import torch
    if torch.cuda.is_initialized():
        raise ValueError("Audio table preparation must remain CPU-only")
    cache = Path(config["cache"])
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    assets.validate_catalog(manifest, 50)
    weights = assets.weight_identity(manifest)
    def find(task):
        times = [t.tolist() for t in schedule_timesteps(8, device="cpu", task=task)]
        expected = assets.identity(weights, times, 5376)
        table = next(t for t in manifest["adaln_tables"] if t["identity"] == expected)
        for row in table["files"]:
            assets.check_table(assets.asset_path(cache, row["file"]), row, expected)
        return table
    t2, visual = find("t2va"), find("ref2va")
    root = args.output_cache.resolve()
    root.mkdir(parents=True, exist_ok=False)
    linked = []
    for row in manifest["groups"] + [r for t in manifest["adaln_tables"] for r in t["files"]]:
        source, target = assets.asset_path(cache, row["file"]), assets.asset_path(root, row["file"])
        if source.stat().st_size != row["bytes"] or digest(source) != row["sha256"]:
            raise ValueError("Official base asset changed before linking")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
        linked.append(dict(file=row["file"], bytes=row["bytes"], sha256=row["sha256"], borrowed_path=str(source)))
    new_manifest, derivations = dict(manifest), []
    new_manifest["adaln_tables"] = list(manifest["adaln_tables"])
    for task, base in (("ref2va_audio", t2), ("ref2va_av", visual)):
        times = [t.tolist() for t in schedule_timesteps(8, device="cpu", task=task)]
        expected = assets.identity(weights, times, 5376)
        directory = assets.directory(expected)
        folder = root / directory
        folder.mkdir(exist_ok=False)
        rows, provenance = [], []
        for i in range(50):
            source = base["files"][i]
            zero_source = t2["files"][i]
            native = load_file(str(cache / source["file"]), device="cpu")
            canonical_zero = load_file(str(cache / zero_source["file"]), device="cpu")["step_0"]
            zero_index = t2["identity"]["timesteps"][0].index(0.)
            output = {}
            for step, clocks in enumerate(times):
                pieces = []
                for clock in clocks:
                    old_clocks = base["identity"]["timesteps"][step]
                    if clock in old_clocks:
                        offset, chosen, name, source_step = old_clocks.index(clock), native[f"step_{step}"], source, step
                    elif clock == 0.:
                        offset, chosen, name, source_step = zero_index, canonical_zero, zero_source, 0
                    else:
                        raise ValueError("An audio-reference clock is not covered by official constant rows")
                    piece = chosen[offset * 3:(offset + 1) * 3].clone()
                    if tuple(piece.shape) != (3, 5376 * 6) or piece.dtype != torch.bfloat16 or not bool(torch.isfinite(piece).all()):
                        raise ValueError("Invalid official constant row")
                    pieces.append(piece)
                    provenance.append(dict(block=i, target_step=step, clock=clock, source_file=name["file"],
                        source_sha256=name["sha256"], source_step=source_step, source_clock_index=offset,
                        row_sha256=tensor_record(piece)["sha256"]))
                output[f"step_{step}"] = torch.cat(pieces, dim=0).contiguous()
            target = folder / f"{i:02d}.safetensors"
            save_file(output, str(target))
            with target.open("r+b") as stream:
                os.fsync(stream.fileno())
            row = dict(index=i, file=f"{directory}/{i:02d}.safetensors", bytes=target.stat().st_size, sha256=digest(target))
            assets.check_table(target, row, expected)
            rows.append(row)
            print(f"Derived {task} block {i + 1}/50", flush=True)
        producer = dict(profile="derived_constant_rows_exp_v1", official_model_revision=MODEL_REVISION,
            base_family=base["directory"], canonical_zero_family=t2["directory"], quality_approved=False,
            boundary="Exact reuse of specified official BF16 rows; not a bitwise reevaluation of new-batch GEMMs.")
        derivation = dict(task=task, directory=directory, producer=producer, rows=provenance)
        write_json_new(folder / "row-provenance.json", derivation)
        new_manifest["adaln_tables"].append(dict(identity=expected, directory=directory, files=rows, producer=producer))
        derivations.append(dict(path=str(folder / "row-provenance.json"), sha256=digest(folder / "row-provenance.json")))
    assets.validate_catalog(new_manifest, 50)
    write_json_new(root / "manifest.json", new_manifest)
    write_json_new(root / "t8-reference-derivation.json", dict(schema="t8-freevideo-reference-tables-v1",
        official_manifest_sha256=digest(cache / "manifest.json"), model_revision=MODEL_REVISION,
        linked=linked, derivations=derivations, manifest_sha256=digest(root / "manifest.json"), cuda_initialized=torch.cuda.is_initialized()))
    if torch.cuda.is_initialized():
        raise ValueError("Unexpected CUDA initialization during constant-row preparation")
    print(canonical(dict(cache=str(root), manifest_sha256=digest(root / "manifest.json"), cuda_initialized=False)))


if __name__ == "__main__":
    main()
