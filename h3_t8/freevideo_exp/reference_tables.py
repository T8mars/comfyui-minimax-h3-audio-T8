"""Read back every derived audio-reference row against official source bytes."""
import json
from pathlib import Path

from .runtime import digest, tensor_record


def audit(root, official):
    from freevideo_engine import adaln_assets as assets
    from freevideo_engine.adaln import schedule_timesteps
    from safetensors.torch import load_file
    import torch
    root, official = Path(root).resolve(strict=True), Path(official).resolve(strict=True)
    marker = json.loads((root / "t8-reference-derivation.json").read_text(encoding="utf8"))
    baseline = json.loads((official / "manifest.json").read_text(encoding="utf8"))
    current = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    if (marker.get("schema") != "t8-freevideo-reference-tables-v1" or marker.get("official_manifest_sha256") != digest(official / "manifest.json")
            or marker.get("manifest_sha256") != digest(root / "manifest.json")
            or {k:v for k,v in baseline.items() if k != "adaln_tables"} != {k:v for k,v in current.items() if k != "adaln_tables"}
            or current["adaln_tables"][:len(baseline["adaln_tables"])] != baseline["adaln_tables"]):
        raise ValueError("Derived FreeVideo reference cache differs from its official base")
    assets.validate_catalog(current, 50)
    original_rows = baseline["groups"] + [r for table in baseline["adaln_tables"] for r in table["files"]]
    linked = marker["linked"]
    if len(linked) != len(original_rows):
        raise ValueError("Derived cache has an incomplete immutable base")
    for row, proof in zip(original_rows, linked):
        if any(row[k] != proof[k] for k in ("file", "bytes", "sha256")) or Path(proof["borrowed_path"]).resolve() != assets.asset_path(official, row["file"]).resolve():
            raise ValueError("Derived base-link provenance changed")
        path = assets.asset_path(root, row["file"])
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Derived immutable base asset changed")
    weights = assets.weight_identity(baseline)
    def table(task):
        expected = assets.identity(weights, [t.tolist() for t in schedule_timesteps(8, device="cpu", task=task)], 5376)
        return next(t for t in current["adaln_tables"] if t["identity"] == expected)
    t2 = table("t2va")
    if len(current["adaln_tables"]) != len(baseline["adaln_tables"]) + 2 or len(marker["derivations"]) != 2:
        raise ValueError("Derived cache must add exactly the two audio-reference task tables")
    compared = 0
    for task, base_task, marker_row in zip(("ref2va_audio", "ref2va_av"), ("t2va", "ref2va"), marker["derivations"]):
        target, source = table(task), table(base_task)
        proof_path = root / target["directory"] / "row-provenance.json"
        if Path(marker_row["path"]).resolve() != proof_path or digest(proof_path) != marker_row["sha256"]:
            raise ValueError("Derived constant-row provenance changed or escaped its cache")
        proof = json.loads(proof_path.read_text(encoding="utf8"))
        if (proof.get("task") != task or proof.get("directory") != target["directory"] or proof.get("producer") != target["producer"]
                or target["producer"].get("profile") != "derived_constant_rows_exp_v1"):
            raise ValueError("Invalid derived constant-row producer")
        lookup = {(r["block"], r["target_step"], r["clock"]):r for r in proof["rows"]}
        expected_count = 50 * sum(len(t) for t in target["identity"]["timesteps"])
        if len(lookup) != len(proof["rows"]) or len(lookup) != expected_count:
            raise ValueError("Incomplete or duplicate derived constant-row proof")
        for i in range(50):
            row = target["files"][i]
            assets.check_table(assets.asset_path(root, row["file"]), row, target["identity"])
            new = load_file(str(root / row["file"]), device="cpu")
            base = load_file(str(official / source["files"][i]["file"]), device="cpu")
            zero = load_file(str(official / t2["files"][i]["file"]), device="cpu")
            for step, clocks in enumerate(target["identity"]["timesteps"]):
                for index, clock in enumerate(clocks):
                    original_clocks = source["identity"]["timesteps"][step]
                    if clock in original_clocks:
                        old, at_step, at_clock, origin = base, step, original_clocks.index(clock), source["files"][i]
                    elif clock == 0.:
                        old, at_step, at_clock, origin = zero, 0, t2["identity"]["timesteps"][0].index(0.), t2["files"][i]
                    else:
                        raise ValueError("Uncovered derived constant clock")
                    piece = new[f"step_{step}"][index*3:(index+1)*3]
                    expected = old[f"step_{at_step}"][at_clock*3:(at_clock+1)*3]
                    provenance = lookup[i, step, clock]
                    piece_sha, expected_sha = tensor_record(piece)["sha256"], tensor_record(expected)["sha256"]
                    if (piece.dtype != torch.bfloat16 or not bool(torch.isfinite(piece).all()) or piece_sha != expected_sha
                            or provenance["source_file"] != origin["file"] or provenance["source_sha256"] != origin["sha256"]
                            or provenance["source_step"] != at_step or provenance["source_clock_index"] != at_clock
                            or provenance["row_sha256"] != piece_sha):
                        raise ValueError("Derived constant row is not byte-exact to its specified official row")
                    compared += 1
    return dict(profile="derived_constant_rows_exp_v1", compared_three_modality_rows=compared,
                manifest_sha256=digest(root / "manifest.json"), cuda_initialized=torch.cuda.is_initialized())
