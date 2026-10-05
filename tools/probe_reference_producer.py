"""Explicit CPU native VAE residency probe; no encode, sampling or model download."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def difference(before, after, path=""):
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(before.keys() | after.keys()):
            name = path + "/" + key
            if key not in before:
                result.append({"path": name, "before": "absent", "after": after[key]})
            elif key not in after:
                result.append({"path": name, "before": before[key], "after": "absent"})
            else:
                result.extend(difference(before[key], after[key], name))
        return result
    return [] if before == after else [{"path": path, "before": before, "after": after}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--vae-name", required=True)
    parser.add_argument("--role", choices=("video_vae", "audio_vae"), required=True)
    parser.add_argument("--package-name", help="Optional exact installed saved asset, no encode")
    parser.add_argument("--package-sha", help="Its actual saved SHA256")
    parser.add_argument("--roles-json", help="Explicit presence declaration for the saved asset")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any((args.package_name, args.package_sha, args.roles_json)) and not all(
            (args.package_name, args.package_sha, args.roles_json)):
        raise ValueError("Saved-asset probe needs exact filename, SHA and role declaration together")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(root / "artifacts/development/radar-r6-20261005"):
        raise ValueError("Use a new owned diagnostic file")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    sys.path[:0] = [str(args.core.resolve()), str(root)]
    sys.argv = ["reference-producer-probe", "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.cli_args  # noqa: F401
    import comfy.model_management as mm
    import folder_paths
    import nodes
    import torch
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location("h3_audio_t8_pkg", root / "__init__.py",
        submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from h3_audio_t8_pkg.progressive_producers import _native_producer_description
    from h3_audio_t8_pkg.core import ensure_h3_audio_vae_non_aligned_crop_compat
    if args.vae_name not in folder_paths.get_filename_list("vae"):
        raise ValueError("Choose one exact installed VAE")
    vae = nodes.VAELoader().load_vae(args.vae_name)[0]
    if args.role == "audio_vae":
        ensure_h3_audio_vae_non_aligned_crop_compat(vae)
    before = _native_producer_description(vae, args.role)
    mm.load_models_gpu([vae.patcher], force_full_load=True)
    after = _native_producer_description(vae, args.role)
    changes = difference(before, after)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU probe initialized CUDA")
    receipt = {"vae_name": args.vae_name, "role": args.role,
               "patcher_type": type(vae.patcher).__name__, "changes": changes,
               "cuda_initialized": False, "encoded": False, "sampling_NFE": 0}
    if args.package_name:
        from h3_audio_t8_pkg.reference_package import load_package, resolve_installed_package
        from h3_audio_t8_pkg.reference_runtime import capture_set
        asset = load_package(resolve_installed_package(args.package_name),
                             expected_sha256=args.package_sha)
        refs, mapping = capture_set([asset], args.roles_json)
        actual = refs.conditioning_inputs(vae if args.role == "video_vae" else None,
                                         vae if args.role == "audio_vae" else None)
        receipt["saved_asset"] = {"filename": args.package_name, "file_sha256": asset.file_sha256,
            "manifest_sha256": asset.verify()["sha256"], "mapping": mapping,
            "actual_producers": actual["actual_producers"], "same_native_producer_verified": True,
            "fresh_Qwen_encoded": False, "full_cold_sampler_verified": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(output), "differences": len(changes),
                      "paths": sorted({row["path"].rsplit("/", 1)[-1] for row in changes})}))


if __name__ == "__main__":
    main()
