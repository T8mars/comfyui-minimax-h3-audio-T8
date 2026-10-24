"""Prepare or explicitly stream-convert a plain native H3 base, on CPU only.

No downloads, checkpoint code, expert/DMD merge, quantization or node imports.
The source argument is the directory containing transformer/. A complete file
manifest is mandatory; the parent's download verified.json is accepted directly.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
import types


def module():
    name = "_t8_ptd_plain_base_cpu"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "h3_t8")]
        sys.modules[name] = package
    return importlib.import_module(name + ".ptd_base_conversion")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true", help="Default: metadata only, no Torch or payload")
    modes.add_argument("--convert", action="store_true", help="Explicit full SHA scans and CPU byte conversion")
    parser.add_argument("--job", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path, help="Optional new JSON report, never overwritten")
    parser.add_argument("--chunk-bytes", type=int, default=8 * 1024**2)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.chunk_bytes <= 0 or args.threads <= 0:
        parser.error("chunk-bytes and CPU threads must be positive integers")
    if args.report is not None and args.report.exists():
        parser.error("Report already exists; choose a new receipt path")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    start = time.perf_counter()
    api = module()
    manifest = api._json(args.manifest.read_bytes())
    if args.convert:
        if args.job is None or args.output is None:
            parser.error("--convert requires explicit --job and --output")
        import torch
        torch.set_num_threads(args.threads)
        result = api.run_conversion(args.source, args.job, args.output, manifest,
            chunk_bytes=args.chunk_bytes, progress=lambda row: print(json.dumps(row), flush=True))
    else:
        result = api.prepare(args.source, manifest)
        result["status"] = "metadata_prepared_payloads_and_complete_SHA_not_verified"
    result["wall_seconds"] = time.perf_counter() - start
    result["Torch_imported"] = "torch" in sys.modules
    result["CUDA_initialized"] = "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized()
    result["downloaded_or_environment_installed"] = False
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        api.atomic_json(args.report, result)
    summary = {key: value for key, value in result.items() if key not in ("targets", "files", "config", "normalized_config")}
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
