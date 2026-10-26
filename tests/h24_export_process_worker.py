"""Owned CPU export process for real restart tests; never starts a Core server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types


def main():
    parser = argparse.ArgumentParser()
    for name in ("core", "user", "input", "project", "film", "job"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--pause", type=Path)
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.argv = ["owned-export-worker", "--cpu"]
    sys.path.insert(0, options.core)
    import torch
    torch.set_num_threads(2)
    # Import the real export/store implementation without registering hundreds
    # of unrelated generation nodes or starting an inference server.
    package = types.ModuleType("h3_audio_t8_pkg")
    package.__path__ = [str(root / "h3_t8")]
    sys.modules[package.__name__] = package
    from h3_audio_t8_pkg import director_film_export as exporter
    from h3_audio_t8_pkg.director_project import ProjectStore
    store = ProjectStore(options.user, options.input)
    original = exporter._run
    calls = []

    def run(arguments, log):
        name = Path(arguments[-1]).name
        if options.pause is not None and name == "clip-0001.mov":
            # The first subprocess.run has returned; no encoder child is live
            # at this deliberately chosen process-interruption boundary.
            options.pause.write_text(json.dumps({"pid": os.getpid(), "calls": calls,
                "before_second_encoder_started": True, "cuda_initialized": torch.cuda.is_initialized()}),
                encoding="utf8")
            time.sleep(60)
            raise RuntimeError("Owned test pause expired")
        calls.append(name)
        return original(arguments, log)

    exporter._run = run
    result = exporter.run_export(store, options.project, options.film, options.job)
    print(json.dumps({"job": result, "calls": calls, "cuda_initialized": torch.cuda.is_initialized(),
        "implementation_sha256": hashlib.sha256(Path(exporter.__file__).read_bytes()).hexdigest()}), flush=True)
    assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    main()
