"""Load an exact frozen Face ER-SDE result in a fresh CPU interpreter."""
import json
import sys

import comfy.cli_args

comfy.cli_args.args.cpu = True

import conftest  # noqa: F401,E402  - bootstrap the local extension as a package
import torch  # noqa: E402

from h3_audio_t8_pkg.modular_sampling.face_stage import (  # noqa: E402
    audit_multiface_face_stage, audit_parity_face_stage, audit_window_face_stage,
)
from h3_audio_t8_pkg.modular_sampling.storage import load_stage  # noqa: E402
from test_face_refine_parity_advanced import _locked_av, _plan  # noqa: E402
from test_modular_face_multiface import _job as multiface_job  # noqa: E402
from test_modular_face_window import _job as window_job  # noqa: E402


def main():
    root, path, digest, variant = sys.argv[1:]
    assert not torch.cuda.is_initialized()
    output, _, _, result, _ = load_stage(root, path, digest, "native_high")
    torch.manual_seed(411)
    if variant == "parity":
        frames = torch.zeros((5, 64, 96, 3))
        face_plan = _plan(frames)[0]
        latent = _locked_av()
        candidate, audit = audit_parity_face_stage(result, face_plan, frames, latent)
    elif variant == "multiface":
        parent, frames, face_plan = multiface_job()
        latent = _locked_av()
        candidate, audit = audit_multiface_face_stage(result, face_plan, frames, parent, latent)
    elif variant == "window":
        parent, source_audio, window_plan, frames, window_audio, mapping, face_plan = window_job()
        latent = _locked_av(frame_count=22)
        candidate, audit = audit_window_face_stage(result, face_plan, frames, parent,
            window_plan, mapping, source_audio, window_audio, latent)
    else:
        raise ValueError("Unknown Face cold-load variant")
    receipt = result.verify()
    assert candidate is output
    assert receipt["portable_identity"] and receipt["verified_recipe_completion"]
    assert json.loads(audit)["variant"] == variant
    print(json.dumps({"status": "cold_face_source_bound", "variant": variant,
                      "receipt_sha256": receipt["receipt_sha256"], "cuda_initialized": torch.cuda.is_initialized()}))


if __name__ == "__main__":
    main()
