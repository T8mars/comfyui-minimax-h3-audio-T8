from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "h3_audio_t8_nfe_restart_probe_pkg"


def _load_package() -> None:
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)


_load_package()

from h3_audio_t8_nfe_restart_probe_pkg.nfe_resume_advanced import (  # noqa: E402
    read_nfe_resume_checkpoint,
    sample_minimax_h3_dual_clock_euler_resumable,
)
from h3_audio_t8_nfe_restart_probe_pkg.sampling import native_flow_sigmas  # noqa: E402


class RuntimeModel:
    def __init__(self, noise, latent_image, *, fail_on_call=None):
        self.noise = noise.clone()
        self.latent_image = latent_image.clone()
        self.sigmas = None
        self.inner_model = None
        self.calls = 0
        self.fail_on_call = fail_on_call

    def __call__(self, x, sigma, **_kwargs):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("intentional process-one interruption")
        scale = sigma.reshape((sigma.shape[0],) + (1,) * (x.ndim - 1))
        return x * (0.2 + 0.1 * scale) + 0.05


def config(root: Path, sigmas, *, resume_state=None, write_enabled=True):
    return {
        "full_sigmas": sigmas,
        "resume_state": resume_state,
        "write_enabled": write_enabled,
        "target": root / "process_state.h3nfe.safetensors",
        "session_id": (
            resume_state["payload"]["session_id"]
            if resume_state is not None
            else "fedcba9876543210fedcba9876543210"
        ),
        "model_contract_id": "base=sha256:process-test; wrappers=none",
        "run_contract_sha256": "B" * 64,
        "max_chunk_bytes": 1024 * 1024,
        "allow_replace_existing": resume_state is not None,
    }


def sample(model, x, sigmas, checkpoint_config):
    return sample_minimax_h3_dual_clock_euler_resumable(
        model,
        x,
        sigmas,
        extra_args={"seed": 77, "model_options": {}},
        disable=True,
        video_values=2,
        packed_values=4,
        shift_video=12.0,
        shift_audio=3.0,
        audio_velocity_is_raw=True,
        checkpoint_config=checkpoint_config,
    )


def main() -> None:
    mode = sys.argv[1]
    root = Path(sys.argv[2])
    handoff = Path(sys.argv[3])
    root.mkdir(parents=True, exist_ok=True)
    sigmas = native_flow_sigmas(4, 12.0)
    noise = torch.tensor([[0.125, -0.25, 0.5, -0.75]], dtype=torch.float32)
    latent = torch.tensor([[0.05, 0.1, 0.15, 0.2]], dtype=torch.float32)
    if mode == "save":
        expected = sample(
            RuntimeModel(noise, latent),
            noise.clone(),
            sigmas,
            config(root / "reference", sigmas, write_enabled=False),
        )
        try:
            sample(
                RuntimeModel(noise, latent, fail_on_call=3),
                noise.clone(),
                sigmas,
                config(root, sigmas),
            )
        except RuntimeError as exc:
            if "intentional process-one interruption" not in str(exc):
                raise
        state = read_nfe_resume_checkpoint(root, "process_state.h3nfe.safetensors")
        handoff.write_text(
            json.dumps(
                {
                    "save_pid": __import__("os").getpid(),
                    "completed_steps": state["payload"]["completed_steps"],
                    "expected": expected.tolist(),
                    "file_sha256": state["file_sha256"],
                }
            ),
            encoding="utf-8",
        )
        print(handoff.read_text(encoding="utf-8"))
        return
    if mode == "resume":
        prior = json.loads(handoff.read_text(encoding="utf-8"))
        state = read_nfe_resume_checkpoint(root, "process_state.h3nfe.safetensors")
        output = sample(
            RuntimeModel(torch.zeros_like(noise), torch.zeros_like(latent)),
            torch.full_like(noise, 123.0),
            sigmas[state["payload"]["completed_steps"] :],
            config(root, sigmas, resume_state=state, write_enabled=False),
        )
        expected = torch.tensor(prior["expected"], dtype=output.dtype)
        result = {
            "save_pid": prior["save_pid"],
            "resume_pid": __import__("os").getpid(),
            "completed_steps": state["payload"]["completed_steps"],
            "file_sha256": state["file_sha256"],
            "bit_exact": bool(torch.equal(output, expected)),
        }
        print(json.dumps(result))
        if not result["bit_exact"]:
            raise SystemExit(2)
        return
    raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
