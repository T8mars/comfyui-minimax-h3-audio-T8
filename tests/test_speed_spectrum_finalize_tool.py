from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import torch

from h3_audio_t8_pkg.speed_advanced import accumulate_spectrum_dataset
from h3_audio_t8_pkg.speed_spectrum_storage import save_spectrum_dataset_file
from h3_audio_t8_pkg.tools.finalize_h3_speed_spectrum_dataset import finalize_file


def test_finalize_tool_cli_bootstraps_project_package_without_pythonpath():
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "tools" / "finalize_h3_speed_spectrum_dataset.py"),
            "--help",
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Finalize one persisted H3 SPEED spectrum dataset" in completed.stdout


def test_finalize_tool_loads_persisted_dataset_and_keeps_small_probe_research(tmp_path):
    generator = torch.Generator().manual_seed(71)
    latent = torch.randn(2, 24, 4, 16, 24, generator=generator)
    flattened = latent.reshape(-1, 1, 16, 24)
    for _ in range(3):
        flattened = torch.nn.functional.avg_pool2d(
            flattened, kernel_size=3, stride=1, padding=1
        )
    latent = flattened.reshape(2, 24, 4, 16, 24)
    dataset, _ = accumulate_spectrum_dataset(
        latent,
        batch_id="two-clips",
        task_family="T2VA",
        checkpoint_fingerprint="sha256:model",
        vae_fingerprint="sha256:vae",
        max_temporal_samples=4,
    )
    save_spectrum_dataset_file(
        dataset,
        root=tmp_path,
        dataset_name="probe",
        overwrite=False,
    )
    result = finalize_file(
        storage_root=tmp_path,
        dataset_name="probe",
        profile_name="probe-profile",
        minimum_r_squared=0.0,
        minimum_independent_clips=100,
    )
    assert result["storage"]["action"] == "load"
    assert result["profile"]["independent_clip_count"] == 2
    assert result["profile"]["status"] == "research_probe_only"
    assert result["profile"]["validated_for_delta_optimal"] is False
