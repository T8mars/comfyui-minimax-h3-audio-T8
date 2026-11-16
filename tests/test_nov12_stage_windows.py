"""Real Stage/automatic-reuse consumers retain conditional LoRA execution."""

import json

import torch
from comfy_extras.nodes_custom_sampler import BasicGuider

from h3_audio_t8_pkg.lora_windows import create_window_hook
from h3_audio_t8_pkg.modular_sampling import results, automatic_reuse
from test_modular_native_explicit import inputs
from test_progressive_sampling_runtime import conditioning
from test_stage_reuse import _write_adapter


def selected(tmp_path):
    bound, _, original = inputs()
    args = list(bound)
    path = tmp_path / "window.safetensors"
    _write_adapter(path, 124)
    hooks, _ = create_window_hook(args[1].model_patcher, path, 0.7, 0.0, 0.65)
    cond = conditioning()
    cond[0][1]["hooks"] = hooks
    args[1] = BasicGuider.execute(args[1].model_patcher, cond).result[0]
    return args


def test_two_explicit_Stage_runs_execute_windows_without_false_portable_identity(
    tmp_path,
):
    args = selected(tmp_path)
    first = results.sample_stage(*args)
    second = results.sample_stage(*args)
    for output in (first, second):
        receipt = output[2].verify()
        assert not receipt["cache_reuse_authorized"]
        assert receipt["execution"]["denoiser_evaluations"] == 4
    for a, b in zip(first[0]["samples"].unbind(), second[0]["samples"].unbind()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_automatic_reuse_executes_windows_and_does_not_fake_a_hit(tmp_path):
    args = selected(tmp_path)
    for _ in range(2):
        output = automatic_reuse.sample(
            *args[:5],
            tmp_path / "cache",
            mode="reuse_completed",
            stage_role="native_low",
        )
        report = json.loads(output[3])
        assert report["status"] == "uncertified_executed"
        assert report["artifact_path"] is None
        assert all(torch.isfinite(t).all() for t in output[0]["samples"].unbind())
