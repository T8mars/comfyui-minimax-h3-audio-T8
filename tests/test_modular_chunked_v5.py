"""v5 explicit global lift + one-window joint AV parity with the old executor."""
from types import SimpleNamespace
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import comfy.nested_tensor
import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_parity as parity
from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as legacy
from h3_audio_t8_pkg.modular_sampling import chunked_v5
from h3_audio_t8_pkg.modular_sampling.chunked_v5_storage_nodes import (
    MiniMaxH3ChunkedV5WindowLoadEXPT8, MiniMaxH3ChunkedV5WindowSaveEXPT8,
)
from h3_audio_t8_pkg.modular_sampling.chunked_v5_storage import _prepared_sha, load_window, save_window
from h3_audio_t8_pkg.modular_sampling.results import _input_identity
from test_chunked_two_pass_parity import _plan


def make_v5_harness(monkeypatch):
    calls = {"lift": 0, "noise": 0, "sample": 0, "sampler_bind": 0}

    def lift(latent, *_args):
        calls["lift"] += 1
        video, audio = latent["samples"].unbind()
        out = dict(latent)
        out["samples"] = comfy.nested_tensor.NestedTensor((
            video.repeat_interleave(2, -1).repeat_interleave(2, -2), audio,
        ))
        return out, 64, 64, "{}"

    def sample(piece, _positive, _model, _noise, _sampler, _sigmas,
               _negative, _cfg, *, prepared_noise):
        calls["sample"] += 1
        assert prepared_noise.is_nested
        video, audio = piece["samples"].unbind()
        video_mask, audio_mask = piece["noise_mask"].unbind()
        return comfy.nested_tensor.NestedTensor((video + video_mask * 2,
                                                  audio + audio_mask * 3))

    def bind(_model, _piece, sampler):
        calls["sampler_bind"] += 1
        return sampler

    class Noise:
        seed = 23

        def generate_noise(self, latent):
            calls["noise"] += 1
            return comfy.nested_tensor.NestedTensor(tuple(
                torch.ones_like(item) for item in latent["samples"].unbind()
            ))

    def sample_minimax_h3_dual_clock_euler():
        pass

    sample_minimax_h3_dual_clock_euler._minimax_h3_shift_video = 12.
    sample_minimax_h3_dual_clock_euler._minimax_h3_shift_audio = 3.
    monkeypatch.setattr(legacy, "learned_upscale_h3_av_latent", lift)
    monkeypatch.setattr(legacy, "sample_piece", sample)
    monkeypatch.setattr(parity, "rebind_dual_clock_sampler", bind)
    monkeypatch.setattr(chunked_v5, "rebind_dual_clock_sampler", bind)
    sampler = SimpleNamespace(sampler_function=sample_minimax_h3_dual_clock_euler)
    source = {"samples": comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, 15, 2, 2), torch.zeros(1, 32, 2, 90),
    ))}
    positive = [[torch.zeros(1), {}]]
    return calls, source, positive, Noise(), sampler, torch.tensor(
        parity.UPSTREAM_REFINE_VIDEO_SIGMAS[4], dtype=torch.float32,
    ), _plan()


@pytest.fixture
def v5_harness(monkeypatch):
    return make_v5_harness(monkeypatch)


def test_v5_prepared_identity_ignores_only_lift_runtime_telemetry(v5_harness):
    _calls, source, _positive, noise, _sampler, _sigmas, plan = v5_harness
    lifted, receipt, _ = chunked_v5.lift_standard_joint(source, plan)
    prepared, _ = chunked_v5.prepare_standard_joint(source, lifted, receipt, plan, noise)
    report = {"model": {"sha256": "a" * 64, "cache_hit": False},
              "geometry": {"output_width": 64, "memory_warning": "first"},
              "memory_before": {"free_mib": 100}, "memory_after_release": {"free_mib": 200},
              "gpu_weights_released": True, "cpu_cache_cleared": False,
              "status": "ok"}
    first = replace(prepared, lift=replace(receipt, upscale_report=report))
    changed_telemetry = {**report, "model": {**report["model"], "cache_hit": True},
                         "geometry": {**report["geometry"], "memory_warning": None},
                         "memory_before": {"free_mib": 50},
                         "memory_after_release": {"free_mib": 300},
                         "gpu_weights_released": False, "cpu_cache_cleared": True}
    second = replace(prepared, lift=replace(receipt, upscale_report=changed_telemetry))
    assert _prepared_sha(first) == _prepared_sha(second)
    changed_model = {**changed_telemetry, "model": {**changed_telemetry["model"],
                                                    "sha256": "b" * 64}}
    assert _prepared_sha(first) != _prepared_sha(replace(
        prepared, lift=replace(receipt, upscale_report=changed_model)))
    changed_noise = replace(prepared, video_noise=prepared.video_noise.clone())
    changed_noise.video_noise[0, 0, 0, 0, 0] += 1
    assert _prepared_sha(first) != _prepared_sha(replace(
        changed_noise, lift=replace(receipt, upscale_report=report)))


def test_v5_explicit_two_windows_match_legacy_joint_av_tensor_for_tensor(v5_harness):
    calls, source, positive, noise, sampler, sigmas, plan = v5_harness
    legacy_output, _ = legacy.execute_chunked_two_pass_upscale(
        object(), positive, source, noise, sampler, sigmas, plan,
    )
    lifted, receipt, _ = chunked_v5.lift_standard_joint(source, plan)
    assert len(receipt.segments) == 2
    prepared, _ = chunked_v5.prepare_standard_joint(source, lifted, receipt, plan, noise)
    first, first_result, first_report = chunked_v5.sample_standard_window(
        object(), positive, source, lifted, prepared, plan, noise, sampler, sigmas, 0,
    )
    assert first_result.index == 0 and '"completed": false' in first_report
    final, final_result, final_report = chunked_v5.sample_standard_window(
        object(), positive, source, lifted, prepared, plan, noise, sampler, sigmas, 1,
        first_result,
    )
    assert final_result.index == 1 and '"completed": true' in final_report
    for new, old in zip(final["samples"].unbind(), legacy_output["samples"].unbind()):
        assert torch.equal(new, old)
    assert final["samples"].tensors[1].shape[-1] == 90
    assert not torch.equal(final["samples"].tensors[1], source["samples"].tensors[1])
    assert calls == {"lift": 2, "noise": 2, "sample": 4, "sampler_bind": 4}
    assert first["samples"].tensors[0].shape[2] < final["samples"].tensors[0].shape[2]


def test_v5_rejects_missing_previous_and_post_lift_mutation(v5_harness):
    _calls, source, positive, noise, sampler, sigmas, plan = v5_harness
    lifted, receipt, _ = chunked_v5.lift_standard_joint(source, plan)
    prepared, _ = chunked_v5.prepare_standard_joint(source, lifted, receipt, plan, noise)
    with pytest.raises(ValueError, match="previous PASS2 result"):
        chunked_v5.sample_standard_window(
            object(), positive, source, lifted, prepared, plan, noise, sampler, sigmas, 1,
        )
    lifted["samples"].tensors[0][0, 0, 0, 0, 0] = 7
    with pytest.raises(ValueError, match="identity changed"):
        chunked_v5.sample_standard_window(
            object(), positive, source, lifted, prepared, plan, noise, sampler, sigmas, 0,
        )


def test_v5_rejects_nonstandard_refine_sigmas_before_sampling(v5_harness):
    calls, source, positive, noise, sampler, _sigmas, plan = v5_harness
    lifted, receipt, _ = chunked_v5.lift_standard_joint(source, plan)
    prepared, _ = chunked_v5.prepare_standard_joint(source, lifted, receipt, plan, noise)
    with pytest.raises(ValueError, match="four standard intervals"):
        chunked_v5.sample_standard_window(
            object(), positive, source, lifted, prepared, plan, noise, sampler,
            torch.linspace(1, 0, 5), 0,
        )
    assert calls["sample"] == 0


def test_v5_explicit_window_freeze_resumes_only_second_joint_av_window(
        v5_harness, tmp_path):
    calls, source, positive, noise, sampler, sigmas, plan = v5_harness
    lifted, lift_receipt, _ = chunked_v5.lift_standard_joint(source, plan)
    prepared, _ = chunked_v5.prepare_standard_joint(
        source, lifted, lift_receipt, plan, noise)
    _first_output, first, _ = chunked_v5.sample_standard_window(
        object(), positive, source, lifted, prepared, plan, noise, sampler, sigmas, 0)
    expected, _second, _ = chunked_v5.sample_standard_window(
        object(), positive, source, lifted, prepared, plan, noise, sampler, sigmas, 1,
        first)
    _output, _result, path, digest, manifest_json = save_window(
        first, source, lifted, prepared, plan, tmp_path)
    manifest = json.loads(manifest_json)
    assert manifest["automatic_cache_reuse"] is False
    assert manifest["execution_identity_certified"] is False

    fresh_source = {"samples": comfy.nested_tensor.NestedTensor(tuple(
        value.clone() for value in source["samples"].unbind()))}
    fresh_lifted, fresh_receipt, _ = chunked_v5.lift_standard_joint(
        fresh_source, plan)
    fresh_prepared, _ = chunked_v5.prepare_standard_joint(
        fresh_source, fresh_lifted, fresh_receipt, plan, noise)
    count_before = calls["sample"]
    loaded_output, loaded, report_json = load_window(
        fresh_source, fresh_lifted, fresh_prepared, plan, tmp_path, path, digest, 0)
    assert loaded.index == 0 and loaded_output is loaded.output_latent
    assert json.loads(report_json)["automatic_cache_reuse"] is False
    assert calls["sample"] == count_before
    actual, _final, _ = chunked_v5.sample_standard_window(
        object(), positive, fresh_source, fresh_lifted, fresh_prepared,
        plan, noise, sampler, sigmas, 1, loaded)
    assert calls["sample"] == count_before + 1
    for got, want in zip(actual["samples"].unbind(), expected["samples"].unbind(),
                         strict=True):
        assert torch.equal(got, want)


def test_v5_frozen_window_resumes_only_second_window_in_new_python_process(
        v5_harness, tmp_path):
    _calls, source, positive, noise, sampler, sigmas, plan = v5_harness
    lifted, lift_receipt, _ = chunked_v5.lift_standard_joint(source, plan)
    prepared, _ = chunked_v5.prepare_standard_joint(
        source, lifted, lift_receipt, plan, noise)
    _first_output, first, _ = chunked_v5.sample_standard_window(
        object(), positive, source, lifted, prepared, plan, noise, sampler, sigmas, 0)
    expected, _second, _ = chunked_v5.sample_standard_window(
        object(), positive, source, lifted, prepared, plan, noise, sampler, sigmas, 1,
        first)
    _output, _result, artifact_path, artifact_sha, _manifest = save_window(
        first, source, lifted, prepared, plan, tmp_path)
    worker = Path(__file__).with_name("chunked_v5_storage_worker.py")
    run = subprocess.run(
        [sys.executable, str(worker), str(tmp_path), artifact_path, artifact_sha],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        timeout=120, check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads(run.stdout.strip().splitlines()[-1])
    assert report["loaded_index"] == 0
    assert report["sample_count"] == 1
    assert report["video_identity"] == _input_identity(expected["samples"].tensors[0])
    assert report["audio_identity"] == _input_identity(expected["samples"].tensors[1])


def test_v5_four_windows_freeze_after_third_then_cold_resume_only_fourth(
        v5_harness, tmp_path):
    calls, _source, positive, noise, sampler, sigmas, _plan_15 = v5_harness
    source = {"samples": comfy.nested_tensor.NestedTensor((
        torch.zeros(1, 24, 57, 2, 2), torch.zeros(1, 32, 2, 320),
    ))}
    plan = _plan(temporal_chunk_frames=85, temporal_overlap_frames=34)
    lifted, receipt, _ = chunked_v5.lift_standard_joint(source, plan)
    assert len(receipt.segments) == 4
    prepared, _ = chunked_v5.prepare_standard_joint(source, lifted, receipt, plan, noise)
    previous = None
    frozen = None
    for index in range(4):
        expected, previous, _ = chunked_v5.sample_standard_window(
            object(), positive, source, lifted, prepared, plan, noise,
            sampler, sigmas, index, previous)
        if index == 2:
            frozen = previous
    assert calls["sample"] == 4 and frozen.index == 2
    _out, _result, path, digest, _manifest = save_window(
        frozen, source, lifted, prepared, plan, tmp_path)
    worker = Path(__file__).with_name("chunked_v5_storage_worker.py")
    run = subprocess.run(
        [sys.executable, str(worker), str(tmp_path), path, digest, "4"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        timeout=120, check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    report = json.loads(run.stdout.strip().splitlines()[-1])
    assert report["loaded_index"] == 2 and report["sample_count"] == 1
    assert report["video_identity"] == _input_identity(expected["samples"].tensors[0])
    assert report["audio_identity"] == _input_identity(expected["samples"].tensors[1])


def test_v5_window_freeze_rejects_wrong_source_noise_sha_and_corruption(
        v5_harness, tmp_path):
    _calls, source, positive, noise, sampler, sigmas, plan = v5_harness
    lifted, lift_receipt, _ = chunked_v5.lift_standard_joint(source, plan)
    prepared, _ = chunked_v5.prepare_standard_joint(
        source, lifted, lift_receipt, plan, noise)
    _output, first, _ = chunked_v5.sample_standard_window(
        object(), positive, source, lifted, prepared, plan, noise, sampler, sigmas, 0)
    _output, _result, path, digest, _manifest = save_window(
        first, source, lifted, prepared, plan, tmp_path)
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_window(source, lifted, prepared, plan, tmp_path, path, "0" * 64, 0)
    with pytest.raises(ValueError, match="traversal|escapes|relative"):
        load_window(source, lifted, prepared, plan, tmp_path, "../manifest.json", digest, 0)
    with pytest.raises(ValueError, match="does not match"):
        load_window(source, lifted, prepared, plan, tmp_path, path, digest, 1)
    altered_source = {"samples": comfy.nested_tensor.NestedTensor(tuple(
        value.clone() for value in source["samples"].unbind()))}
    altered_source["samples"].tensors[0][..., 0, 0] += 1
    changed_lift, changed_receipt, _ = chunked_v5.lift_standard_joint(altered_source, plan)
    changed_prepared, _ = chunked_v5.prepare_standard_joint(
        altered_source, changed_lift, changed_receipt, plan, noise)
    with pytest.raises(ValueError, match="does not match"):
        load_window(altered_source, changed_lift, changed_prepared, plan,
                    tmp_path, path, digest, 0)
    prepared.video_noise[0, 0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="does not match"):
        load_window(source, lifted, prepared, plan, tmp_path, path, digest, 0)
    prepared.video_noise[0, 0, 0, 0, 0] -= 1
    with pytest.raises(ValueError, match="geometry or content"):
        save_window(replace(first, index=1), source, lifted, prepared, plan, tmp_path)
    state = tmp_path / path.replace("manifest.json", "state.safetensors")
    with state.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="size or SHA"):
        load_window(source, lifted, prepared, plan, tmp_path, path, digest, 0)


def test_v5_window_storage_nodes_are_append_only():
    save = MiniMaxH3ChunkedV5WindowSaveEXPT8.GET_NODE_INFO_V1()
    load = MiniMaxH3ChunkedV5WindowLoadEXPT8.GET_NODE_INFO_V1()
    assert save["output_name"] == ["cumulative_av_latent", "window_result",
                                   "artifact_path", "artifact_sha256", "report_json"]
    assert load["output_name"] == ["cumulative_av_latent", "window_result", "report_json"]
