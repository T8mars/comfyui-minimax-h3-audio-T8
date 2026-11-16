"""Real tiny Core DMAD interruption/retry; not trained media qualification."""
import copy
import json
import sys

import pytest
import torch
import comfy.model_management as mm
import latent_preview
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise, SamplerCustomAdvanced

from h3_audio_t8_pkg.dmad_renoise import prepare_dmad_renoise
from h3_audio_t8_pkg.modular_sampling import automatic_reuse
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import conditioning, latent


class ForwardCounts:
    """Observe without installing a hook into the cache-authenticated model."""
    def __init__(self, network):
        self.network = network
        self.code = network.forward.__func__.__code__
        self.parent = sys.getprofile()
        self.reset()

    def reset(self):
        self.attempted = self.completed = 0

    def __call__(self, frame, event, value):
        if frame.f_code is self.code and frame.f_locals.get("self") is self.network:
            if event == "call":
                self.attempted += 1
            elif event == "return" and value is not None:
                self.completed += 1
        if self.parent is not None:
            self.parent(frame, event, value)


def callback_interrupt(monkeypatch):
    original = latent_preview.prepare_callback
    observations = []

    def prepare(*args, **kwargs):
        parent = original(*args, **kwargs)

        def callback(step, *values):
            if parent is not None:
                parent(step, *values)
            observations.append(step)
            if step == 1:
                mm.interrupt_current_processing(True)

        return callback

    monkeypatch.setattr(latent_preview, "prepare_callback", prepare)
    return observations, original


@pytest.mark.parametrize("steps", [4, 8])
def test_cached_guider_real_interrupt_then_same_seed_retry_is_bitexact(monkeypatch, tmp_path, steps):
    source = latent()
    before = [x.clone() for x in source["samples"].unbind()]
    positive = conditioning("i2va")
    old_condition = copy.deepcopy(positive)
    selected, sampler, sigmas, _ = prepare_dmad_renoise(model(), source, steps)
    guider = BasicGuider.execute(selected, positive).result[0]
    noise = RandomNoise.execute(121201).result[0]
    calls = ForwardCounts(selected.model.diffusion_model)
    sys.setprofile(calls)
    try:
        clean = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, source).result
        assert calls.completed == calls.attempted == steps
        callbacks, original = callback_interrupt(monkeypatch)
        calls.reset()
        try:
            with pytest.raises(mm.InterruptProcessingException):
                SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, source)
        finally:
            mm.interrupt_current_processing(False)
            monkeypatch.setattr(latent_preview, "prepare_callback", original)
        assert callbacks == [0, 1] and calls.completed == 2
        assert 2 <= calls.attempted <= 3  # Core may check interruption inside the next forward.
        interrupted = dict(attempted=calls.attempted, completed=calls.completed)
        calls.reset()
        retried = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, source).result
        assert calls.completed == calls.attempted == steps
        for old_output, retry_output in zip(clean, retried):
            for a, b in zip(old_output["samples"].unbind(), retry_output["samples"].unbind()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        changed = SamplerCustomAdvanced.execute(
            RandomNoise.execute(noise.seed + 1).result[0], guider, sampler, sigmas, source
        ).result[0]
        assert any(not torch.equal(a, b) for a, b in zip(
            retried[0]["samples"].unbind(), changed["samples"].unbind()))
    finally:
        sys.setprofile(calls.parent)
        mm.interrupt_current_processing(False)
    assert all(torch.equal(a, b) for a, b in zip(before, source["samples"].unbind()))
    assert torch.equal(positive[0][0], old_condition[0][0])
    assert torch.equal(positive[0][1]["minimax_keyframes"][0]["latent"],
                       old_condition[0][1]["minimax_keyframes"][0]["latent"])
    (tmp_path / "DMAD-retry-evidence.json").write_text(json.dumps(dict(
        steps=steps, interrupted=interrupted, retry_completed=steps,
        same_cached_guider=True, same_seed_outputs_bitexact=True,
        changed_seed_changes_output=True, source_and_keyframe_unmodified=True,
        consumer="actual tiny random Core LCM/FLOW_AV", trained_model=False,
        cuda_initialized=torch.cuda.is_initialized())), "utf8")
    assert not torch.cuda.is_initialized()


def test_stage_reuse_interrupt_never_publishes_a_partial_dmad_completion(monkeypatch, tmp_path):
    source = latent()
    selected, sampler, sigmas, _ = prepare_dmad_renoise(model(), source, 8)
    guider = BasicGuider.execute(selected, conditioning()).result[0]
    noise = RandomNoise.execute(121202).result[0]
    callbacks, original = callback_interrupt(monkeypatch)
    calls = ForwardCounts(selected.model.diffusion_model)
    sys.setprofile(calls)
    try:
        try:
            with pytest.raises(mm.InterruptProcessingException):
                automatic_reuse.sample(noise, guider, sampler, sigmas, source, tmp_path,
                                       mode="reuse_completed")
        finally:
            mm.interrupt_current_processing(False)
            monkeypatch.setattr(latent_preview, "prepare_callback", original)
        assert callbacks == [0, 1] and calls.completed == 2
        assert 2 <= calls.attempted <= 3
        interrupted = dict(attempted=calls.attempted, completed=calls.completed)
        assert not list(tmp_path.rglob("manifest.json"))
        calls.reset()
        first = automatic_reuse.sample(noise, guider, sampler, sigmas, source, tmp_path,
                                       mode="reuse_completed")
        assert calls.completed == calls.attempted == 8
        report = json.loads(first[3])
        assert report["status"] in ("miss_saved", "uncertified_executed")
        calls.reset()
        replay = automatic_reuse.sample(noise, guider, sampler, sigmas, source, tmp_path,
                                        mode="reuse_completed")
        next_report = json.loads(replay[3])
        if report["status"] == "miss_saved":
            assert next_report["status"] == "verified_hit" and calls.attempted == 0
            assert next_report["current_denoiser_evaluations"] == 0
        else:
            assert next_report["status"] == "uncertified_executed" and calls.completed == calls.attempted == 8
        for a, b in zip(first[0]["samples"].unbind(), replay[0]["samples"].unbind()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        (tmp_path / "DMAD-Stage-retry-evidence.json").write_text(json.dumps(dict(
            interrupted=interrupted, partial_completion_published=False,
            first_success_status=report["status"], first_success_completed=8,
            replay_status=next_report["status"], replay_completed=calls.completed,
            output_bitexact=True, consumer="actual tiny random Core Stage and LCM/FLOW_AV",
            trained_model=False, cuda_initialized=torch.cuda.is_initialized())), "utf8")
    finally:
        sys.setprofile(calls.parent)
        mm.interrupt_current_processing(False)
    assert not torch.cuda.is_initialized()
