"""External noise is the original video rescheduler, not another sampler."""
import json

import pytest
import torch
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_custom_sampler import RandomNoise, Noise_EmptyNoise

from h3_audio_t8_pkg import freenoise_advanced as legacy
from h3_audio_t8_pkg.modular_sampling.noise import build_noise, operator_identity
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from test_modular_native_dual import inputs
from test_progressive_sampling_runtime import latent
from test_fast_h3_v2_core_sampler import model


@pytest.mark.parametrize("mode", legacy.MODES)
@pytest.mark.parametrize("ratio", [0., .65, 1.])
@pytest.mark.parametrize("segment", [0, 3])
@pytest.mark.parametrize("batch_indices", [None, [1, 3]])
def test_exact_legacy_noise_with_seed_batch_and_segment(mode, ratio, segment, batch_indices):
    source = latent()
    if batch_indices is not None:
        source["samples"] = NestedTensor(tuple(value.repeat(2, *([1] * (value.ndim - 1)))
                                                  for value in source["samples"].unbind()))
        source["batch_index"] = batch_indices
    delegate = RandomNoise.execute(37).result[0]
    planned, _ = legacy.build_free_noise_model(model(), mode=mode, base_seed=55, reuse_ratio=ratio)
    expected, _ = legacy.reschedule_h3_noise(delegate.generate_noise(source),
        config=legacy.free_noise_config(planned), segment_index=segment)
    direct, report = build_noise(delegate, mode, 55, ratio, segment)
    inherited, _ = build_noise(delegate, "from_model_plan", segment_index=segment, model=planned)
    for provider in (direct, inherited):
        assert provider.seed == 37
        actual = provider.generate_noise(source)
        assert all(torch.equal(a, b) for a, b in zip(actual.unbind(), expected.unbind()))
        assert operator_identity(provider)["portable"] is True
    assert json.loads(report)["noise_generated"] is False
    assert not torch.cuda.is_initialized()


def test_audio_same_object_and_source_called_once():
    class RecordingNoise:
        seed = 19

        def __init__(self):
            self.calls = []

        def generate_noise(self, source):
            value = RandomNoise.execute(self.seed).result[0].generate_noise(source)
            self.calls.append(value)
            return value

    source = RecordingNoise()
    wrapped, _ = build_noise(source, "paper_permutation", segment_index=2)
    result = wrapped.generate_noise(latent())
    assert len(source.calls) == 1
    assert result.unbind()[1] is source.calls[0].unbind()[1]
    assert operator_identity(wrapped)["portable"] is False


def test_disabled_and_absent_model_plan_are_exact_bypass():
    for source in (RandomNoise.execute(31).result[0], Noise_EmptyNoise()):
        assert build_noise(source)[0] is source
        assert build_noise(source, "from_model_plan", model=model())[0] is source
    with pytest.raises(ValueError, match="MODEL"):
        build_noise(source, "from_model_plan")


@pytest.mark.parametrize("segment", [0, 2])
def test_external_noise_result_identity_and_sampling_are_stable(segment):
    args = list(inputs()[0])
    args[0] = build_noise(args[0], "variance_preserving_blend", 71, .65, segment)[0]
    first = sample_stage(*args)[2].verify()
    hot = sample_stage(*args)[2].verify()
    assert first["portable_identity"] is hot["portable_identity"] is True
    assert first["request_sha256"] == hot["request_sha256"]
    assert first["request"]["noise_operator"]["segment_index"] == segment
    assert first["execution"]["denoiser_evaluations"] == 4


def test_noise_provider_mutation_during_sampling_rejects_completion():
    args = list(inputs()[0])
    args[0] = build_noise(args[0], "variance_preserving_blend", 71, .65, 2)[0]
    original = args[1].sample

    def change_after(*values, **kwargs):
        output = original(*values, **kwargs)
        object.__setattr__(args[0], "segment_index", 3)
        return output

    args[1].sample = change_after
    with pytest.raises(ValueError, match="noise provider changed"):
        sample_stage(*args)
