"""A completed LOW x0 is a separate, source-bound handoff, not x_sigma."""

import json

import pytest
import torch

from h3_audio_t8_pkg.modular_sampling.fast_h3_v2_handoff import completed_low_x0
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2_handoff_nodes import (
    MiniMaxH3FastH3V2CompletedLowX0EXPT8,
)
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from h3_audio_t8_pkg.modular_sampling.storage import load_stage, save_stage
from test_modular_results import inputs


def test_completed_low_exposes_exact_denoised_output_and_original_receipt():
    result = sample_stage(*inputs("low_0_4"))[2]
    x0, report_json = completed_low_x0(result)
    report = json.loads(report_json)
    assert x0 is result.denoised_output and x0 is not result.output
    assert report["request_sha256"] == result.verify()["request_sha256"]
    assert report["status"] == "verified_low_x0"
    assert MiniMaxH3FastH3V2CompletedLowX0EXPT8.GET_NODE_INFO_V1()["output"] == ["LATENT", "STRING"]
    assert not torch.cuda.is_initialized()


def test_high_result_cannot_be_mistaken_for_completed_low():
    result = sample_stage(*inputs("high_4_8"))[2]
    with pytest.raises(ValueError, match="not the exact FastH3 V2 LOW"):
        completed_low_x0(result)


def test_exact_frozen_low_result_exposes_same_x0_without_resampling(tmp_path):
    result = sample_stage(*inputs("low_0_4"))[2]
    path, digest, _ = save_stage(result, tmp_path)
    frozen = load_stage(tmp_path, path, digest, "low_0_4")[3]
    x0, report_json = completed_low_x0(frozen)
    assert json.loads(report_json)["request_sha256"] == result.verify()["request_sha256"]
    for actual, original in zip(x0["samples"].unbind(), result.denoised_output["samples"].unbind()):
        assert torch.equal(actual, original) and actual.data_ptr() != original.data_ptr()


def test_unverified_or_mutated_result_never_reaches_learned_handoff():
    args = list(inputs("low_0_4"))
    args[2].sampler_function = lambda denoiser, x, sigmas, **kwargs: x
    incomplete = sample_stage(*args)[2]
    with pytest.raises(ValueError, match="did not complete"):
        completed_low_x0(incomplete)
    completed = sample_stage(*inputs("low_0_4"))[2]
    completed.denoised_output["samples"].unbind()[0].add_(.01)
    with pytest.raises(ValueError, match="mutated"):
        completed_low_x0(completed)
