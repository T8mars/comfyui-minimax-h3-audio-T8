"""A stage must project back to its exact raw MODEL before current-job use."""

from copy import copy

import torch

from h3_audio_t8_pkg.long_video_dual_identity import stage_model_identity
from h3_audio_t8_pkg.modular_sampling.eav import EAVConfig, apply_stage_eav
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2 import build_stage
from h3_audio_t8_pkg.modular_sampling.fast_h3_v2_stage_attest import _raw_model_from_stage
from test_fast_h3_v2_core_sampler import model
from test_progressive_sampling_runtime import latent


def test_native_v2_stage_projects_to_its_raw_model_identity():
    raw, source = model(), latent()
    prepared, _, sigmas, context, _ = build_stage(raw, source, "low_0_4", "dense_compat_exp")
    assert _raw_model_from_stage(prepared) == stage_model_identity(raw)
    effected, _, _ = apply_stage_eav(prepared, sigmas, source, context, EAVConfig(mode="report_only"))
    assert _raw_model_from_stage(effected) == stage_model_identity(raw)


def test_v2_raw_identity_ignores_only_dormant_core_sampling_module():
    raw = model()
    changed = raw.clone()
    changed.model = copy(raw.model)
    changed.model._modules = dict(raw.model._modules)

    class DormantSampling(torch.nn.Module):
        def __init__(self, noise_scale):
            super().__init__()
            self.noise_scale = noise_scale

    changed.model._modules["model_sampling"] = DormantSampling(
        getattr(raw.model.model_sampling, "noise_scale", None))
    assert stage_model_identity(changed) != stage_model_identity(raw)
    assert stage_model_identity(changed, _omit_dormant_sampling=True) == stage_model_identity(
        raw, _omit_dormant_sampling=True)
