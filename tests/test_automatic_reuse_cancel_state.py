"""Cancelled native Core sampling must not poison the next request."""
import json

import pytest
import torch
import comfy.utils
import comfy.model_management as mm
from comfy_extras.nodes_custom_sampler import BasicGuider, RandomNoise

from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.modular_sampling import automatic_reuse as auto
from test_fast_h3_v2_core_sampler import model
from test_modular_rf_restart import source_latent
from test_progressive_sampling_runtime import conditioning
from test_stage_reuse import _write_adapter
from h3_audio_t8_pkg.h3_lora_compat_advanced import load_minimax_h3_lora_model


def native_inputs(adapter):
    selected = model()
    selected, _ = load_minimax_h3_lora_model(selected, adapter, .7)
    source = source_latent(True)
    configured, sampler, sigmas = sampling.setup_dual_clock_sampling(selected, source, 4, 12., 3.)
    return RandomNoise.execute(31).result[0], BasicGuider.execute(configured, conditioning()).result[0], sampler, sigmas, source


def test_after_actual_cancel_next_native_lora_result_matches_fresh_control(tmp_path):
    adapter = tmp_path/'adapter.safetensors'
    _write_adapter(adapter, 71)
    control = auto.sample(*native_inputs(adapter), tmp_path/'control', mode='reuse_completed')
    args = native_inputs(adapter)
    hook = comfy.utils.PROGRESS_BAR_HOOK
    cancelled = []
    def cancel(value, total, preview, **kwargs):
        if not cancelled and total == 4 and value == 1:
            cancelled.append(True)
            mm.interrupt_current_processing(True)
    try:
        comfy.utils.set_progress_bar_global_hook(cancel)
        with pytest.raises(mm.InterruptProcessingException):
            auto.sample(*args, tmp_path/'interrupted', mode='reuse_completed')
    finally:
        comfy.utils.set_progress_bar_global_hook(hook)
        mm.interrupt_current_processing(False)
    assert cancelled and not list((tmp_path/'interrupted').rglob('entry.json'))
    retried = auto.sample(*args, tmp_path/'interrupted', mode='reuse_completed')
    assert json.loads(retried[3])['status'] == 'miss_saved'
    for before, after in zip(control[:2], retried[:2]):
        for a, b in zip(before['samples'].unbind(), after['samples'].unbind()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
