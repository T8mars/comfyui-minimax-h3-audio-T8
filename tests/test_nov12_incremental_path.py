"""Combined real CPU Stage paths; hook uncertainty never destroys valid LOW."""

import json
from copy import deepcopy

import pytest
import torch
from comfy_extras.nodes_custom_sampler import BasicGuider

from h3_audio_t8_pkg.learned_latent_upscale_advanced import reconcile_two_pass_h3_latent
from h3_audio_t8_pkg.lora_windows import create_window_hook
from h3_audio_t8_pkg.modular_sampling import automatic_reuse
from test_modular_native_explicit import inputs
from test_modular_rf_restart import source_latent
from test_progressive_sampling_runtime import conditioning
from test_stage_reuse import _write_adapter


def test_changed_HIGH_window_and_failed_delivery_preserve_actual_completed_LOW(
    tmp_path,
):
    _, _, low = inputs()
    cache = tmp_path / "cache"
    first = automatic_reuse.sample(
        *low, cache, mode="reuse_completed", stage_role="native_low"
    )
    assert json.loads(first[3])["status"] == "miss_saved"
    prepared, positive, _ = reconcile_two_pass_h3_latent(
        first[1], source_latent(True), conditioning(), "auto"
    )
    path = tmp_path / "content.safetensors"
    _write_adapter(path, 171)
    for strength in (0.3, -0.6):
        _, _, high = inputs(stage="native_high", source=prepared)
        high = list(high)
        selected = high[1].model_patcher
        hooks, _ = create_window_hook(selected, path, strength, 0.4, 0.9)
        cond = [[entry[0], {**entry[1], "hooks": hooks}] for entry in positive]
        high[1] = BasicGuider.execute(selected, cond).result[0]
        output = automatic_reuse.sample(
            *high, cache, mode="reuse_completed", stage_role="native_high"
        )
        assert json.loads(output[3])["status"] == "uncertified_executed"
        # A delivery failure after this return cannot erase or relabel LOW.
        hit = automatic_reuse.sample(
            *low, cache, mode="reuse_completed", stage_role="native_low"
        )
        report = json.loads(hit[3])
        assert (
            report["status"] == "verified_hit"
            and report["current_denoiser_evaluations"] == 0
        )
        for actual, saved in zip(
            hit[1]["samples"].unbind(), first[1]["samples"].unbind()
        ):
            assert torch.equal(actual, saved)


@pytest.mark.parametrize("changed_field", ["text", "keyframe"])
def test_actual_changed_condition_content_invalidates_only_the_affected_LOW_request(
    tmp_path,
    changed_field,
):
    _, _, original = inputs()
    original = list(original)
    first_condition = conditioning("i2va")
    first_condition[0][1]["minimax_keyframes"][0]["latent"] = torch.full(
        (1, 24, 1, *original[4]["samples"].unbind()[0].shape[-2:]), 0.2
    )
    original[1] = BasicGuider.execute(
        original[1].model_patcher, first_condition
    ).result[0]
    cache = tmp_path / "cache"
    first = automatic_reuse.sample(
        *original, cache, mode="reuse_completed", stage_role="native_low"
    )
    assert json.loads(first[3])["status"] == "miss_saved"
    # Actual text or keyframe tensor values change while shapes and counts
    # remain identical. A same-shape declaration is not a content identity.
    changed = list(original)
    positive = deepcopy(first_condition)
    if changed_field == "text":
        positive[0][0] = positive[0][0] + 0.2
    else:
        positive[0][1]["minimax_keyframes"][0]["latent"] += 0.2
    changed[1] = BasicGuider.execute(original[1].model_patcher, positive).result[0]
    next_result = automatic_reuse.sample(
        *changed, cache, mode="reuse_completed", stage_role="native_low"
    )
    assert json.loads(next_result[3])["status"] == "miss_saved"
    old_again = automatic_reuse.sample(
        *original, cache, mode="reuse_completed", stage_role="native_low"
    )
    assert json.loads(old_again[3])["status"] == "verified_hit"
