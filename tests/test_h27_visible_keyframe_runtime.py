"""Real tiny Core50+2 HF sampling with explicit visible endpoints, not media quality."""

import pytest
import torch
from h3_audio_t8_pkg.modular_sampling import hyperflow_fresh as fresh
from h3_audio_t8_pkg.multikeyframe_advanced import (
    ACTUAL_FRAME_INDEX,
    append_keyframe_plan,
)
from h3_audio_t8_pkg.nodes_director_keyframes import (
    MiniMaxH3DirectorKeyframeConditioningEXPT8,
)
from h3_audio_t8_pkg.prompt_relay_advanced import build_prompt_relay_plan
import test_h21_hyperflow_keyframe_runtime as hf_cases

from test_h21_director_keyframe_runtime import (
    CountingNativeLikeFakeClip,
    actual_payload,
)
from helpers import FakeAudioVAE

actual_base = hf_cases.actual_base
actual_video_encoder = hf_cases.actual_video_encoder
run_route, check_original = hf_cases.run_route, hf_cases.check_original


@pytest.mark.parametrize("relay", [False, True])
@pytest.mark.parametrize("route", ["single8", "combined", "separate"])
def test_actual_core_hf_visible_tail_before_hidden_middle_and_next_request(
    actual_base, actual_video_encoder, relay, route
):
    base, weights, original = actual_base
    image = torch.linspace(0.05, 0.95, 32 * 32 * 3).reshape(1, 32, 32, 3)

    def prepare():
        selected = fresh.install(base, weights)[0]
        middle = append_keyframe_plan(
            None, image * 0.9, "frame", 15, 0.999, "stretch", True
        )
        kwargs = dict(
            model=selected,
            clip=CountingNativeLikeFakeClip(),
            video_vae=actual_video_encoder,
            audio_vae=FakeAudioVAE(),
            prompt="A continuous shot.",
            width=32,
            height=32,
            length=22,
            task_type="FL2VA",
            audio_mode="native",
            audio_denoise_strength=0.35,
            add_source_as_reference=True,
            prompt_primary_audio_ordinal=0,
            strict_prompt_tags=True,
            ref_image_size="match",
            reference_video_policy="official_2_to_15s",
            first_frame=image,
            last_frame=image * 0.8,
            first_frame_index=2,
            last_frame_index=12,
            keyframe_plan=middle,
        )
        if relay:
            kwargs.update(
                prompt_relay_plan=build_prompt_relay_plan(
                    "A continuous shot.",
                    "Raise hand.\nTurn.",
                    22,
                    "auto_equal",
                    "",
                    "paper_v1",
                    0.1,
                    False,
                    False,
                )[0],
                execution_mode="apply_exp",
                query_chunk_rows=64,
            )
        result = MiniMaxH3DirectorKeyframeConditioningEXPT8.execute(**kwargs).result
        payload = actual_payload(*result[:3])
        assert [item[ACTUAL_FRAME_INDEX] for item in payload["keyframes"]] == [
            2,
            12,
            15,
        ]
        return result

    low, high = prepare(), prepare()
    run_route(low, high, route)
    check_original(base, original)
    # A fresh next request must not inherit a previous forward/phase owner.
    run_route(prepare(), prepare(), route)
    check_original(base, original)
