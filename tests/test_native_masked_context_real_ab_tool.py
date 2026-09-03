from __future__ import annotations

from copy import deepcopy

from tools.run_native_masked_context_ab import (
    CLASSICAL_MANDARIN_SPEECH_PROMPT,
    CLASSICAL_CONTINUATION_MUSIC_PROMPT,
    CORRECTED_FL2V_ALPHA8_LORA,
    CONTINUATION_SEED,
    INSTRUMENTAL_MUSIC_PROMPT,
    LEGACY_GENERIC_EMA_LORA,
    NEW_EMA_B_LORA,
    RENDER_FRAMES,
    ROUTES,
    _result_success,
    build_prompt,
)


def _prompt(route: str):
    return build_prompt(route, chain_id="same_chain", run_id="fixed_run")


def test_segment_zero_creates_the_only_context_checkpoint():
    prompt = _prompt("segment0")
    assert prompt["6"]["inputs"]["segment_index"] == 0
    assert prompt["6"]["inputs"]["is_final_segment"] is False
    assert prompt["6"]["inputs"]["minimum_render_frames"] == RENDER_FRAMES
    assert prompt["8"]["inputs"]["context_audio"] == "video_only"
    assert prompt["13"]["class_type"] == "MiniMaxH3LongVideoContextSaveT8"
    assert prompt["19"]["class_type"] == "MiniMaxH3LongVideoColorMatchT8Advanced"
    assert prompt["19"]["inputs"]["enabled"] is True
    assert prompt["16"]["inputs"]["images"] == ["19", 0]
    assert "18" not in prompt


def test_continuation_pair_has_one_intended_topology_difference():
    soft = _prompt("soft_context")
    hard = _prompt("hard_mask_plan_b")
    assert "13" not in soft and "13" not in hard
    assert "18" not in soft
    assert hard["18"] == {
        "class_type": "MiniMaxH3NativeMaskedVideoContextT8Advanced",
        "inputs": {
            "av_latent": ["8", 2],
            "context": ["7", 0],
            "planner_report_json": ["6", 9],
            "conditioning_report_json": ["8", 6],
        },
    }
    assert soft["9"]["inputs"]["av_latent"] == ["8", 2]
    assert soft["12"]["inputs"]["latent_image"] == ["8", 2]
    assert hard["9"]["inputs"]["av_latent"] == ["18", 0]
    assert hard["12"]["inputs"]["latent_image"] == ["18", 0]
    for prompt in (soft, hard):
        assert prompt["19"] == {
            "class_type": "MiniMaxH3LongVideoColorMatchT8Advanced",
            "inputs": {
                "frames": ["15", 0],
                "context": ["7", 0],
                "chain_id": ["6", 0],
                "segment_index": ["6", 1],
                "enabled": True,
                "reference_frames": 5,
                "transition_frames": 24,
                "strength": 1.0,
                "minimum_jump": 0.0005,
                "maximum_offset": 0.02,
                "scene_cut_threshold": 0.18,
            },
        }
        assert prompt["16"]["inputs"]["images"] == ["19", 0]

    normalized_hard = deepcopy(hard)
    normalized_hard.pop("18")
    normalized_hard["9"]["inputs"]["av_latent"] = ["8", 2]
    normalized_hard["12"]["inputs"]["latent_image"] = ["8", 2]
    normalized_hard["17"]["inputs"]["filename_prefix"] = soft["17"]["inputs"][
        "filename_prefix"
    ]
    assert normalized_hard == soft


def test_pair_uses_exact_same_continuation_contract_and_never_injects_old_audio():
    soft = _prompt("soft_context")
    hard = _prompt("hard_mask_plan_b")
    for prompt in (soft, hard):
        assert prompt["6"]["inputs"]["segment_index"] == 1
        assert prompt["6"]["inputs"]["is_final_segment"] is True
        assert prompt["8"]["inputs"]["context_audio"] == "video_only"
        assert prompt["10"]["inputs"]["noise_seed"] == CONTINUATION_SEED
        assert prompt["9"]["inputs"]["steps"] == 4
        assert prompt["9"]["inputs"]["shift_video"] == 12.0
        assert prompt["9"]["inputs"]["shift_audio"] == 3.0
        assert prompt["9"]["inputs"]["sampler_name"] == "euler"
        assert prompt["9"]["inputs"]["scheduler"] == "native_flow"
    assert soft["8"]["inputs"] == hard["8"]["inputs"]
    assert soft["10"]["inputs"] == hard["10"]["inputs"]


def test_runner_route_order_is_segment_then_soft_then_plan_b():
    assert ROUTES == ("segment0", "soft_context", "hard_mask_plan_b")


def test_segment_zero_checkpoint_records_native_av_sampler_contract():
    prompt = _prompt("segment0")
    assert prompt["13"]["inputs"]["sampling_summary"] == (
        "4-step euler/native_flow ComfyUI ModelSamplingAV shift12/3"
    )


def test_diagnostic_resolution_can_be_lowered_without_changing_audio_contract():
    prompt = build_prompt(
        "segment0",
        chain_id="same_chain",
        run_id="fixed_run",
        audio_profile="instrumental_music",
        width=416,
        height=224,
    )
    assert prompt["8"]["inputs"]["width"] == 416
    assert prompt["8"]["inputs"]["height"] == 224
    assert prompt["9"]["inputs"]["sampler_name"] == "euler"


def test_same_prompt_can_run_an_eight_nfe_audio_diagnostic():
    prompt = build_prompt(
        "segment0",
        chain_id="same_chain",
        run_id="fixed_run",
        audio_profile="classical_mandarin_speech",
        width=416,
        height=224,
        steps=8,
    )
    assert prompt["9"]["inputs"]["steps"] == 8
    assert prompt["9"]["inputs"]["sampler_name"] == "euler"
    assert prompt["13"]["inputs"]["sampling_summary"] == (
        "8-step euler/native_flow ComfyUI ModelSamplingAV shift12/3"
    )


def test_corrected_fl2v_lora_is_a_single_variable_segment_zero_diagnostic():
    baseline = build_prompt(
        "segment0",
        chain_id="same_chain",
        run_id="fixed_run",
        audio_profile="classical_mandarin_speech",
        width=416,
        height=224,
        steps=4,
    )
    corrected = build_prompt(
        "segment0",
        chain_id="same_chain",
        run_id="fixed_run",
        audio_profile="classical_mandarin_speech",
        width=416,
        height=224,
        steps=4,
        turbo_lora=CORRECTED_FL2V_ALPHA8_LORA,
    )
    assert baseline["5"]["inputs"]["lora_name"] != corrected["5"]["inputs"][
        "lora_name"
    ]
    assert corrected["5"]["inputs"]["lora_name"] == CORRECTED_FL2V_ALPHA8_LORA
    assert CORRECTED_FL2V_ALPHA8_LORA in corrected["13"]["inputs"]["model_id"]

    normalized = deepcopy(corrected)
    normalized["5"]["inputs"]["lora_name"] = baseline["5"]["inputs"]["lora_name"]
    normalized["13"]["inputs"]["model_id"] = baseline["13"]["inputs"]["model_id"]
    assert normalized == baseline


def test_new_step600_ema_b_is_default_and_legacy_ema_requires_explicit_selection():
    current = build_prompt(
        "segment0",
        chain_id="same_chain",
        run_id="fixed_run",
        audio_profile="classical_mandarin_speech",
        width=416,
        height=224,
        steps=4,
    )
    legacy = build_prompt(
        "segment0",
        chain_id="same_chain",
        run_id="fixed_run",
        audio_profile="classical_mandarin_speech",
        width=416,
        height=224,
        steps=4,
        turbo_lora=LEGACY_GENERIC_EMA_LORA,
    )
    assert current["5"]["inputs"]["lora_name"] == NEW_EMA_B_LORA
    assert NEW_EMA_B_LORA in current["13"]["inputs"]["model_id"]
    assert legacy["5"]["inputs"]["lora_name"] == LEGACY_GENERIC_EMA_LORA
    assert current["5"]["inputs"]["lora_name"] != legacy["5"]["inputs"]["lora_name"]


def test_instrumental_music_profile_is_identical_across_pair_and_excludes_ambience():
    soft = build_prompt(
        "soft_context",
        chain_id="same_chain",
        run_id="fixed_run",
        audio_profile="instrumental_music",
    )
    hard = build_prompt(
        "hard_mask_plan_b",
        chain_id="same_chain",
        run_id="fixed_run",
        audio_profile="instrumental_music",
    )
    assert soft["8"]["inputs"]["prompt"] == INSTRUMENTAL_MUSIC_PROMPT
    assert hard["8"]["inputs"]["prompt"] == INSTRUMENTAL_MUSIC_PROMPT
    music_prompt = INSTRUMENTAL_MUSIC_PROMPT.lower()
    assert "instrumental synthwave background music" in music_prompt
    assert "steady 96 bpm" in music_prompt
    assert "no rain sound" in music_prompt
    assert "no footsteps" in music_prompt
    assert "no hiss" in music_prompt
    assert "no clipping" in music_prompt
    assert soft["8"]["inputs"] == hard["8"]["inputs"]


def test_classical_mandarin_speech_profile_has_one_exact_dialogue_block():
    prompt = build_prompt(
        "segment0",
        chain_id="same_chain",
        run_id="fixed_run",
        audio_profile="classical_mandarin_speech",
        width=416,
        height=224,
    )
    text = prompt["8"]["inputs"]["prompt"]
    assert text == CLASSICAL_MANDARIN_SPEECH_PROMPT
    assert text.count("<d>") == 1
    assert text.count("</d>") == 1
    assert "<d>[Chinese] 你在哪里</d>" in text
    assert "classical chamber music" in text
    assert "solo cello" in text
    assert "acoustic piano" in text
    assert "No additional words" in text
    assert prompt["9"]["inputs"]["sampler_name"] == "euler"


def test_classical_full_pair_requests_dialogue_once_then_music_only_continuation():
    segment_zero = build_prompt(
        "segment0",
        chain_id="same_chain",
        run_id="fixed_run",
        audio_profile="classical_mandarin_speech",
        width=416,
        height=224,
    )
    soft = build_prompt(
        "soft_context",
        chain_id="same_chain",
        run_id="fixed_run",
        audio_profile="classical_mandarin_speech",
        width=416,
        height=224,
    )
    hard = build_prompt(
        "hard_mask_plan_b",
        chain_id="same_chain",
        run_id="fixed_run",
        audio_profile="classical_mandarin_speech",
        width=416,
        height=224,
    )
    assert segment_zero["8"]["inputs"]["prompt"] == CLASSICAL_MANDARIN_SPEECH_PROMPT
    assert soft["8"]["inputs"]["prompt"] == CLASSICAL_CONTINUATION_MUSIC_PROMPT
    assert hard["8"]["inputs"]["prompt"] == CLASSICAL_CONTINUATION_MUSIC_PROMPT
    assert "<d>" not in CLASSICAL_CONTINUATION_MUSIC_PROMPT
    assert "remains silent throughout this continuation" in CLASSICAL_CONTINUATION_MUSIC_PROMPT
    assert "music from the previous segment is already in progress" in (
        CLASSICAL_CONTINUATION_MUSIC_PROMPT
    )
    assert "no dialogue, vocals" in CLASSICAL_CONTINUATION_MUSIC_PROMPT
    assert soft["8"]["inputs"] == hard["8"]["inputs"]


def test_execution_success_never_hides_a_strict_media_decode_failure():
    phase = {"terminal": {"type": "execution_success"}}
    assert _result_success(phase, {"strict_decode_passed": True}) is True
    assert _result_success(phase, {"strict_decode_passed": False}) is False
    assert _result_success(phase, None) is False
