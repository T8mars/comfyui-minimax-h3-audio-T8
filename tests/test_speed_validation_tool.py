from __future__ import annotations

from h3_audio_t8_pkg.tools.build_h3_speed_validation_pair import build_t2va_pair


def _build():
    return build_t2va_pair(
        width=1056,
        height=608,
        length=124,
        steps=20,
        seed=2608184001,
        prompt="controlled prompt",
        scales="0.5,1.0",
        transition_sigma="0.85",
        shift_video=12.0,
        shift_audio=3.0,
        model_name="model.safetensors",
        clip_name="clip.safetensors",
        video_vae_name="video.safetensors",
        audio_vae_name="audio.safetensors",
        filename_prefix="MiniMaxH3/test",
    )


def test_speed_validation_pair_freezes_all_control_variables():
    baseline, speed, manifest = _build()
    controlled = manifest["controlled"]
    assert controlled["width"] == 1056
    assert controlled["height"] == 608
    assert controlled["length"] == 124
    assert controlled["steps"] == 20
    assert controlled["seed"] == 2608184001
    assert baseline["1"]["inputs"] == speed["1"]["inputs"]
    assert baseline["2"]["inputs"] == speed["2"]["inputs"]
    assert baseline["3"]["inputs"] == speed["3"]["inputs"]
    assert baseline["4"]["inputs"] == speed["4"]["inputs"]
    assert baseline["5"]["inputs"]["prompt"] == speed["6"]["inputs"]["prompt"]
    assert baseline["8"]["class_type"] == "MiniMaxH3SPEEDModalityStableNoiseT8Advanced"
    assert baseline["8"]["inputs"]["seed"] == speed["7"]["inputs"]["seed"]
    assert baseline["6"]["inputs"]["steps"] == speed["5"]["inputs"]["steps"]
    assert speed["7"]["inputs"]["execution_scope"] == "strict_t2va_stock20"
    assert manifest["treatment"]["total_nfe_unchanged"] is True
    assert controlled["noise_contract"] == "modality_stable_nested_av_v1"


def test_speed_validation_pair_uses_real_output_nodes_and_distinct_prefixes():
    baseline, speed, _ = _build()
    for prompt in (baseline, speed):
        assert prompt["10"]["class_type"] == "MiniMaxH3AVDecodeT8"
        assert prompt["11"]["class_type"] == "CreateVideo"
        assert prompt["12"]["class_type"] == "SaveVideo"
        assert prompt["13"]["class_type"] == "SaveText"
    baseline_prefix = baseline["12"]["inputs"]["filename_prefix"]
    speed_prefix = speed["12"]["inputs"]["filename_prefix"]
    assert baseline_prefix.endswith("baseline_stock20")
    assert speed_prefix.endswith("speed_stock20")
    assert baseline_prefix != speed_prefix
    assert baseline["13"]["inputs"]["text"] == ["5", 5]
    assert speed["13"]["inputs"]["text"] == ["7", 4]
