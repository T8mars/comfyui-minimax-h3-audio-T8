from __future__ import annotations

import numpy as np

from tools.analyze_native_masked_context_ab import (
    AUDIO_RATE,
    _blind_html,
    _resource_claim,
    _select_face_near_reference,
    _ssim,
    audio_quality_metrics,
    audio_seam_metrics,
    blind_mapping,
    seam_video_metrics,
)


def test_resource_claim_matches_the_512_mib_gate_result():
    passed = _resource_claim(True)
    failed = _resource_claim(False)
    assert "kept at least" in passed
    assert "does not establish general 16GB safety" in passed
    assert "fell below" in failed
    assert "do not claim general 16GB safety" in failed


def test_ssim_is_one_for_identical_frames():
    frame = np.full((64, 96, 3), 127, dtype=np.uint8)
    assert _ssim(frame, frame) == 1.0


def test_seam_video_metrics_are_finite_and_descriptive():
    frames = []
    for index in range(6):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[:, index : index + 20] = 255
        frames.append(frame)
    report = seam_video_metrics(frames[:3], frames[3:])
    assert 0.0 <= report["boundary_frame_mae_0_to_1"] <= 1.0
    assert np.isfinite(report["flow_vector_jump_from_before"])
    assert "do not select a winner" in report["interpretation_boundary"]


def test_audio_jump_proxy_detects_a_larger_boundary_step():
    length = AUDIO_RATE
    continuous_left = np.zeros((length, 2), dtype=np.float32)
    continuous_right = np.zeros((length, 2), dtype=np.float32)
    jumped_right = np.ones((length, 2), dtype=np.float32) * 0.5
    continuous = audio_seam_metrics(continuous_left, continuous_right)
    jumped = audio_seam_metrics(continuous_left, jumped_right)
    assert jumped["boundary_sample_step_peak"] > continuous["boundary_sample_step_peak"]
    assert jumped["head_50ms_rms"] > continuous["head_50ms_rms"]


def test_blind_mapping_is_deterministic_and_complete():
    first = blind_mapping("a" * 64, "b" * 64)
    second = blind_mapping("a" * 64, "b" * 64)
    assert first == second
    assert set(first) == {"A", "B"}
    assert set(first.values()) == {"soft_context", "hard_mask_plan_b"}


def test_audio_quality_metrics_report_clipping_and_noise_proxies_without_ranking():
    samples = np.linspace(-0.5, 0.5, AUDIO_RATE, dtype=np.float32)
    stereo = np.stack((samples, samples), axis=1)
    report = audio_quality_metrics(stereo)
    assert report["peak_abs"] == 0.5
    assert report["clipping_sample_fraction_at_0p999"] == 0.0
    assert 0.0 <= report["high_band_10k_to_16k_energy_ratio"] <= 1.0
    assert 0.0 <= report["mean_spectral_flatness"] <= 1.0
    assert report["finite"] is True
    assert "human listening remains required" in report["interpretation_boundary"]


def test_music_blind_page_discloses_raw_native_soundtrack_without_route_mapping(tmp_path):
    page = tmp_path / "review.html"
    _blind_html(page, 123, audio_profile="instrumental_music")
    html = page.read_text(encoding="utf-8")
    assert "纯器乐合成器背景音乐" in html
    assert "没有后期降噪" in html
    assert "soft_context" not in html
    assert "hard_mask_plan_b" not in html


def test_classical_speech_blind_page_uses_the_actual_scene_contract(tmp_path):
    page = tmp_path / "review.html"
    _blind_html(page, 123, audio_profile="classical_mandarin_speech")
    html = page.read_text(encoding="utf-8")
    assert "你在哪里" in html
    assert "续段要求人物静默" in html
    assert "人物面部、肤色、灯光和背景" in html
    assert "雨伞" not in html
    assert "soft_context" not in html
    assert "hard_mask_plan_b" not in html


def test_color_match_blind_page_explains_shared_default_without_route_mapping(tmp_path):
    page = tmp_path / "review.html"
    _blind_html(
        page,
        123,
        audio_profile="classical_mandarin_speech",
        color_match_enabled=True,
    )
    html = page.read_text(encoding="utf-8")
    assert "A/B 两条都启用同一个默认 Color Match" in html
    assert "全局 Lab 色彩/对比度" in html
    assert "8x5局部分区" in html
    assert "总改变量最大0.02" in html
    assert "24帧内渐隐" in html
    assert "Color Match不改音频或原生latent" in html
    assert "音频接缝淡化" in html
    assert "soft_context" not in html
    assert "hard_mask_plan_b" not in html


def test_face_selection_tracks_the_previous_face_instead_of_largest_false_detection():
    reference = np.array([169, 33, 80, 112, *([0] * 10), 0.95], dtype=np.float32)
    false_large = np.array([-4, 64, 106, 158, *([0] * 10), 0.99], dtype=np.float32)
    true_continuation = np.array([168, 34, 81, 114, *([0] * 10), 0.91], dtype=np.float32)
    selected = _select_face_near_reference(
        [false_large, true_continuation], reference
    )
    assert selected is true_continuation
    assert _select_face_near_reference([false_large, true_continuation], None) is false_large
    assert _select_face_near_reference([], reference) is None
