from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import pytest
import torch

import comfy.nested_tensor
from comfy.ldm.minimax.model import MiniMaxH3Model

from h3_audio_t8_pkg.nodes_speed_advanced import SPEED_ADVANCED_NODE_CLASSES
from h3_audio_t8_pkg.speed_advanced import (
    SPEED_PLAN_SCHEMA,
    SPEED_PROFILE_SCHEMA,
    SPEED_SOURCE_SCHEMA,
    SPEED_SPECTRUM_DATASET_SCHEMA,
    SPEED_DATASET_PROVENANCE_SCHEMA,
    SPEED_SOURCE_ENTRY_SCHEMA,
    StageConditioning,
    _build_stage,
    _build_empty_t2va_stage,
    _profile_binding,
    _source_set_sha256,
    _apply_speed_scoped_headroom,
    _release_h3_residency_between_stages,
    _restore_speed_scoped_headroom,
    _task_support,
    _weight_patch_contract,
    _ensure_native_h3_model,
    activation_time,
    accumulate_spectrum_dataset,
    align_sigma,
    build_spectrum_profile,
    build_speed_plan,
    build_speed_source,
    dct_expand_official,
    fit_h3_spatial_power_spectrum,
    finalize_spectrum_dataset,
    kappa,
    modality_stable_h3_noise,
    power_spectrum,
    prepare_speed_calibration_window,
    recover_raw_flow_state,
    reindex_joint_audio_state,
    resolve_stage_shapes,
    solve_segment_noise,
)
from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE, make_audio


def test_official_speed_equations_match_reference_formulas():
    power = power_spectrum(12.0, 219.484718, 2.422687)
    expected_activation = 1.0 / (
        1.0 + math.sqrt(0.01 / (power * (1.0 + power - 0.01)))
    )
    expected_kappa = 2.0 / (1.0 + 0.75)
    assert activation_time(power, 0.01) == pytest.approx(expected_activation)
    assert kappa(0.75, 2.0) == pytest.approx(expected_kappa)
    assert align_sigma(0.75, 2.0) == pytest.approx(0.75 * expected_kappa)


def test_stage_shapes_snap_to_32_without_aspect_distortion():
    stages = resolve_stage_shapes(1056, 608, [0.5, 1.0])
    assert [(stage["width"], stage["height"]) for stage in stages] == [
        (544, 320),
        (1056, 608),
    ]
    assert stages[0]["latent_width"] == 34
    assert stages[0]["latent_height"] == 20
    assert stages[-1]["latent_width"] % 2 == 0
    assert stages[-1]["latent_height"] % 2 == 0
    assert stages[0]["snap_anisotropy"] < 0.05


def test_manual_plan_preserves_exact_nfe_and_aligns_each_transition():
    plan, report_json = build_speed_plan(
        width=1056,
        height=608,
        steps=20,
        scales="0.5,1.0",
        transition_mode="manual_sigmas",
        manual_transition_sigmas="0.85",
        delta=0.01,
        shift_video=12.0,
        transform="dct",
        profile_policy="require_validated_profile",
        fallback_policy="error",
    )
    assert plan["schema"] == SPEED_PLAN_SCHEMA
    assert plan["nfe"] == 20
    assert sum(segment["nfe"] for segment in plan["segments"]) == 20
    transition = plan["transitions"][0]
    assert transition["ratio"] == pytest.approx(2.0)
    assert transition["actual_grid_ratio"] == pytest.approx(
        math.sqrt((66 / 34) * (38 / 20))
    )
    assert transition["actual_grid_ratio"] != pytest.approx(transition["ratio"])
    assert transition["aligned_sigma"] > transition["sigma"]
    assert transition["aligned_sigma"] == pytest.approx(
        align_sigma(transition["sigma"], transition["ratio"])
    )
    assert json.loads(report_json)["official_method"]["wan_constants_reused"] is False


def test_three_stage_plan_has_strict_segments_and_same_total_nfe():
    plan, _ = build_speed_plan(
        width=1344,
        height=768,
        steps=20,
        scales="0.4,0.7,1.0",
        transition_mode="manual_sigmas",
        manual_transition_sigmas="0.94,0.78",
        delta=0.01,
        shift_video=12.0,
        transform="dct",
        profile_policy="require_validated_profile",
        fallback_policy="error",
    )
    assert len(plan["stages"]) == 3
    assert len(plan["transitions"]) == 2
    assert len(plan["segments"]) == 3
    assert plan["nfe"] == 20
    assert all(segment["nfe"] > 0 for segment in plan["segments"])
    assert plan["transitions"][0]["step_index"] < plan["transitions"][1]["step_index"]


def test_delta_optimal_rejects_single_clip_probe_unless_explicitly_allowed():
    profile = {
        "schema": SPEED_PROFILE_SCHEMA,
        "profile_name": "probe",
        "status": "research_probe_only",
        "validated_for_delta_optimal": False,
        "checkpoint_fingerprint": "sha256:test",
        "vae_fingerprint": "sha256:test-vae",
        "fit": {
            "amplitude": 200.0,
            "beta": 2.0,
            "r_squared": 0.95,
            "latent_shape": [1, 24, 37, 38, 66],
        },
    }
    kwargs = dict(
        width=1056,
        height=608,
        steps=20,
        scales="0.5,1.0",
        transition_mode="delta_optimal",
        manual_transition_sigmas="0.85",
        delta=0.01,
        shift_video=12.0,
        transform="dct",
        fallback_policy="error",
        spectrum_profile=profile,
    )
    with pytest.raises(ValueError, match="research probe"):
        build_speed_plan(profile_policy="require_validated_profile", **kwargs)
    plan, _ = build_speed_plan(profile_policy="allow_research_profile", **kwargs)
    assert plan["profile"]["amplitude"] == 200.0
    assert plan["profile"]["latent_contract"] == {
        "channels": 24,
        "frames": 37,
        "height": 38,
        "width": 66,
    }
    wrong_grid = {
        **profile,
        "fit": {**profile["fit"], "latent_shape": [1, 24, 37, 36, 64]},
    }
    with pytest.raises(ValueError, match="latent grid mismatch"):
        build_speed_plan(
            profile_policy="allow_research_profile",
            **{**kwargs, "spectrum_profile": wrong_grid},
        )


def test_delta_profile_binding_rejects_task_and_fingerprint_mismatches():
    plan = {
        "transition_mode": "delta_optimal",
        "profile_policy": "require_validated_profile",
        "width": 736,
        "height": 416,
        "profile": {
            "task_family": "T2VA",
            "checkpoint_fingerprint": "sha256:model-a",
            "vae_fingerprint": "sha256:vae-a",
            "latent_contract": {
                "channels": 24,
                "frames": 37,
                "height": 26,
                "width": 46,
            },
        },
    }
    source = {
        "resolved_task": "t2va",
        "checkpoint_fingerprint": "sha256:model-a",
        "vae_fingerprint": "sha256:vae-a",
        "length": 124,
    }
    assert _profile_binding(plan, source)["status"] == "matched"
    # Real SHA-256 digests are hexadecimal; exercise case normalization with valid values.
    case_plan = {
        **plan,
        "profile": {
            **plan["profile"],
            "checkpoint_fingerprint": "sha256:" + "a" * 64,
            "vae_fingerprint": "sha256:" + "b" * 64,
        },
    }
    case_source = {
        **source,
        "checkpoint_fingerprint": "sha256:" + "A" * 64,
        "vae_fingerprint": "sha256:" + "B" * 64,
    }
    assert _profile_binding(case_plan, case_source)["status"] == "matched"
    with pytest.raises(ValueError, match="task mismatch"):
        _profile_binding(plan, {**source, "resolved_task": "ref2va"})
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _profile_binding(plan, {**source, "vae_fingerprint": "sha256:vae-b"})
    with pytest.raises(ValueError, match="requires runtime source fingerprints"):
        _profile_binding(plan, {**source, "checkpoint_fingerprint": "unrecorded"})
    with pytest.raises(ValueError, match="latent grid mismatch"):
        _profile_binding(plan, {**source, "length": 141})
    with pytest.raises(ValueError, match="latent grid mismatch"):
        _profile_binding({**plan, "width": 768}, source)


def test_spectrum_fit_recovers_a_positive_power_law_and_marks_probe_status():
    torch.manual_seed(7)
    latent = torch.randn(1, 24, 8, 16, 24)
    flattened = latent.reshape(-1, 1, 16, 24)
    for _ in range(3):
        flattened = torch.nn.functional.avg_pool2d(
            flattened, kernel_size=3, stride=1, padding=1
        )
    latent = flattened.reshape(1, 24, 8, 16, 24)
    fit = fit_h3_spatial_power_spectrum(latent, max_temporal_samples=4)
    assert fit["amplitude"] > 0.0
    assert fit["beta"] > 0.0
    assert math.isfinite(fit["r_squared"])
    profile, report_json = build_spectrum_profile(
        latent,
        profile_name="unit_probe",
        task_family="T2VA",
        checkpoint_fingerprint="sha256:model",
        vae_fingerprint="sha256:vae",
        independent_clip_count=1,
        minimum_r_squared=0.0,
        max_temporal_samples=4,
    )
    assert profile["validated_for_delta_optimal"] is False
    assert profile["status"] == "research_probe_only"
    assert json.loads(report_json)["schema"] == SPEED_PROFILE_SCHEMA


def test_declared_spectrum_evidence_cannot_promote_one_actual_clip():
    torch.manual_seed(17)
    latent = torch.randn(1, 24, 4, 16, 24)
    flattened = latent.reshape(-1, 1, 16, 24)
    for _ in range(3):
        flattened = torch.nn.functional.avg_pool2d(
            flattened, kernel_size=3, stride=1, padding=1
        )
    latent = flattened.reshape(1, 24, 4, 16, 24)
    profile, _ = build_spectrum_profile(
        latent,
        profile_name="false_dataset_claim",
        task_family="T2VA",
        checkpoint_fingerprint="sha256:model",
        vae_fingerprint="sha256:vae",
        independent_clip_count=100,
        minimum_r_squared=0.0,
        max_temporal_samples=4,
    )
    assert profile["actual_batch_entries"] == 1
    assert profile["declared_evidence_present_in_input"] is False
    assert profile["provenance_complete"] is True
    assert profile["validated_for_delta_optimal"] is False


def _smoothed_spectrum_latent(seed: int, batch: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(batch, 24, 4, 16, 24, generator=generator)
    flattened = latent.reshape(-1, 1, 16, 24)
    for _ in range(3):
        flattened = torch.nn.functional.avg_pool2d(
            flattened, kernel_size=3, stride=1, padding=1
        )
    return flattened.reshape(batch, 24, 4, 16, 24)


def test_spectrum_dataset_two_batches_match_one_direct_fit_without_gpu_retention():
    first = _smoothed_spectrum_latent(31, 2)
    second = _smoothed_spectrum_latent(37, 3)
    dataset, first_report = accumulate_spectrum_dataset(
        first,
        batch_id="batch-a",
        task_family="T2VA",
        checkpoint_fingerprint="sha256:model-a",
        vae_fingerprint="sha256:vae-a",
        max_temporal_samples=4,
    )
    dataset, second_report = accumulate_spectrum_dataset(
        second,
        previous_dataset=dataset,
        batch_id="batch-b",
        task_family="T2VA",
        checkpoint_fingerprint="sha256:model-a",
        vae_fingerprint="sha256:vae-a",
        max_temporal_samples=4,
    )
    profile, profile_report = finalize_spectrum_dataset(
        dataset,
        profile_name="five_clip_probe",
        minimum_r_squared=0.0,
        minimum_independent_clips=100,
    )
    direct = fit_h3_spatial_power_spectrum(
        torch.cat((first, second), dim=0), max_temporal_samples=4
    )
    assert dataset["schema"] == SPEED_SPECTRUM_DATASET_SCHEMA
    assert dataset["power_sum"].device.type == "cpu"
    assert dataset["power_sum"].dtype == torch.float64
    assert dataset["power_sum"].data_ptr() != first.data_ptr()
    assert dataset["independent_clip_count"] == 5
    assert dataset["batch_ids"] == ["batch-a", "batch-b"]
    assert profile["fit"]["amplitude"] == pytest.approx(direct["amplitude"])
    assert profile["fit"]["beta"] == pytest.approx(direct["beta"])
    assert profile["fit"]["r_squared"] == pytest.approx(direct["r_squared"])
    assert profile["status"] == "research_probe_only"
    assert profile["validated_for_delta_optimal"] is False
    assert "power_sum" not in json.loads(first_report)
    assert "power_sum" not in json.loads(second_report)
    assert json.loads(profile_report)["dataset"]["batch_count"] == 2


def test_spectrum_dataset_rejects_duplicate_and_mismatched_inputs():
    latent = _smoothed_spectrum_latent(41, 1)
    dataset, _ = accumulate_spectrum_dataset(
        latent,
        batch_id="batch-a",
        task_family="FL2VA",
        checkpoint_fingerprint="sha256:model-a",
        vae_fingerprint="sha256:vae-a",
        max_temporal_samples=4,
    )
    kwargs = {
        "previous_dataset": dataset,
        "task_family": "FL2VA",
        "checkpoint_fingerprint": "sha256:model-a",
        "vae_fingerprint": "sha256:vae-a",
        "max_temporal_samples": 4,
    }
    with pytest.raises(ValueError, match="Duplicate.*batch_id"):
        accumulate_spectrum_dataset(latent, batch_id="batch-a", **kwargs)
    with pytest.raises(ValueError, match="repeats a clip spectrum"):
        accumulate_spectrum_dataset(latent, batch_id="batch-b", **kwargs)
    with pytest.raises(ValueError, match="checkpoint_fingerprint mismatch"):
        accumulate_spectrum_dataset(
            _smoothed_spectrum_latent(43, 1),
            batch_id="batch-c",
            **{**kwargs, "checkpoint_fingerprint": "sha256:model-b"},
        )
    with pytest.raises(ValueError, match="latent/settings contract mismatch"):
        accumulate_spectrum_dataset(
            _smoothed_spectrum_latent(47, 1)[:, :, :, :, :16],
            batch_id="batch-d",
            **kwargs,
        )

    corrupted = dict(dataset)
    corrupted["batch_sizes"] = [2]
    with pytest.raises(ValueError, match="do not sum to clip count"):
        finalize_spectrum_dataset(
            corrupted,
            profile_name="corrupted",
            minimum_r_squared=0.0,
        )


def test_spectrum_dataset_count_without_reviewed_source_binding_stays_research():
    latent = _smoothed_spectrum_latent(53, 100)
    dataset, _ = accumulate_spectrum_dataset(
        latent,
        batch_id="dataset-100",
        task_family="Ref2VA",
        checkpoint_fingerprint="sha256:model-ref",
        vae_fingerprint="sha256:vae-ref",
        max_temporal_samples=4,
    )
    profile, _ = finalize_spectrum_dataset(
        dataset,
        profile_name="ref2va-100",
        minimum_r_squared=0.0,
        minimum_independent_clips=100,
    )
    assert dataset["independent_clip_count"] == 100
    assert len(dataset["clip_fingerprints"]) == 100
    assert profile["status"] == "research_probe_only"
    assert profile["validated_for_delta_optimal"] is False
    assert profile["validation_checks"]["provenance_complete"] is False
    assert profile["dataset"]["batch_count"] == 1
    with pytest.raises(ValueError, match="cannot be lower than 100"):
        finalize_spectrum_dataset(
            dataset,
            profile_name="unsafe",
            minimum_r_squared=0.0,
            minimum_independent_clips=99,
        )


def test_spectrum_dataset_requires_exact_reviewed_source_set_for_validation():
    source_entries = []
    for index in range(100):
        source_entries.append(
            {
                "schema": SPEED_SOURCE_ENTRY_SCHEMA,
                "batch_id": f"natural-{index:03d}",
                "source_file_sha256": hashlib.sha256(
                    f"source-{index}".encode()
                ).hexdigest().upper(),
                "decoded_window_sha256": hashlib.sha256(
                    f"decoded-{index}".encode()
                ).hexdigest().upper(),
            }
        )
    provenance = {
        "schema": SPEED_DATASET_PROVENANCE_SCHEMA,
        "source_kind": "independent_natural_video_corpus",
        "dataset_id": "example/natural-videos",
        "dataset_revision": "fixed-revision",
        "dataset_license": "apache-2.0",
        "source_shards": [
            {
                "shard": "00000/000000.tar",
                "lfs_oid": "A" * 64,
                "fetch_report_sha256": "B" * 64,
            }
        ],
        "curation_report_sha256": "C" * 64,
        "review_report_sha256": "E" * 64,
        "selection_policy": "sha256_rank",
        "selected_source_count": 100,
        "selected_source_set_sha256": _source_set_sha256(source_entries),
        "independence_reviewed": True,
        "content_diversity_reviewed": True,
        "raw_media_redistributed": False,
    }
    dataset = None
    for index, source_entry in enumerate(source_entries):
        dataset, _ = accumulate_spectrum_dataset(
            _smoothed_spectrum_latent(1000 + index, 1),
            previous_dataset=dataset,
            batch_id=source_entry["batch_id"],
            task_family="T2VA",
            checkpoint_fingerprint="sha256:model",
            vae_fingerprint="sha256:vae",
            max_temporal_samples=4,
            dataset_provenance_json=json.dumps(provenance),
            source_entry_json=json.dumps(source_entry),
        )
    profile, _ = finalize_spectrum_dataset(
        dataset,
        profile_name="reviewed-natural-100",
        minimum_r_squared=0.0,
        minimum_independent_clips=100,
    )
    assert profile["validated_for_delta_optimal"] is True
    assert profile["provenance_complete"] is True
    assert profile["validation_checks"]["selected_source_set_matches"] is True
    assert profile["dataset"]["dataset_provenance"]["dataset_revision"] == "fixed-revision"

    tampered = dict(dataset)
    tampered["dataset_provenance"] = {
        **dataset["dataset_provenance"],
        "selected_source_set_sha256": "D" * 64,
    }
    denied, _ = finalize_spectrum_dataset(
        tampered,
        profile_name="tampered",
        minimum_r_squared=0.0,
        minimum_independent_clips=100,
    )
    assert denied["validated_for_delta_optimal"] is False
    assert denied["validation_checks"]["selected_source_set_matches"] is False


def test_strict_scope_is_exactly_t2va_native_stock20():
    source = {"resolved_task": "t2va", "audio_mode": "native"}
    assert _task_support(source, "strict_t2va_stock20", 20, 12.0, 3.0)[0] is True
    assert _task_support(source, "strict_t2va_stock20", 8, 12.0, 3.0)[0] is False
    assert _task_support(source, "strict_t2va_stock20", 20, 11.0, 3.0)[0] is False
    assert _task_support(source, "strict_t2va_stock20", 20, 12.0, 4.0)[0] is False
    assert _task_support(
        {**source, "resolved_task": "fl2va"}, "strict_t2va_stock20", 20, 12.0, 3.0
    )[0] is False
    assert _task_support(
        {**source, "audio_mode": "lock_source"}, "strict_t2va_stock20", 20, 12.0, 3.0
    )[0] is False
    assert _task_support(
        {**source, "first_frame": torch.zeros(1, 32, 32, 3)},
        "strict_t2va_stock20",
        20,
        12.0,
        3.0,
    )[0] is False
    assert _task_support(
        {**source, "ref_images": {"ref_image_0": torch.zeros(1, 32, 32, 3)}},
        "strict_t2va_stock20",
        20,
        12.0,
        3.0,
    )[0] is False


def test_turbo8_scope_is_exactly_media_free_t2va_native_8step():
    source = {"resolved_task": "t2va", "audio_mode": "native"}
    assert _task_support(source, "turbo8_t2va_research_exp", 8, 12.0, 3.0)[0]
    assert not _task_support(source, "turbo8_t2va_research_exp", 20, 12.0, 3.0)[0]
    assert not _task_support(
        {**source, "first_frame": torch.zeros(1, 32, 32, 3)},
        "turbo8_t2va_research_exp",
        8,
        12.0,
        3.0,
    )[0]


def test_weight_patch_contract_distinguishes_stock_and_turbo_scopes():
    class FakeModel:
        def __init__(self, patches):
            self.patches = patches

    stock = FakeModel({})
    patched = FakeModel({"diffusion_model.test": [(1.0, object())]})
    assert _weight_patch_contract(stock, "strict_t2va_stock20")[0]
    assert not _weight_patch_contract(patched, "strict_t2va_stock20")[0]
    assert not _weight_patch_contract(stock, "turbo8_t2va_research_exp")[0]
    supported, _reason, report = _weight_patch_contract(
        patched, "turbo8_t2va_research_exp"
    )
    assert supported is True
    assert report["has_weight_patches"] is True
    assert report["lora_identity_verified_by_runtime"] is False


def _native_h3_patcher_for_conflict_test(model_options=None, *, extra_conds=None):
    class FakeBase:
        pass

    class FakePatcher:
        pass

    base = FakeBase()
    base.diffusion_model = object.__new__(MiniMaxH3Model)
    base.extra_conds = extra_conds or (lambda **_kwargs: None)
    patcher = FakePatcher()
    patcher.model = base
    patcher.model_options = model_options or {}
    return patcher


@pytest.mark.parametrize(
    "model_options",
    [
        {"transformer_options": {"wrappers": {"dit": object()}}},
        {"transformer_options": {"callbacks": {"dit": object()}}},
        {"transformer_options": {"patches": {"dit": object()}}},
        {
            "transformer_options": {
                "patches_replace": {"dit": {("double_block", 0): object()}}
            }
        },
        {"model_function_wrapper": object()},
        {"sampler_post_cfg_function": [object()]},
    ],
)
def test_speed_fails_closed_on_wrappers_and_block_replacements(model_options):
    with pytest.raises(ValueError, match="refuses"):
        _ensure_native_h3_model(_native_h3_patcher_for_conflict_test(model_options))


def test_speed_fails_closed_on_long_video_or_multikeyframe_scoped_model():
    def patched_extra_conds(**_kwargs):
        return None

    patched_extra_conds._t8_long_video_patch_version = "test"
    with pytest.raises(ValueError, match="scoped MODEL patches"):
        _ensure_native_h3_model(
            _native_h3_patcher_for_conflict_test(extra_conds=patched_extra_conds)
        )


def test_strict_t2va_stage_reuses_text_and_rebuilds_only_empty_av_canvas():
    positive = object()
    mux_audio = object()
    template = StageConditioning(
        positive=positive,
        latent={},
        mux_audio=mux_audio,
        conditioned_prompt="fixed prompt",
        media_map="{}",
        report="initial stage",
    )
    stage = _build_empty_t2va_stage(
        {"length": 124},
        width=544,
        height=320,
        template=template,
    )
    video, audio = stage.latent["samples"].unbind()
    assert stage.positive is positive
    assert stage.mux_audio is mux_audio
    assert stage.conditioned_prompt == "fixed prompt"
    assert stage.route == "reused_t2va_text_plus_stage_empty_av"
    assert tuple(video.shape) == (1, 24, 37, 20, 34)
    assert tuple(audio.shape) == (1, 32, 2, 207)
    assert torch.count_nonzero(video) == 0
    assert torch.count_nonzero(audio) == 0


@pytest.mark.parametrize(
    ("task", "mode", "first", "last", "expected_audio_mask"),
    [
        ("I2VA", "lock_source", True, False, 0.0),
        ("FL2VA", "remix_source", True, True, 0.35),
        ("L2VA", "native", False, True, None),
    ],
)
def test_multimodal_stage_rebuilds_keyframes_and_audio_mask_per_canvas(
    task, mode, first, last, expected_audio_mask
):
    source, _ = build_speed_source(
        clip=FakeClip(),
        video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(),
        prompt="controlled multimodal test",
        length=124,
        task_type=task,
        audio_mode=mode,
        audio_denoise_strength=0.35,
        add_source_as_reference=False,
        prompt_primary_audio_ordinal=0,
        strict_prompt_tags=True,
        ref_image_size="match",
        reference_video_policy="official_2_to_15s",
        checkpoint_fingerprint="unrecorded",
        vae_fingerprint="unrecorded",
        drive_audio=make_audio() if mode != "native" else None,
        final_audio=None,
        first_frame=torch.zeros(1, 96, 160, 3) if first else None,
        last_frame=torch.ones(1, 96, 160, 3) if last else None,
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
    )
    stages = [_build_stage(source, 256, 128), _build_stage(source, 512, 256)]
    for stage, expected_hw in zip(stages, ((8, 16), (16, 32))):
        video, _audio = stage.latent["samples"].unbind()
        assert tuple(video.shape[-2:]) == expected_hw
        metadata = stage.positive[0][1]
        keyframes = metadata["minimax_keyframes"]
        assert len(keyframes) == int(first) + int(last)
        assert all(tuple(item["latent"].shape[-2:]) == expected_hw for item in keyframes)
        if expected_audio_mask is None:
            assert "noise_mask" not in stage.latent
        else:
            _video_mask, audio_mask = stage.latent["noise_mask"].unbind()
            assert torch.all(audio_mask == expected_audio_mask)


def test_dct_expansion_matches_scipy_reference_and_is_deterministic():
    scipy_fft = pytest.importorskip("scipy.fft")
    source = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) / 10.0
    expanded, aligned, report = dct_expand_official(
        source,
        5,
        7,
        sigma=0.6,
        ratio=math.sqrt((5 / 3) * (7 / 4)),
        seed=123,
        chunk_size=1,
    )
    generator = torch.Generator(device="cpu").manual_seed(123)
    coefficients = torch.randn((1, 5, 7), generator=generator).numpy() * 0.6
    coefficients[0, :3, :4] = scipy_fft.dctn(source[0].numpy(), type=2, norm="ortho")
    reference = scipy_fft.idctn(coefficients[0], type=2, norm="ortho")
    reference *= kappa(0.6, math.sqrt((5 / 3) * (7 / 4)))
    assert torch.allclose(expanded[0], torch.from_numpy(reference), atol=2e-5, rtol=2e-5)
    repeated, repeated_aligned, _ = dct_expand_official(
        source,
        5,
        7,
        sigma=0.6,
        ratio=math.sqrt((5 / 3) * (7 / 4)),
        seed=123,
        chunk_size=1,
    )
    assert torch.equal(expanded, repeated)
    assert aligned == pytest.approx(repeated_aligned)
    assert report["new_coefficients"] == 23
    other_chunk, _, _ = dct_expand_official(
        source,
        5,
        7,
        sigma=0.6,
        ratio=math.sqrt((5 / 3) * (7 / 4)),
        seed=123,
        chunk_size=2,
    )
    assert torch.equal(expanded, other_chunk)


def test_audio_reindex_and_segment_transport_round_trip():
    video = torch.randn(1, 24, 2, 4, 6)
    audio = torch.randn(1, 32, 2, 8)
    external = comfy.nested_tensor.NestedTensor((video, audio))
    sigma = 0.6
    audio_scale = 4.0
    raw = recover_raw_flow_state(external, sigma=sigma, audio_scale=audio_scale)
    raw_video, raw_audio = raw.unbind()
    assert torch.allclose(raw_video, video * 0.4)
    assert torch.allclose(raw_audio, audio * 1.6)

    target_video = torch.randn_like(video)
    target_audio = torch.randn_like(audio)
    target = comfy.nested_tensor.NestedTensor((target_video, target_audio))
    sigma_to = 0.8
    reindexed_audio = reindex_joint_audio_state(
        raw_audio,
        target_audio * audio_scale,
        sigma_from=sigma,
        sigma_to=sigma_to,
    )
    desired = comfy.nested_tensor.NestedTensor((raw_video, reindexed_audio))
    noise = solve_segment_noise(
        desired,
        target,
        sigma=sigma_to,
        audio_scale=audio_scale,
        noise_scale=1.0,
    )
    noise_video, noise_audio = noise.unbind()
    reconstructed_video = sigma_to * noise_video + (1 - sigma_to) * target_video
    reconstructed_audio = sigma_to * noise_audio + (1 - sigma_to) * target_audio * audio_scale
    assert torch.allclose(reconstructed_video, raw_video)
    assert torch.allclose(reconstructed_audio, reindexed_audio)


def test_modality_stable_noise_keeps_audio_equal_across_video_canvas_sizes():
    audio = torch.zeros(1, 32, 2, 8)
    small = {
        "samples": comfy.nested_tensor.NestedTensor(
            (torch.zeros(1, 24, 2, 4, 6), audio.clone())
        )
    }
    large = {
        "samples": comfy.nested_tensor.NestedTensor(
            (torch.zeros(1, 24, 2, 8, 12), audio.clone())
        )
    }
    small_video, small_audio = modality_stable_h3_noise(small, 123).unbind()
    large_video, large_audio = modality_stable_h3_noise(large, 123).unbind()
    assert small_video.shape != large_video.shape
    assert torch.equal(small_audio, large_audio)


def test_stage_growth_release_targets_only_h3_clone_family(monkeypatch):
    import comfy.model_management as model_management

    events = []
    free_values = iter((128, 1024))
    monkeypatch.setattr(
        model_management,
        "get_free_memory",
        lambda device: next(free_values),
    )
    monkeypatch.setattr(
        model_management,
        "unload_model_and_clones",
        lambda model, unload_additional_models, all_devices: events.append(
            (model, unload_additional_models, all_devices)
        ),
    )
    monkeypatch.setattr(
        model_management,
        "soft_empty_cache",
        lambda: events.append("soft_empty_cache"),
    )
    dummy = type("DummyModel", (), {"load_device": torch.device("cuda")})()
    report = _release_h3_residency_between_stages(dummy)
    assert events == [(dummy, False, False), "soft_empty_cache"]
    assert report["performed"] is True
    assert report["scope"] == "selected_h3_model_and_clones"
    assert report["global_unload_called"] is False
    assert report["free_memory_delta_bytes"] == 896


def test_speed_scoped_headroom_is_temporary_and_never_unloads_models(monkeypatch):
    import comfy.model_management as model_management
    import h3_audio_t8_pkg.speed_advanced as speed_advanced

    calls = []
    monkeypatch.setattr(model_management, "EXTRA_RESERVED_VRAM", 700 * 1024**2)
    monkeypatch.setattr(
        speed_advanced,
        "_speed_dynamic_headroom_control",
        lambda: (lambda value: calls.append(("dynamic", value)), 0, "test"),
    )
    token, report = _apply_speed_scoped_headroom()
    assert model_management.EXTRA_RESERVED_VRAM == int(1.5 * 1024**3)
    assert calls == [("dynamic", int(1.5 * 1024**3))]
    assert report["global_model_unload_called"] is False
    restored = _restore_speed_scoped_headroom(token)
    assert restored["restored"] is True
    assert model_management.EXTRA_RESERVED_VRAM == 700 * 1024**2
    assert calls[-1] == ("dynamic", 0)
    assert _restore_speed_scoped_headroom(token)["restored"] is True


def test_source_resolves_all_media_without_encoding_and_nodes_are_advanced():
    dummy = object()
    image = torch.zeros(1, 64, 64, 3)
    source, report_json = build_speed_source(
        clip=dummy,
        video_vae=dummy,
        audio_vae=dummy,
        prompt="test",
        length=124,
        task_type="FL2VA",
        audio_mode="native",
        audio_denoise_strength=0.35,
        add_source_as_reference=True,
        prompt_primary_audio_ordinal=1,
        strict_prompt_tags=True,
        ref_image_size="match",
        reference_video_policy="official_2_to_15s",
        checkpoint_fingerprint="unrecorded",
        vae_fingerprint="unrecorded",
        drive_audio=None,
        final_audio=None,
        first_frame=image,
        last_frame=image,
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
    )
    assert source["schema"] == SPEED_SOURCE_SCHEMA
    assert source["resolved_task"] == "fl2va"
    assert json.loads(report_json)["resolved_task"] == "fl2va"
    assert len(SPEED_ADVANCED_NODE_CLASSES) == 10
    assert [node.define_schema().node_id for node in SPEED_ADVANCED_NODE_CLASSES[:5]] == [
        "MiniMaxH3SPEEDSpectrumHarvesterT8Advanced",
        "MiniMaxH3SPEEDPlanT8Advanced",
        "MiniMaxH3SPEEDSourceT8Advanced",
        "MiniMaxH3SPEEDSamplerT8Advanced",
        "MiniMaxH3SPEEDModalityStableNoiseT8Advanced",
    ]
    assert [node.define_schema().node_id for node in SPEED_ADVANCED_NODE_CLASSES[5:]] == [
        "MiniMaxH3SPEEDSpectrumDatasetAccumulateT8Advanced",
        "MiniMaxH3SPEEDSpectrumDatasetFinalizeT8Advanced",
        "MiniMaxH3SPEEDSpectrumDatasetFileT8Advanced",
        "MiniMaxH3SPEEDModelVAEFingerprintT8Advanced",
        "MiniMaxH3SPEEDCalibrationWindowT8Advanced",
    ]
    for node in SPEED_ADVANCED_NODE_CLASSES:
        schema = node.define_schema()
        assert schema.is_experimental is True
        assert schema.node_id.endswith("Advanced")
        assert schema.category == "T8/MiniMax H3/SPEED/Experimental"


def test_speed_calibration_window_center_covers_without_anisotropic_stretch():
    frames = torch.linspace(0.0, 1.0, 22 * 160 * 90 * 3, dtype=torch.float32)
    frames = frames.reshape(22, 160, 90, 3)
    selected, frame_count, report_json = prepare_speed_calibration_window(
        frames,
        source_fps=24.0,
        width=64,
        height=32,
        length=22,
        start_seconds=0.0,
        resize_mode="center_cover",
    )
    report = json.loads(report_json)
    assert selected.shape == (22, 32, 64, 3)
    assert frame_count == 22
    assert report["status"] == "aspect_safe_center_cover"
    assert report["source"]["aspect_ratio"] == pytest.approx(90 / 160)
    assert report["target"]["aspect_ratio"] == pytest.approx(2.0)
    assert report["resize"]["anisotropic_stretch"] is False
    assert report["resize"]["cropped_source_pixels_height"] > 0
    assert 0.0 < report["resize"]["retained_source_fraction"] < 1.0
    assert report["sampling"]["unique_source_frames"] == 22


def test_speed_calibration_window_rejects_short_or_stretching_inputs():
    short = torch.zeros((5, 64, 64, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="too short"):
        prepare_speed_calibration_window(
            short,
            source_fps=24.0,
            width=64,
            height=64,
            length=22,
            start_seconds=0.0,
            resize_mode="center_cover",
        )
    with pytest.raises(ValueError, match="only supports aspect-safe"):
        prepare_speed_calibration_window(
            short,
            source_fps=24.0,
            width=64,
            height=64,
            length=5,
            start_seconds=0.0,
            resize_mode="stretch",
        )


@pytest.mark.parametrize(
    ("filename", "task", "scope", "image_count"),
    [
        ("2026-08-18_H3_SPEED_T2VA_Stock20_Advanced_EXP.json", "T2VA", "strict_t2va_stock20", 0),
        ("2026-08-09_H3_SPEED_FL2VA_Stock20_Advanced_EXP.json", "FL2VA", "multimodal_research_exp", 2),
        ("2026-08-09_H3_SPEED_Ref2VA_Stock20_Advanced_EXP.json", "Ref2VA", "multimodal_research_exp", 1),
    ],
)
def test_speed_frontend_workflows_are_importable_and_self_documenting(
    filename, task, scope, image_count
):
    path = Path(__file__).resolve().parents[1] / "examples" / "workflows" / "10-speed" / filename
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert len(nodes) == len(workflow["nodes"])
    assert any(node["type"] == "MiniMaxH3SPEEDPlanT8Advanced" for node in nodes.values())
    source = next(
        node for node in nodes.values() if node["type"] == "MiniMaxH3SPEEDSourceT8Advanced"
    )
    sampler = next(
        node for node in nodes.values() if node["type"] == "MiniMaxH3SPEEDSamplerT8Advanced"
    )
    assert source["widgets_values"][2] == task
    assert sampler["widgets_values"][2] == "fixed"
    assert sampler["widgets_values"][3] == scope
    assert sum(node["type"] == "LoadImage" for node in nodes.values()) == image_count
    notes = [node for node in nodes.values() if node["type"] == "MarkdownNote"]
    assert notes and task in notes[0]["widgets_values"][0]
    links = {link[0]: link for link in workflow["links"]}
    assert len(links) == len(workflow["links"])
    for link_id, origin, origin_slot, target, target_slot, link_type in workflow["links"]:
        assert link_id in links
        assert origin in nodes and target in nodes
        assert link_id in (nodes[origin]["outputs"][origin_slot].get("links") or [])
        assert nodes[target]["inputs"][target_slot]["link"] == link_id
        assert link_type


def test_speed_spectrum_dataset_frontend_workflow_is_safe_and_self_documenting():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "10-speed"
        / "2026-08-19_H3_SPEED_Spectrum_Dataset_Calibration_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    types = [node["type"] for node in nodes.values()]
    assert "MiniMaxH3SPEEDModelVAEFingerprintT8Advanced" in types
    assert "MiniMaxH3SPEEDSpectrumDatasetAccumulateT8Advanced" in types
    assert "MiniMaxH3SPEEDSpectrumDatasetFinalizeT8Advanced" in types
    assert "MiniMaxH3SPEEDCalibrationWindowT8Advanced" in types
    assert "MiniMaxH3SourceMediaWindowT8" not in types
    assert types.count("MiniMaxH3SPEEDSpectrumDatasetFileT8Advanced") == 2
    assert types.count("MarkdownNote") >= 2
    calibration_window = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3SPEEDCalibrationWindowT8Advanced"
    )
    assert calibration_window["widgets_values"] == [
        24.0,
        736,
        416,
        124,
        0.0,
        "center_cover",
    ]
    assert [item["name"] for item in calibration_window["outputs"]] == [
        "frames",
        "frame_count",
        "report_json",
    ]

    file_nodes = [
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3SPEEDSpectrumDatasetFileT8Advanced"
    ]
    load_node = next(node for node in file_nodes if node["widgets_values"][0] == "load")
    save_node = next(node for node in file_nodes if node["widgets_values"][0] == "save")
    expected_dataset = "h3_t2va_vchitect_e068_736x416x124_n100_v1"
    assert load_node["widgets_values"][1] == expected_dataset
    assert save_node["widgets_values"][1] == expected_dataset
    assert load_node["outputs"][0]["links"] is None
    assert save_node["widgets_values"][2:] == [False, False]
    accumulate = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3SPEEDSpectrumDatasetAccumulateT8Advanced"
    )
    assert [item["name"] for item in accumulate["inputs"][-3:]] == [
        "previous_dataset",
        "dataset_provenance_json",
        "source_entry_json",
    ]
    assert accumulate["inputs"][-3]["link"] is None
    assert accumulate["widgets_values"][:5] == [
        "batch_001",
        "T2VA",
        "sha256:connected",
        "sha256:connected",
        32,
    ]
    plan = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3SPEEDPlanT8Advanced"
    )
    assert plan["widgets_values"][4] == "delta_optimal"
    assert plan["widgets_values"][9] == "require_validated_profile"
    assert plan["inputs"][-1]["link"] is not None
    fingerprint = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3SPEEDModelVAEFingerprintT8Advanced"
    )
    assert fingerprint["widgets_values"] == [
        "minimax_h3_fl2va_int8_convrot.safetensors",
        "minimax_h3_video_vae_fp16.safetensors",
    ]
    finalize = next(
        node
        for node in nodes.values()
        if node["type"] == "MiniMaxH3SPEEDSpectrumDatasetFinalizeT8Advanced"
    )
    assert finalize["widgets_values"][0] == (
        "h3_t2va_vchitect_e068_736x416x124_n100_profile_v1"
    )
    note_text = "\n".join(
        node["widgets_values"][0]
        for node in nodes.values()
        if node["type"] == "MarkdownNote"
    )
    assert "R²=0.9951512" in note_text
    assert "禁止稳定默认" in note_text

    links = {link[0]: link for link in workflow["links"]}
    assert len(links) == len(workflow["links"])
    for link_id, origin, origin_slot, target, target_slot, link_type in workflow["links"]:
        assert link_id in (nodes[origin]["outputs"][origin_slot].get("links") or [])
        assert nodes[target]["inputs"][target_slot]["link"] == link_id
        assert nodes[origin]["outputs"][origin_slot]["type"] == link_type
        assert nodes[target]["inputs"][target_slot]["type"] == link_type


def test_calibrated_t2va_frontend_binds_formal_profile_and_denial_notes():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "10-speed"
        / "2026-08-18_H3_SPEED_T2VA_Stock20_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = workflow["nodes"]
    dataset_file = next(
        node
        for node in nodes
        if node["type"] == "MiniMaxH3SPEEDSpectrumDatasetFileT8Advanced"
    )
    assert dataset_file["widgets_values"] == [
        "load",
        "h3_t2va_vchitect_e068_736x416x124_n100_v1",
        False,
        False,
    ]
    plan = next(node for node in nodes if node["type"] == "MiniMaxH3SPEEDPlanT8Advanced")
    assert plan["widgets_values"][:5] == [736, 416, 20, "0.5,1.0", "delta_optimal"]
    sampler = next(
        node for node in nodes if node["type"] == "MiniMaxH3SPEEDSamplerT8Advanced"
    )
    assert sampler["widgets_values"] == [
        3.0,
        2608194001,
        "fixed",
        "strict_t2va_stock20",
        64,
    ]
    note_text = "\n".join(
        node["widgets_values"][0]
        for node in nodes
        if node["type"] == "MarkdownNote"
    )
    assert "248.688秒" in note_text
    assert "总体、运动细节、声音三项也全部选择基线" in note_text
    assert "不是稳定加速、画质增强或16GB安全方案" in note_text
