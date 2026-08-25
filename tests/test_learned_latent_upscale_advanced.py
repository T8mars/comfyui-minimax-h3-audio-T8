from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import comfy.nested_tensor

from h3_audio_t8_pkg import learned_latent_upscale_advanced as learned
from h3_audio_t8_pkg.nodes_learned_latent_upscale_advanced import (
    MiniMaxH3LearnedLatentUpscaleT8Advanced,
    MiniMaxH3LearnedTwoPassParityPlanT8Advanced,
    MiniMaxH3TwoPassAudioAuditT8Advanced,
    MiniMaxH3TwoPassLatentReconcileT8Advanced,
    MiniMaxH3TwoPassSigmaPlanT8Advanced,
)


def _av_latent(height=26, width=46, *, audio_mask=1.0):
    video = torch.randn((1, 24, 7, height, width))
    audio = torch.randn((1, 32, 2, 176))
    video_mask = torch.ones_like(video)
    audio_mask_tensor = torch.full_like(audio, audio_mask)
    return {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "noise_mask": comfy.nested_tensor.NestedTensor((video_mask, audio_mask_tensor)),
        "custom": {"preserved": True},
    }


def test_checkpoint_contract_is_exact_and_rejects_unknown_shapes():
    state = {
        key: torch.empty(shape, device="meta", dtype=torch.float16)
        for key, shape in learned.EXPECTED_STATE_CONTRACT.items()
    }
    report = learned.validate_learned_resizer_state_dict(state)
    assert report["tensor_count"] == 322
    assert report["channels"] == 24
    assert report["base_channels"] == 512

    state["conv_in.weight"] = torch.empty((511, 24, 3, 3, 3), device="meta")
    with pytest.raises(ValueError, match="shape_mismatches"):
        learned.validate_learned_resizer_state_dict(state)


@pytest.mark.parametrize(
    ("mode", "kwargs"),
    [
        ("scale_by", {"scale_by": 1.5}),
        ("target_megapixels", {"target_megapixels": 0.7}),
        ("target_dimensions", {"target_width": 1152, "target_height": 640}),
    ],
)
def test_all_geometry_modes_are_h3_legal_and_aspect_safe(mode, kwargs):
    values = {
        "source_latent_width": 46,
        "source_latent_height": 26,
        "size_mode": mode,
        "scale_by": 1.5,
        "target_megapixels": 0.7,
        "target_width": 1152,
        "target_height": 640,
        "aspect_policy": "preserve_source",
        "max_anisotropy": 1.05,
    }
    values.update(kwargs)
    geometry = learned.learned_upscale_geometry(**values)
    assert geometry["output_width"] % 32 == 0
    assert geometry["output_height"] % 32 == 0
    assert geometry["scale_x"] >= 1.0
    assert geometry["scale_y"] >= 1.0
    assert geometry["anisotropy"] <= 1.05
    assert geometry["output_pixels"] == geometry["output_width"] * geometry["output_height"]
    assert geometry["exceeds_official_reference_area"] is False


def test_geometry_allows_above_official_2mp_reference_area_with_explicit_warning():
    geometry = learned.learned_upscale_geometry(
        source_latent_width=46,
        source_latent_height=26,
        size_mode="target_megapixels",
        scale_by=2.0,
        target_megapixels=4.0,
        target_width=1152,
        target_height=640,
        aspect_policy="preserve_source",
        max_anisotropy=1.05,
    )
    assert geometry["output_pixels"] > learned.H3_OFFICIAL_REFERENCE_PIXELS
    assert geometry["exceeds_official_reference_area"] is True
    assert "Execution is allowed" in geometry["memory_warning"]
    assert geometry["output_width"] % 32 == 0
    assert geometry["output_height"] % 32 == 0


def test_dimension_mode_rejects_large_anisotropic_stretch():
    with pytest.raises(ValueError, match="anisotropic scale"):
        learned.learned_upscale_geometry(
            46,
            26,
            "target_dimensions",
            1.5,
            0.7,
            1344,
            512,
            "honor_dimensions_exp",
            1.05,
        )


class _ResizeNetwork:
    def __call__(self, value, effective_scale, target_size):
        assert effective_scale > 1.0
        return F.interpolate(value, size=target_size, mode="trilinear", align_corners=False)


class _FailNetwork:
    def __call__(self, *_args, **_kwargs):
        raise RuntimeError("synthetic inference failure")


class _FakePatcher:
    def __init__(self, network):
        self.model = network
        self.load_device = torch.device("cpu")


def _patch_model_manager(monkeypatch, network):
    patcher = _FakePatcher(network)
    entry = learned._CachedModel(
        patcher=patcher,
        path=Path("fixture.safetensors"),
        sha256=learned.KNOWN_MODEL_SHA256,
        precision="fp16",
        contract={"tensor_count": 322},
    )
    unloads = []
    monkeypatch.setattr(learned, "_load_cached_model", lambda *_args: (entry, False))
    monkeypatch.setattr(learned.model_management, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        learned.model_management, "intermediate_device", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(learned.model_management, "load_models_gpu", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        learned.model_management,
        "unload_model_and_clones",
        lambda value: unloads.append(value),
    )
    monkeypatch.setattr(learned.model_management, "soft_empty_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(learned, "_drop_cached_entry", lambda _entry: None)
    return patcher, unloads


def test_learned_upscale_preserves_joint_audio_and_resizes_only_video(monkeypatch):
    patcher, unloads = _patch_model_manager(monkeypatch, _ResizeNetwork())
    latent = _av_latent()
    source_video, source_audio = tuple(latent["samples"].unbind())
    _source_video_mask, source_audio_mask = tuple(latent["noise_mask"].unbind())

    output, width, height, report_text = learned.learned_upscale_h3_av_latent(
        latent,
        "fixture.safetensors",
        "target_dimensions",
        1.5,
        0.7,
        1152,
        640,
        "preserve_source",
        1.05,
        "fp16",
        "offload_after",
    )
    output_video, output_audio = tuple(output["samples"].unbind())
    output_video_mask, output_audio_mask = tuple(output["noise_mask"].unbind())
    assert width % 32 == 0 and height % 32 == 0
    assert output_video.shape[:3] == source_video.shape[:3]
    assert output_video.shape[-2:] == (height // 16, width // 16)
    assert output_audio is source_audio
    assert output_audio_mask is source_audio_mask
    assert output_video_mask.shape == output_video.shape
    assert output["custom"] is latent["custom"]
    assert json.loads(report_text)["audio_preserved"] is True
    assert unloads == [patcher]


def test_learned_upscale_always_releases_on_inference_error(monkeypatch):
    patcher, unloads = _patch_model_manager(monkeypatch, _FailNetwork())
    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        learned.learned_upscale_h3_av_latent(
            _av_latent(),
            "fixture.safetensors",
            "scale_by",
            1.5,
            0.7,
            1152,
            640,
            "preserve_source",
            1.05,
            "fp16",
            "keep_loaded",
        )
    assert unloads == [patcher]


def _conditioning(height: int, width: int):
    return [
        [
            torch.zeros((1, 4)),
            {
                "minimax_keyframes": [
                    {
                        "resolved_frame_index": 0,
                        "latent": torch.zeros((1, 24, 1, height, width)),
                    }
                ],
                "minimax_refs": [
                    {
                        "kind": "image",
                        "latent": torch.zeros((1, 24, 1, 16, 16)),
                        "latent_h": 16,
                        "latent_w": 16,
                    }
                ],
            },
        ]
    ]


def test_reconcile_requires_rebuilt_high_resolution_keyframes():
    learned_latent = _av_latent(40, 72)
    template = _av_latent(40, 72)
    with pytest.raises(ValueError, match="Conditioning node again"):
        learned.reconcile_two_pass_h3_latent(
            learned_latent,
            template,
            _conditioning(26, 46),
            "auto",
        )


def test_reconcile_uses_template_audio_for_locked_source_and_preserves_highres_metadata():
    learned_latent = _av_latent(40, 72, audio_mask=1.0)
    template = _av_latent(40, 72, audio_mask=0.0)
    learned_video, learned_audio = tuple(learned_latent["samples"].unbind())
    template_video, template_audio = tuple(template["samples"].unbind())
    template["batch_index"] = [7]
    output, positive, report_text = learned.reconcile_two_pass_h3_latent(
        learned_latent,
        template,
        _conditioning(40, 72),
        "auto",
    )
    output_video, output_audio = tuple(output["samples"].unbind())
    assert output_video is learned_video
    assert output_audio is template_audio
    assert output_audio is not learned_audio
    assert output["batch_index"] is template["batch_index"]
    assert positive is not None
    report = json.loads(report_text)
    assert report["audio_source"] == "highres_template"
    assert report["conditioning"] == {"keyframes": 1, "refs": 1}
    assert template_video.shape == output_video.shape


def test_reconcile_uses_first_pass_audio_when_template_is_fully_denoised():
    first = _av_latent(40, 72, audio_mask=1.0)
    template = _av_latent(40, 72, audio_mask=1.0)
    _video, first_audio = tuple(first["samples"].unbind())
    output, _, report_text = learned.reconcile_two_pass_h3_latent(
        first, template, _conditioning(40, 72), "auto"
    )
    assert tuple(output["samples"].unbind())[1] is first_audio
    assert json.loads(report_text)["audio_source"] == "first_pass"


def test_reconcile_explicitly_locks_first_pass_audio_for_second_pass():
    first = _av_latent(40, 72, audio_mask=1.0)
    template = _av_latent(40, 72, audio_mask=0.2)
    _first_video, first_audio = tuple(first["samples"].unbind())
    template_video_mask, _template_audio_mask = tuple(template["noise_mask"].unbind())

    output, _, report_text = learned.reconcile_two_pass_h3_latent(
        first,
        template,
        _conditioning(40, 72),
        "auto",
        "first_pass",
        0.0,
    )

    _output_video, output_audio = tuple(output["samples"].unbind())
    output_video_mask, output_audio_mask = tuple(output["noise_mask"].unbind())
    assert output_audio is first_audio
    assert output_video_mask is template_video_mask
    assert torch.count_nonzero(output_audio_mask).item() == 0
    report = json.loads(report_text)
    assert report["schema_version"] == 2
    assert report["second_pass_audio_source"] == "first_pass"
    assert report["second_pass_audio_strength"] == 0.0
    assert report["second_pass_audio_locked"] is True
    assert report["second_pass_audio_mask_min"] == 0.0
    assert report["second_pass_audio_mask_max"] == 0.0


def test_reconcile_explicit_audio_contract_builds_video_mask_and_supports_partial_exp():
    first = _av_latent(40, 72)
    template = _av_latent(40, 72)
    template.pop("noise_mask")
    _template_video, template_audio = tuple(template["samples"].unbind())

    output, _, report_text = learned.reconcile_two_pass_h3_latent(
        first,
        template,
        _conditioning(40, 72),
        "first_pass",
        "highres_template",
        0.25,
    )

    output_video_mask, output_audio_mask = tuple(output["noise_mask"].unbind())
    assert tuple(output["samples"].unbind())[1] is template_audio
    assert torch.all(output_video_mask == 1.0)
    assert torch.all(output_audio_mask == 0.25)
    report = json.loads(report_text)
    assert report["second_pass_video_mask_source"] == "generated_all_ones"
    assert report["second_pass_audio_locked"] is False
    assert report["second_pass_audio_mask_min"] == pytest.approx(0.25)
    assert report["second_pass_audio_mask_max"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("source", "strength", "message"),
    [
        ("unknown", 0.0, "second_pass_audio_source"),
        ("first_pass", -0.01, "second_pass_audio_strength"),
        ("first_pass", float("nan"), "second_pass_audio_strength"),
    ],
)
def test_reconcile_rejects_invalid_second_pass_audio_contract(source, strength, message):
    with pytest.raises(ValueError, match=message):
        learned.reconcile_two_pass_h3_latent(
            _av_latent(40, 72),
            _av_latent(40, 72),
            _conditioning(40, 72),
            "auto",
            source,
            strength,
        )


def test_two_pass_audio_audit_accepts_exact_locked_audio_and_relocks_output():
    before = _av_latent(40, 72, audio_mask=0.0)
    after = before.copy()
    before_video, before_audio = tuple(before["samples"].unbind())
    after["samples"] = comfy.nested_tensor.NestedTensor(
        (before_video + 1.0, before_audio.clone())
    )

    output, report_text = learned.audit_two_pass_h3_audio(
        before,
        after,
        expected_audio_strength=0.0,
        fail_on_locked_mismatch=True,
        locked_atol=0.0,
    )

    assert output is not after
    _output_video, output_audio = tuple(output["samples"].unbind())
    assert output_audio is before_audio
    report = json.loads(report_text)
    assert report["status"] == "locked_audio_replaced_exact"
    assert report["audio_exact_equal"] is True
    assert report["audio_relocked_exact"] is True
    assert report["audio_max_abs"] == 0.0
    assert report["audio_rmse"] == 0.0


def test_two_pass_audio_audit_accepts_roundoff_then_replaces_with_exact_input_audio():
    before = _av_latent(40, 72, audio_mask=0.0)
    after = before.copy()
    video, audio = tuple(before["samples"].unbind())
    rounded_audio = audio.clone()
    rounded_audio[..., 0] += 5e-6
    after["samples"] = comfy.nested_tensor.NestedTensor((video, rounded_audio))

    output, report_text = learned.audit_two_pass_h3_audio(
        before, after, 0.0, True, 1e-5
    )

    assert tuple(output["samples"].unbind())[1] is audio
    report = json.loads(report_text)
    assert report["audio_exact_equal"] is False
    assert report["audio_within_tolerance"] is True
    assert report["audio_relocked_exact"] is True


def test_two_pass_audio_audit_fails_closed_when_locked_audio_changes():
    before = _av_latent(40, 72, audio_mask=0.0)
    after = before.copy()
    video, audio = tuple(before["samples"].unbind())
    changed_audio = audio.clone()
    changed_audio[..., 0] += 1e-4
    after["samples"] = comfy.nested_tensor.NestedTensor((video, changed_audio))

    with pytest.raises(ValueError, match="locked audio changed"):
        learned.audit_two_pass_h3_audio(before, after, 0.0, True, 0.0)


def test_two_pass_audio_audit_rejects_missing_or_wrong_mask_contract():
    before = _av_latent(40, 72, audio_mask=0.2)
    after = before.copy()
    with pytest.raises(ValueError, match="expected_audio_strength"):
        learned.audit_two_pass_h3_audio(before, after, 0.0, True, 0.0)

    before.pop("noise_mask")
    with pytest.raises(ValueError, match="has no noise_mask"):
        learned.audit_two_pass_h3_audio(before, after, 0.0, True, 0.0)


def test_reconcile_rejects_reference_shape_metadata_mismatch():
    positive = _conditioning(40, 72)
    positive[0][1]["minimax_refs"][0]["latent_w"] = 15
    with pytest.raises(ValueError, match="metadata/latent mismatch"):
        learned.reconcile_two_pass_h3_latent(
            _av_latent(40, 72), _av_latent(40, 72), positive, "auto"
        )


class _FakeSampling:
    shift = 12.0
    audio_shift = 3.0


class _FakeModel:
    def __init__(self):
        self.model_options = {
            "transformer_options": {"minimax_h3_sigma_shift_audio": 3.0}
        }

    def get_model_object(self, name):
        assert name == "model_sampling"
        return _FakeSampling()


def test_two_pass_sigma_plan_uses_one_base_flow_trajectory_and_actual_shifts():
    coarse, refine, report_text = learned.build_two_pass_sigma_plan(_FakeModel(), 4, 4, 0.5)
    expected_coarse_q = torch.linspace(1.0, 0.5, 5)
    expected_refine_q = torch.linspace(0.5, 0.0, 5)
    assert torch.equal(coarse, learned.shift_sigma(expected_coarse_q, 12.0))
    assert torch.equal(refine, learned.shift_sigma(expected_refine_q, 12.0))
    assert coarse[-1] == refine[0]
    assert refine[-1] == 0
    report = json.loads(report_text)
    assert report["total_nfe"] == 8
    assert report["shift_video"] == 12.0
    assert report["shift_audio"] == 3.0


def test_learned_parity_plan_reproduces_published_refine_sigmas(monkeypatch):
    published_simple = torch.tensor(
        [1.0, 0.98, 0.94, 0.90, 0.86, 0.75, 0.55, 0.30, 0.0]
    )
    monkeypatch.setattr(
        learned.comfy.samplers,
        "calculate_sigmas",
        lambda sampling, scheduler, steps: published_simple,
    )

    class Sampling:
        shift = 6.0
        audio_shift = 3.0

    class Model:
        model_options = {
            "transformer_options": {
                "minimax_h3_sigma_shift_video": 6.0,
                "minimax_h3_sigma_shift_audio": 3.0,
            }
        }

        def get_model_object(self, name):
            assert name == "model_sampling"
            return Sampling()

    coarse, refine, report_text = learned.build_learned_two_pass_parity_plan(
        Model(), 8, 4, 3
    )
    assert torch.equal(coarse, published_simple[:5])
    assert refine.tolist() == pytest.approx([0.9035, 0.6316, 0.3158, 0.0])
    report = json.loads(report_text)
    assert report["previous_linear_q_plan_denied"] is True
    assert report["previous_reference_shift_remap_denied"] is True
    assert report["source_commit"] == learned.UPSTREAM_WORKFLOW_COMMIT
    assert report["coarse_scheduler"] == "comfy.simple"
    assert report["total_nfe"] == 7


def test_learned_parity_plan_keeps_published_raw_video_sigmas_at_shift_12(monkeypatch):
    simple = torch.linspace(1.0, 0.0, 9)
    monkeypatch.setattr(
        learned.comfy.samplers,
        "calculate_sigmas",
        lambda *_args, **_kwargs: simple,
    )
    model = _FakeModel()
    _coarse, refine, report_text = learned.build_learned_two_pass_parity_plan(
        model, 8, 4, 4
    )
    expected = torch.tensor([0.9035, 0.8, 0.6316, 0.3158, 0.0], dtype=torch.float32)
    assert torch.equal(refine, expected)
    report = json.loads(report_text)
    assert report["shift_video"] == 12.0
    expected_audio = learned.shift_sigma(
        learned._inverse_shift_sigma(expected.to(torch.float64), 12.0), 3.0
    )
    assert report["refine_audio_sigmas"] == pytest.approx(expected_audio.tolist())


def test_learned_parity_plan_rejects_unpublished_refine_counts(monkeypatch):
    monkeypatch.setattr(
        learned.comfy.samplers,
        "calculate_sigmas",
        lambda *_args, **_kwargs: torch.linspace(1.0, 0.0, 9),
    )
    with pytest.raises(ValueError, match="one of 3, 4, or 5"):
        learned.build_learned_two_pass_parity_plan(_FakeModel(), 8, 4, 2)


def test_new_nodes_append_without_changing_existing_order():
    import h3_audio_t8_pkg

    classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    ids = [node.define_schema().node_id for node in classes]
    assert len(ids) == 211
    assert ids[125:130] == [
        "MiniMaxH3LearnedLatentUpscaleT8Advanced",
        "MiniMaxH3TwoPassLatentReconcileT8Advanced",
        "MiniMaxH3TwoPassSigmaPlanT8Advanced",
        "MiniMaxH3LearnedTwoPassParityPlanT8Advanced",
        "MiniMaxH3TwoPassDetailMixerT8Advanced",
    ]
    assert ids[130:133] == [
        "MiniMaxH3PromptRelayPlanT8Advanced",
        "MiniMaxH3PromptRelayConditioningT8Advanced",
        "MiniMaxH3PromptRelayQueryRouteT8Advanced",
    ]
    assert ids[133:135] == [
        "MiniMaxH3PromptRelayLongVideoPlanT8Advanced",
        "MiniMaxH3PromptRelayLongVideoConditioningT8Advanced",
    ]
    assert ids[135:137] == [
        "MiniMaxH3PromptPacketRelayPlanT8Advanced",
        "MiniMaxH3PromptRelayEventT8Advanced",
    ]
    assert ids[137] == "MiniMaxH3PromptRelayPreviewT8Advanced"
    assert ids[138] == "MiniMaxH3PromptRelayResourceEstimateT8Advanced"
    assert ids[139] == "MiniMaxH3TwoPassAudioAuditT8Advanced"
    assert ids[140:148] == [
        "MiniMaxH3EnhanceAVideoT8Advanced",
        "MiniMaxH3EnhanceAVideoAuditT8Advanced",
        "MiniMaxH3EnhanceAVideoReferenceComposerT8Advanced",
        "MiniMaxH3EnhanceAVideoSageComposerT8Advanced",
        "MiniMaxH3EnhanceAVideoPromptRelayComposerT8Advanced",
        "MiniMaxH3EnhanceAVideoBlockCacheComposerT8Advanced",
        "MiniMaxH3EnhanceAVideoSTGComposerT8Advanced",
        "MiniMaxH3EnhanceAVideoLongVideoComposerT8Advanced",
    ]
    assert ids[148:155] == [
        "MiniMaxH3MotionOverloadAnalyzeT8Advanced",
        "MiniMaxH3MotionRetimingPrepareT8Advanced",
        "MiniMaxH3MotionRecoveryComposerT8Advanced",
        "MiniMaxH3MotionRecoverAVT8Advanced",
        "MiniMaxH3MotionSegmentPlanT8Advanced",
        "MiniMaxH3MotionWindowCollectT8Advanced",
        "MiniMaxH3MotionAutoGateT8Advanced",
    ]
    assert ids[155:160] == [
        "MiniMaxH3ExternalBlockSwapBridgeT8Advanced",
        "MiniMaxH3LanPaintAVPrepareT8Advanced",
        "MiniMaxH3LanPaintAVCompositeT8Advanced",
        "MiniMaxH3PromptRewriter8BT8Advanced",
        "MiniMaxH3PromptRewriterUnloadT8Advanced",
    ]
    assert ids[160:163] == [
        "MiniMaxH3LightX2VSLAT8Advanced",
        "MiniMaxH3LightX2VSLAAuditT8Advanced",
        "MiniMaxH3LightX2VSLAKJSageComposerT8Advanced",
    ]
    assert ids[163:165] == [
        "MiniMaxH3AudioIntegrityAuditT8Advanced",
        "MiniMaxH3SpeakerRoutingAuditT8Advanced",
    ]
    assert ids[165] == "MiniMaxH3PromptBudgetCompilerT8Advanced"
    assert ids[166:170] == [
        "MiniMaxH3CreatorShotOverrideT8Advanced",
        "MiniMaxH3CreatorWorkspaceT8Advanced",
        "MiniMaxH3CreatorWorkspaceShotSelectT8Advanced",
        "MiniMaxH3CreatorSynchronizedCompareT8Advanced",
    ]
    assert ids[170:172] == [
        "MiniMaxH3ClipProjCompatibilityAuditT8Advanced",
        "MiniMaxH3SolAttnCompatibilityAuditT8Advanced",
    ]
    assert ids[172] == "MiniMaxH3NativeLatentTimelineConcatT8Advanced"
    assert ids[173] == "MiniMaxH3AudioPerceptualDriftAuditT8Advanced"
    assert ids[174:176] == [
        "MiniMaxH3CreatorRunReceiptT8Advanced",
        "MiniMaxH3CreatorResumePlanT8Advanced",
    ]
    assert ids[176:178] == [
        "MiniMaxH3CreatorBackgroundStartT8Advanced",
        "MiniMaxH3CreatorBackgroundRunSelectT8Advanced",
    ]
    assert ids[178] == "MiniMaxH3PromptProviderRouterT8Advanced"
    assert ids[179] == "MiniMaxH3CreatorRetentionPlanT8Advanced"
    assert ids[180] == "MiniMaxH3NativeLatentResumeManifestT8Advanced"
    assert ids[181:183] == [
        "MiniMaxH3NativeLatentCheckpointSaveT8Advanced",
        "MiniMaxH3NativeLatentCheckpointLoadT8Advanced",
    ]
    assert ids[183] == "MiniMaxH3NativeLatentContinuationConcatT8Advanced"
    assert ids[184:187] == [
        "MiniMaxH3RavenStreamingProfileT8Advanced",
        "MiniMaxH3RavenGuardedLoaderT8Advanced",
        "MiniMaxH3RavenRequestAuditT8Advanced",
    ]
    assert ids[187] == "MiniMaxH3NFEResumeSamplerT8Advanced"
    assert ids[188] == "MiniMaxH3CreatorArtifactQuarantineT8Advanced"
    assert ids[189] == "MiniMaxH3PromptSemanticContractAuditT8Advanced"
    assert ids[190] == "MiniMaxH3NFERunContractT8Advanced"
    assert ids[191:208] == [
        "MiniMaxH3SkinFinishT8",
        "MiniMaxH3SkinFinishAdvancedT8",
        "MiniMaxH3SkinFinishPreviewAuditT8Advanced",
        "MiniMaxH3SkinFinishMultiPersonT8Advanced",
        "MiniMaxH3SkinFinishVideoFinalizeT8Advanced",
        "MiniMaxH3SkinFinishVideoStreamT8Advanced",
        "MiniMaxH3SkinFinishTextureGuardT8Advanced",
        "MiniMaxH3SkinFinishSemanticMaskT8Advanced",
        "MiniMaxH3SkinFinishMultiPersonSemanticMaskT8Advanced",
        "MiniMaxH3SkinFinishPersonProfileT8Advanced",
        "MiniMaxH3SkinFinishPerPersonT8Advanced",
        "MiniMaxH3SkinFinishMultiPersonProfileSemanticMaskT8Advanced",
        "MiniMaxH3SkinFinishSafetyAuditT8Advanced",
        "MiniMaxH3SkinFinishFrequencySplitT8Advanced",
        "MiniMaxH3SkinFinishTimelineKeyframeT8Advanced",
        "MiniMaxH3SkinFinishTimelineT8Advanced",
        "MiniMaxH3SkinFinishQualityVideoStreamT8Advanced",
    ]
    assert ids[208] == "MiniMaxH3SkinFinishSpecularFrequencyT8Advanced"
    assert ids[209] == "MiniMaxH3SkinFinishSurfaceT8Advanced"
    assert ids[210] == "MiniMaxH3SkinFinishDichromaticT8Advanced"
    assert ids[94] == "MiniMaxH3LatentUpscaleBy32T8"


def test_node_schemas_are_isolated_experimental_and_safe_by_default():
    learned_schema = MiniMaxH3LearnedLatentUpscaleT8Advanced.define_schema()
    reconcile_schema = MiniMaxH3TwoPassLatentReconcileT8Advanced.define_schema()
    sigma_schema = MiniMaxH3TwoPassSigmaPlanT8Advanced.define_schema()
    parity_schema = MiniMaxH3LearnedTwoPassParityPlanT8Advanced.define_schema()
    audio_audit_schema = MiniMaxH3TwoPassAudioAuditT8Advanced.define_schema()
    learned_inputs = {item.id: item for item in learned_schema.inputs}
    reconcile_inputs = {item.id: item for item in reconcile_schema.inputs}
    sigma_inputs = {item.id: item for item in sigma_schema.inputs}
    assert learned_schema.is_experimental is True
    assert reconcile_schema.is_experimental is True
    assert sigma_schema.is_experimental is True
    assert parity_schema.is_experimental is True
    assert audio_audit_schema.is_experimental is True
    assert audio_audit_schema.is_output_node is True
    assert learned_inputs["release_policy"].default == "offload_after"
    assert learned_inputs["aspect_policy"].default == "preserve_source"
    assert learned_inputs["scale_by"].default == 2.0
    assert reconcile_inputs["audio_policy"].default == "auto"
    assert reconcile_inputs["second_pass_audio_source"].default == "legacy_policy"
    assert reconcile_inputs["second_pass_audio_source"].optional is True
    assert reconcile_inputs["second_pass_audio_strength"].default == 0.0
    assert reconcile_inputs["second_pass_audio_strength"].optional is True
    audio_audit_inputs = {item.id: item for item in audio_audit_schema.inputs}
    assert audio_audit_inputs["expected_audio_strength"].default == 0.0
    assert audio_audit_inputs["fail_on_locked_mismatch"].default is True
    assert audio_audit_inputs["locked_atol"].default == 1e-5
    assert sigma_inputs["coarse_steps"].default == 4
    assert sigma_inputs["refine_steps"].default == 4
    parity_inputs = {item.id: item for item in parity_schema.inputs}
    assert parity_inputs["base_steps"].default == 8
    assert parity_inputs["coarse_steps"].default == 4
    assert parity_inputs["refine_steps"].default == 4
    assert learned_inputs["target_megapixels"].max == 8.0
    assert learned_inputs["target_width"].max == 4096
    assert learned_inputs["target_height"].max == 4096


def test_frontend_two_pass_i2va_workflow_uses_clean_endpoint_and_rebuilt_conditioning():
    workflow_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "13-latent-upscale"
        / "2026-08-21_H3_Learned_Latent_TwoPass_I2VA_Standard_Advanced_EXP.json"
    )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    assert workflow["version"] == 0.4
    nodes = {node["id"]: node for node in workflow["nodes"]}
    types = [node["type"] for node in workflow["nodes"]]
    assert types.count("MiniMaxH3AudioConditioningT8") == 2
    assert types.count("MiniMaxH3DualClockSamplerT8") == 1
    assert types.count("SamplerCustomAdvanced") == 2
    assert types.count("MarkdownNote") >= 5
    assert "MiniMaxH3LearnedLatentUpscaleT8Advanced" in types
    assert "MiniMaxH3TwoPassLatentReconcileT8Advanced" in types
    assert "MiniMaxH3LearnedTwoPassParityPlanT8Advanced" in types
    assert "MiniMaxH3TwoPassDetailMixerT8Advanced" in types
    assert "MiniMaxH3TwoPassAudioAuditT8Advanced" not in types

    low_conditioning = next(node for node in workflow["nodes"] if node["id"] == 7)
    high_conditioning = next(node for node in workflow["nodes"] if node["id"] == 14)
    assert low_conditioning["widgets_values"][1:4] == [736, 416, 124]
    assert high_conditioning["widgets_values"][3] == 124
    assert low_conditioning["widgets_values"][0] == high_conditioning["widgets_values"][0]
    assert low_conditioning["widgets_values"][4] == "I2VA"
    assert high_conditioning["widgets_values"][4] == "I2VA"
    assert low_conditioning["widgets_values"][-1] is False
    assert high_conditioning["widgets_values"][-1] is True

    width_link = next(
        link for link in workflow["links"] if link[1:5] == [13, 1, 14, 3]
    )
    height_link = next(
        link for link in workflow["links"] if link[1:5] == [13, 2, 14, 4]
    )
    assert width_link[5] == "INT"
    assert height_link[5] == "INT"
    assert any(link[1:5] == [12, 1, 13, 0] for link in workflow["links"])
    assert nodes[12]["outputs"][1]["name"] == "denoised_output"
    assert any(link[1:5] == [9, 1, 16, 2] for link in workflow["links"])
    assert any(link[1:5] == [16, 2, 19, 3] for link in workflow["links"])
    assert nodes[13]["widgets_values"][2] == 2.0
    assert nodes[13]["widgets_values"][-1] == "offload_after"
    assert nodes[15]["widgets_values"] == ["auto", "legacy_policy", 0.0]
    assert any(link[1:5] == [19, 0, 20, 0] for link in workflow["links"])
    assert nodes[9]["widgets_values"] == [8, 4, 4]
    assert nodes[16]["widgets_values"][0:2] == [12.0, 3.0]
    assert nodes[16]["widgets_values"][2] is False
    assert nodes[16]["widgets_values"][3] == 3
    note_text = "\n".join(
        node["widgets_values"][0]
        for node in workflow["nodes"]
        if node["type"] == "MarkdownNote"
    )
    assert "8/4/4" in note_text
    assert "8/4/3" not in note_text

    link_ids = [link[0] for link in workflow["links"]]
    assert len(link_ids) == len(set(link_ids))


def test_api_two_pass_i2va_fixture_is_dependency_complete_and_matches_frontend_contract():
    fixture_path = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "api"
        / "learned_latent_two_pass_i2va_advanced_api.json"
    )
    prompt = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert prompt["12"]["inputs"]["sigmas"] == ["9", 0]
    assert prompt["13"]["inputs"]["av_latent"] == ["12", 1]
    assert prompt["16"]["inputs"]["refine_sigmas"] == ["9", 1]
    assert prompt["19"]["inputs"]["sigmas"] == ["16", 2]
    assert prompt["7"]["inputs"]["width"] == 736
    assert prompt["7"]["inputs"]["height"] == 416
    assert "allow_above_reference_area" not in prompt["7"]["inputs"]
    assert prompt["14"]["inputs"]["allow_above_reference_area"] is True
    assert prompt["13"]["inputs"]["size_mode"] == "scale_by"
    assert prompt["13"]["inputs"]["scale_by"] == 2.0
    assert prompt["14"]["inputs"]["width"] == ["13", 1]
    assert prompt["14"]["inputs"]["height"] == ["13", 2]
    assert prompt["13"]["inputs"]["release_policy"] == "offload_after"
    assert prompt["15"]["inputs"]["audio_policy"] == "auto"
    assert prompt["15"]["inputs"]["second_pass_audio_source"] == "legacy_policy"
    assert prompt["15"]["inputs"]["second_pass_audio_strength"] == 0.0
    assert "22" not in prompt
    assert prompt["20"]["inputs"]["av_latent"] == ["19", 0]
    assert prompt["5"]["inputs"]["lora_name"] == (
        "minimax_h3_fl2v_turbo_4step_v0.1_comfyui_alpha8.safetensors"
    )
    assert prompt["5"]["class_type"] == "LoraLoaderBypassModelOnly"
    assert prompt["8"]["inputs"]["shift_video"] == 12.0
    assert prompt["9"]["class_type"] == "MiniMaxH3LearnedTwoPassParityPlanT8Advanced"
    assert prompt["9"]["inputs"]["refine_steps"] == 4
    assert prompt["16"]["class_type"] == "MiniMaxH3TwoPassDetailMixerT8Advanced"
    assert prompt["16"]["inputs"]["enable_tail"] is False
    assert prompt["21"]["class_type"] == "VHS_VideoCombine"
    assert prompt["21"]["inputs"]["format"] == "video/h265-mp4"
