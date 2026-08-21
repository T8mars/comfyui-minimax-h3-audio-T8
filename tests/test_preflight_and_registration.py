from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch

import h3_audio_t8_pkg
from h3_audio_t8_pkg.preflight import run_preflight
from helpers import FakeAudioVAE, FakeVideoVAE, make_audio


def test_all_nodes_register_with_unique_ids_and_valid_schemas():
    extension = h3_audio_t8_pkg.comfy_entrypoint()
    node_classes = asyncio.run(extension.get_node_list())
    schemas = [node.define_schema() for node in node_classes]
    ids = [schema.node_id for schema in schemas]
    assert len(ids) == 145
    assert len(ids) == len(set(ids))
    features = json.loads(
        (Path(__file__).resolve().parents[1] / "features.json").read_text(
            encoding="utf-8"
        )
    )
    assert features["nodes"] == ids
    assert "MiniMaxH3AudioConditioningT8" in ids
    assert "MiniMaxH3DualClockSamplerT8" in ids
    assert "MiniMaxH3MultiRateSamplerEXPT8" in ids
    assert "MiniMaxH3StillConditioningT8" in ids
    assert "MiniMaxH3StillPreflightT8" in ids
    assert "MiniMaxH3StillDecodeT8" in ids
    assert ids[:14] == [
        "MiniMaxH3AudioConditioningT8",
        "MiniMaxH3AudioLatentControlT8",
        "MiniMaxH3DurationPlannerT8",
        "MiniMaxH3AudioWindowT8",
        "MiniMaxH3PromptTagsT8",
        "MiniMaxH3AVDecodeT8",
        "MiniMaxH3AudioMixT8",
        "MiniMaxH3OutputTrimT8",
        "MiniMaxH3PreflightT8",
        "MiniMaxH3DualClockSamplerT8",
        "MiniMaxH3MultiRateSamplerEXPT8",
        "MiniMaxH3StillConditioningT8",
        "MiniMaxH3StillPreflightT8",
        "MiniMaxH3StillDecodeT8",
    ]
    long_video_ids = {
        "MiniMaxH3LongVideoPlannerT8",
        "MiniMaxH3LongVideoContextLoadT8",
        "MiniMaxH3LongVideoConditioningT8",
        "MiniMaxH3LongVideoContextSaveT8",
        "MiniMaxH3LongVideoCandidateSaveT8",
        "MiniMaxH3LongVideoAcceptCandidateT8",
        "MiniMaxH3LongVideoAcceptedContextLoadT8",
        "MiniMaxH3LongVideoComposeAcceptedT8",
        "MiniMaxH3LongVideoOrchestratorT8",
        "MiniMaxH3LongVideoBackgroundStartT8",
        "MiniMaxH3LongVideoAutoQueueT8",
    }
    assert long_video_ids <= set(ids)
    exp_schema = schemas[ids.index("MiniMaxH3MultiRateSamplerEXPT8")]
    assert exp_schema.is_experimental is True
    assert exp_schema.category == "T8/MiniMax H3/Audio/Experimental"
    for still_id in {
        "MiniMaxH3StillConditioningT8",
        "MiniMaxH3StillPreflightT8",
        "MiniMaxH3StillDecodeT8",
    }:
        still_schema = schemas[ids.index(still_id)]
        assert still_schema.is_experimental is True
        assert still_schema.category == "T8/MiniMax H3/Still/Experimental"
    for long_video_id in long_video_ids:
        long_video_schema = schemas[ids.index(long_video_id)]
        assert long_video_schema.is_experimental is True
        assert long_video_schema.category == "T8/MiniMax H3/Long Video/Experimental"
    assert ids[33:35] == [
        "MiniMaxH3SpeechFinalizeT8",
        "MiniMaxH3SpeechStudioT8",
    ]

    speech_ids = {
        "MiniMaxH3VoiceProfileT8",
        "MiniMaxH3SpeechPlanT8",
        "MiniMaxH3SpeechConditioningT8",
        "MiniMaxH3SpeechDecodeT8",
        "MiniMaxH3SpeechVerifyT8",
        "MiniMaxH3SpeechAssembleT8",
        "MiniMaxH3DialogueScriptT8",
        "MiniMaxH3DialogueTurnSelectT8",
        "MiniMaxH3SpeechFinalizeT8",
        "MiniMaxH3SpeechStudioT8",
        "MiniMaxH3SpeechGuardT8",
        "MiniMaxH3SpeechVRAMPreflightT8",
        "MiniMaxH3VoiceLibrarySaveT8",
        "MiniMaxH3VoiceLibraryLoadT8",
        "MiniMaxH3VoiceLibraryDeleteT8",
        "MiniMaxH3SpeechPerformanceT8",
        "MiniMaxH3SpeechADRFitT8",
        "MiniMaxH3SpeechLongFormStartT8",
        "MiniMaxH3SpeechLongFormAcceptT8",
        "MiniMaxH3SpeechLongFormControlT8",
        "MiniMaxH3SpeechLongFormComposeT8",
        "MiniMaxH3JointDialogueConditioningT8",
    }
    assert ids[25:35] == [
        "MiniMaxH3VoiceProfileT8",
        "MiniMaxH3SpeechPlanT8",
        "MiniMaxH3SpeechConditioningT8",
        "MiniMaxH3SpeechDecodeT8",
        "MiniMaxH3SpeechVerifyT8",
        "MiniMaxH3SpeechAssembleT8",
        "MiniMaxH3DialogueScriptT8",
        "MiniMaxH3DialogueTurnSelectT8",
        "MiniMaxH3SpeechFinalizeT8",
        "MiniMaxH3SpeechStudioT8",
    ]
    assert ids[35] == "MiniMaxH3VisualReferenceStrengthEXPT8"
    assert ids[36:48] == [
        "MiniMaxH3SpeechGuardT8",
        "MiniMaxH3SpeechVRAMPreflightT8",
        "MiniMaxH3VoiceLibrarySaveT8",
        "MiniMaxH3VoiceLibraryLoadT8",
        "MiniMaxH3VoiceLibraryDeleteT8",
        "MiniMaxH3SpeechPerformanceT8",
        "MiniMaxH3SpeechADRFitT8",
        "MiniMaxH3SpeechLongFormStartT8",
        "MiniMaxH3SpeechLongFormAcceptT8",
        "MiniMaxH3SpeechLongFormControlT8",
        "MiniMaxH3SpeechLongFormComposeT8",
        "MiniMaxH3JointDialogueConditioningT8",
    ]
    assert ids[48:51] == [
        "MiniMaxH3SourceMediaWindowT8",
        "MiniMaxH3SourceAVPrepareT8",
        "MiniMaxH3AVLatentSeparateT8",
    ]
    assert ids[51:54] == [
        "MiniMaxH3DialogueBoundaryAnalyzerT8",
        "MiniMaxH3DialogueSafeMasterT8",
        "MiniMaxH3TimedAudioBedLockT8",
    ]
    assert ids[54:56] == [
        "MiniMaxH3KeyframePlanT8Advanced",
        "MiniMaxH3MultiKeyframeConditioningT8Advanced",
    ]
    for multikeyframe_id in ids[54:56]:
        multikeyframe_schema = schemas[ids.index(multikeyframe_id)]
        assert multikeyframe_schema.is_experimental is True
        assert multikeyframe_schema.category == "T8/MiniMax H3/Conditioning/Experimental"
    assert ids[56:59] == [
        "MiniMaxH3HybridPairInspectorT8Advanced",
        "MiniMaxH3HybridArtifactBuilderT8Advanced",
        "MiniMaxH3HybridModelLoaderT8Advanced",
    ]
    for hybrid_id in ids[56:59]:
        hybrid_schema = schemas[ids.index(hybrid_id)]
        assert hybrid_schema.is_experimental is True
        assert hybrid_schema.category == "T8/MiniMax H3/Models/Experimental"
    assert ids[59] == "MiniMaxH3VRAMPolicyT8Advanced"
    vram_schema = schemas[59]
    assert vram_schema.is_experimental is True
    assert vram_schema.category == "T8/MiniMax H3/Models/Experimental"
    assert ids[60] == "MiniMaxH3HybridArtifactMaintenanceT8Advanced"
    maintenance_schema = schemas[60]
    assert maintenance_schema.is_experimental is True
    assert maintenance_schema.is_output_node is True
    assert maintenance_schema.category == "T8/MiniMax H3/Models/Experimental"
    maintenance_inputs = {item.id: item for item in maintenance_schema.inputs}
    assert maintenance_inputs["action"].default == "inspect_only"
    assert maintenance_inputs["confirm_action"].default is False
    assert ids[61:73] == [
        "MiniMaxH3HybridCompatibilityAuditT8Advanced",
        "MiniMaxH3EnvironmentAuditT8Advanced",
        "MiniMaxH3ActivationChunkT8Advanced",
        "MiniMaxH3QwenReferencePrefixCacheT8Advanced",
        "MiniMaxH3QwenPrefixCacheStatsT8Advanced",
        "MiniMaxH3UnifiedCastT8Advanced",
        "MiniMaxH3SoundCanvasT8Advanced",
        "MiniMaxH3PromptCompilerT8Advanced",
        "MiniMaxH3StudioTimelineT8Advanced",
        "MiniMaxH3StudioShotSelectT8Advanced",
        "MiniMaxH3SelectiveSegmentRepairT8Advanced",
        "MiniMaxH3RepairSegmentSelectT8Advanced",
    ]
    assert ids[73:77] == [
        "MiniMaxH3SelectiveRepairBindT8Advanced",
        "MiniMaxH3SelectiveRepairStageT8Advanced",
        "MiniMaxH3SelectiveRepairAcceptT8Advanced",
        "MiniMaxH3SelectiveRepairComposeT8Advanced",
    ]
    assert ids[77] == "MiniMaxH3ScheduledDriveAudioInjectionT8Advanced"
    compatibility_schema = schemas[61]
    assert ids[78] == "MiniMaxH3AVDecodeSafetyT8Advanced"
    av_decode_schema = schemas[78]
    assert av_decode_schema.is_experimental is True
    assert av_decode_schema.category == "T8/MiniMax H3/Audio/Experimental"
    av_decode_inputs = {item.id: item for item in av_decode_schema.inputs}
    assert av_decode_inputs["mode"].default == "preflight_only"
    assert av_decode_inputs["enforcement"].default == "report_only"
    assert ids[79:81] == [
        "MiniMaxH3ContextIRProviderT8Advanced",
        "MiniMaxH3ContextIRPromptCompilerT8Advanced",
    ]
    for context_ir_schema in schemas[79:81]:
        assert context_ir_schema.is_experimental is True
        assert context_ir_schema.category == "T8/MiniMax H3/Studio/Experimental"
    assert compatibility_schema.is_experimental is True
    assert ids[81:83] == [
        "MiniMaxH3ReelDeliveryPlanT8Advanced",
        "MiniMaxH3ReelDeliveryComposeT8Advanced",
    ]
    for reel_schema in schemas[81:83]:
        assert reel_schema.is_experimental is True
        assert reel_schema.category == "T8/MiniMax H3/Studio/Experimental"
    assert compatibility_schema.is_output_node is False
    assert ids[83:86] == [
        "MiniMaxH3TrajectoryProbeT8Advanced",
        "MiniMaxH3TrajectoryCheckpointSaveT8Advanced",
        "MiniMaxH3TrajectoryCheckpointLoadT8Advanced",
    ]
    for trajectory_schema in schemas[83:86]:
        assert trajectory_schema.is_experimental is True
        assert trajectory_schema.category == "T8/MiniMax H3/Models/Experimental"
    assert ids[86:90] == [
        "MiniMaxH3AVSigmaTailSubdivisionT8Advanced",
        "MiniMaxH3MotionQualityAuditT8Advanced",
        "MiniMaxH3AVSigmaSameNFERedistributionT8Advanced",
        "MiniMaxH3MotionRepairPlanT8Advanced",
    ]
    for motion_quality_schema in schemas[86:90]:
        assert motion_quality_schema.is_experimental is True
        assert motion_quality_schema.category == "T8/MiniMax H3/Quality/Experimental"
    assert ids[90:94] == [
        "MiniMaxH3FaceRefinePlanT8Advanced",
        "MiniMaxH3FaceRefineConditioningT8Advanced",
        "MiniMaxH3FaceRefineSamplerT8Advanced",
        "MiniMaxH3FaceRefineStitchAuditT8Advanced",
    ]
    for face_refine_schema in schemas[90:94]:
        assert face_refine_schema.is_experimental is True
        assert face_refine_schema.category == "T8/MiniMax H3/Quality/Experimental"
    assert ids[94] == "MiniMaxH3LatentUpscaleBy32T8"
    latent_upscale_schema = schemas[94]
    assert latent_upscale_schema.is_experimental is False
    assert latent_upscale_schema.category == "T8/MiniMax H3/Latent"
    assert ids[95:101] == [
        "MiniMaxH3FaceRefineParityPlanT8Advanced",
        "MiniMaxH3FaceRefineParityLatentT8Advanced",
        "MiniMaxH3FaceRefinePerFrameDenoiseT8Advanced",
        "MiniMaxH3FaceRefineParityStitchT8Advanced",
        "MiniMaxH3FaceRefineQualityGateT8Advanced",
        "MiniMaxH3FaceRefineManual512RelativeBaselineT8Advanced",
    ]
    for parity_schema in schemas[95:101]:
        assert parity_schema.is_experimental is True
        assert parity_schema.category == (
            "T8/MiniMax H3/Quality/Experimental/Face Refine Parity"
        )
    assert ids[101:107] == [
        "MiniMaxH3FaceCharacterProfileT8Advanced",
        "MiniMaxH3FaceCastMergeT8Advanced",
        "MiniMaxH3SAM31MultiPersonTrackT8Advanced",
        "MiniMaxH3FaceTrackAssignT8Advanced",
        "MiniMaxH3MultiFaceRepairJobT8Advanced",
        "MiniMaxH3MultiFaceCompositeT8Advanced",
    ]
    for multiface_schema in schemas[101:107]:
        assert multiface_schema.is_experimental is True
        assert multiface_schema.category == (
            "T8/MiniMax H3/Quality/Experimental/Face Refine Multi-Person"
        )
    assert ids[107:109] == [
        "MiniMaxH3DynamicCFGGuiderT8Advanced",
        "MiniMaxH3DynamicGuidanceAuditT8Advanced",
    ]
    for dynamic_guidance_schema in schemas[107:109]:
        assert dynamic_guidance_schema.is_experimental is True
        assert dynamic_guidance_schema.category == "T8/MiniMax H3/Quality/Experimental"
    assert ids[109:114] == [
        "MiniMaxH3AVTailDetailScheduleT8Advanced",
        "MiniMaxH3ModelTimeBiasSamplerT8Advanced",
        "MiniMaxH3RectifiedFlowRestartSamplerT8Advanced",
        "MiniMaxH3SpatioTemporalGuidanceT8Advanced",
        "MiniMaxH3TemporalDetailEnhanceT8Advanced",
    ]
    for detail_schema in schemas[109:115]:
        assert detail_schema.is_experimental is True
        assert detail_schema.category == "T8/MiniMax H3/Quality/Experimental"
    assert ids[125:128] == [
        "MiniMaxH3LearnedLatentUpscaleT8Advanced",
        "MiniMaxH3TwoPassLatentReconcileT8Advanced",
        "MiniMaxH3TwoPassSigmaPlanT8Advanced",
    ]
    for learned_upscale_schema in schemas[125:128]:
        assert learned_upscale_schema.is_experimental is True
        assert learned_upscale_schema.category == "T8/MiniMax H3/Latent/Experimental"
    assert ids[130:133] == [
        "MiniMaxH3PromptRelayPlanT8Advanced",
        "MiniMaxH3PromptRelayConditioningT8Advanced",
        "MiniMaxH3PromptRelayQueryRouteT8Advanced",
    ]
    for prompt_relay_schema in schemas[130:133]:
        assert prompt_relay_schema.is_experimental is True
        assert prompt_relay_schema.category == "T8/MiniMax H3/Conditioning/Experimental"
    assert ids[133:135] == [
        "MiniMaxH3PromptRelayLongVideoPlanT8Advanced",
        "MiniMaxH3PromptRelayLongVideoConditioningT8Advanced",
    ]
    for prompt_relay_long_video_schema in schemas[133:135]:
        assert prompt_relay_long_video_schema.is_experimental is True
        assert prompt_relay_long_video_schema.category == (
            "T8/MiniMax H3/Long Video/Experimental"
        )
    assert ids[135:137] == [
        "MiniMaxH3PromptPacketRelayPlanT8Advanced",
        "MiniMaxH3PromptRelayEventT8Advanced",
    ]
    for prompt_packet_schema in schemas[135:137]:
        assert prompt_packet_schema.is_experimental is True
        assert prompt_packet_schema.category == (
            "T8/MiniMax H3/Conditioning/Experimental"
        )
    assert ids[137] == "MiniMaxH3PromptRelayPreviewT8Advanced"
    assert schemas[137].is_experimental is True
    assert schemas[137].is_output_node is True
    assert schemas[137].category == "T8/MiniMax H3/Conditioning/Experimental"
    assert ids[138] == "MiniMaxH3PromptRelayResourceEstimateT8Advanced"
    assert schemas[138].is_experimental is True
    assert ids[139] == "MiniMaxH3TwoPassAudioAuditT8Advanced"
    assert schemas[139].is_experimental is True
    assert schemas[139].is_output_node is True
    assert schemas[139].category == "T8/MiniMax H3/Latent/Experimental"
    assert ids[140:145] == [
        "MiniMaxH3EnhanceAVideoT8Advanced",
        "MiniMaxH3EnhanceAVideoAuditT8Advanced",
        "MiniMaxH3EnhanceAVideoReferenceComposerT8Advanced",
        "MiniMaxH3EnhanceAVideoSageComposerT8Advanced",
        "MiniMaxH3EnhanceAVideoPromptRelayComposerT8Advanced",
    ]
    for eav_schema in schemas[140:145]:
        assert eav_schema.is_experimental is True
        assert eav_schema.category == "T8/MiniMax H3/Quality/Experimental"
    assert schemas[138].is_output_node is True
    assert schemas[138].category == "T8/MiniMax H3/Conditioning/Experimental"
    tail_detail_inputs = {item.id: item for item in schemas[109].inputs}
    assert tail_detail_inputs["extra_tail_steps"].default == 1
    assert tail_detail_inputs["spacing"].default == "video_sigma_linear"
    sigma_tail_inputs = {item.id: item for item in schemas[86].inputs}
    assert sigma_tail_inputs["mode"].default == "report_only"
    assert sigma_tail_inputs["extra_substeps"].default == 0
    assert sigma_tail_inputs["profile"].default == "turbo_standard8"
    assert sigma_tail_inputs["accept_turbo_schedule_ood"].default is False
    quality_audit_inputs = {item.id: item for item in schemas[87].inputs}
    assert quality_audit_inputs["roi_mode"].default == "full_frame"
    same_nfe_inputs = {item.id: item for item in schemas[88].inputs}
    assert same_nfe_inputs["mode"].default == "report_only"
    assert same_nfe_inputs["tail_power"].default == 1.6
    assert same_nfe_inputs["profile"].default == "turbo_standard8"
    assert same_nfe_inputs["accept_turbo_schedule_ood"].default is False
    motion_repair_inputs = {item.id: item for item in schemas[89].inputs}
    assert motion_repair_inputs["audit_scope"].default == "single_shot"
    assert motion_repair_inputs["mapping_basis"].default == "suggested_repair_window"
    assert schemas[89].is_output_node is True
    assert compatibility_schema.category == "T8/MiniMax H3/Models/Experimental"
    compatibility_inputs = {item.id: item for item in compatibility_schema.inputs}
    assert compatibility_inputs["enforcement"].default == "report_only"
    assert compatibility_inputs["require_applied_vram_policy"].default is False
    assert compatibility_inputs["positive"].optional is True

    environment_schema = schemas[62]
    assert environment_schema.is_experimental is True
    assert environment_schema.is_output_node is True
    assert environment_schema.category == "T8/MiniMax H3/Models/Experimental"
    environment_inputs = {item.id: item for item in environment_schema.inputs}
    assert environment_inputs["enforcement"].default == "report_only"
    assert environment_inputs["model"].optional is True
    assert environment_inputs["positive"].optional is True


    activation_schema = schemas[63]
    assert activation_schema.is_experimental is True
    assert activation_schema.is_output_node is False
    assert activation_schema.category == "T8/MiniMax H3/Models/Experimental"
    activation_inputs = {item.id: item for item in activation_schema.inputs}
    assert activation_inputs["mode"].default == "report_only"
    assert activation_inputs["chunk_rows"].default == 256
    assert activation_inputs["preserve_short_path"].default is True

    qwen_schema = schemas[64]
    assert qwen_schema.is_experimental is True
    assert qwen_schema.category == "T8/MiniMax H3/Conditioning/Experimental"
    qwen_inputs = {item.id: item for item in qwen_schema.inputs}
    assert qwen_inputs["mode"].default == "report_only"
    qwen_stats_schema = schemas[65]
    assert qwen_stats_schema.is_experimental is True
    assert qwen_stats_schema.is_output_node is True
    assert qwen_stats_schema.category == "T8/MiniMax H3/Conditioning/Experimental"
    studio_schemas = schemas[66:73]
    assert len(studio_schemas) == 7
    assert all(schema.is_experimental for schema in studio_schemas)
    assert all(schema.category == "T8/MiniMax H3/Studio/Experimental" for schema in studio_schemas)
    timeline_inputs = {item.id: item for item in studio_schemas[3].inputs}
    assert timeline_inputs["split_long_shots"].default is True
    repair_inputs = {item.id: item for item in studio_schemas[5].inputs}
    assert repair_inputs["selection_policy"].default == "manual"
    assert repair_inputs["repair_mode"].default == "auto"
    repair_execution_schemas = schemas[73:77]
    assert all(schema.is_experimental for schema in repair_execution_schemas)
    assert all(
        schema.category == "T8/MiniMax H3/Studio/Experimental"
        for schema in repair_execution_schemas
    )
    assert repair_execution_schemas[2].is_output_node is True
    assert repair_execution_schemas[3].is_output_node is True
    accept_inputs = {item.id: item for item in repair_execution_schemas[2].inputs}
    assert accept_inputs["accept_repair"].default is False
    assert accept_inputs["replace_existing"].default is False

    assert ids[23:25] == [
        "MiniMaxH3LongVideoBackgroundStartT8",
        "MiniMaxH3LongVideoAutoQueueT8",
    ]
    for speech_id in speech_ids:
        speech_schema = schemas[ids.index(speech_id)]
        assert speech_schema.is_experimental is True
        assert speech_schema.category == "T8/MiniMax H3/Speech/Experimental"

    visual_strength = schemas[ids.index("MiniMaxH3VisualReferenceStrengthEXPT8")]
    assert visual_strength.is_experimental is True
    assert visual_strength.category == "T8/MiniMax H3/Conditioning/Experimental"

    for source_av_id in ids[48:51]:
        source_av_schema = schemas[ids.index(source_av_id)]
        assert source_av_schema.is_experimental is True
        assert source_av_schema.category == "T8/MiniMax H3/Source AV/Experimental"

    for dialogue_audio_id in ids[51:54]:
        dialogue_audio_schema = schemas[ids.index(dialogue_audio_id)]
        assert dialogue_audio_schema.is_experimental is True
        assert dialogue_audio_schema.category == "T8/MiniMax H3/Speech/Experimental"

    timed_bed = schemas[ids.index("MiniMaxH3TimedAudioBedLockT8")]
    timed_inputs = {item.id: item for item in timed_bed.inputs}
    assert timed_inputs["tail_denoise_strength"].default == 0.0
    assert timed_inputs["transition_seconds"].default == 0.0
    assert timed_inputs["audio_latent_fit_policy"].default == "strict"

    source_media = schemas[ids.index("MiniMaxH3SourceMediaWindowT8")]
    media_inputs = {item.id: item for item in source_media.inputs}
    assert media_inputs["length"].default == 124
    assert media_inputs["short_video_policy"].default == "strict"
    assert media_inputs["short_audio_policy"].default == "pad_silence"
    assert media_inputs["source_audio"].optional is True

    source_prepare = schemas[ids.index("MiniMaxH3SourceAVPrepareT8")]
    source_inputs = {item.id: item for item in source_prepare.inputs}
    assert source_inputs["video_mode"].default == "remix"
    assert source_inputs["video_denoise_strength"].default == 0.5
    assert source_inputs["audio_mode"].default == "lock"
    assert source_inputs["audio_fit_policy"].default == "fit_to_video_generate_tail"
    assert source_inputs["dtype_device_policy"].default == "match_video"

    voice_profile = schemas[ids.index("MiniMaxH3VoiceProfileT8")]
    rights = next(item for item in voice_profile.inputs if item.id == "rights_confirmed")
    assert rights.default is False

    studio = schemas[ids.index("MiniMaxH3SpeechStudioT8")]
    studio_inputs = {item.id: item for item in studio.inputs}
    assert studio_inputs["steps"].default == 20
    assert studio_inputs["sampler_name"].default == "res_multistep"
    assert studio_inputs["scheduler"].default == "simple"
    assert studio_inputs["release_policy"].default == "clear_execution_cache"

    background_start = schemas[ids.index("MiniMaxH3LongVideoBackgroundStartT8")]
    mode = next(item for item in background_start.inputs if item.id == "execution_mode")
    release = next(item for item in background_start.inputs if item.id == "release_policy")
    assert mode.default == "review_only"
    assert mode.options == ["review_only", "auto_accept_and_continue"]
    assert release.default == "clear_execution_cache"
    assert release.options == [
        "keep_loaded",
        "clear_execution_cache",
        "unload_all_models",
    ]

    long_conditioning = schemas[ids.index("MiniMaxH3LongVideoConditioningT8")]
    first_frame_reuse = next(
        item for item in long_conditioning.inputs if item.id == "first_frame_reuse"
    )
    assert long_conditioning.inputs[-4].id == "first_frame_reuse"
    assert long_conditioning.inputs[-3].id == "persistent_identity_image"
    assert long_conditioning.inputs[-3].optional is True
    strategy = long_conditioning.inputs[-2]
    assert strategy.id == "persistent_identity_strategy"
    assert strategy.default == "single_reference"
    assert strategy.options == ["single_reference", "scene_plus_identity"]
    interval = long_conditioning.inputs[-1]
    assert interval.id == "persistent_identity_interval"
    assert interval.default == 1
    assert interval.optional is True
    assert first_frame_reuse.default == "segment0_only"
    assert first_frame_reuse.options == [
        "segment0_only",
        "persistent_identity_reference",
    ]


def test_task_type_frontend_labels_preserve_canonical_backend_values():
    extension = h3_audio_t8_pkg.comfy_entrypoint()
    node_classes = asyncio.run(extension.get_node_list())
    conditioning = next(
        node for node in node_classes
        if node.define_schema().node_id == "MiniMaxH3AudioConditioningT8"
    )
    task_type = next(
        item for item in conditioning.define_schema().inputs
        if item.id == "task_type"
    )
    highres_opt_in = next(
        item for item in conditioning.define_schema().inputs
        if item.id == "allow_above_reference_area"
    )
    assert task_type.options == [
        "auto", "T2VA", "I2VA", "FL2VA", "L2VA", "Ref2VA", "Hybrid",
    ]
    assert highres_opt_in.default is False
    assert highres_opt_in.optional is True
    assert conditioning.define_schema().inputs[-1].id == "allow_above_reference_area"

    package_root = Path(__file__).resolve().parents[1]
    assert h3_audio_t8_pkg.WEB_DIRECTORY == "./web"
    frontend = (package_root / "web" / "task_type_labels.js").read_text(encoding="utf-8")
    for label in {
        "auto — 自动判断",
        "T2VA — 文生音视频",
        "I2VA — 图生音视频（首帧）",
        "FL2VA — 首尾帧生音视频",
        "L2VA — 尾帧生音视频",
        "Ref2VA — 参考生音视频",
        "Hybrid — 关键帧+参考混合生成",
    }:
        assert label in frontend
    assert '"MiniMaxH3LongVideoConditioningT8"' in frontend
    assert "toBackendValue(widget.value)" in frontend

    background_frontend = (
        package_root / "web" / "long_video_background.js"
    ).read_text(encoding="utf-8")
    assert '"MiniMaxH3LongVideoBackgroundStartT8"' in background_frontend
    for action in {"pause", "resume", "cancel"}:
        assert f'"{action}"' in background_frontend
    assert "/minimax_h3_t8/long_video/background" in background_frontend


def test_dual_clock_sampler_appends_optional_choices_without_reordering_legacy_widgets():
    extension = h3_audio_t8_pkg.comfy_entrypoint()
    node_classes = asyncio.run(extension.get_node_list())
    sampler_node = next(
        node for node in node_classes
        if node.define_schema().node_id == "MiniMaxH3DualClockSamplerT8"
    )
    inputs = sampler_node.define_schema().inputs

    assert [item.id for item in inputs] == [
        "model",
        "av_latent",
        "steps",
        "shift_video",
        "shift_audio",
        "sampler_name",
        "scheduler",
    ]
    sampler_name = inputs[-2]
    scheduler = inputs[-1]
    assert sampler_name.optional is True
    assert sampler_name.default == "dual_clock_euler"
    assert sampler_name.options[0] == "dual_clock_euler"
    assert "euler" in sampler_name.options
    assert scheduler.optional is True
    assert scheduler.default == "native_flow"
    assert scheduler.options[0] == "native_flow"
    assert "normal" in scheduler.options


def test_preflight_reports_alignment_audio_and_reference_guidance():
    ready, warning_count, report = run_preflight(
        1344, 768, 123, "lock_source", video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        drive_audio=make_audio(1, value=0),
        ref_videos={"ref_video_1": torch.zeros((20, 32, 32, 3))},
    )
    data = json.loads(report)
    assert ready is True
    assert warning_count >= 3
    assert data["facts"]["aligned_frames"] == 124
    assert data["facts"]["video_vae_kind"] == "video"
    assert not any("swapped" in warning for warning in data["warnings"])


def test_preflight_allows_1080p_area_and_blocks_only_above_it():
    ready, warning_count, report = run_preflight(1920, 1088, 124, "native")
    data = json.loads(report)
    assert ready is True
    assert warning_count >= 1
    assert data["facts"]["pixels"] == 1920 * 1088
    assert any("VRAM" in warning for warning in data["warnings"])

    ready, _, report = run_preflight(1952, 1088, 124, "lock_source")
    assert ready is False
    assert len(json.loads(report)["errors"]) == 2


def test_preflight_distinguishes_h3_video_and_audio_vaes_by_latent_contract():
    ready, _, report = run_preflight(
        1344,
        768,
        124,
        "native",
        video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(),
    )
    data = json.loads(report)
    assert ready is True
    assert data["facts"]["video_vae_kind"] == "video"
    assert data["facts"]["audio_vae_kind"] == "audio"

    ready, _, report = run_preflight(
        1344,
        768,
        124,
        "native",
        video_vae=FakeAudioVAE(),
        audio_vae=FakeVideoVAE(),
    )
    data = json.loads(report)
    assert ready is False
    assert any("video_vae is an H3 audio VAE" in error for error in data["errors"])
    assert any("audio_vae is an H3 video VAE" in error for error in data["errors"])


def test_example_api_workflow_is_valid_and_references_existing_nodes():
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "api" / "audio_lock_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    custom_types = {value["class_type"] for value in workflow.values() if value["class_type"].endswith("T8")}
    assert custom_types == {
        "MiniMaxH3AudioWindowT8", "MiniMaxH3AudioConditioningT8",
        "MiniMaxH3AVDecodeT8", "MiniMaxH3OutputTrimT8",
    }
    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_dual_clock_example_uses_one_coherent_sampling_setup():
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "api" / "dual_clock_4step_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    dual_nodes = [value for value in workflow.values() if value["class_type"] == "MiniMaxH3DualClockSamplerT8"]
    assert len(dual_nodes) == 1
    assert dual_nodes[0]["inputs"]["steps"] == 4
    assert not any(value["class_type"] in {"MiniMaxH3SigmaShift", "KSamplerSelect", "BasicScheduler"}
                   for value in workflow.values())
    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_multikeyframe_advanced_api_is_isolated_and_wired_to_the_cloned_model():
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "api" / "multikeyframe_advanced_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))

    plan_ids = [
        node_id for node_id, node in workflow.items()
        if node["class_type"] == "MiniMaxH3KeyframePlanT8Advanced"
    ]
    assert len(plan_ids) == 2
    assert workflow[plan_ids[1]]["inputs"]["previous_plan"] == [plan_ids[0], 0]
    assert [workflow[node_id]["inputs"]["position"] for node_id in plan_ids] == [33.0, 67.0]
    assert [workflow[node_id]["inputs"]["visual_noise_aug"] for node_id in plan_ids] == [
        0.999,
        0.999,
    ]

    conditioning_id = next(
        node_id for node_id, node in workflow.items()
        if node["class_type"] == "MiniMaxH3MultiKeyframeConditioningT8Advanced"
    )
    conditioning = workflow[conditioning_id]
    assert conditioning["inputs"]["keyframe_plan"] == [plan_ids[1], 0]
    assert conditioning["inputs"]["first_frame_noise_aug"] == 0.999
    assert conditioning["inputs"]["last_frame_noise_aug"] == 0.999

    sampler_id = next(
        node_id for node_id, node in workflow.items()
        if node["class_type"] == "MiniMaxH3DualClockSamplerT8"
    )
    sampler = workflow[sampler_id]
    assert sampler["inputs"]["model"] == [conditioning_id, 0]
    assert sampler["inputs"]["av_latent"] == [conditioning_id, 2]

    guider = next(node for node in workflow.values() if node["class_type"] == "BasicGuider")
    assert guider["inputs"]["conditioning"] == [conditioning_id, 1]
    sample = next(
        node for node in workflow.values() if node["class_type"] == "SamplerCustomAdvanced"
    )
    assert sample["inputs"]["latent_image"] == [conditioning_id, 2]

    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_multikeyframe_advanced_frontend_workflow_is_consistent_and_opt_in():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples" / "workflows" / "08-multi-keyframe" / "2026-08-09_H3_MultiKeyframe_Advanced_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert len(nodes) == len(workflow["nodes"])

    types = [node["type"] for node in workflow["nodes"]]
    assert types.count("MiniMaxH3KeyframePlanT8Advanced") == 2
    assert types.count("MiniMaxH3MultiKeyframeConditioningT8Advanced") == 1
    assert "MiniMaxH3AudioConditioningT8" not in types
    assert not any(node_type.startswith("MiniMaxH3LongVideo") for node_type in types)
    assert not any(node_type.startswith("MiniMaxH3Motion") for node_type in types)

    plans = [
        node for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3KeyframePlanT8Advanced"
    ]
    assert [node["widgets_values"][1] for node in plans] == [33.0, 67.0]
    assert [node["widgets_values"][2] for node in plans] == [0.999, 0.999]
    assert plans[1]["inputs"][6]["name"] == "previous_plan"
    assert plans[1]["inputs"][6]["link"] is not None

    conditioning = next(
        node for node in workflow["nodes"]
        if node["type"] == "MiniMaxH3MultiKeyframeConditioningT8Advanced"
    )
    assert conditioning["inputs"][21]["name"] == "first_frame"
    assert conditioning["inputs"][22]["name"] == "last_frame"
    assert conditioning["inputs"][23]["name"] == "keyframe_plan"
    assert all(conditioning["inputs"][index]["link"] is not None for index in (21, 22, 23))

    for link_id, source, output_slot, target, input_slot, _ in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])


def test_long_video_api_example_is_isolated_retry_safe_and_trimmed():
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "api" / "long_video_segment_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    custom_types = {
        value["class_type"] for value in workflow.values()
        if value["class_type"].startswith("MiniMaxH3")
    }
    assert {
        "MiniMaxH3LongVideoPlannerT8",
        "MiniMaxH3LongVideoContextLoadT8",
        "MiniMaxH3LongVideoConditioningT8",
        "MiniMaxH3LongVideoContextSaveT8",
        "MiniMaxH3DualClockSamplerT8",
        "MiniMaxH3AVDecodeT8",
        "MiniMaxH3OutputTrimT8",
    } <= custom_types
    planner_id = next(key for key, value in workflow.items()
                      if value["class_type"] == "MiniMaxH3LongVideoPlannerT8")
    save = next(value for value in workflow.values()
                if value["class_type"] == "MiniMaxH3LongVideoContextSaveT8")
    assert save["inputs"]["save_context"] == [planner_id, 8]
    class_types = {value["class_type"] for value in workflow.values()}
    assert {"CreateVideo", "SaveVideo"} <= class_types
    assert "VHS_VideoCombine" not in class_types
    assert not any(value["class_type"].startswith("MiniMaxH3Motion") for value in workflow.values())
    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_long_video_candidate_api_separates_preview_from_accepted_state():
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "api" / "long_video_candidate_accept_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    types = {value["class_type"] for value in workflow.values()}
    assert {
        "MiniMaxH3LongVideoPlannerT8",
        "MiniMaxH3LongVideoAcceptedContextLoadT8",
        "MiniMaxH3LongVideoConditioningT8",
        "MiniMaxH3LongVideoCandidateSaveT8",
        "MiniMaxH3LongVideoAcceptCandidateT8",
    } <= types
    assert "MiniMaxH3LongVideoContextSaveT8" not in types
    accepted_loader_id = next(
        key for key, value in workflow.items()
        if value["class_type"] == "MiniMaxH3LongVideoAcceptedContextLoadT8"
    )
    candidate = next(
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3LongVideoCandidateSaveT8"
    )
    assert candidate["inputs"]["parent_candidate_id"] == [accepted_loader_id, 2]
    assert candidate["inputs"]["parent_manifest_revision"] == [accepted_loader_id, 3]
    seed_id = next(
        key for key, value in workflow.items() if value["class_type"] == "PrimitiveInt"
    )
    noise = next(value for value in workflow.values() if value["class_type"] == "RandomNoise")
    assert noise["inputs"]["noise_seed"] == [seed_id, 0]
    assert candidate["inputs"]["seed"] == [seed_id, 0]
    conditioning_id = next(
        key for key, value in workflow.items()
        if value["class_type"] == "MiniMaxH3LongVideoConditioningT8"
    )
    assert candidate["inputs"]["prompt"] == [conditioning_id, 4]
    review = next(
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3LongVideoAcceptCandidateT8"
    )
    assert review["inputs"]["accept_candidate"] is False
    assert review["inputs"]["replace_policy"] == "reject_existing"
    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_long_video_compose_api_requires_an_explicit_final_segment():
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "api" / "long_video_compose_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    assert list(workflow.values())[0]["class_type"] == "MiniMaxH3LongVideoComposeAcceptedT8"
    assert list(workflow.values())[0]["inputs"]["require_final_segment"] is True
    assert list(workflow.values())[0]["inputs"]["audio_seam_policy"] == "cosine_bridge"


def test_long_video_auto_resume_api_drives_segment_prompt_and_seed_from_one_plan():
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "api" / "long_video_auto_resume_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    orchestrator_id = next(
        key for key, value in workflow.items()
        if value["class_type"] == "MiniMaxH3LongVideoOrchestratorT8"
    )
    orchestrator = workflow[orchestrator_id]
    assert orchestrator["inputs"]["total_duration_seconds"] == 60.0
    assert orchestrator["inputs"]["render_window_frames"] == 124
    assert orchestrator["inputs"]["context_frames"] == 22
    assert not any(value["class_type"] == "PrimitiveInt" for value in workflow.values())
    conditioning = next(
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3LongVideoConditioningT8"
    )
    noise = next(value for value in workflow.values() if value["class_type"] == "RandomNoise")
    candidate = next(
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3LongVideoCandidateSaveT8"
    )
    assert conditioning["inputs"]["prompt"] == [orchestrator_id, 10]
    assert noise["inputs"]["noise_seed"] == [orchestrator_id, 11]
    assert candidate["inputs"]["seed"] == [orchestrator_id, 11]
    sampler = next(
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3DualClockSamplerT8"
    )
    assert sampler["inputs"]["steps"] == [orchestrator_id, 16]
    assert sampler["inputs"]["shift_video"] == [orchestrator_id, 17]
    assert sampler["inputs"]["shift_audio"] == [orchestrator_id, 18]
    assert sampler["inputs"]["sampler_name"] == [orchestrator_id, 19]
    assert sampler["inputs"]["scheduler"] == [orchestrator_id, 20]
    assert candidate["inputs"]["sampling_summary"] == [orchestrator_id, 21]
    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_long_video_background_api_is_explicit_and_queues_through_one_terminal():
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "api" / "long_video_background_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    start_id = next(
        key for key, value in workflow.items()
        if value["class_type"] == "MiniMaxH3LongVideoBackgroundStartT8"
    )
    start = workflow[start_id]
    assert start["inputs"] == {
        "chain_id": "h3_background_demo",
        "execution_mode": "auto_accept_and_continue",
        "max_retries": 1,
        "retry_delay_seconds": 2.0,
        "release_policy": "clear_execution_cache",
    }
    orchestrator = next(
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3LongVideoOrchestratorT8"
    )
    assert orchestrator["inputs"]["chain_id"] == [start_id, 0]
    terminal = next(
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3LongVideoAutoQueueT8"
    )
    assert terminal["inputs"]["job_id"] == [start_id, 2]
    assert terminal["inputs"]["auto_accept"] == [start_id, 1]
    assert terminal["inputs"]["compose_when_complete"] is True
    assert not any(
        value["class_type"] == "MiniMaxH3LongVideoAcceptCandidateT8"
        for value in workflow.values()
    )
    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_long_video_frontend_workflow_has_consistent_links_and_no_global_motion_node():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples" / "workflows" / "04-long-video" / "2026-08-09_H3_Long_Video_22F_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert len(nodes) == len(workflow["nodes"])
    assert workflow["last_link_id"] == len(workflow["links"])
    assert not any(node["type"].startswith("MiniMaxH3Motion") for node in nodes.values())
    node_types = {node["type"] for node in nodes.values()}
    assert {"CreateVideo", "SaveVideo"} <= node_types
    assert "VHS_VideoCombine" not in node_types
    for link_id, source, output_slot, target, input_slot, _ in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])


def test_long_video_accepted_frontend_workflow_is_review_first_and_consistent():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples" / "workflows" / "04-long-video" / "2026-08-09_H3_Long_Video_Accepted_22F_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert len(nodes) == len(workflow["nodes"])
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    types = {node["type"] for node in nodes.values()}
    assert {
        "MiniMaxH3LongVideoAcceptedContextLoadT8",
        "MiniMaxH3LongVideoCandidateSaveT8",
        "MiniMaxH3LongVideoAcceptCandidateT8",
        "PrimitiveInt",
    } <= types
    assert "MiniMaxH3LongVideoContextLoadT8" not in types
    assert "MiniMaxH3LongVideoContextSaveT8" not in types
    assert "CreateVideo" not in types and "SaveVideo" not in types
    review = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoAcceptCandidateT8"
    )
    assert review["widgets_values"] == [False, "reject_existing", True]
    candidate = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoCandidateSaveT8"
    )
    assert candidate["inputs"][7]["name"] == "parent_candidate_id"
    assert candidate["inputs"][7]["link"] is not None
    assert candidate["inputs"][8]["name"] == "parent_manifest_revision"
    assert candidate["inputs"][8]["link"] is not None
    seed_node = next(node for node in nodes.values() if node["type"] == "PrimitiveInt")
    noise = next(node for node in nodes.values() if node["type"] == "RandomNoise")
    assert noise["inputs"][0]["link"] in seed_node["outputs"][0]["links"]
    assert candidate["inputs"][13]["link"] in seed_node["outputs"][0]["links"]
    conditioning = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoConditioningT8"
    )
    assert candidate["inputs"][12]["link"] in conditioning["outputs"][4]["links"]
    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_long_video_auto_resume_frontend_workflow_has_one_timeline_source():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples" / "workflows" / "04-long-video" / "2026-08-09_H3_Long_Video_Auto_Resume_22F_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    types = {node["type"] for node in nodes.values()}
    assert "MiniMaxH3LongVideoOrchestratorT8" in types
    assert "MiniMaxH3LongVideoPlannerT8" not in types
    assert "PrimitiveInt" not in types
    orchestrator = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoOrchestratorT8"
    )
    assert orchestrator["widgets_values"][1:4] == [60.0, 124, 22]
    assert orchestrator["widgets_values"][7] == "increment"
    assert orchestrator["widgets_values"][8:] == [
        4, 12.0, 3.0, "dual_clock_euler", "native_flow",
    ]
    conditioning = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoConditioningT8"
    )
    noise = next(node for node in nodes.values() if node["type"] == "RandomNoise")
    candidate = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoCandidateSaveT8"
    )
    assert conditioning["inputs"][8]["link"] in orchestrator["outputs"][10]["links"]
    assert noise["inputs"][0]["link"] in orchestrator["outputs"][11]["links"]
    assert candidate["inputs"][13]["link"] in orchestrator["outputs"][11]["links"]
    sampler = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3DualClockSamplerT8"
    )
    for input_slot, output_slot in zip(range(2, 7), range(16, 21), strict=True):
        assert sampler["inputs"][input_slot]["link"] in (
            orchestrator["outputs"][output_slot]["links"]
        )
    assert candidate["inputs"][11]["link"] in orchestrator["outputs"][21]["links"]
    review = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoAcceptCandidateT8"
    )
    assert review["widgets_values"][0] is False
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_long_video_background_frontend_workflow_has_explicit_controller_links():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples" / "workflows" / "04-long-video" / "2026-08-09_H3_Long_Video_Background_22F_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    start = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoBackgroundStartT8"
    )
    terminal = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoAutoQueueT8"
    )
    orchestrator = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoOrchestratorT8"
    )
    assert start["widgets_values"] == [
        "h3_background_demo",
        "auto_accept_and_continue",
        1,
        2.0,
        "clear_execution_cache",
    ]
    assert "MiniMaxH3LongVideoAcceptCandidateT8" not in {
        node["type"] for node in nodes.values()
    }
    links = {link[0]: link for link in workflow["links"]}
    chain_link = links[orchestrator["inputs"][0]["link"]]
    job_link = links[terminal["inputs"][1]["link"]]
    auto_link = links[terminal["inputs"][2]["link"]]
    assert chain_link[1:5] == [start["id"], 0, orchestrator["id"], 0]
    assert job_link[1:5] == [start["id"], 2, terminal["id"], 1]
    assert auto_link[1:5] == [start["id"], 1, terminal["id"], 2]
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(links)
    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_scene_plus_identity_background_workflow_wires_two_images_and_exp_policy():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "04-long-video"
        / "2026-08-09_H3_Long_Video_Background_22F_ScenePlusIdentity_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}

    conditioning = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoConditioningT8"
    )
    full_scene = next(
        node for node in nodes.values()
        if node.get("title", "").startswith("0a. Full scene")
    )
    identity_crop = next(
        node for node in nodes.values()
        if node.get("title", "").startswith("0b. Same-subject")
    )
    inputs = {value["name"]: value for value in conditioning["inputs"]}

    assert conditioning["widgets_values"][-3:] == [
        "persistent_identity_reference",
        "scene_plus_identity",
        1,
    ]
    assert links[inputs["first_frame"]["link"]][1:5] == [
        full_scene["id"], 0, conditioning["id"], 22,
    ]
    assert links[inputs["persistent_identity_image"]["link"]][1:5] == [
        identity_crop["id"], 0, conditioning["id"], 25,
    ]

    start = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoBackgroundStartT8"
    )
    orchestrator = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3LongVideoOrchestratorT8"
    )
    assert start["widgets_values"][0] == orchestrator["widgets_values"][0]
    assert start["widgets_values"][0] == "h3_background_scene_identity_demo"
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(links)

    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_background_control_routes_offload_blocking_manager_calls():
    source = (
        Path(__file__).resolve().parents[1] / "long_video_routes.py"
    ).read_text(encoding="utf-8")
    assert "await asyncio.to_thread(BACKGROUND_JOBS.pause, chain_id)" in source
    assert "await asyncio.to_thread(BACKGROUND_JOBS.resume, chain_id)" in source
    assert "await asyncio.to_thread(BACKGROUND_JOBS.cancel, chain_id)" in source


def test_multirate_exp_example_is_independent_and_uses_eight_joint_calls():
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "api" / "multirate_exp_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    exp_nodes = [
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3MultiRateSamplerEXPT8"
    ]
    assert len(exp_nodes) == 1
    assert exp_nodes[0]["inputs"]["video_steps"] == 4
    assert exp_nodes[0]["inputs"]["audio_steps"] == 8
    assert not any(
        value["class_type"] in {
            "MiniMaxH3DualClockSamplerT8",
            "MiniMaxH3SigmaShift",
            "KSamplerSelect",
            "BasicScheduler",
        }
        for value in workflow.values()
    )
    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_still_image_edit_example_uses_ref2va_without_incompatible_lora():
    path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "api" / "still_image_edit_api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    types = {value["class_type"] for value in workflow.values()}
    assert "MiniMaxH3StillConditioningT8" in types
    assert "MiniMaxH3StillPreflightT8" in types
    assert "MiniMaxH3StillDecodeT8" in types
    assert "MiniMaxH3DualClockSamplerT8" in types
    assert not any("Lora" in node_type for node_type in types)

    unet = next(value for value in workflow.values() if value["class_type"] == "UNETLoader")
    assert "ref2va" in unet["inputs"]["unet_name"]
    conditioning = next(
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3StillConditioningT8"
    )
    assert conditioning["inputs"]["target_mode"] == "short_video_22_frames"
    assert conditioning["inputs"]["canvas_mode"] == "custom"
    assert conditioning["inputs"]["width"] == 512
    assert conditioning["inputs"]["height"] == 512
    sampler = next(
        value for value in workflow.values()
        if value["class_type"] == "MiniMaxH3DualClockSamplerT8"
    )
    assert sampler["inputs"]["steps"] == 20

    node_ids = set(workflow)
    for node in workflow.values():
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in node_ids


def test_frontend_workflows_cover_stable_and_both_exp_step_counts():
    workflow_dir = Path(__file__).resolve().parents[1] / "examples" / "workflows" / "01-basic-generation"
    expected = {
        "2026-08-06_H3_Turbo_Stable_4V4A.json": ("MiniMaxH3DualClockSamplerT8", [4, 12.0, 3.0]),
        "2026-08-06_H3_Turbo_EXP_4V8A.json": ("MiniMaxH3MultiRateSamplerEXPT8", [4, 8, 12.0, 3.0]),
        "2026-08-06_H3_Turbo_EXP_4V10A.json": ("MiniMaxH3MultiRateSamplerEXPT8", [4, 10, 12.0, 3.0]),
    }

    for filename, (sampler_type, sampler_widgets) in expected.items():
        workflow = json.loads((workflow_dir / filename).read_text(encoding="utf-8"))
        assert workflow["version"] == 0.4
        assert workflow["last_node_id"] == max(node["id"] for node in workflow["nodes"])
        assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])

        nodes = {node["id"]: node for node in workflow["nodes"]}
        types = {node["type"] for node in nodes.values()}
        assert "LoraLoaderBypassModelOnly" in types
        assert "MiniMaxH3SigmaShift" not in types
        assert "BasicScheduler" not in types
        assert "KSamplerSelect" not in types

        sampler_nodes = [node for node in nodes.values() if node["type"] == sampler_type]
        assert len(sampler_nodes) == 1
        assert sampler_nodes[0]["widgets_values"] == sampler_widgets

        unet = next(node for node in nodes.values() if node["type"] == "UNETLoader")
        assert unet["widgets_values"][0] == "minimax_h3_fl2va_int8_convrot.safetensors"
        lora = next(
            node for node in nodes.values() if node["type"] == "LoraLoaderBypassModelOnly"
        )
        assert lora["widgets_values"][0] == (
            "minimax_h3_turbo_4步加速ema_comfyui.safetensors"
        )

        for link_id, source_id, source_slot, target_id, target_slot, link_type in workflow["links"]:
            source = nodes[source_id]
            target = nodes[target_id]
            assert link_id in source["outputs"][source_slot]["links"]
            assert target["inputs"][target_slot]["link"] == link_id
            assert source["outputs"][source_slot]["type"] == link_type
            assert target["inputs"][target_slot]["type"] == link_type


def test_frontend_audio_input_workflows_cover_three_source_modes_and_output_routing():
    workflow_dir = Path(__file__).resolve().parents[1] / "examples" / "workflows" / "02-audio-control"
    expected = {
        "2026-08-06_H3_Audio_Lock_Source_Stable_4V4A.json": ("lock_source", 6),
        "2026-08-06_H3_Audio_Remix_Source_Stable_4V4A.json": ("remix_source", 11),
        "2026-08-06_H3_Audio_Reference_Only_Stable_4V4A.json": ("reference_only", 11),
    }

    for filename, (audio_mode, final_audio_source_id) in expected.items():
        workflow = json.loads((workflow_dir / filename).read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}
        conditioning = next(
            node for node in nodes.values()
            if node["type"] == "MiniMaxH3AudioConditioningT8"
        )
        audio_window = next(
            node for node in nodes.values()
            if node["type"] == "MiniMaxH3AudioWindowT8"
        )
        output_trim = next(
            node for node in nodes.values()
            if node["type"] == "MiniMaxH3OutputTrimT8"
        )
        sampler = next(
            node for node in nodes.values()
            if node["type"] == "MiniMaxH3DualClockSamplerT8"
        )
        conditioning_inputs = {
            value["name"]: value for value in conditioning["inputs"]
        }

        assert workflow["version"] == 0.4
        assert workflow["last_node_id"] == max(nodes)
        assert workflow["last_link_id"] == max(links)
        assert conditioning["widgets_values"][1:3] == [736, 416]
        assert conditioning["widgets_values"][5] == audio_mode
        assert "<Audio 1>" in conditioning["widgets_values"][0]
        assert sampler["widgets_values"] == [
            4, 12.0, 3.0, "dual_clock_euler", "native_flow",
        ]
        assert links[conditioning_inputs["drive_audio"]["link"]][1:5] == [
            audio_window["id"], 0, conditioning["id"], 15,
        ]
        assert links[conditioning_inputs["length"]["link"]][1:5] == [
            audio_window["id"], 1, conditioning["id"], 6,
        ]
        final_audio_link = links[
            next(value for value in output_trim["inputs"] if value["name"] == "audio")["link"]
        ]
        assert final_audio_link[1] == final_audio_source_id
        assert final_audio_link[2] == (2 if audio_mode == "lock_source" else 1)

        for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
            assert nodes[target]["inputs"][input_slot]["link"] == link_id
            assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
            assert nodes[source]["outputs"][output_slot]["type"] == link_type
            assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_frontend_still_edit_workflow_uses_native_22_frame_ref2va_target():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "workflows"
        / "03-image-video-edit"
        / "2026-08-07_H3_Still_Edit_22Frames_EXP.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))
    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(node["id"] for node in workflow["nodes"])
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])

    nodes = {node["id"]: node for node in workflow["nodes"]}
    types = {node["type"] for node in nodes.values()}
    assert {
        "MiniMaxH3StillConditioningT8",
        "MiniMaxH3StillPreflightT8",
        "MiniMaxH3StillDecodeT8",
        "MiniMaxH3DualClockSamplerT8",
        "SaveImage",
    } <= types
    assert not any("Lora" in node_type for node_type in types)

    unet = next(node for node in nodes.values() if node["type"] == "UNETLoader")
    assert unet["widgets_values"][0] == (
        "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    )
    conditioning = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3StillConditioningT8"
    )
    assert conditioning["widgets_values"][1:7] == [
        "custom",
        512,
        512,
        "short_video_22_frames",
        0.999,
        "generate_and_discard",
    ]
    sampler = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3DualClockSamplerT8"
    )
    assert sampler["widgets_values"] == [20, 12.0, 3.0]

    for link_id, source_id, source_slot, target_id, target_slot, link_type in workflow["links"]:
        source = nodes[source_id]
        target = nodes[target_id]
        assert link_id in source["outputs"][source_slot]["links"]
        assert target["inputs"][target_slot]["link"] == link_id
        assert (
            source["outputs"][source_slot]["type"] == link_type
            or link_type == "*"
        )
        assert target["inputs"][target_slot]["type"] == link_type


def test_dialogue_safe_master_examples_require_verified_independent_stems():
    root = Path(__file__).resolve().parents[1]
    api = json.loads((root / "tests" / "fixtures" / "api" / "dialogue_safe_master_api.json").read_text(
        encoding="utf-8"
    ))
    analyzer = next(
        node for node in api.values()
        if node["class_type"] == "MiniMaxH3DialogueBoundaryAnalyzerT8"
    )
    master = next(
        node for node in api.values()
        if node["class_type"] == "MiniMaxH3DialogueSafeMasterT8"
    )
    analyzer_id = next(key for key, value in api.items() if value is analyzer)
    assert master["inputs"]["speech_accepted"] == [analyzer_id, 2]
    assert master["inputs"]["music_fit_policy"] == "strict"
    assert master["inputs"]["ambience_fit_policy"] == "strict"
    assert master["inputs"]["sfx_fit_policy"] == "strict"
    assert master["inputs"]["target_duration_seconds"] == 10.0
    load_titles = {
        node.get("_meta", {}).get("title", "")
        for node in api.values()
        if node["class_type"] == "LoadAudio"
    }
    assert any("independent speech stem" in title for title in load_titles)
    assert any("music stem" in title for title in load_titles)
    assert any("ambience stem" in title for title in load_titles)
    assert any("SFX stem" in title for title in load_titles)

    frontend = json.loads(
        (root / "examples" / "workflows" / "05-speech-dialogue" / "2026-08-10_H3_Dialogue_Safe_Master_EXP.json")
        .read_text(encoding="utf-8")
    )
    nodes = {node["id"]: node for node in frontend["nodes"]}
    assert frontend["last_node_id"] == max(nodes)
    assert frontend["last_link_id"] == max(link[0] for link in frontend["links"])
    assert {
        "MiniMaxH3DialogueBoundaryAnalyzerT8",
        "MiniMaxH3DialogueSafeMasterT8",
    } <= {node["type"] for node in nodes.values()}
    for link_id, source, output_slot, target, input_slot, link_type in frontend["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


def test_timed_background_bed_example_is_opt_in_and_routes_locked_latent_twice():
    root = Path(__file__).resolve().parents[1]
    api = json.loads((root / "tests" / "fixtures" / "api" / "dialogue_timed_bed_lock_api.json").read_text(
        encoding="utf-8"
    ))
    ids_by_type = {
        value["class_type"]: key
        for key, value in api.items()
    }
    timed_id = ids_by_type["MiniMaxH3TimedAudioBedLockT8"]
    timed = api[timed_id]
    sampler_setup = api[ids_by_type["MiniMaxH3DualClockSamplerT8"]]
    sampler = api[ids_by_type["SamplerCustomAdvanced"]]
    boundary_id = ids_by_type["PrimitiveFloat"]

    assert timed["inputs"]["tail_lock_start_seconds"] == [boundary_id, 0]
    assert timed["inputs"]["tail_denoise_strength"] == 0.0
    assert timed["inputs"]["transition_seconds"] == 0.0
    assert timed["inputs"]["audio_latent_fit_policy"] == "fit_reported"
    assert sampler_setup["inputs"]["av_latent"] == [timed_id, 0]
    assert sampler["inputs"]["latent_image"] == [timed_id, 0]
    assert sampler_setup["inputs"]["steps"] == 4

    frontend = json.loads(
        (root / "examples" / "workflows" / "05-speech-dialogue" / "2026-08-09_H3_Dialogue_Timed_Background_Bed_Lock_EXP.json")
        .read_text(encoding="utf-8")
    )
    nodes = {node["id"]: node for node in frontend["nodes"]}
    timed_frontend = next(
        node for node in nodes.values()
        if node["type"] == "MiniMaxH3TimedAudioBedLockT8"
    )
    assert timed_frontend["widgets_values"] == [1.0, 0.0, 0.0, "fit_reported"]
    assert frontend["last_node_id"] == max(nodes)
    assert frontend["last_link_id"] == max(link[0] for link in frontend["links"])
    for link_id, source, output_slot, target, input_slot, link_type in frontend["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type


@pytest.mark.parametrize(
    ("filename", "advanced_type"),
    [
        (
            "2026-08-18_H3_Hanfu_Tail_Detail_3Step_Advanced_EXP.json",
            "MiniMaxH3AVTailDetailScheduleT8Advanced",
        ),
        (
            "2026-08-18_H3_Hanfu_Model_Time_Bias_Advanced_EXP.json",
            "MiniMaxH3ModelTimeBiasSamplerT8Advanced",
        ),
        (
            "2026-08-18_H3_Hanfu_RF_Restart_Advanced_EXP.json",
            "MiniMaxH3RectifiedFlowRestartSamplerT8Advanced",
        ),
        (
            "2026-08-18_H3_Hanfu_STG_Advanced_EXP.json",
            "MiniMaxH3SpatioTemporalGuidanceT8Advanced",
        ),
        (
            "2026-08-18_H3_Hanfu_Temporal_Detail_Advanced_EXP.json",
            "MiniMaxH3TemporalDetailEnhanceT8Advanced",
        ),
        (
            "2026-08-18_H3_Hanfu_Detail_Mixer_Advanced_EXP.json",
            "MiniMaxH3DetailMixerSamplerT8Advanced",
        ),
    ],
)
def test_h3_detail_advanced_frontend_examples_are_importable_and_documented(
    filename,
    advanced_type,
):
    root = Path(__file__).resolve().parents[1]
    workflow = json.loads(
        (root / "examples" / "workflows" / "07-motion-detail" / filename).read_text(encoding="utf-8")
    )
    nodes = {node["id"]: node for node in workflow["nodes"]}
    types = {node["type"] for node in nodes.values()}

    assert workflow["version"] == 0.4
    assert workflow["last_node_id"] == max(nodes)
    assert workflow["last_link_id"] == max(link[0] for link in workflow["links"])
    assert advanced_type in types
    assert "MarkdownNote" in types
    assert (
        "MiniMaxH3DualClockSamplerT8" in types
        or advanced_type
        in {
            "MiniMaxH3ModelTimeBiasSamplerT8Advanced",
            "MiniMaxH3RectifiedFlowRestartSamplerT8Advanced",
            "MiniMaxH3DetailMixerSamplerT8Advanced",
        }
    )
    assert "MiniMaxH3AudioConditioningT8" in types
    note = next(node for node in nodes.values() if node["type"] == "MarkdownNote")
    assert len(note["widgets_values"][0]) >= 100
    if advanced_type == "MiniMaxH3DetailMixerSamplerT8Advanced":
        mixer = next(node for node in nodes.values() if node["type"] == advanced_type)
        assert mixer["widgets_values"] == [
            8,
            12.0,
            3.0,
            True,
            1,
            "video_sigma_linear",
            "turbo_standard8",
            True,
            -0.025,
            0.7,
            0.95,
            "video_sigma",
            True,
            0.35,
            "25",
            0.25,
            0.85,
            False,
            0.15,
            3,
            2608183001,
        ]
        assert sum(node["type"] == "MarkdownNote" for node in nodes.values()) == 4
        assert "MiniMaxH3TemporalDetailEnhanceT8Advanced" in types

    for link_id, source, output_slot, target, input_slot, link_type in workflow["links"]:
        assert nodes[target]["inputs"][input_slot]["link"] == link_id
        assert link_id in (nodes[source]["outputs"][output_slot].get("links") or [])
        assert nodes[source]["outputs"][output_slot]["type"] == link_type
        assert nodes[target]["inputs"][input_slot]["type"] == link_type
