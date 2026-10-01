"""Opt-in stage nodes. Legacy algorithms and workflow contracts remain intact."""


def node_classes():
    from .legacy_pass_through_nodes import (
        MiniMaxH3NativeLatentCheckpointPassThroughSaveEXPT8,
        MiniMaxH3MotionOverloadAnalyzePassThroughEXPT8,
    )
    from .pass_through_audit_nodes import (
        MiniMaxH3MotionStageAuditPassThroughEXPT8,
        MiniMaxH3StageEAVAuditPassThroughEXPT8,
    )
    from .fast_h3_v2_handoff_nodes import NODES as fast_h3_v2_handoff_nodes
    from .fast_h3_v2_continuation_nodes import NODES as fast_h3_v2_continuation_nodes
    from .fast_h3_v2_job_nodes import NODES as fast_h3_v2_job_nodes
    from .stage_unet_loader_nodes import NODES as stage_unet_loader_nodes
    from .progressive_nodes import NODES as progressive_nodes
    from .progressive_effect_nodes import NODES as progressive_effect_nodes
    from .avatar_nodes import NODES as avatar_nodes
    from .continuation_nodes import NODES as continuation_nodes
    from .hyperflow_nodes import NODES as hyperflow_nodes
    from .hyperflow_storage_nodes import NODES as hyperflow_storage_nodes
    from .hyperflow_effect_nodes import NODES as hyperflow_effect_nodes
    from .hyperflow_fresh_nodes import NODES as hyperflow_fresh_nodes
    from .hyperflow_p7_nodes import NODES as hyperflow_p7_nodes
    from .hyperflow_p7_storage_nodes import NODES as hyperflow_p7_storage_nodes
    from .hyperflow_p7_effect_nodes import NODES as hyperflow_p7_effect_nodes
    from .hyperflow_p7_delivery_nodes import NODES as hyperflow_p7_delivery_nodes
    from .speed_nodes import NODES as speed_nodes
    from .speed_storage_nodes import NODES as speed_storage_nodes
    from .chunked_source_nodes import NODES as chunked_source_nodes
    from .chunked_stage_nodes import NODES as chunked_stage_nodes
    from .chunked_v5_nodes import NODES as chunked_v5_nodes
    from .h16_nodes import NODES as h16_nodes
    from .h16_native_source_nodes import MiniMaxH3H16VerifiedNativeSourceEXPT8
    from .face_nodes import NODES as face_nodes
    from .face_nodes import MiniMaxH3FaceRelayBindEXPT8, MiniMaxH3FaceLocalRelayBindEXPT8
    from .motion_nodes import NODES as motion_nodes
    from .motion_storage_nodes import MiniMaxH3MotionFrozenFirstPassLoadEXPT8
    from .audio_refine_nodes import NODES as audio_refine_nodes
    from .audio_refine_storage_nodes import NODES as audio_refine_storage_nodes
    from .ltx_rgb_nodes import NODES as ltx_rgb_nodes
    from .prepared_ltx_nodes import NODES as prepared_ltx_nodes
    from .ltx_latent_nodes import NODES as ltx_latent_nodes
    from .ltx_latent_sample_nodes import NODES as ltx_latent_sample_nodes
    from .chunked_v1_relay_nodes import NODES as chunked_v1_relay_nodes
    from .chunked_v1_storage_nodes import NODES as chunked_v1_storage_nodes
    from .chunked_v5_storage_nodes import NODES as chunked_v5_storage_nodes
    from .nodes import (MiniMaxH3FastH3V2StageSetupEXPT8, MiniMaxH3StageEAVConfigEXPT8,
                        MiniMaxH3StageEAVApplyEXPT8, MiniMaxH3StageEAVAuditEXPT8,
                        MiniMaxH3StageSamplerEXPT8, MiniMaxH3StageSaveEXPT8, MiniMaxH3StageLoadEXPT8,
                        MiniMaxH3NativeDualStageSetupEXPT8, MiniMaxH3NativeDualHandoffEXPT8,
                        MiniMaxH3ManualPassStageSetupEXPT8, MiniMaxH3StageNoiseEXPT8,
                        MiniMaxH3RFBaseStageSetupEXPT8, MiniMaxH3RFHandoffEXPT8, MiniMaxH3RFRestartStageSetupEXPT8,
                        MiniMaxH3NativeStageBindEXPT8, MiniMaxH3PDDStageSetupEXPT8, MiniMaxH3VDNStageSetupEXPT8,
                        MiniMaxH3VDNRelayApplyEXPT8, MiniMaxH3VDNRelayAuditEXPT8)
    return [MiniMaxH3FastH3V2StageSetupEXPT8, MiniMaxH3StageEAVConfigEXPT8,
            MiniMaxH3StageEAVApplyEXPT8, MiniMaxH3StageEAVAuditEXPT8,
            MiniMaxH3StageSamplerEXPT8, MiniMaxH3StageSaveEXPT8, MiniMaxH3StageLoadEXPT8,
            MiniMaxH3NativeDualStageSetupEXPT8, MiniMaxH3NativeDualHandoffEXPT8,
            MiniMaxH3ManualPassStageSetupEXPT8, MiniMaxH3StageNoiseEXPT8,
            MiniMaxH3RFBaseStageSetupEXPT8, MiniMaxH3RFHandoffEXPT8, MiniMaxH3RFRestartStageSetupEXPT8,
            MiniMaxH3NativeStageBindEXPT8, MiniMaxH3PDDStageSetupEXPT8, MiniMaxH3VDNStageSetupEXPT8,
            MiniMaxH3VDNRelayApplyEXPT8, MiniMaxH3VDNRelayAuditEXPT8, *progressive_nodes, *progressive_effect_nodes,
            *avatar_nodes, *continuation_nodes, *hyperflow_nodes, *hyperflow_storage_nodes, *hyperflow_effect_nodes,
            *hyperflow_fresh_nodes, *hyperflow_p7_nodes, *hyperflow_p7_storage_nodes,
            *hyperflow_p7_effect_nodes, *hyperflow_p7_delivery_nodes, *speed_nodes, *speed_storage_nodes,
            *chunked_source_nodes, *chunked_stage_nodes, *chunked_v5_nodes, *h16_nodes, *face_nodes,
            *motion_nodes, *audio_refine_nodes, *ltx_rgb_nodes, *prepared_ltx_nodes,
            *ltx_latent_nodes, *fast_h3_v2_handoff_nodes, *stage_unet_loader_nodes,
            *audio_refine_storage_nodes, *chunked_v1_relay_nodes, *chunked_v1_storage_nodes,
            *chunked_v5_storage_nodes, MiniMaxH3H16VerifiedNativeSourceEXPT8,
            *ltx_latent_sample_nodes, MiniMaxH3MotionFrozenFirstPassLoadEXPT8,
            *fast_h3_v2_continuation_nodes, *fast_h3_v2_job_nodes,
            MiniMaxH3FaceRelayBindEXPT8, MiniMaxH3FaceLocalRelayBindEXPT8,
            # Add only after the complete legacy and modular registration prefix.
            MiniMaxH3NativeLatentCheckpointPassThroughSaveEXPT8,
            MiniMaxH3MotionOverloadAnalyzePassThroughEXPT8,
            MiniMaxH3MotionStageAuditPassThroughEXPT8,
            MiniMaxH3StageEAVAuditPassThroughEXPT8]
