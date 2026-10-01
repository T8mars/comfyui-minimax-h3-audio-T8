"""Complete rollout scope, not a generic sampler or a qualification claim.

Each row names actual source symbols and distinct numerical handoffs. Existing
public stages remain available even where this rollout has not qualified them.
This module is inert stdlib data: no Core import, weight load or sampling.
"""
from dataclasses import dataclass


DIMENSIONS = ("public_stages", "editable_workflow", "external_eav", "external_relay",
              "stage_resume", "legacy_regression", "gpu_mechanical", "human_review")


@dataclass(frozen=True)
class Source:
    path: str
    symbol: str


@dataclass(frozen=True)
class Route:
    id: str
    phase: str
    name: str
    stages: tuple[str, ...]
    handoff: str
    variants: tuple[str, ...]
    sources: tuple[Source, ...]
    public_nodes: tuple[str, ...] = ()
    repeated_stages: bool = False


def src(module, symbol):
    return Source("h3_t8/" + module + ".py", symbol)


def learned(symbol):
    return src("learned_latent_upscale_advanced", symbol)


DUAL = src("long_video_dual_model_runner", "DualModelSegmentRunner._stage_sampling")
DUAL_HANDOFF = src("long_video_dual_model_runner", "DualModelSegmentRunner._reconcile")
UPSCALE = learned("learned_upscale_h3_av_latent")
RECONCILE = learned("reconcile_two_pass_h3_latent")
PARITY = learned("build_learned_two_pass_parity_plan")
CHUNKED = src("chunked_two_pass_upscale_advanced", "execute_chunked_two_pass_upscale")
CHUNKED_SOURCE_NODE = "MiniMaxH3ChunkedSourceSegmentEXPT8"
CHUNKED_SOURCE = (
    src("modular_sampling/chunked_source", "slice_chunked_source"),
    src("modular_sampling/chunked_source_nodes", CHUNKED_SOURCE_NODE + ".execute"),
)
CHUNKED_STAGE_NODES = ("MiniMaxH3ChunkedPass2PrepareEXPT8",
                       "MiniMaxH3ChunkedLearnedLiftEXPT8",
                       "MiniMaxH3ChunkedPass2SegmentEXPT8")
CHUNKED_STAGES = (
    *(src("modular_sampling/chunked_stages", name) for name in
      ("prepare_chunked_pass2", "lift_chunked_segment", "sample_chunked_pass2")),
    *(src("modular_sampling/chunked_stage_nodes", node + ".execute")
      for node in CHUNKED_STAGE_NODES),
)
CHUNKED_EAV_NODES = ("MiniMaxH3ChunkedPass2EAVApplyEXPT8",
                     "MiniMaxH3ChunkedPass2EAVAuditEXPT8")
CHUNKED_EAV = (
    *(src("modular_sampling/chunked_effects", name) for name in
      ("bind_chunked_eav", "validate_stage", "audit_chunked_eav")),
    *(src("modular_sampling/chunked_stage_nodes", node + ".execute")
      for node in CHUNKED_EAV_NODES),
)
CHUNKED_RELAY_NODES = ("MiniMaxH3ChunkedPass2RelayBindEXPT8",
                       "MiniMaxH3ChunkedPass2RelayAuditEXPT8")
CHUNKED_RELAY = (
    *(src("modular_sampling/chunked_relay", name) for name in
      ("bind_full_clip_relay", "assert_relay_binding", "audit_full_clip_relay")),
    *(src("modular_sampling/chunked_stage_nodes", node + ".execute")
      for node in CHUNKED_RELAY_NODES),
)
CHUNKED_V1_LOCAL_RELAY_NODES = ("MiniMaxH3ChunkedV1RelayProjectEXPT8",
                                "MiniMaxH3ChunkedV1RelayAuditEXPT8")
CHUNKED_V1_LOCAL_RELAY = (
    *(src("modular_sampling/chunked_v1_relay", name) for name in
      ("project_v1_local_relay", "assert_v1_local_relay_binding", "audit_v1_local_relay")),
    *(src("modular_sampling/chunked_v1_relay_nodes", node + ".execute")
      for node in CHUNKED_V1_LOCAL_RELAY_NODES),
    *(Source("tools/build_modular_chunked_v1_relay_workflow.py", name) for name in
      ("add_v1_relay_frontend", "add_v1_relay_api", "build_pair")),
)
CHUNKED_V1_STORAGE_NODES = ("MiniMaxH3ChunkedV1SegmentSaveEXPT8",
                            "MiniMaxH3ChunkedV1SegmentLoadEXPT8")
CHUNKED_V1_STORAGE = (
    *(src("modular_sampling/chunked_v1_storage", name) for name in
      ("verify_segment", "save_segment", "load_segment")),
    *(src("modular_sampling/chunked_v1_storage_nodes", node + ".execute")
      for node in CHUNKED_V1_STORAGE_NODES),
    *(Source("tools/build_modular_chunked_v1_storage_workflow.py", name) for name in
      ("freeze_frontend", "resume_frontend", "freeze_api", "resume_api", "build_storage_pair")),
)
CHUNKED_V5_NODES = ("MiniMaxH3ChunkedV5GlobalLiftEXPT8",
                    "MiniMaxH3ChunkedV5PrepareEXPT8",
                    "MiniMaxH3ChunkedV5PASS2WindowEXPT8",
                    "MiniMaxH3ChunkedV5EAVApplyEXPT8",
                    "MiniMaxH3ChunkedV5EAVAuditEXPT8",
                    "MiniMaxH3ChunkedV5RelayProjectEXPT8",
                    "MiniMaxH3ChunkedV5RelayAuditEXPT8",
                    "MiniMaxH3ChunkedV5WindowSaveEXPT8",
                    "MiniMaxH3ChunkedV5WindowLoadEXPT8",
                    "MiniMaxH3ChunkedV5VerifiedNativeSourceEXPT8")
CHUNKED_V5 = (
    *(src("modular_sampling/chunked_v5", name) for name in
      ("lift_standard_joint", "prepare_standard_joint", "sample_standard_window")),
    *(src("modular_sampling/chunked_v5_effects", name) for name in
      ("bind_v5_eav", "audit_v5_eav")),
    *(src("modular_sampling/chunked_v5_relay", name) for name in
      ("project_v5_relay", "audit_v5_relay")),
    *(src("modular_sampling/chunked_v5_storage", name) for name in
      ("verify_window", "save_window", "load_window")),
    src("modular_sampling/chunked_v5_native_source", "verified_native_source"),
    *(src("modular_sampling/chunked_v5_nodes", node + ".execute")
      for node in CHUNKED_V5_NODES[:-3]),
    *(src("modular_sampling/chunked_v5_storage_nodes", node + ".execute")
      for node in CHUNKED_V5_NODES[-3:]),
)
CHUNKED_V5_WORKFLOW = (
    Source("tools/build_modular_chunked_v5_workflow.py", "split_frontend"),
    Source("tools/build_modular_chunked_v5_workflow.py", "split_api"),
    Source("tools/build_modular_chunked_v5_workflow.py", "add_eav_frontend"),
    Source("tools/build_modular_chunked_v5_workflow.py", "add_eav_api"),
    Source("tools/build_modular_chunked_v5_workflow.py", "add_relay_frontend"),
    Source("tools/build_modular_chunked_v5_workflow.py", "add_relay_api"),
    Source("tools/build_modular_chunked_v5_storage_workflow.py", "freeze_frontend"),
    Source("tools/build_modular_chunked_v5_storage_workflow.py", "resume_frontend"),
    Source("tools/build_modular_chunked_v5_storage_workflow.py", "candidate_api"),
    Source("tools/build_modular_chunked_v5_storage_workflow.py", "build_pair"),
)
H16_SPLIT_NODES = ("MiniMaxH3H16Pass2PlanEXPT8",
                   "MiniMaxH3H16Pass2WindowEXPT8",
                   "MiniMaxH3H16EAVApplyEXPT8",
                   "MiniMaxH3H16EAVAuditEXPT8",
                   "MiniMaxH3H16RelayProjectEXPT8",
                   "MiniMaxH3H16RelayAuditEXPT8",
                   "MiniMaxH3H16WindowSaveEXPT8",
                   "MiniMaxH3H16WindowLoadEXPT8")
H16_SPLIT = (
    *(src("modular_sampling/h16_stages", name)
      for name in ("build_h16_plan", "h16_window_piece", "sample_h16_pass2")),
    *(src("modular_sampling/h16_effects", name)
      for name in ("bind_h16_eav", "audit_h16_eav")),
    *(src("modular_sampling/h16_relay", name)
      for name in ("project_h16_relay", "assert_h16_relay_binding", "audit_h16_relay")),
    *(src("modular_sampling/h16_storage", name)
      for name in ("verify_window", "save_window", "load_window")),
    *(src("modular_sampling/h16_nodes", node + ".execute")
      for node in H16_SPLIT_NODES),
    src("chunked_two_pass_upscale_advanced", "_spatial_resample"),
    Source("tools/build_modular_h16_workflow.py", "split_frontend"),
    Source("tools/build_modular_h16_workflow.py", "add_eav_frontend"),
    Source("tools/build_modular_h16_workflow.py", "add_relay_frontend"),
    Source("tools/build_modular_h16_workflow.py", "split_api"),
)
CHUNKED_WORKFLOW = (
    Source("tools/build_modular_chunked_workflow.py", "split_workflow"),
    Source("tools/build_modular_chunked_workflow.py", "split_api_graph"),
    Source("tools/build_modular_chunked_workflow.py", "add_external_eav_frontend"),
    Source("tools/build_modular_chunked_workflow.py", "add_external_eav_api"),
    Source("tools/build_modular_chunked_workflow.py", "add_external_relay_frontend"),
    Source("tools/build_modular_chunked_workflow.py", "add_external_relay_api"),
)
NATIVE_DUAL_STAGE = src("modular_sampling/native_dual", "build_stage")
NATIVE_DUAL_HANDOFF = src("modular_sampling/native_dual", "reconcile")
NATIVE_DUAL_NODES = ("MiniMaxH3NativeDualStageSetupEXPT8", "MiniMaxH3NativeDualHandoffEXPT8")
NATIVE_EXPLICIT = src("modular_sampling/native_explicit", "bind_stage")
NATIVE_EXPLICIT_NODE = "MiniMaxH3NativeStageBindEXPT8"
PROGRESSIVE_NODES = (
    "MiniMaxH3ProgressiveStagePlanEXPT8", "MiniMaxH3ProgressiveStageConditioningEXPT8",
    "MiniMaxH3ProgressiveLowStageEXPT8", "MiniMaxH3ProgressiveLiftInputEXPT8",
    "MiniMaxH3ProgressiveHighHandoffEXPT8", "MiniMaxH3ProgressiveHighStageEXPT8",
    "MiniMaxH3ProgressiveLowSaveEXPT8", "MiniMaxH3ProgressiveLowLoadEXPT8",
    "MiniMaxH3ProgressiveHighSaveEXPT8", "MiniMaxH3ProgressiveHighLoadEXPT8")
PROGRESSIVE_EFFECT_NODES = (
    "MiniMaxH3ProgressiveRelayStageApplyEXPT8", "MiniMaxH3ProgressiveLowEAVApplyEXPT8",
    "MiniMaxH3ProgressiveHighEAVApplyEXPT8", "MiniMaxH3ProgressiveLowEffectsAuditEXPT8",
    "MiniMaxH3ProgressiveHighEffectsAuditEXPT8")
AVATAR_NODES = ("MiniMaxH3AvatarSourceBindEXPT8", "MiniMaxH3AvatarLowStageEXPT8",
                "MiniMaxH3AvatarHighHandoffEXPT8", "MiniMaxH3AvatarDeliveryAuditEXPT8")
CONTINUATION_NODES = ("MiniMaxH3ContinuationSourceEXPT8", "MiniMaxH3ContinuationContextsEXPT8",
    "MiniMaxH3ContinuationPlanEXPT8", "MiniMaxH3ContinuationConditioningEXPT8",
    "MiniMaxH3ContinuationLowStageEXPT8", "MiniMaxH3ContinuationHighHandoffEXPT8", "MiniMaxH3ContinuationHighStageEXPT8",
    "MiniMaxH3ContinuationRelayProjectEXPT8", "MiniMaxH3ContinuationRelayApplyEXPT8",
    "MiniMaxH3ContinuationLowEAVApplyEXPT8", "MiniMaxH3ContinuationHighEAVApplyEXPT8", "MiniMaxH3ContinuationDeliveryEXPT8")
HYPERFLOW_FRESH_NODES = tuple("MiniMaxH3HyperFlowFresh" + name + "EXPT8"
                             for name in ("StageSetup", "LiftInput", "StageAudit", "StageLoad", "Loader"))
HYPERFLOW_FRESH_SOURCES = (
    *(src("modular_sampling/hyperflow_fresh", name) for name in ("install", "build_stage", "execution", "lift_input", "verify_receipt")),
    *(src("modular_sampling/hyperflow_fresh_nodes", node + ".execute") for node in HYPERFLOW_FRESH_NODES),
    src("modular_sampling/results", "sample_stage"), src("modular_sampling/storage", "save_stage"),
    src("modular_sampling/storage", "load_stage"), Source("tools/build_modular_hyperflow_fresh_workflow.py", "split_graph"))
HYPERFLOW_P7_NODES = ("MiniMaxH3HyperFlowP7AcceptedParentEXPT8",
    "MiniMaxH3HyperFlowP7PrepareContextsEXPT8", "MiniMaxH3HyperFlowP7ConditioningEXPT8",
    "MiniMaxH3HyperFlowP7InitialSegmentEXPT8", "MiniMaxH3HyperFlowP7LowSetupEXPT8",
    "MiniMaxH3HyperFlowP7LowResultEXPT8", "MiniMaxH3HyperFlowP7LearnedLiftEXPT8",
    "MiniMaxH3HyperFlowP7HighHandoffEXPT8", "MiniMaxH3HyperFlowP7HighSetupEXPT8",
    "MiniMaxH3HyperFlowP7HighResultEXPT8")
HYPERFLOW_P7_STORAGE_NODES = ("MiniMaxH3HyperFlowP7LowSaveEXPT8",
    "MiniMaxH3HyperFlowP7LowLoadEXPT8", "MiniMaxH3HyperFlowP7HighSaveEXPT8",
    "MiniMaxH3HyperFlowP7HighLoadEXPT8")
HYPERFLOW_P7_EFFECT_NODES = ("MiniMaxH3HyperFlowP7RelayProjectEXPT8",
    "MiniMaxH3HyperFlowP7RelayApplyEXPT8")
HYPERFLOW_P7_DELIVERY_NODES = ("MiniMaxH3HyperFlowP7CandidateSaveEXPT8",
    "MiniMaxH3HyperFlowP7CandidateAcceptEXPT8")
SPEED_STAGE_NODES = ("MiniMaxH3SPEEDStageSetupEXPT8", "MiniMaxH3SPEEDStageSampleEXPT8",
                     "MiniMaxH3SPEEDDCTTransitionEXPT8", "MiniMaxH3SPEEDRelayApplyEXPT8",
                     "MiniMaxH3SPEEDStageSaveEXPT8", "MiniMaxH3SPEEDStageLoadEXPT8")


ROUTES = (
    Route("S01", "M2", "Native base-flow split", ("coarse", "refine"),
          "prediction_x0 -> learned upscale -> rebuilt native AV template; base-flow restart", ("restart_base_noise",),
          (learned("build_two_pass_sigma_plan"), UPSCALE, RECONCILE, NATIVE_EXPLICIT),
          ("MiniMaxH3TwoPassSigmaPlanT8Advanced", NATIVE_EXPLICIT_NODE)),
    Route("S02", "M2", "Published LBH parity", ("coarse", "refine"),
          "prediction_x0 -> learned upscale; independent published raw-video HIGH sigma table", ("4+3", "4+4", "4+5"),
          (PARITY, UPSCALE, RECONCILE, NATIVE_EXPLICIT),
          ("MiniMaxH3LearnedTwoPassParityPlanT8Advanced", NATIVE_EXPLICIT_NODE)),
    Route("S03", "M2", "Complete first trajectory then native tail refinement", ("complete_first", "native_refine"),
          "completed AV -> learned upscale -> fresh tail noise; not partial4+4", ("complete8", "stock20", "vdn_to_native_tail"),
            (DUAL, PARITY, Source("tools/build_vdn_two_pass_workflows.py", "build_prompt"), NATIVE_EXPLICIT,
             Source("tools/build_modular_vdn_native_workflow.py", "split_graph"),
             Source("tools/build_modular_vdn_relay_workflow.py", "split_graph")),
          ("MiniMaxH3DualClockSamplerT8", "MiniMaxH3LearnedTwoPassParityPlanT8Advanced", NATIVE_EXPLICIT_NODE)),
    Route("S04", "M2", "PDD learned4+4", ("low_0_4", "high_4_8"),
          "prediction_x0 -> learned upscale; dynamic head and original nine-point grid", ("FL2VA", "Ref2VA"),
          (src("pdd_advanced", "build_pdd_8step_setup"), src("pdd_advanced", "pdd_runtime_sigmas"), UPSCALE, RECONCILE,
           src("modular_sampling/pdd_stages", "build_stage"), src("modular_sampling/pdd_dynamic", "describe"),
           src("modular_sampling/pdd_dynamic", "project")), ("MiniMaxH3PDDStageSetupEXPT8",)),
    Route("S05", "M2", "OpenVDN complete plus native stage tail", ("complete_stage", "fresh_tail"),
          "completed AV -> learned upscale -> own VDN grid tail with fresh noise", ("dmd8", "stage_b50"),
          (src("vdn_two_pass", "setup_vdn_refine"), UPSCALE, src("modular_sampling/vdn_stages", "build_stage"),
             src("modular_sampling/vdn_identity", "inspect"), src("modular_sampling/vdn_identity", "weight_execution"),
             src("modular_sampling/vdn_effects", "wrap"),
             Source("tools/build_modular_vdn_workflow.py", "split_graph"),
             src("modular_sampling/vdn_relay", "apply"), src("modular_sampling/vdn_relay", "project"),
             src("modular_sampling/vdn_relay_math", "window"), src("modular_sampling/vdn_relay_math", "linear"),
             Source("tools/build_modular_vdn_relay_workflow.py", "split_graph")),
            ("MiniMaxH3VDNRefinePlanT8Advanced", "MiniMaxH3VDNStageSetupEXPT8",
             "MiniMaxH3VDNRelayApplyEXPT8", "MiniMaxH3VDNRelayAuditEXPT8")),
    Route("S06", "M2", "Dual MODEL partial LOW4", ("low4", "high_refine"),
          "LOW prediction_x0 -> learned upscale -> joint audio continuation and original prefix policy", ("4+3", "4+4", "4+5"),
          (DUAL, DUAL_HANDOFF, UPSCALE, src("long_video_dual_model_runner", "DualModelSegmentRunner.run"),
           NATIVE_DUAL_STAGE, NATIVE_DUAL_HANDOFF), NATIVE_DUAL_NODES),
    Route("S07", "M2", "Dual MODEL complete LOW20", ("low20", "high_refine"),
          "completed LOW audio; retain explicit complete-audio policy, independent HIGH grid", ("20+3", "20+4", "20+5"),
          (DUAL, DUAL_HANDOFF, UPSCALE, src("long_video_dual_model_runner", "DualModelSegmentRunner.__init__"),
           NATIVE_DUAL_STAGE, NATIVE_DUAL_HANDOFF), NATIVE_DUAL_NODES),
    Route("S08", "M1", "FastH3 V2 Dual learned4+4", ("low_0_4", "high_4_8"),
          "prediction_x0 -> existing learned/reconcile -> absolute DMD4:8; joint audio unfinished after LOW", ("dense_compat_exp", "trained_vsa_exp", "long_video"),
          (src("fast_h3_v2_advanced", "sample_v2_euler"), DUAL, src("modular_sampling/fast_h3_v2", "build_stage"),
           src("modular_sampling/fast_h3_v2_handoff", "completed_low_x0"),
           src("modular_sampling/fast_h3_v2_handoff_nodes", "MiniMaxH3FastH3V2CompletedLowX0EXPT8.execute"),
           src("modular_sampling/stage_unet_loader", "load_unet_after_stage"),
           src("modular_sampling/stage_unet_loader_nodes", "MiniMaxH3StageUNETLoaderAfterEXPT8.execute"),
           Source("tools/build_modular_fast_h3_v2_workflow.py", "split_graph_with_results"),
           Source("tools/build_modular_fast_h3_v2_workflow.py", "resume_high_graph")),
          ("MiniMaxH3FastH3V2StageSetupEXPT8", "MiniMaxH3FastH3V2CompletedLowX0EXPT8",
           "MiniMaxH3StageUNETLoaderAfterEXPT8")),
    Route("S09", "M2", "Manual second pass in both long-video runners", ("complete_first", "manual_second"),
          "complete first AV -> separate noise preparation -> manual tail; not tail_subdivide", ("plain_loop", "effects_loop"),
          (src("long_video_in_node_loop_advanced", "_sample_one_segment"),
           src("long_video_in_node_loop_effects_advanced", "_sample_prepared_segment"),
           src("long_video_in_node_loop_effects_advanced", "run_long_video_in_node_loop_effects"),
           src("long_video_sampling_plan_advanced", "resolve_long_video_sample_schedules"),
           src("modular_sampling/manual_pass", "build_stage"), src("modular_sampling/noise", "build_noise")),
          ("MiniMaxH3ManualPassStageSetupEXPT8", "MiniMaxH3StageNoiseEXPT8")),
    Route("S10", "M3", "Progressive single", ("low", "high_continuation"),
          "clean_video + audio_next; learned lift video without replacing audio_next", ("initialized", "standard"),
          (src("progressive_sampling_runtime", "sample_progressive_h3"), src("progressive_sampling_runtime", "_native_stage"),
           src("modular_sampling/progressive", "build_plan"), src("modular_sampling/progressive", "prepare_low_source"),
           src("modular_sampling/progressive", "sample_low"), src("modular_sampling/progressive", "lift_input"),
           src("modular_sampling/progressive", "prepare_high"), src("modular_sampling/progressive", "sample_high"),
           src("modular_sampling/progressive_storage", "save_boundary"), src("modular_sampling/progressive_storage", "load_boundary"),
           src("modular_sampling/progressive_high_result", "sample_high_result"),
           src("modular_sampling/progressive_high_result", "save_result"),
           src("modular_sampling/progressive_high_result", "load_result"),
             *(src("modular_sampling/progressive_nodes", name + ".execute") for name in PROGRESSIVE_NODES),
             *(src("modular_sampling/progressive_effect_nodes", name + ".execute") for name in PROGRESSIVE_EFFECT_NODES),
             *(src("modular_sampling/progressive_effects", name) for name in ("apply_relay", "apply_eav", "execution", "audit")),
             src("modular_sampling/progressive_model_identity", "native_model_identity"),
             src("modular_sampling/progressive_effect_identity", "project"),
             Source("tools/build_modular_progressive_effect_workflow.py", "split_graph"),
             Source("tools/build_modular_progressive_workflow.py", "split_graph")), PROGRESSIVE_NODES + PROGRESSIVE_EFFECT_NODES),
    Route("S11", "M3", "Progressive continuation long video", ("low", "high_continuation"),
          "typed continuation state; segment conditioning/accepted context outside stage execution", ("long_video", "prepared_continuation"),
          (src("progressive_continuation_runtime", "sample_progressive_continuation"), src("progressive_sampling_runtime", "sample_progressive_h3"),
           *(src("modular_sampling/continuation", name) for name in ("capture_source", "prepare_contexts", "prepare_phase",
               "build_plan", "sample_low", "prepare_high", "sample_high", "deliver")),
           *(src("modular_sampling/continuation_effects", name) for name in ("project_relay", "apply_relay", "apply_eav")),
           src("modular_sampling/continuation_identity", "project"),
           *(src("modular_sampling/continuation_nodes", name + ".execute") for name in CONTINUATION_NODES),
           Source("tools/build_modular_continuation_workflow.py", "split_graph")), CONTINUATION_NODES),
    Route("S12", "M3", "Avatar progressive", ("initialized_low", "initialized_high"),
          "Progressive clean_video + audio_next with bound native DriveAudio masks", ("avatar", "initialized_drive_audio"),
          (src("avatar_progressive_entry", "sample_avatar_progressive"), src("progressive_sampling_runtime", "sample_progressive_h3"),
           src("modular_sampling/progressive", "sample_low"), src("modular_sampling/progressive", "prepare_high"),
           src("modular_sampling/progressive", "sample_high"),
           src("modular_sampling/avatar", "bind_source"), src("modular_sampling/avatar", "sample_low"),
           src("modular_sampling/avatar", "prepare_high"), src("modular_sampling/avatar", "deliver"),
           Source("tools/build_modular_avatar_workflow.py", "split_graph")), AVATAR_NODES),
    Route("S13", "M3", "HyperFlow continuous split", ("head", "continuation_tail"),
          "raw model-space x_sigma, no new HIGH noise or learned upscale", ("1+7", "4+4", "7+1"),
          (src("hyperflow_two_pass_advanced", "sample_hyperflow_split"),
           src("modular_sampling/hyperflow", "sample_head"), src("modular_sampling/hyperflow", "sample_tail"),
           src("modular_sampling/hyperflow", "sample_tail_result"), src("modular_sampling/hyperflow", "ContinuousBoundary.verify"),
           src("modular_sampling/hyperflow", "ContinuousResult.verify"),
           src("modular_sampling/hyperflow_nodes", "MiniMaxH3HyperFlowHeadStageEXPT8.execute"),
           src("modular_sampling/hyperflow_nodes", "MiniMaxH3HyperFlowTailStageEXPT8.execute"),
           Source("tools/build_modular_hyperflow_workflow.py", "split_graph"),
           src("modular_sampling/hyperflow_identity", "model_identity"),
           src("modular_sampling/hyperflow_storage", "save_boundary"), src("modular_sampling/hyperflow_storage", "load_boundary"),
           src("modular_sampling/hyperflow_storage", "save_result"), src("modular_sampling/hyperflow_storage", "load_result"),
           src("modular_sampling/hyperflow_storage_nodes", "MiniMaxH3HyperFlowHeadSaveEXPT8.execute"),
           src("modular_sampling/hyperflow_storage_nodes", "MiniMaxH3HyperFlowHeadLoadEXPT8.execute"),
           src("modular_sampling/hyperflow_storage_nodes", "MiniMaxH3HyperFlowTailSaveEXPT8.execute"),
           src("modular_sampling/hyperflow_storage_nodes", "MiniMaxH3HyperFlowTailLoadEXPT8.execute"),
           Source("tools/build_modular_hyperflow_storage_workflow.py", "split_graph"),
           *(src("modular_sampling/hyperflow_effects", name) for name in ("bind_head", "bind_tail", "execution", "audit")),
           *(src("modular_sampling/hyperflow_effect_nodes", "MiniMaxH3HyperFlow" + name + "EXPT8.execute")
             for name in ("HeadEffectsBind", "TailEffectsBind", "HeadEffectsAudit", "TailEffectsAudit")),
           Source("tools/build_modular_hyperflow_effect_workflow.py", "split_graph")),
          ("MiniMaxH3HyperFlowSplitT8Advanced", "MiniMaxH3HyperFlowHeadStageEXPT8", "MiniMaxH3HyperFlowTailStageEXPT8",
           "MiniMaxH3HyperFlowHeadSaveEXPT8", "MiniMaxH3HyperFlowHeadLoadEXPT8",
           "MiniMaxH3HyperFlowTailSaveEXPT8", "MiniMaxH3HyperFlowTailLoadEXPT8",
           "MiniMaxH3HyperFlowHeadEffectsBindEXPT8", "MiniMaxH3HyperFlowTailEffectsBindEXPT8",
           "MiniMaxH3HyperFlowHeadEffectsAuditEXPT8", "MiniMaxH3HyperFlowTailEffectsAuditEXPT8")),
    Route("S14", "M3", "HyperFlow complete8 plus fresh4", ("full8", "fresh4"),
          "completed AV -> learned lift -> own two-time grid with fresh noise;12 NFE", ("8+4",),
          (src("nodes_hyperflow_advanced", "MiniMaxH3HyperFlowRefineSamplerT8Advanced.execute"), UPSCALE,
           *HYPERFLOW_FRESH_SOURCES),
          ("MiniMaxH3HyperFlowSamplerT8Advanced", "MiniMaxH3HyperFlowRefineSamplerT8Advanced", *HYPERFLOW_FRESH_NODES)),
    Route("S15", "M3", "HyperFlow partial4 plus fresh4", ("partial4", "fresh4"),
          "LOW prediction_x0, not raw unfinished output; own two-time grid", ("partial4+4",),
          (src("nodes_hyperflow_advanced", "MiniMaxH3HyperFlowPartialRefineSamplerT8Advanced.execute"), UPSCALE,
           *HYPERFLOW_FRESH_SOURCES),
          ("MiniMaxH3HyperFlowCoarseSamplerT8Advanced", "MiniMaxH3HyperFlowPartialRefineSamplerT8Advanced", *HYPERFLOW_FRESH_NODES)),
    Route("S16", "M3", "HyperFlow P7 long video", ("segment_low", "segment_high"),
          "P7 single-segment states; accepted RGB context and independent durable stage cache", ("T2VA_native",),
          (src("hyperflow_long_video_exp/runner", "HyperFlowLongVideoSegmentRunner"),
           src("modular_sampling/hyperflow_p7", "capture_parent"),
           src("modular_sampling/hyperflow_p7", "P7Parent.prepare_contexts"),
           src("modular_sampling/hyperflow_p7", "prepare_phase"),
           *(src("modular_sampling/hyperflow_p7", name) for name in
             ("capture_initial", "setup_low", "bind_low", "lift_low", "handoff_high", "setup_high", "bind_high")),
           src("modular_sampling/hyperflow_identity", "_project_long_video_patch"),
           *(src("modular_sampling/hyperflow_p7_storage", name) for name in
             ("stage_root", "save_low", "load_low", "save_high", "load_high")),
           *(src("modular_sampling/hyperflow_p7_effects", name) for name in
             ("_accepted_start", "project_relay", "apply_relay")),
           *(src("modular_sampling/hyperflow_p7_delivery", name) for name in
             ("_source", "_parent_identity", "verify_candidate", "save_candidate", "accept_candidate")),
           *(src("modular_sampling/hyperflow_p7_nodes", node + ".execute") for node in HYPERFLOW_P7_NODES),
           *(src("modular_sampling/hyperflow_p7_storage_nodes", node + ".execute")
             for node in HYPERFLOW_P7_STORAGE_NODES),
           *(src("modular_sampling/hyperflow_p7_effect_nodes", node + ".execute")
             for node in HYPERFLOW_P7_EFFECT_NODES),
           *(src("modular_sampling/hyperflow_p7_delivery_nodes", node + ".execute")
             for node in HYPERFLOW_P7_DELIVERY_NODES)),
          (*HYPERFLOW_P7_NODES, *HYPERFLOW_P7_STORAGE_NODES, *HYPERFLOW_P7_EFFECT_NODES,
           *HYPERFLOW_P7_DELIVERY_NODES)),
    Route("S17", "M3", "SPEED N-stage", ("stage_i", "stage_i_plus_1"),
          "DCT expansion, joint audio reindex, anchored RF state and solved segment noise; not learned upscale", ("multi_resolution", "multi_stage"),
          (src("speed_advanced", "execute_speed_sampling"), src("speed_advanced", "dct_expand_official"),
           src("speed_advanced", "reindex_joint_audio_state"), src("speed_advanced", "solve_segment_noise"),
           src("modular_sampling/speed_transition", "transition_speed_stage"),
           src("modular_sampling/speed_effects", "apply_relay"),
           src("modular_sampling/speed_effects", "capture_owner"),
           src("modular_sampling/speed_effects", "project_identity"),
           *(src("modular_sampling/speed_stages", name) for name in
             ("prepare_speed_stage", "sample_speed_stage", "handoff_speed_stage")),
           *(src("modular_sampling/speed_storage", name) for name in
             ("save_speed_stage", "load_speed_stage", "fingerprint_speed_stage")),
           *(src("modular_sampling/speed_nodes", node + ".execute") for node in SPEED_STAGE_NODES[:4]),
           *(src("modular_sampling/speed_storage_nodes", node + ".execute") for node in SPEED_STAGE_NODES[4:])),
          SPEED_STAGE_NODES, repeated_stages=True),
    Route("S18", "M4", "Chunked v1", ("first", "chunked_pass2"),
          "per-piece video refinement; input audio retained", ("video_only_legacy",),
          (CHUNKED, *CHUNKED_SOURCE, *CHUNKED_STAGES, *CHUNKED_EAV,
           src("modular_sampling/chunked_relay", "_assert_paired_conditioning"),
           src("modular_sampling/chunked_relay", "assert_relay_binding"),
           *CHUNKED_V1_LOCAL_RELAY, *CHUNKED_V1_STORAGE, *CHUNKED_WORKFLOW,
           *(Source("tools/build_modular_chunked_workflow.py", name) for name in
             ("v1_source_from_v2", "append_second_frontend_segment", "append_second_api_segment"))),
          (CHUNKED_SOURCE_NODE, *CHUNKED_STAGE_NODES, *CHUNKED_EAV_NODES,
           *CHUNKED_V1_LOCAL_RELAY_NODES, *CHUNKED_V1_STORAGE_NODES)),
    Route("S19", "M4", "Chunked v2", ("first", "chunked_pass2"),
          "global target noise sliced across windows, not independently redrawn per window", ("global_noise",),
          (CHUNKED, src("chunked_two_pass_upscale_advanced", "build_chunked_two_pass_global_noise_plan"),
           *CHUNKED_SOURCE, *CHUNKED_STAGES, *CHUNKED_EAV, *CHUNKED_RELAY, *CHUNKED_WORKFLOW),
          (CHUNKED_SOURCE_NODE, *CHUNKED_STAGE_NODES, *CHUNKED_EAV_NODES, *CHUNKED_RELAY_NODES)),
    Route("S20", "M4", "Chunked v3", ("first", "chunked_pass2"),
          "low-sigma AV initialization; distinct from v1 full-noise contract", ("low_sigma",),
          (CHUNKED, src("chunked_two_pass_upscale_advanced", "build_chunked_two_pass_low_sigma_plan"),
           *CHUNKED_SOURCE, *CHUNKED_STAGES, *CHUNKED_EAV, *CHUNKED_RELAY, *CHUNKED_WORKFLOW,
           Source("tools/build_modular_chunked_workflow.py", "v3_source_from_v4")),
          (CHUNKED_SOURCE_NODE, *CHUNKED_STAGE_NODES, *CHUNKED_EAV_NODES, *CHUNKED_RELAY_NODES)),
    Route("S21", "M4", "Chunked v4", ("first", "chunked_pass2"),
          "inherited video masks, global AV noise, time anchors and explicit overlap policy", ("masked_low_sigma", "full_clip", "guarded_overlap"),
          (CHUNKED, src("chunked_two_pass_upscale_advanced", "build_chunked_two_pass_masked_low_sigma_plan"),
           *CHUNKED_SOURCE, *CHUNKED_STAGES, *CHUNKED_EAV, *CHUNKED_RELAY, *CHUNKED_WORKFLOW),
          (CHUNKED_SOURCE_NODE, *CHUNKED_STAGE_NODES, *CHUNKED_EAV_NODES, *CHUNKED_RELAY_NODES)),
    Route("S22", "M4", "Chunked v5", ("partial4", "joint_pass2_windows"),
          "one global learned lift then per-window remaining4 joint AV; total calls grow with windows", ("standard_joint_4plus4",),
          (CHUNKED, src("chunked_two_pass_parity", "execute_standard_chunked"),
           *CHUNKED_V5, *CHUNKED_V5_WORKFLOW),
          CHUNKED_V5_NODES, repeated_stages=True),
    Route("S23", "M4", "H16 chunked PASS2", ("first", "h16_pass2"),
          "same-size learned no-op; explicit audio fallback; denoised_output is output placeholder, not x0", ("DeciiaChunkedPass2Sampler",),
          (src("nodes_h16_chunked_pass2", "DeciiaChunkedPass2Sampler"),
           *CHUNKED_SOURCE, *CHUNKED_STAGES, *H16_SPLIT),
          ("DeciiaChunkedPass2Sampler", CHUNKED_SOURCE_NODE,
           *CHUNKED_STAGE_NODES[:2], *H16_SPLIT_NODES), repeated_stages=True),
    Route("S24", "M4", "Face refinement family", ("source_generation", "region_refinement"),
          "region/window conditioning and sampling remain editable; stitch is not a sampling stage", ("standard", "parity", "per_frame", "sampler_mask", "multi_face", "window", "window_studio"),
          (src("face_refine_advanced", "setup_face_refine_sampling"), src("face_refine_parity_advanced", "apply_face_refine_per_frame_denoise"),
           src("face_refine_sampler_mask_advanced", "apply_face_refine_sampler_mask_patch"),
           src("multiface_refine_advanced", "build_multiface_repair_job"), src("face_refine_window_advanced", "extract_face_refine_window"),
           src("face_refine_window_studio_advanced", "prepare_face_refine_window_studio"),
           src("modular_sampling/face_stage", "bind_standard_face_stage"),
           src("modular_sampling/face_stage", "audit_standard_face_stage"),
           src("modular_sampling/face_stage", "bind_parity_face_stage"),
           src("modular_sampling/face_stage", "audit_parity_face_stage"),
           src("modular_sampling/face_stage", "bind_multiface_face_stage"),
           src("modular_sampling/face_stage", "audit_multiface_face_stage"),
           src("modular_sampling/face_stage", "bind_window_face_stage"),
           src("modular_sampling/face_stage", "audit_window_face_stage"),
           src("modular_sampling/face_relay", "bind_standard_face_relay"),
           src("modular_sampling/face_local_relay", "bind_local_face_relay"),
           src("modular_sampling/face_nodes", "MiniMaxH3FaceStageBindEXPT8.execute"),
           src("modular_sampling/face_nodes", "MiniMaxH3FaceStageAuditEXPT8.execute"),
           src("modular_sampling/face_nodes", "MiniMaxH3FaceParityStageBindEXPT8.execute"),
           src("modular_sampling/face_nodes", "MiniMaxH3FaceParityStageAuditEXPT8.execute"),
           src("modular_sampling/face_nodes", "MiniMaxH3MultiFaceStageBindEXPT8.execute"),
           src("modular_sampling/face_nodes", "MiniMaxH3MultiFaceStageAuditEXPT8.execute"),
           src("modular_sampling/face_nodes", "MiniMaxH3FaceWindowStageBindEXPT8.execute"),
           src("modular_sampling/face_nodes", "MiniMaxH3FaceWindowStageAuditEXPT8.execute"),
           src("modular_sampling/face_nodes", "MiniMaxH3FaceRelayBindEXPT8.execute"),
           src("modular_sampling/face_nodes", "MiniMaxH3FaceLocalRelayBindEXPT8.execute")),
          ("MiniMaxH3FaceRefineSamplerT8Advanced", "MiniMaxH3FaceStageBindEXPT8",
           "MiniMaxH3FaceStageAuditEXPT8", "MiniMaxH3FaceParityStageBindEXPT8",
           "MiniMaxH3FaceParityStageAuditEXPT8", "MiniMaxH3MultiFaceStageBindEXPT8",
           "MiniMaxH3MultiFaceStageAuditEXPT8", "MiniMaxH3FaceWindowStageBindEXPT8",
           "MiniMaxH3FaceWindowStageAuditEXPT8", "MiniMaxH3FaceRelayBindEXPT8",
           "MiniMaxH3FaceLocalRelayBindEXPT8"),
          repeated_stages=True),
    Route("S25", "M4", "Motion recovery", ("source_generation", "motion_refinement"),
          "retiming/recovery mapping and V2V tail plus explicit audio choice", ("full_clip", "windowed"),
          (src("motion_recovery_advanced", "compose_motion_recovery"), src("motion_recovery_advanced", "plan_motion_segment"),
           src("modular_sampling/motion_stage", "bind_motion_stage"),
           src("modular_sampling/motion_stage", "audit_motion_stage"),
           src("modular_sampling/motion_nodes", "MiniMaxH3MotionStageBindEXPT8.execute"),
           src("modular_sampling/motion_nodes", "MiniMaxH3MotionStageAuditEXPT8.execute")),
          ("MiniMaxH3MotionRecoveryComposerT8Advanced", "MiniMaxH3MotionStageBindEXPT8",
           "MiniMaxH3MotionStageAuditEXPT8")),
    Route("S26", "M4", "Audio-only refinement", ("video_generation", "audio_refinement"),
          "video locked, audio resampled; can append third stage to spatial dual pass; delivery separate", ("dual_clock", "dual_model", "compatibility", "long_delivery", "third_stage"),
          (src("audio_refine_advanced", "setup_audio_refine"), src("audio_refine_advanced", "setup_audio_refine_dual_model"),
           src("audio_refine_advanced", "setup_audio_refine_compatibility"), src("audio_refine_advanced", "split_audio_refine_long_video_delivery"),
           src("modular_sampling/audio_refine_stage", "bind_audio_refine_stage"),
           src("modular_sampling/audio_refine_stage", "audit_audio_refine_stage"),
           src("modular_sampling/audio_refine_stage", "audit_audio_refine_tail_delivery"),
           src("modular_sampling/audio_refine_effects", "bind_tail_effects"),
           src("modular_sampling/audio_refine_effects", "tail_effects_guider"),
           src("modular_sampling/audio_refine_effect_nodes", "MiniMaxH3AudioRefineEffectsBindEXPT8.execute"),
           src("modular_sampling/audio_refine_effect_nodes", "MiniMaxH3AudioRefineEffectsGuiderEXPT8.execute"),
           Source("tools/build_modular_audio_tail_effect_workflows.py", "add_tail_effects"),
           Source("tools/build_modular_audio_refine_workflows.py", "split_frontend"),
           Source("tools/build_modular_audio_refine_workflows.py", "split_api"),
           Source("tools/validate_modular_audio_refine_core.py", "validate"),
           src("modular_sampling/audio_refine_nodes", "_AudioRefineBind.execute"),
           src("modular_sampling/audio_refine_nodes", "_AudioRefineAudit.execute"),
           src("modular_sampling/audio_refine_nodes", "MiniMaxH3AudioRefineTailDeliveryAuditEXPT8.execute")),
          ("MiniMaxH3AudioRefineDualClockSetupT8Advanced", "MiniMaxH3AudioRefineDualModelSetupT8Advanced", "MiniMaxH3AudioRefineCompatibilitySetupT8Advanced",
           "MiniMaxH3AudioRefineDualClockStageBindEXPT8", "MiniMaxH3AudioRefineDualClockStageAuditEXPT8",
           "MiniMaxH3AudioRefineDualModelStageBindEXPT8", "MiniMaxH3AudioRefineDualModelStageAuditEXPT8",
           "MiniMaxH3AudioRefineCompatStageBindEXPT8", "MiniMaxH3AudioRefineCompatStageAuditEXPT8",
           "MiniMaxH3AudioRefineTailDeliveryAuditEXPT8",
           "MiniMaxH3AudioRefineEffectsBindEXPT8", "MiniMaxH3AudioRefineEffectsGuiderEXPT8")),
    Route("S27", "M4", "H3 Super to LTX", ("h3_draft", "ltx_refine"),
          "RGB/LTX VAE or distinct learned-latent bridge; independent LTX condition/noise/decode, original H3 audio bypass",
          ("stage2", "identity_preserve", "learned_latent_adapter"),
          (src("sol_engine_h3_super_advanced", "prepare_h3_draft_for_ltx_refiner"), src("sol_engine_h3_super_advanced", "setup_ltx_stage2_refiner"),
           src("sol_engine_h3_super_advanced", "setup_ltx_identity_preserve_refiner"),
           src("h3_ltx_latent_contract", "convert_video_latent"),
           src("nodes_h3_ltx_latent_adapter", "MiniMaxH3LTXLatentAdapterEXPT8.execute"),
           src("modular_sampling/ltx_rgb_stage", "bind_ltx_rgb_stage"),
           src("modular_sampling/ltx_rgb_stage", "audit_ltx_rgb_stage"),
           src("modular_sampling/ltx_rgb_nodes", "MiniMaxH3LTXRGBStageBindEXPT8.execute"),
           src("modular_sampling/ltx_rgb_nodes", "MiniMaxH3LTXRGBStageAuditEXPT8.execute"),
           src("modular_sampling/ltx_latent_stage", "bind_ltx_latent_stage"),
           src("modular_sampling/ltx_latent_stage", "audit_ltx_latent_stage"),
           src("modular_sampling/ltx_latent_sample_stage", "sample_ltx_latent_stage"),
           src("modular_sampling/ltx_latent_sample_stage", "audit_completed_ltx_latent_stage"),
           src("modular_sampling/ltx_latent_stage", "decode_original_h3_audio"),
           src("modular_sampling/ltx_latent_nodes", "MiniMaxH3LTXLearnedStageBindEXPT8.execute"),
           src("modular_sampling/ltx_latent_nodes", "MiniMaxH3LTXLearnedStageAuditEXPT8.execute"),
           src("modular_sampling/ltx_latent_nodes", "MiniMaxH3LTXOriginalAudioDecodeEXPT8.execute"),
           src("modular_sampling/ltx_latent_sample_nodes", "MiniMaxH3LTXLearnedStageSampleEXPT8.execute"),
           src("modular_sampling/ltx_latent_sample_nodes", "MiniMaxH3LTXLearnedStageCertifiedAuditEXPT8.execute"),
           Source("tools/build_modular_ltx_rgb_workflows.py", "split_frontend"),
           Source("tools/validate_modular_ltx_rgb_core.py", "validate"),
           Source("tools/build_modular_ltx_latent_workflow.py", "build_prompt"),
           Source("tools/validate_modular_ltx_latent_core.py", "validate"),
           src("modular_sampling/ltx_rgb_source_storage", "save_source"),
           src("modular_sampling/ltx_rgb_source_storage", "load_source"),
           src("modular_sampling/ltx_rgb_source_nodes", "MiniMaxH3LTXRGBSourceSaveEXPT8.execute"),
           src("modular_sampling/ltx_rgb_source_nodes", "MiniMaxH3LTXRGBSourceLoadEXPT8.execute"),
           Source("tools/build_modular_ltx_rgb_source_workflows.py", "build"),
           src("modular_sampling/ltx_eav", "apply_ltx_eav"),
           src("modular_sampling/ltx_eav", "audit_ltx_eav"),
           src("modular_sampling/ltx_effect_nodes", "MiniMaxH3LTXEAVApplyEXPT8.execute"),
           src("modular_sampling/ltx_effect_nodes", "MiniMaxH3LTXEAVAuditEXPT8.execute"),
           src("modular_sampling/ltx_relay_plan", "build_ltx_relay_plan"),
           src("modular_sampling/ltx_relay_nodes", "MiniMaxH3LTXPromptRelayPlanEXPT8.execute"),
           src("modular_sampling/ltx_relay_text", "encode_ltx_relay_conditioning"),
           src("modular_sampling/ltx_relay_nodes", "MiniMaxH3LTXPromptRelayEncodeEXPT8.execute"),
           src("modular_sampling/ltx_relay_apply", "apply_ltx_relay"),
           src("modular_sampling/ltx_relay_apply", "audit_ltx_relay"),
           src("modular_sampling/ltx_relay_nodes", "MiniMaxH3LTXPromptRelayApplyEXPT8.execute"),
           src("modular_sampling/ltx_relay_nodes", "MiniMaxH3LTXPromptRelayAuditEXPT8.execute")),
          ("MiniMaxH3SolEngineLTXRefinerSetupT8Advanced", "MiniMaxH3SolEngineLTXIdentityRefinerSetupT8Advanced",
           "MiniMaxH3LTXLatentAdapterEXPT8", "MiniMaxH3LTXRGBStageBindEXPT8", "MiniMaxH3LTXRGBStageAuditEXPT8",
           "MiniMaxH3LTXLearnedStageBindEXPT8", "MiniMaxH3LTXLearnedStageAuditEXPT8",
           "MiniMaxH3LTXOriginalAudioDecodeEXPT8", "MiniMaxH3LTXLearnedStageSampleEXPT8",
           "MiniMaxH3LTXLearnedStageCertifiedAuditEXPT8",
           "MiniMaxH3LTXRGBSourceSaveEXPT8", "MiniMaxH3LTXRGBSourceLoadEXPT8",
           "MiniMaxH3LTXEAVApplyEXPT8", "MiniMaxH3LTXEAVAuditEXPT8",
           "MiniMaxH3LTXPromptRelayPlanEXPT8", "MiniMaxH3LTXPromptRelayEncodeEXPT8",
           "MiniMaxH3LTXPromptRelayApplyEXPT8", "MiniMaxH3LTXPromptRelayAuditEXPT8")),
    Route("S28", "M4", "Prepared LTX worker", ("prepared_generation", "ltx_refine"),
          "prepared bundle -> SHA-bound refined latent artifact -> explicit load or separate decode; retain subprocess identity", ("ltx_refine",),
          (src("prepared_backend/ltx_prepared_refinement_worker", "main"),
           src("prepared_backend/ltx_refined_decode_worker", "main"),
           src("modular_sampling/prepared_ltx_stage", "run_generation"),
           src("modular_sampling/prepared_ltx_stage", "load_generation"),
           src("modular_sampling/prepared_ltx_stage", "run_decode"),
           src("modular_sampling/prepared_ltx_nodes", "MiniMaxH3PreparedLTXGenerateEXPT8.execute"),
           src("modular_sampling/prepared_ltx_nodes", "MiniMaxH3PreparedLTXLoadGenerationEXPT8.execute"),
           src("modular_sampling/prepared_ltx_nodes", "MiniMaxH3PreparedLTXDecodeEXPT8.execute"),
           Source("tools/build_modular_prepared_ltx_workflows.py", "build_prompt"),
           Source("tools/validate_modular_prepared_ltx_core.py", "validate"),
           src("modular_sampling/prepared_ltx_effects_nodes", "bind_prepared_ltx_effects"),
           src("modular_sampling/prepared_ltx_effects_nodes", "plan_prepared_ltx_relay"),
           src("modular_sampling/prepared_ltx_relay_cache", "run_cache_encoding"),
           src("modular_sampling/prepared_ltx_relay_cache_nodes", "MiniMaxH3PreparedLTXRelayEncodeEXPT8.execute"),
           src("prepared_backend/ltx_external_effects", "PreparedLTXEffects.installed"),
           Source("tools/build_formal_prepared_ltx_effect_workflows.py", "build_prompt")),
          ("MiniMaxH3PreparedLTXGenerateEXPT8", "MiniMaxH3PreparedLTXLoadGenerationEXPT8",
           "MiniMaxH3PreparedLTXDecodeEXPT8", "MiniMaxH3PreparedLTXEffectsBindEXPT8",
           "MiniMaxH3PreparedLTXRelayPlanEXPT8", "MiniMaxH3PreparedLTXRelayEncodeEXPT8")),
    Route("S29", "M2", "Rectified-flow restart", ("base_descent", "restart_descent"),
          "clean endpoint -> joint AV RF re-noise -> second descent; independent restart_seed; disabled exact bypass", ("standalone", "detail_mixer", "two_pass_detail_mixer", "third_stage"),
          (src("detail_sampling_advanced", "setup_rectified_flow_restart_sampling"), src("detail_sampling_advanced", "setup_detail_mixer_sampling"),
           src("detail_sampling_advanced", "setup_two_pass_detail_mixer_sampling"),
           src("modular_sampling/rf_restart", "prepare_handoff"), src("modular_sampling/rf_restart", "build_restart_stage"),
           src("modular_sampling/rf_restart", "RFRestartSampler.sample"),
           src("modular_sampling/rf_stages", "build_base_stage"), src("modular_sampling/rf_stages", "build_restart_stage"),
           src("modular_sampling/rf_stages", "handoff"), src("modular_sampling/detail_effects", "forward_plan")),
          ("MiniMaxH3RFBaseStageSetupEXPT8", "MiniMaxH3RFHandoffEXPT8", "MiniMaxH3RFRestartStageSetupEXPT8")),
)


def validate_catalogue(routes=ROUTES):
    ids = [route.id for route in routes]
    required = [f"S{index:02d}" for index in range(1, 30)]
    if ids[:len(required)] != required or ids != [f"S{index:02d}" for index in range(1, len(ids) + 1)]:
        raise ValueError("The full ordered S01-S29 scope and append-only route IDs are required")
    for route in routes:
        if (route.phase not in ("M1", "M2", "M3", "M4", "M5") or len(route.stages) < 2
                or not route.name or not route.handoff or not route.variants or not route.sources
                or len(set(route.stages)) != len(route.stages)):
            raise ValueError(f"Incomplete recipe contract: {route.id}")
        for source in route.sources:
            if (not source.path.endswith(".py") or source.path.startswith(("/", "\\"))
                    or ":" in source.path or "\\" in source.path or ".." in source.path.split("/")
                    or not source.symbol or any(not item.isidentifier() for item in source.symbol.split("."))):
                raise ValueError(f"Invalid source binding: {route.id}")
    return routes
