from __future__ import annotations

from comfy_api.latest import io

from .enhance_a_video_advanced import (
    EAV_MODES,
    EAV_RUNTIME_TYPE,
    EAV_SAGE_TASK_SCOPES,
    EAV_SAMPLING_PROFILES,
    build_eav_block_cache_model,
    build_eav_long_video_model,
    build_eav_model,
    build_eav_prompt_relay_model,
    build_eav_stg_model,
    finalize_eav_runtime,
)


CATEGORY = "T8/MiniMax H3/Quality/Experimental"
EAVRuntimeIO = io.Custom(EAV_RUNTIME_TYPE)


class MiniMaxH3EnhanceAVideoT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3EnhanceAVideoT8Advanced",
            display_name="MiniMax H3 Enhance-A-Video / FETA (Advanced EXP)",
            description=(
                "Experimental H3 full-3D adapter for Enhance-A-Video/FETA. It computes "
                "the paper temporal CFI only from target-video Q/K, in exact low-memory "
                "chunks, and directly scales only target-video attention output rows."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "Stock20: connect the native H3 model with no LoRA/cache/attention patch. "
                        "Turbo8: connect the corrected 208-module Alpha8 bypass LoRA first, then "
                        "this node; the runtime contract is checked strictly."
                    ),
                ),
                io.Sigmas.Input(
                    "sigmas",
                    tooltip="Connect the exact Stock20 or Turbo8 SIGMAS sent to the sampler.",
                ),
                io.Combo.Input(
                    "mode",
                    options=list(EAV_MODES),
                    default="report_only",
                    tooltip=(
                        "disabled is exact bypass; report_only measures CFI/g without changing "
                        "the output; apply_exp enables the experimental gain. tau=0 is not off."
                    ),
                ),
                io.Float.Input(
                    "tau",
                    default=4.0,
                    min=-32.0,
                    max=32.0,
                    step=0.25,
                    tooltip=(
                        "Paper enhancement weight. 4 is an upstream candidate, not a validated "
                        "H3 optimum; always compare against disabled/report_only with the same seed."
                    ),
                ),
                io.Float.Input(
                    "start_video_progress",
                    default=0.0,
                    min=0.0,
                    max=0.99,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "end_video_progress",
                    default=1.0,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Int.Input(
                    "max_workspace_mib",
                    default=32,
                    min=4,
                    max=512,
                    step=4,
                    advanced=True,
                    tooltip=(
                        "Upper planning budget for chunked temporal score buffers. This is not "
                        "a whole-workflow VRAM guarantee."
                    ),
                ),
                io.Float.Input(
                    "g_hard_limit",
                    default=1.5,
                    min=1.0,
                    max=3.0,
                    step=0.01,
                    advanced=True,
                    tooltip=(
                        "Fail closed when the observed gain exceeds this value. The node refuses "
                        "the run instead of silently clamping and calling it paper-equivalent."
                    ),
                ),
                io.Combo.Input(
                    "sampling_profile",
                    options=list(EAV_SAMPLING_PROFILES),
                    default="stock20",
                    tooltip=(
                        "stock20 rejects every LoRA. turbo8_alpha8 requires the corrected "
                        "208-module Alpha8 bypass LoRA at strength 1.0 before this node."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                EAVRuntimeIO.Output("runtime"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_eav_model(**kwargs))


class MiniMaxH3EnhanceAVideoAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3EnhanceAVideoAuditT8Advanced",
            display_name="MiniMax H3 Enhance-A-Video Runtime Audit (Advanced EXP)",
            description=(
                "Place after sampling. It verifies the selected schedule forward count, all 50 H3 DiT "
                "attention measurements, observed g range, and chunk workspace without "
                "changing the AV latent."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Latent.Input("av_latent"),
                EAVRuntimeIO.Input("runtime"),
            ],
            outputs=[
                io.Latent.Output("av_latent"),
                io.String.Output("report_json"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, av_latent, runtime):
        latent, report_json = finalize_eav_runtime(av_latent, runtime)
        return io.NodeOutput(latent, report_json, ui={"text": (report_json,)})


class MiniMaxH3EnhanceAVideoReferenceComposerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3EnhanceAVideoReferenceComposerT8Advanced",
            display_name=(
                "MiniMax H3 Enhance-A-Video Ref2VA / Hybrid Composer (Advanced EXP)"
            ),
            description=(
                "Isolated Stock20 composer for native Ref2VA or Hybrid conditioning. It audits "
                "the exact reference-segment layout, computes FETA only from target-video Q/K, "
                "and never directly scales reference, text, condition, or target-audio rows."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "Connect an unpatched native Stock20 H3 model. LoRA, Prompt Relay, Sage, "
                        "BlockCache, STG, Long Video and model-Hybrid artifacts remain rejected."
                    ),
                ),
                io.Sigmas.Input(
                    "sigmas",
                    tooltip="Connect the exact 20-step Stock20 SIGMAS sent to the sampler.",
                ),
                io.Combo.Input(
                    "mode",
                    options=list(EAV_MODES),
                    default="report_only",
                    tooltip=(
                        "disabled is exact bypass; report_only audits without modifying output; "
                        "apply_exp enables FETA. tau=0 is not an off switch."
                    ),
                ),
                io.Float.Input("tau", default=4.0, min=-32.0, max=32.0, step=0.25),
                io.Float.Input(
                    "start_video_progress",
                    default=0.0,
                    min=0.0,
                    max=0.99,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "end_video_progress",
                    default=1.0,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Int.Input(
                    "max_workspace_mib",
                    default=32,
                    min=4,
                    max=512,
                    step=4,
                    advanced=True,
                    tooltip="FETA score-buffer planning budget only; not total workflow VRAM.",
                ),
                io.Float.Input(
                    "g_hard_limit",
                    default=1.5,
                    min=1.0,
                    max=3.0,
                    step=0.01,
                    advanced=True,
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                EAVRuntimeIO.Output("runtime"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(
            *build_eav_model(
                **kwargs,
                sampling_profile="stock20",
                allowed_tasks=("Ref2VA", "Hybrid"),
                allow_reference_blocks=True,
                composer_profile="native_reference_stock20_v1",
            )
        )


class MiniMaxH3EnhanceAVideoSageComposerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3EnhanceAVideoSageComposerT8Advanced",
            display_name="MiniMax H3 Enhance-A-Video + Strict Sage (Advanced EXP)",
            description=(
                "Owns both the H3 FETA route and a strict SageAttention HND backend. "
                "It refuses external Sage object patches and never silently falls back to "
                "PyTorch attention. Use this node instead of stacking the standalone EAV "
                "node with a third-party MiniMax H3 Sage patch."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "Connect a native H3 model with no attention/object/block patch. "
                        "For turbo8_alpha8, apply only the corrected 208-module Alpha8 "
                        "bypass LoRA before this node."
                    ),
                ),
                io.Sigmas.Input(
                    "sigmas",
                    tooltip="Connect the exact SIGMAS sent to the sampler.",
                ),
                io.Combo.Input(
                    "task_scope",
                    options=list(EAV_SAGE_TASK_SCOPES),
                    default="visual",
                    tooltip=(
                        "visual: T2VA/I2VA/FL2VA/L2VA. reference: Ref2VA/Hybrid and "
                        "Stock20 only. Scope is explicit so task routing cannot silently drift."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=list(EAV_MODES),
                    default="report_only",
                    tooltip=(
                        "disabled returns the original model and runs neither Sage nor FETA. "
                        "report_only still runs Sage but does not apply the FETA gain. "
                        "apply_exp runs both."
                    ),
                ),
                io.Float.Input(
                    "tau",
                    default=4.0,
                    min=-32.0,
                    max=32.0,
                    step=0.25,
                    tooltip="Experimental FETA weight; 4 is a candidate, not an H3 optimum.",
                ),
                io.Float.Input(
                    "start_video_progress",
                    default=0.0,
                    min=0.0,
                    max=0.99,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "end_video_progress",
                    default=1.0,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Int.Input(
                    "max_workspace_mib",
                    default=32,
                    min=4,
                    max=512,
                    step=4,
                    advanced=True,
                    tooltip="FETA score-buffer budget only; not a whole-workflow VRAM claim.",
                ),
                io.Float.Input(
                    "g_hard_limit",
                    default=1.5,
                    min=1.0,
                    max=3.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Combo.Input(
                    "sampling_profile",
                    options=list(EAV_SAMPLING_PROFILES),
                    default="stock20",
                    tooltip=(
                        "reference scope is intentionally Stock20-only. visual scope also "
                        "accepts the audited corrected Alpha8 Turbo8 path."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                EAVRuntimeIO.Output("runtime"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        task_scope = str(kwargs.pop("task_scope"))
        sampling_profile = str(kwargs.get("sampling_profile", "stock20"))
        if task_scope == "visual":
            allowed_tasks = ("T2VA", "I2VA", "FL2VA", "L2VA")
            allow_reference_blocks = False
        elif task_scope == "reference":
            if sampling_profile != "stock20":
                raise ValueError(
                    "H3 EAV + Strict Sage reference scope currently requires stock20"
                )
            allowed_tasks = ("Ref2VA", "Hybrid")
            allow_reference_blocks = True
        else:
            raise ValueError(f"Unknown H3 EAV + Strict Sage task scope {task_scope!r}")
        return io.NodeOutput(
            *build_eav_model(
                **kwargs,
                allowed_tasks=allowed_tasks,
                allow_reference_blocks=allow_reference_blocks,
                composer_profile=f"strict_sage_{task_scope}_v1",
                attention_backend="strict_sage_hnd",
            )
        )


class MiniMaxH3EnhanceAVideoPromptRelayComposerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3EnhanceAVideoPromptRelayComposerT8Advanced",
            display_name="MiniMax H3 Enhance-A-Video + Prompt Relay (Advanced EXP)",
            description=(
                "Explicitly replaces one authenticated Prompt Relay wrapper with a single "
                "combined owner. Relay performs its local-event attention route first; FETA "
                "then scales only target-video output rows. It does not add model forwards."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "Connect the MODEL output of Prompt Relay Conditioning in apply_exp "
                        "with at least two active events. For turbo8_alpha8, apply the corrected "
                        "Alpha8 bypass LoRA after Relay and before this composer."
                    ),
                ),
                io.Sigmas.Input(
                    "sigmas",
                    tooltip="Connect the exact SIGMAS sent to the sampler.",
                ),
                io.Combo.Input(
                    "mode",
                    options=list(EAV_MODES),
                    default="report_only",
                    tooltip=(
                        "disabled preserves Prompt Relay exactly and disables only EAV; "
                        "report_only measures FETA after Relay without scaling; apply_exp "
                        "enables the target-video gain."
                    ),
                ),
                io.Float.Input(
                    "tau",
                    default=4.0,
                    min=-32.0,
                    max=32.0,
                    step=0.25,
                    tooltip="Experimental FETA weight; 4 is a candidate, not an H3 optimum.",
                ),
                io.Float.Input(
                    "start_video_progress",
                    default=0.0,
                    min=0.0,
                    max=0.99,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "end_video_progress",
                    default=1.0,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Int.Input(
                    "max_workspace_mib",
                    default=32,
                    min=4,
                    max=512,
                    step=4,
                    advanced=True,
                    tooltip="FETA score-buffer budget only; not total workflow VRAM.",
                ),
                io.Float.Input(
                    "g_hard_limit",
                    default=1.5,
                    min=1.0,
                    max=3.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Combo.Input(
                    "sampling_profile",
                    options=list(EAV_SAMPLING_PROFILES),
                    default="stock20",
                    tooltip=(
                        "stock20 supports the bound visual/reference task. turbo8_alpha8 is "
                        "strictly limited to T2VA with the corrected 208-hook bypass LoRA."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                EAVRuntimeIO.Output("runtime"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_eav_prompt_relay_model(**kwargs))


class MiniMaxH3EnhanceAVideoBlockCacheComposerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3EnhanceAVideoBlockCacheComposerT8Advanced",
            display_name="MiniMax H3 Enhance-A-Video + BlockCache (Advanced EXP)",
            description=(
                "Explicit Stock20 composer for the separately installed T8 BlockCache. "
                "It runs FETA only in blocks that actually execute and audits one measurement "
                "on a cache hit versus 50 on a full forward."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "Connect DualClock MODEL after MiniMax H3 Block Cache (T8). The first "
                        "contract requires CPU cache and refuses every other wrapper/patch."
                    ),
                ),
                io.Sigmas.Input(
                    "sigmas",
                    tooltip="Connect the exact native Stock20 SIGMAS sent to the sampler.",
                ),
                io.Combo.Input(
                    "mode",
                    options=list(EAV_MODES),
                    default="report_only",
                    tooltip=(
                        "disabled preserves the exact BlockCache MODEL; report_only audits "
                        "FETA without gain; apply_exp enables target-video gain."
                    ),
                ),
                io.Float.Input(
                    "tau",
                    default=4.0,
                    min=-32.0,
                    max=32.0,
                    step=0.25,
                    tooltip="Experimental FETA weight; 4 is a candidate, not an H3 optimum.",
                ),
                io.Float.Input(
                    "start_video_progress",
                    default=0.0,
                    min=0.0,
                    max=0.99,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "end_video_progress",
                    default=1.0,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Int.Input(
                    "max_workspace_mib",
                    default=32,
                    min=4,
                    max=512,
                    step=4,
                    advanced=True,
                    tooltip="FETA score-buffer budget only; not total workflow VRAM.",
                ),
                io.Float.Input(
                    "g_hard_limit",
                    default=1.5,
                    min=1.0,
                    max=3.0,
                    step=0.01,
                    advanced=True,
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                EAVRuntimeIO.Output("runtime"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_eav_block_cache_model(**kwargs))


class MiniMaxH3EnhanceAVideoSTGComposerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3EnhanceAVideoSTGComposerT8Advanced",
            display_name="MiniMax H3 Enhance-A-Video + STG (Advanced EXP)",
            description=(
                "One explicit Stock20 owner for target-video FETA and skip-block STG. "
                "FETA is applied consistently to both the main and weak branches; the "
                "runtime audit verifies the exact extra joint audio-video forwards."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "Connect an unpatched native H3 model. Do not stack standalone STG, "
                        "BlockCache, Prompt Relay, Sage, Long Video or another guidance hook."
                    ),
                ),
                io.Sigmas.Input(
                    "sigmas",
                    tooltip="Connect the exact native Stock20 SIGMAS sent to the sampler.",
                ),
                io.Combo.Input(
                    "mode",
                    options=list(EAV_MODES),
                    default="report_only",
                    tooltip=(
                        "disabled keeps STG active but disables only EAV for a controlled "
                        "baseline; report_only measures FETA on both branches; apply_exp "
                        "also enables the target-video gain."
                    ),
                ),
                io.Float.Input("tau", default=4.0, min=-32.0, max=32.0, step=0.25),
                io.Float.Input(
                    "start_video_progress",
                    default=0.0,
                    min=0.0,
                    max=0.99,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "end_video_progress",
                    default=1.0,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Int.Input(
                    "max_workspace_mib",
                    default=32,
                    min=4,
                    max=512,
                    step=4,
                    advanced=True,
                    tooltip="FETA score-buffer budget only; not total workflow VRAM.",
                ),
                io.Float.Input(
                    "g_hard_limit",
                    default=1.5,
                    min=1.0,
                    max=3.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "stg_scale",
                    default=0.35,
                    min=0.0,
                    max=5.0,
                    step=0.05,
                    tooltip=(
                        "Strength of the extra weak-branch guidance. 0.35 is a conservative "
                        "starting point, not a universal quality optimum."
                    ),
                ),
                io.String.Input(
                    "stg_double_blocks",
                    default="25",
                    tooltip="Comma-separated H3 double blocks skipped only by the weak branch.",
                ),
                io.Float.Input(
                    "stg_start_progress",
                    default=0.25,
                    min=0.0,
                    max=0.99,
                    step=0.01,
                ),
                io.Float.Input(
                    "stg_end_progress",
                    default=0.85,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                ),
                io.Float.Input(
                    "shift_video",
                    default=12.0,
                    min=0.01,
                    max=64.0,
                    step=0.1,
                    advanced=True,
                    tooltip="Must match the native H3 video clock used by the schedule.",
                ),
                io.Float.Input(
                    "rescale",
                    default=0.0,
                    min=0.0,
                    max=0.0,
                    step=0.01,
                    advanced=True,
                    tooltip="Must remain 0; shared AV global-std rescale is not validated.",
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                EAVRuntimeIO.Output("runtime"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_eav_stg_model(**kwargs))


class MiniMaxH3EnhanceAVideoLongVideoComposerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3EnhanceAVideoLongVideoComposerT8Advanced",
            display_name="MiniMax H3 Enhance-A-Video + Long Video (Advanced EXP)",
            description=(
                "Keeps Long Video Conditioning as the scoped layout owner and adds one "
                "fresh EAV runtime per Stock20 segment. Segment/context inputs bind resume "
                "state and prevent a consumed audit token from leaking across segments."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="Connect the MODEL output of the matching Long Video Conditioning node.",
                ),
                io.Sigmas.Input(
                    "sigmas",
                    tooltip="Connect the exact native Stock20 SIGMAS sent to this segment sampler.",
                ),
                io.Int.Input(
                    "segment_index",
                    default=0,
                    min=0,
                    max=99999,
                    force_input=True,
                    tooltip="Connect Long Video Planner segment_index.",
                ),
                io.Int.Input(
                    "context_frames",
                    default=0,
                    min=0,
                    max=39,
                    force_input=True,
                    tooltip="Connect Long Video Planner context_frames (0, 5, 22 or 39).",
                ),
                io.Combo.Input(
                    "mode",
                    options=list(EAV_MODES),
                    default="report_only",
                    tooltip=(
                        "disabled preserves the exact Long Video MODEL; report_only audits "
                        "FETA without gain; apply_exp enables target-video gain."
                    ),
                ),
                io.Float.Input("tau", default=4.0, min=-32.0, max=32.0, step=0.25),
                io.Float.Input(
                    "start_video_progress",
                    default=0.0,
                    min=0.0,
                    max=0.99,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "end_video_progress",
                    default=1.0,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Int.Input(
                    "max_workspace_mib",
                    default=32,
                    min=4,
                    max=512,
                    step=4,
                    advanced=True,
                    tooltip="FETA score-buffer budget only; not total workflow VRAM.",
                ),
                io.Float.Input(
                    "g_hard_limit",
                    default=1.5,
                    min=1.0,
                    max=3.0,
                    step=0.01,
                    advanced=True,
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                EAVRuntimeIO.Output("runtime"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_eav_long_video_model(**kwargs))

ENHANCE_A_VIDEO_ADVANCED_NODE_CLASSES = [
    MiniMaxH3EnhanceAVideoT8Advanced,
    MiniMaxH3EnhanceAVideoAuditT8Advanced,
    MiniMaxH3EnhanceAVideoReferenceComposerT8Advanced,
    MiniMaxH3EnhanceAVideoSageComposerT8Advanced,
    MiniMaxH3EnhanceAVideoPromptRelayComposerT8Advanced,
    MiniMaxH3EnhanceAVideoBlockCacheComposerT8Advanced,
    MiniMaxH3EnhanceAVideoSTGComposerT8Advanced,
    MiniMaxH3EnhanceAVideoLongVideoComposerT8Advanced,
]
