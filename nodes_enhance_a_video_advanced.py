from __future__ import annotations

from comfy_api.latest import io

from .enhance_a_video_advanced import (
    EAV_MODES,
    EAV_RUNTIME_TYPE,
    EAV_SAGE_TASK_SCOPES,
    EAV_SAMPLING_PROFILES,
    build_eav_model,
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

ENHANCE_A_VIDEO_ADVANCED_NODE_CLASSES = [
    MiniMaxH3EnhanceAVideoT8Advanced,
    MiniMaxH3EnhanceAVideoAuditT8Advanced,
    MiniMaxH3EnhanceAVideoReferenceComposerT8Advanced,
    MiniMaxH3EnhanceAVideoSageComposerT8Advanced,
]
