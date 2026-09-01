from __future__ import annotations

import folder_paths
from comfy_api.latest import io

from .sla_attention_advanced import SLA_LORA_FILENAME
from .sla_precision_v2_advanced import (
    RECOMMENDED_SCHEDULE,
    RUNTIME_TYPE,
    SCHEDULE_POLICIES,
    apply_sla_dynamic_lora_bypass,
    finalize_sla_precision_v2_runtime,
    patch_sla_precision_v2,
)


CATEGORY = "T8/MiniMax H3/Performance/Experimental"
SLAPrecisionRuntimeIO = io.Custom(RUNTIME_TYPE)
DENSE_BACKENDS = (
    "pytorch",
    "comfy_kitchen",
    "sage:auto",
    "sage:qk_int8_pv_fp16_cuda",
    "sage:qk_int8_pv_fp16_triton",
    "sage:qk_int8_pv_fp8_cuda",
    "sage:qk_int8_pv_fp8_cuda++",
    "auto",
)


def _lora_options() -> list[str]:
    names = list(folder_paths.get_filename_list("loras"))
    if SLA_LORA_FILENAME in names:
        names.remove(SLA_LORA_FILENAME)
    return [SLA_LORA_FILENAME, *names]


class MiniMaxH3SLADynamicLoRABypassV2T8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SLADynamicLoRABypassV2T8Advanced",
            display_name="MiniMax H3 SLA Dynamic LoRA Bypass V2 (Advanced EXP)",
            description=(
                "Loads the selected H3 SLA LoRA as a model-only dynamic residual. "
                "Quantized base weights are not merged or re-quantized. Place this "
                "before SLA Precision V2 Attention."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "lora_name",
                    options=_lora_options(),
                    default=SLA_LORA_FILENAME,
                ),
            ],
            outputs=[io.Model.Output("model"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, lora_name):
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        return io.NodeOutput(*apply_sla_dynamic_lora_bypass(model, path))


class MiniMaxH3SLAPrecisionV2T8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SLAPrecisionV2T8Advanced",
            display_name="MiniMax H3 SLA Precision V2 Attention (Advanced EXP)",
            description=(
                "Precision-first MiniMax H3 SLA attention derived from PlagueKind "
                "v1.4.3 at pinned commit 066ada9: FP32 routing, direct Triton sparse "
                "attention, logical sigma-step tracking, first/last dense protection "
                "and precise language/audio segment protection. Input MODEL must already "
                "carry the SLA LoRA; the recommended route uses the matching dynamic "
                "bypass node."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Sigmas.Input("sigmas"),
                io.Combo.Input(
                    "schedule_policy",
                    options=list(SCHEDULE_POLICIES),
                    default=RECOMMENDED_SCHEDULE,
                ),
                io.Float.Input(
                    "sparsity_ratio", default=0.90, min=0.60, max=0.95, step=0.05
                ),
                io.Combo.Input(
                    "block_size", options=["32", "64", "128"], default="32"
                ),
                io.Int.Input(
                    "min_seq_len", default=8192, min=0, max=1_000_000, step=1024
                ),
                io.Int.Input("dense_last_steps", default=1, min=0, max=8),
                io.Boolean.Input("protect_audio", default=True),
                io.String.Input("dense_steps", default="0", optional=True),
                io.Combo.Input(
                    "dense_backend",
                    options=list(DENSE_BACKENDS),
                    default="comfy_kitchen",
                    optional=True,
                ),
                io.Boolean.Input("disable_fp16_accum", default=True, optional=True),
                io.Boolean.Input("stabilize_motion", default=False, optional=True),
                io.Boolean.Input("reference_protection", default=False, optional=True),
            ],
            outputs=[
                io.Model.Output("model"),
                SLAPrecisionRuntimeIO.Output("runtime"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        sigmas,
        schedule_policy,
        sparsity_ratio,
        block_size,
        min_seq_len,
        dense_last_steps,
        protect_audio,
        dense_steps="0",
        dense_backend="comfy_kitchen",
        disable_fp16_accum=True,
        stabilize_motion=False,
        reference_protection=False,
    ):
        return io.NodeOutput(
            *patch_sla_precision_v2(
                model,
                sigmas,
                schedule_policy=schedule_policy,
                sparsity_ratio=sparsity_ratio,
                block_size=int(block_size),
                min_seq_len=min_seq_len,
                dense_last_steps=dense_last_steps,
                dense_steps=dense_steps,
                dense_backend=dense_backend,
                protect_audio=protect_audio,
                disable_fp16_accum=disable_fp16_accum,
                stabilize_motion=stabilize_motion,
                reference_protection=reference_protection,
            )
        )


class MiniMaxH3SLAPrecisionV2AuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SLAPrecisionV2AuditT8Advanced",
            display_name="MiniMax H3 SLA Precision V2 Runtime Audit (Advanced EXP)",
            description=(
                "Fails closed after sampling unless logical NFE, 50 H3 blocks per "
                "sparse step, dense boundaries, protected audio/language blocks and "
                "zero kernel fallback are all observed."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Latent.Input("av_latent"),
                SLAPrecisionRuntimeIO.Input("runtime"),
            ],
            outputs=[io.Latent.Output("av_latent"), io.String.Output("report_json")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, av_latent, runtime):
        latent, report_json = finalize_sla_precision_v2_runtime(av_latent, runtime)
        return io.NodeOutput(latent, report_json, ui={"text": (report_json,)})


SLA_PRECISION_V2_ADVANCED_NODE_CLASSES = [
    MiniMaxH3SLADynamicLoRABypassV2T8Advanced,
    MiniMaxH3SLAPrecisionV2T8Advanced,
    MiniMaxH3SLAPrecisionV2AuditT8Advanced,
]
