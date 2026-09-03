from __future__ import annotations

import json

from comfy_api.latest import io

from .vdn_h3_advanced import (
    STAGES,
    audit_vdn_runtime,
    available_vdn_roots,
    compose_vdn_model,
    setup_vdn_execution,
)


CATEGORY = "T8/MiniMax H3/Performance/Experimental"


class MiniMaxH3VDNRuntimeAuditT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VDNRuntimeAuditT8Advanced",
            display_name="MiniMax H3 OpenVDN Runtime Audit (Advanced EXP/T8)",
            description=(
                "Read-only audit for the pinned OpenVDN MiniMax H3 branch and adapters. "
                "Checks exact asset identity, native H3 structure, attention ownership and "
                "the explicit structural-base exception without loading the 4.28GB branch. "
                "The report also exposes the separately governed weight-license boundary."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "vdn_root",
                    options=available_vdn_roots(),
                    default=available_vdn_roots()[0],
                ),
                io.Combo.Input("stage", options=list(STAGES), default="stage_dmd_8nfe"),
                io.Boolean.Input("verify_hashes", default=True, advanced=True),
                io.Boolean.Input(
                    "allow_structural_base",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "Explicit EXP exception for a structurally matching Comfy H3 base "
                        "whose exact official BF16 revision provenance is unavailable."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Boolean.Output("ready"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        vdn_root,
        stage,
        verify_hashes=True,
        allow_structural_base=False,
    ):
        ready, report = audit_vdn_runtime(
            model,
            vdn_root,
            stage,
            verify_hashes=verify_hashes,
            allow_structural_base=allow_structural_base,
        )
        return io.NodeOutput(
            model,
            ready,
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        )


class MiniMaxH3VDNModelComposerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VDNModelComposerT8Advanced",
            display_name="MiniMax H3 OpenVDN Model Composer (Advanced EXP/T8)",
            description=(
                "Clones a clean native H3 MODEL, applies the pinned OpenVDN default/turbo "
                "adapters, and attaches the 50-layer VDN branch through Comfy ModelPatcher. "
                "v1 is strict T2VA only and rejects every competing attention owner. Users "
                "must review the MiniMax H3 Community License before running the weights."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "vdn_root",
                    options=available_vdn_roots(),
                    default=available_vdn_roots()[0],
                ),
                io.Combo.Input("stage", options=list(STAGES), default="stage_dmd_8nfe"),
                io.Boolean.Input("verify_hashes", default=True, advanced=True),
                io.Boolean.Input(
                    "allow_structural_base",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "Required for the current local INT8/ConvRot H3 base. This records "
                        "an explicit unproven-base EXP exception; it does not relabel it exact."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        vdn_root,
        stage,
        verify_hashes=True,
        allow_structural_base=False,
    ):
        return io.NodeOutput(
            *compose_vdn_model(
                model,
                vdn_root,
                stage,
                verify_hashes=verify_hashes,
                allow_structural_base=allow_structural_base,
            )
        )


class MiniMaxH3VDNExecutionPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VDNExecutionPlanT8Advanced",
            display_name="MiniMax H3 OpenVDN Execution Plan (Advanced EXP/T8)",
            description=(
                "Builds the exact OpenVDN sampling contract from the connected MODEL: "
                "DMD uses 8 NFE, Stage B uses 50 NFE, both use native AV Euler and "
                "video/audio shifts 12/3. No user-overridable mismatched step count."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[io.Model.Input("model"), io.Latent.Input("av_latent")],
            outputs=[
                io.Model.Output("model"),
                io.Sampler.Output("sampler"),
                io.Sigmas.Output("sigmas"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, model, av_latent):
        return io.NodeOutput(*setup_vdn_execution(model, av_latent))


VDN_H3_ADVANCED_NODE_CLASSES = [
    MiniMaxH3VDNRuntimeAuditT8Advanced,
    MiniMaxH3VDNModelComposerT8Advanced,
    MiniMaxH3VDNExecutionPlanT8Advanced,
]
