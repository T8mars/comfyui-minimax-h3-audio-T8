"""Experimental, opt-in Veda predictor / Flex H3 T2VA nodes."""

from __future__ import annotations

import json
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .veda_runtime import VedaBundleRef, VedaRuntime, apply_veda, inspect_bundle


folder_paths.add_model_folder_path(
    "veda_scorers", str(Path(folder_paths.models_dir) / "veda_scorers"))


class MiniMaxH3VedaBundleEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VedaBundleEXPT8",
            display_name="H3 Veda · Bundle (T2VA EXP)",
            category="T8/MiniMax H3/Acceleration/Experimental",
            is_experimental=True,
            description="Select the author's predictor bundle from models/veda_scorers. "
                        "This is a tile-score predictor, not an H3 UNET or Turbo LoRA. "
                        "The recommended teacher recipe uses a separate Turbo v4 step600 LoRA "
                        "and explicit eight-step sampling; the training step is not NFE. "
                        "Checks format and SHA; does not run inference or download weights.",
            inputs=[io.Combo.Input(
                "bundle_name",
                options=[name for name in folder_paths.get_filename_list("veda_scorers")
                         if Path(name).suffix.lower() == ".safetensors"] or ["放入 models/veda_scorers"],
            )],
            outputs=[io.Custom("T8_VEDA_BUNDLE").Output("bundle"),
                     io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, bundle_name):
        if bundle_name == "放入 models/veda_scorers":
            raise ValueError("Place the released Veda predictor .safetensors in models/veda_scorers")
        path = Path(folder_paths.get_full_path_or_raise("veda_scorers", bundle_name))
        reference = inspect_bundle(path)
        return io.NodeOutput(reference, json.dumps(reference.report(), ensure_ascii=False, indent=2))


class MiniMaxH3VedaApplyEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VedaApplyEXPT8",
            display_name="H3 Veda · Apply Flex Sparse (T2VA EXP)",
            category="T8/MiniMax H3/Acceleration/Experimental",
            is_experimental=True,
            description="Explicit MODEL clone. Only exact T2VA video grids in the bundle are allowed; "
                        "no silent resize or unsupported-plan fallback. Masked calls preserve the "
                        "original dense backend and report zero sparse coverage for those calls. "
                        "Relay-style partial-query calls also delegate with an explicit audit. "
                        "Uses PyTorch Flex, not official FA4. "
                        "Windows apply_exp requires --disable-dynamic-vram due to a verified "
                        "native crash. Optional fused tile I/O is experimental and does not "
                        "establish a speedup. Existing unverified Relay/EAV/DiT owners "
                        "are preserved with an explicit Veda-bypassed audit, not sparse coverage.",
            inputs=[
                io.Model.Input("model"),
                io.Custom("T8_VEDA_BUNDLE").Input("bundle"),
                io.Combo.Input("mode", options=["report_only", "apply_exp"], default="report_only"),
                io.Float.Input("keep_ratio", default=0.1, min=0.01, max=1., step=0.01),
                io.Int.Input("max_copy_mib", default=64, min=1, max=256),
                io.Boolean.Input("fused_tile_io_exp", default=False, advanced=True),
            ],
            outputs=[io.Model.Output("model"), io.Custom("T8_VEDA_RUNTIME").Output("runtime")],
        )

    @classmethod
    def execute(cls, model, bundle: VedaBundleRef, mode="report_only",
                keep_ratio=0.1, max_copy_mib=64, fused_tile_io_exp=False):
        patched, runtime = apply_veda(
            model, bundle, mode=mode, keep_ratio=keep_ratio,
            max_copy_mib=max_copy_mib, fused_tile_io=fused_tile_io_exp)
        return io.NodeOutput(patched, runtime)


class MiniMaxH3VedaAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VedaAuditEXPT8",
            display_name="H3 Veda · Audit After Sample (T2VA EXP)",
            category="T8/MiniMax H3/Acceleration/Experimental",
            is_experimental=True,
            is_output_node=True,
            description="Place after the actual sampler to observe Veda calls. "
                        "This report never certifies video quality or end-to-end speedup.",
            inputs=[io.Latent.Input("av_latent"),
                    io.Custom("T8_VEDA_RUNTIME").Input("runtime")],
            outputs=[io.Latent.Output("av_latent"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, av_latent, runtime: VedaRuntime):
        if type(runtime) is not VedaRuntime:
            raise TypeError("Use the paired H3 Veda Apply runtime")
        report = json.dumps(runtime.snapshot(), ensure_ascii=False, indent=2)
        return io.NodeOutput(av_latent, report, ui={"text": (report,)})


VEDA_SPARSE_NODE_CLASSES = [
    MiniMaxH3VedaBundleEXPT8,
    MiniMaxH3VedaApplyEXPT8,
    MiniMaxH3VedaAuditEXPT8,
]
