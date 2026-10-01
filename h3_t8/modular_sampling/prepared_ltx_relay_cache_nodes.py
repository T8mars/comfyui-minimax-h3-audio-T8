"""Append-only explicit Prepared LTX event cache encoder, separate from sampling."""
import json
import folder_paths
from comfy_api.latest import io

from .ltx_relay_plan import PLAN_TYPE
from .prepared_ltx_nodes import BUNDLE, CATEGORY


class MiniMaxH3PreparedLTXRelayEncodeEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        names = folder_paths.get_filename_list("text_encoders")
        return io.Schema(node_id=cls.__name__, display_name="Prepared LTX · Encode Relay Event Caches (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Explicit isolated native INT8 Gemma/AV connector event encoding. No diffusion sampler "
                "or VAE. Requires the pinned prepared-cache provider and its matching native INT8 checkpoints. "
                "64 GiB free host RAM and 2 GiB free GPU runtime reserve. Changed text/source needs a new cache_id.",
            inputs=[BUNDLE.Input("prepared_bundle"), io.Custom(PLAN_TYPE).Input("ltx_relay_plan"),
                io.Combo.Input("text_encoder_name", options=names),
                io.String.Input("cache_id", default="prepared_ltx_relay_01"),
                io.Boolean.Input("resume_existing", default=True),
                io.String.Input("serial_lease_path", default="", advanced=True)],
            outputs=[io.String.Output("relay_cache_manifest_path"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, prepared_bundle, ltx_relay_plan, text_encoder_name, cache_id,
                resume_existing, serial_lease_path):
        from comfy.model_management import throw_exception_if_processing_interrupted
        from .prepared_ltx_relay_cache import run_cache_encoding
        path = (folder_paths.get_full_path_or_raise("text_encoders", text_encoder_name)
                if ltx_relay_plan.get("events") else "")
        manifest, report = run_cache_encoding(prepared_bundle, ltx_relay_plan, gemma_path=path,
            output_directory=folder_paths.get_output_directory(), cache_id=cache_id,
            resume_existing=resume_existing, lease_path=serial_lease_path,
            interrupt=throw_exception_if_processing_interrupted)
        return io.NodeOutput(manifest, json.dumps(report, ensure_ascii=False, indent=2))

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")


NODES = [MiniMaxH3PreparedLTXRelayEncodeEXPT8]
