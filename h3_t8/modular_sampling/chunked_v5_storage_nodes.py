"""Append-only public explicit storage nodes for Chunked v5 joint AV windows."""
from pathlib import Path

from comfy_api.latest import io
import folder_paths

from .chunked_v5_nodes import CATEGORY, PLAN, PREPARED, RESULT
from .chunked_v5_native_source import verified_native_source
from .chunked_v5_storage import load_window, save_window, verify_window
from .results import canonical
from .storage import fingerprint_stage


def _v5_store_root():
    return Path(folder_paths.get_output_directory()) / "MiniMaxH3" / "chunked_v5_window_artifacts"


class MiniMaxH3ChunkedV5WindowSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked v5 · Freeze ONE Joint AV Window (T8 EXP)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="Explicitly freeze one completed v5 PASS2 window with cumulative joint AV. "
                        "Copy both manifest path and exact SHA. This is not automatic MODEL/condition cache reuse.",
            inputs=[io.Custom(RESULT).Input("window_result"),
                    io.Latent.Input("partial4_denoised_output"),
                    io.Latent.Input("lifted_full_av"), io.Custom(PREPARED).Input("prepared"),
                    io.Custom(PLAN).Input("plan"),
                    io.Boolean.Input("confirm_save", default=False)],
            outputs=[io.Latent.Output("cumulative_av_latent"),
                     io.Custom(RESULT).Output("window_result"),
                     io.String.Output("artifact_path"), io.String.Output("artifact_sha256"),
                     io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, window_result, partial4_denoised_output, lifted_full_av,
                prepared, plan, confirm_save=False):
        if not confirm_save:
            verify_window(window_result, partial4_denoised_output, lifted_full_av,
                          prepared, plan)
            report = canonical({"status": "not_saved_confirm_save_false",
                                "automatic_cache_reuse": False})
            return io.NodeOutput(window_result.output_latent, window_result, "", "", report,
                                 ui={"text": (report,)})
        output, result, path, digest, report = save_window(
            window_result, partial4_denoised_output, lifted_full_av, prepared,
            plan, _v5_store_root(),
        )
        return io.NodeOutput(output, result, path, digest, report,
                             ui={"text": ("artifact_path: " + path,
                                          "artifact_sha256: " + digest, report)})


class MiniMaxH3ChunkedV5WindowLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Chunked v5 · Load Exact Frozen Window (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Load an explicitly selected completed v5 window by exact manifest SHA. "
                        "Recheck current source, global lift, prepared noise, plan and window number; "
                        "later windows may run without prior-window sampling. Today's MODEL/conditioning "
                        "is not claimed equivalent to the frozen execution.",
            inputs=[io.Latent.Input("partial4_denoised_output"),
                    io.Latent.Input("lifted_full_av"), io.Custom(PREPARED).Input("prepared"),
                    io.Custom(PLAN).Input("plan"),
                    io.Int.Input("expected_window_index", default=0, min=0, max=9999),
                    io.String.Input("artifact_path", default=""),
                    io.String.Input("artifact_sha256", default="")],
            outputs=[io.Latent.Output("cumulative_av_latent"),
                     io.Custom(RESULT).Output("window_result"),
                     io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, partial4_denoised_output, lifted_full_av, prepared, plan,
                expected_window_index, artifact_path, artifact_sha256):
        return io.NodeOutput(*load_window(
            partial4_denoised_output, lifted_full_av, prepared, plan,
            _v5_store_root(), artifact_path, artifact_sha256, expected_window_index,
        ))

    @classmethod
    def fingerprint_inputs(cls, artifact_path, artifact_sha256, **_kwargs):
        try:
            return fingerprint_stage(_v5_store_root(), artifact_path)
        except (ValueError, OSError, RuntimeError):
            return float("nan")


class MiniMaxH3ChunkedV5VerifiedNativeSourceEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="H3 Chunked v5 · Verify Native Partial4 Source (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Connect every receipt output of the native AV checkpoint Load. "
                        "Recheck its external manifest, then remove only Load provenance "
                        "so the original partial4 source identity can bind v5 windows.",
            inputs=[io.Latent.Input("av_latent"),
                    io.Boolean.Input("resume_verified"),
                    io.String.Input("checkpoint_id"),
                    io.String.Input("content_sha256"),
                    io.String.Input("file_sha256"),
                    io.String.Input("manifest_json"),
                    io.String.Input("report_json")],
            outputs=[io.Latent.Output("partial4_denoised_output"),
                     io.String.Output("verification_report_json")],
        )

    @classmethod
    def execute(cls, av_latent, resume_verified, checkpoint_id,
                content_sha256, file_sha256, manifest_json, report_json):
        return io.NodeOutput(*verified_native_source(
            av_latent, resume_verified, checkpoint_id, content_sha256,
            file_sha256, manifest_json, report_json,
        ))


NODES = [MiniMaxH3ChunkedV5WindowSaveEXPT8,
         MiniMaxH3ChunkedV5WindowLoadEXPT8,
         MiniMaxH3ChunkedV5VerifiedNativeSourceEXPT8]
