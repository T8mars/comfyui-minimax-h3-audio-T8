"""Append-only explicit P7 LOW/HIGH save/load nodes."""
from comfy_api.latest import io

from . import hyperflow_p7_storage as frozen
from .hyperflow_p7_nodes import HIGH_INPUT, HIGH_RESULT, LOW_RESULT, PHASE, schema


class MiniMaxH3HyperFlowP7LowSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Save Exact LOW 0:4",
            "Write a new immutable P7-only LOW artifact. Output path and SHA are mandatory for explicit "
            "HIGH-only restoration; old HyperFlow cache and general stage store are not reused. "
            "Unknown executable stacks cannot claim portable restoration.",
            [io.Custom(LOW_RESULT).Input("low_result")],
            [io.Custom(LOW_RESULT).Output("low_result"), io.Latent.Output("low_x0"),
             io.String.Output("artifact_path"), io.String.Output("artifact_sha256"),
             io.String.Output("report_json")])

    @classmethod
    def execute(cls, low_result):
        return io.NodeOutput(*frozen.save_low(low_result))


class MiniMaxH3HyperFlowP7LowLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Load Exact LOW for HIGH Only",
            "Reload a selected completed LOW manifest by path and SHA under this chain/segment's separate "
            "P7 namespace. Recheck the accepted parent and freshly prepared LOW phase, but do not load "
            "a LOW MODEL, draw LOW noise or sample LOW. Frozen selection is not an automatic cache hit "
            "against edited LOW model/effects/noise settings.",
            [io.Custom(PHASE).Input("low_phase"), io.String.Input("artifact_path", default=""),
             io.String.Input("artifact_sha256", default="")],
            [io.Custom(LOW_RESULT).Output("low_result"), io.Latent.Output("low_x0"),
             io.String.Output("report_json")])

    @classmethod
    def execute(cls, low_phase, artifact_path, artifact_sha256):
        return io.NodeOutput(*frozen.load_low(low_phase, artifact_path, artifact_sha256))

    @classmethod
    def fingerprint_inputs(cls, low_phase, artifact_path, artifact_sha256):
        try:
            if low_phase is None:
                return frozen.fingerprint_selected(artifact_path, frozen.LOW_STAGE)
            return frozen.fingerprint(low_phase, artifact_path)
        except (ValueError, OSError, RuntimeError):
            return float("nan")


class MiniMaxH3HyperFlowP7HighSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Save Exact HIGH 4:8",
            "Write a new immutable P7-only completed joint AV result with the exact handoff and "
            "sampler receipt. No old P7 cache migration or implicit segment acceptance.",
            [io.Custom(HIGH_RESULT).Input("high_result")],
            [io.Custom(HIGH_RESULT).Output("high_result"), io.Latent.Output("completed_av"),
             io.String.Output("artifact_path"), io.String.Output("artifact_sha256"),
             io.String.Output("report_json")])

    @classmethod
    def execute(cls, high_result):
        return io.NodeOutput(*frozen.save_high(high_result))


class MiniMaxH3HyperFlowP7HighLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return schema(cls, "Load Exact Completed HIGH",
            "Reload a selected completed HIGH manifest by path and SHA. Recheck current P7 parent, "
            "LOW lift, HIGH conditions and reconciled handoff; no diffusion run or implicit acceptance.",
            [io.Custom(HIGH_INPUT).Input("high_handoff"), io.String.Input("artifact_path", default=""),
             io.String.Input("artifact_sha256", default="")],
            [io.Custom(HIGH_RESULT).Output("high_result"), io.Latent.Output("completed_av"),
             io.String.Output("report_json")])

    @classmethod
    def execute(cls, high_handoff, artifact_path, artifact_sha256):
        return io.NodeOutput(*frozen.load_high(high_handoff, artifact_path, artifact_sha256))

    @classmethod
    def fingerprint_inputs(cls, high_handoff, artifact_path, artifact_sha256):
        try:
            if high_handoff is None:
                return frozen.fingerprint_selected(artifact_path, frozen.HIGH_STAGE)
            return frozen.fingerprint(high_handoff.phase, artifact_path)
        except (ValueError, OSError, RuntimeError):
            return float("nan")


NODES = [MiniMaxH3HyperFlowP7LowSaveEXPT8, MiniMaxH3HyperFlowP7LowLoadEXPT8,
         MiniMaxH3HyperFlowP7HighSaveEXPT8, MiniMaxH3HyperFlowP7HighLoadEXPT8]
