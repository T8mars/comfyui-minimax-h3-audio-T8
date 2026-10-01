"""Append-only explicit SPEED stage Save/Load nodes."""
from comfy_api.latest import io

from . import speed_storage as frozen
from .speed_nodes import CATEGORY, PLAN, RESULT, SPEC


class MiniMaxH3SPEEDStageSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 SPEED · Save ONE Frozen Stage (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Writes a new immutable SPEED-only artifact by completed typed result. "
                        "Returns an exact path and SHA for explicit later selection. No automatic cache lookup. "
                        "Unknown MODEL, noise or effect stacks cannot claim portable restoration; "
                        "authenticated external Stage EAV and SPEED Relay calls are supported.",
            inputs=[io.Custom(RESULT).Input("stage_result")],
            outputs=[io.Custom(RESULT).Output("stage_result"),
                     io.String.Output("artifact_path"), io.String.Output("artifact_sha256"),
                     io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, stage_result):
        return io.NodeOutput(*frozen.save_speed_stage(stage_result))


class MiniMaxH3SPEEDStageLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 SPEED · Load Exact Previous Stage (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Loads the selected completed previous stage by path+SHA under the SPEED-only store. "
                        "Connect prior_spec to the next Stage Setup and completed_stage to DCT Transition; "
                        "no previous MODEL, noise, conditioning encode or sampler runs. This freezes a selected "
                        "past stage, not an automatic cache hit against edited past-stage settings.",
            inputs=[io.Custom(PLAN).Input("speed_plan"),
                    io.Int.Input("seed", default=2608184001, min=0, max=0xFFFFFFFFFFFFFFFF),
                    io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01),
                    io.Int.Input("next_stage_index", default=1, min=1, max=99),
                    io.String.Input("artifact_path", default=""),
                    io.String.Input("artifact_sha256", default="")],
            outputs=[io.Custom(RESULT).Output("completed_stage"),
                     io.Custom(SPEC).Output("previous_spec"),
                     io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, speed_plan, seed, shift_audio, next_stage_index, artifact_path, artifact_sha256):
        return io.NodeOutput(*frozen.load_speed_stage(
            speed_plan, seed, shift_audio, next_stage_index, artifact_path, artifact_sha256,
        ))

    @classmethod
    def fingerprint_inputs(cls, speed_plan, seed, shift_audio, next_stage_index,
                           artifact_path, artifact_sha256):
        try:
            return frozen.fingerprint_speed_stage(artifact_path)
        except (ValueError, OSError, RuntimeError):
            return float("nan")


NODES = [MiniMaxH3SPEEDStageSaveEXPT8, MiniMaxH3SPEEDStageLoadEXPT8]
