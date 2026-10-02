"""Explicit opt-in RF clock correction; existing node schemas stay intact."""
from comfy_api.latest import io
from .nodes import MiniMaxH3RFRestartStageSetupEXPT8
from . import rf_stages


class MiniMaxH3RFRestartJointClockSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        old = MiniMaxH3RFRestartStageSetupEXPT8.define_schema()
        return io.Schema(node_id="MiniMaxH3RFRestartJointClockSetupEXPT8",
            display_name="H3 RF · Restart Joint Clock (No Double Audio Rebase) (T8 EXP)",
            category="T8/MiniMax H3/Modular Sampling/Experimental", is_experimental=True,
            description="Explicit corrected RF restart: each AV stream is re-noised once on its own clock. "
                "Keeps joint AV sampling and original inpaint anchor/noise/masks. Legacy RF nodes/graphs remain "
                "unchanged; replace only Restart Setup to opt in. Audio quality still needs listening review.",
            inputs=old.inputs, outputs=old.outputs)

    @classmethod
    def execute(cls, model, rf_handoff, shift_video=12., shift_audio=3., restart_video_sigma=.15,
                restart_steps=3, restart_seed=1234):
        prepared, sampler, sigmas, context, report = rf_stages.build_restart_stage(model, rf_handoff,
            shift_video, shift_audio, restart_video_sigma, restart_steps, restart_seed,
            audio_start_policy="already_joint_renoised")
        return io.NodeOutput(prepared, sampler, sigmas, rf_handoff.completed_av, context, report)


NODES = [MiniMaxH3RFRestartJointClockSetupEXPT8]
