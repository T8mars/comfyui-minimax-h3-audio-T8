from __future__ import annotations

from comfy_api.latest import io

from .trajectory_probe_advanced import (
    build_trajectory_probe,
    canonical_json,
    load_trajectory_checkpoint,
    save_trajectory_checkpoint,
    prepare_trajectory_model,
    TrajectoryResumeNoise,
)


CATEGORY = "T8/MiniMax H3/Models/Experimental"
TrajectoryIO = io.Custom("H3_T8_TRAJECTORY_PROBE")


class MiniMaxH3TrajectoryProbeT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TrajectoryProbeT8Advanced",
            display_name="MiniMax H3 Trajectory Probe / 采样轨迹探针 (Advanced)",
            description=(
                "Splits only the stateless T8 dual_clock_euler sigma schedule and binds the "
                "checkpoint contract to the exact same-session MODEL/SAMPLER identity. It does "
                "not sample or write files."
            ),
            category=CATEGORY,
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("av_latent"),
                io.Int.Input("split_step", default=2, min=1, max=9999),
                io.Float.Input(
                    "maximum_checkpoint_mib",
                    default=4096.0,
                    min=1.0,
                    max=65536.0,
                    step=64.0,
                ),
                io.Int.Input("noise_seed", default=0, min=0, max=0xffffffffffffffff),
            ],
            outputs=[
                TrajectoryIO.Output("trajectory_contract"),
                io.Sigmas.Output("high_sigmas"),
                io.Sigmas.Output("low_sigmas"),
                io.String.Output("report_json"),
                io.Model.Output("trajectory_model"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        sampler,
        sigmas,
        av_latent,
        split_step,
        maximum_checkpoint_mib,
        noise_seed,
    ):
        trajectory_model = prepare_trajectory_model(model)
        contract, high, low = build_trajectory_probe(
            trajectory_model,
            sampler,
            sigmas,
            split_step,
            maximum_checkpoint_mib,
            av_latent,
            noise_seed,
        )
        return io.NodeOutput(
            contract,
            high,
            low,
            canonical_json(contract),
            trajectory_model,
        )


class MiniMaxH3TrajectoryCheckpointSaveT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TrajectoryCheckpointSaveT8Advanced",
            display_name="MiniMax H3 Trajectory Checkpoint Save / 轨迹保存 (Advanced)",
            description=(
                "Atomically saves the first-stage sampled AV latent only after explicit "
                "confirmation. The checkpoint stays inside the ComfyUI output directory."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                TrajectoryIO.Input("trajectory_contract"),
                io.Latent.Input("latent"),
                io.String.Input("checkpoint_name", default="h3_probe"),
                io.Boolean.Input("confirm_save", default=False),
            ],
            outputs=[
                io.String.Output("checkpoint_path"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, trajectory_contract, latent, checkpoint_name, confirm_save):
        path, report = save_trajectory_checkpoint(
            trajectory_contract,
            latent,
            checkpoint_name,
            confirm_save,
        )
        return io.NodeOutput(path, canonical_json(report))


class MiniMaxH3TrajectoryCheckpointLoadT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TrajectoryCheckpointLoadT8Advanced",
            display_name="MiniMax H3 Trajectory Checkpoint Load / 轨迹续跑 (Advanced)",
            description=(
                "Loads an internal-x_sigma checkpoint produced through Trajectory Probe trajectory_model. "
                "Connect checkpoint_latent, remaining_sigmas, resume_noise, and the same "
                "trajectory_model to the second SamplerCustomAdvanced stage."
            ),
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[
                io.String.Input("checkpoint_path", default="", force_input=True),
                io.Model.Input("model"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
            ],
            outputs=[
                io.Latent.Output("checkpoint_latent"),
                io.Sigmas.Output("remaining_sigmas"),
                io.String.Output("report_json"),
                io.Noise.Output("resume_noise"),
            ],
        )

    @classmethod
    def execute(cls, checkpoint_path, model, sampler, sigmas):
        latent, report, remaining = load_trajectory_checkpoint(
            checkpoint_path,
            model,
            sampler,
            sigmas,
        )
        return io.NodeOutput(
            latent,
            remaining,
            canonical_json(report),
            TrajectoryResumeNoise(report.get("noise_seed", 0)),
        )

    @classmethod
    def fingerprint_inputs(cls, checkpoint_path, **_kwargs):
        from pathlib import Path

        path = Path(str(checkpoint_path or ""))
        try:
            stat = path.stat()
            return f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            return f"missing:{checkpoint_path}"


TRAJECTORY_PROBE_ADVANCED_NODE_CLASSES = [
    MiniMaxH3TrajectoryProbeT8Advanced,
    MiniMaxH3TrajectoryCheckpointSaveT8Advanced,
    MiniMaxH3TrajectoryCheckpointLoadT8Advanced,
]
