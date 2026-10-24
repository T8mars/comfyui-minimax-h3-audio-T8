"""Independent native two-time HyperFlow nodes (full-structure H3 only)."""

from __future__ import annotations

import json
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .hyperflow_runtime_advanced import install_hyperflow
from .hyperflow_sampling_advanced import build_hyperflow_plan, setup_hyperflow_sampler
from .hyperflow_weights_advanced import load_hyperflow_original


CATEGORY = "T8/MiniMax H3/Performance/Experimental"
HYPERFLOW_PLAN_IO = io.Custom("T8_H3_HYPERFLOW_PLAN_V1")
_MODEL_FOLDER = Path(folder_paths.models_dir) / "hyperflow" / "loras"
folder_paths.add_model_folder_path("hyperflow", str(_MODEL_FOLDER))


def _weight_options() -> list[str]:
    dedicated = folder_paths.get_filename_list("hyperflow")
    # Existing raw originals in the regular LoRA directory remain discoverable,
    # but the selector never claims generic acceleration LoRAs are compatible.
    legacy = [
        name for name in folder_paths.get_filename_list("loras")
        if "hyperflow" in Path(name).name.lower() and name.lower().endswith(".safetensors")
    ]
    return [*(f"hyperflow/{name}" for name in dedicated), *(f"loras/{name}" for name in legacy)] or ["missing_hyperflow_weights"]


def _resolve(selection: str) -> Path:
    if selection.startswith("hyperflow/"):
        category, name = "hyperflow", selection.removeprefix("hyperflow/")
    elif selection.startswith("loras/"):
        category, name = "loras", selection.removeprefix("loras/")
    else:
        raise FileNotFoundError("Place the original HyperFlow safetensors in models/hyperflow/loras")
    return Path(folder_paths.get_full_path_or_raise(category, name))


class MiniMaxH3HyperFlowLoaderT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HyperFlowLoaderT8Advanced",
            display_name="MiniMax H3 HyperFlow Loader (Two-Time EXP/T8)",
            category=CATEGORY,
            is_experimental=True,
            description=(
                "Loads the ORIGINAL 632-tensor HyperFlow adapter, maps every LoRA target, "
                "and installs its separate endpoint-time branch. Requires a full non-pruned "
                "MiniMax H3 base and the typed HyperFlow sampler. It does not substitute "
                "a generic acceleration LoRA or silently degrade to a single time."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("hyperflow_file", options=_weight_options()),
            ],
            outputs=[io.Model.Output("model"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, hyperflow_file):
        weights = load_hyperflow_original(_resolve(hyperflow_file))
        patched, _, report = install_hyperflow(model, weights)
        return io.NodeOutput(patched, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


class MiniMaxH3HyperFlowPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HyperFlowPlanT8Advanced",
            display_name="MiniMax H3 HyperFlow 8-Step Plan (EXP/T8)",
            category=CATEGORY,
            is_experimental=True,
            description=(
                "Creates the fixed 9-point/8-forward AV interval plan from the loaded "
                "HyperFlow artifact. It is bound to that exact MODEL, not a bare SIGMAS list."
            ),
            inputs=[io.Model.Input("model")],
            outputs=[HYPERFLOW_PLAN_IO.Output("hyperflow_plan"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model):
        plan = build_hyperflow_plan(model)
        return io.NodeOutput(plan, json.dumps(plan.as_report(), ensure_ascii=False, indent=2, allow_nan=False))


class MiniMaxH3HyperFlowSamplerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HyperFlowSamplerT8Advanced",
            display_name="MiniMax H3 HyperFlow AV Sampler (EXP/T8)",
            category=CATEGORY,
            is_experimental=True,
            description=(
                "Binds the HyperFlow MODEL and typed 8-interval plan to T8 dual-clock Euler. "
                "Connect outputs to the normal SamplerCustomAdvanced alongside native H3 "
                "conditioning; the old H3 sampler and standard 4+4 route are unchanged."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("av_latent"),
                HYPERFLOW_PLAN_IO.Input("hyperflow_plan"),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Sampler.Output("sampler"),
                io.Sigmas.Output("sigmas"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, model, av_latent, hyperflow_plan):
        patched, sampler, sigmas = setup_hyperflow_sampler(model, av_latent, hyperflow_plan)
        return io.NodeOutput(
            patched, sampler, sigmas,
            json.dumps(hyperflow_plan.as_report(), ensure_ascii=False, indent=2, allow_nan=False),
        )


class MiniMaxH3HyperFlowSplitT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HyperFlowSplitT8Advanced",
            display_name="MiniMax H3 HyperFlow Continuous Split (EXP/T8)",
            category=CATEGORY,
            is_experimental=True,
            description=(
                "Two independently content-patched HyperFlow MODELs over one trained 8-step "
                "AV trajectory, with exact x_sigma handoff and no new noise. This is a "
                "same-resolution 4+4 validation, not the legacy Director 4+4 and not "
                "a low-to-high upscale. Clone one common full H3 base for both stages."
            ),
            inputs=[
                io.Model.Input("model_low"), io.Model.Input("model_high"),
                io.Conditioning.Input("positive"), io.Latent.Input("av_latent"),
                io.Int.Input("seed", default=20260921, min=0, max=2**64 - 1, control_after_generate=True),
                io.Int.Input("split_interval", default=4, min=1, max=7),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1),
                io.Conditioning.Input("negative", optional=True),
                io.Conditioning.Input("positive_high", optional=True),
                io.Conditioning.Input("negative_high", optional=True),
            ],
            outputs=[io.Latent.Output("av_latent"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model_low, model_high, positive, av_latent, seed, split_interval, cfg,
                negative=None, positive_high=None, negative_high=None):
        from .hyperflow_two_pass_advanced import sample_hyperflow_split
        from comfy.utils import ProgressBar
        progress = ProgressBar(8)

        def callback(index, _prediction, _state, _total):
            progress.update_absolute(index + 1, 8)

        return io.NodeOutput(*sample_hyperflow_split(
            model_low, model_high, positive, av_latent, seed=seed,
            split_interval=split_interval, cfg=cfg, negative=negative,
            positive_high=positive_high, negative_high=negative_high,
            callback=callback,
        ))


class MiniMaxH3HyperFlowTailPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HyperFlowTailPlanT8Advanced",
            display_name="MiniMax H3 HyperFlow Tail 4 Plan (EXP/T8)",
            category=CATEGORY,
            is_experimental=True,
            description="Absolute trained intervals 4:8, for explicit new-noise 8+4 upscale experiment only.",
            inputs=[io.Model.Input("model")],
            outputs=[HYPERFLOW_PLAN_IO.Output("hyperflow_plan"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model):
        plan = build_hyperflow_plan(model, 4, 8)
        return io.NodeOutput(plan, json.dumps(plan.as_report(), ensure_ascii=False, allow_nan=False))


class MiniMaxH3HyperFlowHeadPlanT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HyperFlowHeadPlanT8Advanced",
            display_name="MiniMax H3 HyperFlow Head 4 Plan (EXP/T8)",
            category=CATEGORY,
            is_experimental=True,
            description="Absolute trained intervals 0:4 for explicit partial-x0 low-to-high 4+4 upscale only.",
            inputs=[io.Model.Input("model")],
            outputs=[HYPERFLOW_PLAN_IO.Output("hyperflow_plan"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model):
        plan = build_hyperflow_plan(model, 0, 4)
        return io.NodeOutput(plan, json.dumps(plan.as_report(), ensure_ascii=False, allow_nan=False))


class MiniMaxH3HyperFlowCoarseSamplerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HyperFlowCoarseSamplerT8Advanced",
            display_name="MiniMax H3 HyperFlow Partial 4 Sampler (EXP/T8)",
            category=CATEGORY,
            is_experimental=True,
            description=(
                "Runs only absolute intervals 0:4. Feed SamplerCustomAdvanced's "
                "denoised_output (socket 1), not its nonterminal x_sigma result, "
                "to the learned 3D upscaler. HIGH then starts a separately "
                "re-noised 4:8 tail; this is not a continuous trajectory."
            ),
            inputs=[io.Model.Input("model"), io.Latent.Input("av_latent"),
                    HYPERFLOW_PLAN_IO.Input("hyperflow_plan")],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"),
                     io.Sigmas.Output("sigmas"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, av_latent, hyperflow_plan):
        if (hyperflow_plan.start_interval, hyperflow_plan.stop_interval) != (0, 4):
            raise ValueError("HyperFlow partial upscale requires the absolute 0:4 Head Plan")
        patched, sampler, sigmas = setup_hyperflow_sampler(
            model, av_latent, hyperflow_plan, internal_continuation=True,
        )
        report = {
            **hyperflow_plan.as_report(),
            "recipe": "hyperflow4plus4_partial_x0_upscale_exp_v1",
            "handoff": "low_denoised_output_learned_3d_high_new_noise",
            "quality_status": "unverified_experimental",
        }
        return io.NodeOutput(patched, sampler, sigmas, json.dumps(report, ensure_ascii=False, allow_nan=False))


class MiniMaxH3HyperFlowRefineSamplerT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HyperFlowRefineSamplerT8Advanced",
            display_name="MiniMax H3 HyperFlow Tail New-Noise Sampler (EXP/T8)",
            category=CATEGORY,
            is_experimental=True,
            description=(
                "Only for LOW full8 -> learned 3D latent upscale -> HIGH 4-interval new-noise "
                "experiment (12 total NFE). This is not a seamless 4+4 continuation. "
                "Connect the HIGH stage MODEL, upscaled clean AV latent and Tail Plan; "
                "the generic SamplerCustomAdvanced supplies a distinct noise seed."
            ),
            inputs=[
                io.Model.Input("model"), io.Latent.Input("av_latent"),
                HYPERFLOW_PLAN_IO.Input("hyperflow_plan"),
            ],
            outputs=[
                io.Model.Output("model"), io.Sampler.Output("sampler"),
                io.Sigmas.Output("sigmas"), io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, model, av_latent, hyperflow_plan):
        if (hyperflow_plan.start_interval, hyperflow_plan.stop_interval) != (4, 8):
            raise ValueError("HyperFlow refine requires the absolute 4:8 Tail Plan")
        patched, sampler, sigmas = setup_hyperflow_sampler(
            model, av_latent, hyperflow_plan,
            internal_continuation=True, new_noise_restart=True,
        )
        report = {
            **hyperflow_plan.as_report(),
            "recipe": "hyperflow8plus4_new_noise_upscale_exp_v1",
            "handoff": "explicit_clean_latent_new_noise_dual_av_clocks",
            "quality_status": "unverified_experimental",
        }
        return io.NodeOutput(patched, sampler, sigmas, json.dumps(report, ensure_ascii=False, allow_nan=False))


class MiniMaxH3HyperFlowPartialRefineSamplerT8Advanced(MiniMaxH3HyperFlowRefineSamplerT8Advanced):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "MiniMaxH3HyperFlowPartialRefineSamplerT8Advanced"
        schema.display_name = "MiniMax H3 HyperFlow Partial 4+4 Refine (EXP/T8)"
        schema.description = (
            "Only for LOW absolute 0:4 denoised_output -> learned 3D upscale "
            "-> HIGH absolute 4:8 new-noise sampler (8 total NFE)."
        )
        return schema

    @classmethod
    def execute(cls, model, av_latent, hyperflow_plan):
        if (hyperflow_plan.start_interval, hyperflow_plan.stop_interval) != (4, 8):
            raise ValueError("HyperFlow partial 4+4 refine requires the absolute 4:8 Tail Plan")
        patched, sampler, sigmas = setup_hyperflow_sampler(
            model, av_latent, hyperflow_plan,
            internal_continuation=True, new_noise_restart=True,
        )
        report = {
            **hyperflow_plan.as_report(),
            "recipe": "hyperflow4plus4_partial_x0_upscale_exp_v1",
            "handoff": "low_partial_x0_learned_3d_high_new_noise_dual_av_clocks",
            "quality_status": "unverified_experimental",
        }
        return io.NodeOutput(patched, sampler, sigmas, json.dumps(report, ensure_ascii=False, allow_nan=False))


class MiniMaxH3HyperFlowLatentParityT8Advanced(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HyperFlowLatentParityT8Advanced",
            display_name="MiniMax H3 HyperFlow Latent Parity Audit (EXP/T8)",
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            description="Read-only video/audio latent difference between continuous split and single8.",
            inputs=[io.Latent.Input("single_av_latent"), io.Latent.Input("split_av_latent")],
            outputs=[io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, single_av_latent, split_av_latent):
        import torch
        from .core import nested_av_parts

        baseline = nested_av_parts(single_av_latent)
        candidate = nested_av_parts(split_av_latent)
        measures = {}
        for name, a, b in zip(("video", "audio"), baseline, candidate):
            if a.shape != b.shape:
                raise ValueError(f"HyperFlow parity {name} shape mismatch")
            delta = a.detach().to(torch.float32) - b.detach().to(torch.float32)
            measures[name] = {
                "shape": list(a.shape),
                "max_abs": float(delta.abs().max()),
                "mean_abs": float(delta.abs().mean()),
                "rms": float(delta.square().mean().sqrt()),
            }
        report = json.dumps({
            "schema": "t8.minimax_h3.hyperflow.latent_parity.v1",
            "status": "measured_not_quality_certified", "measures": measures,
        }, ensure_ascii=False, allow_nan=False)
        return io.NodeOutput(report, ui={"text": (report,)})


HYPERFLOW_ADVANCED_NODE_CLASSES = [
    MiniMaxH3HyperFlowLoaderT8Advanced,
    MiniMaxH3HyperFlowPlanT8Advanced,
    MiniMaxH3HyperFlowSamplerT8Advanced,
    MiniMaxH3HyperFlowSplitT8Advanced,
    MiniMaxH3HyperFlowHeadPlanT8Advanced,
    MiniMaxH3HyperFlowCoarseSamplerT8Advanced,
    MiniMaxH3HyperFlowTailPlanT8Advanced,
    MiniMaxH3HyperFlowRefineSamplerT8Advanced,
    MiniMaxH3HyperFlowPartialRefineSamplerT8Advanced,
    MiniMaxH3HyperFlowLatentParityT8Advanced,
]
