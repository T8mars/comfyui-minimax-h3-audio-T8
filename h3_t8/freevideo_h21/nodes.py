"""Append-only v4 entrypoints; the project owner registers this independent list."""
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from . import runtime, stages
from .source import canonical

MODEL = io.Custom("H3_T8_FREEVIDEO_H21_MODEL")
STAGE = io.Custom("H3_T8_FREEVIDEO_H21_STAGE")
CATEGORY = "T8/MiniMax H3/FreeVideo/H21 v4 Experimental"


def output(stage):
    from comfy.nested_tensor import NestedTensor
    stages.validate(stage)
    return {"samples": NestedTensor((stage.video.clone(), stage.audio.clone()))}


def callbacks(total):
    from comfy.model_management import throw_exception_if_processing_interrupted
    from comfy.utils import ProgressBar
    bar = ProgressBar(total)
    return throw_exception_if_processing_interrupted, lambda completed, count: bar.update_absolute(completed, count)


class MiniMaxH3FreeVideoH21LoaderEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="FreeVideo H21 v4 · 独立模型配置", category=CATEGORY,
            is_experimental=True, description="显式新v4配置；不读取或修改旧FreeVideo默认。实际SHA在执行前后复核。",
            inputs=[io.String.Input("runtime_config", default="")],
            outputs=[MODEL.Output("freevideo_model"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, runtime_config):
        model = runtime.load_model(runtime_config)
        config = runtime.config_for(model)
        return io.NodeOutput(model, canonical(dict(schema=config["schema"], revision=config["freevideo_revision"],
            prepared_tiers=config["prepared_tiers"], prepared_tasks=config["prepared_tasks"],
            GPU_quality_qualified=False, full_SHA_at_execution=True)))


class MiniMaxH3FreeVideoH21SamplerEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="FreeVideo H21 v4 · 新四档采样", category=CATEGORY,
            is_experimental=True, description="新Light/Medium均8+3，High20 EasyCache-AV，Max20全算。"
                "FULL自动执行真实learned3D；LOW只完成8步供外置放大及HIGH3。小画布回落如实记录。",
            inputs=[MODEL.Input("freevideo_model"), io.Conditioning.Input("conditioning"),
                io.Combo.Input("tier", options=["light", "medium", "high", "max"], default="light"),
                io.Combo.Input("role", options=["FULL", "LOW"], default="FULL"),
                io.Int.Input("width", default=1024, min=256, step=32),
                io.Int.Input("height", default=576, min=256, step=32),
                io.Int.Input("frames", default=124, min=39, step=17),
                io.Int.Input("seed", default=171, min=0, max=2**64 - 1, control_after_generate=True),
                io.Clip.Input("clip_to_offload", optional=True)],
            outputs=[io.Latent.Output("av_latent"), STAGE.Output("completed_stage"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, freevideo_model, conditioning, tier="light", role="FULL", width=1024, height=576, frames=124, seed=171,
                clip_to_offload=None):
        from ..freevideo_exp.nodes import offload_supplied_clip
        offload_supplied_clip(clip_to_offload)
        interrupt, progress = callbacks(11 if role == "FULL" and tier in ("light", "medium") else 8 if role == "LOW" else 20)
        stage = runtime.sample(freevideo_model, conditioning, width, height, frames, seed,
                               tier=tier, role=role, interrupt=interrupt, progress=progress)
        return io.NodeOutput(output(stage), stage, stage.receipt_json)


class MiniMaxH3FreeVideoH21HIGH3EXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="FreeVideo H21 v4 · 独立HIGH3", category=CATEGORY,
            is_experimental=True, description="接v4完成LOW及真实放大、裁边到目标画布的AV。只跑3步，保留LOW音频；"
                "MODEL/有序LoRA可显式独立选择，不把换模型当禁令。",
            inputs=[MODEL.Input("freevideo_model"), io.Conditioning.Input("conditioning"), STAGE.Input("completed_low"),
                io.Latent.Input("lifted_av_latent"), io.Int.Input("seed", default=171, min=0, max=2**64 - 1),
                io.Clip.Input("clip_to_offload", optional=True)],
            outputs=[io.Latent.Output("av_latent"), STAGE.Output("completed_stage"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, freevideo_model, conditioning, completed_low, lifted_av_latent, seed=171, clip_to_offload=None):
        from ..core import nested_av_parts
        from ..freevideo_exp.nodes import offload_supplied_clip
        receipt = stages.validate(completed_low)
        video, audio = nested_av_parts(lifted_av_latent)
        if stages.tensor_record(audio) != stages.tensor_record(completed_low.audio):
            raise ValueError("External lift changed completed LOW audio")
        canvas = receipt["plan"]["actual_canvas"]
        interrupt, progress = callbacks(3)
        offload_supplied_clip(clip_to_offload)
        stage = runtime.sample(freevideo_model, conditioning, canvas["width"], canvas["height"], canvas["frames"], seed,
            tier=receipt["plan"]["tier"], role="HIGH", low=completed_low, lifted=video, interrupt=interrupt, progress=progress)
        return io.NodeOutput(output(stage), stage, stage.receipt_json)


class MiniMaxH3FreeVideoH21LoRAEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="FreeVideo H21 v4 · 有序LoRA", category=CATEGORY,
            is_experimental=True, inputs=[MODEL.Input("freevideo_model"),
                io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras")),
                io.Float.Input("strength", default=1.)],
            outputs=[MODEL.Output("freevideo_model"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, freevideo_model, lora_name, strength=1.):
        path = lora_name if strength == 0 else folder_paths.get_full_path_or_raise("loras", lora_name)
        model = runtime.with_lora(freevideo_model, path, strength)
        return io.NodeOutput(model, canonical(dict(ordered_slots=list(model.loras))))


class MiniMaxH3FreeVideoH21StageSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="FreeVideo H21 v4 · 保存完成阶段", category=CATEGORY,
            is_experimental=True, is_output_node=True, inputs=[STAGE.Input("completed_stage")],
            outputs=[STAGE.Output("completed_stage"), io.String.Output("manifest_path"), io.String.Output("manifest_sha256")])

    @classmethod
    def execute(cls, completed_stage):
        path, sha = stages.save(completed_stage, Path(folder_paths.get_output_directory()) / "T8-FreeVideo/h21-stages")
        return io.NodeOutput(completed_stage, path, sha, ui={"text": [path, sha]})


class MiniMaxH3FreeVideoH21StageLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="FreeVideo H21 v4 · 冷加载完成阶段", category=CATEGORY,
            is_experimental=True, inputs=[io.String.Input("manifest_path", default=""), io.String.Input("manifest_sha256", default="")],
            outputs=[io.Latent.Output("av_latent"), STAGE.Output("completed_stage"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, manifest_path, manifest_sha256):
        stage = stages.load(manifest_path, manifest_sha256)
        return io.NodeOutput(output(stage), stage, stage.receipt_json)


def effect_schema(schema, node_id, display):
    schema.node_id, schema.category, schema.display_name = node_id, CATEGORY, display
    schema.inputs[0] = MODEL.Input("freevideo_model")
    schema.outputs[0] = MODEL.Output("freevideo_model")
    return schema


class MiniMaxH3FreeVideoH21EAVEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        from ..freevideo_exp.nodes import MiniMaxH3FreeVideoEAVEXPT8
        return effect_schema(MiniMaxH3FreeVideoEAVEXPT8.define_schema(), cls.__name__, "FreeVideo H21 v4 · 外置EAV")

    @classmethod
    def execute(cls, freevideo_model, **kwargs):
        from .effects import with_eav
        model = with_eav(freevideo_model, **kwargs)
        return io.NodeOutput(model, model.effects_json)


class MiniMaxH3FreeVideoH21PromptRelayEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        from ..freevideo_exp.nodes import MiniMaxH3FreeVideoPromptRelayEXPT8
        return effect_schema(MiniMaxH3FreeVideoPromptRelayEXPT8.define_schema(), cls.__name__, "FreeVideo H21 v4 · 外置Relay条件")

    @classmethod
    def execute(cls, **kwargs):
        from .effects import build_relay_conditioning
        return io.NodeOutput(*build_relay_conditioning(**kwargs))


NODES = [MiniMaxH3FreeVideoH21LoaderEXPT8, MiniMaxH3FreeVideoH21SamplerEXPT8, MiniMaxH3FreeVideoH21HIGH3EXPT8,
         MiniMaxH3FreeVideoH21LoRAEXPT8, MiniMaxH3FreeVideoH21StageSaveEXPT8, MiniMaxH3FreeVideoH21StageLoadEXPT8,
         MiniMaxH3FreeVideoH21EAVEXPT8, MiniMaxH3FreeVideoH21PromptRelayEXPT8]
