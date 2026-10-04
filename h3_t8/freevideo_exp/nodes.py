"""Independent FreeVideo LOW/HIGH stages; no inactive LOW graph on cold HIGH."""
from pathlib import Path
import os
import folder_paths
from comfy_api.latest import io
from .runtime import (av_output, canonical, config_for, load_model, load_stage, sample,
                      save_stage, validate_stage, with_lora)
from .split_nodes import NODES as SPLIT_NODES

MODEL = io.Custom("H3_T8_FREEVIDEO_MODEL")
STAGE = io.Custom("H3_T8_FREEVIDEO_STAGE")
CATEGORY = "T8/MiniMax H3/FreeVideo/Experimental"


def default_config():
    return os.environ.get("T8_FREEVIDEO_RUNTIME_CONFIG") or str(Path(folder_paths.get_user_directory()) / "default/T8/freevideo-runtime.json")


def callbacks():
    import comfy.model_management
    import comfy.utils
    bar = comfy.utils.ProgressBar(8)
    return comfy.model_management.throw_exception_if_processing_interrupted, lambda n, total: bar.update_absolute(n, total)


def offload_supplied_clip(clip):
    if clip is None:
        return
    import comfy.model_management
    function = getattr(comfy.model_management, "unload_model_and_clones", None)
    if not callable(function) or not getattr(clip, "patcher", None):
        raise ValueError("This Core cannot explicitly offload the supplied CLIP; use a CPU encoder or update the runtime")
    # Only the explicitly connected CLIP family, not unload_all_models().
    function(clip.patcher, unload_additional_models=False)


class MiniMaxH3FreeVideoLoaderEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="FreeVideo · FP8 流式模型加载 (T8 EXP)", category=CATEGORY,
            is_experimental=True, description="独立 FreeVideo Python/VDN 引擎，不是普通MODEL。先运行prepare工具；模型只读复用。外置效果使用本家族EAV/Relay节点。",
            inputs=[io.String.Input("runtime_config", default="", tooltip="留空读取 user/default/T8/freevideo-runtime.json；不修改主环境")],
            outputs=[MODEL.Output("freevideo_model"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, runtime_config=""):
        model = load_model(runtime_config.strip() or default_config())
        config = config_for(model)
        return io.NodeOutput(model, canonical(dict(model_revision=config["model_revision"],
            model_assets=config["model_asset_count"], boundary=config["boundary"], validated="metadata_only_full_SHA_at_execution")))


class MiniMaxH3FreeVideoLoRAEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="FreeVideo · 串联在线 LoRA (T8 EXP)", category=CATEGORY,
            is_experimental=True, description="串联多个普通attention/FF LoRA，逐目标匹配实际VDN权重。零强度旁路，不隐式融合或下载AdaLN原投影。",
            inputs=[MODEL.Input("freevideo_model"), io.Combo.Input("lora_name", options=folder_paths.get_filename_list("loras") or ["None"]),
                    io.Float.Input("strength", default=1.0, min=-4.0, max=4.0, step=.01)],
            outputs=[MODEL.Output("freevideo_model"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, freevideo_model, lora_name, strength=1.0):
        path = lora_name if strength == 0 else folder_paths.get_full_path_or_raise("loras", lora_name)
        model = with_lora(freevideo_model, path, strength)
        return io.NodeOutput(model, canonical(dict(slots=len(model.loras), disabled=strength == 0, deferred_target_check=strength != 0)))


class MiniMaxH3FreeVideoLOWEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="FreeVideo · LOW 完整八步 (T8 EXP)", category=CATEGORY,
            is_experimental=True, description="固定官方DMD8/AV12-3和匹配AdaLN表，完成8 NFE才输出标准AV latent。可接外置3D放大/VAE。",
            inputs=[MODEL.Input("freevideo_model"), io.Conditioning.Input("conditioning"),
                    io.Int.Input("width", default=448, min=256, max=4096, step=32),
                    io.Int.Input("height", default=256, min=256, max=4096, step=32),
                    io.Int.Input("frames", default=124, min=39, max=2000, step=17),
                    io.Int.Input("seed", default=171, min=0, max=2**63-1, control_after_generate=True),
                    io.Clip.Input("clip_to_offload", optional=True, tooltip="显式释放本次条件编码使用的CLIP，给独立引擎留显存；不卸载其他模型")],
            outputs=[io.Latent.Output("av_latent"), STAGE.Output("completed_stage"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, freevideo_model, conditioning, width=448, height=256, frames=124, seed=171, clip_to_offload=None):
        interrupt, progress = callbacks()
        offload_supplied_clip(clip_to_offload)
        stage = sample(freevideo_model, conditioning, width, height, frames, seed, interrupt=interrupt, progress=progress)
        return io.NodeOutput(av_output(stage), stage, stage.receipt_json)


class MiniMaxH3FreeVideoHIGHEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="FreeVideo · HIGH 独立尾部细化 (T8 EXP)", category=CATEGORY,
            is_experimental=True, description="独立模型/条件/seed，接LOW完成态和外置放大后的标准AV latent；HIGH保留LOW音频，原DMD8尾2步。冷图不依赖LOW节点。",
            inputs=[MODEL.Input("freevideo_model"), io.Conditioning.Input("conditioning"), STAGE.Input("completed_low"),
                    io.Latent.Input("lifted_av_latent"), io.Int.Input("seed", default=172, min=0, max=2**63-1, control_after_generate=True),
                    io.Int.Input("tail_steps", default=2, min=1, max=7),
                    io.Clip.Input("clip_to_offload", optional=True, tooltip="仅释放显式接入的条件编码CLIP")],
            outputs=[io.Latent.Output("av_latent"), STAGE.Output("completed_stage"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, freevideo_model, conditioning, completed_low, lifted_av_latent, seed=172, tail_steps=2, clip_to_offload=None):
        from ..core import nested_av_parts
        receipt = validate_stage(completed_low, "LOW")
        video, _ = nested_av_parts(lifted_av_latent)
        interrupt, progress = callbacks()
        offload_supplied_clip(clip_to_offload)
        stage = sample(freevideo_model, conditioning, video.shape[-1] * 16, video.shape[-2] * 16,
                       receipt["geometry"]["frames"], seed, low=completed_low, lifted=lifted_av_latent,
                       tail_steps=tail_steps, interrupt=interrupt, progress=progress)
        return io.NodeOutput(av_output(stage), stage, stage.receipt_json)


class MiniMaxH3FreeVideoStageSaveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="FreeVideo · 保存完成阶段 (T8 EXP)", category=CATEGORY,
            is_experimental=True, is_output_node=True, description="create-only safetensors+SHA manifest；不覆盖旧缓存。",
            inputs=[STAGE.Input("completed_stage")], outputs=[STAGE.Output("completed_stage"),
                    io.String.Output("manifest_path"), io.String.Output("manifest_sha256")])

    @classmethod
    def execute(cls, completed_stage):
        path, sha = save_stage(completed_stage, Path(folder_paths.get_output_directory()) / "T8-FreeVideo/stages")
        return io.NodeOutput(completed_stage, path, sha, ui={"text": [path, sha]})


class MiniMaxH3FreeVideoStageLoadEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="FreeVideo · 冷加载完成阶段 (T8 EXP)", category=CATEGORY,
            is_experimental=True, description="填实际保存path和完整SHA，不接受占位/partial或其他家族缓存。Cold HIGH只连接本节点，不连旧LOW。",
            inputs=[io.String.Input("manifest_path", default=""), io.String.Input("manifest_sha256", default=""),
                    io.Combo.Input("role", options=["LOW", "HIGH"], default="LOW")],
            outputs=[io.Latent.Output("av_latent"), STAGE.Output("completed_stage"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, manifest_path, manifest_sha256, role="LOW"):
        stage = load_stage(manifest_path, manifest_sha256, role)
        return io.NodeOutput(av_output(stage), stage, stage.receipt_json)


class MiniMaxH3FreeVideoEAVEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="FreeVideo · 外置 EAV (T8 EXP)", category=CATEGORY,
            is_experimental=True, description="LOW/HIGH各自独立分支。全头FETA，目标视频softmax/linear投影前增益；不改音频行/时钟/NFE。report_only不改变数值。",
            inputs=[MODEL.Input("freevideo_model"), io.Combo.Input("mode", options=["disabled", "report_only", "apply_exp"], default="report_only"),
                io.Float.Input("tau", default=4., min=-32., max=32.), io.Float.Input("start", default=.15, min=0., max=1.),
                io.Float.Input("end", default=.9, min=0., max=1.), io.Int.Input("workspace_mib", default=32, min=1, max=1024),
                io.Float.Input("g_hard_limit", default=1.5, min=1., max=10.)],
            outputs=[MODEL.Output("freevideo_model"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, freevideo_model, **kwargs):
        from .effects import with_eav
        model = with_eav(freevideo_model, **kwargs)
        return io.NodeOutput(model, model.eav)


class MiniMaxH3FreeVideoPromptRelayEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        from ..nodes_prompt_relay_advanced import MiniMaxH3PromptRelayConditioningT8Advanced
        original = MiniMaxH3PromptRelayConditioningT8Advanced.define_schema()
        inputs = [MODEL.Input("freevideo_model")]
        for item in original.inputs[1:]:
            if item.id == "audio_mode":
                item = io.Combo.Input("audio_mode", options=["native", "reference_only"], default="native")
            if item.id == "task_type":
                item = io.Combo.Input("task_type", options=["T2VA", "auto", "I2VA", "FL2VA", "L2VA", "Ref2VA"], default="T2VA")
            if item.id == "width":
                item.default = 448
            if item.id == "height":
                item.default = 256
            inputs.append(item)
        inputs.append(io.Int.Input("relay_workspace_mib", default=64, min=4, max=1024, advanced=True,
            tooltip="明确限定新增bias/linear scan/批量text seed的估算预算，不是整个引擎VRAM上限"))
        return io.Schema(node_id=cls.__name__, display_name="FreeVideo · 外置 Prompt Relay 条件 (T8 EXP)", category=CATEGORY,
            is_experimental=True, description="连接现有Plan/Query Route。原生编码并绑定精确token；两个阶段分别接独立计划。保留VDN原window keyset，linear使用显式实验性加权text seed，不冒称论文等价。",
            inputs=inputs, outputs=[MODEL.Output("freevideo_model"), *original.outputs[1:]])

    @classmethod
    def execute(cls, **kwargs):
        from .effects import build_relay_conditioning
        return io.NodeOutput(*build_relay_conditioning(**kwargs))


NODES = [MiniMaxH3FreeVideoLoaderEXPT8, MiniMaxH3FreeVideoLoRAEXPT8, MiniMaxH3FreeVideoLOWEXPT8,
         MiniMaxH3FreeVideoHIGHEXPT8, MiniMaxH3FreeVideoStageSaveEXPT8, MiniMaxH3FreeVideoStageLoadEXPT8]
NODES += [MiniMaxH3FreeVideoEAVEXPT8, MiniMaxH3FreeVideoPromptRelayEXPT8]

# Imported after the original eight class definitions to retain their schemas,
# order and defaults; MID is a different socket from completed LOW.
NODES += SPLIT_NODES
