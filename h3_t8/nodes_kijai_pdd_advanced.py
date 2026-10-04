"""Append-only Kijai relative-head setup; old T8 PDD schema stays untouched."""
from __future__ import annotations

import folder_paths
from comfy_api.latest import io

from .kijai_pdd_advanced import build_kijai_pdd_8step_setup


class MiniMaxH3KijaiPDD8StepSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        names = [name for name in folder_paths.get_filename_list("loras")
                 if "acc-8step" in name.lower() and name.lower().endswith(("_comfy.safetensors", "_comfy.sft"))]
        return io.Schema(
            node_id="MiniMaxH3KijaiPDD8StepSetupEXPT8",
            display_name="MiniMax H3 Kijai PDD 8-Step · Full / Pruned (T8 EXP)",
            category="T8/MiniMax H3/Performance/Experimental",
            is_experimental=True,
            description="Kijai *_Acc-8Step_comfy / pruned_comfy 原件。保留全部32组视频/音频头及AdaLN/bias；"
                        "接现有PDD Stage Setup实现绝对0:4/4:8分离，EAV/Relay可外置。不要接旧T8绝对bank格式。",
            inputs=[io.Model.Input("model"), io.Latent.Input("av_latent"),
                    io.Combo.Input("pdd_lora_name", options=names or ["安装 Kijai Acc-8Step 原件"],
                                   tooltip="选择与底模对应的Full/Pruned、FL2VA/Ref2VA文件；真实形状不符会报错。"),
                    io.Combo.Input("base_variant", options=["FL2VA", "Ref2VA"], default="FL2VA",
                                   tooltip="同形状不证明训练基模相同，必须正确选择FL/Ref。"),
                    io.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01,
                                   tooltip="alpha/rank已烘焙。通常1.0；较低值同时插值全部backbone和输出头，未作效果认证。")],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"),
                     io.Sigmas.Output("sigmas"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, av_latent, pdd_lora_name, base_variant="FL2VA", strength=1.0):
        path = folder_paths.get_full_path_or_raise("loras", pdd_lora_name)
        return io.NodeOutput(*build_kijai_pdd_8step_setup(model, av_latent, path,
                                                      base_variant=base_variant, strength=strength))


NODES = [MiniMaxH3KijaiPDD8StepSetupEXPT8]
