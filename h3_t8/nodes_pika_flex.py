from comfy_api.latest import io

from .pika_flex import prepare_pika_flex


class MiniMaxH3PikaFlexSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 Pika Flex · 6步采样准备 (EXP/T8)",
            category="T8/MiniMax H3/Sampling/Experimental", is_experimental=True,
            description="先用现有LoRA加载器加载Pika FL2VA或Ref2VA，建议强度1。"
                        "本节点提供作者6步sigma、LCM重新加噪及12/3音视频时钟；连接BasicGuider。"
                        "不会加载或清除LoRA，不改提示词、参考素材、首尾帧及遮罩。"
                        "已有加速/风格LoRA均保留，叠加效果由用户判断。",
            inputs=[io.Model.Input("model"), io.Latent.Input("av_latent")],
            outputs=[io.Model.Output("model"), io.Sampler.Output("sampler"),
                     io.Sigmas.Output("sigmas"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, model, av_latent):
        return io.NodeOutput(*prepare_pika_flex(model, av_latent))


NODES = [MiniMaxH3PikaFlexSetupEXPT8]
