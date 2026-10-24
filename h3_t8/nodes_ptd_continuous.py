"""Unregistered opt-in, same-canvas PTD 2+2 / 3+3 nodes."""
import json

from comfy_api.latest import io

from .ptd_continuous import inspect_stage, run_head, run_tail

CATEGORY = "T8/MiniMax H3/Performance/Experimental"
HEAD = io.Custom("T8_PTD_HEAD")
COMPLETE = io.Custom("T8_PTD_COMPLETION")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class MiniMaxH3PTDContinuousHEAD_EXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="H3 PTD 连续 HEAD · 2或3步 (EXP/T8)", category=CATEGORY,
            is_experimental=True,
            inputs=[io.Noise.Input("noise"), io.Guider.Input("guider"), io.Latent.Input("av_latent")],
            outputs=[HEAD.Output("head"), io.String.Output("report_json")],
            description="Guider 使用已选择的PTD MODEL。4步前2／6步前3，保存真实内部FP32 noisy AV；不是x0、不是标准4+4，不写冷缓存。原完整PTD入口不变。")

    @classmethod
    def execute(cls, noise, guider, av_latent):
        head, report = run_head(noise, guider, av_latent)
        return io.NodeOutput(head, _json(report))


class MiniMaxH3PTDContinuousTAIL_EXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="H3 PTD 连续 TAIL · 接续全局专家 (EXP/T8)", category=CATEGORY,
            is_experimental=True,
            inputs=[io.Guider.Input("guider"), HEAD.Input("head")],
            outputs=[io.Latent.Output("av_latent"), COMPLETE.Output("completion"), io.String.Output("report_json")],
            description="直接续接HEAD内部状态与原物理AV时钟，不新随机噪声、不重置专家索引；TAIL Guider保留用户阶段MODEL/有序LoRA。仅同画布2+2/3+3，未实现放大或冷TAIL。")

    @classmethod
    def execute(cls, guider, head):
        latent, completion, report = run_tail(guider, head)
        return io.NodeOutput(latent, completion, _json(report))


class MiniMaxH3PTDContinuousAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,
            display_name="H3 PTD 连续阶段真实审计 (EXP/T8)", category=CATEGORY,
            is_experimental=True, is_output_node=True,
            inputs=[io.AnyType.Input("stage")], outputs=[io.String.Output("report_json")],
            description="只接受实际已执行的typed HEAD/完成对象并核内容；其它latent或伪造JSON不能作为完成证明。无GPU/画质/portable资格。")

    @classmethod
    def execute(cls, stage):
        value = _json(inspect_stage(stage))
        return io.NodeOutput(value, ui={"text": [value]})

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")


NODES = [MiniMaxH3PTDContinuousHEAD_EXPT8, MiniMaxH3PTDContinuousTAIL_EXPT8,
         MiniMaxH3PTDContinuousAuditEXPT8]
