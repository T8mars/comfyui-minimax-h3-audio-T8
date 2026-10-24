"""Independent MC experimental nodes, not inserted into any default graph.

MC algorithms adapted under GNU GPL version 3; Copyright (C) 2026 NikoDemon80.
T8 modifications dated 2026-10-21. See licenses/MC_MOTION_CONTEXT_NOTICE.md.
"""
from comfy_api.latest import io
from collections.abc import Mapping

from .mc_seam_experimental import canonical, join_match, refine_blend, sigma_tail

CATEGORY = "T8/MiniMax H3/Long Video/Experimental"
LEGAL = " 改编自NikoDemon80 Motion Context固定518c083d，GPL3、无担保；完整许可见licenses/MC_MOTION_CONTEXT_GPL3.txt。"


def _request_connected(cls, name, requested):
    if not requested:
        return []
    hidden = cls.hidden
    prompt = getattr(hidden, "prompt", None) if hidden else None
    unique_id = getattr(hidden, "unique_id", None) if hidden else None
    if isinstance(prompt, Mapping) and unique_id is not None:
        inputs = prompt.get(str(unique_id), {}).get("inputs", {})
        if name not in inputs:
            return []  # An optional unwired input cannot become a strong link.
    return [name]


class MiniMaxH3MCRefineBlendEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 MC 保头细化混合 · Refine Blend (EXP/T8)",
            category=CATEGORY, is_experimental=True,
            description=("独立可选后处理，默认关闭。refined必须由显式上游采样器提供，本节点不采样。"
                         "保持22原帧=7个原生latent单元，随后按物理帧渐变。默认原音频不动；"
                         "clip_index=0为首段全取refined，不自动读取任何Chain界面。"
                         "保留LATENT元数据，不保证画质或额外采样收益。" + LEGAL),
            inputs=[io.Latent.Input("original"), io.Boolean.Input("enabled", default=False),
                    io.Int.Input("hold_frames", default=22, min=0),
                    io.Int.Input("blend_frames", default=22, min=0),
                    io.Combo.Input("audio", options=["original", "refined"], default="original"),
                    io.Int.Input("clip_index", default=1),
                    io.Latent.Input("refined", optional=True, lazy=True)],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
            outputs=[io.Latent.Output("samples"), io.String.Output("report_json")])

    @classmethod
    def check_lazy_status(cls, original, enabled=False, hold_frames=22,
                          blend_frames=22, audio="original", clip_index=1, refined=None):
        return _request_connected(cls, "refined", enabled and refined is None)

    @classmethod
    def execute(cls, original, enabled=False, hold_frames=22, blend_frames=22,
                audio="original", clip_index=1, refined=None):
        output, report = refine_blend(original, refined, enabled=enabled, hold_frames=hold_frames,
                                      blend_frames=blend_frames, audio=audio, clip_index=clip_index)
        return io.NodeOutput(output, canonical(report))


class MiniMaxH3MCJoinMatchEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 MC 接点匹配 · Join Match (EXP/T8)",
            category=CATEGORY, is_experimental=True,
            description=("独立可选latent统计处理，默认关闭。原算法source_whole_clip会改头部pin；"
                         "keep_pin_T8是明确的新变体，恢复前22帧对应7单元，不冒称原算法。"
                         "context_length须与实际已采样pin一致，不能从均值推断真实闪光/硬切应消除；"
                         "所有结果须从接点到片尾审片。原音频和元数据保留，不增加NFE。" + LEGAL),
            inputs=[io.Latent.Input("samples"), io.Boolean.Input("enabled", default=False),
                    io.Float.Input("strength", default=1.0, step=.05),
                    io.Int.Input("context_length", default=22, min=0),
                    io.Combo.Input("variant", options=["keep_pin_T8_exp", "source_whole_clip_exp"], default="keep_pin_T8_exp"),
                    io.Combo.Input("tuning", options=["auto", "manual"], default="auto"),
                    io.Float.Input("min_excess", default=1.5, advanced=True),
                    io.Boolean.Input("match_spread", default=True),
                    io.Float.Input("min_excess_spread", default=1.5, advanced=True),
                    io.Boolean.Input("match_detail", default=True),
                    io.Float.Input("min_excess_detail", default=1.5, advanced=True),
                    io.Latent.Input("context_latent", optional=True, lazy=True)],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
            outputs=[io.Latent.Output("samples"), io.String.Output("report_json")])

    @classmethod
    def check_lazy_status(cls, samples, enabled=False, strength=1.0, context_length=22,
                          context_latent=None, **kwargs):
        return _request_connected(cls, "context_latent", enabled and strength != 0 and context_length != 0 and context_latent is None)

    @classmethod
    def execute(cls, samples, enabled=False, strength=1.0, context_length=22,
                variant="keep_pin_T8_exp", tuning="auto", min_excess=1.5,
                match_spread=True, min_excess_spread=1.5, match_detail=True,
                min_excess_detail=1.5, context_latent=None):
        output, report = join_match(samples, context_latent, enabled=enabled, strength=strength,
                                    context_length=context_length, variant=variant, tuning=tuning,
                                    min_excess=min_excess, match_spread=match_spread,
                                    min_excess_spread=min_excess_spread, match_detail=match_detail,
                                    min_excess_detail=min_excess_detail)
        return io.NodeOutput(output, canonical(report))


class MiniMaxH3MCSigmaTailEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__, display_name="H3 MC 独立Sigma尾部 · Sigma Tail (EXP/T8)",
            category=CATEGORY, is_experimental=True,
            description=("默认关闭、原SIGMAS对象直通。开启只生成独立的新尾部表，不自动接入HF/PTD/4+4，"
                         "不修改任何原训练表或步数。此表并未获蒸馏模型兼容/画质认证，需用户显式接线；"
                         "strength=-1是公式的真实零分母错误；其它可计算选择执行并报告，不以widget意图硬禁。"
                         "后续联合采样可能改变音画，并非原声保持保证。" + LEGAL),
            inputs=[io.Sigmas.Input("sigmas"), io.Boolean.Input("enabled", default=False),
                    io.Int.Input("tail_steps", default=3, min=0),
                    io.Float.Input("strength", default=.5, step=.05)],
            outputs=[io.Sigmas.Output("sigmas"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, sigmas, enabled=False, tail_steps=3, strength=.5):
        output, report = sigma_tail(sigmas, enabled=enabled, tail_steps=tail_steps, strength=strength)
        return io.NodeOutput(output, canonical(report))


NODES = [MiniMaxH3MCRefineBlendEXPT8, MiniMaxH3MCJoinMatchEXPT8, MiniMaxH3MCSigmaTailEXPT8]
