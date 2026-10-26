"""Explicit periodic AV wiring; registry inclusion is owned by the project."""
from __future__ import annotations

from dataclasses import dataclass, field
import json

from comfy_api.latest import io
from comfy.patcher_extension import WrappersMP

from .audio_ops import decode_av_latent
from .av_loop_clock import LoopPlan
from .av_loop_runtime import KEY, _parts, audit_loop, configure_loop, decode_loop
from .core import nested_av_parts
from .patch_stack_policy import warn_patch_stack


PLAN_IO = io.Custom("T8_H3_PERIODIC_AV_PLAN")
CATEGORY = "T8/MiniMax H3/Experimental"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class PeriodicAVDecodePlan:
    """In-memory geometry, not a receipt binding a sampled tensor to a MODEL."""

    enabled: bool
    geometry: LoopPlan | None = None
    _binding: tuple = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_binding", self._signature())

    def _signature(self):
        if type(self.enabled) is not bool:
            raise ValueError("periodic decode enabled must be boolean")
        if not self.enabled:
            if self.geometry is not None:
                raise ValueError("disabled decode plan must not claim periodic geometry")
            return (False,)
        if not isinstance(self.geometry, LoopPlan):
            raise TypeError("enabled decode plan needs actual LoopPlan geometry")
        self.geometry.__post_init__()
        return (True, self.geometry.length, self.geometry.pad_clips, self.geometry.views)

    def verify(self):
        if self._signature() != self._binding:
            raise ValueError("periodic decode plan changed after setup")


def infer_loop_plan(av_latent, pad_clips=1, views=2):
    video, audio = nested_av_parts(av_latent)
    ticks = video.shape[2]
    if ticks < 7 or (ticks - 2) % 5:
        raise ValueError("periodic setup needs actual video T=5k+2 with k>=1")
    plan = LoopPlan(17 * ((ticks - 2) // 5) + 5, pad_clips=pad_clips, views=views)
    _parts((video, audio), plan)
    return plan


def paired_control_owners(model):
    """Bind only complete observed native interfaces, regardless of class name.

    A bound diffusion method alone is insufficient: every declared injection
    layer must have the same owner's actual block-patch entry in its chain.
    Opaque methods/closures remain in the user's stack and are not certified.
    """
    wrappers = getattr(model, "wrappers", {}).get(WrappersMP.DIFFUSION_MODEL, {})
    options = getattr(model, "model_options", {}).get("transformer_options", {})
    patches = options.get("patches_replace", {}).get("dit", {})
    owners, seen, warnings = [], set(), []
    methods = ("diffusion_model_wrapper", "prepare_control_latent", "before_block", "after_block",
               "to", "cleanup", "models", "register")
    fields = ("control_video", "mask", "source_video", "control_latent", "control_latent_shape",
              "control_stream", "pristine_stream", "active", "model_patch", "vae")
    for entries in wrappers.values():
        for wrapper in entries:
            owner = getattr(wrapper, "__self__", None)
            if owner is None:
                warnings.append("opaque_diffusion_wrapper_preserved_without_temporal_binding")
                continue
            if id(owner) in seen:
                continue
            try:
                bound = getattr(owner, "diffusion_model_wrapper", None)
                complete = (all(callable(getattr(owner, name, None)) for name in methods)
                            and all(hasattr(owner, name) for name in fields)
                            and callable(getattr(wrapper, "__func__", None))
                            and getattr(bound, "__self__", None) is owner
                            and getattr(bound, "__func__", None) is getattr(wrapper, "__func__", None))
                layers = getattr(getattr(getattr(owner, "model_patch", None), "model", None),
                                 "injection_layers", None)
                if (not complete or not isinstance(layers, (tuple, list)) or not layers
                        or any(type(index) is not int or index < 0 for index in layers)):
                    warnings.append("unknown_bound_wrapper_preserved_without_temporal_binding")
                    continue
                matched = set()
                for index in layers:
                    patch = patches.get(("double_block", index))
                    chain = set()
                    while patch is not None and id(patch) not in chain:
                        chain.add(id(patch))
                        if (getattr(patch, "control_patch", None) is owner
                                and getattr(patch, "block_index", None) == index
                                and hasattr(patch, "previous") and callable(patch)):
                            matched.add(index)
                        patch = getattr(patch, "previous", None)
                if matched != set(layers):
                    warnings.append("incomplete_control_patch_pair_preserved_without_temporal_binding")
                    continue
                owners.append(owner)
                seen.add(id(owner))
                if callable(getattr(owner, "verify_derived", None)):
                    warnings.append("derived_cache_only_region_gate_clock_and_closure_not_adapted")
            except Exception as error:
                warnings.append("control_binding_inspection_delegated:" + type(error).__name__)
    return tuple(owners), list(dict.fromkeys(warnings))


class MiniMaxH3PeriodicAVSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="H3 音画循环 · 设置 (EXP/T8)",
            category=CATEGORY, is_experimental=True,
            description="默认关闭，原MODEL直通。开启后从实际AV latent推导17k+5时钟，"
                        "在模型预测中周期补边与旋转；第二视图只融合音频。"
                        "MODEL接原采样器，原AV latent仍接采样输入，计划接循环解码。"
                        "不修改LoRA、采样器、步数或sigma；非空区域控制尚无循环时间适配。",
            inputs=[io.Model.Input("model"), io.Latent.Input("av_latent"),
                    io.Boolean.Input("enabled", default=False),
                    io.Int.Input("pad_clips", default=1, min=1, advanced=True,
                                 tooltip="每侧补边的17帧组数；3的倍数可消除音频补边tick取整误差，计算量随之增加。"),
                    io.Int.Input("views", default=2, min=1, max=2, advanced=True,
                                 tooltip="1或2；视频只用第一视图。奇数组数的第二视图不是精确半圈。")],
            outputs=[io.Model.Output("model"), PLAN_IO.Output("loop_plan"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, av_latent, enabled=False, pad_clips=1, views=2):
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        if not enabled:
            return io.NodeOutput(model, PeriodicAVDecodePlan(False), canonical({
                "status": "disabled_native_path", "requested_enabled": False,
                "decode_policy": "existing_full_native_decode", "quality_accepted": False}))
        plan = infer_loop_plan(av_latent, pad_clips, views)
        controls, warnings = paired_control_owners(model)
        for warning in warnings:
            warn_patch_stack("periodic AV: " + warning)
        selected, report = configure_loop(model, plan, control_owners=controls)
        installed = report.get("schema") == KEY
        route = PeriodicAVDecodePlan(installed, plan if installed else None)
        report = {**report, "requested_enabled": True, "loop_decode_enabled": installed,
                  "complete_control_pairs_bound": len(controls) if installed else 0,
                  "control_binding_warnings": warnings, "sample_result_binding_proven": False}
        return io.NodeOutput(selected, route, canonical(report))


class MiniMaxH3PeriodicAVDecodeEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="H3 音画循环 · 解码 (EXP/T8)",
            category=CATEGORY, is_experimental=True,
            description="采样结果与设置节点的计划配对。关闭时调用既有完整原生解码；"
                        "开启时两VAE周期补边，视频17k+5裁为17k，音频按24fps帧时钟裁剪。"
                        "无交叉淡化、重采样或自动质量认证；计划只表示几何，不是结果来源回执。",
            inputs=[io.Latent.Input("av_latent"), PLAN_IO.Input("loop_plan"),
                    io.Vae.Input("video_vae"), io.Vae.Input("audio_vae")],
            outputs=[io.Image.Output("images"), io.Audio.Output("audio"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, av_latent, loop_plan, video_vae, audio_vae):
        if type(loop_plan) is not PeriodicAVDecodePlan:
            raise TypeError("connect the typed periodic AV plan from the setup node")
        loop_plan.verify()
        if not loop_plan.enabled:
            images, audio = decode_av_latent(av_latent, video_vae, audio_vae)[:2]
            report = {"status": "native_decode_delegated", "periodic_crop_applied": False,
                      "video_frames": images.shape[0], "audio_samples": audio["waveform"].shape[-1],
                      "quality_accepted": False}
        else:
            images, audio, report = decode_loop(av_latent, video_vae, audio_vae, loop_plan.geometry)
            report = {**report, "periodic_crop_applied": True, "sample_result_binding_proven": False}
        return io.NodeOutput(images, audio, canonical(report))


class MiniMaxH3PeriodicAVAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="H3 音画循环 · 运行审计 (EXP/T8)",
            category=CATEGORY, is_experimental=True, is_output_node=True,
            description="after_sampling仅建立执行顺序。报告该MODEL实际预测/视图次数、补噪和时钟；"
                        "不据配置推断采样步数、成片来源、无缝效果或GPU通用资格。",
            inputs=[io.Model.Input("model"), io.AnyType.Input("after_sampling")],
            outputs=[io.String.Output("report_json")])

    @classmethod
    def execute(cls, model, after_sampling):
        report = canonical({**audit_loop(model), "sample_result_binding_proven": False})
        return io.NodeOutput(report, ui={"text": [report]})

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return float("nan")


NODES = [MiniMaxH3PeriodicAVSetupEXPT8, MiniMaxH3PeriodicAVDecodeEXPT8, MiniMaxH3PeriodicAVAuditEXPT8]
