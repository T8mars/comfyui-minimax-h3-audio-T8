"""Thin Director source geometry and genuine typed HyperFlow HEAD audit nodes.

The runtime observation is mutable, dependency-bound evidence, not an immutable
control execution owner or a portable completion receipt. Ordinary samplers
continue to use the existing LATENT regional audit.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import uuid

from comfy_api.latest import io
import torch

from .control_region_runtime import ControlRegionRuntime
from .control_regions import FRAME_BINS, GEOMETRY_SCHEMA, RegionGate, canonical
from .core import nested_av_parts
from .director_control_source import prepare_director_control_source
from .modular_sampling.hyperflow import ContinuousBoundary
from .modular_sampling.hyperflow_nodes import BOUNDARY
from .nodes_control_region import CATEGORY, STATE


def _row_json(value):
    if not isinstance(value, str):
        raise ValueError("spec_json必须是单条控制对象的JSON字符串")
    try:
        row = json.loads(value)
        canonical(row)  # Reject JSON NaN/Infinity, including inactive extra data.
    except (TypeError, ValueError) as error:
        raise ValueError("spec_json不是有效的有限JSON控制对象") from error
    if not isinstance(row, dict) or "version" in row or "items" in row:
        raise ValueError(
            "spec_json只接收含UUID的单条控制对象，不接收版本封装或控制列表"
        )
    return row


def _runtime_report(state, boundary):
    for name in ("run", "calls", "gated_calls", "target_row_evaluations"):
        value = getattr(state, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"Regional runtime {name} must be a nonnegative integer")
    if state.gated_calls > state.calls or bool(state.gated_calls) != bool(
        state.target_row_evaluations
    ):
        raise ValueError(
            "Regional runtime residual/gated/target-row statistics disagree"
        )
    if not isinstance(state.reasons, Mapping) or any(
        not isinstance(key, str) or type(value) is not int or value < 0
        for key, value in state.reasons.items()
    ):
        raise ValueError(
            "Regional runtime preserved reasons must contain nonnegative counts"
        )
    if not isinstance(state.errors, list) or any(
        not isinstance(error, str) for error in state.errors
    ):
        raise ValueError(
            "Regional runtime errors must be a list of observed error strings"
        )
    if not isinstance(state.stage, str) or not state.stage.strip():
        raise ValueError("Regional runtime must retain its explicit stage label")
    if state.geometry is None:
        if state.gate is not None or state.gated_calls:
            raise ValueError(
                "Regional runtime has gated residuals without actual geometry"
            )
        return state.report()
    geometry = state.geometry
    if not isinstance(geometry, dict) or geometry.get("schema") != GEOMETRY_SCHEMA:
        raise ValueError(
            "Regional runtime requires its actual versioned control geometry"
        )
    payload = {
        key: value for key, value in geometry.items() if key != "geometry_sha256"
    }
    if hashlib.sha256(canonical(payload).encode()).hexdigest() != geometry.get(
        "geometry_sha256"
    ):
        raise ValueError("Regional runtime geometry contents changed")
    video, _ = nested_av_parts(boundary.scaffold)
    target = {
        "width": int(video.shape[-1]) * 16,
        "height": int(video.shape[-2]) * 16,
        "length": sum(
            FRAME_BINS[index % len(FRAME_BINS)] for index in range(video.shape[-3])
        ),
    }
    if geometry.get("target") != target:
        raise ValueError(
            "Regional runtime geometry differs from the actual HEAD canvas/frame bins"
        )
    if state.gate is None:
        if geometry.get("region") is not None or state.gated_calls:
            raise ValueError("Regional runtime lost its requested region gate")
    else:
        gate = state.gate
        if not isinstance(gate, RegionGate):
            raise ValueError(
                "Regional runtime gate must be the actual prepared RegionGate"
            )
        mask = gate.mask
        if (
            not isinstance(mask, torch.Tensor)
            or mask.is_nested
            or mask.layout != torch.strided
            or mask.device.type == "meta"
            or mask.ndim != 3
            or min(mask.shape) <= 0
            or not mask.is_floating_point()
            or not bool(torch.isfinite(mask).all())
            or not bool(((mask >= 0) & (mask <= 1)).all())
        ):
            raise ValueError(
                "Regional runtime mask must remain finite actual coverage in [0,1]"
            )
        if gate.report() != geometry.get("region") or (
            gate.width,
            gate.height,
            gate.length,
        ) != (target["width"], target["height"], target["length"]):
            raise ValueError("Regional runtime region mask or target geometry changed")
    rows_per_call = (
        int(video.shape[-3]) * (int(video.shape[-2]) // 2) * (int(video.shape[-1]) // 2)
    )
    if state.target_row_evaluations != state.gated_calls * rows_per_call:
        raise ValueError(
            "Regional runtime target-row coverage differs from the actual HEAD packed video"
        )
    return state.report()


class MiniMaxH3DirectorControlSourceEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="H3 导演台 · 控制源时轴与几何 (EXP/T8)",
            category=CATEGORY,
            is_experimental=True,
            description="仅按实际输入FPS选择控制视频帧，保留原RGB尺寸；分别准备目标区域MASK与原源像素crop。没有PTS，不认证VFR时轴，不估计姿态／深度，不处理音频或采样。",
            inputs=[
                io.Image.Input("control_video"),
                io.String.Input("spec_json", default="", multiline=True),
                io.Int.Input("width", default=512, min=32, step=32),
                io.Int.Input("height", default=288, min=32, step=32),
                io.Int.Input("length", default=124, min=5, step=17),
                io.Combo.Input(
                    "source_kind", options=["image", "video"], default="image"
                ),
                io.Float.Input("source_fps", default=24.0),
                io.String.Input("stage", default="SINGLE"),
            ],
            outputs=[
                io.Image.Output("control_video"),
                io.Mask.Output("region_mask"),
                io.String.Output("crop_plan_json"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(
        cls,
        control_video,
        spec_json,
        width,
        height,
        length,
        source_kind,
        source_fps,
        stage,
    ):
        frames, mask, crop, report = prepare_director_control_source(
            control_video,
            _row_json(spec_json),
            width=width,
            height=height,
            length=length,
            source_kind=source_kind,
            source_fps=source_fps,
            stage=stage,
        )
        return io.NodeOutput(frames, mask, crop, canonical(report))


class MiniMaxH3DirectorControlHeadAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="H3 导演台 · HEAD区域控制审计 (EXP/T8)",
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            description="真实HEAD typed Boundary接此节点后再保存或接TAIL。校验原完成收据及区域实际调用；原对象透传，不重采样／抽噪，不授予不可变控制owner或可移植身份。",
            inputs=[
                STATE.Input("region_state"),
                io.Custom(BOUNDARY).Input("continuous_boundary"),
            ],
            outputs=[
                io.Custom(BOUNDARY).Output("continuous_boundary"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, region_state, continuous_boundary):
        if not isinstance(region_state, ControlRegionRuntime):
            raise TypeError("Connect the actual H3 regional control runtime")
        if not isinstance(continuous_boundary, ContinuousBoundary):
            raise TypeError(
                "Connect the actual typed HyperFlow continuous HEAD Boundary, not LATENT"
            )
        receipt = continuous_boundary.verify()
        report = {
            "schema": "t8.director.control_head_audit.v1",
            "boundary_receipt_sha256": receipt["receipt_sha256"],
            "head_absolute_interval": receipt["request"]["plan"]["absolute_interval"],
            "head_actual_apply_intervals": receipt["execution"][
                "actual_apply_intervals"
            ],
            "head_portable_identity": receipt["execution"]["portable_identity"],
            "region": _runtime_report(region_state, continuous_boundary),
            "runtime_binding": "dependency_only_not_immutable_execution_owner",
            "sampling_calls": 0,
            "portable_cache_identity": False,
            "quality_accepted": False,
        }
        text = canonical(report)
        return io.NodeOutput(continuous_boundary, text, ui={"text": [text]})

    @classmethod
    def fingerprint_inputs(cls, **_kwargs):
        return uuid.uuid4().hex


NODES = [MiniMaxH3DirectorControlSourceEXPT8, MiniMaxH3DirectorControlHeadAuditEXPT8]
