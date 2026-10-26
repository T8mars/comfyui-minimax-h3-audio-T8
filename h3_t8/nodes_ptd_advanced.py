"""Unregistered explicit PTD experimental entry points; old schemas unchanged."""

import json
from pathlib import Path
import folder_paths
from comfy_api.latest import io
from .ptd_advanced import get_controller, load_ptd_setup, sampler_and_sigmas

CATEGORY = "T8/MiniMax H3/Performance/Experimental"


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def checkpoint_path(name):
    path = Path(name)
    if path.is_absolute():
        return path
    root = (Path(folder_paths.models_dir) / "ptd").resolve()
    path = (root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(
            "Relative PTD checkpoint path must remain within models/ptd; use an explicitly selected absolute file otherwise"
        )
    return path


class MiniMaxH3PTDSetupEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="H3 PTD 专家加载与采样 · Hop Setup (EXP/T8)",
            category=CATEGORY,
            is_experimental=True,
            description="读取文件元数据选择 EPFL 4／6 步或 stacked-hop Compact PTD 完整网格；Compact 支持 schema1 行量化存储，加载时还原 BF16，不代表 INT8 推理。只读 safetensors，不执行 DCP/pickle。原用户底模和 LoRA 保留，实际覆盖见报告；指定 Compact 4 步已完成本地实跑，其他组合与完整画质需另验。",
            inputs=[
                io.Model.Input("model"),
                io.String.Input(
                    "checkpoint",
                    default="ptd_h3_4step.safetensors",
                    tooltip="models/ptd 下文件名或明确选择的绝对路径；本节点不会下载或转换权重",
                ),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Sampler.Output("sampler"),
                io.Sigmas.Output("sigmas"),
                io.String.Output("report_json"),
            ],
        )

    @classmethod
    def execute(cls, model, checkpoint):
        selected, sampler, sigmas, report = load_ptd_setup(
            model, checkpoint_path(checkpoint)
        )
        return io.NodeOutput(selected, sampler, sigmas, _json(report))

    @classmethod
    def fingerprint_inputs(cls, checkpoint, **_kwargs):
        # Cache invalidation only: this stat identity is not a content receipt.
        # Full file SHA/format validation remains in load_ptd_setup on reload.
        # Do not read multi-GB checkpoint contents just because seed changed.
        from .hybrid_model import file_stat_fingerprint

        return "ptd-checkpoint-stat-v1:" + file_stat_fingerprint(
            (checkpoint_path(checkpoint),)
        )


class MiniMaxH3PTDSamplerEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="H3 PTD 采样器与完整网格 (EXP/T8)",
            category=CATEGORY,
            is_experimental=True,
            description="从 PTD MODEL 取得对应 EPFL 或 Compact 合同的采样器和完整网格。不可切半、接普通 Euler 或冒充旧 4+4。Compact 暂无连续 HEAD/TAIL 状态交接协议。",
            inputs=[io.Model.Input("model")],
            outputs=[io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas")],
        )

    @classmethod
    def execute(cls, model):
        controller = get_controller(model)
        return io.NodeOutput(*sampler_and_sigmas(controller))


class MiniMaxH3PTDAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="H3 PTD 实际专家与时钟审计 (EXP/T8)",
            category=CATEGORY,
            is_experimental=True,
            is_output_node=True,
            inputs=[io.Model.Input("model"), io.AnyType.Input("after_sampling")],
            outputs=[io.String.Output("report_json")],
            description="排序输入只建立依赖；报告实际 hop／专家／AV 时钟，不把输入 latent 当完成证明，不授予缓存或GPU质量资格。",
        )

    @classmethod
    def execute(cls, model, after_sampling):
        value = _json(get_controller(model).report())
        return io.NodeOutput(value, ui={"text": [value]})

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")


NODES = [MiniMaxH3PTDSetupEXPT8, MiniMaxH3PTDSamplerEXPT8, MiniMaxH3PTDAuditEXPT8]
