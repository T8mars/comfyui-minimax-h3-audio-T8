"""Unregistered explicit PTD experimental entry points; old schemas unchanged."""

import json
from pathlib import Path
import folder_paths
from comfy_api.latest import io
from .ptd_advanced import get_controller, load_ptd_setup
from .ptd_sampling import make_sampler, ptd_sigmas

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
            description="独立 PTD 4／6 步逐 hop 专家，不是普通 LoRA、PDD 或 4+4。只读可信已转换 safetensors，不执行 DCP/pickle。原用户底模和 LoRA 保留，未知组合风险见报告；训练模型/GPU/画质未验。",
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


class MiniMaxH3PTDSamplerEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="H3 PTD 采样器与完整网格 (EXP/T8)",
            category=CATEGORY,
            is_experimental=True,
            description="从 PTD MODEL 取得匹配的完整 4／6 步网格。不可切半、接普通 Euler 或冒充旧 4+4。",
            inputs=[io.Model.Input("model")],
            outputs=[io.Sampler.Output("sampler"), io.Sigmas.Output("sigmas")],
        )

    @classmethod
    def execute(cls, model):
        controller = get_controller(model)
        return io.NodeOutput(make_sampler(controller), ptd_sigmas(controller.recipe))


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
