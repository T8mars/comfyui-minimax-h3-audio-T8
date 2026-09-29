"""Append-only Semantic Bridge configuration and CONDITIONING application."""
import math
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .semantic_bridge import BridgeConfig, apply_bridge, canonical, file_sha

BridgeIO = io.Custom("T8_SEMANTIC_BRIDGE")
CATEGORY = "T8/MiniMax H3/Semantic Bridge"


def model_paths():
    """Keep extra_model_paths entries and disambiguate duplicate relative names."""
    default = Path(folder_paths.models_dir) / "semantic_bridge"
    roots = [Path(path) for path in folder_paths.folder_names_and_paths.get("semantic_bridge", ([], set()))[0]]
    if default not in roots:
        roots.append(default)
    grouped = {}
    for root in roots:
        if root.is_dir():
            for path in sorted(root.rglob("*.safetensors")):
                if ".cache" not in path.relative_to(root).parts:
                    grouped.setdefault(path.relative_to(root).as_posix(), set()).add(str(path.resolve()))
    result = {}
    for name, paths in sorted(grouped.items()):
        for path in sorted(paths):
            label = name if len(paths) == 1 else f"{name} [{path}]"
            result[label] = path
    return result


def resolve_model(name):
    path = model_paths().get(name)
    if path is None:
        raise FileNotFoundError("Semantic Bridge model not found. Install an author .safetensors in models/semantic_bridge and refresh the list.")
    return path


class MiniMaxH3SemanticBridgeConfigT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SemanticBridgeConfigT8",
            display_name="H3 Semantic Bridge / 模型与设置 (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="Semantic Bridge 文本条件适配。原版/BUNNY 默认0.10；T8 动漫战斗 Transformer 权重按模型卡设 1.0、per_token、all_tokens、chunk_tokens=0。模型放入 models/semantic_bridge；参考音频/唱歌可能退化。",
            inputs=[
                io.Combo.Input("model_name", options=list(model_paths()) or ["No Semantic Bridge models installed"],
                    tooltip="原版/BUNNY：https://huggingface.co/t8star/Semantic-Bridge-Comfy ；T8动漫战斗：https://huggingface.co/t8star/semantic_bridge_T8-comic-combat 。放在 models/semantic_bridge/t8_compat；不是LoRA或H3主模型。"),
                io.Boolean.Input("enabled", default=True),
                io.Float.Input("alpha", default=0.10, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("magnitude_match", options=["per_token", "global", "none"], default="per_token"),
                io.Combo.Input("token_scope", options=["all_tokens", "text_only_preserve_reference"], default="all_tokens",
                               tooltip="all_tokens复现原作者；text_only仅修改原生tag=1行，仍不能保证歌声。"),
                io.Combo.Input("device", options=["auto", "cpu", "cuda"], default="auto", advanced=True),
                io.Int.Input("chunk_tokens", default=256, min=0, max=65536, advanced=True,
                             tooltip="0 = whole sequence; required by some trained Transformer bridges."),
            ], outputs=[BridgeIO.Output("semantic_bridge"), io.String.Output("report_json")],
        )

    @classmethod
    def validate_inputs(cls, model_name, enabled=True, alpha=0.10):
        # Naming these fields opts them out of Core's combo/range checks.
        # Preserve alpha validation even for bypass; leave every other field
        # to Core. Linked inputs are None during validation and are checked
        # again by execute once their producers have actually run.
        if alpha is not None:
            try:
                if not math.isfinite(alpha) or not 0 <= alpha <= 1:
                    return "Bridge alpha must be finite and between 0 and 1"
            except TypeError:
                return "Bridge alpha must be numeric"
        if enabled is False or alpha == 0:
            return True
        if enabled is None or alpha is None or model_name is None:
            return True
        try:
            resolve_model(model_name)
        except (FileNotFoundError, OSError) as error:
            return str(error)
        return True

    @classmethod
    def fingerprint_inputs(cls, model_name, enabled=True, alpha=0.10, **kwargs):
        if not enabled or alpha == 0:
            return "disabled"
        return file_sha(resolve_model(model_name))

    @classmethod
    def execute(cls, model_name, enabled=True, alpha=0.10, magnitude_match="per_token",
                token_scope="all_tokens", device="auto", chunk_tokens=256):
        active = enabled and alpha != 0
        path = resolve_model(model_name) if active else ""
        config = BridgeConfig(path, file_sha(path) if active else "", alpha,
                              magnitude_match, token_scope, device, chunk_tokens, enabled)
        return io.NodeOutput(config, canonical({"identity": config.identity(), "model_name": model_name,
                                               "loaded": False, "warning": "EXP; no universal quality guarantee"}))


class MiniMaxH3SemanticBridgeApplyT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SemanticBridgeApplyT8",
            display_name="H3 Semantic Bridge / 应用条件 (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="接在原生条件编码之后、采样之前。Relay请使用内部Bridge入口；不要重复增强。",
            inputs=[io.Conditioning.Input("conditioning"), BridgeIO.Input("semantic_bridge")],
            outputs=[io.Conditioning.Output("conditioning"), io.String.Output("report_json")],
        )

    @classmethod
    def execute(cls, conditioning, semantic_bridge):
        result, report = apply_bridge(conditioning, semantic_bridge)
        return io.NodeOutput(result, canonical(report))


SEMANTIC_BRIDGE_NODE_CLASSES = [MiniMaxH3SemanticBridgeConfigT8, MiniMaxH3SemanticBridgeApplyT8]
