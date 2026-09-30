"""Append-only Semantic Bridge configuration and CONDITIONING application."""
import math
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .semantic_bridge import BridgeConfig, apply_bridge, canonical, file_sha, compose_bridges
from .semantic_bridge_profiles import inspect_bridge_profile, validate_profile_settings

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
            description="手动参数，旧默认0.10不变。新手请用‘自动模型参数’节点按文件内容读取推荐值。动漫战斗固定1.0/per_token/all_tokens/chunk=0；武术v1不是1.0。固定契约不匹配会在预检报错。多桥用显式‘组合’，不要串多个Apply。",
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
    def validate_inputs(cls, model_name, enabled=True, alpha=0.10, magnitude_match="per_token",
                        token_scope="all_tokens", device="auto", chunk_tokens=256):
        # Naming these fields opts them out of Core's combo/range checks.
        # Preserve all range/enum checks when opting fields into custom validation.
        # Linked inputs are None during validation and are checked
        # again by execute once their producers have actually run.
        if alpha is not None:
            try:
                if not math.isfinite(alpha) or not 0 <= alpha <= 1:
                    return "Bridge alpha must be finite and between 0 and 1"
            except TypeError:
                return "Bridge alpha must be numeric"
        for value, choices, name in ((magnitude_match, ("per_token", "global", "none"), "magnitude_match"),
                                     (token_scope, ("all_tokens", "text_only_preserve_reference"), "token_scope"),
                                     (device, ("auto", "cpu", "cuda"), "device")):
            if value is not None and value not in choices:
                return f"Unknown Bridge {name}"
        if chunk_tokens is not None and (type(chunk_tokens) is not int or not 0 <= chunk_tokens <= 65536):
            return "Bridge chunk_tokens must be an integer between 0 and 65536"
        if enabled is False or alpha == 0:
            return True
        if enabled is None or alpha is None or model_name is None:
            return True
        try:
            path = resolve_model(model_name)
            if all(value is not None for value in (magnitude_match, token_scope, chunk_tokens)):
                validate_profile_settings(inspect_bridge_profile(path), alpha=alpha,
                    magnitude_match=magnitude_match, token_scope=token_scope, chunk_tokens=chunk_tokens)
        except (ValueError, TypeError, FileNotFoundError, OSError) as error:
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
        profile = inspect_bridge_profile(path) if active else None
        config = BridgeConfig(path, profile["sha256"] if active else "", alpha,
                              magnitude_match, token_scope, device, chunk_tokens, enabled)
        warnings = (validate_profile_settings(profile, alpha=alpha, magnitude_match=magnitude_match,
                    token_scope=token_scope, chunk_tokens=chunk_tokens) if active else [])
        return io.NodeOutput(config, canonical({"identity": config.identity(), "model_name": model_name,
                                               "model_profile": profile, "warnings": warnings,
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


class MiniMaxH3SemanticBridgeAutoConfigT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="H3 Semantic Bridge / 自动模型参数 (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="按权重内容SHA／固定元数据契约自动配置。原版/BUNNY .10；已识别武术v1 .12；动漫战斗1.0。"
                        "不会按文件名猜测或把所有trans都设1。未知训练模型请用手动节点。report_json显示实际参数。",
            inputs=[io.Combo.Input("model_name", options=list(model_paths()) or ["No Semantic Bridge models installed"]),
                    io.Boolean.Input("enabled", default=True),
                    io.Combo.Input("device", options=["auto", "cpu", "cuda"], default="auto", advanced=True)],
            outputs=[BridgeIO.Output("semantic_bridge"), io.String.Output("report_json")])

    @classmethod
    def validate_inputs(cls, model_name, enabled=True):
        if enabled is False or enabled is None or model_name is None:
            return True
        try:
            profile = inspect_bridge_profile(resolve_model(model_name))
            if profile["settings"] is None:
                return "No reliable automatic preset for this weight; use the manual configuration and its model card"
        except (ValueError, TypeError, OSError) as error:
            return str(error)
        return True

    @classmethod
    def fingerprint_inputs(cls, model_name, enabled=True, **kwargs):
        return file_sha(resolve_model(model_name)) if enabled else "disabled"

    @classmethod
    def execute(cls, model_name, enabled=True, device="auto"):
        if not enabled:
            return io.NodeOutput(BridgeConfig("", "", enabled=False, device=device),
                                 canonical({"enabled": False, "loaded": False}))
        path = resolve_model(model_name)
        profile = inspect_bridge_profile(path)
        settings = profile["settings"]
        if settings is None:
            raise ValueError("No reliable automatic preset for this weight; use manual configuration and the model card")
        config = BridgeConfig(path, profile["sha256"], device=device, **settings)
        validate_profile_settings(profile, **settings)
        return io.NodeOutput(config, canonical({"model_name": model_name, "model_profile": profile,
            "effective_parameters": settings, "identity": config.identity(), "loaded": False,
            "warning": "EXP; presets are not a quality guarantee"}))


class MiniMaxH3SemanticBridgeComposeT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="H3 Semantic Bridge / 多桥组合 (T8 EXP)",
            category=CATEGORY, is_experimental=True,
            description="显式按 first→second 串行应用，每桥保留自己的强度／契约；不是LoRA加法融合。"
                        "继续串接本组合节点可加载最多8桥，最后只接一次Apply或Relay/内循环Bridge插口。顺序影响效果，"
                        "重复同SHA拒绝，变化换chain_id；多桥可能损害画面／声音。",
            inputs=[BridgeIO.Input("first"), BridgeIO.Input("second"), io.Boolean.Input("enabled", default=True)],
            outputs=[BridgeIO.Output("semantic_bridge"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, first, second, enabled=True):
        config = compose_bridges(first, second, enabled)
        return io.NodeOutput(config, canonical({"identity": config.identity(), "operation": "ordered_serial",
            "loaded": False, "warning": "Order matters; this is not LoRA merging or human quality approval"}))


# Registered only after the complete current root extension prefix.
SEMANTIC_BRIDGE_EXTRA_NODE_CLASSES = [MiniMaxH3SemanticBridgeAutoConfigT8, MiniMaxH3SemanticBridgeComposeT8]
