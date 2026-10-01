"""Capability inventory used by the Director D3 hand-off panel.

This is deliberately an inventory and dependency gate, not a generic graph
builder.  D3 features have different state contracts (long-video resume,
Bridge/Relay projection, post-processing, or Meridian geometry), so the
Director must show the real registered entry points and send users to the
corresponding native workflow until each recipe has its own end-to-end receipt.
"""

from __future__ import annotations

from typing import Iterable, Mapping


CAPABILITY_SPECS: tuple[Mapping[str, object], ...] = (
    {
        "id": "long_video",
        "label": "连续长片／二采",
        "entry_nodes": (
            "MiniMaxH3LongVideoPlannerT8",
            "MiniMaxH3LongVideoConditioningT8",
            "MiniMaxH3LongVideoOrchestratorT8",
            "MiniMaxH3LongVideoComposeAcceptedT8",
        ),
        "route": "native_long_video_workflow",
        "note": "保留上下文、取消、恢复与已验收接缝；不把单段生成拼成连续长片。",
    },
    {
        "id": "semantic_bridge",
        "label": "Semantic Bridge",
        "entry_nodes": (
            "MiniMaxH3SemanticBridgeConfigT8",
            "MiniMaxH3SemanticBridgeApplyT8",
        ),
        "route": "native_semantic_bridge_workflow",
        "note": "Bridge 只改变条件链，不改既有采样、音频、时间偏置或接缝默认。",
    },
    {
        "id": "prompt_relay",
        "label": "Prompt Relay",
        "entry_nodes": (
            "MiniMaxH3PromptRelayPlanT8Advanced",
            "MiniMaxH3PromptRelayConditioningT8Advanced",
        ),
        "route": "native_prompt_relay_workflow",
        "note": "按正式 Relay Plan→Conditioning 合同编译，原文空白与角色映射可审计。",
    },
    {
        "id": "fast_h3_v2",
        "label": "FastH3 V2／Tao",
        "entry_nodes": (
            "MiniMaxH3FastH3V2SetupEXPT8",
            "MiniMaxH3FastH3V2RuntimeAuditEXPT8",
            "MiniMaxH3FastH3V2DualModelLongVideoEXPT8",
        ),
        "route": "native_fast_h3_v2_workflow",
        "note": "现有导演台编译为单镜 8 步；双 MODEL 4+4 分离图仍走原生实验路线。训练 VSA 不支持活动 Prompt Relay 时间偏置，二者同开须显式选择 Dense。",
    },
    {
        "id": "eav",
        "label": "EAV／增强音视频",
        "entry_nodes": (
            "MiniMaxH3EnhanceAVideoPromptRelayComposerT8Advanced",
            "MiniMaxH3EnhanceAVideoLongVideoComposerT8Advanced",
        ),
        "route": "native_eav_workflow",
        "note": "EAV 与 Relay／长片有独立 wrapper 合同，未验收组合不会被导演台强行叠加。",
    },
    {
        "id": "low_vram",
        "label": "LowVRAM／ChunkFFN",
        "entry_nodes": (
            "MiniMaxH3LowVRAMAttentionT8Advanced",
            "MiniMaxH3ChunkFeedForwardT8Advanced",
        ),
        "route": "native_memory_workflow",
        "note": "显存策略是显式可审计节点，不偷偷改变普通 H3 默认路径。",
    },
    {
        "id": "taeh3_preview",
        "label": "TAEH3 实时采样预览",
        "entry_nodes": (
            "MiniMaxH3TAEH3PreviewCapabilityT8Advanced",
            "MiniMaxH3TAEH3SamplingPreviewEXPT8",
        ),
        "route": "native_taeh3_preview_workflow",
        "note": "预览是中间 x0 近似图，不是最终 VAE 成片；取消必须回到原生采样入口。",
    },
    {
        "id": "topaz",
        "label": "Topaz 高清放大",
        "entry_nodes": (
            "MiniMaxH3TopazEnvironmentEXPT8",
            "MiniMaxH3TopazVideoEXPT8",
        ),
        "route": "native_topaz_workflow",
        "note": "输入可为普通下载视频或 H3 输出；环境检测、取消与安全 H264 交付需分别通过。",
    },
    {
        "id": "meridian",
        "label": "Meridian",
        "entry_nodes": (
            "MiniMaxH3MeridianConfigEXPT8",
            "MiniMaxH3MeridianMaterialEXPT8",
            "MiniMaxH3MeridianCameraEXPT8",
            "MiniMaxH3MeridianGenerateEXPT8",
        ),
        "route": "native_meridian_workflow",
        "note": "几何权重、ConvRot INT8 与完整 P2/P3/P4 质量门禁独立记录，不从 D2 生成结果冒充。",
    },
)


def inspect_director_capabilities(
    node_ids: Iterable[str] | None = None,
    *,
    model_names: Iterable[str] | None = None,
) -> dict[str, object]:
    """Return a truthful, JSON-safe D3 inventory for the current Core.

    ``model_names`` is intentionally only reported as context.  Each feature
    owns different model/weight checks, so a filename-only global gate would be
    misleading; its native workflow performs the final dependency validation.
    """

    if node_ids is None:
        try:
            import nodes

            node_ids = nodes.NODE_CLASS_MAPPINGS.keys()
        except Exception:
            node_ids = ()
    available = {str(value) for value in node_ids}
    models = sorted({str(value) for value in (model_names or ())})
    rows = []
    for spec in CAPABILITY_SPECS:
        required = [str(value) for value in spec["entry_nodes"]]
        missing = [value for value in required if value not in available]
        rows.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "state": "ready" if not missing else "missing_entry_nodes",
                "entry_nodes": required,
                "missing_entry_nodes": missing,
                "route": spec["route"],
                "note": spec["note"],
                "execution": "native_workflow_only_until_gpu_receipt",
            }
        )
    return {
        "schema": "t8.minimax_h3.director_capabilities.v1",
        "capabilities": rows,
        "model_inventory_context": models,
        "warning": "ready means registered entry nodes only; it is not a GPU or human quality acceptance.",
    }
