"""Truthful D3 route registry and per-shot dependency preflight.

The Director D3 surface is intentionally a router, not a second implementation
of nine unrelated runtimes.  Each capability keeps its native workflow and
entry nodes; this module makes the current project/shot, installed nodes and
model inventory visible before the user leaves the Director for that route.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from .director_capabilities import CAPABILITY_SPECS
from .director_project import ProjectStore, canonical, compile_project, validate_project


ROUTE_WORKFLOWS: Mapping[str, tuple[str, ...]] = {
    "long_video": (
        "examples/workflows/04-long-video/2026-09-12_H3_Dual_Model_Long_Video_4plus4_Joint_AV_Dialogue_Aligned_24s_EXP.json",
        "examples/workflows/04-long-video/2026-09-15_H3_Dual_4plus4_Accepted_Picture_T8_LowVRAM_EXP.json",
    ),
    "semantic_bridge": (
        "examples/workflows/34-semantic-bridge/2026-09-17_H3_SemanticBridge_DualIndependent_8s_Advanced.json",
        "examples/workflows/34-semantic-bridge/README.md",
    ),
    "prompt_relay": (
        "examples/workflows/14-prompt-relay/2026-08-20_H3_Prompt_Relay_I2VA_Stock20_Advanced_EXP.json",
        "examples/workflows/14-prompt-relay/2026-08-20_H3_Prompt_Relay_Ref2VA_Stock20_Advanced_EXP.json",
    ),
    "fast_h3_v2": (
        "examples/workflows/34-fasth3-v2/FastH3_V2_Trained_VSA_73f_h1c1_EXP.json",
        "examples/workflows/34-fasth3-v2/FastH3_V2_Split_01_First_LOW_HIGH_Relay_EAV_EXP.json",
        "examples/workflows/34-fasth3-v2/FastH3_V2_Split_02_First_Cold_HIGH_EXP.json",
        "examples/workflows/34-fasth3-v2/FastH3_V2_Split_03_Accepted_Second_LOW_HIGH_EXP.json",
        "examples/workflows/34-fasth3-v2/FastH3_V2_Split_04_Accepted_Second_Cold_HIGH_EXP.json",
        "examples/workflows/34-fasth3-v2/FastH3_V2_Split_05_Review_Explicit_Accept_EXP.json",
        "examples/workflows/34-fasth3-v2/FastH3_V2_Split_06_Compose_Accepted_EXP.json",
        "examples/workflows/34-fasth3-v2/README.md",
        "examples/workflows/34-fasth3-v2/FAST_H3_V2_README.md",
    ),
    "eav": (
        "examples/workflows/04-long-video/2026-09-11_H3_In_Node_Long_Video_Prompt_Relay_EAV_Dialogue_Segment_Aligned_Stock20_Advanced_EXP.json",
        "examples/workflows/36-avatar-voice/2026-09-18_T8_voice_eav_on_EXP.json",
    ),
    "low_vram": (
        "examples/workflows/04-long-video/2026-09-15_H3_Dual_4plus4_Accepted_Picture_T8_LowVRAM_EXP.json",
        "docs/H3_MEMORY_NODES_EXP.md",
    ),
    "taeh3_preview": (
        "docs/TAEH3_SAMPLING_PREVIEW_EXP.md",
        "examples/workflows/04-long-video/2026-09-12_H3_Dual_Model_Long_Video_4plus4_Joint_AV_Dialogue_Aligned_24s_EXP.json",
    ),
    "topaz": (
        "examples/workflows/31-topaz/2026-09-11_H3_Topaz_Video_EXP.json",
        "examples/workflows/31-topaz/2026-09-11_H3_Topaz_Environment_EXP.json",
    ),
    "meridian": (
        "examples/workflows/37-meridian/2026-09-18_T8_Meridian_video_source_camera_EXP.json",
        "examples/workflows/37-meridian/README.md",
    ),
}

# These contracts are intentionally short and operational.  They are shown in
# the hand-off dialog and copied into the route package so a user can continue
# in the native workflow without guessing which part the Director did (and did
# not) compile.
ROUTE_CONTRACTS: Mapping[str, Mapping[str, object]] = {
    "long_video": {
        "mode": "native_only",
        "steps": ["打开连续长片工作流", "替换源视频／分段输入", "保留规划、接缝、取消与恢复节点"],
        "requires": ["连续源视频", "分段或上下文配置", "可用 H3 模型"],
        "boundary": "导演台不会把单镜头拼成连续长片；原生工作流负责上下文和二采状态。",
    },
    "semantic_bridge": {
        "mode": "native_only",
        "steps": ["选择 models/semantic_bridge 下的转换权重", "接入 Bridge 条件链", "沿原采样和音频链运行"],
        "requires": ["MiniMaxH3_SemanticBridge_v1_T8_Compat.safetensors 或同等兼容权重"],
        "boundary": "Bridge 只改条件链，不替换采样器、音频或接缝策略。",
    },
    "prompt_relay": {
        "mode": "native_only",
        "steps": ["填写 Relay Plan 时间段", "把 Plan 接到 Conditioning", "沿原生 H3 采样链运行"],
        "requires": ["Prompt Relay Plan／Conditioning 节点", "与输入类型匹配的 H3 模型"],
        "boundary": "空白片段和角色映射必须在原生 Plan 中确认，导演台不替用户猜测。",
    },
    "fast_h3_v2": {
        "mode": "native_only",
        "steps": ["旧单镜入口仍用 V2 Setup／Runtime Audit", "分离 4+4 按 01～06 教学图先生成未接受候选、再分别审阅接受与拼接", "逐个核对真实模型、父片和阶段回执"],
        "requires": ["fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors", "FastH3 V2 入口节点"],
        "boundary": "D3 只下载原生副本，不把当前镜头编译成分离图；旧单镜默认仍是原图。01～06 是可编辑 EXP，须安装新阶段/来源节点并填首帧和父片回执；活动 Relay 与训练 VSA 不兼容，需显式 Dense，不能静默换后端。",
    },
    "eav": {
        "mode": "native_only",
        "steps": ["准备视频与音频输入", "选择 EAV／Relay 组合入口", "先做短段试听再扩展时长"],
        "requires": ["EAV Composer 节点", "清晰的音频驱动或参考音频", "输入视频／首帧"],
        "boundary": "EAV wrapper 有自己的音视频时序合同，不由单镜头生成器代替。",
    },
    "low_vram": {
        "mode": "native_only",
        "steps": ["在 MODEL 链接入 LowVRAM", "需要时继续接 ChunkFFN", "保持顺序并记录显存策略"],
        "requires": ["LowVRAM Attention 节点", "ChunkFFN 节点（可选）"],
        "boundary": "显存优化是显式 MODEL patch，不会静默改变默认 H3 路线。",
    },
    "taeh3_preview": {
        "mode": "native_only",
        "steps": ["启用 TAEH3 预览入口", "采样中观察 x0 近似图", "取消后回到原生队列状态"],
        "requires": ["TAEH3 Preview Capability／Sampling Preview 节点", "对应 TAEH3 资源"],
        "boundary": "预览图是中间近似，不是最终 VAE 成片；不能用来替代交付验收。",
    },
    "topaz": {
        "mode": "native_only",
        "steps": ["先运行环境检测", "把 H3 或普通下载视频接入高清放大", "确认取消和 H.264 交付"],
        "requires": ["正式 Topaz Video AI 与 models 目录（可由环境节点自动发现）", "可读的视频输入"],
        "boundary": "输入不限定为 H3 结果；环境、模型和取消能力由 Topaz 原生节点最终核验。",
    },
    "meridian": {
        "mode": "native_only",
        "steps": ["准备图像／相机／材质输入", "选择 ConvRot INT8 权重", "按 P2/P3/P4 质量门禁验收"],
        "requires": ["Meridian ConvRot INT8 ComfyUI 权重", "VGGT-Omega 几何权重（若路线需要）", "Meridian 原生入口节点"],
        "boundary": "没有几何权重时只可做结构预检，不能声称 Meridian 成片已完成。",
    },
}


def _route_file(path: str) -> Path:
    """Resolve one allow-listed native workflow below the formal package root."""

    root = Path(__file__).resolve().parents[1]
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError("原生路线文件路径越界")
    return candidate


def handoff_d3_route(capability: str, file_path: str | None = None) -> dict[str, object]:
    """Return a concrete, download-ready native workflow handoff.

    This deliberately does not rewrite the workflow into a fake Director graph or
    queue it.  The allow-list is the contract: the user receives the exact checked-
    in native JSON/README, plus existence and format metadata, so missing optional
    runtimes remain visible instead of becoming a dead button.
    """

    if capability not in ROUTE_WORKFLOWS:
        raise ValueError("未知 D3 原生路线")
    allowed = tuple(ROUTE_WORKFLOWS[capability])
    selected = file_path or allowed[0]
    # Exact legacy aliases only: do not turn a missing path into a glob search.
    if capability == 'fast_h3_v2':
        legacy = {path.replace('/34-fasth3-v2/', '/10-speed/'): path for path in allowed}
        selected = legacy.get(selected, selected)
    if selected not in allowed:
        raise ValueError("该文件不属于所选 D3 原生路线")
    rows: list[dict[str, object]] = []
    for relative in allowed:
        path = _route_file(relative)
        exists = path.is_file()
        row: dict[str, object] = {
            "path": relative,
            "name": path.name,
            "kind": "workflow" if path.suffix.lower() == ".json" else "readme",
            "exists": exists,
        }
        if exists:
            row["bytes"] = path.stat().st_size
        rows.append(row)
    chosen_path = _route_file(selected)
    if not chosen_path.is_file():
        raise FileNotFoundError(f"原生路线文件不存在：{selected}")
    raw = chosen_path.read_bytes()
    result: dict[str, object] = {
        "schema": "t8.minimax_h3.director_d3_handoff.v1",
        "capability": capability,
        "files": rows,
        "selected": selected,
        "name": chosen_path.name,
        "kind": "workflow" if chosen_path.suffix.lower() == ".json" else "readme",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "warning": "这是仓库内核验过的原生工作流／说明副本，不会自动把当前导演台镜头映射进去，也不会排队。打开后请按该路线的输入和模型门禁操作。",
    }
    if chosen_path.suffix.lower() == ".json":
        try:
            text = raw.decode("utf-8")
            result["content"] = json.loads(text)
            # Keep the original bytes available to the browser download so a
            # handoff does not silently reformat or reorder a native workflow.
            result["text"] = text
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"原生工作流 JSON 无法读取：{chosen_path.name}") from error
    else:
        result["text"] = raw.decode("utf-8")
    return result


def export_d3_package(
    capability: str,
    project: Mapping[str, object] | None = None,
    shot_id: str | None = None,
    store: ProjectStore | None = None,
) -> dict[str, object]:
    """Export a truthful, self-contained hand-off manifest for native routes.

    The package is JSON rather than a zip on purpose: it is inspectable, diffable
    and safe to download from the Director.  Native workflow text is preserved
    byte-for-byte; local media are represented by stable IDs and server paths,
    never copied or silently re-encoded.
    """

    if capability not in ROUTE_WORKFLOWS:
        raise ValueError("未知 D3 原生路线")
    checked = validate_project(project) if project is not None else None
    preflight = inspect_d3_routes(
        checked, shot_id, store, capability=capability
    ) if checked is not None else inspect_d3_routes(capability=capability)
    files: list[dict[str, object]] = []
    for relative in ROUTE_WORKFLOWS[capability]:
        path = _route_file(relative)
        if not path.is_file():
            continue
        raw = path.read_bytes()
        item: dict[str, object] = {
            "path": relative,
            "name": path.name,
            "kind": "workflow" if path.suffix.lower() == ".json" else "readme",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "text": raw.decode("utf-8"),
        }
        if path.suffix.lower() == ".json":
            item["content"] = json.loads(item["text"])
        files.append(item)
    assets: list[dict[str, object]] = []
    if checked is not None:
        for asset in checked.get("assets", []):
            if not isinstance(asset, Mapping):
                continue
            assets.append({
                key: asset.get(key)
                for key in ("id", "kind", "name", "server_path", "duration", "width", "height")
                if key in asset
            })
    contract = dict(ROUTE_CONTRACTS.get(capability, {}))
    package: dict[str, object] = {
        "schema": "t8.minimax_h3.director_d3_route_package.v1",
        "capability": capability,
        "contract": contract,
        "preflight": preflight,
        "files": files,
        "assets": assets,
        "project": json.loads(canonical(checked)) if checked is not None else None,
        "selected_shot": shot_id,
        "warning": "路线包只保存当前项目快照、稳定素材身份和原生工作流副本；不会上传媒体、改写工作流或自动排队。",
    }
    return package


def _model_names() -> dict[str, list[str]]:
    try:
        import folder_paths

        result = {}
        for folder in (
            "diffusion_models",
            "loras",
            "vae",
            "semantic_bridge",
            "meridian",
        ):
            try:
                result[folder] = sorted(str(value) for value in folder_paths.get_filename_list(folder))
            except (KeyError, TypeError, AttributeError):
                result[folder] = []
        # Semantic Bridge recursively resolves its own nested model roots and
        # therefore is not required to register a folder_paths category.
        if not result.get("semantic_bridge"):
            try:
                from .nodes_semantic_bridge import model_paths

                result["semantic_bridge"] = sorted(model_paths())
            except (ImportError, OSError, RuntimeError):
                pass
        return result
    except Exception:
        return {}


def _dependency_rows(capability_id: str, inventory: Mapping[str, Iterable[str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_names = [name.lower() for values in inventory.values() for name in values]

    def add(label: str, ok: bool, detail: str) -> None:
        rows.append({"id": label, "ok": bool(ok), "detail": detail})

    if capability_id == "semantic_bridge":
        matches = [name for name in inventory.get("semantic_bridge", ()) if name.lower().endswith(".safetensors")]
        add("bridge_weights", bool(matches), "已发现 " + ", ".join(matches[:3]) if matches else "models/semantic_bridge 没有 .safetensors")
    elif capability_id == "fast_h3_v2":
        matches = [name for name in all_names if "fasth3" in name or "fast_h3" in name]
        add("fasth3_assets", bool(matches), "已发现 FastH3 相关模型/LoRA" if matches else "未发现 FastH3 V2 模型或 LoRA；仍可打开配方但不能排队")
    elif capability_id == "meridian":
        matches = [name for name in inventory.get("meridian", ()) if "meridian" in name.lower()]
        add("meridian_weights", bool(matches), "已发现 Meridian 资源" if matches else "未发现 models/meridian 下的权重；Omega/ConvRot 需单独授权与安装")
    elif capability_id == "topaz":
        matches = [name for name in all_names if "topaz" in name or "iris" in name or "apollo" in name]
        discovered = None
        try:
            # Read-only filesystem metadata only.  This deliberately does not
            # launch Topaz, inspect a license, or claim that a model loaded.
            from .topaz_runtime import discover_official_installation
            discovered = discover_official_installation()
        except (ImportError, OSError, RuntimeError, ValueError):
            discovered = None
        if discovered:
            add("topaz_runtime", True,
                "已发现正式 Topaz 程序、models 定义和候选权重；环境节点仍需做签名／滤镜／实际加载核验")
        else:
            add("topaz_runtime", bool(matches),
                "节点负责运行时环境检查；模型/可执行文件由 Topaz 节点再次核验" if matches
                else "未发现完整 Topaz 程序和 models 目录；可清空环境节点三项路径并启用自动发现，或手动填写")
    else:
        add("native_runtime", True, "入口节点执行时继续做其专属依赖、显存与输入检查")
    return rows


def inspect_d3_routes(
    project: Mapping[str, object] | None = None,
    shot_id: str | None = None,
    store: ProjectStore | None = None,
    *,
    node_ids: Iterable[str] | None = None,
    model_inventory: Mapping[str, Iterable[str]] | None = None,
    capability: str | None = None,
) -> dict[str, object]:
    """Return a per-shot D3 hand-off report without queuing or mutating state."""

    if node_ids is None:
        try:
            import nodes

            node_ids = nodes.NODE_CLASS_MAPPINGS.keys()
        except Exception:
            node_ids = ()
    available = {str(value) for value in node_ids}
    inventory = dict(model_inventory or _model_names())
    compile_report = None
    selected_shot = None
    if project is not None:
        checked = validate_project(project)
        if store is not None:
            compile_report = compile_project(checked, store, shot_id=shot_id)
        selected_shot = next(
            (shot for shot in checked["doc"]["shots"] if shot["id"] == shot_id), None
        )
        if shot_id and selected_shot is None:
            raise ValueError("请选择项目内有效的镜头 UUID")

    rows = []
    specs = [spec for spec in CAPABILITY_SPECS if capability in (None, spec["id"])]
    for spec in specs:
        capability_id = str(spec["id"])
        required = [str(value) for value in spec["entry_nodes"]]
        missing = [value for value in required if value not in available]
        deps = _dependency_rows(capability_id, inventory)
        deps_ok = all(bool(row["ok"]) for row in deps)
        compile_ok = bool(compile_report is None or compile_report.get("ready"))
        state = (
            "ready_for_native_workflow"
            if not missing and deps_ok and compile_ok
            else "blocked_missing_entry_or_dependency"
        )
        rows.append(
            {
                "id": capability_id,
                "label": spec["label"],
                "state": state,
                "entry_nodes": required,
                "missing_entry_nodes": missing,
                "dependencies": deps,
                "route": spec["route"],
                "workflows": list(ROUTE_WORKFLOWS.get(capability_id, ())),
                "contract": dict(ROUTE_CONTRACTS.get(capability_id, {})),
                "shot_id": selected_shot.get("id") if selected_shot else None,
                "shot_name": selected_shot.get("name") if selected_shot else None,
                "compile_ready": compile_ok,
                "execution_boundary": "handoff_to_native_workflow; no implicit queue or quality claim",
                "next_action": "打开对应原生工作流并保存为项目副本，再按该节点的真实输入/模型门禁执行",
            }
        )
    return {
        "schema": "t8.minimax_h3.director_d3_preflight.v1",
        "capabilities": rows,
        "project_id": project.get("id") if project else None,
        "shot_id": shot_id,
        "compile": compile_report,
        "warning": "ready_for_native_workflow 只代表当前镜头可安全交接到原生路线，不代表 GPU、人审或发布通过。",
    }
