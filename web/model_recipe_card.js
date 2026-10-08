import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { currentRecipe, editableRecipeFields, previewWidgetRecipe, validateFreshPreview } from "./model_recipe_diff.mjs";

// This action adds no widget, socket, serialization field or auto callback.
const RESOURCE_WIDGETS = {
    unet_name: "diffusion_models", vae_name: "vae", clip_name: "clip",
    clip_name1: "clip", clip_name2: "clip", clip_name3: "clip",
    lora_name: "loras",
};

function categoryFor(node, name) {
    if (name === "model_name" && ["MiniMaxH3SemanticBridgeConfigT8", "MiniMaxH3SemanticBridgeAutoConfigT8"].includes(node.comfyClass)) {
        return "semantic_bridge";
    }
    return RESOURCE_WIDGETS[name];
}

function cell(tag, text) {
    const item = document.createElement(tag);
    item.textContent = text;
    item.style.cssText = "padding:6px;border-bottom:1px solid #555;text-align:left;overflow-wrap:anywhere;";
    return item;
}

function showEvidence(output, data, node) {
    output.replaceChildren();
    const card = data.inspection?.recipe_card;
    const table = document.createElement("table");
    table.style.cssText = "width:100%;border-collapse:collapse;";
    const heading = document.createElement("tr");
    heading.append(cell("th", "字段"), cell("th", "值"), cell("th", "证据范围"));
    table.append(heading);
    for (const [name, key] of [["任务", "task"], ["剪枝结构", "structure"], ["存储dtype", "storage_dtypes"],
        ["量化布局", "quantization_layouts"], ["训练rank", "training_rank"], ["训练alpha", "training_alpha"],
        ["文件中的未限定alpha（不自动套用）", "unscoped_alpha_metadata"],
        ["sigma／clock", "sigma_and_clock"], ["VAE像素倍率", "vae_pixel_multiplier"]]) {
        if (!card?.[key]) continue;
        const item = card[key], row = document.createElement("tr");
        row.append(cell("td", name), cell("td", item.value === null ? "unknown" : JSON.stringify(item.value)), cell("td", item.source));
        table.append(row);
    }
    if (data.profile) {
        for (const [name, value] of Object.entries({ "Bridge架构": data.profile.architecture, ...data.profile.settings })) {
            const row = document.createElement("tr");
            row.append(cell("td", name), cell("td", JSON.stringify(value)), cell("td", data.profile.settings_source));
            table.append(row);
        }
    }
    const runtime = Object.fromEntries((node.widgets || []).filter(w =>
        ["strength_model", "strength_clip", "weight_dtype", "alpha", "magnitude_match", "token_scope", "chunk_tokens"].includes(w.name))
        .map(w => [w.name, w.value]));
    const row = document.createElement("tr");
    row.append(cell("td", "当前运行选择"), cell("td", JSON.stringify(runtime)), cell("td", "当前widget；不是训练alpha／已执行结果"));
    table.append(row);
    const details = document.createElement("details"), summary = document.createElement("summary"), raw = document.createElement("pre");
    summary.textContent = "完整报告／来源与限制";
    raw.textContent = JSON.stringify(data, null, 2);
    raw.style.cssText = "white-space:pre-wrap;overflow-wrap:anywhere;";
    details.append(summary, raw);
    output.append(table, details);
}

function addRecipeEditor(dialog, node, nodeData, onApplied) {
    let fields = editableRecipeFields(node, nodeData);
    if (!Object.keys(fields).length) {
        dialog.append(cell("p", "此节点只读查看；导入修改暂限静态Core加载器／Bridge配置。未知节点、连接输入及自定义回调不迁移，仍可原样运行。"));
        return null;
    }
    const section = document.createElement("details"), title = document.createElement("summary");
    title.textContent = "导入／替换配方 · 先看差异，再明确确认（可撤销）";
    const note = document.createElement("p");
    note.textContent = "只改本节点已有的明确字段，不自动保存、不排队、不改其他节点或连线。普通模型下拉操作保持原样。";
    const editor = document.createElement("textarea");
    editor.setAttribute("aria-label", "声明式模型配方JSON");
    editor.style.cssText = "width:100%;height:160px;background:#15171a;color:#eee;font-family:monospace;";
    editor.value = JSON.stringify(currentRecipe(node, fields), null, 2);
    const status = document.createElement("p"), difference = document.createElement("table");
    difference.style.cssText = "width:100%;border-collapse:collapse;";
    const previewButton = document.createElement("button"), applyButton = document.createElement("button");
    previewButton.textContent = "预览精确差异（不改画布）";
    applyButton.textContent = "我确认，只应用上表差异";
    applyButton.disabled = true;
    let preview, graph, serialized, nodeSnapshot, editorText;
    editor.addEventListener("input", () => { applyButton.disabled = true; preview = null; status.textContent = "配方已编辑；请重新预览。"; });
    previewButton.onclick = () => {
        applyButton.disabled = true;
        preview = null;
        difference.replaceChildren();
        try {
            fields = editableRecipeFields(node, nodeData);
            const candidate = previewWidgetRecipe(editor.value, node, fields);
            graph = node.graph;
            if (!graph || graph.getNodeById(node.id) !== node) throw new Error("节点已移除或画布已更换。");
            serialized = JSON.stringify(graph.serialize());
            if (serialized.length > 4 * 1024 * 1024) throw new Error("此大画布仅提供只读卡；请用原节点控件修改。");
            nodeSnapshot = JSON.stringify(node.serialize());
            editorText = editor.value;
            for (const change of candidate.changes) {
                const row = document.createElement("tr");
                row.append(cell("td", change.field), cell("td", JSON.stringify(change.before)), cell("td", "→ " + JSON.stringify(change.after)));
                difference.append(row);
            }
            preview = candidate;
            status.textContent = `${preview.changes.length}个明确差异。尚未应用、保存或排队；其他字段和连线保持。`;
            applyButton.disabled = !preview.changes.length;
        } catch (error) { status.textContent = String(error); }
    };
    applyButton.onclick = () => {
        applyButton.disabled = true;
        let entered = false;
        try {
            if (!preview || editorText !== editor.value || graph !== node.graph || graph.getNodeById(node.id) !== node ||
                JSON.stringify(graph.serialize()) !== serialized || JSON.stringify(node.serialize()) !== nodeSnapshot) {
                throw new Error("画布／节点／配方在预览后发生变化，请重新预览；本次未应用。");
            }
            fields = editableRecipeFields(node, nodeData);
            const fresh = validateFreshPreview(preview, editor.value, node, fields);
            graph.beforeChange();
            entered = true;
            // Only static declared fields are supported. No import-time callback/code execution.
            for (const change of fresh.changes) fields[change.field].widget.value = change.after;
            graph.afterChange();
            entered = false;
            node.setDirtyCanvas?.(true, true);
            status.textContent = "已按明确确认应用。未自动保存／排队；可用 ComfyUI 撤销，保存由你决定。";
            preview = null;
            onApplied();
        } catch (error) {
            if (entered) {
                for (const change of preview.changes) fields[change.field].widget.value = change.before;
                graph.afterChange();
            }
            status.textContent = `${String(error)} 未标记成功。`;
        }
    };
    section.append(title, note, editor, previewButton, difference, status, applyButton);
    dialog.append(section);
    return settings => {
        const selected = Object.fromEntries(Object.entries(settings || {}).filter(([name]) => Object.hasOwn(fields, name)));
        editor.value = JSON.stringify({ ...currentRecipe(node, fields), settings: { ...currentRecipe(node, fields).settings, ...selected } }, null, 2);
        preview = null;
        applyButton.disabled = true;
        status.textContent = "已把内容绑定推荐值填入编辑框；还未应用，必须先预览再确认。";
        section.open = true;
    };
}

async function showCard(node, nodeData) {
    const resources = (node.widgets || []).filter(widget => categoryFor(node, widget.name));
    // Resolve only an explicit field. Never pick the "first similar" installed model.
    if (!resources.length) return;
    const dialog = document.createElement("dialog");
    dialog.style.cssText = "max-width:820px;width:85vw;background:#202328;color:#eee;border:1px solid #777;padding:20px;";
    const title = document.createElement("h3");
    title.textContent = "T8 模型配方卡 · 证据与差异";
    const note = document.createElement("p");
    note.textContent = "训练 rank / alpha ≠ LoRA strength；Bridge alpha 独立。缺证据显示 unknown，不自动换模型或改步数。";
    dialog.style.maxHeight = "85vh";
    dialog.style.overflow = "auto";
    const output = document.createElement("div");
    output.style.cssText = "overflow-wrap:anywhere;max-height:42vh;overflow:auto;";
    const close = document.createElement("button");
    close.textContent = "关闭";
    close.onclick = () => dialog.close();
    dialog.append(title, note);
    const overridden = node.comfyClass === "MiniMaxH3HyperVAE2xLoaderEXPT8" &&
        String(node.widgets?.find(w => w.name === "absolute_path")?.value || "").trim().replace(/^"+|"+$/g, "");
    if (overridden) {
        dialog.append(cell("p", "此HyperVAE节点 absolute_path 覆盖了 vae_name。目录下拉项不是实际执行权重，本卡不拿它冒充真实文件；保留覆盖设置，实际倍率暂为 unknown。不会通过此接口读取任意绝对路径。"));
    }
    const resourceButtons = [];
    let inspectionEpoch = 0;
    for (const widget of resources) {
        const button = document.createElement("button");
        button.textContent = `${widget.name}: ${String(widget.value)}`;
        button.style.margin = "4px";
        button.disabled = Boolean(overridden);
        resourceButtons.push({ widget, button });
        button.onclick = async () => {
            const epoch = ++inspectionEpoch, selection = widget.value;
            output.textContent = "读取所选文件头…";
            try {
                const response = await api.fetchApi("/minimax_h3_t8/director/model-recipe/inspect", {
                    method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ category: categoryFor(node, widget.name), selection }),
                });
                const data = await response.json();
                if (epoch !== inspectionEpoch || selection !== widget.value || !dialog.open) return;
                if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
                // textContent: filenames and metadata are untrusted, never HTML.
                showEvidence(output, data, node);
                if (data.profile?.settings && fillRecommendation && node.comfyClass === "MiniMaxH3SemanticBridgeConfigT8") {
                    const recommend = document.createElement("button");
                    recommend.textContent = "将内容绑定Bridge推荐值填入差异编辑框（不应用）";
                    recommend.onclick = () => fillRecommendation(data.profile.settings);
                    output.append(recommend);
                }
            } catch (error) {
                if (epoch !== inspectionEpoch || !dialog.open) return;
                output.textContent = `未读取；画布未改动。${String(error)}`;
            }
        };
        dialog.append(button);
    }
    dialog.append(output);
    const fillRecommendation = addRecipeEditor(dialog, node, nodeData, () => {
        inspectionEpoch++;
        output.replaceChildren(cell("p", "当前字段已更新；旧报告不再表示当前选择。需要时重新点击所选资源读取。"));
        for (const { widget, button } of resourceButtons) button.textContent = `${widget.name}: ${String(widget.value)}`;
    });
    dialog.append(close);
    dialog.addEventListener("close", () => dialog.remove(), { once: true });
    document.body.append(dialog);
    dialog.showModal();
}

app.registerExtension({
    name: "T8.ModelRecipeCard",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const inputNames = Object.keys({ ...nodeData.input?.required, ...nodeData.input?.optional });
        if (!inputNames.some(name => RESOURCE_WIDGETS[name]) &&
            !["MiniMaxH3SemanticBridgeConfigT8", "MiniMaxH3SemanticBridgeAutoConfigT8"].includes(nodeData.name)) return;
        const previous = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (...args) {
            const result = previous?.apply(this, args);
            const options = args[1];
            if (Array.isArray(options)) options.push({ content: "T8 模型配方卡（证据／差异）", callback: () => showCard(this, nodeData) });
            return result;
        };
    },
});
