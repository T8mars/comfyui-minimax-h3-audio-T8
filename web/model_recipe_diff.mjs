// Declarative widget recipes only: never execute imported code or replace a graph.
export const RECIPE_SCHEMA = "t8.widget-recipe.v1";
const TYPES = new Set([
    "UNETLoader", "CLIPLoader", "VAELoader", "LoraLoader", "LoraLoaderModelOnly",
    "MiniMaxH3SemanticBridgeConfigT8", "MiniMaxH3SemanticBridgeAutoConfigT8",
]);
const FIELDS = new Set([
    "unet_name", "weight_dtype", "clip_name", "type", "device", "vae_name",
    "lora_name", "strength_model", "strength_clip", "model_name", "enabled",
    "alpha", "magnitude_match", "token_scope", "chunk_tokens",
]);
const own = (o, key) => Object.prototype.hasOwnProperty.call(o, key);
const scalar = value => value === null || typeof value === "string" || typeof value === "boolean" ||
    (typeof value === "number" && Number.isFinite(value));
const equal = (a, b) => typeof a === typeof b && a === b;
const plain = o => o !== null && typeof o === "object" && !Array.isArray(o) &&
    Object.getPrototypeOf(o) === Object.prototype;

export function editableRecipeFields(node, nodeData) {
    if (!TYPES.has(node.comfyClass)) return {};
    const specs = { ...nodeData.input?.required, ...nodeData.input?.optional };
    const result = {};
    for (const widget of node.widgets || []) {
        const name = widget.name, spec = specs[name];
        if (!FIELDS.has(name) || !Array.isArray(spec) || !scalar(widget.value)) continue;
        if ((node.widgets || []).filter(w => w.name === name).length !== 1) continue;
        // Converted/linked inputs must keep their producer, not a hidden widget override.
        if (node.inputs?.some(i => i.name === name && i.link != null)) continue;
        const kind = spec[0];
        if (!Array.isArray(kind) && !["INT", "FLOAT", "BOOLEAN", "STRING"].includes(kind)) continue;
        Object.defineProperty(result, name, { value: { widget, spec }, enumerable: true });
    }
    return result;
}

export function currentRecipe(node, fields) {
    return { schema: RECIPE_SCHEMA, node_type: node.comfyClass,
        settings: Object.fromEntries(Object.entries(fields).map(([name, value]) => [name, value.widget.value])) };
}

function checkValue(name, value, spec) {
    if (!scalar(value) || value === null || (typeof value === "string" && value.length > 4096)) {
        throw new Error(`字段 ${name} 必须是有界的明确标量，不能是 null／代码／对象。`);
    }
    const [kind, options = {}] = spec;
    if (Array.isArray(kind)) {
        if (!kind.some(choice => equal(choice, value))) throw new Error(`字段 ${name} 不是当前节点的可选值；不猜替代模型。`);
    } else if ((kind === "INT" && !Number.isSafeInteger(value)) ||
               (kind === "FLOAT" && typeof value !== "number") ||
               (kind === "BOOLEAN" && typeof value !== "boolean") ||
               (kind === "STRING" && typeof value !== "string")) {
        throw new Error(`字段 ${name} 类型不符：需要 ${kind}。`);
    }
    if (typeof value === "number" &&
        ((Number.isFinite(options.min) && value < options.min) ||
         (Number.isFinite(options.max) && value > options.max))) throw new Error(`字段 ${name} 超出节点原有范围。`);
}

export function previewWidgetRecipe(text, node, fields) {
    if (typeof text !== "string" || text.length > 65536) throw new Error("配方JSON过大。最多65536字符。");
    const recipe = JSON.parse(text);
    if (!plain(recipe) || Object.keys(recipe).sort().join(",") !== "node_type,schema,settings" ||
        recipe.schema !== RECIPE_SCHEMA || recipe.node_type !== node.comfyClass || !plain(recipe.settings)) {
        throw new Error("只接受当前节点类型的 t8.widget-recipe.v1 声明式配方；不是工作流或脚本导入器。");
    }
    if (!Object.keys(recipe.settings).length || Object.keys(recipe.settings).length > 128) throw new Error("配方字段数不合法。");
    const current = currentRecipe(node, fields).settings;
    const changes = [];
    for (const [name, after] of Object.entries(recipe.settings)) {
        if (!own(fields, name)) throw new Error(`字段 ${name} 不可在此预览修改（未知／连接／不支持）；原节点仍可正常运行。`);
        checkValue(name, after, fields[name].spec);
        if (!equal(current[name], after)) changes.push({ field: name, before: current[name], after });
    }
    return { current, proposed: recipe.settings, changes, applied: false, saved: false, queued: false };
}

export function validateFreshPreview(preview, text, node, fields) {
    const fresh = previewWidgetRecipe(text, node, fields);
    if (JSON.stringify(preview) !== JSON.stringify(fresh)) throw new Error("预览后字段或配方已改变，请重新预览。");
    return fresh;
}
