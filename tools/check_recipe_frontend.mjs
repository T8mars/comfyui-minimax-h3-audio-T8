// Exercise actual editor/menu source; fake DOM proves behavior, not browser layout.
import assert from "node:assert/strict";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { resolve, relative } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import * as helpers from "../web/model_recipe_diff.mjs";

class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.style = {}; this.listeners = new Map(); this.value = ""; }
    append(...items) { this.children.push(...items); }
    replaceChildren(...items) { this.children = items; this.textContent = ""; }
    setAttribute(name, value) { this[name] = value; }
    addEventListener(name, callback) { this.listeners.set(name, callback); }
    showModal() { this.open = true; }
    close() { this.open = false; this.listeners.get("close")?.(); }
    remove() { this.removed = true; }
}
let extension;
const body = new Element("body"), requests = [];
const context = vm.createContext({ ...helpers, Object, JSON, console,
    app: { registerExtension: value => { extension = value; } },
    api: { fetchApi: async (path, options) => { requests.push({ path, options });
        return { ok: true, json: async () => ({ inspection: { recipe_card: {
            task: { value: "<script>untrusted</script>", source: "metadata_declaration" } } } }) }; } },
    document: { body, createElement: tag => new Element(tag) } });
const source = readFileSync(new URL("../web/model_recipe_card.js", import.meta.url), "utf8").replace(/^import .*;\r?\n/gm, "");
vm.runInContext(source, context);
const info = { name: "VAELoader", input: { required: { vae_name: [["old.safetensors", "new.safetensors"]] } } };
class Node {
    constructor() { this.id = 8; this.comfyClass = "VAELoader"; this.inputs = []; this.widgets = [{ name: "vae_name", value: "old.safetensors" }]; }
    getExtraMenuOptions(_, options) { options.push({ content: "previous" }); return 17; }
    serialize() { return { id: this.id, type: this.comfyClass, widgets_values: this.widgets.map(w => w.value), inputs: this.inputs, properties: { unrelated: "keep" } }; }
    setDirtyCanvas() { this.dirty = true; }
}
await extension.beforeRegisterNodeDef(Node, info);
const node = new Node(), undo = [];
const graph = { getNodeById: id => id === node.id ? node : null,
    serialize: () => ({ nodes: [node.serialize()], links: [], extra: { keep: true } }),
    beforeChange: () => { undo.push(node.widgets[0].value); }, afterChange: () => {} };
node.graph = graph;
const find = (root, predicate) => predicate(root) ? root : root.children.map(c => find(c, predicate)).find(Boolean);
const menu = [];
assert.equal(node.getExtraMenuOptions(null, menu), 17);
assert.equal(menu[0].content, "previous");
await menu[1].callback();
const dialog = body.children.at(-1), editor = find(dialog, e => e.tag === "textarea");
const preview = find(dialog, e => e.textContent === "预览精确差异（不改画布）");
const apply = find(dialog, e => e.textContent === "我确认，只应用上表差异");
assert.equal(node.widgets.length, 1);
assert.equal(apply.disabled, true);
const initial = JSON.stringify(graph.serialize());
editor.value = JSON.stringify({ schema: helpers.RECIPE_SCHEMA, node_type: "VAELoader", settings: { vae_name: "new.safetensors" } });
preview.onclick();
assert.equal(JSON.stringify(graph.serialize()), initial);
assert.equal(apply.disabled, false);
// Editing after preview, even without dispatching input, is rejected.
editor.value += " ";
apply.onclick();
assert.equal(JSON.stringify(graph.serialize()), initial);
preview.onclick();
node.inputs.push({ name: "vae_name", link: 91 });
apply.onclick();
assert.equal(node.widgets[0].value, "old.safetensors");
assert.equal(undo.length, 0);
node.inputs = [];
preview.onclick();
apply.onclick();
assert.equal(node.widgets[0].value, "new.safetensors");
assert.deepEqual(undo, ["old.safetensors"]);
assert.equal(node.widgets.length, 1);
assert.equal(requests.length, 0); // no header fetch until explicit inspection; no save/queue ever
node.widgets[0].value = undo.pop();
assert.equal(JSON.stringify(graph.serialize()), initial);
const fields = helpers.editableRecipeFields(node, info);
for (const settings of [{ vae_name: "guess.safetensors" }, { __proto__: "bad", unknown: 1 }, { vae_name: null }]) {
    assert.throws(() => helpers.previewWidgetRecipe(JSON.stringify({ schema: helpers.RECIPE_SCHEMA, node_type: "VAELoader", settings }), node, fields));
}
assert.throws(() => helpers.previewWidgetRecipe(JSON.stringify({ schema: helpers.RECIPE_SCHEMA, node_type: "Other", settings: { vae_name: "old.safetensors" } }), node, fields));
assert.deepEqual(helpers.editableRecipeFields({ ...node, comfyClass: "UnknownLoader" }, info), {});
// A third-party resource loader gets the read-only card, not widget migration.
// Its original menu, dynamic callbacks, native selection and saved fields stay
// owned by that loader. This is fake-DOM behavior, not a third-party GPU test.
let thirdPartyCalls = 0;
class ThirdParty {
    constructor() {
        this.id = 91; this.comfyClass = "ThirdPartyLoader";
        this.inputs = [{ name: "model", link: 37 }];
        this.widgets = [{ name: "vae_name", value: "custom.safetensors", callback: () => { thirdPartyCalls++; } },
            { name: "custom_mode", value: "user-choice" }];
        this.properties = { owner: "third-party", untouched: true };
    }
    getExtraMenuOptions(_, options) { options.push({ content: "third-party" }); return 31; }
    serialize() { return { id: this.id, type: this.comfyClass, inputs: this.inputs,
        widgets_values: this.widgets.map(w => w.value), properties: this.properties }; }
}
const originalThirdPartySerializer = ThirdParty.prototype.serialize;
await extension.beforeRegisterNodeDef(ThirdParty, { name: "ThirdPartyLoader",
    input: { required: { vae_name: [["custom.safetensors"]], custom_mode: [["user-choice"]] } } });
const thirdParty = new ThirdParty(), thirdPartyMenu = [];
const thirdPartySaved = JSON.stringify(thirdParty.serialize());
assert.equal(thirdParty.getExtraMenuOptions(null, thirdPartyMenu), 31);
assert.equal(thirdPartyMenu[0].content, "third-party");
await thirdPartyMenu[1].callback();
const thirdPartyDialog = body.children.at(-1);
assert.ok(find(thirdPartyDialog, e => e.textContent?.startsWith("此节点只读查看")));
assert.equal(find(thirdPartyDialog, e => e.tag === "textarea"), undefined);
assert.equal(ThirdParty.prototype.serialize, originalThirdPartySerializer);
assert.equal(JSON.stringify(thirdParty.serialize()), thirdPartySaved);
thirdParty.widgets[0].callback();
assert.equal(thirdPartyCalls, 1);
thirdPartyDialog.close();
assert.equal(requests.length, 0);
const inspect = find(dialog, e => e.tag === "button" && e.textContent.startsWith("vae_name:"));
await inspect.onclick();
assert.equal(requests.length, 1);
assert.equal(requests[0].path, "/minimax_h3_t8/director/model-recipe/inspect");
assert.ok(find(dialog, e => e.textContent === '"<script>untrusted</script>"'));
assert.equal(find(dialog, e => e.innerHTML !== undefined), undefined);
dialog.close();
assert.equal(dialog.removed, true);
const result = { status: "pass", scope: "actual_extension_fake_DOM_not_browser", no_save_or_queue: true,
    preserved_widget_order_and_links: true, stale_preview_rejected: true, declared_schema_validated: true,
    unknown_loader_readonly_original_callback_and_serialization_preserved: true,
    source_sha256: Object.fromEntries(["../web/model_recipe_card.js", "../web/model_recipe_diff.mjs", "./check_recipe_frontend.mjs"]
        .map(name => [name, createHash("sha256").update(readFileSync(new URL(name, import.meta.url))).digest("hex")])) };
if (process.argv[2] === "--receipt") {
    const root = fileURLToPath(new URL("../artifacts/development/radar-r6-20261005/", import.meta.url));
    const path = resolve(process.argv[3]), inside = relative(root, path);
    assert.ok(inside && !inside.startsWith("..") && !inside.includes(":"));
    assert.equal(existsSync(path), false);
    writeFileSync(path, JSON.stringify(result, null, 2), { flag: "wx" });
}
console.log(JSON.stringify(result));
