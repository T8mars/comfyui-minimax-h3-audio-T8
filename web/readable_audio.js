import { app } from "../../scripts/app.js";

const IDS = new Set(["MiniMaxH3NodeSourceDiagnosticT8", "MiniMaxH3AudioSourceExplanationT8", "MiniMaxH3PostprocessSaveEXPT8"]);

app.registerExtension({
    name: "minimax-h3-audio-t8.readable-audio",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!IDS.has(nodeData.name)) return;
        const created = nodeType.prototype.onNodeCreated;
        const executed = nodeType.prototype.onExecuted;
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            this._t8ReadableCleanup?.();
            const text = document.createElement("textarea");
            text.readOnly = true;
            text.setAttribute("aria-label", "T8只读诊断说明");
            text.style.cssText = "box-sizing:border-box;width:100%;height:100%;min-height:200px;resize:none;white-space:pre-wrap;overflow:auto;padding:12px;font:14px/1.6 system-ui,sans-serif;color:var(--fg-color,#ddd);background:var(--comfy-input-bg,#222)";
            text.value = nodeData.name === "MiniMaxH3PostprocessSaveEXPT8"
                ? "后处理状态：尚未运行。成功片和原片为独立出口；失败不伪装为增强成功。运行后显示持久状态及原片SHA。"
                : "运行后显示来源／音频说明。此面板不修改设置，不会启动采样。";
            const stop = (event) => event.stopPropagation();
            text.addEventListener("pointerdown", stop);
            text.addEventListener("wheel", stop);
            const reportWidget = this.addDOMWidget("t8_readable_report", "text", text, {
                serialize: false, getMinHeight: () => 200, getMaxHeight: () => 600,
            });
            // Current Core also checks the widget itself. Keep this new node's
            // transient receipt out of saved values; leave the old nodes alone.
            if (nodeData.name === "MiniMaxH3PostprocessSaveEXPT8") reportWidget.serialize = false;
            this._t8ReadableRender = (value) => {
                text.value = String(Array.isArray(value) ? value[0] ?? "" : value ?? "");
            };
            this._t8ReadableCleanup = () => {
                text.removeEventListener("pointerdown", stop);
                text.removeEventListener("wheel", stop);
                text.remove();
                this._t8ReadableRender = this._t8ReadableCleanup = null;
            };
            this.setSize?.([Math.max(this.size?.[0] ?? 0, 560), Math.max(this.size?.[1] ?? 0, 380)]);
            return result;
        };
        nodeType.prototype.onExecuted = function (message) {
            const result = executed?.apply(this, arguments);
            this._t8ReadableRender?.(message?.text);
            return result;
        };
        nodeType.prototype.onRemoved = function () {
            this._t8ReadableCleanup?.();
            return removed?.apply(this, arguments);
        };
    },
});
