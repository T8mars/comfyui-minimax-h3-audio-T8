import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_ID = "MiniMaxH3LongVideoBackgroundStartT8";
const BASE = "/minimax_h3_t8/long_video/background";

function chainId(node) {
    return String(node.widgets?.find((widget) => widget.name === "chain_id")?.value ?? "").trim();
}

function notify(summary, detail, severity = "info") {
    const toast = app.extensionManager?.toast;
    if (toast?.add) {
        toast.add({ severity, summary, detail, life: severity === "error" ? 8000 : 5000 });
    } else {
        console[severity === "error" ? "error" : "info"](`${summary}: ${detail}`);
    }
}

function statusText(state) {
    const progress = Number.isFinite(state.accepted_count)
        ? ` · accepted ${state.accepted_count}`
        : "";
    const retry = Number.isFinite(state.retry_count)
        ? ` · retry ${state.retry_count}/${state.max_retries}`
        : "";
    const location = state.runtime_location ? ` · ${state.runtime_location}` : "";
    const recovery = state.recovery_required
        ? state.recovery_action === "compose_accepted"
            ? " · 已接受完成，请运行合成"
            : " · 需重新排队工作流一次"
        : "";
    return `${state.state ?? "unknown"}${progress}${retry}${location}${recovery}`;
}

async function request(node, action = "status") {
    const chain = chainId(node);
    if (!chain) {
        notify("MiniMax H3 background", "chain_id 不能为空", "error");
        return;
    }
    const suffix = action === "status" ? "" : `/${action}`;
    const response = await api.fetchApi(`${BASE}/${encodeURIComponent(chain)}${suffix}`, {
        method: action === "status" ? "GET" : "POST",
        cache: "no-store",
    });
    const payload = await response.json();
    if (!response.ok) {
        notify("MiniMax H3 background", payload.error ?? `HTTP ${response.status}`, "error");
        return;
    }
    node._minimaxH3BackgroundState = payload;
    node.setDirtyCanvas?.(true, true);
    notify("MiniMax H3 background", statusText(payload), "success");
}

function addControl(node, label, action) {
    const widget = node.addWidget("button", label, null, () => {
        request(node, action).catch((error) => {
            notify("MiniMax H3 background", String(error), "error");
        });
    });
    widget.options ??= {};
    widget.options.serialize = false;
}

app.registerExtension({
    name: "minimax-h3-audio-t8.long-video-background",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) {
            return;
        }
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            addControl(this, "status / 状态", "status");
            addControl(this, "pause / 当前段后暂停", "pause");
            addControl(this, "resume / 继续", "resume");
            addControl(this, "cancel / 取消", "cancel");
            return result;
        };
    },
});
