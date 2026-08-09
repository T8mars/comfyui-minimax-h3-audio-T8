import { app } from "../../scripts/app.js";

const NODE_ID = "MiniMaxH3AudioConditioningT8";
const RH_NODE_IDS = [
    "MiniMaxH3AudioConditioningT8",
    "MiniMaxH3AudioLatentControlT8",
    "MiniMaxH3DurationPlannerT8",
    "MiniMaxH3AudioWindowT8",
    "MiniMaxH3PromptTagsT8",
    "MiniMaxH3AVDecodeT8",
    "MiniMaxH3AudioMixT8",
    "MiniMaxH3OutputTrimT8",
    "MiniMaxH3PreflightT8",
    "MiniMaxH3DualClockSamplerT8",
    "MiniMaxH3MultiRateSamplerEXPT8",
    "MiniMaxH3StillConditioningT8",
    "MiniMaxH3StillPreflightT8",
    "MiniMaxH3StillDecodeT8",
];

// Keep the backend/API values unchanged. Only the widget-facing strings are
// localized so existing API prompts and old workflows remain compatible.
const TASK_TYPE_LABELS = new Map([
    ["auto", "auto — 自动判断"],
    ["T2VA", "T2VA — 文生音视频"],
    ["I2VA", "I2VA — 图生音视频（首帧）"],
    ["FL2VA", "FL2VA — 首尾帧生音视频"],
    ["L2VA", "L2VA — 尾帧生音视频"],
    ["Ref2VA", "Ref2VA — 参考生音视频"],
    ["Hybrid", "Hybrid — 关键帧+参考混合生成"],
]);

const DISPLAY_TO_TASK_TYPE = new Map(
    [...TASK_TYPE_LABELS].map(([taskType, label]) => [label, taskType]),
);
const DISPLAY_VALUES = [...TASK_TYPE_LABELS.values()];

function toDisplayValue(value) {
    if (TASK_TYPE_LABELS.has(value)) {
        return TASK_TYPE_LABELS.get(value);
    }
    return DISPLAY_TO_TASK_TYPE.has(value) ? value : TASK_TYPE_LABELS.get("auto");
}

function toBackendValue(value) {
    return DISPLAY_TO_TASK_TYPE.get(value) ?? value;
}

function localizeTaskTypeWidget(node) {
    const widget = node.widgets?.find((candidate) => candidate.name === "task_type");
    if (!widget) {
        return;
    }

    widget.options ??= {};
    widget.options.values = DISPLAY_VALUES;
    widget.value = toDisplayValue(widget.value);

    if (!widget._minimaxH3TaskTypeLocalized) {
        widget._minimaxH3TaskTypeLocalized = true;
        widget.serializeValue = () => toBackendValue(widget.value);
    }
}

app.registerExtension({
    name: "minimax-h3-audio-t8.task-type-labels",
    rh: {
        type: "nodes",
        nodes: RH_NODE_IDS,
    },
    async setup() {
        if (typeof app.ensureNodesRegistered !== "function") return;
        await app.ensureNodesRegistered(new Set(RH_NODE_IDS));
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            localizeTaskTypeWidget(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            // Legacy workflows store canonical values such as "I2VA". Convert
            // them after LiteGraph restores widgets_values and before ComfyUI's
            // invalid-combo reset pass runs.
            localizeTaskTypeWidget(this);
            return result;
        };
    },
});
