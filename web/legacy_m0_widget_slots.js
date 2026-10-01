import { app } from "../../scripts/app.js";

// Preserve evidenced old widget layouts in memory only. Newer ComfyUI
// serializes seed controls and connected-widget placeholders that those old
// workflow files did not contain. Never touch an already named/current node.
const TAIL_SEED_COUNTS = new Map([
    ["MiniMaxH3CADSVisualReferenceT8Advanced", 6],
    ["MiniMaxH3DetailMixerSamplerT8Advanced", 21],
    ["MiniMaxH3RectifiedFlowRestartSamplerT8Advanced", 6],
    ["MiniMaxH3TrajectoryProbeT8Advanced", 3],
    ["MiniMaxH3TwoPassDetailMixerT8Advanced", 19],
]);

export function migrateLegacyM0WidgetSlots(graphData) {
    if (!Array.isArray(graphData?.nodes)) return 0;
    let count = 0;
    graphData.nodes = graphData.nodes.map((node) => {
        if (!node || node.widgets_values_named != null) return node;
        const values = node.widgets_values;
        if (TAIL_SEED_COUNTS.has(node.type) && Array.isArray(values) &&
            values.length === TAIL_SEED_COUNTS.get(node.type) &&
            Number.isInteger(values.at(-1)) && values.at(-1) >= 0) {
            count += 1;
            return { ...node, widgets_values: [...values, "fixed"] };
        }
        if (node.type === "MiniMaxH3FlashVSRRestoreT8Advanced" &&
            Array.isArray(values) && values.length === 4 &&
            (values[0] === 2 || values[0] === 4) &&
            Number.isInteger(values[1]) && values[1] >= 0 &&
            typeof values[2] === "boolean" &&
            ["offload_after", "clear_after", "keep_loaded"].includes(values[3])) {
            count += 1;
            return { ...node, widgets_values: [...values.slice(0, 2), "fixed", ...values.slice(2)] };
        }
        if (node.type === "MiniMaxH3MotionSegmentPlanT8Advanced" &&
            Array.isArray(values) && values.length === 4 &&
            Number.isInteger(values[0]) && Number.isInteger(values[1]) &&
            Number.isInteger(values[2]) &&
            (values[3] === "hot_ranges_only" || values[3] === "full_clip")) {
            count += 1;
            return { ...node, widgets_values: [...values.slice(0, 2), "fixed", ...values.slice(2)] };
        }
        if (node.type === "MiniMaxH3FaceRefineWindowExtractT8Advanced" &&
            Array.isArray(values) && values.length === 1 &&
            (values[0] === "reject" || values[0] === "edge_hold_exp") &&
            node.inputs?.some((pin) => pin.name === "window_index" &&
                Number.isInteger(pin.link))) {
            count += 1;
            return { ...node, widgets_values: [0, values[0]] };
        }
        const linked = ["conditioned_prompt", "media_map_json", "conditioning_report"];
        if (node.type === "MiniMaxH3NFERunContractT8Advanced" &&
            Array.isArray(values) && values.length === 1 &&
            Number.isInteger(values[0]) && values[0] >= 1 && values[0] <= 64 &&
            linked.every((name) => node.inputs?.some((pin) =>
                pin.name === name && Number.isInteger(pin.link)))) {
            count += 1;
            return { ...node, widgets_values: ["", "", "", values[0]] };
        }
        return node;
    });
    return count;
}

app.registerExtension({
    name: "minimax-h3-audio-t8.legacy-m0-widget-slots",
    beforeConfigureGraph(graphData) {
        migrateLegacyM0WidgetSlots(graphData);
    },
});
