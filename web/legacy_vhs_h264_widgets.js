import { app } from "../../scripts/app.js";

// Two evidenced legacy H.264 ten-position layouts. Current VHS restores a
// named object correctly but misassigns these positional values. Only change
// the in-memory import; never rewrite a user's workflow file or touch other
// VHS formats.
const FIELDS = [
    "frame_rate", "loop_count", "filename_prefix", "format", "pix_fmt",
    "crf", "save_metadata", "trim_to_audio", "pingpong", "save_output",
];
const API_ORDER_FIELDS = [
    "crf", "filename_prefix", "format", "frame_rate", "loop_count",
    "pingpong", "pix_fmt", "save_metadata", "save_output", "trim_to_audio",
];

function isLegacyH264(values) {
    return Array.isArray(values) && values.length === FIELDS.length &&
        Number.isInteger(values[0]) && values[0] > 0 && values[0] <= 120 &&
        Number.isInteger(values[1]) && values[1] >= 0 &&
        typeof values[2] === "string" && values[2].length > 0 &&
        values[3] === "video/h264-mp4" &&
        (values[4] === "yuv420p" || values[4] === "yuv420p10le") &&
        Number.isInteger(values[5]) && values[5] >= 0 && values[5] <= 100 &&
        values.slice(6).every((value) => typeof value === "boolean");
}

function isLegacyApiOrderH264(values) {
    return Array.isArray(values) && values.length === API_ORDER_FIELDS.length &&
        Number.isInteger(values[0]) && values[0] >= 0 && values[0] <= 100 &&
        typeof values[1] === "string" && values[1].length > 0 &&
        values[2] === "video/h264-mp4" &&
        Number.isInteger(values[3]) && values[3] > 0 && values[3] <= 120 &&
        Number.isInteger(values[4]) && values[4] >= 0 &&
        typeof values[5] === "boolean" &&
        (values[6] === "yuv420p" || values[6] === "yuv420p10le") &&
        values.slice(7).every((value) => typeof value === "boolean");
}

export function migrateLegacyVhsH264Graph(graphData) {
    if (!Array.isArray(graphData?.nodes)) return 0;
    let count = 0;
    graphData.nodes = graphData.nodes.map((node) => {
        if (node?.type !== "VHS_VideoCombine" || node.widgets_values_named != null) return node;
        const order = isLegacyH264(node.widgets_values) ? FIELDS :
            isLegacyApiOrderH264(node.widgets_values) ? API_ORDER_FIELDS : null;
        if (!order) return node;
        count += 1;
        return {
            ...node,
            widgets_values: Object.fromEntries(
                FIELDS.map((field) => [field, node.widgets_values[order.indexOf(field)]]),
            ),
        };
    });
    return count;
}

app.registerExtension({
    name: "minimax-h3-audio-t8.legacy-vhs-h264-widgets",
    beforeConfigureGraph(graphData) {
        migrateLegacyVhsH264Graph(graphData);
    },
});
