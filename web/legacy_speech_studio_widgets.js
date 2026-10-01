import { app } from "../../scripts/app.js";

// Older SpeechStudio graphs saved 21 positional values before ComfyUI added
// the seed's control_after_generate widget. Without this in-memory bridge,
// every value after seed shifts to the wrong input during native import.
const RELEASE_POLICIES = new Set([
    "keep_loaded", "unload_all_models", "clear_execution_cache",
]);

function isLegacySpeech(values) {
    return Array.isArray(values) && values.length === 21 &&
        Number.isInteger(values[0]) && values[0] >= 0 &&
        Number.isInteger(values[1]) && values[1] >= 0 &&
        typeof values[2] === "number" && Number.isFinite(values[2]) && values[2] > 0 &&
        typeof values[5] === "string" && typeof values[6] === "string" &&
        RELEASE_POLICIES.has(values[20]);
}

export function migrateLegacySpeechStudioGraph(graphData) {
    if (!Array.isArray(graphData?.nodes)) return 0;
    let count = 0;
    graphData.nodes = graphData.nodes.map((node) => {
        if (node?.type !== "MiniMaxH3SpeechStudioT8" ||
            node.widgets_values_named != null || !isLegacySpeech(node.widgets_values)) {
            return node;
        }
        count += 1;
        return {
            ...node,
            widgets_values: [
                ...node.widgets_values.slice(0, 2), "fixed",
                ...node.widgets_values.slice(2),
            ],
        };
    });
    return count;
}

app.registerExtension({
    name: "minimax-h3-audio-t8.legacy-speech-studio-widgets",
    beforeConfigureGraph(graphData) {
        migrateLegacySpeechStudioGraph(graphData);
    },
});
