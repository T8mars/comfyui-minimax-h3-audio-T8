import { app } from "../../scripts/app.js";

// Some frozen workflows wrote the note text as a scalar. Current MarkdownNote
// expects a one-element widget array and otherwise saves the default "#".
// Repair only the in-memory import; the source workflow stays byte-for-byte.
export function migrateLegacyMarkdownNotes(graphData) {
    if (!Array.isArray(graphData?.nodes)) return 0;
    let count = 0;
    graphData.nodes = graphData.nodes.map((node) => {
        if (node?.type !== "MarkdownNote" || node.widgets_values_named != null ||
            typeof node.widgets_values !== "string") return node;
        count += 1;
        return { ...node, widgets_values: [node.widgets_values] };
    });
    return count;
}

app.registerExtension({
    name: "minimax-h3-audio-t8.legacy-markdown-note-widgets",
    beforeConfigureGraph(graphData) {
        migrateLegacyMarkdownNotes(graphData);
    },
});
