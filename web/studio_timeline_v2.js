import { app } from "../../scripts/app.js";

const NODE_ID = "MiniMaxH3StudioTimelineT8Advanced";

function asText(value) {
    if (Array.isArray(value)) {
        return String(value[0] ?? "");
    }
    return String(value ?? "");
}

function parseTimeline(value) {
    try {
        const parsed = JSON.parse(asText(value));
        return parsed && Array.isArray(parsed.shots) ? parsed : null;
    } catch {
        return null;
    }
}

function makeTimelineView(node) {
    const root = document.createElement("div");
    root.style.cssText = [
        "box-sizing:border-box",
        "padding:8px",
        "width:100%",
        "min-height:130px",
        "max-height:280px",
        "overflow:auto",
        "font:12px/1.35 system-ui,sans-serif",
        "color:var(--fg-color,#ddd)",
        "background:rgba(0,0,0,.18)",
        "border:1px solid rgba(255,255,255,.12)",
        "border-radius:6px",
    ].join(";");

    const summary = document.createElement("div");
    summary.style.cssText = "margin-bottom:7px;font-weight:600";
    summary.textContent = "运行后显示镜头时间轴 / Timeline appears after execution";
    root.appendChild(summary);

    const track = document.createElement("div");
    track.style.cssText = "display:flex;gap:3px;min-height:30px;margin-bottom:8px";
    root.appendChild(track);

    const rows = document.createElement("div");
    root.appendChild(rows);

    node._t8TimelineRender = (payload) => {
        const timeline = parseTimeline(payload);
        if (!timeline) {
            summary.textContent = "时间轴数据不可读 / Timeline data unavailable";
            track.replaceChildren();
            rows.replaceChildren();
            return;
        }
        summary.textContent =
            `${timeline.project_id} · ${timeline.shot_count} 镜头 · ` +
            `${Number(timeline.total_duration_seconds).toFixed(3)}s · ` +
            `${timeline.total_frames} frames @ ${timeline.fps}fps`;
        track.replaceChildren();
        rows.replaceChildren();
        const total = Math.max(1e-9, Number(timeline.total_duration_seconds));
        for (const shot of timeline.shots) {
            const segment = document.createElement("div");
            segment.title =
                `#${shot.index} ${shot.id} | ${Number(shot.start_seconds).toFixed(3)}-` +
                `${Number(shot.end_seconds).toFixed(3)}s | seed ${shot.seed}`;
            segment.style.cssText = [
                `flex:${Math.max(0.01, Number(shot.end_seconds) - Number(shot.start_seconds)) / total}`,
                "min-width:8px",
                "height:30px",
                "border-radius:3px",
                "background:hsl(" + ((Number(shot.index) * 61) % 360) + " 55% 48%)",
                "opacity:.82",
            ].join(";");
            track.appendChild(segment);

            const row = document.createElement("div");
            row.style.cssText =
                "display:grid;grid-template-columns:34px minmax(70px,1fr) 90px 64px;" +
                "gap:6px;padding:3px 0;border-top:1px solid rgba(255,255,255,.08)";
            const prompt = String(shot.prompt ?? "").replace(/\s+/g, " ").trim();
            const indexCell = document.createElement("span");
            indexCell.textContent = `#${shot.index}`;
            const idCell = document.createElement("span");
            idCell.title = prompt;
            idCell.textContent = String(shot.id ?? "");
            const timeCell = document.createElement("span");
            timeCell.textContent =
                `${Number(shot.start_seconds).toFixed(2)}–` +
                `${Number(shot.end_seconds).toFixed(2)}s`;
            const frameCell = document.createElement("span");
            frameCell.textContent = `${shot.frame_count}F`;
            row.append(indexCell, idCell, timeCell, frameCell);
            rows.appendChild(row);
        }
        node.setDirtyCanvas?.(true, true);
    };

    return root;
}

app.registerExtension({
    name: "minimax-h3-audio-t8.studio-timeline-v2",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) {
            return;
        }
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const root = makeTimelineView(this);
            this.addDOMWidget("t8_timeline_preview", "timeline", root, {
                serialize: false,
                getMinHeight: () => 130,
                getMaxHeight: () => 280,
            });
            this.setSize?.([Math.max(this.size?.[0] ?? 0, 480), Math.max(this.size?.[1] ?? 0, 520)]);
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            this._t8TimelineRender?.(message?.t8_studio_timeline);
            return result;
        };
    },
});
