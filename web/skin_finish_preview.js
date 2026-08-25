import { app } from "../../scripts/app.js";

const NODE_ID = "MiniMaxH3SkinFinishPreviewAuditT8Advanced";
const UI_SCHEMA = "h3_t8_skin_finish_preview_ui/v1";

function asText(value) {
    if (Array.isArray(value)) {
        return String(value[0] ?? "");
    }
    return String(value ?? "");
}

function parsePayload(value) {
    try {
        const parsed = JSON.parse(asText(value));
        return parsed?.schema === UI_SCHEMA ? parsed : null;
    } catch {
        return null;
    }
}

function clampPosition(value) {
    const number = Number(value);
    return Number.isFinite(number) ? Math.max(0, Math.min(1, number)) : 0.5;
}

function makeElement(tagName, cssText = "") {
    const element = document.createElement(tagName);
    element.style.cssText = cssText;
    return element;
}

function makeSkinFinishPreview(node) {
    const root = makeElement(
        "div",
        [
            "box-sizing:border-box",
            "padding:8px",
            "width:100%",
            "font:12px/1.35 system-ui,sans-serif",
            "color:var(--fg-color,#ddd)",
            "background:rgba(0,0,0,.18)",
            "border:1px solid rgba(255,255,255,.12)",
            "border-radius:6px",
        ].join(";"),
    );

    const summary = makeElement("div", "margin-bottom:7px;font-weight:600");
    summary.textContent = "运行后显示代理对比 / Proxy comparison appears after execution";
    root.appendChild(summary);

    const stage = makeElement(
        "div",
        [
            "position:relative",
            "width:100%",
            "aspect-ratio:16/9",
            "overflow:hidden",
            "background:#111",
            "border:1px solid rgba(255,255,255,.16)",
            "border-radius:4px",
        ].join(";"),
    );
    root.appendChild(stage);

    const sourceImage = makeElement(
        "img",
        "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;user-select:none",
    );
    sourceImage.alt = "Skin Finish source proxy";
    sourceImage.draggable = false;
    stage.appendChild(sourceImage);

    const candidateImage = makeElement(
        "img",
        "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;user-select:none",
    );
    candidateImage.alt = "Skin Finish candidate proxy";
    candidateImage.draggable = false;
    stage.appendChild(candidateImage);

    const sourceLabel = makeElement(
        "span",
        "position:absolute;left:6px;top:5px;padding:2px 5px;background:rgba(0,0,0,.68);border-radius:3px",
    );
    sourceLabel.textContent = "SOURCE / 原片";
    stage.appendChild(sourceLabel);

    const candidateLabel = makeElement(
        "span",
        "position:absolute;right:6px;top:5px;padding:2px 5px;background:rgba(0,0,0,.68);border-radius:3px",
    );
    candidateLabel.textContent = "CANDIDATE / 候选";
    stage.appendChild(candidateLabel);

    const divider = makeElement(
        "div",
        "position:absolute;top:0;bottom:0;width:2px;background:#fff;box-shadow:0 0 4px #000;pointer-events:none",
    );
    stage.appendChild(divider);

    const controls = makeElement(
        "div",
        "display:grid;grid-template-columns:minmax(0,1fr) auto;gap:7px;align-items:center;margin-top:8px",
    );
    root.appendChild(controls);

    const slider = makeElement("input", "width:100%;accent-color:#75d6ff");
    slider.type = "range";
    slider.min = "0";
    slider.max = "100";
    slider.step = "1";
    slider.value = "50";
    slider.title = "左侧原片，右侧候选；拖动只改变浏览器代理预览";
    controls.appendChild(slider);

    const applyButton = makeElement(
        "button",
        "padding:5px 8px;color:inherit;background:rgba(60,150,190,.28);border:1px solid rgba(117,214,255,.55);border-radius:4px;cursor:pointer",
    );
    applyButton.type = "button";
    applyButton.textContent = "写回位置 / Apply";
    controls.appendChild(applyButton);

    const positionText = makeElement("div", "margin-top:5px;color:#9edfff");
    root.appendChild(positionText);
    const safetyText = makeElement("div", "margin-top:4px;opacity:.82");
    safetyText.textContent =
        "代理图仅供定位；全分辨率输出用于判断。写回后需手动排队，不会接受候选。";
    root.appendChild(safetyText);

    let latestPayload = null;

    function renderPosition(position) {
        const safe = clampPosition(position);
        const percent = safe * 100;
        slider.value = String(Math.round(percent));
        candidateImage.style.clipPath = `inset(0 0 0 ${percent}%)`;
        divider.style.left = `calc(${percent}% - 1px)`;
        positionText.textContent =
            `原片 ${percent.toFixed(0)}% | 候选 ${(100 - percent).toFixed(0)}% | ` +
            `comparison_position=${safe.toFixed(2)}`;
    }

    slider.addEventListener("input", () => renderPosition(Number(slider.value) / 100));
    applyButton.addEventListener("click", () => {
        const widget = node.widgets?.find((item) => item.name === "comparison_position");
        if (!widget) {
            summary.textContent = "未找到 comparison_position；请刷新ComfyUI前端";
            return;
        }
        const value = clampPosition(Number(slider.value) / 100);
        widget.value = value;
        widget.callback?.(value);
        node.graph?.setDirtyCanvas?.(true, true);
        node.setDirtyCanvas?.(true, true);
        summary.textContent =
            `已写回 comparison_position=${value.toFixed(2)}；请手动排队以更新全分辨率输出`;
    });

    node._t8SkinFinishPreviewRender = (value) => {
        const payload = parsePayload(value);
        latestPayload = payload;
        if (!payload) {
            summary.textContent = "Skin Finish代理预览数据不可读";
            sourceImage.removeAttribute("src");
            candidateImage.removeAttribute("src");
            return;
        }
        renderPosition(payload.comparison_position);
        if (payload.status !== "READY") {
            summary.textContent =
                `帧 ${payload.frame_index + 1}/${payload.frame_count} · 代理图不可用，` +
                "请检查全分辨率输出";
            sourceImage.removeAttribute("src");
            candidateImage.removeAttribute("src");
            return;
        }
        stage.style.aspectRatio = `${payload.proxy_width} / ${payload.proxy_height}`;
        sourceImage.src = String(payload.source_data_url ?? "");
        candidateImage.src = String(payload.candidate_data_url ?? "");
        summary.textContent =
            `帧 ${payload.frame_index + 1}/${payload.frame_count} · ` +
            `${payload.proxy_width}×${payload.proxy_height}代理 · ` +
            `音频 ${payload.audio_status} · ${payload.review_status}`;
    };

    root.addEventListener("pointerdown", (event) => event.stopPropagation());
    root.addEventListener("wheel", (event) => event.stopPropagation());
    renderPosition(latestPayload?.comparison_position ?? 0.5);
    return root;
}

app.registerExtension({
    name: "minimax-h3-audio-t8.skin-finish-preview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) {
            return;
        }
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const root = makeSkinFinishPreview(this);
            this.addDOMWidget("t8_skin_finish_preview", "preview", root, {
                serialize: false,
                getMinHeight: () => 300,
                getMaxHeight: () => 520,
            });
            this.setSize?.([Math.max(this.size?.[0] ?? 0, 520), Math.max(this.size?.[1] ?? 0, 660)]);
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            this._t8SkinFinishPreviewRender?.(message?.t8_skin_finish_preview);
            return result;
        };
    },
});
