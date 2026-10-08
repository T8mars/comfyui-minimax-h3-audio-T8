// Pure authoring helpers. No graph execution, network, cache or identity inference.
export const COLORS = {red:"#ff3030", blue:"#3070ff", yellow:"#ffdf00", cyan:"#00dfff",
    green:"#20df50", magenta:"#ef30ef", orange:"#ff9000", white:"#ffffff", black:"#000000"};

export function validateSpec(value, mode="render_rectangles") {
    if (!value || typeof value !== "object" || Array.isArray(value) ||
        Object.keys(value).sort().join(",") !== "markers,relations" ||
        !Array.isArray(value.markers) || value.markers.length<2 || value.markers.length>64 ||
        !Array.isArray(value.relations) || value.relations.length<1 || value.relations.length>128)
        throw Error("需要2–64个明确标记及1–128条人物→目标关系。");
    if (!["render_rectangles","provided_marked"].includes(mode)) throw Error("未知准备模式。");
    const ids=new Map(), id=/^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$/;
    const bounded=s=>typeof s==="string" && s.trim() && new TextEncoder().encode(s).length<=4096;
    for (const m of value.markers) {
        if (!m || typeof m!=="object" || typeof m.marker_id!=="string" || !id.test(m.marker_id) || ids.has(m.marker_id) ||
            !["actor","target"].includes(m.kind) || !bounded(m.description) ||
            Object.keys(m).some(k=>!["marker_id","kind","color","description","role_id","xyxy"].includes(k)) ||
            (m.kind==="actor" && (typeof m.role_id!=="string" || !id.test(m.role_id))) ||
            (m.kind==="target" && Object.hasOwn(m,"role_id"))) throw Error("标记ID／角色／描述有误。");
        if (typeof m.color!=="string" || m.color.length>32) throw Error("颜色必须为名称或#RRGGBB字符串。");
        const rgb=COLORS[m.color.toLowerCase()] ?? m.color.toLowerCase();
        if (!/^#[0-9a-f]{6}$/.test(rgb)) throw Error("颜色使用名称或#RRGGBB。");
        if (m.xyxy!=null) {
            const p=m.xyxy;
            if (!Array.isArray(p) || p.length!==4 || p.some(n=>typeof n!=="number" || !Number.isFinite(n)) ||
                !(0<=p[0] && p[0]<p[2] && p[2]<=1 && 0<=p[1] && p[1]<p[3] && p[3]<=1))
                throw Error("矩形需要有限、非零且0–1范围的xyxy。");
        } else if (mode==="render_rectangles") throw Error("自动画框必须有坐标，手绘模式可无坐标。");
        ids.set(m.marker_id,m);
    }
    const used=new Set();
    for (const r of value.relations) {
        if (!r || typeof r!=="object" || Object.keys(r).sort().join(",")!=="action,actor_marker_id,target_marker_id" ||
            ids.get(r.actor_marker_id)?.kind!=="actor" || ids.get(r.target_marker_id)?.kind!=="target" || !bounded(r.action))
            throw Error("关系必须明确指向人物框和目标框，并写动作。");
        used.add(r.actor_marker_id); used.add(r.target_marker_id);
    }
    if (used.size!==ids.size) throw Error("每个框都要有对应关系；同色和共享目标允许。");
    if (new TextEncoder().encode(JSON.stringify(value)).length>65536) throw Error("标记JSON超过64KiB。");
    return value;
}

export function normalizedBox(start, end, width, height) {
    if (!Array.isArray(start) || !Array.isArray(end) || start.length!==2 || end.length!==2) throw Error("拖动需要源平面上的两个点。");
    if (![...start,...end,width,height].every(Number.isFinite) || width<=0 || height<=0) throw Error("源尺寸／拖动坐标无效。");
    const clamp=n=>Math.min(1,Math.max(0,n));
    const p=[clamp(Math.min(start[0],end[0])/width),clamp(Math.min(start[1],end[1])/height),
        clamp(Math.max(start[0],end[0])/width),clamp(Math.max(start[1],end[1])/height)];
    if (p[0]>=p[2] || p[1]>=p[3]) throw Error("不能画零面积框。");
    return p;
}

export function setDraft(node, value, sourceSHA="") {
    const get=name=>node.widgets?.find(w=>w.name===name);
    const target=get("markers_json"), mode=get("mode"), binding=get("expected_source_sha256");
    if (!target || !mode || !binding) throw Error("找不到实际工作流字段。");
    validateSpec(value,mode.value);
    if (typeof sourceSHA!=="string" || (sourceSHA && !/^[0-9a-f]{64}$/.test(sourceSHA))) throw Error("预览源SHA无效。");
    // Actual serialized widgets, not an unsaved hidden editor plan.
    target.value=JSON.stringify(value);
    binding.value=sourceSHA;
    node.setDirtyCanvas?.(true,true);
    return target.value;
}

export async function rgbSHA(imageData) {
    const rgb=new Float32Array(imageData.width*imageData.height*3);
    for (let i=0,j=0;i<imageData.data.length;i+=4) {
        rgb[j++]=imageData.data[i]/255; rgb[j++]=imageData.data[i+1]/255; rgb[j++]=imageData.data[i+2]/255;
    }
    const bytes=new Uint8Array(await crypto.subtle.digest("SHA-256",rgb));
    return [...bytes].map(b=>b.toString(16).padStart(2,"0")).join("");
}
