import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { COLORS, normalizedBox, rgbSHA, setDraft, validateSpec } from "./visual_marker_editor.mjs";

const ID="MiniMaxH3VisualMarkerPrepareEXPT8";
const widget=(node,name)=>node.widgets?.find(w=>w.name===name);
function el(tag,text,parent) {
    const value=document.createElement(tag);
    if (text!=null) value.textContent=text;
    parent?.appendChild(value);
    return value;
}

function upstreamSource(node) {
    const input=node.inputs?.find(x=>x.name==="image");
    const link=app.graph.links[input?.link];
    const origin=link && app.graph.getNodeById(link.origin_id);
    if (origin?.type!=="LoadImage") return null;
    const value=widget(origin,"image")?.value;
    if (typeof value!=="string" || !value) return null;
    const parts=value.replace(/\s*\[input\]$/," ").trim().replaceAll("\\","/").split("/");
    const filename=parts.pop();
    return api.apiURL("/view?"+new URLSearchParams({filename,subfolder:parts.join("/"),type:"input"}));
}

export function openMarkerEditor(node) {
    node._t8MarkerClose?.();
    const dialog=el("dialog",null,document.body);
    dialog.setAttribute("aria-label","H3 视觉标记框选编辑器");
    dialog.style.cssText="width:min(94vw,1050px);max-height:92vh;overflow:auto;background:#17212d;color:#eef3ff;padding:18px";
    el("h3","视觉标记：人物 → 目标",dialog);
    el("p","选一个标记后在图上拖框；描述、颜色和关系可在JSON中修改。同色及共享目标合法。预览仅示意，不是生成画面的硬坐标。",dialog);
    el("p","读取上游LoadImage或选择同一张RGB底图，不会替换IMAGE接线、运行采样或自动保存。应用时写入实际控件，并绑定预览RGB；运行时输入不同会明确报错。",dialog);
    const status=el("p","尚无预览；可只编辑JSON。",dialog);
    const controls=el("div",null,dialog);
    const load=el("button","读取上游LoadImage底图",controls);
    const file=el("input",null,controls); file.type="file"; file.accept="image/png,image/jpeg,image/webp";
    const choice=el("select",null,controls); choice.setAttribute("aria-label","正在编辑的标记");
    const canvas=el("canvas",null,dialog);
    canvas.style.cssText="display:block;max-width:100%;max-height:55vh;object-fit:contain;touch-action:none;margin:12px 0;border:1px solid #5e718a";
    canvas.width=640; canvas.height=360;
    const text=el("textarea",null,dialog); text.setAttribute("aria-label","标记JSON");
    text.style.cssText="display:block;width:98%;height:190px;font-family:monospace";
    text.value=widget(node,"markers_json")?.value ?? "";
    let image=null, sourceSHA=widget(node,"expected_source_sha256")?.value ?? "", start=null, ownedURL=null, spec=null, disposed=false, imageEpoch=0;
    const ctx=canvas.getContext("2d",{willReadFrequently:true});
    function refresh() {
        const selected=choice.value;
        spec=JSON.parse(text.value);
        choice.replaceChildren();
        for (const marker of spec.markers ?? []) {
            const option=el("option",`${marker.marker_id} · ${marker.kind}`,choice); option.value=marker.marker_id;
        }
        if ([...choice.options].some(x=>x.value===selected)) choice.value=selected;
        draw();
    }
    function draw() {
        ctx.clearRect(0,0,canvas.width,canvas.height);
        if (image) ctx.drawImage(image,0,0);
        for (const m of spec?.markers ?? []) {
            if (!Array.isArray(m.xyxy) || m.xyxy.length!==4) continue;
            const [x0,y0,x1,y1]=m.xyxy;
            ctx.strokeStyle=COLORS[m.color?.toLowerCase()] ?? m.color;
            ctx.lineWidth=m.marker_id===choice.value?4:2;
            ctx.strokeRect(x0*canvas.width,y0*canvas.height,(x1-x0)*canvas.width,(y1-y0)*canvas.height);
            ctx.fillStyle=ctx.strokeStyle; ctx.font="16px sans-serif";
            ctx.fillText(m.marker_id,x0*canvas.width+4,y0*canvas.height+18);
        }
    }
    async function loadImage(url) {
        const epoch=++imageEpoch;
        apply.disabled=true;
        status.textContent="正在读取底图并计算实际RGB绑定；完成前不能应用。";
        try {
            const next=new Image(); next.src=url; await next.decode();
            if (disposed || epoch!==imageEpoch) return;
            if (next.naturalWidth*next.naturalHeight>4*1024**2) throw Error("底图超过4MP准备预算，请显式缩图后使用。");
            image=next; canvas.width=next.naturalWidth; canvas.height=next.naturalHeight;
            ctx.drawImage(next,0,0);
            const currentSHA=await rgbSHA(ctx.getImageData(0,0,canvas.width,canvas.height));
            if (disposed || epoch!==imageEpoch) return;
            sourceSHA=currentSHA;
            status.textContent=`底图 ${canvas.width}×${canvas.height} · RGB ${sourceSHA.slice(0,16)}；SHA在节点执行时核对。`;
            refresh();
        } catch (error) { if (disposed || epoch!==imageEpoch) return; image=null; status.textContent=error.message; }
        finally {if (!disposed && epoch===imageEpoch) apply.disabled=false;}
    }
    load.onclick=()=>{const url=upstreamSource(node); if (url) void loadImage(url); else status.textContent="当前IMAGE不是直接LoadImage；请选与实际输入相同的底图。";};
    file.onchange=()=>{const selected=file.files?.[0]; if (!selected) return; if (ownedURL) URL.revokeObjectURL(ownedURL); ownedURL=URL.createObjectURL(selected); void loadImage(ownedURL);};
    choice.onchange=draw;
    text.oninput=()=>{try {refresh();} catch (error) {status.textContent=error.message;}};
    const point=event=>{const r=canvas.getBoundingClientRect(); return [(event.clientX-r.left)*canvas.width/r.width,(event.clientY-r.top)*canvas.height/r.height];};
    canvas.onpointerdown=event=>{if (!image || !choice.value) return; start=point(event); canvas.setPointerCapture(event.pointerId);};
    canvas.onpointercancel=()=>{start=null;};
    canvas.onpointerup=event=>{
        if (!start) return;
        try {
            const box=normalizedBox(start,point(event),canvas.width,canvas.height);
            spec.markers.find(x=>x.marker_id===choice.value).xyxy=box;
            text.value=JSON.stringify(spec,null,2); refresh();
        } catch (error) {status.textContent=error.message;} finally {start=null;}
    };
    const apply=el("button","应用到节点（不运行、不保存）",dialog);
    apply.onclick=()=>{
        try {const value=JSON.parse(text.value); validateSpec(value,widget(node,"mode")?.value); setDraft(node,value,sourceSHA); status.textContent="已更新真实工作流控件。请另行保存画布；未排队或采样。";}
        catch (error) {status.textContent=error.message;}
    };
    const close=el("button","关闭",dialog);
    const cleanup=()=>{disposed=true; if (ownedURL) URL.revokeObjectURL(ownedURL); dialog.remove(); if (node._t8MarkerClose===cleanup) delete node._t8MarkerClose;};
    node._t8MarkerClose=cleanup; close.onclick=cleanup;
    dialog.addEventListener("cancel",event=>{event.preventDefault(); cleanup();});
    try {refresh();} catch (error) {status.textContent=error.message;}
    dialog.showModal();
    return dialog;
}

app.registerExtension({
    name:"T8.VisualMarker.Editor",
    async beforeRegisterNodeDef(nodeType,nodeData) {
        if (nodeData.name!==ID) return;
        const created=nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated=function(...args) {
            const result=created?.apply(this,args);
            this.addWidget("button","框选编辑（不运行）",null,()=>openMarkerEditor(this),{serialize:false});
            return result;
        };
        const removed=nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved=function(...args) {this._t8MarkerClose?.(); return removed?.apply(this,args);};
    },
});
