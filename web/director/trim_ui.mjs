import { createModalAccess } from './modal_ui.mjs';

// This changes only the adopted playback/export range, never the source media.
export function directorTrimRange(start,end,frames,fps) {
    if(!Number.isSafeInteger(frames)||frames<1||!Number.isFinite(fps)||fps<=0)throw Error('无法确认成片的真实帧数或帧率');
    if(!Number.isSafeInteger(start)||!Number.isSafeInteger(end)||start<0||end<=start||end>frames)throw Error(`请输入整数帧：0 ≤ 入帧 < 出帧 ≤ ${frames}（出帧不包含）`);
    return {in_frame:start,out_frame:end};
}

export function createTrimEditor(root,ctx) {
    const dialog=document.createElement('dialog');dialog.className='w-frame-dialog';dialog.dataset.trimDialog='';
    dialog.innerHTML='<header class="o-row o-between"><h3>采用范围 · 不修改原片</h3><button type="button" data-trim-close>关闭</button></header><p data-trim-status role="status"></p><div data-trim-body class="o-gap"></div><div class="o-row o-gap"><button type="button" data-trim-reset>恢复全片（可撤销）</button><button type="button" data-trim-apply disabled>保存采用范围</button></div>';
    root.append(dialog);
    const modal=createModalAccess(dialog,{root}),status=dialog.querySelector('[data-trim-status]'),body=dialog.querySelector('[data-trim-body]'),apply=dialog.querySelector('[data-trim-apply]');
    let sequence=0,origin=null,snapshot=null,source=null,key=null,media=null;
    const live=ticket=>dialog.open&&ticket===sequence&&origin===ctx.context()&&snapshot===ctx.snapshot()&&source===ctx.shot().id&&key===ctx.shot().adoptedResultId;
    function close(){sequence++;dialog.close();body.replaceChildren();}
    dialog.querySelector('[data-trim-close]').onclick=close;
    dialog.addEventListener('cancel',()=>{sequence++;});
    function readRange(){return directorTrimRange(body.querySelector('[data-trim-in]').valueAsNumber,body.querySelector('[data-trim-out]').valueAsNumber,media.frames,media.fps);}
    function update(){
        if(!media)return;
        try{const range=readRange();body.querySelector('[data-trim-seconds]').textContent=`${(range.in_frame/media.fps).toFixed(4)}–${(range.out_frame/media.fps).toFixed(4)} 秒 · 共 ${range.out_frame-range.in_frame} 帧 / ${((range.out_frame-range.in_frame)/media.fps).toFixed(4)} 秒`;apply.disabled=false;}
        catch(error){body.querySelector('[data-trim-seconds]').textContent=error.message;apply.disabled=true;}
    }
    async function open(){
        if(root.querySelector('dialog[open]')){ctx.notify('请先关闭当前编辑窗口。');return;}
        const shot=ctx.shot();if(!shot.adoptedResultId&&!shot.filmTrim){ctx.notify('请先采用一个成功版本。');return;}
        const ticket=++sequence;origin=ctx.context();snapshot=ctx.snapshot();source=shot.id;key=shot.adoptedResultId;media=null;apply.disabled=true;body.replaceChildren();
        dialog.querySelector('[data-trim-reset]').disabled=!shot.filmTrim;
        status.textContent='正在核对采用版本的真实帧数与帧率；不生成视频…';modal.show();
        if(!key){status.textContent='采用版本缺失；可明确恢复全片以移除旧范围，未替换原片。';return;}
        try{
            const film=await ctx.prepare(key);
            if(!live(ticket)){if(dialog.open&&sequence===ticket)status.textContent='草稿或采用版本已变化，请关闭后重新准备。';return;}
            if(!film?.ready||film.entries?.length!==1)throw Error('无法确认唯一采用成片，请重新核对版本。');
            media=film.entries[0].media;directorTrimRange(0,media.frames,media.frames,media.fps);
            const range=shot.filmTrim||{in_frame:0,out_frame:media.frames};
            status.textContent=`采用成片：${media.frames} 帧 · ${media.fps} fps。只影响整片串播和导出，不改提示词、生成时长或原文件。`;
            body.innerHTML=`<div class="o-grid"><label>入帧（包含）<input type="number" min="0" max="${media.frames-1}" step="1" required data-trim-in></label><label>出帧（不包含）<input type="number" min="1" max="${media.frames}" step="1" required data-trim-out></label></div><p data-trim-seconds role="status"></p><p class="o-tip">帧号从 0 开始。例如入帧 2、出帧 10，采用第 2–9 帧。原片和所有生成版本均保留。</p>`;
            body.querySelector('[data-trim-in]').value=range.in_frame;body.querySelector('[data-trim-out]').value=range.out_frame;update();
        }catch(error){if(live(ticket))status.textContent=error.message+'；旧范围未修改，可恢复全片或关闭后重试。';}
    }
    body.addEventListener('input',update);
    dialog.addEventListener('click',event=>{
        const button=event.target.closest('button');if(!button||button.disabled||!button.matches('[data-trim-apply],[data-trim-reset]'))return;
        if(!live(sequence)){status.textContent='工程或采用版本已变化，未覆盖范围；请关闭后重新准备。';return;}
        try{const range=button.hasAttribute('data-trim-reset')?null:readRange();ctx.apply(source,key,range);close();}
        catch(error){status.textContent=error.message;}
    });
    new MutationObserver(()=>{if(dialog.open&&origin!==ctx.context())close();}).observe(root,{childList:true,subtree:true});
    return {open};
}
