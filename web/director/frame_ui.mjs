import { createModalAccess } from './modal_ui.mjs';

// Explicit frozen-version frame selection, never a CSS/video canvas screenshot.
export function createFramePicker(root,ctx){
    const dialog=document.createElement('dialog');dialog.className='w-frame-dialog';dialog.dataset.frameDialog='';
    dialog.innerHTML='<header class="o-row o-between"><h3>成片取帧 → 下一镜首帧</h3><button type="button" data-frame-close>关闭</button></header><p data-frame-status role="status"></p><div data-frame-body></div>';
    root.append(dialog);
    const modal=createModalAccess(dialog,{root});
    const status=dialog.querySelector('[data-frame-status]'),body=dialog.querySelector('[data-frame-body]');
    let sequence=0,origin=null,snapshot=null,film=null,source=null,picked=null;
    const live=ticket=>dialog.open&&ticket===sequence&&origin===ctx.context()&&snapshot===ctx.snapshot();
    function close(){sequence++;picked=null;dialog.close();body.replaceChildren();}
    dialog.querySelector('[data-frame-close]').onclick=close;dialog.addEventListener('cancel',close);
    function checkCommit(){const confirm=body.querySelector('[data-frame-replace]');body.querySelector('[data-frame-apply]').disabled=!picked||(confirm&&!confirm.checked);}
    function targetChanged(){
        const id=body.querySelector('[data-frame-target]').value,row=ctx.targets(source).find(row=>row.id===id);
        body.querySelector('[data-frame-warning]').innerHTML=row?.first?'<label><input type="checkbox" data-frame-replace>确认替换下一镜已有首帧（原素材和原成片保留）</label>':'';
        checkCommit();
    }
    async function open(key){
        if(root.querySelector('dialog[open]')){ctx.notify('请先关闭当前编辑窗口。');return;}
        const ticket=++sequence;origin=ctx.context();snapshot=ctx.snapshot();source=ctx.current();picked=null;film=null;
        body.replaceChildren();status.textContent='正在核对所选版本并固定媒体副本；不生成视频…';modal.show();
        try{
            const result=await ctx.prepare(key);
            if(!live(ticket)){if(dialog.open&&ticket===sequence)status.textContent='草稿已变化，请关闭后重新准备。';return;}
            if(!result)throw Error('草稿已变化，请关闭后重新选择。');film=result;
            const entry=film.entries[0],frames=entry.media.frames,targets=ctx.targets(source);
            status.textContent=`成片 ${frames} 帧 · ${entry.media.width}×${entry.media.height} · ${entry.media.fps} fps；默认末帧。`;
            body.innerHTML=`<p class="o-tip">按解码顺序选取完整原尺寸 PNG，不裁切、不放大。预览会创建独立输入文件；确认后才加入工程。只是静态接镜，不继承音频或长片 latent，也不保证人物连续。</p><label>帧号（0–${frames-1}）<input type="number" min="0" max="${frames-1}" step="1" value="${frames-1}" required data-frame-number></label><button type="button" data-frame-extract>提取并预览该帧</button><div data-frame-preview></div><label>应用到<select data-frame-target>${targets.map(row=>`<option value="${ctx.escape(row.id)}">${ctx.escape(row.name)}</option>`).join('')}<option value="new">在当前镜后新建一镜</option></select></label><div data-frame-warning></div><button type="button" data-frame-apply disabled>确认用作首帧（可撤销）</button>`;
            targetChanged();
        }catch(error){if(live(ticket))status.textContent=error.message;}
    }
    body.addEventListener('input',event=>{if(event.target.matches('[data-frame-number]')){picked=null;body.querySelector('[data-frame-preview]').replaceChildren();checkCommit();}});
    body.addEventListener('change',event=>{if(event.target.matches('[data-frame-target]'))targetChanged();if(event.target.matches('[data-frame-replace]'))checkCommit();});
    body.addEventListener('click',async event=>{
        const button=event.target.closest('button');if(!button||button.disabled)return;
        if(button.hasAttribute('data-frame-extract')){
            const input=body.querySelector('[data-frame-number]');if(!input.reportValidity())return;
            const number=input.valueAsNumber,ticket=sequence;picked=null;checkCommit();button.disabled=true;status.textContent='正在解码并提取完整帧…';
            try{
                const result=await ctx.extract(film,number);
                if(!live(ticket)||input.valueAsNumber!==number){if(dialog.open&&ticket===sequence)status.textContent='草稿或帧号已变化，请重新准备；未改动镜头。';return;}
                picked=result;
                body.querySelector('[data-frame-preview]').innerHTML=`<img src="${ctx.escape(result.asset.url)}" alt="第 ${number} 帧完整预览"><p>第 ${number} 帧 · ${result.seconds.toFixed(3)} 秒 · ${result.asset.width}×${result.asset.height}</p>`;
                status.textContent='仅已提取预览，尚未改动镜头。';checkCommit();
            }catch(error){if(live(ticket))status.textContent=error.message;}
            finally{button.disabled=false;}
        }
        if(button.hasAttribute('data-frame-apply')){
            if(!live(sequence)){status.textContent='工程已变化，请关闭后重新准备；未覆盖任何镜头。';return;}
            try{ctx.apply(source,body.querySelector('[data-frame-target]').value,picked.asset,!!body.querySelector('[data-frame-replace]:checked'));close();}
            catch(error){status.textContent=error.message;}
        }
    });
    new MutationObserver(()=>{if(dialog.open&&origin!==ctx.context())close();}).observe(root,{childList:true,subtree:true});
    return {open};
}
