import {createModalAccess} from './modal_ui.mjs';

export function createMaskAuthor({root,api}) {
    const dialog=document.createElement('dialog');dialog.className='o-modal w-mask-author';dialog.dataset.maskAuthor='';
    dialog.style.cssText='width:min(1050px,94vw);max-height:90dvh;overflow:auto';
    dialog.innerHTML=`<div class="o-row o-between"><h3>逐帧绘制遮罩</h3><button type="button" data-mask-close>关闭</button></div>
      <p>只编辑已采用版本的遮罩。未画帧为空；复制仅作用于明确选择的帧段，不自动跟踪，也不修改原片或声音。</p>
      <div class="o-row" style="display:flex;align-items:center;gap:8px;flex-wrap:wrap"><button type="button" data-mask-prev>上一帧</button><input data-mask-frame type="range" aria-label="源片帧" style="width:auto;min-width:100px;flex:1"><button type="button" data-mask-next>下一帧</button><output data-mask-position></output></div>
      <div class="o-row"><button type="button" data-mask-tool="paint" aria-pressed="true">画笔</button><button type="button" data-mask-tool="erase" aria-pressed="false">擦除</button><label>画笔像素 <input data-mask-width type="number" min="1" value="16" step="1" style="width:6em"></label><button type="button" data-mask-undo>撤销</button><button type="button" data-mask-clear>本帧清空</button></div>
      <div data-mask-plane style="position:relative;margin:auto"><img data-mask-image alt="实际源帧" style="display:block;width:100%;height:100%"><canvas data-mask-canvas style="position:absolute;inset:0;width:100%;height:100%;touch-action:none;opacity:.45;cursor:crosshair"></canvas></div>
      <div class="o-row"><label>复制到源帧 <input data-mask-copy-start type="number" step="1" style="width:7em"></label><label>至（含） <input data-mask-copy-end type="number" step="1" style="width:7em"></label><button type="button" data-mask-copy>复制当前帧遮罩</button></div>
      <div class="o-row"><button type="button" data-mask-save>保存遮罩稿</button><button type="button" data-mask-export>保存并导出逐帧 MASK</button><a data-mask-download hidden download="frame-masks.zip">下载已保存的 MASK</a></div><p data-mask-status role="status"></p>`;
    root.append(dialog);const modal=createModalAccess(dialog,{root}),$=s=>dialog.querySelector(s),canvas=$('[data-mask-canvas]'),context=canvas.getContext('2d');
    let film=null,index=0,authorId=null,draft=null,saved='',frame=0,tool='paint',undo=[],drag=null,serial=0,ready=false,busy=false,previewInFlight=false,pendingPreview=null,previewTimer=null;
    const status=$('[data-mask-status]'),slider=$('[data-mask-frame]');
    const dirty=()=>draft&&JSON.stringify(draft)!==saved;
    function mayClose(){return !dialog.open||!dirty()||confirm('遮罩稿尚未保存。关闭后会丢失本次未保存绘制，仍要关闭？');}
    function close(){if(!mayClose())return false;serial++;ready=false;dialog.close();return true;}
    function changed(){status.textContent='遮罩稿已更改，尚未保存。';$('[data-mask-download]').hidden=true;}
    function snapshot(){undo.push(structuredClone(draft.strokes));}
    function draw(){
        context.clearRect(0,0,canvas.width,canvas.height);
        for(const stroke of draft?.strokes[String(frame)]||[]){
            context.globalCompositeOperation=stroke.tool==='erase'?'destination-out':'source-over';
            context.strokeStyle=context.fillStyle='#ff5045';context.lineWidth=stroke.width*Math.min(canvas.width,canvas.height);context.lineCap=context.lineJoin='round';
            context.beginPath();stroke.points.forEach(([x,y],i)=>i?context.lineTo(x*(canvas.width-1),y*(canvas.height-1)):context.moveTo(x*(canvas.width-1),y*(canvas.height-1)));context.stroke();
            for(const [x,y] of stroke.points){context.beginPath();context.arc(x*(canvas.width-1),y*(canvas.height-1),context.lineWidth/2,0,2*Math.PI);context.fill();}
        }
        context.globalCompositeOperation='source-over';
    }
    async function readLatest(){
        if(previewInFlight||busy||pendingPreview===null||!draft||!dialog.open)return;
        const selectedFrame=pendingPreview,selectedFilm=film,selectedIndex=index,source=draft.source,ticket=serial;
        pendingPreview=null;previewInFlight=true;
        try{const result=await api.frame(selectedFilm,selectedIndex,selectedFrame);if(ticket!==serial||!dialog.open)return;
            if(!result||result.frame!==frame||result.media_sha256!==source.media_sha256)throw Error('源片或项目已切换，未套用旧帧');
            const image=$('[data-mask-image]');image.onload=()=>{if(ticket===serial&&dialog.open){ready=true;image.style.opacity='1';canvas.setAttribute('aria-busy','false');draw();}};image.src=result.data_url;
        }catch(error){if(ticket===serial)status.textContent='源帧未读取：'+error.message;}
        finally{previewInFlight=false;if(pendingPreview!==null&&dialog.open)queueMicrotask(readLatest);}
    }
    async function select(next){
        if(!draft||busy)return;
        if(drag){draft.strokes=undo.pop();drag=null;}
        const source=draft.source;if(!Number.isInteger(next)||next<source.in_frame||next>=source.out_frame)return;
        frame=next;slider.value=String(frame);$('[data-mask-position]').textContent=`源片第 ${frame} 帧 · ${(frame/source.fps).toFixed(3)} 秒`;
        ready=false;canvas.setAttribute('aria-busy','true');$('[data-mask-image]').style.opacity='.35';draw();serial++;
        pendingPreview=frame;clearTimeout(previewTimer);previewTimer=setTimeout(readLatest,80);
    }
    function point(event){const b=canvas.getBoundingClientRect();return [Math.max(0,Math.min(1,(event.clientX-b.left)/b.width)),Math.max(0,Math.min(1,(event.clientY-b.top)/b.height))];}
    canvas.addEventListener('pointerdown',event=>{
        if(!ready||busy||event.button!==0)return;const width=Number($('[data-mask-width]').value);if(!Number.isFinite(width)||width<=0){status.textContent='画笔像素必须为正数。';return;}
        snapshot();const stroke={tool,width:width/Math.min(canvas.width,canvas.height),points:[point(event)]};
        (draft.strokes[String(frame)]??=[]).push(stroke);drag={pointer:event.pointerId,stroke};canvas.setPointerCapture(event.pointerId);draw();event.preventDefault();
    });
    canvas.addEventListener('pointermove',event=>{if(drag?.pointer!==event.pointerId)return;drag.stroke.points.push(point(event));draw();event.preventDefault();});
    canvas.addEventListener('pointerup',event=>{if(drag?.pointer!==event.pointerId)return;drag.stroke.points.push(point(event));drag=null;canvas.releasePointerCapture(event.pointerId);draw();changed();});
    canvas.addEventListener('pointercancel',()=>{if(!drag)return;draft.strokes=undo.pop();drag=null;draw();});
    dialog.addEventListener('close',()=>{serial++;ready=false;drag=null;pendingPreview=null;clearTimeout(previewTimer);});
    canvas.addEventListener('lostpointercapture',()=>{if(drag){draft.strokes=undo.pop();drag=null;draw();}});
    slider.addEventListener('input',()=>select(Number(slider.value)));
    dialog.addEventListener('cancel',event=>{event.preventDefault();close();});
    async function save(){
        const ticket=serial;const result=await api.save(film,index,authorId,draft);if(ticket!==serial||!dialog.open)return false;
        if(!result)throw Error('项目已切换；当前稿件保留');draft=result;saved=JSON.stringify(draft);status.textContent='遮罩稿已保存，可重新打开继续编辑。';return true;
    }
    dialog.addEventListener('click',async event=>{
        const button=event.target.closest('button');if(!button)return;
        if(button.hasAttribute('data-mask-close')){close();return;}
        if(busy)return;
        if(button.dataset.maskTool){tool=button.dataset.maskTool;dialog.querySelectorAll('[data-mask-tool]').forEach(x=>x.setAttribute('aria-pressed',String(x===button)));return;}
        if(!draft)return;
        if(button.hasAttribute('data-mask-prev')){select(frame-1);return;}
        if(button.hasAttribute('data-mask-next')){select(frame+1);return;}
        if(button.hasAttribute('data-mask-clear')){snapshot();draft.strokes[String(frame)]=[];draw();changed();return;}
        if(button.hasAttribute('data-mask-undo')){if(undo.length){draft.strokes=undo.pop();draw();changed();}return;}
        if(button.hasAttribute('data-mask-copy')){
            const a=Number($('[data-mask-copy-start]').value),b=Number($('[data-mask-copy-end]').value),source=draft.source;
            if(!Number.isInteger(a)||!Number.isInteger(b)||a>b||a<source.in_frame||b>=source.out_frame){status.textContent='复制范围须在本镜已采用的整数帧范围内。';return;}
            snapshot();const current=structuredClone(draft.strokes[String(frame)]||[]);for(let i=a;i<=b;i++)draft.strokes[String(i)]=structuredClone(current);draw();changed();return;
        }
        if(button.hasAttribute('data-mask-save')||button.hasAttribute('data-mask-export')){
            busy=true;slider.disabled=true;
            try{if(await save()&&button.hasAttribute('data-mask-export')){const ticket=serial,url=await api.export(film,index,authorId);if(ticket===serial&&dialog.open&&url){const a=$('[data-mask-download]');a.href=url;a.hidden=false;status.textContent='逐帧 MASK 已导出。未画帧为空，原片与原声保留。';}}}
            catch(error){status.textContent='未完成：'+error.message+'；当前绘制保留，没有自动重试。';}
            finally{busy=false;slider.disabled=false;if(pendingPreview!==null)readLatest();}
        }
    });
    return {mayClose,close,dispose(){serial++;dialog.remove();},async open(selected,selectedIndex){
        if(dialog.open&&!mayClose())return;film=selected;index=selectedIndex;ready=false;busy=true;draft=null;undo=[];drag=null;serial++;modal.show();status.textContent='读取该采用版本的遮罩稿…';
        const key='t8director.maskAuthor:'+film.id+':'+index;try{authorId=localStorage.getItem(key);}catch{/* local storage is optional */}
        if(!authorId){authorId=crypto.randomUUID();try{localStorage.setItem(key,authorId);}catch{/* save remains available */}}
        const ticket=serial;
        try{const result=await api.load(film,index,authorId);if(ticket!==serial||!dialog.open)return;if(!result)throw Error('项目已切换');
            draft=result;saved=JSON.stringify(draft);const s=draft.source;canvas.width=s.width;canvas.height=s.height;
            $('[data-mask-plane]').style.cssText+=`;aspect-ratio:${s.width}/${s.height};width:min(100%,${50*s.width/s.height}dvh)`;
            slider.min=String(s.in_frame);slider.max=String(s.out_frame-1);slider.disabled=false;$('[data-mask-copy-start]').value=String(s.in_frame);$('[data-mask-copy-end]').value=String(s.in_frame);
            busy=false;status.textContent='可绘制当前帧。遮挡帧请清空；复制后逐帧检查。';await select(s.in_frame);
        }catch(error){if(ticket===serial){status.textContent='无法打开遮罩稿：'+error.message;busy=false;}}
    }};
}
