// Optional control geometry editor. Pointer previews never mutate project data.
import {newControlRow,normalizeControls,controlRows,controlSamplingMode,controlDraftIssues,containedRect,dragControlBox} from './control_state.mjs';

export function createControlEditor({root,shot,doc,assets,catalog,mutate,uuid,escape:esc,upload,notify,canvas=()=>null,refreshCanvas}){
    const openShots=new Set(),advanced=new Set(),tabs=new Map(),drawNext=new Set();let drag=null,canvasEpoch=0;
    const style=document.createElement('link');style.rel='stylesheet';style.href=new URL('./control_ui.css',import.meta.url).href;document.head.append(style);
    const rowFor=(s,id)=>controlRows(s).find(row=>row.id===id);
    const plan=s=>s.controlRegions??={version:1,items:[]};
    const normalized=row=>normalizeControls({version:1,items:[row]}).items[0];
    const resizeObserver=globalThis.ResizeObserver?new ResizeObserver(entries=>{
        for(const {target:plane}of entries){
            if(!plane.isConnected||plane.closest('[data-control-owner]')?.dataset.controlOwner!==shot().id)continue;
            if(drag?.plane===plane)cancelDrag(true);
            const row=rowFor(shot(),plane.closest('[data-control-row]').dataset.controlRow);if(row)paint(plane,normalized(row)[plane.dataset.controlPlane].xyxy_normalized);
        }
    }):null;
    function cancelDrag(refresh=false){if(!drag)return;const current=drag;drag=null;paint(current.plane,current.original);try{current.plane.releasePointerCapture(current.pointer);}catch{/* Detached/cancelled element. */}if(refresh)queueMicrotask(refreshGeometry);}
    function remember(){
        cancelDrag();canvasEpoch++;resizeObserver?.disconnect();
        for(const node of root.querySelectorAll('[data-control-owner]')){
            const owner=node.dataset.controlOwner;if(node.open)openShots.add(owner);else openShots.delete(owner);
            for(const item of node.querySelectorAll('[data-control-advanced]')){
                const key=owner+':'+item.dataset.controlAdvanced;if(item.open)advanced.add(key);else advanced.delete(key);
            }
        }
    }
    function commit(id,fn){remember();if(!rowFor(shot(),id))return;openShots.add(shot().id);mutate(s=>{const row=rowFor(s,id);Object.assign(row,normalized(row));fn(row,s);});}
    function targetRatio(s){
        const checked=canvas();if(checked?.width>0&&checked?.height>0)return checked.width/checked.height;
        const d=doc(),value=d.sharedRatio?d.ratio:s.ownRatio||s.ratio;
        const parts=String(value||'').split(':').map(Number);if(parts.length===2&&parts.every(x=>Number.isFinite(x)&&x>0))return parts[0]/parts[1];
        const media=[...(['first','ends'].includes(s.mode)?[s.first]:[]),...(s.mode==='ends'?[s.last]:[]),...(d.sharedRefs||[]),...(s.mode==='text'?[]:s.refs||[])].map(id=>assets().get(id)).find(a=>a?.kind==='image'&&a?.width>0&&a?.height>0);
        return media?media.width/media.height:16/9;
    }
    const input=(key,value,label,type='number',extra='')=>`<label>${label}<input type="${type}" ${type==='number'?'step="any"':''} data-control-key="${key}" value="${esc(value)}" aria-label="${label}" ${extra}></label>`;
    const select=(key,value,label,choices)=>`<label>${label}<select data-control-key="${key}" aria-label="${label}">${choices.map(([id,name])=>`<option value="${esc(id)}" ${id===value?'selected':''}>${esc(name)}</option>`).join('')}</select></label>`;
    const check=(key,value,label)=>`<label class="o-check"><input type="checkbox" data-control-key="${key}" ${value?'checked':''}>${label}</label>`;
    function refreshCatalog(){
        for(const list of root.querySelectorAll('[data-control-row] datalist'))list.innerHTML=(catalog()?.fun_control_names||[]).map(item=>`<option value="${esc(item.value??item)}">${esc(item.label??item.value??item)}</option>`).join('');
    }
    function geometry(s,row,asset){
        const tab=tabs.get(s.id+':'+row.id)||'region',kind=tab==='crop'?'crop':'region',box=row[kind].xyxy_normalized;
        const media=asset&&!asset.missing?(asset.kind==='image'?`<img src="${esc(asset.url)}" alt="控制源原图，不裁剪">`:`<video src="${esc(asset.url)}" preload="metadata" muted playsinline aria-label="控制源视频原始画面"></video>`):'<span class="c-placeholder">先选择已预处理的控制图片／视频</span>';
        return `<div class="c-geometry"><div class="o-row c-tabs"><button type="button" data-control-action="tab-region" aria-pressed="${kind==='region'}">目标区域</button><button type="button" data-control-action="tab-crop" aria-pressed="${kind==='crop'}">源图裁框</button><button type="button" data-control-action="redraw" aria-pressed="${drawNext.has(s.id+':'+row.id+':'+kind)}">重新画框</button><button type="button" data-control-action="reset-box">恢复整幅</button></div>
            <p class="o-tip">${kind==='region'?'成片画布坐标（0–1）；区域内施加控制，不是抠图，也不是严格禁止区域外变化。':'原始控制素材坐标；允许超出源图，外部按零填充。不会修改上传素材。'}</p>${kind==='region'?`<p class="o-tip" data-control-canvas-status>${canvas()?`${canvas().width} × ${canvas().height} · 服务端实际画布`:'比例示意 · 展开后读取服务端实际画布'}</p>`:''}
            <div class="c-plane ${kind==='crop'?'c-source':'c-target'}" data-control-plane="${kind}" data-control-active="${kind==='region'?row.region.mode==='rect':row.crop.enabled}" style="--c-ratio:${kind==='crop'&&asset?.width&&asset?.height?asset.width/asset.height:targetRatio(s)}" aria-label="${kind==='crop'?'源图裁框':'目标区域'}画布">
                ${kind==='crop'?media:'<span class="c-placeholder">成片画布 · 拖动划定控制区域</span>'}
                <div class="c-selection" data-control-selection><span>${kind==='crop'?'源裁框':'目标区域'}</span>${['nw','n','ne','e','se','s','sw','w'].map(handle=>`<i data-control-handle="${handle}" class="c-handle c-${handle}"></i>`).join('')}</div>
            </div><div class="c-coordinates">${['左 x₁','上 y₁','右 x₂','下 y₂'].map((label,i)=>input(kind+'.xyxy_normalized.'+i,box[i],label)).join('')}</div>
            <p class="o-tip" data-control-live>拖空白重新画框；拖框移动；拖边角缩放。拖动中仅预览，松开保存一次，可撤销。</p><p class="o-tip">整幅／关闭裁框时，数值仅存草稿；拖框会开启对应功能。源裁框超出原图的部分只在数值中保留，预览边界会裁掉辅助线。</p></div>`;
    }
    function render(host){
        const s=shot(),mode=controlSamplingMode(doc(),s),node=document.createElement('details');node.className='o-details c-editor';node.dataset.controlEditor='';node.dataset.controlOwner=s.id;node.open=openShots.has(s.id);
        let rows;try{rows=normalizeControls(s.controlRegions)?.items||[];}catch(error){node.innerHTML=`<summary>局部画面控制 · 版本待核对</summary><p role="alert">${esc(error.message)}</p>`;host.append(node);return;}
        const choices=[...assets().values()].filter(a=>['image','video'].includes(a.kind));
        node.innerHTML=`<summary>局部画面控制 <small>· ${controlRows(s,true,mode).length||'可选'}${controlRows(s,true,mode).length?' 项生效':''}</small></summary><p class="o-tip">使用<strong>已经预处理</strong>的控制图／视频（深度、姿态或线稿等），这里不会自动提取姿态／深度。无需开启 Prompt Relay，底模与加速配置沿用模型设置。</p>
            <div class="c-list">${rows.map((row,i)=>{
                const asset=assets().get(row.asset_id),missing=row.asset_id&&!choices.some(a=>a.id===row.asset_id),listId='c-model-'+row.id,active=controlRows({controlRegions:{version:1,items:[row]}},true,mode).length>0;
                return `<article class="c-row" data-control-row="${esc(row.id)}"><div class="o-row o-between">${check('enabled',row.enabled,`控制 ${i+1}`)}<span class="c-status">${active?'参与生成':'草稿保留 · 不生效'}</span><button type="button" data-control-action="up" aria-label="向前叠加控制 ${i+1}" ${i===0?'disabled':''}>↑</button><button type="button" data-control-action="down" aria-label="向后叠加控制 ${i+1}" ${i===rows.length-1?'disabled':''}>↓</button><button type="button" data-control-action="remove" aria-label="删除局部控制 ${i+1}">删除</button></div>
                    <div class="o-two"><label>已预处理控制素材<select data-control-key="asset_id" aria-label="控制 ${i+1} 素材"><option value="">选择图片／视频</option>${missing?`<option selected value="${esc(row.asset_id)}">原素材缺失，请重连</option>`:''}${choices.map(a=>`<option value="${esc(a.id)}" ${row.asset_id===a.id?'selected':''}>${esc(a.name)}${a.missing?'（需重连）':''}</option>`).join('')}</select></label><label>Fun Control 模型<input type="text" data-control-key="control_net_name" value="${esc(row.control_net_name)}" list="${listId}" placeholder="选择或填写模型文件名" aria-label="控制 ${i+1} 模型"><datalist id="${listId}">${(catalog()?.fun_control_names||[]).map(a=>`<option value="${esc(a.value??a)}">${esc(a.label??a.value??a)}</option>`).join('')}</datalist></label></div>
                    <div class="o-row c-upload"><button type="button" data-control-action="upload">上传控制素材</button><span class="o-tip">${asset?esc(asset.name)+' · '+esc(asset.width||'?')+' × '+esc(asset.height||'?'):'不作为 @image／@video 参考编号'}</span></div>
                    <div class="c-parameters">${input('strength',row.strength,'控制强度')}${input('start_percent',row.start_percent,'采样开始（0–1）')}${input('end_percent',row.end_percent,'采样结束（0–1）')}</div>
                    <details data-control-advanced="${esc(row.id)}" ${advanced.has(s.id+':'+row.id)?'open':''}><summary>区域、裁框与高级选项</summary><div class="o-two">${select('region.mode',row.region.mode,'控制范围',[['full','整幅画面'],['rect','指定矩形区域']])}${check('crop.enabled',row.crop.enabled,'启用源图裁框')}</div>${geometry(s,row,asset)}
                    <div class="o-two">${select('control_kind',row.control_kind,'素材类别',[['depth','深度'],['pose','姿态'],['canny','Canny 边缘'],['HED','HED 边缘'],['MLSD','MLSD 直线'],['custom','其他预处理控制']])}${select('fit_mode',row.fit_mode,'控制图适配目标',[['exact','精确尺寸'],['center_crop','等比居中裁剪'],['stretch','拉伸']])}
                    ${select('crop.fit',row.crop.fit,'源裁框适配',[['contain','完整容纳'],['cover','铺满裁剪'],['stretch','拉伸'],['exact','精确尺寸']])}<div class="o-row">${check('crop.flip_x',row.crop.flip_x,'源图左右翻转')}${check('crop.flip_y',row.crop.flip_y,'源图上下翻转')}</div>
                    ${input('source.start_seconds',row.source.start_seconds,'控制视频起点／秒')}${select('source.short_video_policy',row.source.short_video_policy,'控制视频不足时',[['hold_last_frame','保持最后一帧'],['strict','报错，不补帧']])}</div>
                    <div class="o-row c-stages">${check('stages.low',row.stages.low,'LOW 一采')}${check('stages.high',row.stages.high,'HIGH 二采')}<span class="o-tip">${mode==='single'?'当前为单采：使用 SINGLE，忽略 LOW/HIGH 开关。':mode==='high_only'?'当前从 TAIL 恢复：仅 HIGH 生效，LOW 草稿保留，不参与当前生成。':'两项均关闭时只保留草稿，不参与生成。'}</span></div></details></article>`;
            }).join('')}</div><button type="button" data-control-action="add">＋ 添加局部控制</button><div class="c-issues" role="status">${controlDraftIssues(doc(),s,assets()).map(message=>`<p class="o-tip">${esc(message)}</p>`).join('')}</div><p class="o-tip">多项按列表顺序叠加。未知模型文件名交给实际加载器核对，不自动替换。真实效果与双采组合仍需审片确认；图形预览不代表 GPU 验收。</p>`;
        host.append(node);for(const plane of node.querySelectorAll('[data-control-plane]')){
            const row=rows.find(r=>r.id===plane.closest('[data-control-row]').dataset.controlRow);paint(plane,row[plane.dataset.controlPlane].xyxy_normalized);
            resizeObserver?.observe(plane);
            plane.querySelector('video')?.addEventListener('loadedmetadata',event=>{const video=event.target;if(row.source.start_seconds>=0&&row.source.start_seconds<video.duration)video.currentTime=row.source.start_seconds;paint(plane,row[plane.dataset.controlPlane].xyxy_normalized);});
            plane.querySelector('img')?.addEventListener('load',()=>paint(plane,row.crop.xyxy_normalized));
        }
    }
    function contentRect(plane){
        const view=plane.getBoundingClientRect(),media=plane.querySelector('img,video');
        const width=media?.naturalWidth||media?.videoWidth,height=media?.naturalHeight||media?.videoHeight;
        return plane.dataset.controlPlane==='crop'&&width&&height?containedRect(view,width,height):view;
    }
    async function refreshGeometry(){
        if(!refreshCanvas||drag)return;
        const epoch=canvasEpoch,owner=shot().id;
        try{const actual=canvas()||await refreshCanvas();if(!actual||epoch!==canvasEpoch||owner!==shot().id||drag)return;
            for(const plane of root.querySelectorAll('[data-control-plane="region"]')){plane.style.setProperty('--c-ratio',actual.width/actual.height);const row=rowFor(shot(),plane.closest('[data-control-row]').dataset.controlRow);if(row)paint(plane,normalized(row).region.xyxy_normalized);}
            for(const label of root.querySelectorAll('[data-control-canvas-status]'))label.textContent=`${actual.width} × ${actual.height} · 服务端实际画布`;
        }catch(error){if(epoch===canvasEpoch&&owner===shot().id)for(const label of root.querySelectorAll('[data-control-canvas-status]'))label.textContent='比例示意，未确认实际画布：'+error.message;}
    }
    function paint(plane,box,preview=false){
        const selection=plane.querySelector('[data-control-selection]');if(!selection)return;
        const view=plane.getBoundingClientRect(),rect=contentRect(plane);
        Object.assign(selection.style,{display:preview||plane.dataset.controlActive==='true'?'block':'none',left:(rect.left-view.left+box[0]*rect.width)+'px',top:(rect.top-view.top+box[1]*rect.height)+'px',width:((box[2]-box[0])*rect.width)+'px',height:((box[3]-box[1])*rect.height)+'px'});
        const live=plane.parentElement.querySelector('[data-control-live]');if(live)live.textContent=box.map(x=>Number(x.toFixed(4))).join('，')+' · 拖动预览不改草稿，松开保存一次';
    }
    root.addEventListener('toggle',event=>{const node=event.target,owner=node.closest('[data-control-owner]')?.dataset.controlOwner;if(!owner)return;
        if(node.matches('[data-control-editor]')){if(node.open)openShots.add(owner);else openShots.delete(owner);}
        else if(node.matches('[data-control-advanced]')){const key=owner+':'+node.dataset.controlAdvanced;if(node.open)advanced.add(key);else advanced.delete(key);}
        if(node.open)for(const plane of node.querySelectorAll('[data-control-plane]')){const row=rowFor(shot(),plane.closest('[data-control-row]').dataset.controlRow);if(row)paint(plane,normalized(row)[plane.dataset.controlPlane].xyxy_normalized);}
        if(node.open&&node.matches('[data-control-advanced]'))refreshGeometry();
    },true);
    root.addEventListener('click',event=>{
        const button=event.target.closest('[data-control-action]');if(!button)return;
        const owner=button.closest('[data-control-owner]')?.dataset.controlOwner;if(owner!==shot().id)return;
        const id=button.closest('[data-control-row]')?.dataset.controlRow,row=rowFor(shot(),id),action=button.dataset.controlAction;
        if(action==='upload'){if(row)upload(id,row.asset_id??null);return;}
        if(action==='redraw'&&row){const kind=tabs.get(shot().id+':'+id)||'region',key=shot().id+':'+id+':'+kind;if(drawNext.has(key))drawNext.delete(key);else drawNext.add(key);button.setAttribute('aria-pressed',String(drawNext.has(key)));return;}
        remember();openShots.add(shot().id);
        if(action==='add'){const id=uuid();mutate(s=>plan(s).items.push(newControlRow(id)));root.querySelector(`[data-control-row="${id}"] select`)?.focus();}
        else if(action==='remove'&&row)mutate(s=>{plan(s).items=controlRows(s).filter(r=>r.id!==id);});
        else if(['up','down'].includes(action)&&row)mutate(s=>{const rows=plan(s).items,index=rows.findIndex(r=>r.id===id),target=index+(action==='up'?-1:1);if(target>=0&&target<rows.length)[rows[index],rows[target]]=[rows[target],rows[index]];});
        else if(action==='reset-box'&&row){const kind=tabs.get(shot().id+':'+id)||'region';commit(id,row=>{row[kind].xyxy_normalized=[0,0,1,1];if(kind==='region')row.region.mode='full';else row.crop.enabled=false;});}
        else if(action.startsWith('tab-')&&row){tabs.set(shot().id+':'+id,action.slice(4));const host=root.querySelector('[data-control-editor]').parentElement;root.querySelector('[data-control-editor]').remove();render(host);refreshGeometry();}
    });
    root.addEventListener('change',event=>{
        const element=event.target;if(!element.matches('[data-control-key]'))return;
        const owner=element.closest('[data-control-owner]')?.dataset.controlOwner;if(owner!==shot().id)return;
        const id=element.closest('[data-control-row]')?.dataset.controlRow,key=element.dataset.controlKey;
        const value=element.type==='checkbox'?element.checked:element.type==='number'?element.valueAsNumber:key==='asset_id'?element.value||null:element.value;
        if(element.type==='number'&&!Number.isFinite(value)){element.setCustomValidity('请输入有限数字，原草稿保留');element.reportValidity();notify('该输入尚未保存，请填写有限数字；原草稿保留。');return;}
        element.setCustomValidity('');commit(id,(row,s)=>{const path=key.split('.');let holder=row;while(path.length>1)holder=holder[path.shift()];holder[path[0]]=value;if(key==='asset_id'&&value)s.selected=value;});
    });
    root.addEventListener('input',event=>{if(event.target.matches('[data-control-key]'))event.target.setCustomValidity('');});
    root.addEventListener('pointerdown',event=>{
        const plane=event.target.closest('[data-control-plane]');if(drag||!plane||event.button!==0||plane.closest('[data-control-owner]').dataset.controlOwner!==shot().id)return;
        const row=rowFor(shot(),plane.closest('[data-control-row]').dataset.controlRow),kind=plane.dataset.controlPlane;if(!row)return;
        const rect=contentRect(plane);if(!rect.width||!rect.height)return;
        const start={x:(event.clientX-rect.left)/rect.width,y:(event.clientY-rect.top)/rect.height};
        if(start.x<0||start.x>1||start.y<0||start.y>1)return;
        const handle=event.target.closest('[data-control-handle]')?.dataset.controlHandle||'',action=drawNext.has(shot().id+':'+row.id+':'+kind)?'draw':handle?'resize':event.target.closest('[data-control-selection]')?'move':'draw';
        const original=[...normalized(row)[kind].xyxy_normalized];drag={plane,pointer:event.pointerId,owner:shot().id,id:row.id,kind,rect,start,original,next:original,action,handle};
        plane.setPointerCapture(event.pointerId);event.preventDefault();
    });
    root.addEventListener('pointermove',event=>{if(!drag||event.pointerId!==drag.pointer)return;const d=drag;
        d.next=dragControlBox(d.original,d.start,{x:(event.clientX-d.rect.left)/d.rect.width,y:(event.clientY-d.rect.top)/d.rect.height},d.action,d.handle,d.kind==='region');paint(d.plane,d.next,true);event.preventDefault();
    });
    root.addEventListener('pointerup',event=>{if(!drag||event.pointerId!==drag.pointer)return;const d=drag;drag=null;try{d.plane.releasePointerCapture(d.pointer);}catch{/* Capture already gone. */}
        if(d.owner!==shot().id||!rowFor(shot(),d.id)||d.next.some(x=>!Number.isFinite(x))||d.next[2]<=d.next[0]||d.next[3]<=d.next[1]||d.next.every((x,i)=>x===d.original[i])){paint(d.plane,d.original);refreshGeometry();return;}
        drawNext.delete(d.owner+':'+d.id+':'+d.kind);commit(d.id,row=>{row[d.kind].xyxy_normalized=d.next;if(d.kind==='region')row.region.mode='rect';else row.crop.enabled=true;});
    });
    root.addEventListener('pointercancel',event=>{if(drag?.pointer===event.pointerId)cancelDrag(true);});root.addEventListener('lostpointercapture',event=>{if(drag?.pointer===event.pointerId)cancelDrag(true);});
    return {render,remember,refreshCatalog};
}
