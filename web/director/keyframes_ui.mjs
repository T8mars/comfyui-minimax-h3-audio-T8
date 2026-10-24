// Optional image/time editor. Existing project history owns every committed edit.
import {middleRows, resolveMiddlePosition} from './keyframes_state.mjs';

export function createKeyframeEditor({root,shot,assets,mutate,uuid,escape:esc,upload,notify}) {
    const openShots=new Set(),openAdvanced=new Set();
    const style=document.createElement('link');
    style.rel='stylesheet';style.href=new URL('./keyframes_ui.css',import.meta.url).href;
    document.head.append(style);
    const rowFor=(s,id)=>middleRows(s).find(row=>row.id===id);
    function plan(s){return s.middleKeyframes??={version:1,time_basis:'visible_delivery',items:[]};}
    function remember(){
        // Native toggle events can still be queued when a change rerenders.
        // Read live DOM before replacement, using its owner rather than current shot.
        for(const node of root.querySelectorAll('[data-middle-owner]')){
            const owner=node.dataset.middleOwner;
            if(node.open)openShots.add(owner);else openShots.delete(owner);
            for(const advanced of node.querySelectorAll('[data-middle-advanced]')){
                const key=owner+':'+advanced.dataset.middleAdvanced;
                if(advanced.open)openAdvanced.add(key);else openAdvanced.delete(key);
            }
        }
    }
    function hint(s,row){
        if(row.enabled===false)return '已停用，图片与位置草稿保留，不参与生成。';
        if(!row.asset_id||!assets().has(row.asset_id)||assets().get(row.asset_id).missing)return '请选择已上传的图片，或关闭这一项。';
        try{
            const clock=resolveMiddlePosition(s,row);
            const collision=middleRows(s,true).some(other=>{
                if(other.id===row.id)return false;
                try{return resolveMiddlePosition(s,other).frame===clock.frame;}catch{return false;}
            });
            return `帧 ${clock.frame} · ${(clock.seconds).toFixed(3)} 秒${collision?' · 与另一项同帧，请调整位置':''}${clock.visible_in_delivery?'':' · 在交付范围外，不会出现在最终裁片中'}`;
        }catch(error){return error.message+'；原请求保留，不自动截断或移动。';}
    }
    function render(host){
        const s=shot(),rows=middleRows(s),active=middleRows(s,true).length;
        const node=document.createElement('details');node.className='o-details k-editor';node.dataset.middleEditor='';node.dataset.middleOwner=s.id;
        node.open=openShots.has(s.id);
        node.innerHTML=`<summary>中间关键帧 <small>· ${active?active+' 项启用':'可选'}</small></summary>
            <p class="o-tip">指定“这张图片出现在何时”，不是逐张切镜。普通镜头使用首尾模式并提供首帧、尾帧；原音驱动可在首帧模式添加中间图，不需要尾帧。普通提示词不必开启 Relay。</p>
            <label>定位时轴<select data-middle-basis aria-label="中间关键帧定位时轴">
            <option value="visible_delivery" ${(s.middleKeyframes?.time_basis||'visible_delivery')==='visible_delivery'?'selected':''}>最终可见片段（推荐）</option>
            <option value="generated_canvas" ${s.middleKeyframes?.time_basis==='generated_canvas'?'selected':''}>完整生成帧轴（高级，含裁掉的尾部）</option></select></label>
            <p class="o-tip">24fps，帧号从 0 开始；改时长时，秒／帧位置保持原值，百分比重新计算。越界或同帧会提示，不悄悄修改。</p>
            <div data-middle-list>${rows.map((row,index)=>{
                const asset=assets().get(row.asset_id),images=[...assets().values()].filter(a=>a.kind==='image');
                const selectedMissing=row.asset_id&&!images.some(a=>a.id===row.asset_id);
                return `<article class="k-row" data-middle-row="${esc(row.id)}">
                    <div class="o-row o-between"><label class="o-check"><input type="checkbox" data-middle-key="enabled" ${row.enabled!==false?'checked':''} aria-label="启用中间关键帧 ${index+1}">第 ${index+1} 项</label><button type="button" data-keyframe-action="remove" aria-label="删除中间关键帧 ${index+1}">删除</button></div>
                    <div class="k-image">${asset?.kind==='image'?`<img src="${esc(asset.url)}" alt="中间关键帧 ${index+1} 原图预览">`:''}<label>关键帧图片<select data-middle-key="asset_id" aria-label="中间关键帧 ${index+1} 图片"><option value="">选择素材库图片</option>${selectedMissing?`<option value="${esc(row.asset_id)}" selected>图片缺失，请重连</option>`:''}${images.map(a=>`<option value="${esc(a.id)}" ${a.id===row.asset_id?'selected':''}>${esc(a.name)}${a.missing?'（需重连）':''}</option>`).join('')}</select></label><button type="button" data-keyframe-action="upload">上传图片</button></div>
                    <div class="o-two"><label>定位方式<select data-middle-key="position_mode" aria-label="中间关键帧 ${index+1} 定位方式">${[['percent','百分比 %'],['seconds','秒'],['frame','帧号']].map(([value,label])=>`<option value="${value}" ${row.position_mode===value?'selected':''}>${label}</option>`).join('')}</select></label><label>位置<input type="number" step="any" data-middle-key="position" value="${esc(row.position)}" aria-label="中间关键帧 ${index+1} 位置"></label></div>
                    <p class="o-tip" data-middle-hint role="status">${esc(hint(s,row))}</p>
                    <details data-middle-advanced="${esc(row.id)}" ${openAdvanced.has(s.id+':'+row.id)?'open':''}><summary>图像处理（高级）</summary><div class="o-two"><label>原始混噪值<input type="number" min="0" max="1" step="0.001" data-middle-key="visual_noise_aug" value="${esc(row.visual_noise_aug??.999)}" aria-label="中间关键帧 ${index+1} 原始混噪值"></label><label>适配画布<select data-middle-key="resize_mode" aria-label="中间关键帧 ${index+1} 缩放方式"><option value="center_crop" ${(row.resize_mode||'center_crop')==='center_crop'?'selected':''}>等比居中裁剪</option><option value="stretch" ${row.resize_mode==='stretch'?'selected':''}>拉伸到画布</option></select></label></div><p class="o-tip">默认 0.999，与原关键帧节点一致；这是模型原始混噪参数，不是“相似度”。图片预览不裁剪，实际送模按此项处理。</p></details>
                </article>`;
            }).join('')}</div><button type="button" data-keyframe-action="add">＋ 添加中间关键帧</button>
            <p class="o-tip">多图增加编码时间与显存。LOW/HIGH 分别按各自画布编码；完整模型效果请审片确认。</p>`;
        host.append(node);
    }
    root.addEventListener('toggle',event=>{
        const node=event.target,owner=node.closest('[data-middle-owner]')?.dataset.middleOwner;
        if(!owner)return;
        if(node.matches('[data-middle-editor]')){
            if(node.open)openShots.add(owner);else openShots.delete(owner);
        }else if(node.matches('[data-middle-advanced]')){
            const key=owner+':'+node.dataset.middleAdvanced;
            if(node.open)openAdvanced.add(key);else openAdvanced.delete(key);
        }
    },true);
    root.addEventListener('click',event=>{
        const button=event.target.closest('[data-keyframe-action]');if(!button)return;
        remember();
        const s=shot(),row=rowFor(s,button.closest('[data-middle-row]')?.dataset.middleRow);
        const action=button.dataset.keyframeAction;
        if(action==='upload'){if(row)upload(row.id,row.asset_id??null);return;}
        openShots.add(s.id);
        if(action==='remove'){if(row)mutate(next=>{plan(next).items=middleRows(next).filter(item=>item.id!==row.id);});return;}
        if(action==='add'){
            const id=uuid();
            mutate(next=>plan(next).items.push({id,asset_id:null,position_mode:'percent',position:50,
                visual_noise_aug:.999,resize_mode:'center_crop',enabled:true}));
            root.querySelector(`[data-middle-row="${id}"] select`)?.focus();
        }
    });
    root.addEventListener('change',event=>{
        const input=event.target;
        if(input.matches('[data-middle-basis],[data-middle-key]'))remember();
        if(input.matches('[data-middle-basis]')){openShots.add(shot().id);mutate(s=>{plan(s).time_basis=input.value;});return;}
        if(!input.matches('[data-middle-key]'))return;
        const id=input.closest('[data-middle-row]')?.dataset.middleRow,key=input.dataset.middleKey;
        if(!rowFor(shot(),id))return;
        const value=key==='enabled'?input.checked:['position','visual_noise_aug'].includes(key)?input.valueAsNumber:input.value||null;
        if(['position','visual_noise_aug'].includes(key)&&(!Number.isFinite(value)||(key==='visual_noise_aug'&&(value<0||value>1)))){
            input.setCustomValidity(key==='position'?'请输入有限的位置数字':'原始混噪值必须在 0 到 1 之间');
            input.reportValidity();notify('该输入尚未保存，请修正数字；原草稿保留。');return;
        }
        input.setCustomValidity('');openShots.add(shot().id);
        mutate(s=>{const row=rowFor(s,id);row[key]=value;
            if(key==='asset_id'&&value){s.tray=[...new Set([...(s.tray||[]),value])];s.selected=value;}});
    });
    root.addEventListener('input',event=>{if(event.target.matches('[data-middle-key]'))event.target.setCustomValidity('');});
    return {render,remember};
}
