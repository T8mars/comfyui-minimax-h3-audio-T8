import {CREATION_KINDS} from './creation_library.mjs';

export function createCreationLibrary(root,ctx){
    const dialog=document.createElement('dialog');dialog.className='w-creation-dialog';dialog.dataset.creationDialog='';dialog.setAttribute('aria-labelledby','director-creation-title');
    dialog.innerHTML='<header class="o-row o-between"><h3 id="director-creation-title">创作收藏 · 当前工程</h3><button type="button" data-creation-close>关闭</button></header><p class="o-tip">收藏随项目/JSON/工程包保存，不是上传到外部素材平台。应用前先看范围，确认后一次撤销；不自动生成。</p><div class="w-creation-controls"><label>类别<select data-creation-kind>'+Object.entries(CREATION_KINDS).map(([kind,name])=>`<option value="${kind}">${name}</option>`).join('')+'</select></label><label>名称<input data-creation-name maxlength="200" placeholder="给本次收藏起名"></label><button type="button" data-creation-save>收藏当前内容</button><button type="button" data-creation-copy hidden>直接复制当前参数</button></div><p data-creation-capture-tip class="o-tip"></p><label>查找收藏<input type="search" data-creation-search placeholder="按名称查找"></label><section data-creation-options hidden><p>复制哪些参数？默认只复制采样与 LoRA，不改底模。</p><div class="o-row"><label><input type="checkbox" data-creation-field="sampling" checked>采样与 LoRA</label><label><input type="checkbox" data-creation-field="d3">D3 增强</label><label><input type="checkbox" data-creation-field="duration">时长</label><label><input type="checkbox" data-creation-field="ratio">独立画幅</label></div><label>目标镜头<select data-creation-targets><option value="current">当前镜头</option><option value="selected">勾选的镜头</option><option value="all">全部镜头</option></select></label></section><p data-creation-status role="status"></p><div data-creation-list></div><section data-creation-preview hidden></section>';
    root.append(dialog);
    const $=selector=>dialog.querySelector(selector),kind=$('[data-creation-kind]'),name=$('[data-creation-name]'),status=$('[data-creation-status]'),preview=$('[data-creation-preview]');
    let origin=null,target=null,pending=null,ticket=null,returnFocus=null;
    function invalidate(){pending=null;ticket=null;preview.hidden=true;preview.replaceChildren();}
    function live(){if(origin!==ctx.context()){close();ctx.notify('工程已切换，旧收藏窗口已关闭。');return false;}return true;}
    function render(){
        const query=$('[data-creation-search]').value.trim().toLocaleLowerCase(),items=ctx.items().filter(row=>row.kind===kind.value&&row.name.toLocaleLowerCase().includes(query));
        $('[data-creation-list]').innerHTML=items.map(item=>`<article><strong>${ctx.escape(item.name)}</strong><div class="o-row"><button type="button" data-creation-use="${item.id}">预览使用</button><button type="button" data-creation-rename="${item.id}">重命名</button><button type="button" data-creation-delete="${item.id}">删除收藏</button></div></article>`).join('')||'<p class="o-tip">没有匹配收藏。可先在本镜填写内容，再收藏。</p>';
        $('[data-creation-copy]').hidden=kind.value!=='recipe';$('[data-creation-options]').hidden=kind.value!=='recipe';
        $('[data-creation-capture-tip]').textContent={snippet:'来源：'+ctx.targetLabel(target)+'。有选中文字就保存选区，否则保存整段；使用时插入原光标处，不覆盖原文。',template:'保存本镜写作草稿、时间、素材用途和生效采样/D3快照。使用时新建镜头，不覆盖旧镜头，不保存采用成片/种子/底模。',recipe:'保存本镜生效采样与多 LoRA、D3、画幅和时长。复制时逐项选择范围，不把底模误当本镜设置。',group:'保存本镜全部素材的稳定身份与顺序。使用时图片/视频加入本镜参考，音频只加入素材栏，不自动指定声音用途。'}[kind.value];
    }
    function open(){
        if(root.querySelector('dialog[open]')){ctx.notify('请先保存或关闭当前编辑窗口，再打开创作收藏。');return;}
        const opener=document.activeElement;
        // The menu folds after activation, so its hidden child cannot receive focus back.
        returnFocus=opener?.closest('.w-menu')?.querySelector(':scope > summary')||opener;
        origin=ctx.context();target=ctx.target();name.value='';$('[data-creation-search]').value='';invalidate();render();status.textContent='';dialog.showModal();name.focus();
    }
    function close(){invalidate();dialog.close();origin=null;}
    function showPlan(item){
        if(!live())return;invalidate();
        try{
            const fields=[...dialog.querySelectorAll('[data-creation-field]:checked')].map(el=>el.dataset.creationField);
            const plan=ctx.plan(item,target,fields,$('[data-creation-targets]').value);
            pending=plan;ticket=ctx.snapshot();preview.hidden=false;
            preview.innerHTML=`<h4>确认使用：${ctx.escape(item.name)}</h4><ul>${plan.notes.map(note=>`<li>${ctx.escape(note)}</li>`).join('')}</ul>${plan.missing.length?`<p>以下引用在目标位置没有相同素材，会保留为 missing 待绑定：${plan.missing.map(ctx.escape).join('、')}</p><label><input type="checkbox" data-creation-missing>确认保留待绑定占位，不误用同编号素材</label>`:''}<details><summary>查看本次变更详情</summary><pre>${ctx.escape(JSON.stringify(plan.detail,null,2))}</pre></details><button type="button" class="o-primary" data-creation-commit ${plan.missing.length?'disabled':''}>确认应用（可撤销）</button>`;
            status.textContent='只是预览；还没有修改镜头。';preview.scrollIntoView({block:'nearest'});
        }catch(error){status.textContent=error.message;}
    }
    $('[data-creation-close]').addEventListener('click',close);dialog.addEventListener('close',()=>{
        if(dialog.open)return;
        invalidate();origin=null;
        if(!root.querySelector('dialog[open]')&&returnFocus?.isConnected)returnFocus.focus({preventScroll:true});
        returnFocus=null;
    });
    kind.addEventListener('change',()=>{invalidate();render();});$('[data-creation-search]').addEventListener('input',render);
    $('[data-creation-options]').addEventListener('change',()=>{invalidate();status.textContent='参数或目标范围已变化，请重新预览。';});
    $('[data-creation-save]').addEventListener('click',()=>{if(!live())return;try{const item=ctx.capture(kind.value,name.value,target);ctx.save(item);invalidate();render();status.textContent='已收藏到本工程草稿，请保存项目；可撤销。';}catch(error){status.textContent=error.message;}});
    $('[data-creation-copy]').addEventListener('click',()=>{if(!live())return;try{showPlan(ctx.capture('recipe',name.value.trim()||'本镜参数',target));}catch(error){status.textContent=error.message;}});
    dialog.addEventListener('change',event=>{if(event.target.matches('[data-creation-missing]'))$('[data-creation-commit]').disabled=!event.target.checked;});
    dialog.addEventListener('click',event=>{
        const button=event.target.closest('button');if(!button||button.disabled)return;
        if(button.hasAttribute('data-creation-commit')){
            if(!live())return;
            if(!pending||ticket!==ctx.snapshot()){invalidate();status.textContent='工程已变化，请重新预览后应用。';return;}
            try{ctx.apply(pending);close();}catch(error){status.textContent=error.message;}return;
        }
        const id=button.dataset.creationUse||button.dataset.creationRename||button.dataset.creationDelete;if(!id||!live())return;
        const item=ctx.items().find(row=>row.id===id);if(!item){invalidate();render();return;}
        if(button.dataset.creationUse)showPlan(item);
        else if(button.dataset.creationRename){const next=prompt('新的收藏名称',item.name);if(next!==null)try{ctx.rename(id,next);invalidate();render();status.textContent='收藏已重命名，可撤销。';}catch(error){status.textContent=error.message;}}
        else if(confirm('只删除收藏“'+item.name+'”？已应用的镜头和原素材不会删除。')){ctx.remove(id);invalidate();render();status.textContent='已删除收藏，可撤销；没有删除原素材。';}
    });
    new MutationObserver(()=>{if(dialog.open&&origin!==ctx.context())close();}).observe(root,{childList:true,subtree:true});
    return {open};
}
