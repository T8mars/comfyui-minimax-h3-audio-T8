// Read-only task polling. No retry, queue or cancellation endpoint is called here.
export function createTaskDrawer(root, ctx) {
    const panel=document.createElement('dialog');panel.className='w-task-drawer';panel.dataset.taskDrawer='';
    panel.setAttribute('aria-labelledby','director-task-title');
    panel.innerHTML='<header><h3 id="director-task-title">任务列表</h3><button type="button" data-task-close aria-label="关闭任务列表">关闭</button></header><p class="o-tip">仅当前项目 · 每 5 秒刷新。关闭列表不停止生成；后台标签暂停查询。定位镜头不改变批次。</p><div class="w-task-controls"><button type="button" data-task-refresh>刷新状态</button><label><input type="checkbox" data-task-auto checked>自动刷新</label></div><p data-task-state role="status"></p><div data-task-content></div>';
    root.append(panel);
    const $=selector=>panel.querySelector(selector),body=$('[data-task-content]'),state=$('[data-task-state]');
    let origin=null,timer=null,controller=null,epoch=0,returnFocus=null,rendered='';
    function stop(){clearTimeout(timer);timer=null;controller?.abort();controller=null;epoch++;$('[data-task-refresh]').disabled=false;}
    function close(){stop();origin=null;if(panel.open)panel.close();}
    function schedule(){clearTimeout(timer);if(panel.open&&$('[data-task-auto]').checked&&!document.hidden)timer=setTimeout(refresh,5000);}
    function locate(id){
        const shot=ctx.shots().find(row=>row.id===id);
        if(!shot)return '原镜头已不在当前草稿';
        return `<button type="button" data-task-locate="${ctx.escape(id)}">第 ${ctx.shots().indexOf(shot)+1} 镜 · ${ctx.escape(shot.name||'未命名')}</button>`;
    }
    function paint(data){
        const batch=data.batch;
        const frozen=batch?`<section><h4>冻结批次 · ${batch.items.filter(row=>row.state==='success').length}/${batch.items.length}</h4><p class="o-tip">提交时顺序与种子固定。继续或暂停请使用底栏；本列表不会提交任务。</p><ol>${batch.items.map((row,i)=>`<li>批次 ${i+1} · ${locate(row.shot_id)} · ${ctx.escape(ctx.label(row.state))}</li>`).join('')}</ol></section>`:'';
        const content=frozen+(data.batchError?`<p class="o-warning">旧批次暂不可读：${ctx.escape(data.batchError)}。未重发。</p>`:'')+`<ul class="w-task-list">${data.records.map(row=>`<li>${locate(row.shot_id)}<span>${ctx.escape(ctx.label(row.state))}</span><small>${row.submitted_at?ctx.escape(new Date(row.submitted_at*1000).toLocaleString()):'时间未记录'}</small></li>`).join('')||'<li>尚无生成记录。</li>'}</ul>`;
        if(rendered===content)return;
        const focus=document.activeElement?.dataset.taskLocate,scroll=body.scrollTop;
        body.innerHTML=content;rendered=content;body.scrollTop=scroll;
        if(focus)[...body.querySelectorAll('[data-task-locate]')].find(button=>button.dataset.taskLocate===focus)?.focus({preventScroll:true});
    }
    async function refresh(){
        if(!panel.open||document.hidden)return;
        if(origin!==ctx.context()){close();return;}
        stop();const sequence=epoch;controller=new AbortController();
        $('[data-task-refresh]').disabled=true;state.textContent='正在读取状态…';
        try{
            const data=await ctx.load(controller.signal);
            if(sequence!==epoch||!panel.open)return;
            if(!data||origin!==ctx.context()){close();return;}
            paint(data);state.textContent='已更新 '+new Date().toLocaleTimeString()+' · 只查询，不重复提交';
        }catch(error){if(sequence===epoch&&error.name!=='AbortError')state.textContent='状态读取失败，下面保留的是上次结果：'+error.message+'。不会自动重试生成。';}
        finally{if(sequence===epoch){controller=null;$('[data-task-refresh]').disabled=false;schedule();}}
    }
    function open(){
        if(panel.open){refresh();return;}
        if(root.querySelector('dialog[open]')){ctx.notify('请先关闭当前编辑窗口，再打开任务列表。');return;}
        origin=ctx.context();rendered='';body.replaceChildren();returnFocus=document.activeElement;
        panel.show();$('[data-task-close]').focus();refresh();
    }
    $('[data-task-close]').addEventListener('click',close);
    $('[data-task-refresh]').addEventListener('click',refresh);
    $('[data-task-auto]').addEventListener('change',()=>{$('[data-task-auto]').checked?refresh():clearTimeout(timer);});
    panel.addEventListener('click',event=>{const button=event.target.closest('[data-task-locate]');if(!button)return;if(origin!==ctx.context()){close();return;}const id=button.dataset.taskLocate;close();ctx.locate(id);});
    panel.addEventListener('keydown',event=>{if(event.key==='Escape'&&!event.isComposing){event.preventDefault();event.stopImmediatePropagation();close();}});
    panel.addEventListener('close',()=>{stop();origin=null;if(returnFocus?.isConnected&&(document.activeElement===document.body||panel.contains(document.activeElement)))returnFocus.focus({preventScroll:true});});
    document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();else if(panel.open&&$('[data-task-auto]').checked)refresh();});
    window.addEventListener('pagehide',close);
    new MutationObserver(()=>{if(panel.open&&origin!==ctx.context())close();}).observe(root,{childList:true,subtree:true});
    return {open,close};
}
