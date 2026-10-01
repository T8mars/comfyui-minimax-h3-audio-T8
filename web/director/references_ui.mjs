// UI suggestions only. Asset identity, role changes and undo belong to the editor.
export function referenceQuery(value, start, end=start) {
    if(typeof value!=='string'||!Number.isInteger(start)||start<0||start>value.length||start!==end)return null;
    const match=value.slice(0,start).match(/@([\p{L}\p{N}_./-]{0,80})$/u);
    if(!match||(match[1]&&/^[\p{L}\p{N}_]/u.test(value.slice(start))))return null;
    return {start:start-match[0].length,end:start,query:match[1],value};
}
export function filterReferenceChoices(choices, query) {
    const term=query.toLocaleLowerCase();
    return choices.filter(item=>item.token.slice(1).toLocaleLowerCase().startsWith(term)||item.name.toLocaleLowerCase().includes(term));
}
export function createReferenceCandidates(root, ctx) {
    const selector='[data-global],[data-field="simplePrompt"],[data-field="prompt"],[data-key="text"],[data-editor]';
    const popup=document.createElement('section');popup.className='w-reference-popup';popup.hidden=true;
    const title=document.createElement('p');title.textContent='插入素材引用 · 不改变本镜素材用途';
    const list=document.createElement('div');list.id='director-reference-options';list.setAttribute('role','listbox');list.setAttribute('aria-label','素材引用候选');
    const hint=document.createElement('small');hint.setAttribute('role','status');
    popup.append(title,list,hint);root.append(popup);
    let active=null,choices=[],selected=0,composing=false;
    function close(){
        if(active){active.field.setAttribute('aria-expanded','false');active.field.removeAttribute('aria-activedescendant');active.field.removeAttribute('aria-controls');}
        active=null;choices=[];popup.hidden=true;
    }
    function highlight(){
        for(const [index,node] of [...list.children].entries())node.setAttribute('aria-selected',String(index===selected));
        const option=list.children[selected];
        if(option){
            active.field.setAttribute('aria-activedescendant',option.id);
            const box=list.getBoundingClientRect(),row=option.getBoundingClientRect();
            if(row.top<box.top)list.scrollTop-=box.top-row.top;
            else if(row.bottom>box.bottom)list.scrollTop+=row.bottom-box.bottom;
        }
        else active.field.removeAttribute('aria-activedescendant');
    }
    function position(){
        if(!active)return;
        const rect=active.field.getBoundingClientRect(),width=Math.min(440,window.innerWidth-24);
        popup.style.width=width+'px';popup.style.left=Math.max(12,Math.min(rect.left,window.innerWidth-width-12))+'px';
        popup.style.top=Math.max(12,Math.min(rect.bottom+4,window.innerHeight-280))+'px';
    }
    function update(field){
        if(composing||!field?.matches?.(selector)){close();return;}
        const range=referenceQuery(field.value,field.selectionStart,field.selectionEnd),context=ctx.context(field);
        if(!range||!context){close();return;}
        close();active={...range,field,context};selected=0;
        const matched=filterReferenceChoices(ctx.choices(field),range.query);choices=matched.slice(0,12);
        list.replaceChildren();
        choices.forEach((item,index)=>{
            const button=document.createElement('button');button.type='button';button.tabIndex=-1;button.dataset.referenceChoice=String(index);button.id=`director-reference-option-${index}`;button.setAttribute('role','option');
            if(item.kind==='image'&&item.url){const img=document.createElement('img');img.src=item.url;img.alt='';button.append(img);}
            const text=document.createElement('span'),name=document.createElement('strong'),role=document.createElement('small');
            name.textContent=item.token+' · '+item.name;role.textContent=item.hint||item.role;
            text.append(name,role);button.append(text);list.append(button);
        });
        hint.textContent=choices.length?`${matched.length} 项${matched.length>12?' · 仅显示前 12 项，可继续输入名称缩小范围':''} · ↑↓选择，Enter 插入，Esc 关闭`:'没有匹配项；可先从素材库加入本镜。Esc 关闭。';
        (field.closest('dialog')||root).append(popup);
        position();
        popup.hidden=false;field.setAttribute('aria-autocomplete','list');field.setAttribute('aria-expanded','true');field.setAttribute('aria-controls',list.id);highlight();
    }
    function choose(index){
        const ticket=active,item=choices[index];
        if(!ticket||!item)return;
        if(!ticket.field.isConnected||ticket.field.value!==ticket.value||ctx.context(ticket.field)!==ticket.context){close();return;}
        close();ctx.insert(item.id,ticket);
    }
    popup.addEventListener('pointerdown',event=>event.preventDefault());
    popup.addEventListener('click',event=>{const button=event.target.closest('[data-reference-choice]');if(button)choose(Number(button.dataset.referenceChoice));});
    root.addEventListener('compositionstart',event=>{if(event.target.matches(selector)){composing=true;close();}});
    root.addEventListener('compositionend',event=>{if(event.target.matches(selector)){composing=false;update(event.target);}});
    root.addEventListener('focusout',event=>{if(event.target.matches(selector))composing=false;});
    root.addEventListener('input',event=>{if(event.target.matches(selector)&&!event.isComposing)update(event.target);});
    root.addEventListener('click',event=>{if(event.target.matches(selector))update(event.target);else if(!popup.contains(event.target))close();});
    root.addEventListener('focusin',event=>{if(active&&event.target!==active.field&&!popup.contains(event.target))close();});
    root.addEventListener('keyup',event=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)&&event.target.matches(selector))update(event.target);});
    root.addEventListener('keydown',event=>{
        if(composing||event.isComposing||event.keyCode===229||!active||event.target!==active.field)return;
        if(event.key==='Tab'){close();return;}
        if(event.key==='Escape'){event.preventDefault();event.stopImmediatePropagation();close();return;}
        if(!choices.length)return;
        if(event.key==='Enter'){event.preventDefault();event.stopImmediatePropagation();choose(selected);}
        else if(['ArrowUp','ArrowDown'].includes(event.key)){event.preventDefault();event.stopImmediatePropagation();selected=(selected+(event.key==='ArrowUp'?-1:1)+choices.length)%choices.length;highlight();}
    },true);
    root.addEventListener('scroll',event=>{if(active&&!popup.contains(event.target)){if(document.activeElement===active.field)position();else close();}},true);
    window.addEventListener('resize',close);
    new MutationObserver(()=>{if(active&&(!active.field.isConnected||ctx.context(active.field)!==active.context))close();}).observe(root,{childList:true,subtree:true});
    return {close};
}
