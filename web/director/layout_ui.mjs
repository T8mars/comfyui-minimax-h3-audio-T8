// Device-local preferences only. Layout never mutates a project or a frozen batch.
export function normalizedLayout(value = {}) {
    const amount=Number(value?.split);
    return {mode:['parallel','writing','review'].includes(value?.mode)?value.mode:'parallel',split:Number.isFinite(amount)?Math.min(65,Math.max(40,amount)):53};
}
export function createDirectorLayout(root, controls) {
    const key='t8director.layout.v1';
    let state=normalizedLayout(),effectiveMode=null;
    try{state=normalizedLayout(JSON.parse(localStorage.getItem(key)||'{}'));}catch{/* Storage is optional. */}
    controls.innerHTML='<p class="o-tip">布局仅保存在当前浏览器，不改变项目。</p><div class="o-row"><button data-focus-layout="parallel">并排</button><button data-focus-layout="writing">专注写作</button><button data-focus-layout="review">专注审片</button></div><label>剧本列宽 <input type="range" min="40" max="65" step="1" data-column-width aria-label="剧本列宽百分比"></label><button data-reset-layout>恢复默认布局</button>';
    const workspace=root.querySelector('.o-workspace'),editor=root.querySelector('.w-editor'),stage=root.querySelector('.o-stage');
    const separator=document.createElement('div');separator.className='w-resizer';separator.tabIndex=0;
    separator.setAttribute('role','separator');separator.setAttribute('aria-label','调整剧本与预览列宽');separator.setAttribute('aria-orientation','vertical');separator.setAttribute('aria-valuemin','40');separator.setAttribute('aria-valuemax','65');
    workspace.append(separator);
    const range=controls.querySelector('[data-column-width]');
    function place(){
        const bounds=workspace.getBoundingClientRect(),edge=editor.getBoundingClientRect();
        separator.style.left=(edge.right-bounds.left-4)+'px';
    }
    function apply(save=true){
        state=normalizedLayout(state);
        const wide=window.innerWidth>800,nextMode=wide?state.mode:'parallel';
        const enteringWriting=nextMode==='writing'&&effectiveMode!=='writing';
        effectiveMode=nextMode;root.dataset.focusLayout=effectiveMode;
        root.style.setProperty('--w-write-fr',state.split+'fr');root.style.setProperty('--w-preview-fr',(100-state.split)+'fr');
        separator.hidden=root.dataset.focusLayout!=='parallel'||window.innerWidth<=800;
        separator.setAttribute('aria-valuenow',String(state.split));separator.setAttribute('aria-valuetext',state.split+'% 剧本宽度');
        range.value=state.split;range.disabled=!wide||effectiveMode!=='parallel';
        controls.querySelectorAll('[data-focus-layout]').forEach(button=>{
            button.setAttribute('aria-pressed',String(button.dataset.focusLayout===effectiveMode));
            button.disabled=!wide&&button.dataset.focusLayout!=='parallel';
            button.title=!wide?'窄屏自动上下显示；宽屏会恢复已保存的布局。':'';
        });
        if(enteringWriting)stage.querySelectorAll('video,audio').forEach(media=>media.pause());
        place();
        if(save)try{localStorage.setItem(key,JSON.stringify(state));}catch{/* Layout still works in memory. */}
    }
    controls.addEventListener('click',event=>{
        const button=event.target.closest('button');if(!button)return;
        if(button.dataset.focusLayout)state.mode=button.dataset.focusLayout;
        else if(button.hasAttribute('data-reset-layout'))state=normalizedLayout();else return;
        apply();
    });
    range.addEventListener('input',()=>{state.split=Number(range.value);apply();});
    separator.addEventListener('keydown',event=>{
        const values={ArrowLeft:state.split-1,ArrowRight:state.split+1,Home:40,End:65};
        if(!(event.key in values))return;event.preventDefault();state.split=values[event.key];apply();
    });
    separator.addEventListener('pointerdown',event=>{if(event.button!==0)return;event.preventDefault();separator.setPointerCapture(event.pointerId);separator.focus();});
    separator.addEventListener('pointermove',event=>{
        if(!separator.hasPointerCapture(event.pointerId))return;
        const left=editor.getBoundingClientRect().left,right=workspace.getBoundingClientRect().right;
        if(right>left){state.split=Math.round((event.clientX-left)/(right-left)*100);apply();}
    });
    separator.addEventListener('pointerup',event=>{if(separator.hasPointerCapture(event.pointerId))separator.releasePointerCapture(event.pointerId);});
    window.addEventListener('resize',()=>apply(false));
    new ResizeObserver(place).observe(workspace);
    apply(false);
}
