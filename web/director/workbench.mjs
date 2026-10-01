// Presentation only: keep the existing project, controls, event handlers and services.
import { directorSamplingSummary } from './sampling_ui.mjs';
import { directorDraftIssues } from './preflight_ui.mjs';
import { createDirectorLayout } from './layout_ui.mjs';
import { createAssetFilters, createLibraryFilters } from './assets_ui.mjs';
import { createFilmViewer } from './film_ui.mjs';
import { createVersionComparison } from './compare_ui.mjs';
import { createFramePicker } from './frame_ui.mjs';
import { createTrimEditor } from './trim_ui.mjs';
import { createBundleDialog } from './bundle_ui.mjs';
import { createReferenceCandidates } from './references_ui.mjs';
import { createTaskDrawer } from './tasks_ui.mjs';
import { createStoryboardDialog } from './storyboard_ui.mjs';
import { createCreationLibrary } from './creation_ui.mjs';
import { directorTaskLabel } from './session.mjs';
export function createDirectorWorkbench(ctx) {
    const { root, doc, shot, assets, results, videos } = ctx;
    const $ = selector => root.querySelector(selector);
    const $$ = selector => [...root.querySelectorAll(selector)];
    const css = document.createElement('link');
    css.rel = 'stylesheet';
    css.href = new URL('./workbench.css', import.meta.url).href;
    document.head.append(css);
    root.dataset.workbench = 'parallel';
    root.setAttribute('aria-label','曜石导演台');
    document.title = '曜石导演台 · 并排创作台';
    let activePanel = null;

    function menu(label, className = '') {
        const node = document.createElement('details');
        node.className = 'w-menu ' + className;
        node.innerHTML = `<summary>${label}</summary><div class="w-menu-body"></div>`;
        return node;
    }
    const header = $('.o-header');
    $('.o-brand h2').textContent = '曜石导演台';
    $('.o-eyebrow').textContent = '并排创作台';
    const project = $('.o-project');
    const projectActions = project.querySelector('.o-row');
    const actions = document.createElement('div');
    actions.className = 'w-header-actions';
    actions.append($('[data-action="model-settings"]'), $('[data-service="save"]'));
    const more = menu('更多');
    const moreBody = more.querySelector('.w-menu-body');
    moreBody.append(...projectActions.children);
    moreBody.append($('[data-action="undo"]'), $('[data-action="redo"]'));
    const creation=createCreationLibrary(root,{...ctx.creation,escape:ctx.esc,notify:ctx.notify});
    const creationButton=document.createElement('button');creationButton.type='button';creationButton.dataset.creationOpen='';creationButton.textContent='创作收藏';creationButton.addEventListener('click',creation.open);moreBody.prepend(creationButton);
    const oldBadge = $('.o-badge');
    oldBadge.parentElement.remove();
    moreBody.append(oldBadge);
    const bundle=createBundleDialog({root,escape:ctx.esc,prepare:ctx.prepareBundle,build:ctx.buildBundle,upload:ctx.uploadBundle,apply:ctx.applyBundle,lastImport:ctx.lastBundleImport,notify:ctx.notify});
    const bundleButton=document.createElement('button');bundleButton.type='button';bundleButton.dataset.bundleOpen='';bundleButton.textContent='工程包 ZIP';bundleButton.addEventListener('click',()=>bundle.open());moreBody.prepend(bundleButton);
    const note = $('.o-note');
    moreBody.append(note);
    projectActions.remove();
    actions.append(more);
    if (window.parent === window) {
        const back = document.createElement('a');
        back.href = '/'; back.className = 'w-back'; back.textContent = '返回画布';
        actions.append(back);
    }
    header.append(actions);
    const save = $('[data-save]');
    const compactSave=document.createElement('span');
    compactSave.className='w-compact-save';compactSave.setAttribute('role','status');
    save.after(compactSave);
    const updateSave=()=>{
        const text=save.textContent;
        save.title=text;compactSave.title=text;
        compactSave.textContent=navigator.onLine===false?'离线 · 草稿保留':/失败|未完成|冲突/.test(text)?'保存未完成':/正在保存/.test(text)?'保存中…':/请保存|请再次保存|未保存/.test(text)?'草稿未保存':/已真实保存|已载入项目/.test(text)?'已保存':'本地草稿';
    };
    updateSave();
    new MutationObserver(updateSave).observe(save, {childList:true,subtree:true,characterData:true});
    window.addEventListener('online',updateSave);window.addEventListener('offline',updateSave);

    const workspace = $('.o-workspace');
    const stage = $('.o-stage');
    const shots = $('.o-shotbar');
    const shotActions = shots.querySelector('.o-row > .o-row');
    const shotMenu = menu('镜头操作');
    shotMenu.querySelector('.w-menu-body').append($('[data-action="duplicate"]'), $('[data-action="delete-shot"]'));
    const storyboard=createStoryboardDialog(root,{escape:ctx.esc,notify:ctx.notify,context:ctx.taskContext,snapshot:ctx.creationSnapshot,apply:ctx.importStoryboard});
    const importStoryboard=document.createElement('button');importStoryboard.type='button';importStoryboard.dataset.storyboardOpen='';importStoryboard.textContent='导入文本分镜';importStoryboard.addEventListener('click',storyboard.open);shotMenu.querySelector('.w-menu-body').append(importStoryboard);
    const movement=document.createElement('div');movement.className='o-row';
    movement.innerHTML='<button type="button" data-shot-up>上移本镜</button><button type="button" data-shot-down>下移本镜</button>';
    shotMenu.querySelector('.w-menu-body').append(movement);
    movement.querySelector('[data-shot-up]').addEventListener('click',()=>ctx.moveShot(shot().id,doc().shots.indexOf(shot())-1));
    movement.querySelector('[data-shot-down]').addEventListener('click',()=>ctx.moveShot(shot().id,doc().shots.indexOf(shot())+1));
    let draggingShot=null;
    shots.addEventListener('dragstart',e=>{const card=e.target.closest('[data-shot]');if(!card)return;draggingShot=card.dataset.shot;e.dataTransfer.setData('application/x-t8-shot',draggingShot);e.dataTransfer.effectAllowed='move';e.stopPropagation();});
    shots.addEventListener('dragover',e=>{if(draggingShot&&e.target.closest('[data-shot]')){e.preventDefault();e.dataTransfer.dropEffect='move';}});
    shots.addEventListener('drop',e=>{const card=e.target.closest('[data-shot]');if(!card||!draggingShot)return;e.preventDefault();e.stopPropagation();const id=draggingShot;draggingShot=null;ctx.moveShot(id,doc().shots.findIndex(s=>s.id===card.dataset.shot));});
    shots.addEventListener('dragend',()=>{draggingShot=null;});
    shots.addEventListener('change',e=>{if(e.target.matches('[data-shot-check]')){ctx.selectShot(e.target.dataset.shotCheck,e.target.checked);syncSelection();}});
    const shotFooter = document.createElement('div');
    shotFooter.className = 'w-shot-actions';
    shotFooter.append($('[data-action="new"]'), shotMenu);
    shotActions.remove();
    shots.append(shotFooter);
    shots.querySelector('h3').firstChild.textContent = '镜头列表 ';
    const editor = document.createElement('section');
    editor.className = 'w-editor'; editor.setAttribute('aria-label', '当前镜头剧本');
    editor.innerHTML = '<div class="w-shot-heading"><h2 data-workbench-heading></h2><div data-workbench-name></div></div>';
    for (const selector of ['[data-global-zone]','[data-writing-zone]','[data-simple-zone]','[data-local-zone]']) editor.append($(selector));
    editor.append($('[data-events]').closest('details'));
    const settings = document.createElement('section');
    settings.className = 'w-settings';
    settings.innerHTML = '<h3>基础设置</h3><div class="w-setting-tiles" data-workbench-tiles></div><button class="w-enhancements" data-workbench-panel="d3"><span>增强与性能</span><small data-workbench-enhancements></small></button>';
    settings.append($('[data-inspector]'));
    const variation = document.createElement('div');
    variation.className = 'w-variation';
    variation.innerHTML = '<label>本镜种子<input data-shot-seed type="text" inputmode="numeric" pattern="[0-9]+" required aria-label="本镜种子" title="普通生成复用这个种子；换种子生成会保存新的种子。相同种子不保证跨环境逐像素一致。"></label><button type="button" data-service="new-variation">换种子生成一版</button>';
    settings.append(variation);
    editor.append(settings);
    const readiness=document.createElement('details');
    readiness.className='w-readiness';
    readiness.innerHTML='<summary data-draft-status></summary><p class="o-tip">自动检查仅看当前草稿，不读取模型、不提交任务。文件与尺寸请手动预检；依赖另查。</p><ul data-draft-issues></ul><div class="o-row"><button type="button" data-action="check">文件与尺寸预检</button><button type="button" data-service="d3-preflight">依赖检查</button></div>';
    readiness.querySelector('[data-draft-issues]').addEventListener('click',e=>{const button=e.target.closest('[data-draft-field]');if(button)focusIssue(button.dataset.draftField);});
    editor.append(readiness);
    workspace.prepend(shots, editor);

    const mediaWorkbench = $('.o-media-workbench');
    const thumbs = $('[data-thumbs]');
    const mediaSection = document.createElement('section');
    mediaSection.className = 'w-assets';
    mediaSection.innerHTML = '<div class="w-assets-heading"><h3>本镜素材 <small data-workbench-asset-count></small></h3><div class="o-row" data-workbench-asset-actions></div></div>';
    const mediaActions = mediaSection.querySelector('[data-workbench-asset-actions]');
    const add = stage.querySelector(':scope > .o-row [data-action="add"]');
    add.textContent = '添加素材';
    mediaActions.append(add);
    const mediaMenu = menu('素材操作');
    mediaMenu.querySelector('.w-menu-body').append($('[data-action="all"]'), $('[data-action="pair"]'), $('[data-action="zoom"]'), $('[data-action="toggle-preview"]'));
    mediaActions.append(mediaMenu);
    mediaSection.append(thumbs, $('.o-trayhelp'), $('[data-insert-target]'));
    const assetFilters=createAssetFilters(mediaSection,{assets,shared:()=>doc().sharedRefs||[],selected:()=>shot().selected});
    mediaMenu.querySelector('.w-menu-body').append(mediaSection.querySelector('.o-trayhelp'));
    const insertTarget=mediaSection.querySelector('[data-insert-target]');
    new MutationObserver(()=>{insertTarget.title=insertTarget.textContent;}).observe(insertTarget,{childList:true,characterData:true,subtree:true});
    const selectedActions=document.createElement('div');
    selectedActions.className='w-selection-actions';
    selectedActions.setAttribute('aria-label','选中素材操作');
    thumbs.after(selectedActions);
    mediaWorkbench.after(mediaSection);
    const versionBar=document.createElement('section');
    versionBar.className='w-versions';
    versionBar.setAttribute('aria-label','本镜生成版本');
    versionBar.innerHTML='<label>生成版本<select data-result-version aria-label="查看生成版本"></select></label><div class="o-row"><button type="button" data-adopt-version>采用此版</button><button type="button" data-version-details>版本详情</button></div><small data-version-status role="status"></small>';
    mediaWorkbench.after(versionBar);
    versionBar.querySelector('[data-result-version]').addEventListener('change',e=>ctx.previewVersion(e.target.value));
    versionBar.querySelector('[data-adopt-version]').addEventListener('click',()=>ctx.adoptVersion(ctx.resultKey(results().get(shot().id))));
    versionBar.querySelector('[data-version-details]').addEventListener('click',()=>ctx.showVersion(results().get(shot().id)));
    const framePicker=createFramePicker(root,ctx.frame);
    const trimEditor=createTrimEditor(root,ctx.trim);
    const trimButton=document.createElement('button');trimButton.type='button';trimButton.dataset.trimOpen='';trimButton.textContent='编辑采用范围';trimButton.addEventListener('click',trimEditor.open);versionBar.querySelector('.o-row').append(trimButton);
    const trimSummary=document.createElement('small');trimSummary.dataset.trimSummary='';versionBar.append(trimSummary);
    const frameButton=document.createElement('button');frameButton.type='button';frameButton.dataset.frameOpen='';frameButton.textContent='取帧接下一镜';
    frameButton.addEventListener('click',()=>framePicker.open(ctx.resultKey(results().get(shot().id))));versionBar.querySelector('.o-row').append(frameButton);
    const comparison=createVersionComparison({root,versions:ctx.versions,resultKey:ctx.resultKey,videos,prepare:ctx.prepareComparison,escape:ctx.esc});
    const compareButton=document.createElement('button');compareButton.type='button';compareButton.dataset.compareOpen='';compareButton.textContent='A/B 对比';
    compareButton.addEventListener('click',()=>comparison.open(ctx.resultKey(results().get(shot().id))));versionBar.querySelector('.o-row').append(compareButton);
    const tabs = $('.o-viewbar .o-tabs');
    const film=createFilmViewer({root,prepare:ctx.prepareFilm,notify:ctx.notify,locate:ctx.locateShot,escape:ctx.esc,startExport:ctx.startFilmExport,exportStatus:ctx.filmExportStatus});
    const filmButton=document.createElement('button');filmButton.type='button';filmButton.dataset.filmOpen='';filmButton.textContent='整片串播';filmButton.addEventListener('click',()=>film.open());tabs.append(filmButton);
    tabs.prepend($('[data-view="output"]'));
    $('[data-view="input"]').textContent = '输入预览';
    const toolsMenu = menu('检查与导出', 'w-tools-menu');
    for (const button of $$('.o-footer [data-service],.o-footer [data-action="check"]')) {
        if (!['job-status','resume-batch','pause-batch'].includes(button.dataset.service)) toolsMenu.querySelector('.w-menu-body').append(button);
    }
    moreBody.prepend(toolsMenu);
    const footerActions = $('.o-footer > .o-row');
    const tasksButton=document.createElement('button');tasksButton.type='button';tasksButton.dataset.service='task-list';tasksButton.textContent='任务列表';footerActions.prepend(tasksButton);
    footerActions.prepend($('[data-action="generate-all"]'));
    const batchMenu=menu('选择生成','w-batch-menu');
    batchMenu.querySelector('.w-menu-body').innerHTML='<button type="button" data-service="generate-selected">生成选中的镜头</button><button type="button" data-service="generate-ungenerated">生成尚无成片的镜头</button><button type="button" data-service="generate-modified">生成配置已修改的镜头</button><p class="o-tip">按镜头列表顺序生成。旧版本无完整快照时不猜测改动，请手动勾选。</p>';
    footerActions.prepend(batchMenu);
    const mobileShots = document.createElement('button');
    mobileShots.className = 'w-mobile-shots'; mobileShots.textContent = '镜头列表';
    mobileShots.setAttribute('aria-expanded','false');
    mobileShots.addEventListener('click', () => { const open = root.dataset.shotsOpen !== 'true'; root.dataset.shotsOpen = String(open); mobileShots.setAttribute('aria-expanded',String(open)); });
    header.prepend(mobileShots);

    // Global uploads keep their original route after the global editor moves columns.
    const globalZone = $('[data-global-zone]');
    globalZone.addEventListener('dragover', e => { if(e.dataTransfer.types.includes('Files')) e.preventDefault(); });
    globalZone.addEventListener('drop', e => { if(e.dataTransfer.files.length){e.preventDefault();ctx.importFiles(e.dataTransfer.files,{target:'global',shot:shot().id});} });
    root.addEventListener('click', e => {
        const panel = e.target.closest('[data-workbench-panel]');
        if(panel){ activePanel = activePanel === panel.dataset.workbenchPanel ? null : panel.dataset.workbenchPanel; syncPanels(); }
        if(e.target.closest('[data-workbench-close]')){ activePanel = null; syncPanels(); }
        if(e.target.closest('[data-workbench-dismiss]'))$('[data-notice]').hidden=true;
        if(e.target.closest('[data-shot]')) {root.dataset.shotsOpen='false';mobileShots.setAttribute('aria-expanded','false');}
    });
    root.addEventListener('keydown', e => { if(e.key==='Escape'){ $$('.w-menu[open]').forEach(m=>m.open=false); root.dataset.shotsOpen='false'; mobileShots.setAttribute('aria-expanded','false'); } });
    // Existing service handlers stop propagation at root. Close menus before that boundary.
    document.addEventListener('click',e=>{
        const button=e.target.closest('.w-menu button');
        if(button && root.contains(button) && !button.disabled){
            const parents=$$('.w-menu[open]').filter(m=>m.contains(button));
            queueMicrotask(()=>parents.forEach(m=>m.open=false));
        }
        for(const m of $$('.w-menu[open]')) if(!m.contains(e.target))m.open=false;
    },true);

    function syncPanels() {
        $$('[data-workbench-section]').forEach(section => section.hidden=section.dataset.workbenchSection!==activePanel);
        $$('[data-workbench-panel]').forEach(button => button.setAttribute('aria-expanded',String(button.dataset.workbenchPanel===activePanel)));
    }
    function syncInspector() {
        const inspector = $('[data-inspector]');
        const sections = [...inspector.children];
        const name = inspector.querySelector('[data-field="name"]');
        if(name){$('[data-workbench-name]').replaceChildren(name.closest('label'));sections[0].hidden=true;}
        for(const [index,key] of ['source','audio','timing','d3'].entries()){
            const section=sections[index+1]; if(!section)continue;
            section.dataset.workbenchSection=key;
            section.querySelector('h3').insertAdjacentHTML('afterend','<button type="button" class="w-close-panel" data-workbench-close aria-label="收起设置">收起</button>');
        }
        syncSummary(); syncPanels();
    }
    function syncSummary() {
        const s=shot();
        const seedField=$('[data-shot-seed]');
        if(document.activeElement!==seedField){seedField.value=String(s.seed??26091901);seedField.setCustomValidity('');}
        $('[data-workbench-heading]').textContent='第 '+String(doc().shots.indexOf(s)+1).padStart(2,'0')+' 镜';
        const mode={text:'文字',first:'首帧',ends:'首尾',refs:'参考素材'}[s.mode];
        const audio={native:'模型生成声音',record:'音频驱动',voice:'参考音色'}[s.sound];
        const samplingSummary=directorSamplingSummary(doc(),s);
        const tiles=[['source','画面来源',mode],['audio','声音',audio],['timing','时长',s.duration+' 秒'],['timing','画幅',s.ratio+(doc().sharedRatio?' · 全片共用':' · 本镜')]];
        const host=$('[data-workbench-tiles]');
        if(!host.children.length) host.innerHTML=tiles.map(([key,label])=>`<button data-workbench-panel="${key}"><small>${label}</small><span></span></button>`).join('')+'<button data-action="model-settings"><small>分辨率</small><span></span></button>';
        [...host.children].forEach((button,i)=>{button.querySelector('span').textContent=i<4?tiles[i][2]:samplingSummary.label;});
        host.lastElementChild.title='目标像素；实际宽高以当前镜头预检为准。'+(samplingSummary.inherited?'采样跟随全片。':'本镜独立采样。');
        const canvas=ctx.checkedCanvas?.();
        if(canvas&&Number.isInteger(canvas.width)&&Number.isInteger(canvas.height)){
            host.lastElementChild.querySelector('span').textContent=`${canvas.width} × ${canvas.height}`;
            host.lastElementChild.title=`本镜预检尺寸 · ${(canvas.width*canvas.height/1000000).toFixed(3)} MP · ${samplingSummary.label}。修改草稿后重新预检。`;
        }
        const issues=directorDraftIssues(doc(),s,[...assets().values()]);
        readiness.querySelector('[data-draft-status]').textContent=issues.length?`草稿检查 · ${issues.length} 项待完善`:'草稿检查 · 未发现基础缺项';
        const issueHTML=issues.map(issue=>`<li>${ctx.esc(issue.message)} <button type="button" data-draft-field="${ctx.esc(issue.field)}">去修正</button></li>`).join('');
        const issueList=readiness.querySelector('[data-draft-issues]');
        if(issueList.dataset.html!==issueHTML){issueList.innerHTML=issueHTML;issueList.dataset.html=issueHTML;}
        const d=ctx.effectiveD3(s);
        const enabled=[d.semantic_bridge?.enabled&&'语义桥接',d.prompt_relay?.enabled&&'提示词接力',d.fast_h3_v2?.enabled&&'FastH3 V2',d.memory?.low_vram&&'低显存',d.memory?.chunk_ffn&&'分块前馈'].filter(Boolean);
        $('[data-workbench-enhancements]').textContent=enabled.length?enabled.join(' · '):'未启用增强';
        $('[data-action="generate-all"]').textContent='按顺序生成全部（'+doc().shots.length+' 镜）';
        $('[data-global-zone] > summary').innerHTML='全片设定 <small>· '+doc().shots.length+' 镜共用</small>';
        for(const button of $$('[data-writing]'))button.textContent=button.dataset.writing==='simple'?'新手模式':'高级模式';
        syncPanels();
    }
    function syncAssets() {
        assetFilters.sync();
        const cards=$$('[data-thumbs] [data-tray-id]');
        $('[data-workbench-asset-count]').textContent='（'+cards.length+'）';
        if(cards.length&&!cards.some(c=>c.dataset.selected==='true'))cards[0].dataset.selected='true';
        for(const card of cards){const a=assets().get(card.dataset.trayId);if(a)card.title=a.name;}
        const selected=cards.find(card=>card.dataset.selected==='true');
        if(!selected)selectedActions.replaceChildren();
        else if(selected.querySelector('.o-quick')){
            selectedActions.replaceChildren(selected.querySelector('.o-quick'));
            const extra=menu('更多');
            const body=extra.querySelector('.w-menu-body');
            for(const control of selected.querySelectorAll('.o-dragline,.o-delete-asset'))body.append(control);
            selectedActions.append(extra);
            selected.dataset.workbenchActions='detached';
        }
    }
    function syncShots() {
        root.dataset.shotsOpen='false';
        mobileShots.setAttribute('aria-expanded','false');
        for(const button of $$('[data-shots] [data-shot]')){
            const s=doc().shots.find(s=>s.id===button.dataset.shot);if(!s)continue;
            button.draggable=true;
            if(!button.parentElement.classList.contains('w-shot-choice')){
                const choice=document.createElement('div');choice.className='w-shot-choice';
                const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.dataset.shotCheck=s.id;
                button.before(choice);choice.append(checkbox,button);
            }
            const checkbox=button.parentElement.querySelector('[data-shot-check]');
            checkbox.checked=ctx.selectedShotIds().includes(s.id);checkbox.setAttribute('aria-label','选择生成：'+s.name);
            const a=[s.first,...(s.refs||[]),...(doc().sharedRefs||[]),...(s.tray||[])].map(id=>assets().get(id)).find(a=>a?.kind==='image');
            let thumb=button.querySelector('.w-shot-thumbnail');
            if(a){if(!thumb){thumb=document.createElement('img');thumb.className='w-shot-thumbnail';thumb.alt='';button.prepend(thumb);}if(thumb.getAttribute('src')!==a.url)thumb.src=a.url;}
            else thumb?.remove();
            button.dataset.hasOutput=String(videos(results().get(s.id)).length>0);
        }
        $('[data-shot-up]').disabled=doc().shots.indexOf(shot())===0;
        $('[data-shot-down]').disabled=doc().shots.indexOf(shot())===doc().shots.length-1;
        syncSelection();
    }
    function syncSelection(){
        const count=doc().shots.filter(s=>ctx.selectedShotIds().includes(s.id)).length;
        const button=$('[data-service="generate-selected"]');button.textContent='生成选中镜头（'+count+'）';button.disabled=!count;
    }
    function syncPreview() {
        const view=ctx.view();
        $$('[data-view]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.view===view)));
        // Background completions update only the result pane, not the prompt editor.
        const output=view==='output';
        versionBar.hidden=!output;
        const records=ctx.versions(),current=results().get(shot().id),key=ctx.resultKey(current);
        const select=versionBar.querySelector('[data-result-version]');
        const options=records.map((item,index)=>{
            const date=item.submitted_at?new Date(item.submitted_at*1000).toLocaleString():'时间未记录';
            const state={success:'已生成',error:'失败',pending:'生成中',cancelled:'已取消'}[item.state]||'待核对';
            return `<option value="${ctx.esc(ctx.resultKey(item))}">${ctx.esc(`${records.length-index} · ${date} · ${state}`)}</option>`;
        }).join('');
        if(select.dataset.options!==options){select.innerHTML='<option value="" disabled>选择一个版本</option>'+options;select.dataset.options=options;}
        select.value=key||'';
        const adopted=shot().adoptedResultId===key&&!!key;
        const adopt=versionBar.querySelector('[data-adopt-version]');
        adopt.textContent=adopted?'已采用':'采用此版';
        adopt.disabled=adopted||current?.state!=='success'||!videos(current).length;
        versionBar.querySelector('[data-version-details]').disabled=!current;
        compareButton.disabled=records.filter(row=>row.state==='success'&&videos(row).length===1).length<2;
        frameButton.disabled=current?.state!=='success'||videos(current).length!==1;
        trimButton.disabled=!shot().adoptedResultId&&!shot().filmTrim;
        const range=shot().filmTrim;
        trimSummary.textContent=range?`整片采用范围：第 ${range.in_frame}–${range.out_frame} 帧（出帧不含）；可编辑或恢复全片。`:'整片采用范围：完整原片；编辑范围不会裁改原文件。';
        versionBar.querySelector('[data-version-status]').textContent=current?.integrity_error||(shot().adoptedResultId&&!records.some(item=>ctx.resultKey(item)===shot().adoptedResultId)?'采用版本暂未找到；不会自动改用其它成片。':adopted?'已固定采用版本；新生成不会替换。':'正在预览；点击采用后用于整片。');
        const player=$('.o-output-player');
        if(player)player.setAttribute('aria-label','第 '+(doc().shots.indexOf(shot())+1)+' 镜'+(adopted?'采用版本':'预览版本'));
        const text=$('.o-output-meta span');
        if(text)text.textContent='第 '+(doc().shots.indexOf(shot())+1)+' 镜 · '+(adopted?'采用版本':'预览版本');
        const stageEmpty=$('[data-stage] .o-empty p');
        if(stageEmpty&&shot().mode==='text'&&!player)stageEmpty.textContent='在中间提示词区描述画面，无需上传首帧。生成后可在这里回看成片。';
        root.dataset.activeView=view;
    }
    function focusIssue(field) {
        let target;
        if(['source','audio','timing','d3'].includes(field)){
            activePanel=field;syncPanels();
            target=$(`[data-workbench-section="${field}"]`);
        }else if(field==='global')target=$('[data-global-zone]');
        else if(field==='events')target=$('[data-events]').closest('details');
        else target=$(shot().writingMode==='simple'?'[data-simple-zone]':'[data-local-zone]');
        if(target?.tagName==='DETAILS')target.open=true;
        target?.scrollIntoView({block:'nearest'});
        target?.querySelector('textarea,input,select,button')?.focus();
    }
    const layoutMenu=menu('布局');moreBody.append(layoutMenu);
    createDirectorLayout(root,layoutMenu.querySelector('.w-menu-body'));
    createReferenceCandidates(root,{context:ctx.referenceContext,choices:ctx.referenceChoices,insert:ctx.insertReference});
    const tasks=createTaskDrawer(root,{context:ctx.taskContext,load:ctx.loadTasks,shots:()=>doc().shots,escape:ctx.esc,label:directorTaskLabel,notify:ctx.notify,locate:ctx.locateTask});
    function filterLibrary(){createLibraryFilters($('[data-dialog]'),{assets,shared:()=>doc().sharedRefs||[],bound:()=>[...new Set([...(doc().sharedRefs||[]),...(shot().tray||[]),...(shot().refs||[]),shot().first,shot().last,shot().audio].filter(Boolean))]});}
    return {syncInspector,syncSummary,syncAssets,syncShots,syncPreview,focusIssue,filterLibrary,openTasks:tasks.open};
}
