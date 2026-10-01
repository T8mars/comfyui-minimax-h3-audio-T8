export function assetMatchesFilter(asset, {query='',kind='all',scope='all',recent=false}={}, shared=[], recentIds=[]) {
    const text=query.trim().toLocaleLowerCase();
    return !!asset && (!text||String(asset.name||'').toLocaleLowerCase().includes(text)) &&
        (kind==='all'||asset.kind===kind) && (scope==='all'||(scope==='shared'?shared.includes(asset.id):!shared.includes(asset.id))) &&
        (!recent||recentIds.includes(asset.id));
}
export function createAssetFilters(host, ctx) {
    const filters=document.createElement('div');filters.className='w-asset-filters';
    filters.innerHTML='<input type="search" data-asset-search placeholder="搜索本镜素材" aria-label="搜索本镜素材"><select data-asset-kind aria-label="素材类型"><option value="all">全部类型</option><option value="image">图片</option><option value="video">视频</option><option value="audio">音频</option></select><select data-asset-scope aria-label="素材范围"><option value="all">全部范围</option><option value="shared">全片参考</option><option value="local">本镜素材</option></select><label><input type="checkbox" data-asset-recent>最近使用</label><button type="button" data-clear-asset-filter>清除筛选</button><small data-asset-filter-count role="status"></small>';
    const thumbs=host.querySelector('[data-thumbs]');thumbs.before(filters);
    const toggle=document.createElement('button');toggle.type='button';toggle.dataset.assetFiltersToggle='';toggle.textContent='筛选';toggle.setAttribute('aria-expanded','false');
    host.querySelector('[data-workbench-asset-actions]').append(toggle);filters.hidden=true;
    filters.querySelector('[data-asset-recent]').nextSibling.textContent='最近查看';
    toggle.addEventListener('click',()=>{filters.hidden=!filters.hidden;toggle.setAttribute('aria-expanded',String(!filters.hidden));if(!filters.hidden)filters.querySelector('[data-asset-search]').focus();});
    let recent=[],lastSelected=null;
    const key='t8director.recentAssets.v1';
    try{const saved=JSON.parse(localStorage.getItem(key)||'[]');if(Array.isArray(saved))recent=saved.filter(id=>typeof id==='string').slice(0,20);}catch{/* Optional preferences. */}
    function apply(){
        const query=filters.querySelector('[data-asset-search]').value,kind=filters.querySelector('[data-asset-kind]').value,scope=filters.querySelector('[data-asset-scope]').value,onlyRecent=filters.querySelector('[data-asset-recent]').checked;
        const cards=[...thumbs.querySelectorAll('[data-tray-id]')];let shown=0;
        for(const card of cards){card.hidden=!assetMatchesFilter(ctx.assets().get(card.dataset.trayId),{query,kind,scope,recent:onlyRecent},ctx.shared(),recent);if(!card.hidden)shown++;}
        const active=!!query.trim()||kind!=='all'||scope!=='all'||onlyRecent;
        const status=filters.querySelector('[data-asset-filter-count]');status.hidden=!active;
        status.textContent=`显示 ${shown}/${cards.length} 项 · 筛选不改变排序或 @ 引用`;
        toggle.textContent=active?`筛选 ${shown}/${cards.length}`:'筛选';
    }
    filters.addEventListener('input',apply);filters.addEventListener('change',apply);
    filters.querySelector('[data-clear-asset-filter]').addEventListener('click',()=>{filters.querySelector('[data-asset-search]').value='';filters.querySelectorAll('select').forEach(select=>select.value='all');filters.querySelector('[data-asset-recent]').checked=false;apply();filters.querySelector('[data-asset-search]').focus();});
    function sync(){
        const id=ctx.selected();
        if(id&&id!==lastSelected){lastSelected=id;recent=[id,...recent.filter(value=>value!==id)].slice(0,20);try{localStorage.setItem(key,JSON.stringify(recent));}catch{/* Keep in-memory recency. */}}
        apply();
    }
    return {sync};
}

// Search all project assets; hiding cards must never change roles or @ numbering.
export function createLibraryFilters(dialog, {assets, shared, bound}) {
    const grid=dialog.querySelector('[data-library-grid]');
    if(!grid)return;
    const controls=document.createElement('section');controls.className='w-library-filters';
    controls.innerHTML='<label>查找素材<input type="search" data-library-search placeholder="输入名称或素材编号" aria-label="搜索工程素材库"></label><label>类型<select data-library-kind><option value="all">全部类型</option><option value="image">图片</option><option value="video">视频</option><option value="audio">音频</option></select></label><label>范围<select data-library-scope><option value="all">当前列表全部</option><option value="shared">全片共享</option><option value="bound">本镜已加入</option><option value="unbound">尚未加入本镜</option></select></label><button type="button" data-library-reset>清除筛选</button><p data-library-count-status role="status"></p>';
    grid.before(controls);
    function filter(){
        const query=controls.querySelector('[data-library-search]').value.trim().toLocaleLowerCase();
        const kind=controls.querySelector('[data-library-kind]').value,scope=controls.querySelector('[data-library-scope]').value;
        const cards=[...grid.querySelectorAll('[data-library-asset]')];let shown=0;
        for(const card of cards){
            const id=card.dataset.libraryAsset,asset=assets().get(id);
            const name=String(asset?.name||'').toLocaleLowerCase();
            card.hidden=!asset||(query&&!name.includes(query)&&!id.toLocaleLowerCase().includes(query))||(kind!=='all'&&asset.kind!==kind)||(scope==='shared'&&!shared().includes(id))||(scope==='bound'&&!bound().includes(id))||(scope==='unbound'&&bound().includes(id));
            if(!card.hidden)shown++;
        }
        controls.querySelector('[data-library-count-status]').textContent=`显示 ${shown}/${cards.length} 项${shown?'':' · 没有匹配素材，可清除筛选'}。筛选不会更改用途或引用编号。`;
    }
    controls.addEventListener('input',filter);controls.addEventListener('change',filter);
    controls.querySelector('[data-library-reset]').addEventListener('click',()=>{controls.querySelector('input').value='';controls.querySelectorAll('select').forEach(select=>select.value='all');filter();controls.querySelector('input').focus();});
    filter();controls.querySelector('input').focus();
}
