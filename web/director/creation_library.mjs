// Project-owned snapshots; plans are pure until the editor confirms one transaction.
const clone=value=>structuredClone(value);
const pick=(value,keys)=>Object.fromEntries(keys.filter(key=>Object.hasOwn(value,key)).map(key=>[key,clone(value[key])]));
const TEMPLATE_FIELDS='name mode sound writingMode simplePrompt prompt simpleInitialized advancedInitialized refs tray first last audio start end duration manualDuration autoDuration ratio ownRatio events'.split(' ');
export const CREATION_KINDS={snippet:'提示词片段',template:'镜头模板',recipe:'采样配方',group:'素材组合'};
const unique=values=>[...new Set(values.filter(Boolean))];
export function creationTokens(doc,shot,assets){
    const media=new Map(assets.map(item=>[item.id,item])),count={image:0,video:0,audio:0};
    return new Map(unique([...(doc.sharedRefs||[]),...(shot.tray||[]),shot.first,shot.last,...(shot.refs||[]),shot.audio]).filter(id=>media.has(id)).map(id=>[id,'@'+media.get(id).kind+(++count[media.get(id).kind])]));
}
export function savedCreationText(text,bindings,tokens){
    const missing=[];
    const value=text.replace(/@(image|video|audio)\d+\b|<(Picture|Video|Audio) \d+>/g,token=>{
        const id=bindings[token],mapped=id&&tokens.get(id);
        if(mapped)return mapped;
        missing.push(token);
        return '@missing_'+token.replace(/^@|[<> ]/g,'');
    });
    return {text:value,missing:unique(missing)};
}
export function creationSampling(doc,shot){
    const raw=shot.samplingInherit===false?shot.sampling:doc.sampling;
    if(shot.samplingInherit===false&&!raw)throw Error('本镜独立采样配置缺失，不能收藏为有效配方。');
    const source=clone(raw||{mode:'single'}),g=doc.generation||{};
    if((source.mode||'single')==='single'){
        source.mode='single';source.resolution_mp??=g.resolution_mp??'auto';
        if(Object.hasOwn(source,'loras'))source.lora_mode??='manual';
        else{
            // Match the existing backend's legacy generation settings, including
            // its auto/none sentinel handling; collection must not change a recipe.
            const legacy=Array.isArray(g.lora)?g.lora:[g.lora??'auto'];
            const explicit=legacy.filter(name=>!['auto','none'].includes(name));
            source.lora_mode=g.lora_mode??(explicit.length?'manual':'auto');
            source.loras=clone(g.loras?.length?g.loras:explicit.map(name=>({name,strength:g.lora_strength??1,enabled:true})));
        }
    }
    // Only the selected sampling mode is a recipe, not inactive dialog drafts.
    delete source.single;delete source.two_pass;delete source.hyperflow;
    return source;
}
function targetRow(doc,target){
    if(target?.kind==='global')return [doc,'global'];
    const shot=doc.shots.find(row=>row.id===target?.shot);
    if(!shot)throw Error('提示词目标镜头已不存在。');
    if(['prompt','simplePrompt'].includes(target.kind))return [shot,target.kind];
    const event=target.kind==='event'&&shot.events.find(row=>row.id===target.event);
    if(!event)throw Error('提示词目标事件已不存在。');
    return [event,'text'];
}
export function captureCreation({doc,shot,assets,kind,name,id,target,d3}){
    name=name.trim();if(!name||name.length>200)throw Error('请填写 1–200 字的收藏名称。');
    const tokens=creationTokens(doc,shot,assets),bindings=Object.fromEntries([...tokens].map(([aid,token])=>[token,aid])),base={id,name,kind};
    if(kind==='snippet'){
        const [row,key]=targetRow(doc,target),value=row[key]||'',selection=target.end>target.start?value.slice(target.start,target.end):value;
        if(!selection.trim()||selection.length>100000)throw Error('先选择或填写非空提示词；单个片段最多 100000 字符。');
        return {...base,text:selection,bindings:Object.fromEntries(Object.entries(bindings).filter(([token])=>new RegExp(token+'\\b').test(selection)))};
    }
    if(kind==='template'){
        const saved=pick(shot,TEMPLATE_FIELDS);
        saved.tray=[...tokens.keys()];saved.refs=unique([...(doc.sharedRefs||[]),...(shot.mode==='text'?[]:saved.refs)]);
        if(saved.mode==='text'&&saved.refs.length)saved.mode='refs';
        saved.samplingInherit=false;saved.sampling=creationSampling(doc,shot);saved.d3Inherit=false;saved.d3=clone(d3);
        saved.ownRatio=shot.ratio;
        return {...base,shot:saved,bindings};
    }
    if(kind==='recipe')return {...base,sampling:creationSampling(doc,shot),d3:clone(d3),ratio:shot.ratio,duration:shot.duration};
    if(kind==='group'){
        if(!tokens.size||tokens.size>200)throw Error('素材组合需要本镜有 1–200 项素材。');
        return {...base,asset_ids:[...tokens.keys()],note:''};
    }
    throw Error('未知收藏种类。');
}
export function addCreation(doc,item){
    const next=clone(doc),library=next.creationLibrary||{version:1,items:[]};
    if(library.items.length>=200)throw Error('本工程创作收藏已达到 200 项。');
    if(library.items.some(row=>row.id===item.id))throw Error('收藏身份重复，未覆盖原项。');
    library.items.push(clone(item));
    if(new TextEncoder().encode(JSON.stringify(library)).byteLength>2*1024**2)throw Error('创作收藏超过 2MiB，请拆分工程。');
    next.creationLibrary=library;return next;
}
export function renameCreation(doc,id,name){
    name=name.trim();if(!name||name.length>200)throw Error('名称须为 1–200 字');
    const next=clone(doc),item=next.creationLibrary?.items.find(row=>row.id===id);
    if(!item)throw Error('收藏已不存在');item.name=name;
    if(new TextEncoder().encode(JSON.stringify(next.creationLibrary)).byteLength>2*1024**2)throw Error('创作收藏超过 2MiB，请拆分工程。');
    return next;
}
export function planCreation({doc,shotId,assets,item,target,fields=[],targets=[shotId],uuid}){
    const next=clone(doc),shot=next.shots.find(row=>row.id===shotId),media=new Map(assets.map(row=>[row.id,row]));
    if(!shot)throw Error('当前镜头已不存在。');
    const missing=[],notes=[];let current=shotId;
    if(item.kind==='snippet'){
        if(target?.kind!=='global'&&target?.shot!==shotId)throw Error('提示词目标已不是当前镜头，请重新选择插入位置。');
        const [row,key]=targetRow(next,target),tokens=creationTokens(next,shot,assets);
        if(target.kind==='global')for(const id of tokens.keys())if(!next.sharedRefs.includes(id))tokens.delete(id);
        const value=savedCreationText(item.text,item.bindings,tokens);missing.push(...value.missing);
        const position=Math.max(0,Math.min(target.start??row[key].length,row[key].length));
        row[key]=row[key].slice(0,position)+value.text+row[key].slice(position);
        if(target.kind==='global')next.shots.forEach(row=>row.rev++);
        else{shot[target.kind==='simplePrompt'?'simpleInitialized':'advancedInitialized']=true;shot.rev++;}
        notes.push('在记住的光标处插入文字，不替换选中文字、不改变素材用途。');
    }else if(item.kind==='template'){
        const saved=clone(item.shot),ids=unique([...saved.tray,...saved.refs,saved.first,saved.last,saved.audio]);
        const lost=ids.filter(id=>!media.has(id)||media.get(id).missing);
        if(lost.length)throw Error('模板有 '+lost.length+' 项素材不可用，请先重连或重新收藏。不会只套用部分素材。');
        saved.id=uuid();saved.events=saved.events.map(event=>({...event,id:uuid()}));saved.rev=1;saved.name=item.name;
        saved.ratio=next.sharedRatio?next.ratio:saved.ownRatio;
        const tokens=creationTokens(next,saved,assets);
        for(const field of ['simplePrompt','prompt']){const value=savedCreationText(saved[field],item.bindings,tokens);saved[field]=value.text;missing.push(...value.missing);}
        saved.events.forEach(event=>{const value=savedCreationText(event.text,item.bindings,tokens);event.text=value.text;missing.push(...value.missing);});
        next.shots.splice(next.shots.indexOf(shot)+1,0,saved);current=saved.id;
        notes.push('在当前镜后新建模板副本；保存的素材/音频用途、写作草稿、时间和独立采样/D3随模板复制。','底模、全片提示词和共享素材仍用当前工程；开启共用画幅时跟随当前全片。种子与采用成片不复制。');
    }else if(item.kind==='recipe'){
        if(!fields.length||fields.some(key=>!['sampling','d3','ratio','duration'].includes(key)))throw Error('请选择至少一个明确的参数类别。');
        if(fields.includes('ratio')&&next.sharedRatio)throw Error('当前开启全片共用画幅。请先关闭后再按镜头复制画幅，避免修改未选镜头。');
        if(!targets.length||targets.some(id=>!next.shots.some(row=>row.id===id)))throw Error('请选择仍在本工程中的目标镜头。');
        for(const row of next.shots.filter(row=>targets.includes(row.id))){
            if(fields.includes('sampling')){row.samplingInherit=false;row.sampling=clone(item.sampling);}
            if(fields.includes('d3')){row.d3Inherit=false;row.d3=clone(item.d3);}
            if(fields.includes('ratio'))row.ownRatio=row.ratio=item.ratio;
            if(fields.includes('duration')){row.autoDuration=false;row.duration=row.manualDuration=item.duration;}
            row.rev++;
        }
        notes.push('只覆盖选择的类别与镜头；采样/D3改为独立快照，后续不跟随全片同类设置。','保留底模、文字、素材、音频用途、种子和采用版本；时长改变不自动缩放时间事件，需再次预检。');
    }else if(item.kind==='group'){
        const before=Object.fromEntries([...creationTokens(next,shot,assets)].map(([id,token])=>[token,id]));
        if(item.asset_ids.some(id=>!media.has(id)||media.get(id).missing))throw Error('组合包含不可用素材，请先重连。不会部分套用。');
        shot.tray=unique([...shot.tray,...item.asset_ids]);
        const visuals=item.asset_ids.filter(id=>['image','video'].includes(media.get(id).kind)&&!next.sharedRefs.includes(id));
        shot.refs=unique([...shot.refs,...visuals]);if(shot.mode==='text'&&visuals.length)shot.mode='refs';shot.rev++;
        const after=creationTokens(next,shot,assets);
        const remap=text=>text.replace(/@(image|video|audio)\d+\b/g,token=>before[token]?(after.get(before[token])||'@missing_'+token.slice(1)):token);
        for(const key of ['simplePrompt','prompt'])shot[key]=remap(shot[key]);
        shot.events.forEach(event=>event.text=remap(event.text));
        notes.push('图片/视频加入本镜参考，音频只加入素材栏，不自动设为驱动或音色；不改全片共享或首尾帧。');
    }else throw Error('未知收藏种类。');
    return {doc:next,current,notes,missing:unique(missing)};
}
export function remapCreationAsset(doc,oldId,newId){
    for(const item of doc.creationLibrary?.items||[]){
        if(item.bindings)for(const [token,id] of Object.entries(item.bindings))if(id===oldId)item.bindings[token]=newId;
        if(item.kind==='group')item.asset_ids=unique(item.asset_ids.map(id=>id===oldId?newId:id));
        if(item.kind==='template'){
            for(const key of ['tray','refs'])item.shot[key]=unique(item.shot[key].map(id=>id===oldId?newId:id));
            for(const key of ['first','last','audio'])if(item.shot[key]===oldId)item.shot[key]=newId;
        }
    }
}
