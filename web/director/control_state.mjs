// Versioned editable control data. No model admission or queue authorization.
const object=value=>value&&typeof value==='object'&&!Array.isArray(value);
const uuid=value=>typeof value==='string'&&/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(value);
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const fields=['id','asset_id','enabled','control_net_name','control_kind','fit_mode','strength','start_percent','end_percent','stages','source','region','crop'];
const nested={stages:['low','high'],source:['start_seconds','short_video_policy'],region:['mode','xyxy_normalized'],crop:['enabled','xyxy_normalized','fit','flip_x','flip_y']};
export function newControlRow(id){
    return {id,asset_id:null,enabled:true,control_net_name:'',control_kind:'depth',fit_mode:'center_crop',
        strength:.7,start_percent:0,end_percent:.75,stages:{low:true,high:true},
        source:{start_seconds:0,short_video_policy:'hold_last_frame'},
        region:{mode:'full',xyxy_normalized:[0,0,1,1]},
        crop:{enabled:false,xyxy_normalized:[0,0,1,1],fit:'contain',flip_x:false,flip_y:false}};
}
function jsonData(value){
    if(value===null||typeof value==='string'||typeof value==='boolean')return;
    if(typeof value==='number'){if(!Number.isFinite(value))throw Error('局部控制包含非有限数字，原草稿保留。');return;}
    if(Array.isArray(value)){value.forEach(jsonData);return;}
    if(object(value)&&[Object.prototype,null].includes(Object.getPrototypeOf(value))){Object.values(value).forEach(jsonData);return;}
    throw Error('局部控制只能包含 JSON 数据，原草稿保留。');
}
export function normalizeControls(plan){
    if(plan==null)return undefined;
    if(!object(plan)||plan.version!==1||!Array.isArray(plan.items))throw Error('局部控制版本或列表不支持，当前工程保留；请使用匹配版本或修正原文件。');
    jsonData(plan);
    const next=structuredClone(plan),ids=new Set();
    for(const row of next.items){
        if(!object(row)||!uuid(row.id)||ids.has(row.id)||(row.asset_id!=null&&!uuid(row.asset_id)))throw Error('局部控制条目身份无效或重复，当前工程保留。');
        ids.add(row.id);
        const defaults=newControlRow(row.id);
        for(const [key,value]of Object.entries(defaults))if(row[key]===undefined)row[key]=structuredClone(value);
        for(const key of Object.keys(nested)){
            if(!object(row[key]))throw Error('局部控制 '+key+' 需要对象，当前工程保留。');
            for(const [name,value]of Object.entries(defaults[key]))if(row[key][name]===undefined)row[key][name]=structuredClone(value);
        }
        const box=value=>Array.isArray(value)&&value.length===4&&value.every(finite);
        if(typeof row.enabled!=='boolean'||typeof row.control_net_name!=='string'
            ||!['depth','pose','canny','HED','MLSD','custom'].includes(row.control_kind)
            ||!['exact','center_crop','stretch'].includes(row.fit_mode)
            ||![row.strength,row.start_percent,row.end_percent,row.source.start_seconds].every(finite)
            ||typeof row.stages.low!=='boolean'||typeof row.stages.high!=='boolean'
            ||!['strict','hold_last_frame'].includes(row.source.short_video_policy)
            ||!['full','rect'].includes(row.region.mode)||!box(row.region.xyxy_normalized)
            ||!box(row.crop.xyxy_normalized)||!['exact','stretch','contain','cover'].includes(row.crop.fit)
            ||['enabled','flip_x','flip_y'].some(key=>typeof row.crop[key]!=='boolean'))throw Error('局部控制参数或坐标格式无效，当前工程保留。');
    }
    return next;
}
export function validateControlDraft(shot){normalizeControls(shot.controlRegions);}
export function validateControlProject(project){
    (project.doc?.shots||[]).forEach(validateControlDraft);
    for(const item of project.doc?.creationLibrary?.items||[])if(item.kind==='template')validateControlDraft(item.shot);
}
export function controlRows(shot,active=false,mode='single'){
    const rows=shot.controlRegions?.items;
    return (Array.isArray(rows)?rows:[]).filter(row=>!active||(row.enabled!==false&&(row.strength??.7)!==0
        &&(mode==='single'||(mode==='high_only'?row.stages?.high!==false:row.stages?.low!==false||row.stages?.high!==false))));
}
export const controlAssetIds=(shot,active=false,mode='single')=>controlRows(shot,active,mode).map(row=>row.asset_id).filter(Boolean);
export const controlSamplingMode=(doc,shot)=>{
    const value=shot.samplingInherit===false?shot.sampling:doc?.sampling;
    if(value?.mode==='hyperflow'&&(value.variant??'single8')==='single8')return 'single';
    if(value?.mode==='hyperflow'&&value.variant==='continuous4plus4separate'&&value.stage_checkpoint?.mode==='resume_tail')return 'high_only';
    return value?.mode||'single';
};
export function controlInputState(shot,mode='single'){
    const normalized=portableControls(shot.controlRegions);
    if(!normalized)return undefined;
    const rows=controlRows({controlRegions:normalized},true,mode);
    if(!rows.length)return undefined;
    return {version:1,items:rows.map(row=>{
        const result=Object.fromEntries(fields.filter(key=>key!=='id').map(key=>[key,structuredClone(row[key])]));
        delete result.enabled;
        if(mode==='single'||mode==='high_only')delete result.stages;
        return result;
    })};
}
export function portableControls(plan){
    const value=normalizeControls(plan);if(!value)return undefined;
    return {version:1,items:value.items.map(row=>{
        const clean=Object.fromEntries(fields.map(key=>[key,structuredClone(row[key])]));
        for(const [key,names]of Object.entries(nested))clean[key]=Object.fromEntries(names.map(name=>[name,structuredClone(row[key][name])]));
        return clean;
    })};
}
export function duplicateControlRows(shot,makeUUID){for(const row of controlRows(shot))row.id=makeUUID();}
export function remapControlAsset(shot,oldId,newId){for(const row of controlRows(shot))if(row.asset_id===oldId)row.asset_id=newId;}
export function removeControlAsset(shot,id){remapControlAsset(shot,id,null);}
export function requiresControlProtocol(project,shotIds){
    return (project.doc?.shots||[]).some(shot=>(!shotIds||shotIds.includes(shot.id))&&controlRows(shot,true,controlSamplingMode(project.doc,shot)).length);
}
export function assertControlProtocol(project,report,shotIds){
    if(requiresControlProtocol(project,shotIds)&&report.extensions?.control_regions_v1!==true)
        throw Error('当前 Core 未确认局部控制协议，未提交生成。请保存项目，待现有任务结束后更新并重启 Core；控制图、区域与裁框草稿保留。');
}
export function controlDraftIssues(doc,shot,media){
    const issues=[];try{validateControlDraft(shot);}catch(error){return [error.message];}
    const active=controlRows(shot,true,controlSamplingMode(doc,shot));
    for(const [index,raw]of active.entries()){
        const row=normalizeControls({version:1,items:[raw]}).items[0],label=`局部控制第 ${index+1} 项：`,asset=media.get(row.asset_id);
        if(!row.control_net_name.trim())issues.push(label+'请选择 Fun Control 模型；可以填写自选文件名。');
        if(!asset||asset.missing||!['image','video'].includes(asset.kind))issues.push(label+'请选择已上传且已预处理的控制图片／视频。');
        if(!(0<=row.start_percent&&row.start_percent<=row.end_percent&&row.end_percent<=1))issues.push(label+'采样时间窗需要 0≤开始≤结束≤1。');
        if(row.source.start_seconds<0)issues.push(label+'控制视频开始时间不能为负。');
        for(const [key,enabled]of [['region',row.region.mode==='rect'],['crop',row.crop.enabled]])if(enabled){
            const box=row[key].xyxy_normalized;
            if(!(box[2]>box[0]&&box[3]>box[1]&&Number.isFinite(box[2]-box[0])&&Number.isFinite(box[3]-box[1])))issues.push(label+(key==='region'?'目标区域':'源裁框')+'需要有限的正宽高。');
            if(key==='region'&&box.some(value=>value<0||value>1))issues.push(label+'目标区域应位于画布 0–1 内；源裁框才可超出源图。');
        }
    }
    return issues;
}

// Geometry is UI-only. Applying a drag changes no model input until pointerup.
export function containedRect(view,width,height){
    const scale=Math.min(view.width/width,view.height/height),w=width*scale,h=height*scale;
    return {left:view.left+(view.width-w)/2,top:view.top+(view.height-h)/2,width:w,height:h};
}
export function dragControlBox(box,start,point,action='draw',handle='',bounded=true){
    const clamp=value=>bounded?Math.max(0,Math.min(1,value)):value;
    const p={x:clamp(point.x),y:clamp(point.y)},dx=p.x-start.x,dy=p.y-start.y;
    let next;
    if(action==='move'){
        const x=bounded?Math.max(-box[0],Math.min(1-box[2],dx)):dx,y=bounded?Math.max(-box[1],Math.min(1-box[3],dy)):dy;
        next=[box[0]+x,box[1]+y,box[2]+x,box[3]+y];
    }else if(action==='resize'){
        next=[...box];if(handle.includes('w'))next[0]=p.x;if(handle.includes('e'))next[2]=p.x;
        if(handle.includes('n'))next[1]=p.y;if(handle.includes('s'))next[3]=p.y;
    }else next=[start.x,start.y,p.x,p.y];
    return [Math.min(next[0],next[2]),Math.min(next[1],next[3]),Math.max(next[0],next[2]),Math.max(next[1],next[3])];
}
