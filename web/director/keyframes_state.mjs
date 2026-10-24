// Pure project state; no upload, graph, model or DOM mutation.
export const middleRows = (shot, active=false) => (shot.middleKeyframes?.items || []).filter(row=>!active || row.enabled!==false);
export const middleAssetIds = (shot, active=false) => middleRows(shot,active).map(row=>row.asset_id).filter(Boolean);
export function validateMiddleDraft(shot){
    const plan=shot.middleKeyframes;if(plan==null)return;
    const object=value=>value&&typeof value==='object'&&!Array.isArray(value);
    const uuid=value=>typeof value==='string'&&/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(value);
    if(!object(plan)||plan.version!==1||!Array.isArray(plan.items)
        ||!['visible_delivery','generated_canvas'].includes(plan.time_basis===undefined?'visible_delivery':plan.time_basis))
        throw Error('中间关键帧版本或列表结构不支持，当前工程保留；请使用匹配版本或修正原文件。');
    const ids=new Set();
    for(const row of plan.items){
        const aug=row?.visual_noise_aug===undefined?.999:row.visual_noise_aug;
        if(!object(row)||!uuid(row.id)||ids.has(row.id)||(row.asset_id!=null&&!uuid(row.asset_id))
            ||!['frame','seconds','percent'].includes(row.position_mode)||typeof row.position!=='number'||!Number.isFinite(row.position)
            ||(row.enabled!==undefined&&typeof row.enabled!=='boolean')
            ||typeof aug!=='number'||!Number.isFinite(aug)||aug<0||aug>1
            ||!['center_crop','stretch'].includes(row.resize_mode===undefined?'center_crop':row.resize_mode))
            throw Error('中间关键帧条目身份或参数无效，当前工程保留；请修正原文件。');
        ids.add(row.id);
    }
}
export function requiresMiddleProtocol(project,shotIds){
    return (project.doc?.shots||[]).some(s=>(!shotIds||shotIds.includes(s.id))&&middleRows(s,true).length);
}
export function assertMiddleProtocol(project,report,shotIds){
    if(requiresMiddleProtocol(project,shotIds)&&report.extensions?.middle_keyframes!==1)
        throw Error('当前 Core 未确认中间关键帧协议，未提交生成。请保存项目，待现有任务结束后更新并重启 Core；图片与位置草稿保留。');
}
export function remapMiddleAsset(shot,oldId,newId){
    for(const row of middleRows(shot))if(row.asset_id===oldId)row.asset_id=newId;
}
export function duplicateMiddleRows(shot,uuid){
    for(const row of middleRows(shot))row.id=uuid();
}
export function removeMiddleAsset(shot,id){
    if(shot.middleKeyframes)shot.middleKeyframes.items=middleRows(shot).filter(row=>row.asset_id!==id);
}
export function middleFrameClock(shot){
    let duration=shot.manualDuration;
    const events=shot.writingMode==='advanced'?(shot.events||[]):[];
    if(shot.autoDuration){
        if(events.length)duration=Math.max(...events.map(row=>row.end));
        else if(shot.sound==='record'&&shot.audio)duration=shot.end-shot.start;
    }
    const visible=Math.ceil(duration*24),requested=Math.max(5,visible);
    if(!Number.isFinite(visible)||visible<1)throw Error('镜头时长无法映射到有限的24fps帧轴');
    return {visible,generated:requested+((5-requested)%17+17)%17,duration};
}
export function resolveMiddlePosition(shot,row){
    const clock=middleFrameClock(shot),basis=shot.middleKeyframes?.time_basis||'visible_delivery';
    if(!['visible_delivery','generated_canvas'].includes(basis))throw Error('未知关键帧时轴');
    const count=basis==='visible_delivery'?clock.visible:clock.generated,value=row.position;
    if(typeof value!=='number'||!Number.isFinite(value))throw Error('请输入有限的位置数字');
    let frame;
    if(row.position_mode==='frame'){
        frame=Math.round(value);
        if(Math.abs(frame-value)>1e-9||!Number.isSafeInteger(frame))throw Error('帧号必须是可精确表达的整数');
    }else if(row.position_mode==='seconds'){
        if(value<0)throw Error('秒数不能为负数');
        frame=Math.floor(value*24+.5);
    }else if(row.position_mode==='percent'){
        if(value<0||value>100)throw Error('百分比必须在0到100之间');
        frame=Math.floor(value/100*(count-1)+.5);
    }else throw Error('未知定位方式');
    if(!Number.isSafeInteger(frame)||frame<=0||frame>=count-1)throw Error(`中间帧须在1到${count-2}，当前为${frame}`);
    return {...clock,basis,frame,seconds:frame/24,visible_in_delivery:frame<clock.visible};
}
export function middleInputState(shot){
    const rows=middleRows(shot,true);
    if(!rows.length)return undefined;
    return {version:shot.middleKeyframes.version,time_basis:shot.middleKeyframes.time_basis||'visible_delivery',
        items:rows.map(row=>{
            let frame;
            try{frame=resolveMiddlePosition(shot,row).frame;}catch{frame={mode:row.position_mode,value:row.position};}
            return {asset_id:row.asset_id,frame,visual_noise_aug:row.visual_noise_aug??.999,resize_mode:row.resize_mode||'center_crop'};
        }).sort((a,b)=>typeof a.frame==='number'&&typeof b.frame==='number'?a.frame-b.frame:JSON.stringify(a).localeCompare(JSON.stringify(b)))};
}
