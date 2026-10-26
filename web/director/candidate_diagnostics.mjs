// Text only: no DOM, requests, mutable project state or runtime cache inference.
const components = {schema:'输入格式',global:'全片设定',draft:'本镜原作者稿／用途',
    references:'参考绑定／顺序',assets:'被引用素材的内容 SHA',evidence:'源证据／确认',
    intent:'创作意图',dependencies:'明确的跨镜依赖',bindings:'有效规则／顺序',options:'候选编译选项'};
const states = {legacy:'无候选，仍用原作者稿',pending:'待审，尚未采用',active:'已显式采用，输入仍一致',
    stale:'候选已失效，请核对后重新编译'};
const seconds = value => Number.isFinite(value) ? String(Number(value.toFixed(6))) : '未知';

export function candidateDiagnosticText(report) {
    const candidate = report.candidate || report;
    const lines = ['候选：' + (states[candidate.status] || '状态未知')];
    if (Array.isArray(candidate.changed_components)) {
        lines.push(candidate.changed_components.length ? '改变的输入：' + candidate.changed_components.map(key => components[key] || key).join('；') : '已记录的输入未变化。');
    } else if (candidate.component_snapshot === 'inconsistent_receipt_declaration') {
        lines.push('总签名与分字段声明不一致，不能准确归因；不自动修复或重签。');
    } else {
        lines.push('旧候选没有分字段快照，不猜历史差异。');
    }
    const timing = report.time_plan;
    if (timing) {
        lines.push(`计划时长：请求 ${seconds(timing.requested_seconds)} 秒／${timing.requested_frames} 帧；对齐 ${timing.aligned_frames} 帧／${seconds(timing.generated_seconds)} 秒（${timing.fps} fps）；交付裁齐 ${timing.delivery_trim_frames} 帧。`);
        lines.push('以上来自当前草稿编译计划，不是实测源 PTS 或生成结果。');
    }
    if (Array.isArray(report.reference_slots)) {
        lines.push('生效参考编号（计划）：');
        if (!report.reference_slots.length) lines.push('本镜没有生效参考槽。');
        for (const row of report.reference_slots) {
            const aliases = (report.reference_aliases || []).filter(item => item.asset_id === row.asset_id && item.native === row.native).map(item => item.alias);
            lines.push(`${row.native || '未知编号'} · ${row.role || '未知用途'} · 素材 ${row.asset_id || '未知'}${aliases.length ? ' · ' + aliases.join(' / ') : ''}`);
        }
    }
    lines.push('本次只读检查不执行采用、保存或排队；这不是 MODEL／Stage 缓存命中证明。');
    return lines.join('\n');
}

const literal = value => value == null ? '未知' : typeof value === 'string' ? value : JSON.stringify(value);
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const loraText = (rows, label, inactive=false) => {
    if(rows!=null&&!Array.isArray(rows))return [label+'：LoRA 列表损坏，未推断加载状态。'];
    const entries=(rows||[]).map((row,index)=>({row,index}));
    const active=entries.filter(({row})=>record(row)&&row.enabled!==false);
    const damaged=entries.filter(({row})=>!record(row));
    return [label+'：'+(inactive?'非活动草稿，不执行':active.length?'按顺序加载':damaged.length?'配置损坏，待核对':'无手动启用项'),
        ...active.map(({row,index})=>`  ${index+1}. ${row.name} × ${row.strength??1} · alpha=${row.file_alpha_policy??'legacy_tensor_only'} · prefix=${row.prefix_policy??'legacy'}`),
        ...damaged.map(({index})=>`  第 ${index+1} 行 LoRA 损坏，其它有效行保留。`)];
};
function planText(config) {
    if(config==null)return ['尚未读取当前编译计划。'];
    if(!record(config))return ['编译计划格式损坏，未推断配置；其它证据层保留。'];
    const lines=[`画面 ${literal(config.task_type)}；声音 ${literal(config.audio_mode)}；交付主轨 ${literal(config.delivery_audio)}`];
    if(config.canvas)lines.push(`计划尺寸 ${config.canvas.width} × ${config.canvas.height}；目标 ${config.canvas.requested_megapixels} MP。`);
    if(config.time)lines.push(`帧数 请求 ${config.time.requested_frames} → 对齐 ${config.time.aligned_frames} → 交付 ${config.time.delivery_trim_frames}（${config.time.fps} fps）。`);
    if(config.generation!=null&&!record(config.generation))lines.push('模型配置损坏，未还原具体选择。');
    if(config.sampling!=null&&!record(config.sampling))lines.push('采样配置损坏，未还原具体选择。');
    const g=record(config.generation)?config.generation:{},s=record(config.sampling)?config.sampling:{mode:config.sampling==null?'single':'未知'};
    lines.push(`共用底模 ${g.unet||'auto'}；文本 ${g.clip||'auto'}；视频 VAE ${g.video_vae||'auto'}；音频 VAE ${g.audio_vae||'auto'}。`);
    lines.push(`采样计划 ${s.mode}${s.variant?' / '+s.variant:s.preset?' / '+s.preset:''}；auto 文件在提交时解析，不是已加载证明。`);
    if(s.mode==='single'){
        if(s.loras==null&&g.loras==null&&g.lora!==undefined)lines.push('旧格式 LoRA 配置：'+literal({lora:g.lora,lora_strength:g.lora_strength})+'；解析以冻结加载节点为准，不猜缺失的现代行设置。');
        else{
            const mode=s.lora_mode??g.lora_mode??'auto';lines.push('单采 LoRA 模式：'+mode);
            if(mode!=='none')lines.push(...loraText(s.loras??g.loras,'单采'));
        }
    }else{
        lines.push(...loraText(s.low_loras,'LOW / HEAD',s.mode==='hyperflow'&&s.stage_checkpoint?.mode==='resume_tail'));
        if(!(s.mode==='hyperflow'&&s.variant==='single8'))lines.push(...loraText(s.high_loras,'HIGH / TAIL'));
        if(s.stage_checkpoint?.mode==='resume_tail')lines.push('仅恢复 TAIL；LOW / HEAD 草稿不执行；HEAD 制品 '+literal(s.stage_checkpoint));
    }
    if(config.audio_selection)lines.push('声音来源与入出点：'+literal(config.audio_selection));
    lines.push('生效参考声明（本检查未读取文件校验当前内容）：');
    if(config.references!=null&&!Array.isArray(config.references))lines.push('参考列表损坏，未推断参考绑定。');
    for(const row of Array.isArray(config.references)?config.references:[]){
        if(!record(row)){lines.push('  参考行损坏，其它有效参考保留。');continue;}
        lines.push(`  ${row.native||'无原生编号'} · ${row.role} · ${row.asset_id} · 声明 SHA ${row.sha256||'未知'}`);
    }
    lines.push('有效提示词：\n'+(config.prompt??'未知'));
    if(config.events?.length)lines.push('局部事件／计划时间：'+literal(config.events));
    return lines;
}
function loraEvidenceText(frozen, completed) {
    const bindings=frozen?.snapshot?.report?.lora_loaders;
    if(!frozen?.complete||!Array.isArray(bindings))return ['无逐条 LoRA 加载证据：旧快照或外部成片未保存阶段／行映射。'];
    if(!bindings.length)return ['冻结配置没有内容 LoRA 加载节点。'];
    if(!completed||!frozen.request_id||!frozen.prompt_id||!completed.request_id||!completed.prompt_id)
        return ['无逐条 LoRA 加载证据：缺少可配对的版本身份。'];
    if(frozen.request_id!==completed.request_id||frozen.prompt_id!==completed.prompt_id)
        return ['逐条 LoRA 报告版本不匹配；没有使用另一版本的输出。'];
    const lines=['逐条 LoRA 加载报告（所选冻结版本）'];
    const counts=new Map();
    for(const row of bindings)counts.set(row?.node_id,(counts.get(row?.node_id)||0)+1);
    for(const binding of bindings){
        const id=binding?.node_id;
        if(typeof id!=='string'||!id){lines.push('LoRA 报告节点绑定损坏，缺少有效节点 ID。');continue;}
        const node=frozen.snapshot.prompt?.[id],input=node?.inputs;
        const label=`${literal(binding?.stage)||'未知阶段'} 第 ${literal(binding?.row_index)} 行 #${id} · ${literal(input?.lora_name)}`;
        if(counts.get(id)!==1||!['SINGLE','LOW','HIGH'].includes(binding?.stage)||!Number.isInteger(binding?.row_index)||binding.row_index<1||
            node?.class_type!=='MiniMaxH3LoRACompatibilityLoaderT8Advanced'||
            !record(input)||typeof input.lora_name!=='string'||!input.lora_name){
            lines.push(label+'：报告节点绑定不匹配或不唯一。');continue;
        }
        const entries=completed.outputs?.[id]?.t8_lora_loader;
        if(entries===undefined||(Array.isArray(entries)&&!entries.length)){
            lines.push(label+'：未观察到加载报告；不猜测已加载或未加载。');continue;
        }
        if(!Array.isArray(entries)||entries.length!==1){
            lines.push(label+'：加载报告不唯一或损坏。');continue;
        }
        const envelope=entries[0];
        let report;
        try{
            if(typeof envelope?.report_json!=='string')throw new Error('report is not text');
            report=JSON.parse(envelope.report_json);
        }catch{
            lines.push(label+'：加载报告损坏，未影响成片播放。');continue;
        }
        const disabled=input.lora_name==='disabled';
        const policies=disabled||
            (report?.alpha_interpretation?.requested_policy===(input.file_alpha_policy??'legacy_tensor_only')&&
             report?.prefix_interpretation?.requested_policy===(input.prefix_policy??'legacy'));
        if(report?.schema!=='t8.minimax_h3.lora_compat.v1'||!['applied','no_compatible_patches','disabled'].includes(report.status)||
            (report.status==='disabled')!==disabled||
            typeof envelope.selected_lora_name!=='string'||!envelope.selected_lora_name||
            envelope.selected_lora_name!==input.lora_name||!Number.isFinite(report.selected_strength)||
            report.selected_strength!==(input.strength_model??1)||!policies){
            lines.push(label+'：加载报告与冻结名称／强度／解析策略不匹配。');continue;
        }
        const count=Number.isInteger(report.applied_patch_count)&&report.applied_patch_count>=0?report.applied_patch_count:'未知';
        const missed=Number.isInteger(report.missed_patch_target_count)&&report.missed_patch_target_count>=0?report.missed_patch_target_count:'未知';
        lines.push(label+`：观察到节点加载报告；状态 ${report.status}，报告注册 ${count}，未注册 ${missed}。`);
    }
    return lines;
}

export function configurationEvidenceText({draft,frozen,completed,draft_error,frozen_error}={}) {
    const lines=['当前草稿编译计划',...planText(draft?.effective_config)];
    if(draft_error)lines.push('草稿读取失败：'+draft_error+'；未伪造计划，其它层单独保留。');
    if(draft?.compilation_errors?.length)lines.push('当前预检未通过：'+draft.compilation_errors.map(row=>row.message).join('；'));
    lines.push('\n已提交冻结配置');
    if(frozen?.complete&&frozen.snapshot){
        const source=frozen.snapshot,shots=source.report?.shots;
        const selected=Array.isArray(shots)?shots.find(row=>record(row)&&row.id===source.shot_id):undefined;
        lines.push(`快照 SHA ${frozen.sha256}；种子 ${literal(source.seed)}；配方 ${literal(source.recipe)}。`);
        if(shots!=null&&(!Array.isArray(shots)||shots.some(row=>!record(row))))lines.push('冻结镜头列表损坏，未猜测缺失行；有效配置及结果仍保留。');
        if(selected)lines.push(...planText({...selected,generation:source.project?.doc?.generation,references:selected.media_map}));
        if(source.prompt!=null&&!record(source.prompt))lines.push('冻结图格式损坏，未推断节点。');
        for(const [id,node] of Object.entries(record(source.prompt)?source.prompt:{})){
            if(!record(node)||!record(node.inputs)||typeof node.class_type!=='string'){lines.push(`冻结图 #${id} 节点／输入损坏，其它节点及结果仍保留。`);continue;}
            const input=node.inputs;
            if(['UNETLoader','CLIPLoader','VAELoader','MiniMaxH3LoRACompatibilityLoaderT8Advanced'].includes(node.class_type)){
                const fields=Object.fromEntries(Object.entries(input).filter(([key])=>['unet_name','clip_name','vae_name','lora_name','strength_model','file_alpha_policy','prefix_policy'].includes(key)));
                lines.push(`冻结图 #${id} ${node.class_type}：${literal(fields)}`);
            }
            if(/Sampler|Scheduler|Sigmas|Clock|HyperFlow.*Plan/.test(node.class_type)){
                const fields=Object.fromEntries(Object.entries(input).filter(([key])=>['sampler_name','scheduler','steps','shift_video','shift_audio','video_shift','audio_shift','cfg','raw_sigmas_json','sigmas_json','split_interval'].includes(key)));
                if(Object.keys(fields).length)lines.push(`冻结采样参数 #${id} ${node.class_type}：${literal(fields)}（不是运行计数）`);
            }
        }
    }else if(frozen_error)lines.push('冻结快照读取失败：'+frozen_error+'；未认证／未还原，不等同旧版缺少快照。');
    else lines.push(completed?.origin==='external'?'外部成片：没有可还原的原始模型／采样快照。':frozen?.reason||'没有可还原的完整生成快照；不从当前草稿猜测历史设置。');
    lines.push('\n已观察结果');
    if(completed){
        lines.push(`记录状态 ${literal(completed.state)}；版本 ${literal(completed.request_id||completed.external_take_id||completed.prompt_id)}。`);
        if(completed.integrity_error)lines.push('需核对：'+completed.integrity_error);
        for(const output of Object.values(completed.outputs||{}))for(const item of Array.isArray(output?.images)?output.images:[])
            if(item?.filename)lines.push('Core 输出记录：'+item.filename+'（此摘要未重新校验文件）');
        if(completed.origin==='external')lines.push('外部成片 SHA 声明：'+literal(completed.media_sha256)+'；不证明采样祖先。');
    }else lines.push('当前尚未选择结果版本。');
    lines.push(...loraEvidenceText(frozen,completed));
    lines.push('加载报告可能来自 Core 缓存重放；不证明本次文件重读、该分支实际前向、文件 SHA 或最终效果。');
    lines.push('实际 NFE／委托次数：未知；计划 steps／total_nfe 不替代实际运行报告。');
    lines.push('未认证 MODEL／Stage 缓存命中或 LOW 可复用；无阶段制品证据时不承诺复用。');
    return lines.join('\n');
}
