import test from 'node:test';
import assert from 'node:assert/strict';
import {candidateDiagnosticText,configurationEvidenceText} from '../web/director/candidate_diagnostics.mjs';

test('candidate explanation distinguishes stale components, timing plan and cache proof',()=>{
    const report = {candidate:{status:'stale',changed_components:['assets','references']},
        time_plan:{requested_seconds:4,requested_frames:96,aligned_frames:107,generated_seconds:107/24,fps:24,delivery_trim_frames:96},
        reference_slots:[{asset_id:'video-id',role:'ref_video',native:'<Video 1>'},
            {asset_id:'video-id',role:'ref_video_audio',native:'<Audio 1>'}],
        reference_aliases:[{asset_id:'video-id',alias:'@ref1',native:'<Video 1>'}]};
    const before = structuredClone(report), text = candidateDiagnosticText(report);
    assert.match(text,/候选已失效/); assert.match(text,/素材的内容 SHA；参考绑定／顺序/);
    assert.match(text,/请求 4 秒／96 帧；对齐 107 帧／4.458333 秒/);
    assert.match(text,/<Video 1> · ref_video · 素材 video-id · @ref1/);
    assert.match(text,/<Audio 1> · ref_video_audio · 素材 video-id\n/);
    assert.match(text,/不是实测源 PTS/); assert.match(text,/不是 MODEL／Stage 缓存命中证明/);
    assert.deepEqual(report,before);
});
test('legacy and inconsistent receipts never invent exact change attribution',()=>{
    assert.match(candidateDiagnosticText({status:'stale',changed_components:null,component_snapshot:'legacy_missing'}),/不猜历史差异/);
    assert.match(candidateDiagnosticText({status:'active',changed_components:null,component_snapshot:'inconsistent_receipt_declaration'}),/不能准确归因/);
});
test('arbitrary strings stay literal, not executed or inserted as HTML',()=>{
    const text = candidateDiagnosticText({candidate:{status:'pending',changed_components:[]},
        reference_slots:[{native:'<script>not executable</script>',role:'ref_image',asset_id:'safe-id'}]});
    assert.match(text,/<script>not executable<\/script>/);
});

test('configuration summary separates current draft, frozen graph and observed output',()=>{
    const report={effective_config:{scope:'draft_plan_not_runtime_evidence',shot_id:'s',prompt:'new text',sampling:{mode:'two_pass',low_loras:[{name:'a',strength:1,file_alpha_policy:'metadata_alpha_per_rank'}],high_loras:[{name:'b',strength:-.2}]},generation:{unet:'auto'},references:[],time:{requested_frames:96,aligned_frames:107,delivery_trim_frames:96,fps:24},canvas:{width:832,height:480}}};
    const frozen={complete:true,sha256:'f'.repeat(64),snapshot:{shot_id:'s',seed:42,recipe:'frozen-route',project:{doc:{generation:{unet:'auto'}}},report:{shots:[{id:'s',prompt:'old text',sampling:{mode:'single'}}]},prompt:{'1':{class_type:'UNETLoader',inputs:{unet_name:'actual.safetensors'}},'6':{class_type:'MiniMaxH3DualClockSamplerT8',inputs:{steps:8}}},sampling:{total_nfe:999}}};
    const completed={state:'success',request_id:'r',outputs:{save:{images:[{filename:'movie.mp4',type:'output'}]}}};
    const input={draft:report,frozen,completed},before=structuredClone(input),text=configurationEvidenceText(input);
    for(const phrase of ['当前草稿','已提交冻结配置','已观察结果','new text','old text','actual.safetensors','movie.mp4','metadata_alpha_per_rank'])assert.ok(text.includes(phrase),phrase);
    assert.match(text,/实际 NFE.*未知/);assert.doesNotMatch(text,/实际 NFE[^\n]*999/);assert.match(text,/未认证.*缓存/);
    assert.deepEqual(input,before);
});
test('legacy and external completion do not acquire invented frozen model or sampling evidence',()=>{
    const text=configurationEvidenceText({completed:{origin:'external',state:'success',external_take_id:'take',media_sha256:'a'.repeat(64)}});
    assert.match(text,/外部成片/);assert.match(text,/没有可还原/);assert.match(text,/实际 NFE.*未知/);
});
test('frozen summary shows real shift keys and a legacy LoRA declaration without inventing auto mode',()=>{
    const text=configurationEvidenceText({frozen:{complete:true,snapshot:{shot_id:'s',project:{doc:{generation:{lora:['old.safetensors'],lora_strength:.3}}},report:{shots:[{id:'s',sampling:{mode:'single'}}]},prompt:{'6':{class_type:'MiniMaxH3DualClockSamplerT8',inputs:{shift_video:3,shift_audio:4,scheduler:'euler',steps:8}}}}}});
    assert.match(text,/旧格式 LoRA 配置/);assert.match(text,/old.safetensors/);assert.doesNotMatch(text,/单采 LoRA 模式：auto/);
    assert.match(text,/"shift_video":3/);assert.match(text,/"shift_audio":4/);assert.match(text,/"scheduler":"euler"/);
});
test('one layer failure remains explicit and never erases the other layer evidence',()=>{
    const text=configurationEvidenceText({draft:{effective_config:{prompt:'valid current draft'}},frozen_error:'快照 SHA 校验失败',completed:{state:'success',prompt_id:'saved-result'}});
    assert.match(text,/valid current draft/);assert.match(text,/冻结快照读取失败：快照 SHA 校验失败/);assert.match(text,/saved-result/);
    assert.doesNotMatch(text,/没有可还原的完整生成快照/);
});

test('TAIL resume labels HEAD rows as inactive drafts, never as executed loaders',()=>{
    const sampling={mode:'hyperflow',variant:'continuous4plus4',low_loras:[{name:'head.safetensors'}],
        high_loras:[{name:'tail.safetensors'}],stage_checkpoint:{mode:'resume_tail',artifact_path:'saved-head',artifact_sha256:'a'.repeat(64)}};
    const text=configurationEvidenceText({draft:{effective_config:{sampling}}});
    assert.match(text,/LOW \/ HEAD：非活动草稿，不执行/);
    assert.doesNotMatch(text,/LOW \/ HEAD：按顺序加载/);
    assert.match(text,/HIGH \/ TAIL：按顺序加载/);
    assert.match(text,/head.safetensors/);
});

for(const damage of ['none','missing','duplicate','bad-json','schema','name','strength','policy','version',
    'fake-disabled','missing-names','bad-inputs','disabled','enabled-status-for-disabled'])test('per-loader observed evidence: '+damage,()=>{
    const binding={node_id:'14',stage:'LOW',row_index:2,row_id:'stable-row'};
    const inputs={lora_name:'directory/user.safetensors',strength_model:3,file_alpha_policy:'metadata_alpha_per_rank'};
    const report={schema:'t8.minimax_h3.lora_compat.v1',status:'applied',selected_strength:3,
        alpha_interpretation:{requested_policy:'metadata_alpha_per_rank'},
        prefix_interpretation:{requested_policy:'legacy'},applied_patch_count:1,missed_patch_target_count:0};
    const envelope={selected_lora_name:inputs.lora_name,report_json:JSON.stringify(report)};
    const frozen={complete:true,request_id:'request-a',prompt_id:'prompt-a',snapshot:{
        prompt:{'14':{class_type:'MiniMaxH3LoRACompatibilityLoaderT8Advanced',inputs}},
        report:{lora_loaders:[binding]}}};
    const completed={state:'error',request_id:'request-a',prompt_id:'prompt-a',outputs:{'14':{t8_lora_loader:[envelope]}}};
    if(damage==='missing')completed.outputs={};
    if(damage==='duplicate')completed.outputs['14'].t8_lora_loader.push({...envelope});
    if(damage==='bad-json')envelope.report_json='{bad';
    if(damage==='schema')envelope.report_json=JSON.stringify({...report,schema:'invented'});
    if(damage==='name')envelope.selected_lora_name='other/user.safetensors';
    if(damage==='strength')envelope.report_json=JSON.stringify({...report,selected_strength:2});
    if(damage==='policy')envelope.report_json=JSON.stringify({...report,prefix_interpretation:{requested_policy:'unique_double_prefix'}});
    if(damage==='version')completed.request_id='request-b';
    if(damage==='fake-disabled')envelope.report_json=JSON.stringify({...report,status:'disabled'});
    if(damage==='missing-names'){delete inputs.lora_name;delete envelope.selected_lora_name;}
    if(damage==='bad-inputs')frozen.snapshot.prompt['14'].inputs=[];
    if(damage==='disabled'||damage==='enabled-status-for-disabled'){
        inputs.lora_name=envelope.selected_lora_name='disabled';
        if(damage==='disabled')envelope.report_json=JSON.stringify({...report,status:'disabled',
            alpha_interpretation:undefined,prefix_interpretation:undefined});
    }
    const before=structuredClone({frozen,completed});
    const text=configurationEvidenceText({frozen,completed});
    if(damage==='none'||damage==='disabled'){
        assert.match(text,/LOW.*第 2 行.*#14/);
        assert.match(text,/观察到.*加载报告/);
        assert.match(text,/注册 1/);
    }else if(damage==='missing')assert.match(text,/未观察到.*加载报告/);
    else if(damage==='version')assert.match(text,/版本不匹配/);
    else assert.match(text,/报告.*(?:不匹配|损坏|不唯一)/);
    assert.match(text,/不证明.*本次.*重读/);
    assert.match(text,/实际 NFE.*未知/);
    assert.deepEqual({frozen,completed},before);
});
test('old snapshots do not gain stage/row evidence from matching filenames',()=>{
    const text=configurationEvidenceText({frozen:{complete:true,snapshot:{prompt:{},report:{}}},
        completed:{state:'success',request_id:'r',outputs:{'14':{t8_lora_loader:[]}}}});
    assert.match(text,/无逐.*LoRA.*证据/);
});

for(const damage of ['shots-object','shots-null-row','prompt-null-node','references-object','references-null-row','loras-object','bindings-null-row',
    'binding-object-id','class-object','mixed-loras'])test('damaged frozen '+damage+' does not erase other evidence layers',()=>{
    const selected={id:'s',prompt:'frozen text',sampling:{mode:'single'},media_map:[]};
    const frozen={complete:true,request_id:'r',prompt_id:'p',snapshot:{shot_id:'s',
        project:{doc:{generation:{}}},report:{shots:[selected],lora_loaders:[]},prompt:{}}};
    if(damage==='shots-object')frozen.snapshot.report.shots={};
    if(damage==='shots-null-row')frozen.snapshot.report.shots=[null,selected];
    if(damage==='prompt-null-node')frozen.snapshot.prompt={'14':null};
    if(damage==='references-object')selected.media_map={};
    if(damage==='references-null-row')selected.media_map=[null];
    if(damage==='loras-object')selected.sampling.loras={};
    if(damage==='bindings-null-row')frozen.snapshot.report.lora_loaders=[null];
    if(damage==='binding-object-id')frozen.snapshot.report.lora_loaders=[{node_id:{toString:null},stage:'LOW',row_index:1}];
    if(damage==='class-object')frozen.snapshot.prompt={'14':{class_type:{toString:null},inputs:{}}};
    if(damage==='mixed-loras')selected.sampling.loras=[{name:'kept.safetensors',enabled:false},null,{name:'active.safetensors',strength:0}];
    const completed={state:'success',request_id:'r',prompt_id:'p',outputs:{save:{images:[{filename:'kept.mp4'}]}}};
    const input={draft:{effective_config:{prompt:'healthy draft'}},frozen,completed},before=structuredClone(input);
    let text;
    assert.doesNotThrow(()=>{text=configurationEvidenceText(input);});
    for(const phrase of ['healthy draft','已提交冻结配置','已观察结果','kept.mp4'])assert.ok(text.includes(phrase),phrase);
    if(damage==='mixed-loras'){
        assert.match(text,/3\. active.safetensors × 0/);
        assert.match(text,/第 2 行 LoRA 损坏/);
    }
    assert.deepEqual(input,before);
});
