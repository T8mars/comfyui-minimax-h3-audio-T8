import test from 'node:test';
import {captureCreation,addCreation,renameCreation,planCreation,creationSampling,remapCreationAsset} from '../web/director/creation_library.mjs';
function creativeFixture(){
    const shot={id:'shot',name:'本镜',mode:'first',sound:'native',writingMode:'simple',simplePrompt:'原文 @image2',prompt:'高级 @image1',simpleInitialized:true,advancedInitialized:true,events:[{id:'event',start:0,end:2,text:'动作 @image2'}],refs:['a'],tray:['a','b'],first:'b',last:null,audio:null,start:0,end:0,duration:4,manualDuration:4,autoDuration:true,ratio:'16:9',ownRatio:'16:9',samplingInherit:true,d3Inherit:true,seed:42,rev:1,adoptedResultId:'accepted'};
    const doc={shots:[shot],global:'全片',sharedRefs:[],sharedRatio:false,ratio:'16:9',sampling:{mode:'two_pass',output_mp:'0.6',low_loras:[{name:'low.safetensors',strength:1,enabled:true},{name:'extra.safetensors',strength:.4,enabled:true}],high_loras:[{name:'high.safetensors',strength:.5,enabled:true}]},generation:{unet:'shared-base'},d3:{memory:{chunk_ffn:true}}};
    return {doc,shot,assets:[{id:'a',kind:'image'},{id:'b',kind:'image'},{id:'c',kind:'image'},{id:'audio',kind:'audio'}],d3:doc.d3,name:'我的收藏',id:'collection',target:{kind:'simplePrompt',shot:shot.id,start:0,end:0}};
}
test('text templates do not activate inactive local reference drafts',()=>{
    const ctx=creativeFixture();ctx.shot.mode='text';
    const item=captureCreation({...ctx,kind:'template'});
    assert.equal(item.shot.mode,'text');assert.deepEqual(item.shot.refs,[]);
    assert.deepEqual(item.shot.tray,['a','b']);
    ctx.doc.sharedRefs=['c'];
    const shared=captureCreation({...ctx,kind:'template'});
    assert.equal(shared.shot.mode,'refs');assert.deepEqual(shared.shot.refs,['c']);
});
test('group addition remaps known aliases but preserves native tags and unknown original text',()=>{
    const ctx=creativeFixture();ctx.shot.tray=[];ctx.shot.refs=[];
    ctx.shot.simplePrompt='@image1 <Picture 1> @image99';
    const plan=planCreation({...ctx,shotId:'shot',item:{kind:'group',asset_ids:['c']}});
    assert.equal(plan.doc.shots[0].simplePrompt,'@image2 <Picture 1> @image99');
});
test('collection sampling matches backend explicit and legacy LoRA defaults',()=>{
    const ctx=creativeFixture();delete ctx.doc.sampling;
    for(const lora of ['none',['none'],'auto']){
        ctx.doc.generation={lora};assert.equal(creationSampling(ctx.doc,ctx.shot).lora_mode,'auto');
    }
    ctx.doc.generation={lora_mode:'none',lora:['old.safetensors']};
    ctx.doc.sampling={mode:'single',loras:[]};
    assert.equal(creationSampling(ctx.doc,ctx.shot).lora_mode,'manual');
    delete ctx.doc.sampling;ctx.doc.generation={lora:['old.safetensors'],loras:[],lora_strength:0};
    assert.deepEqual(creationSampling(ctx.doc,ctx.shot).loras,[{name:'old.safetensors',strength:0,enabled:true}]);
});
test('renaming is bounded and immutable; stale insertion owner is refused',()=>{
    const ctx=creativeFixture(),item=captureCreation({...ctx,kind:'snippet'}),doc=addCreation(ctx.doc,item);
    assert.equal(renameCreation(doc,item.id,' 新名称 ').creationLibrary.items[0].name,'新名称');
    assert.equal(doc.creationLibrary.items[0].name,item.name);
    assert.throws(()=>renameCreation(doc,item.id,'x'.repeat(201)));
    const nearLimit=structuredClone(doc);nearLimit.creationLibrary.items[0].text='x'.repeat(2*1024**2-JSON.stringify(nearLimit.creationLibrary).length+item.text.length);
    assert.throws(()=>renameCreation(nearLimit,item.id,'新'.repeat(200)),/2MiB/);
    assert.throws(()=>planCreation({...ctx,shotId:'shot',item,target:{...ctx.target,shot:'stale'}}),/当前镜头/);
});
test('collections capture selected text and bound identity without sharing mutable settings',()=>{
    const ctx=creativeFixture();ctx.target.start=3;ctx.target.end=10;
    const item=captureCreation({...ctx,kind:'snippet'});
    assert.equal(item.text,'@image2');assert.deepEqual(item.bindings,{'@image2':'b'});
    const next=addCreation(ctx.doc,item);assert.equal(ctx.doc.creationLibrary,undefined);
    assert.throws(()=>addCreation(next,item));
    const template=captureCreation({...ctx,kind:'template'});
    for(const field of ['id','rev','seed','adoptedResultId'])assert.equal(Object.hasOwn(template.shot,field),false);
    ctx.doc.sampling.low_loras[0].strength=.2;
    assert.equal(template.shot.sampling.low_loras[0].strength,1);
});
test('snippet insertion remaps UUID to destination numbering and keeps original selection',()=>{
    const ctx=creativeFixture(),item=captureCreation({...ctx,kind:'snippet'});
    ctx.doc.sharedRefs=['c'];ctx.shot.simplePrompt='保留全部文字';ctx.target.start=2;ctx.target.end=5;
    const plan=planCreation({...ctx,shotId:'shot',item,uuid:()=>''});
    assert.equal(plan.doc.shots[0].simplePrompt,'保留原文 @image3全部文字');
    assert.equal(ctx.shot.simplePrompt,'保留全部文字');assert.deepEqual(plan.missing,[]);
    const globalPlan=planCreation({...ctx,shotId:'shot',item,target:{kind:'global',start:0},uuid:()=>''});
    assert.match(globalPlan.doc.global,/@missing_image2/);assert.equal(globalPlan.missing.length,1);
    assert.deepEqual(globalPlan.doc.sharedRefs,['c']);
});
test('template creates new identities and leaves old adopted result and source untouched',()=>{
    const ctx=creativeFixture(),item=captureCreation({...ctx,kind:'template'});ctx.doc.sharedRefs=['c'];
    let serial=0;
    const plan=planCreation({...ctx,shotId:'shot',item,uuid:()=>`new-${++serial}`});
    assert.equal(plan.doc.shots.length,2);assert.deepEqual(plan.doc.shots[0],ctx.shot);
    const fresh=plan.doc.shots[1];assert.equal(fresh.id,'new-1');assert.equal(fresh.events[0].id,'new-2');
    assert.equal(fresh.adoptedResultId,undefined);assert.equal(fresh.seed,undefined);
    assert.equal(fresh.simplePrompt,'原文 @image3');assert.equal(fresh.prompt,'高级 @image2');
    assert.equal(fresh.sampling.low_loras.length,2);assert.equal(fresh.sampling.high_loras.length,1);
    assert.equal(plan.doc.generation.unet,'shared-base');
    assert.throws(()=>planCreation({...ctx,assets:[],shotId:'shot',item,uuid:()=>''}),/素材不可用/);
});
test('recipe applies only chosen fields to chosen shots with independent LoRA chains',()=>{
    const ctx=creativeFixture(),item=captureCreation({...ctx,kind:'recipe'});
    ctx.doc.shots.push({...structuredClone(ctx.shot),id:'two'},{...structuredClone(ctx.shot),id:'three'});
    const plan=planCreation({...ctx,shotId:'shot',item,fields:['sampling'],targets:['shot','two']});
    assert.equal(plan.doc.shots[0].samplingInherit,false);assert.equal(plan.doc.shots[1].samplingInherit,false);
    assert.deepEqual(plan.doc.shots[2],ctx.doc.shots[2]);assert.equal(plan.doc.shots[0].d3Inherit,true);
    assert.equal(plan.doc.shots[0].adoptedResultId,'accepted');assert.equal(plan.doc.shots[0].seed,42);
    plan.doc.shots[0].sampling.high_loras[0].strength=-1;
    assert.equal(plan.doc.shots[1].sampling.high_loras[0].strength,.5);
    assert.throws(()=>planCreation({...ctx,doc:{...ctx.doc,sharedRatio:true},shotId:'shot',item,fields:['ratio']}),/全片共用/);
    assert.throws(()=>planCreation({...ctx,shotId:'shot',item,fields:['sampling'],targets:[]}),/目标镜头/);
});
test('asset group preserves old alias meaning when extending a legacy role-only tray',()=>{
    const ctx=creativeFixture();ctx.shot.tray=[];ctx.shot.refs=[];ctx.shot.simplePrompt='原图 @image1';
    const item={kind:'group',asset_ids:['c','audio'],name:'组合'};
    const plan=planCreation({...ctx,shotId:'shot',item});
    assert.equal(plan.doc.shots[0].simplePrompt,'原图 @image2');assert.equal(plan.doc.shots[0].first,'b');
    assert.equal(plan.doc.shots[0].sound,'native');assert.equal(plan.doc.shots[0].audio,null);
    assert.deepEqual(plan.doc.shots[0].refs,['c']);
    assert.throws(()=>planCreation({...ctx,shotId:'shot',item:{...item,asset_ids:['missing']}}),/不可用/);
});
test('legacy single LoRA settings and explicit reconnection remain usable in collections',()=>{
    const ctx=creativeFixture();delete ctx.doc.sampling;ctx.doc.generation={lora:['content.safetensors'],lora_strength:0,resolution_mp:'.6'};
    const saved=creationSampling(ctx.doc,ctx.shot);assert.equal(saved.lora_mode,'manual');assert.equal(saved.loras[0].strength,0);
    ctx.doc.creationLibrary={version:1,items:['snippet','template','group'].map(kind=>captureCreation({...ctx,kind}))};
    remapCreationAsset(ctx.doc,'b','replacement');
    assert.equal(ctx.doc.creationLibrary.items[0].bindings['@image2'],'replacement');
    assert.equal(ctx.doc.creationLibrary.items[1].shot.first,'replacement');
    assert.deepEqual(ctx.doc.creationLibrary.items[2].asset_ids,['a','replacement']);
});
import {parseStoryboardText} from '../web/director/storyboard_ui.mjs';
test('storyboard explicit multiline parse preserves prose and invalidates foreign aliases',()=>{
    const result=parseStoryboardText('\uFEFF# 开场 | 4秒\r\n  原文：@image1 / <Audio 2>\r\n\\# 这行是正文\r\n\r\n# <第二镜> | 3.5s\r\n后文。');
    assert.equal(result.errors.length,0);
    assert.equal(result.total,7.5);
    assert.equal(result.unbound,2);
    assert.equal(result.rows[0].prompt,'  原文：@missing_image1 / @missing_audio2\n# 这行是正文');
    assert.equal(result.rows[1].name,'<第二镜>');
    assert.equal(result.rows[1].prompt,'后文。');
});
for(const text of ['', '没有标题的故事', '# 开场 | 0秒\n空', '# 开场 | 一分钟\n原文', '# 开场 | 4秒', '# '+ '名'.repeat(201)+' | 4s\n正文'])test('storyboard refuses ambiguous or empty shot: '+text.slice(0,30),()=>{
    assert.ok(parseStoryboardText(text).errors.length);
});
test('storyboard limits batches and input, and never ignores malformed middle rows',()=>{
    assert.ok(parseStoryboardText(Array.from({length:201},(_,i)=>`# 第${i}镜 | 4秒\n正文`).join('\n')).errors.length);
    assert.throws(()=>parseStoryboardText('a'.repeat(1024*1024+1)));
    const result=parseStoryboardText('# 合法 | 4秒\n保持\n# 错误无时长\n不能丢弃\n# 结束 | 5s\n继续');
    assert.equal(result.rows.length,2);
    assert.equal(result.errors.length,2);
});
import {referenceQuery, filterReferenceChoices} from '../web/director/references_ui.mjs';
test('reference candidates only replace the typed query at a collapsed caret',()=>{
    assert.deepEqual(referenceQuery('原文 @角色 后文',6),{start:3,end:6,query:'角色',value:'原文 @角色 后文'});
    assert.equal(referenceQuery('@image12',4),null);
    assert.equal(referenceQuery('@image1',1,4),null);
    assert.equal(referenceQuery('plain text',10),null);
    assert.equal(referenceQuery('@角色',-1),null);
    assert.deepEqual(filterReferenceChoices([{token:'@image1',name:'角色图'},{token:'@audio1',name:'VOICE'}],'角'),[{token:'@image1',name:'角色图'}]);
    assert.equal(filterReferenceChoices([{token:'@audio1',name:'VOICE'}],'voice').length,1);
});
import assert from 'node:assert/strict';
import { directorSeed, newDirectorSeed, directorResultKey, mergeDirectorResults, selectDirectorResult, moveDirectorShot, directorShotInputKey } from '../web/director/session.mjs';
import { directorSamplingSummary } from '../web/director/sampling_ui.mjs';
import { directorDraftIssues } from '../web/director/preflight_ui.mjs';
test('unbound imported aliases are reported only in active prompt drafts',()=>{
    const shot={id:'s',mode:'text',sound:'native',writingMode:'simple',simplePrompt:'完整',prompt:'@missing_image1',manualDuration:4};
    assert.equal(directorDraftIssues({},shot,[]).length,0);
    assert.match(directorDraftIssues({global:'@missing_image2'},shot,[])[0].message,/未绑定/);
    assert.match(directorDraftIssues({},{...shot,writingMode:'advanced'},[])[0].message,/未绑定/);
});
import { normalizedLayout } from '../web/director/layout_ui.mjs';
import { assetMatchesFilter } from '../web/director/assets_ui.mjs';
import { filmClipTimes } from '../web/director/film_ui.mjs';
import { comparisonWindow } from '../web/director/compare_ui.mjs';

for(const mode of ['single','two_pass','hyperflow'])test(`effective MP: ${mode}, active scope only`,()=>{
    const global={mode,resolution_mp:'0.4',output_mp:'0.6',single:{resolution_mp:'1'}};
    const project={sampling:global,generation:{resolution_mp:'0.8'}};
    const shot={samplingInherit:false,sampling:{mode,resolution_mp:'0.5',output_mp:'1'}};
    assert.equal(directorSamplingSummary(project,{}).mp,mode==='single'?'0.4':'0.6');
    assert.equal(directorSamplingSummary(project,shot).mp,mode==='single'?'0.5':'1');
    assert.match(directorSamplingSummary(project,{}).label,/目标/);
    global.output_mp='auto';global.resolution_mp='auto';
    assert.equal(directorSamplingSummary(project,{}).label,'自动 · 待预检');
});
test('missing or invalid local sampling is not shown as valid global MP',()=>{
    const project={sampling:{mode:'hyperflow',output_mp:'0.6'},generation:{resolution_mp:'0.4'}};
    assert.equal(directorSamplingSummary(project,{samplingInherit:false}).mp,null);
    assert.equal(directorSamplingSummary(project,{samplingInherit:false,sampling:{mode:'bad'}}).mp,null);
    assert.equal(directorSamplingSummary({generation:project.generation},{}).mp,'0.4');
    assert.equal(directorSamplingSummary({sampling:{mode:'two_pass'},generation:project.generation},{}).mp,'auto');
});
for(const value of [0,1,26091901,'4294967295',Number.MAX_SAFE_INTEGER])test(`seed exact roundtrip ${value}`,()=>{
    assert.equal(directorSeed({seed:value}),Number(value));
    assert.equal(directorSeed(JSON.parse(JSON.stringify({seed:value}))),Number(value));
});
for(const seed of [-1,1.5,'',true,'1e2',Infinity,NaN,Number.MAX_SAFE_INTEGER+1])test(`invalid seed rejected ${seed}`,()=>{
    assert.throws(()=>directorSeed({seed}));
});
test('new variation cannot accidentally retain the same seed',()=>{
    const random={getRandomValues:array=>{array[0]=4294967295;return array;}};
    assert.equal(newDirectorSeed(4294967295,random),0);
    assert.equal(newDirectorSeed(4,random),4294967295);
    assert.equal(directorSeed({}),26091901);
});
const version=(id,time,state='success')=>({shot_id:'shot',prompt_id:id,state,submitted_at:time,outputs:state==='success'?{'12':{images:[{filename:id+'.mp4',type:'output'}]}}:{}});
test('all versions retained; failure or pending never replaces adopted version',()=>{
    const old=version('old',1),newer=version('new',2),failure=version('failed',3,'error');
    const records=mergeDirectorResults([old,newer],[failure,version('old',4,'pending')]);
    assert.equal(records.length,3);
    assert.equal(selectDirectorResult(records,'old').state,'success');
    assert.equal(selectDirectorResult(records).prompt_id,'new');
    assert.equal(selectDirectorResult(records,'missing'),undefined);
});
test('verified imported media corruption invalidates stale success without substituting another version',()=>{
    const original=version('adopted',1),other=version('other',2);
    const corrupted={...original,state:'error',outputs:{},integrity_error:'SHA mismatch'};
    const records=mergeDirectorResults([original,other],[corrupted]);
    assert.equal(selectDirectorResult(records,'adopted').state,'error');
    assert.deepEqual(selectDirectorResult(records,'adopted').outputs,{});
    const repaired=mergeDirectorResults(records,[{...original,integrity_error:null}]);
    assert.equal(selectDirectorResult(repaired,'adopted').state,'success');
    assert.equal(selectDirectorResult(repaired,'adopted').integrity_error,null);
});
test('server enrichment keeps other history and static legacy identities',()=>{
    const old=version('old',1),newer=version('new',2);
    const records=mergeDirectorResults([newer],[old,{...newer,request_id:'request',snapshot_available:true}]);
    assert.equal(records.length,2);assert.equal(records[0].snapshot_available,true);
    const legacy={shot_id:'shot',outputs:{legacy:{images:[{filename:'a.mp4',subfolder:'folder\\shot'}]}}};
    const moved=structuredClone(legacy);moved.outputs.legacy.images[0].subfolder='folder/shot';
    assert.equal(directorResultKey(legacy),directorResultKey(moved));
    assert.notEqual(directorResultKey(legacy),directorResultKey({...legacy,shot_id:'other'}));
});
test('shot reordering retains UUID, seed, adoption and does not mutate original array',()=>{
    const a={id:'a',seed:42,adoptedResultId:'accepted'},b={id:'b',seed:9},c={id:'c',seed:80};
    const shots=[a,b,c],next=moveDirectorShot(shots,'c',0);
    assert.deepEqual(next,[c,a,b]);assert.deepEqual(shots,[a,b,c]);
    assert.equal(next[1],a);assert.equal(moveDirectorShot(shots,'a',-1),shots);
    assert.equal(moveDirectorShot(shots,'missing',0),shots);
});
test('modified detection ignores title, revision, selection and inactive prompt drafts',()=>{
    const shot={id:'a',seed:42,name:'First',rev:1,mode:'text',sound:'native',writingMode:'simple',simplePrompt:'hello',prompt:'old advanced draft'};
    const doc={shots:[shot],global:'common',sampling:{mode:'single',resolution_mp:'0.6'}};
    const before=directorShotInputKey(doc,shot);
    const renamed={...shot,name:'Renamed',rev:99,selected:'other',prompt:'edited inactive draft',adoptedResultId:'accepted'};
    assert.equal(directorShotInputKey(doc,renamed),before);
    assert.notEqual(directorShotInputKey(doc,{...renamed,simplePrompt:'new prompt'}),before);
    assert.notEqual(directorShotInputKey({...doc,global:'changed'},renamed),before);
    assert.notEqual(directorShotInputKey(doc,{...renamed,seed:43}),before);
});
test('local sampling does not inherit changes to unused global sampling',()=>{
    const shot={id:'a',samplingInherit:false,sampling:{mode:'hyperflow',variant:'single8',output_mp:'0.6',low_loras:[],high_loras:[{name:'unused'}]}};
    const doc={generation:{unet:'base'},sampling:{mode:'single',resolution_mp:'1'}};
    const before=directorShotInputKey(doc,shot);
    assert.equal(directorShotInputKey({...doc,sampling:{mode:'two_pass',output_mp:'0.4'}},shot),before);
    assert.equal(directorShotInputKey(doc,{...shot,sampling:{...shot.sampling,high_loras:[]}}),before);
    assert.notEqual(directorShotInputKey({...doc,generation:{unet:'new-base'}},shot),before);
});

const draftShot={id:'draft',mode:'refs',sound:'native',writingMode:'simple',simplePrompt:'女人微笑',manualDuration:4,autoDuration:true,refs:[],events:[]};
test('draft checks count shared references and do not request inactive end frame or native audio',()=>{
    assert.deepEqual(directorDraftIssues({sharedRefs:['image']},draftShot,[{id:'image',kind:'image'}]),[]);
    assert.deepEqual(directorDraftIssues({}, {...draftShot,mode:'text'}),[]);
    assert.deepEqual(directorDraftIssues({}, {...draftShot,mode:'ends',last:'image'},[{id:'image',kind:'image'}]),[]);
});
test('draft checks inspect only active writing mode and audio requirements',()=>{
    assert.equal(directorDraftIssues({}, {...draftShot,mode:'text',simplePrompt:'',prompt:'inactive'}).filter(issue=>issue.field==='prompt').length,1);
    assert.deepEqual(directorDraftIssues({}, {...draftShot,mode:'text',writingMode:'advanced',simplePrompt:'',prompt:'',events:[{text:'笑',start:0,end:2}]}),[]);
    const driven={...draftShot,mode:'first',first:'image',sound:'record',audio:'audio',start:0,end:2,simplePrompt:''};
    const assets=[{id:'image',kind:'image'},{id:'audio',kind:'audio',duration:2}];
    assert.deepEqual(directorDraftIssues({},driven,assets),[]);
    assert.ok(directorDraftIssues({}, {...driven,end:3},assets).some(issue=>issue.field==='audio'));
});
test('draft hints catch missing assets and invalid timing without claiming file or dependency checks',()=>{
    assert.ok(directorDraftIssues({sharedRefs:['gone']},draftShot).some(issue=>issue.field==='source'));
    assert.ok(directorDraftIssues({}, {...draftShot,mode:'text',manualDuration:0}).some(issue=>issue.field==='timing'));
    assert.ok(directorDraftIssues({}, {...draftShot,mode:'text',writingMode:'advanced',events:[{text:'笑',start:0,end:0.1}]}).some(issue=>issue.field==='events'));
});
test('layout preferences have bounded widths and recover from corrupt storage',()=>{
    assert.deepEqual(normalizedLayout(),{mode:'parallel',split:53});
    assert.deepEqual(normalizedLayout(null),{mode:'parallel',split:53});
    assert.deepEqual(normalizedLayout({mode:'bad',split:999}),{mode:'parallel',split:65});
    assert.deepEqual(normalizedLayout({mode:'review',split:-20}),{mode:'review',split:40});
});
test('modified detection ignores disabled LoRA drafts and row metadata',()=>{
    const shot={id:'a'},row={id:'row-a',name:'a.safetensors',strength:0.7,enabled:true,source:'user'};
    const doc={sampling:{mode:'two_pass',low_loras:[row],high_loras:[]}};
    const revised={sampling:{...doc.sampling,low_loras:[{...row,id:'row-b',source:'imported'},{name:'b',enabled:false,strength:1}]}};
    assert.equal(directorShotInputKey(doc,shot),directorShotInputKey(revised,shot));
    revised.sampling.low_loras[0].strength=0.5;
    assert.notEqual(directorShotInputKey(doc,shot),directorShotInputKey(revised,shot));
});
test('material filters preserve identity and distinguish type, scope and recency',()=>{
    const asset={id:'a',name:'女人正面.PNG',kind:'image'};
    assert.equal(assetMatchesFilter(asset,{query:' 正面.png ',kind:'image',scope:'shared'},['a']),true);
    assert.equal(assetMatchesFilter(asset,{scope:'local'},['a']),false);
    assert.equal(assetMatchesFilter(asset,{kind:'audio'}),false);
    assert.equal(assetMatchesFilter(asset,{recent:true},[],['a']),true);
    assert.equal(assetMatchesFilter(asset,{recent:true},[],['other']),false);
    assert.equal(assetMatchesFilter(undefined),false);
});
test('film uses explicit source-frame ranges rather than the mutable shot duration',()=>{
    assert.deepEqual(filmClipTimes({media:{fps:24},in_frame:24,out_frame:72}),{start:1,end:3,duration:2});
    for(const entry of [{media:{fps:0},in_frame:0,out_frame:4},{media:{fps:24},in_frame:2,out_frame:1},{media:{fps:24},in_frame:0.5,out_frame:4}])assert.throws(()=>filmClipTimes(entry));
});
test('comparison uses common actual duration with independent source offsets',()=>{
    const result=comparisonWindow([{media:{fps:24},in_frame:24,out_frame:96},{media:{fps:24},in_frame:0,out_frame:48}]);
    assert.equal(result.duration,2);assert.equal(result.times[0].start,1);assert.equal(result.times[1].start,0);
    assert.throws(()=>comparisonWindow([]));
});
