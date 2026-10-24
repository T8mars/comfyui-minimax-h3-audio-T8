import test from 'node:test';
import assert from 'node:assert/strict';
import {newControlRow,normalizeControls,portableControls,validateControlProject,controlRows,controlAssetIds,controlSamplingMode,controlInputState,duplicateControlRows,remapControlAsset,removeControlAsset,requiresControlProtocol,assertControlProtocol,controlDraftIssues,containedRect,dragControlBox} from '../web/director/control_state.mjs';
import {directorShotInputKey} from '../web/director/session.mjs';
import {captureCreation,planCreation,remapCreationAsset,creationTokens} from '../web/director/creation_library.mjs';
import {radarCandidateInputState} from '../web/director/radar_state.mjs';
import {directorDraftIssues} from '../web/director/preflight_ui.mjs';

const id=n=>'00000000-0000-4000-8000-'+String(n).padStart(12,'0');
const row=()=>({...newControlRow(id(11)),asset_id:id(1),control_net_name:'my-unlisted-model.safetensors'});
const shot=()=>({id:id(10),mode:'text',sound:'native',simplePrompt:'A room.',writingMode:'simple',prompt:'',events:[],tray:[],refs:[],first:null,last:null,audio:null,ratio:'16:9',duration:4,manualDuration:4,seed:1,rev:1,controlRegions:{version:1,items:[row()]}});
const media=()=>[{id:id(1),name:'depth.png',kind:'image',width:512,height:512,sha256:'a'.repeat(64)}];
const doc=()=>({shots:[shot()],sharedRefs:[],global:'',sharedRatio:true,ratio:'16:9',sampling:{mode:'single'},generation:{}});

test('optional drafts preserve old projects and exact contract defaults',()=>{
    const old={id:id(10)};assert.equal(normalizeControls(undefined),undefined);assert.deepEqual(controlRows(old),[]);assert.equal(controlInputState(old),undefined);
    const raw={version:1,items:[{id:id(11)}]},before=structuredClone(raw),normalized=normalizeControls(raw);
    assert.deepEqual(raw,before);assert.deepEqual(normalized.items[0],newControlRow(id(11)));
    assert.equal(normalized.items[0].crop.enabled,false);assert.equal(normalized.items[0].source.short_video_policy,'hold_last_frame');
});
test('malformed version, data, duplicate identity and nested types reject without rewriting',()=>{
    for(const corrupt of [{version:2,items:[]},{version:true,items:[]},{version:1,items:[row(),row()]},{version:1,items:[{...row(),id:'x'}]},{version:1,items:[{...row(),stages:[]}]},{version:1,items:[{...row(),strength:NaN}]},{version:1,items:[{...row(),extra:{bad:Infinity}}]}])assert.throws(()=>normalizeControls(corrupt));
    const project={doc:doc()};project.doc.creationLibrary={version:1,items:[{kind:'template',shot:{controlRegions:{version:7,items:[]}}}]};assert.throws(()=>validateControlProject(project));
});
test('portable controls whitelist nested/private data while editable draft preserves it',()=>{
    const s=shot();s.controlRegions.private='/secret';s.controlRegions.items[0].private='/secret';s.controlRegions.items[0].source.private='/secret';
    assert.equal(normalizeControls(s.controlRegions).items[0].source.private,'/secret');
    for(const value of [portableControls(s.controlRegions),controlInputState(s)])assert.ok(!JSON.stringify(value).includes('/secret'));
});
test('inactive and stage semantics are real input semantics, not model admission',()=>{
    const d=doc(),s=d.shots[0];s.controlRegions.items[0].stages={low:false,high:false};
    assert.equal(controlSamplingMode(d,s),'single');assert.equal(controlRows(s,true,'single').length,1);assert.equal(controlRows(s,true,'two_pass').length,0);
    assert.equal(controlInputState(s,'two_pass'),undefined);assert.equal(controlInputState(s).items[0].stages,undefined);
    d.sampling={mode:'hyperflow',variant:'single8'};assert.equal(controlSamplingMode(d,s),'single');
    s.controlRegions.items[0].enabled=false;assert.equal(controlInputState(s),undefined);assert.deepEqual(controlAssetIds(s),[id(1)]);assert.deepEqual(controlAssetIds(s,true),[]);
    s.controlRegions.items[0].enabled=true;s.controlRegions.items[0].strength=0;assert.equal(controlInputState(s),undefined);
    s.controlRegions.items[0].strength=-1;assert.equal(controlDraftIssues(d,s,new Map(media().map(a=>[a.id,a]))).length,0);
});
test('actual errors stay visible, crop exterior legal, full/disabled geometry drafts retained',()=>{
    const d=doc(),s=d.shots[0],r=s.controlRegions.items[0],assets=new Map(media().map(a=>[a.id,a]));
    r.crop.enabled=true;r.crop.xyxy_normalized=[-.3,-.1,1.2,1.5];assert.deepEqual(controlDraftIssues(d,s,assets),[]);
    r.region.mode='rect';r.region.xyxy_normalized=[-.1,0,.8,1];assert.match(controlDraftIssues(d,s,assets).join(' '),/0–1/);
    r.region.mode='full';r.start_percent=.9;r.end_percent=.1;r.control_net_name='';r.asset_id=null;
    const messages=directorDraftIssues(d,s,media()).filter(x=>x.field==='controlRegions');assert.equal(messages.length,3);
    r.enabled=false;assert.equal(controlDraftIssues(d,s,assets).length,0);
});
test('HyperFlow resume_tail ignores incomplete LOW-only drafts and retains strict HIGH inputs',()=>{
    const d=doc(),s=d.shots[0],r=s.controlRegions.items[0];d.sampling={mode:'hyperflow',variant:'continuous4plus4separate',stage_checkpoint:{mode:'resume_tail'}};
    r.stages={low:true,high:false};r.asset_id=null;r.control_net_name='';assert.equal(controlSamplingMode(d,s),'high_only');
    assert.equal(controlInputState(s,'high_only'),undefined);assert.deepEqual(controlDraftIssues(d,s,new Map()),[]);assert.doesNotThrow(()=>assertControlProtocol({doc:d},{}));
    const inactive=directorShotInputKey(d,s,[]);r.crop.xyxy_normalized=[1,1,0,0];assert.equal(directorShotInputKey(d,s,[]),inactive);
    r.stages.high=true;assert.equal(controlDraftIssues(d,s,new Map()).length,2);assert.throws(()=>assertControlProtocol({doc:d},{}));
    r.asset_id=id(1);r.control_net_name='custom.safetensors';r.crop.enabled=false;const before=directorShotInputKey(d,s,media());r.stages.low=false;assert.equal(directorShotInputKey(d,s,media()),before);
});
test('active protocol extension acknowledgement only, legacy and inactive need none',()=>{
    const p={doc:doc()};assert.equal(requiresControlProtocol(p),true);assert.throws(()=>assertControlProtocol(p,{}),/未提交/);
    assert.doesNotThrow(()=>assertControlProtocol(p,{extensions:{control_regions_v1:true}}));
    assert.throws(()=>assertControlProtocol(p,{extensions:{control_regions_v1:1}}));
    assert.doesNotThrow(()=>assertControlProtocol(p,{},[id(99)]));p.doc.shots[0].controlRegions.items[0].enabled=false;assert.doesNotThrow(()=>assertControlProtocol(p,{}));
});
test('row duplication, removal and reconnect keep source draft but renew control identity',()=>{
    const s=shot(),old=structuredClone(s);duplicateControlRows(s,()=>id(12));assert.equal(s.controlRegions.items[0].id,id(12));assert.equal(old.controlRegions.items[0].id,id(11));
    remapControlAsset(s,id(1),id(2));assert.equal(s.controlRegions.items[0].asset_id,id(2));removeControlAsset(s,id(2));assert.equal(s.controlRegions.items.length,1);assert.equal(s.controlRegions.items[0].asset_id,null);
});
test('input identity observes source SHA and active geometry, ignores private/row/stage/dormant drafts',()=>{
    const d=doc(),s=d.shots[0],a=media(),key=()=>directorShotInputKey(d,s,a),base=key();
    s.controlRegions.items[0].id=id(50);s.controlRegions.items[0].source.private='ignored';s.controlRegions.items[0].stages={low:false,high:false};assert.equal(key(),base);
    s.controlRegions.items[0].strength=.9;assert.notEqual(key(),base);s.controlRegions.items[0].strength=.7;a[0].sha256='b'.repeat(64);assert.notEqual(key(),base);
    s.controlRegions.items[0].enabled=false;const inactive=key();s.controlRegions.items[0].crop.xyxy_normalized=[.8,.9,.1,.2];a[0].sha256='c'.repeat(64);assert.equal(key(),inactive);
    d.sampling={mode:'two_pass'};s.controlRegions.items[0].enabled=true;const bothOff=key();s.controlRegions.items[0].control_net_name='new.safetensors';assert.equal(key(),bothOff);
});
test('control-only media never receives reference aliases but template carries and reconnects it',()=>{
    const d=doc(),s=d.shots[0],a=media();assert.equal(creationTokens(d,s,a).size,0);
    s.controlRegions.items[0].source.private='secret';const item=captureCreation({doc:d,shot:s,assets:a,kind:'template',name:'Control',id:id(30),d3:{}});
    assert.equal(item.shot.controlRegions.items[0].asset_id,id(1));assert.equal(item.shot.controlRegions.items[0].source.private,undefined);assert.deepEqual(item.bindings,{});
    const applied=planCreation({doc:d,shotId:s.id,assets:a,item,uuid:(()=>{let n=40;return()=>id(n++);})()});assert.notEqual(applied.doc.shots[1].controlRegions.items[0].id,id(11));assert.equal(applied.doc.shots[1].controlRegions.items[0].asset_id,id(1));
    d.creationLibrary={items:[item]};remapCreationAsset(d,id(1),id(2));assert.equal(d.creationLibrary.items[0].shot.controlRegions.items[0].asset_id,id(2));
    assert.throws(()=>planCreation({doc:d,shotId:s.id,assets:[],item,uuid:()=>id(40)}),/素材不可用/);
});
test('candidate invalidation tracks only active control sources and options',()=>{
    const d=doc(),s=d.shots[0],a=media();s.promptCandidate={active:true,options:{}};
    const before=JSON.stringify(radarCandidateInputState(d,s,a));s.controlRegions.items[0].region.mode='rect';assert.notEqual(JSON.stringify(radarCandidateInputState(d,s,a)),before);
    s.controlRegions.items[0].enabled=false;const off=JSON.stringify(radarCandidateInputState(d,s,a));s.controlRegions.items[0].strength=.4;a[0].sha256='z';assert.equal(JSON.stringify(radarCandidateInputState(d,s,a)),off);
});
test('geometry preserves aspect, continuous reverse draw/move/resize and unbounded source crop',()=>{
    assert.deepEqual(containedRect({left:10,top:20,width:200,height:100},100,200),{left:85,top:20,width:50,height:100});
    assert.deepEqual(dragControlBox([0,0,1,1],{x:.8,y:.7},{x:.2,y:.1}),[.2,.1,.8,.7]);
    assert.deepEqual(dragControlBox([.2,.2,.5,.5],{x:.3,y:.3},{x:1.5,y:1.5},'move'),[.7,.7,1,1]);
    assert.deepEqual(dragControlBox([.2,.2,.5,.5],{x:.5,y:.5},{x:1.5,y:1.5},'resize','se',false),[.2,.2,1.5,1.5]);
    assert.deepEqual(dragControlBox([0,0,1,1],{x:.3,y:.3},{x:-.2,y:1.4},'draw','',false),[-.2,.3,.3,1.4]);
});
