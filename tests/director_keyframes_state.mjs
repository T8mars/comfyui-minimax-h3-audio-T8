import test from 'node:test';
import assert from 'node:assert/strict';
import {middleFrameClock,resolveMiddlePosition,middleInputState,middleAssetIds,duplicateMiddleRows,removeMiddleAsset,remapMiddleAsset,assertMiddleProtocol,validateMiddleDraft} from '../web/director/keyframes_state.mjs';
import {directorShotInputKey} from '../web/director/session.mjs';
import {directorDraftIssues} from '../web/director/preflight_ui.mjs';

const make=()=>({id:'shot',mode:'ends',sound:'native',first:'first',last:'last',refs:[],tray:[],
    writingMode:'simple',simplePrompt:'原稿',prompt:'',events:[],duration:4,manualDuration:4,autoDuration:false,
    middleKeyframes:{version:1,time_basis:'visible_delivery',items:[{id:'row',asset_id:'middle',position_mode:'percent',position:95,enabled:true}]}});
const doc=shot=>({shots:[shot],global:'',sharedRefs:[],sharedRatio:true,ratio:'16:9',sampling:{mode:'single'}});
const assets=[{id:'first',kind:'image',sha256:'a'},{id:'last',kind:'image',sha256:'b'},{id:'middle',kind:'image',sha256:'c'}];

test('visible delivery and generated canvas are distinct explicit clocks',()=>{
    const shot=make(),row=shot.middleKeyframes.items[0];
    assert.deepEqual(middleFrameClock(shot),{duration:4,visible:96,generated:107});
    assert.equal(resolveMiddlePosition(shot,row).frame,90);
    shot.middleKeyframes.time_basis='generated_canvas';
    assert.equal(resolveMiddlePosition(shot,row).frame,101);
    assert.equal(resolveMiddlePosition(shot,row).visible_in_delivery,false);
});
test('duration edits preserve absolute requests and half-up semantics',()=>{
    const shot=make(),row=shot.middleKeyframes.items[0];
    row.position_mode='seconds';row.position=.0625;
    assert.equal(resolveMiddlePosition(shot,row).frame,2);
    row.position=2;shot.manualDuration=8;
    assert.equal(resolveMiddlePosition(shot,row).frame,48);
    row.position_mode='percent';row.position=50;
    assert.equal(resolveMiddlePosition(shot,row).frame,96);
});
test('large fractional frame, out of range and nonfinite values are not clamped',()=>{
    const shot=make(),row=shot.middleKeyframes.items[0];
    row.position_mode='seconds';row.position=8;
    assert.throws(()=>resolveMiddlePosition(shot,row));
    shot.manualDuration=1e8;row.position_mode='frame';row.position=1e9+.25;
    assert.throws(()=>resolveMiddlePosition(shot,row));
    row.position=NaN;assert.throws(()=>resolveMiddlePosition(shot,row));
    shot.manualDuration=1e308;assert.throws(()=>middleFrameClock(shot));
});
test('input identity excludes row IDs, editing order and disabled draft inputs',()=>{
    const shot=make();shot.middleKeyframes.items.push({id:'other',asset_id:'last',position_mode:'frame',position:10});
    const expected=middleInputState(shot);
    shot.middleKeyframes.items.reverse();shot.middleKeyframes.items.forEach(row=>row.id+='copy');
    shot.middleKeyframes.items.push({id:'disabled',asset_id:'missing',position_mode:'seconds',position:999,enabled:false});
    assert.deepEqual(middleInputState(shot),expected);
    assert.deepEqual(middleAssetIds(shot,true),['last','middle']);
});
test('Director request identity changes with active source bytes or frame requests',()=>{
    const shot=make(),project=doc(shot),before=directorShotInputKey(project,shot,assets);
    shot.middleKeyframes.items[0].id='editing-only';
    assert.equal(directorShotInputKey(project,shot,assets),before);
    assert.notEqual(directorShotInputKey(project,shot,assets.map(row=>row.id==='middle'?{...row,sha256:'new'}:row)),before);
    shot.middleKeyframes.items[0].position=50;
    assert.notEqual(directorShotInputKey(project,shot,assets),before);
});
test('no active middle plan keeps the exact old Director input identity',()=>{
    const shot=make();delete shot.middleKeyframes;
    const before=directorShotInputKey(doc(shot),shot,assets);
    shot.middleKeyframes={version:1,items:[{id:'draft',enabled:false,asset_id:'absent'}]};
    assert.equal(directorShotInputKey(doc(shot),shot,assets),before);
});
test('duplicate, reconnect, remove and undo snapshots retain independent row identities',()=>{
    const shot=make(),saved=structuredClone(shot);let index=0;
    duplicateMiddleRows(shot,()=>`new-${++index}`);
    assert.equal(shot.middleKeyframes.items[0].id,'new-1');
    assert.equal(shot.middleKeyframes.items[0].asset_id,'middle');
    remapMiddleAsset(shot,'middle','replacement');
    assert.equal(shot.middleKeyframes.items[0].asset_id,'replacement');
    removeMiddleAsset(shot,'replacement');assert.equal(shot.middleKeyframes.items.length,0);
    assert.equal(saved.middleKeyframes.items[0].id,'row');assert.equal(saved.middleKeyframes.items[0].asset_id,'middle');
});
test('old Core cannot silently ignore active keyframes; unrelated shots keep old protocol',()=>{
    const shot=make(),project={doc:doc(shot)};
    assert.throws(()=>assertMiddleProtocol(project,{ready:true},['shot']),/未确认中间关键帧协议/);
    assertMiddleProtocol(project,{ready:true,extensions:{middle_keyframes:1}},['shot']);
    assertMiddleProtocol(project,{ready:true},['other-shot']);
    shot.middleKeyframes.items[0].enabled=false;
    assertMiddleProtocol(project,{ready:true},['shot']);
});
test('snapshot validation is lossless for old/valid drafts and rejects invalid fields before hydration',()=>{
    const shot=make(),uuid='53d762a0-bc34-4d12-9876-748d2f9ed789';
    shot.middleKeyframes.items[0].id=uuid;shot.middleKeyframes.items[0].asset_id=uuid;
    const before=structuredClone(shot);validateMiddleDraft(shot);assert.deepEqual(shot,before);
    validateMiddleDraft({});validateMiddleDraft({middleKeyframes:null});
    for(const mutate of [s=>s.middleKeyframes.version=99,s=>s.middleKeyframes.items={},
        s=>s.middleKeyframes.time_basis=null,s=>s.middleKeyframes.items.push({...s.middleKeyframes.items[0]}),
        s=>s.middleKeyframes.items[0].id='not-uuid',s=>s.middleKeyframes.items[0].enabled=null,
        s=>s.middleKeyframes.items[0].visual_noise_aug=null,s=>s.middleKeyframes.items[0].visual_noise_aug=Infinity,
        s=>s.middleKeyframes.items[0].resize_mode=null,s=>s.middleKeyframes.items[0].position=null]){
        const bad=structuredClone(shot);mutate(bad);assert.throws(()=>validateMiddleDraft(bad));
    }
});

test('recording first-frame recipe can keep middle images without inventing a last frame',()=>{
    const shot=make();shot.mode='first';shot.last=null;shot.sound='record';
    shot.audio='recording';shot.start=0;shot.end=4;
    const media=[...assets,{id:'recording',kind:'audio',duration:4,sha256:'audio'}];
    const before=structuredClone(shot);
    assert.deepEqual(directorDraftIssues(doc(shot),shot,media),[]);
    assert.deepEqual(shot,before);
    shot.sound='native';
    assert.ok(directorDraftIssues(doc(shot),shot,media).some(issue=>issue.field==='middleKeyframes'));
    shot.sound='record';shot.first=null;
    assert.ok(directorDraftIssues(doc(shot),shot,media).some(issue=>issue.field==='middleKeyframes'));
});
