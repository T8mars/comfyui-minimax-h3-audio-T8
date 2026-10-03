import assert from 'node:assert/strict';
import test from 'node:test';
import {createHash, webcrypto} from 'node:crypto';
import {directorShotInputKey, directorUUID} from '../web/director/session.mjs';
import {sha256UTF8} from '../web/director/content_hash.mjs';

const copy = value => structuredClone(value);
function fixture() {
    const source = {asset_id:'source',sha256:'a'.repeat(64)};
    const evidence = {packet:{id:'packet',revision:1,sha256:'b'.repeat(64),source,claims:[]},
        review:{packet_sha256:'b'.repeat(64),revision:1,claim_ids:[]}};
    const shot = {id:'second',mode:'text',sound:'native',writingMode:'simple',simplePrompt:'稿',prompt:'',
        refs:[],tray:[],events:[],seed:42,duration:5,manualDuration:5,autoDuration:false};
    const doc = {global:'全片',sharedRefs:[],ratio:'16:9',sharedRatio:true,
        shots:[{id:'first',sourceEvidence:evidence},shot],sharedSkills:[]};
    const assets = [{id:'source',sha256:'a'.repeat(64),kind:'video'}];
    return {doc,shot,assets};
}
function adopt(shot, options={facts:false,intent:true}) {
    shot.promptCandidate={id:'candidate',schema:'t8.director.prompt_candidate.v1',text:'候选稿',
        active:true,text_sha256:'c'.repeat(64),inputs_sha256:'d'.repeat(64),options,receipt:{}};
}
test('legacy input key remains exact; draft evidence/skills/history do not change it',()=>{
    const {doc,shot,assets}=fixture(), before=directorShotInputKey(doc,shot,assets);
    assert.deepEqual(JSON.parse(before),{global:'全片',sharedRefs:[],mode:'text',sound:'native',
        first:null,last:null,refs:[],tray:[],audio:null,writingMode:'simple',prompt:'稿',events:[],
        duration:5,manualDuration:5,autoDuration:false,ratio:'16:9',seed:42,sampling:{mode:'single'},generation:{},assets:[]});
    adopt(shot);shot.promptCandidate.active=false;
    shot.evidenceHistory=[{private:'nonexecuting'}];shot.intentHistory=[{text:'old'}];shot.candidateHistory=[{text:'old'}];
    shot.creativeIntent={revision:1,text:'编辑草稿',dependencies:[]};doc.skillLibrary=[{text:'库草稿'}];
    assert.equal(directorShotInputKey(doc,shot,assets),before);
});
test('adoption, candidate replacement and revert are execution-key changes',()=>{
    const {doc,shot,assets}=fixture(), original=directorShotInputKey(doc,shot,assets);
    adopt(shot);const adopted=directorShotInputKey(doc,shot,assets);
    assert.notEqual(adopted,original);
    shot.promptCandidate.text='另一个候选';assert.notEqual(directorShotInputKey(doc,shot,assets),adopted);
    shot.promptCandidate.active=false;assert.equal(directorShotInputKey(doc,shot,assets),original);
});
test('cross-shot key consumes exact facts/review/source but not unrelated draft edits',()=>{
    const {doc,shot,assets}=fixture();adopt(shot);
    shot.creativeIntent={revision:1,text:'保持红衣',reason:'',dependencies:[{shot_id:'first',evidence_sha256:'b'.repeat(64)}]};
    const baseline=directorShotInputKey(doc,shot,assets);
    doc.shots[0].simplePrompt='无关草稿';doc.shots[0].name='重命名';shot.intentHistory=[{text:'历史'}];
    assert.equal(directorShotInputKey(doc,shot,assets),baseline);
    for(const mutate of [
        d=>d.shots[0].sourceEvidence.packet.sha256='e'.repeat(64),
        d=>d.shots[0].sourceEvidence.review.revision++,
        d=>d.shots.splice(0,1),
    ]) {
        const next=copy(doc);mutate(next);assert.notEqual(directorShotInputKey(next,shot,assets),baseline);
    }
    assert.notEqual(directorShotInputKey(doc,shot,[{...assets[0],sha256:'f'.repeat(64)}]),baseline);
});
test('only enabled effective skill snapshots and consumed evidence/intent change the key',()=>{
    const {doc,shot,assets}=fixture();adopt(shot,{facts:false,intent:false});
    const rule={id:'rule',enabled:true,snapshot:{text_snapshot:'规则',content_sha256:'e'.repeat(64)},parameters:{}};
    doc.sharedSkills=[rule];const baseline=directorShotInputKey(doc,shot,assets);
    shot.sourceEvidence=copy(doc.shots[0].sourceEvidence);shot.creativeIntent={text:'不用这份',dependencies:[]};
    doc.skillLibrary=[{text:'不影响绑定的更新'}];assert.equal(directorShotInputKey(doc,shot,assets),baseline);
    const disabled={id:'unused',enabled:false,snapshot:{text_snapshot:'不使用'},parameters:{}};
    doc.sharedSkills.push(disabled);assert.equal(directorShotInputKey(doc,shot,assets),baseline);
    shot.skillDisabled=['rule'];assert.notEqual(directorShotInputKey(doc,shot,assets),baseline);
    shot.skillDisabled=[];shot.promptCandidate.options.facts=true;
    const withFacts=directorShotInputKey(doc,shot,assets);
    shot.sourceEvidence.review.revision++;assert.notEqual(directorShotInputKey(doc,shot,assets),withFacts);
});
for(const value of ['', 'abc', '观察🙂\r\n“你好”', 'x'.repeat(55),'x'.repeat(56),'x'.repeat(64),'x'.repeat(1000000)]) {
    test(`native and ordinary-HTTP fallback SHA match actual Node SHA; UTF8 length=${value.length}`,async()=>{
        const expected=createHash('sha256').update(value,'utf8').digest('hex');
        assert.equal(await sha256UTF8(value,null),expected);
        assert.equal(await sha256UTF8(value,webcrypto),expected);
    });
}
test('hash input bounds reject rather than silently cut content',async()=>{
    await assert.rejects(sha256UTF8(null,null));
    await assert.rejects(sha256UTF8('x'.repeat(2097153),null),/2MiB/);
});
test('ordinary HTTP UUID uses getRandomValues without randomUUID',()=>{
    const original=globalThis.crypto;
    Object.defineProperty(globalThis,'crypto',{value:{getRandomValues:b=>webcrypto.getRandomValues(b)},configurable:true});
    try { assert.match(directorUUID(),/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/); }
    finally { Object.defineProperty(globalThis,'crypto',{value:original,configurable:true}); }
});
