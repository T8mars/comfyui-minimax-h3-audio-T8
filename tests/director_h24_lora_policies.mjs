import test from 'node:test';
import assert from 'node:assert/strict';
import {directorShotInputKey} from '../web/director/session.mjs';

const project = () => ({doc:{shots:[],sampling:{mode:'two_pass',low_loras:[{name:'a',strength:.4,enabled:true}],high_loras:[{name:'b',strength:.7,enabled:true}]}},shot:{id:'shot',mode:'text',sound:'native',simplePrompt:'room'}});
const key = p => directorShotInputKey(p.doc,p.shot,[]);
for (const stage of ['low_loras','high_loras']) for (const [field,value] of [['file_alpha_policy','metadata_alpha_per_rank'],['prefix_policy','unique_double_prefix']]) {
    test(`${stage} ${field} is a real active input`,()=>{
        const p=project(),before=key(p);p.doc.sampling[stage][0][field]=value;
        assert.notEqual(key(p),before);
    });
}
test('explicit legacy and absent defaults mean the same sampling input',()=>{
    const p=project(),before=key(p);
    for(const stage of ['low_loras','high_loras'])Object.assign(p.doc.sampling[stage][0],{file_alpha_policy:'legacy_tensor_only',prefix_policy:'legacy'});
    assert.equal(key(p),before);
});
test('disabled and inactive HIGH interpretation is not a generation change',()=>{
    const p=project();p.doc.sampling.low_loras[0].enabled=false;
    const before=key(p);p.doc.sampling.low_loras[0].prefix_policy='unique_double_prefix';assert.equal(key(p),before);
    p.doc.sampling.mode='hyperflow';p.doc.sampling.variant='single8';const single=key(p);
    p.doc.sampling.high_loras[0].file_alpha_policy='metadata_alpha_per_rank';assert.equal(key(p),single);
});
test('ordinary single generation rows include their policies too',()=>{
    const p=project();p.doc.sampling={mode:'single'};p.doc.generation={lora_mode:'manual',loras:[{name:'x',strength:1}]};
    const before=key(p);p.doc.generation.loras[0].prefix_policy='unique_double_prefix';assert.notEqual(key(p),before);
});
test('TAIL-only ignores frozen HEAD drafts but binds the actual artifact and TAIL policy',()=>{
    const p=project();Object.assign(p.doc.sampling,{mode:'hyperflow',variant:'continuous4plus4separate',stage_checkpoint:{mode:'resume_tail',artifact_path:'head',artifact_sha256:'a'.repeat(64)},stage_eav:{head:{tau:4},tail:{tau:3}},stage_relay:{head:{mode:'inherit'},tail:{mode:'inherit'}}});
    const before=key(p);p.doc.sampling.low_loras[0].prefix_policy='unused-draft';p.doc.sampling.stage_eav.head.tau=8;p.doc.sampling.stage_relay.head.mode='custom';
    assert.equal(key(p),before);
    p.doc.sampling.high_loras[0].prefix_policy='unique_double_prefix';assert.notEqual(key(p),before);
    const tail=key(p);p.doc.sampling.stage_checkpoint.artifact_sha256='b'.repeat(64);assert.notEqual(key(p),tail);
    const changedSHA=key(p);p.doc.sampling.stage_checkpoint.artifact_path='other-head';assert.notEqual(key(p),changedSHA);
});
