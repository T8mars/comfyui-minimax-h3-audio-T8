import test from 'node:test';
import assert from 'node:assert/strict';
import {directorShotInputKey} from '../web/director/session.mjs';

function fixture() {
    const shot={id:'shot',mode:'text',sound:'native',writingMode:'simple',simplePrompt:'A room.',
        refs:[],tray:[],first:null,last:null,audio:null,duration:4,ratio:'16:9',
        samplingInherit:true,d3Inherit:true,seed:42};
    const doc={shots:[shot],sharedRefs:[],generation:{},d3:{},sampling:{mode:'two_pass',
        low_loras:[{name:'a.safetensors',strength:1,enabled:true}],high_loras:[]}};
    return {doc,shot};
}

test('LoRA timing changes candidate content identity; row labels do not',()=>{
    const {doc,shot}=fixture(), row=doc.sampling.low_loras[0];
    const before=directorShotInputKey(doc,shot);
    row.id='renamed';row.source='user';
    assert.equal(directorShotInputKey(doc,shot),before);
    row.start_percent=.25;row.end_percent=.75;
    assert.notEqual(directorShotInputKey(doc,shot),before);
});

test('explicit full-range and implicit legacy LoRA have the same input identity',()=>{
    const {doc,shot}=fixture(), before=directorShotInputKey(doc,shot);
    Object.assign(doc.sampling.low_loras[0],{start_percent:0,end_percent:1});
    assert.equal(directorShotInputKey(doc,shot),before);
});

test('changing 2D tier changes content identity',()=>{
    const {doc,shot}=fixture();doc.sampling.upscaler_architecture='2d';doc.sampling.upscaler_sharpness=1;
    const before=directorShotInputKey(doc,shot);doc.sampling.upscaler_sharpness=3;
    assert.notEqual(directorShotInputKey(doc,shot),before);
});
