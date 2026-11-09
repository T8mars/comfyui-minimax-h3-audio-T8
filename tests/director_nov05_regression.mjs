// Actual session consumer with deterministic local HTTP/storage substitutes.
// This does not claim browser rendering, server queue or GPU execution.
import test from 'node:test';
import assert from 'node:assert/strict';
import {harness,project,flush} from './director_audit20_harness.mjs';
import {directorShotInputKey} from '../web/director/session.mjs';

test('decode failure shows saved Stage and explicit retry without automatic submission',async()=>{
  let failed=false;
  const h=harness({route:(req,response)=>{
    if(req.path.endsWith('/generate'))return response({prompt_id:'decode-failed',recipe:'native'});
    if(req.path.includes('/jobs/'))return response(failed?{state:'error',outputs:{},
      stage_reuse:[{node_id:'9',status:'miss_saved',current_denoiser_evaluations:4}]}:{state:'running'});
  }});
  await h.service.restore();
  const run=h.click({action:'generate'});
  const deadline=Date.now()+1000;
  while(!h.intervals.size && Date.now()<deadline)await flush();
  assert.equal(h.intervals.size,1);
  failed=true;
  await [...h.intervals.values()][0]();await run;
  assert.match(h.$('[data-dialog]').innerHTML,/已采样并保存阶段/);
  assert.match(h.$('[data-dialog]').innerHTML,/保持采样设置后重试/);
  assert.equal(h.requests.filter(req=>req.path.endsWith('/generate')).length,1);
});

test('slow compile cannot overwrite a new shot or changed ordered reference bytes',async()=>{
  const p=project(),first=p.doc.shots[0];
  const second={...structuredClone(first),id:'33333333-3333-4333-8333-333333333333',name:'Second'};
  p.doc.shots.push(second);
  let release,number=0;
  const responseFor=req=>({ready:true,shots:[{id:req.body.shot_id,canvas:{width:64,height:32,preprocessing:[]}}]});
  const h=harness({initial:p,route:(req,response)=>{
    if(req.path.endsWith('/compile')){
      number++;
      if(number===1)return new Promise(resolve=>{release=()=>resolve(response(responseFor(req)))});
      return response(responseFor(req));
    }
  }});
  await h.service.restore();await flush();
  const old=h.service.compile();await flush();assert.ok(release);
  h.setCurrent(second.id);
  h.doc().shots[1].mode='refs';h.doc().shots[1].refs=['blue','red'];h.doc().shots[1].tray=['red','blue'];
  h.ctx.assets().set('red',{id:'red',kind:'image',sha256:'a'.repeat(64)});
  h.ctx.assets().set('blue',{id:'blue',kind:'image',sha256:'b'.repeat(64)});
  await h.service.compile();const fresh=h.$('[data-dialog]').innerHTML;
  release();await old;
  assert.equal(h.$('[data-dialog]').innerHTML,fresh);
  assert.deepEqual(h.service.checkedCanvas(),{width:64,height:32,preprocessing:[]});
  h.ctx.assets().get('blue').sha256='c'.repeat(64);
  assert.equal(h.service.checkedCanvas(),null);
  assert.deepEqual(h.doc().shots[1].refs,['blue','red']);
});

test('late completed take keeps its original shot while new dialogue and references remain edited',async()=>{
  const p=project(),first=p.doc.shots[0];
  p.doc.shots.push({...structuredClone(first),id:'33333333-3333-4333-8333-333333333333'});
  let done=false;
  const h=harness({initial:p,route:(req,response)=>{
    if(req.path.endsWith('/generate'))return response({prompt_id:'original-take',recipe:'test-consumer'});
    if(req.path.includes('/jobs/'))return response(done?{state:'success',outputs:{save:{images:[{filename:'done.mp4',type:'output'}]}}}:{state:'running'});
  }});
  await h.service.restore();await flush();
  const generated=h.click({action:'generate'});
  const deadline=Date.now()+1000;
  while(!h.intervals.size && Date.now()<deadline) await flush();
  assert.equal(h.intervals.size,1);
  h.setCurrent(p.doc.shots[1].id);
  const literal='<d>[Chinese]你好，原句  两个空格。</d>\r\n@image1';
  h.doc().shots[1].simplePrompt=literal;h.doc().shots[1].rev++;
  h.doc().shots[1].refs=['second-reference'];
  const edited=structuredClone(h.doc());done=true;
  await [...h.intervals.values()][0]();await generated;
  assert.deepEqual(h.doc(),edited);
  assert.equal(h.versions().find(row=>row.prompt_id==='original-take').shot_id,first.id);
  assert.equal(h.displayed(),undefined);
  assert.equal(h.requests.filter(req=>req.path.endsWith('/generate')).length,1);
  assert.equal(h.requests.find(req=>req.path.endsWith('/generate')).body.project.doc.shots[0].simplePrompt,'Audit');
});

test('ordered reference slots, byte identity and literal dialogue all invalidate input identity',()=>{
  const p=project(),shot=p.doc.shots[0];shot.mode='refs';shot.refs=['red','blue'];shot.tray=['blue','red'];
  shot.simplePrompt='@image1 says <d>[Chinese]保持原句。</d>';
  const assets=[{id:'red',kind:'image',sha256:'a'.repeat(64)},{id:'blue',kind:'image',sha256:'b'.repeat(64)}];
  const original=directorShotInputKey(p.doc,shot,assets);
  shot.refs.reverse();assert.notEqual(directorShotInputKey(p.doc,shot,assets),original);shot.refs.reverse();
  assets[0].sha256='c'.repeat(64);assert.notEqual(directorShotInputKey(p.doc,shot,assets),original);assets[0].sha256='a'.repeat(64);
  shot.simplePrompt=shot.simplePrompt.replace('原句','另一句');assert.notEqual(directorShotInputKey(p.doc,shot,assets),original);
});
