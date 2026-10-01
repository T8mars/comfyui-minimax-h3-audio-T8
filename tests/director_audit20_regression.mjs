import test from 'node:test';
import assert from 'node:assert/strict';
import {harness,project,flush} from './director_audit20_harness.mjs';
import {makeDirectorServices} from '../web/director/session.mjs';

const video={state:'success',outputs:{save:{videos:[{filename:'test.mp4'}]}}};
for(const change of ['save','edit','seed','other-shot','storage-disabled'])test(`lost accepted request retains exact identity and input after ${change}`,async()=>{
  const submissions=[],accepted=new Map();let lost=false;
  const h=harness({route:(req,response)=>{
    if(req.path.endsWith('/generate')){
      submissions.push(req.body);
      const input=JSON.stringify({project:req.body.project,shot_id:req.body.shot_id,seed:req.body.seed});
      if(accepted.has(req.body.request_id))assert.equal(accepted.get(req.body.request_id),input);else accepted.set(req.body.request_id,input);
      if(!lost){lost=true;throw Error('accepted, reply lost');}
      return response({prompt_id:'original',recipe:'test'});
    }
    if(req.path.includes('/jobs/'))return response(video);
  }});
  await h.service.restore();
  if(change==='storage-disabled')h.session.setItem=()=>{throw Error('quota')};
  await h.click({action:'generate'});
  if(change==='save')await h.click({service:'save'});
  if(change==='edit')h.doc().shots[0].simplePrompt='new draft';
  if(change==='seed')h.doc().shots[0].seed=901;
  if(change==='other-shot'){h.setCurrent('other');h.setCurrent(h.doc().shots[0].id);}
  await h.click({action:'generate'});await flush();
  assert.equal(accepted.size,1);assert.equal(submissions.length,2);
  assert.deepEqual(submissions[0],submissions[1]);
  if(change==='edit')assert.equal(h.doc().shots[0].simplePrompt,'new draft');
});

test('confirmed pre-reservation rejection permits corrected input; ambiguous failures do not',async()=>{
  const rows=[];let rejected=true;
  const h=harness({route:(req,response)=>{
    if(req.path.endsWith('/generate')){rows.push(req.body);if(rejected)return response({error:'fix prompt',submission_state:'not_submitted'},400);return response({prompt_id:'ok'});}
    if(req.path.includes('/jobs/'))return response(video);
  }});
  await h.service.restore();await h.click({action:'generate'});rejected=false;
  h.doc().shots[0].simplePrompt='corrected';await h.click({action:'generate'});
  assert.notEqual(rows[0].request_id,rows[1].request_id);assert.equal(rows[1].project.doc.shots[0].simplePrompt,'corrected');
});

test('legacy fingerprint-only pending is upgraded from immutable server input',async()=>{
  const p=project(),frozen=structuredClone(p),submitted=[];
  const h=harness({initial:p,route:(req,response)=>{
    if(req.path.includes('/snapshots/'))return response({complete:true,snapshot:{project:frozen,shot_id:p.current,seed:23}});
    if(req.path.endsWith('/generate')){submitted.push(req.body);return response({prompt_id:'original'});}
    if(req.path.includes('/jobs/'))return response(video);
  }});
  await h.service.restore();h.doc().shots[0].simplePrompt='new draft';
  h.session.setItem(`t8director.request.${p.id}.${p.current}`,JSON.stringify({request_id:'old-request',fingerprint:'old-fingerprint'}));
  await h.click({action:'generate'});
  assert.equal(submitted.length,1);assert.equal(submitted[0].request_id,'old-request');assert.deepEqual(submitted[0].project,frozen);
  assert.equal(h.doc().shots[0].simplePrompt,'new draft');
});

test('unrecoverable legacy pending requires explicit abandon and never automatically submits',async()=>{
  const p=project(),h=harness({initial:p,route:(req,response)=>req.path.includes('/snapshots/')?response({error:'missing'},404):undefined});
  await h.service.restore();h.session.setItem(`t8director.request.${p.id}.${p.current}`,JSON.stringify({request_id:'old-request',fingerprint:'old-fingerprint'}));
  await h.click({action:'generate'});
  assert.equal(h.requests.filter(r=>r.path.endsWith('/generate')).length,0);
  assert.match(h.$('[data-dialog]').innerHTML,/abandon-request/);
  globalThis.confirm=()=>false;await h.click({service:'abandon-request'});
  assert.ok(h.session.getItem(`t8director.request.${p.id}.${p.current}`));
  globalThis.confirm=()=>true;await h.click({service:'abandon-request'});
  assert.equal(h.session.getItem(`t8director.request.${p.id}.${p.current}`),null);
  assert.equal(h.requests.filter(r=>r.path.endsWith('/generate')).length,0);
});

for(const copy of [false,true])for(const keyKind of ['last','backup','draft'])test(`remote save success survives cache failure copy=${copy}/${keyKind}`,async()=>{
  const h=harness();await h.service.restore();const original=h.service.envelope().id,write=h.local.setItem;
  h.local.setItem=(k,v)=>{if(keyKind==='last'?k==='t8director.lastProject':keyKind==='backup'?k.includes(':backup:'):k.startsWith('t8director.draft:'))throw Error('quota');write(k,v)};
  await h.click({service:copy?'copy':'save'});
  assert.equal(h.server.size,1);assert.ok(h.messages.some(m=>m.type==='t8-director:saved'));
  assert.doesNotMatch(h.$('[data-save]').textContent,/保存未完成/);
  if(copy)assert.notEqual(h.service.envelope().id,original);else assert.equal(h.service.envelope().id,original);
  if(keyKind!=='backup'||copy)assert.match(h.$('[data-save]').textContent,/本地恢复保护不完整/);
});

test('edit-only copy clears foreign results, adoption and crop, preserving original and live edits',async()=>{
  const p=project(),shot=p.doc.shots[0];shot.adoptedResultId='old';shot.filmTrim={in_frame:12,out_frame:24};
  const h=harness({initial:p,results:new Map([[p.id,[{...video,prompt_id:'old',shot_id:shot.id}] ]])});
  await h.service.restore();await flush();assert.equal(h.versions().length,1);
  await h.click({service:'copy'});await flush();
  const copied=h.service.envelope(),saved=h.server.get(copied.id);
  assert.notEqual(copied.id,p.id);assert.equal(h.versions().length,0);
  for(const x of [copied,saved]){assert.equal(x.doc.shots[0].adoptedResultId,undefined);assert.equal(x.doc.shots[0].filmTrim,undefined);}
  assert.equal(p.doc.shots[0].adoptedResultId,'old');
  const backup=JSON.parse(h.local.getItem('t8director.draft:audit:backup:'+p.id));assert.equal(backup.doc.shots[0].filmTrim.in_frame,12);
});

for(const editing of ['prompt','asset-only'])test(`copy response loss retries same target and preserves ${editing}`,async()=>{
  const remote=new Map(),targets=[];let first=true;
  const h=harness({route:(req,response)=>{
    if(req.path.includes('/projects/')&&req.method==='POST'){
      targets.push(req.body.project.id);
      if(remote.has(req.body.project.id))return response({error:'conflict'},409);
      remote.set(req.body.project.id,{...structuredClone(req.body.project),revision:1});
      if(first){first=false;throw Error('response lost');}
    }
    if(req.path.includes('/projects/')&&req.method==='GET')return response(remote.get(req.path.split('/').at(-1)));
  }});
  await h.service.restore();const original=h.service.envelope().id;
  await h.click({service:'copy'});assert.equal(h.service.envelope().id,original);
  if(editing==='prompt')h.doc().shots[0].simplePrompt='later editing';else h.ctx.assets().set('new',{id:'new',name:'new asset',kind:'image'});
  await h.click({service:'copy'});
  assert.equal(new Set(targets).size,1);assert.equal(remote.size,1);assert.equal(h.service.envelope().id,targets[0]);
  if(editing==='prompt')assert.equal(h.doc().shots[0].simplePrompt,'later editing');else assert.equal(h.service.envelope().assets.length,1);
  assert.match(h.$('[data-save]').textContent,/后续编辑/);
});

test('selected IDs freeze before slow previous-batch query',async()=>{
  const p=project();p.doc.shots.push({...structuredClone(p.doc.shots[0]),id:'other'});
  let release,slow=false;
  const h=harness({initial:p,route:(req,response)=>{
    if(req.path.endsWith('/batches/old')&&slow)return new Promise(resolve=>{release=()=>resolve(response({complete:true,items:[]}));});
    if(req.path.endsWith('/batches')&&req.method==='POST')return response({id:req.body.batch_id});
    if(req.path.includes('/batches/'))return response({project_id:p.id,complete:true,items:[]});
    if(req.path.endsWith('/batch-features'))return response({selection_version:2});
  }});
  await h.service.restore();await flush();h.local.setItem('t8director.batch:'+p.id,'old');slow=true;
  h.setSelected([p.current]);const running=h.click({service:'generate-selected'});await flush();assert.ok(release);
  h.setSelected(['other']);release();await running;
  const body=h.requests.find(r=>r.path.endsWith('/batches')&&r.method==='POST').body;
  assert.deepEqual(body.shot_ids,[p.current]);
});

test('rejected copy allows corrected draft but keeps its one target identity',async()=>{
  const rows=[];let reject=true;
  const h=harness({route:(req,response)=>{
    if(req.path.includes('/projects/')&&req.method==='POST'){
      rows.push(req.body.project);
      if(reject)return response({error:'invalid title'},400);
    }
  }});
  await h.service.restore();await h.click({service:'copy'});reject=false;
  h.doc().shots[0].simplePrompt='corrected draft';await h.click({service:'copy'});
  assert.equal(rows[0].id,rows[1].id);assert.equal(rows[1].doc.shots[0].simplePrompt,'corrected draft');
  assert.equal(h.service.envelope().id,rows[0].id);
});

test('save works during generation without changing frozen generation',async()=>{
  let running=true;
  const h=harness({route:(req,response)=>{
    if(req.path.endsWith('/generate'))return response({prompt_id:'active'});
    if(req.path.includes('/jobs/'))return response(running?{state:'running'}:video);
  }});
  await h.service.restore();const generating=h.click({action:'generate'});await flush();
  h.doc().shots[0].simplePrompt='edited while generating';await h.click({service:'save'});
  assert.equal(h.server.size,1);assert.equal([...h.server.values()][0].doc.shots[0].simplePrompt,'edited while generating');
  assert.notEqual(h.requests.find(r=>r.path.endsWith('/generate')).body.project.doc.shots[0].simplePrompt,'edited while generating');
  running=false;await [...h.intervals.values()][0]();await generating;
});

test('restricted sessionStorage does not crash service initialization or remote save',async()=>{
  const h=harness();const descriptor=Object.getOwnPropertyDescriptor(globalThis,'sessionStorage');
  try{
    Object.defineProperty(globalThis,'sessionStorage',{configurable:true,get(){throw new DOMException('restricted','SecurityError')}});
    const service=makeDirectorServices(h.ctx);await service.restore();await h.click({service:'save'});
    assert.equal(h.server.size,1);assert.ok(h.messages.some(m=>m.type==='t8-director:saved'));
  }finally{Object.defineProperty(globalThis,'sessionStorage',descriptor);}
});
