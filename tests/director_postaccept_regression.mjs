import assert from 'node:assert/strict';
import test from 'node:test';
import {harness,project,flush} from './director_audit20_harness.mjs';
globalThis.confirm=()=>true;

for(const switched of [false,true])test(`needs-review ends tracking without success or automatic resubmission, switched=${switched}`,async()=>{
 const b=project('bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb');let state='running',submitted=0,finished=false;
 let acknowledge;const submittedOnce=new Promise(resolve=>{acknowledge=resolve});
 const h=harness({route:(r,response)=>{
  if(r.path.endsWith('/projects/'+b.id))return response(b);
  if(r.path.endsWith('/generate')){submitted++;acknowledge();return response({prompt_id:'damaged-receipt'});}
  if(r.path.includes('/jobs/'))return response({state,outputs:{},integrity_error:'Original receipt needs review'});
 }});
 await h.service.restore();const original=JSON.stringify(h.doc());
 const run=h.click({action:'generate'}).then(()=>{finished=true});await submittedOnce;await flush();
 if(switched)await h.click({openProject:b.id});
 assert.equal(h.intervals.size,1,JSON.stringify({notices:h.notices,requests:h.requests,finished}));
 state='needs_review';await [...h.intervals.values()][0]();await flush();
 assert.equal(h.intervals.size,0);assert.equal(finished,true);assert.equal(submitted,1);
 assert.equal(h.session.getItem('t8director.activeJob:audit'),null);
 assert.ok(h.notices.some(x=>x.includes('核对')));
 if(switched){assert.equal(h.service.envelope().id,b.id);assert.equal(h.versions().length,0);}
 else {assert.equal(h.displayed().state,'needs_review');assert.equal(JSON.stringify(h.doc()),original);}
 assert.ok(!h.notices.some(x=>x.includes('已生成')));
 await run;await h.click({action:'generate'});assert.equal(submitted,2);
});

test('Core cancellation ends tracking, releases generation and never auto-resubmits',async()=>{
 let state='running',submitted=0,finished=false;
 let acknowledge;const submittedOnce=new Promise(resolve=>{acknowledge=resolve});
 const h=harness({route:(r,response)=>{
  if(r.path.endsWith('/generate')){submitted++;acknowledge();return response({prompt_id:'cancelled-elsewhere'});}
  if(r.path.includes('/jobs/'))return response({state,outputs:{}});
 }});
 await h.service.restore();
 const run=h.click({action:'generate'}).then(()=>{finished=true});await submittedOnce;await flush();
 state='cancelled';await [...h.intervals.values()][0]();await flush();
 assert.equal(h.intervals.size,0);assert.equal(finished,true);
 assert.equal(h.session.getItem('t8director.activeJob:audit'),null);
 assert.equal(submitted,1);assert.equal(h.displayed().state,'cancelled');
 assert.ok(h.notices.some(text=>text.includes('已取消')&&!text.includes('失败')));
 assert.ok(!h.notices.some(text=>text.startsWith('生成失败')));
 await run;
 // Only a fresh explicit click may submit again, proving busy was released.
 await h.click({action:'generate'});assert.equal(submitted,2);
});

for(const roundtrip of [false,true])test(`batch invalidates old context after project switch, roundtrip=${roundtrip}`,async()=>{
 const a=project(),b=project('bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb');let release,blocked=false,continues=0;
 const status={id:'batch-roundtrip',project_id:a.id,complete:false,next_index:0,items:[{shot_id:a.current,state:'not_submitted'}]};
 const h=harness({route:(r,response)=>{
  if(r.path.endsWith('/projects/'+b.id))return response(b);
  if(r.path.endsWith('/projects/'+a.id))return response(a);
  if(r.path.endsWith('/batches/batch-roundtrip/continue')){continues++;return response({prompt_id:'job'});}
  if(r.path.endsWith('/batches/batch-roundtrip')){if(!blocked){blocked=true;return new Promise(resolve=>{release=()=>resolve(response(status))});}return response(status);}
  if(r.path.includes('/jobs/'))return response({state:'error'});
 }});
 await h.service.restore();h.local.setItem('t8director.batch:'+a.id,status.id);
 const running=h.click({service:'resume-batch'});await flush();
 await h.click({openProject:b.id});if(roundtrip)await h.click({openProject:a.id});
 release();await running;assert.equal(continues,0);
});

for(const kind of ['error','noop'])test(`unconfirmed cancellation ${kind} keeps tracking unknown`,async()=>{
 let state='running';
 let acknowledge;const submittedOnce=new Promise(resolve=>{acknowledge=resolve});
 const h=harness({route:(r,response)=>{
  if(r.path.endsWith('/generate')){acknowledge();return response({prompt_id:'cancel-test'});}
  if(r.path.endsWith('/cancel')){if(kind==='error')throw Error('Network failed');return response({deleted_from_queue:false,interrupted:false});}
  if(r.path.includes('/jobs/'))return response({state});
 }});
 await h.service.restore();const running=h.click({action:'generate'});await submittedOnce;await flush();
 await h.click({service:'cancel-job'});state='unknown';await [...h.intervals.values()][0]();
 assert.ok(h.session.getItem('t8director.activeJob:audit'));
 assert.ok(!h.notices.some(x=>x.startsWith('已取消当前任务')));
 state='error';await [...h.intervals.values()][0]();await running;
});

test('quota recovery requires explicit export and confirmation; new edits invalidate it',async()=>{
 const b=project('bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb');
 const h=harness({route:(r,response)=>r.path.endsWith('/projects/'+b.id)?response(b):undefined});
 await h.service.restore();const original=h.service.envelope().id;
 h.local.setItem=()=>{throw Error('quota')};
 await h.click({openProject:b.id});assert.equal(h.service.envelope().id,original);
 assert.match(h.$('[data-dialog]').innerHTML,/backup-export/);
 await h.click({service:'backup-continue'});assert.equal(h.service.envelope().id,original);
 await h.click({service:'backup-export'});
 h.doc().shots[0].simplePrompt='new edit';
 await h.click({service:'backup-continue'});assert.equal(h.service.envelope().id,original);
 await h.click({openProject:b.id});await h.click({service:'backup-export'});
 await h.click({service:'backup-continue'});assert.equal(h.service.envelope().id,b.id);
});

test('editor recovery is scoped by project and field, never applies to document',async()=>{
 const a=project(),b=project('bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb');
 const h=harness({route:(r,response)=>r.path.endsWith('/projects/'+b.id)?response(b):r.path.endsWith('/projects/'+a.id)?response(a):undefined});
 await h.service.restore();const original=JSON.stringify(h.doc());
 const value={text:'draft',shared:[],tokens:[]};
 assert.equal(h.service.editorRecovery('global',value),true);
 assert.equal(JSON.stringify(h.doc()),original);
 assert.deepEqual(h.service.editorRecovery('global'),value);
 assert.equal(h.service.editorRecovery('other-field'),null);
 await h.click({openProject:b.id});assert.equal(h.service.editorRecovery('global'),null);
 await h.click({openProject:a.id});assert.deepEqual(h.service.editorRecovery('global'),value);
 h.service.editorRecovery('global',null);assert.equal(h.service.editorRecovery('global'),null);
});

test('malformed editor recovery is ignored instead of breaking editor',async()=>{
 const h=harness();await h.service.restore();
 h.local.setItem('t8director.editors:audit','{broken');assert.equal(h.service.editorRecovery('global'),null);
 h.service.editorRecovery('global',{text:'x',shared:[],tokens:[42]});assert.equal(h.service.editorRecovery('global'),null);
});

test('late batch creation response does not renew generation intent after A-B-A',async()=>{
 const a=project(),b=project('bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb');let release,created,continues=0;
 const h=harness({route:(r,response)=>{
  if(r.path.endsWith('/projects/'+a.id))return response(a);
  if(r.path.endsWith('/projects/'+b.id))return response(b);
  if(r.path.endsWith('/batches')&&r.body){created=r.body;return new Promise(resolve=>release=()=>resolve(response({})));}
  if(r.path.endsWith('/continue')){continues++;return response({prompt_id:'job'});}
  if(created&&r.path.endsWith('/batches/'+created.batch_id))return response({id:created.batch_id,project_id:a.id,complete:false,next_index:0,items:[{shot_id:a.current,state:'not_submitted'}]});
  if(r.path.includes('/jobs/'))return response({state:'error'});
 }});
 await h.service.restore();const running=h.click({action:'generate-all'});
 for(let i=0;i<10&&!release;i++)await flush();assert.ok(release);
 await h.click({openProject:b.id});await h.click({openProject:a.id});
 release();await running;assert.equal(continues,0);
 assert.equal(h.local.getItem('t8director.batch:'+a.id),created.batch_id);
});
