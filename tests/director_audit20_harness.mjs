// Isolated CPU harness. The actual session module executes against in-memory
// persistence and HTTP substitutes; no browser, Core, network or GPU is touched.
import {webcrypto} from 'node:crypto';
import {makeDirectorServices, mergeDirectorResults, selectDirectorResult} from '../web/director/session.mjs';
export const clone=structuredClone;
export const flush=()=>new Promise(resolve=>setImmediate(resolve));
export function storage(initial={}) { const values=new Map(Object.entries(initial));return {getItem:k=>values.get(k)??null,setItem:(k,v)=>values.set(k,String(v)),removeItem:k=>values.delete(k),dump:()=>Object.fromEntries(values)}; }
export function project(id='11111111-1111-4111-8111-111111111111') { return {schema:'t8.minimax_h3.director_project',version:2,id,revision:1,title:'Audit project',current:'22222222-2222-4222-8222-222222222222',assets:[],doc:{global:'',sharedRefs:[],sampling:{mode:'single'},shots:[{id:'22222222-2222-4222-8222-222222222222',name:'First',seed:23,simplePrompt:'Audit',writingMode:'simple',mode:'text',sound:'native',events:[],refs:[],tray:[],duration:4,rev:1}]}}; }
export function harness({initial=project(),local=storage(),session=storage({'t8director.tab':'audit'}),server=new Map(),results=new Map(),route=null}={}) {
  globalThis.crypto??=webcrypto;globalThis.localStorage=local;globalThis.sessionStorage=session;
  globalThis.location={href:'http://audit.invalid/ui?project_id='+initial.id,origin:'http://audit.invalid'};
  local.setItem('t8director.draft:'+session.getItem('t8director.tab'),JSON.stringify(initial));
  const notices=[],requests=[],messages=[],listeners={},timers=[],intervals=new Map(),nodes=new Map();let timer=0;
  globalThis.window={addEventListener(){},history:{state:null,replaceState(_a,_b,url){location.href=String(url)}}};
  globalThis.parent={postMessage:m=>messages.push(m)};
  globalThis.MutationObserver=class{observe(){}};
  globalThis.setTimeout=fn=>{timers.push(fn);return ++timer};globalThis.setInterval=fn=>{intervals.set(++timer,fn);return timer};globalThis.clearInterval=id=>intervals.delete(id);
  const node=()=>({dataset:{},textContent:'',hidden:true,open:false,innerHTML:'',click(){},insertAdjacentHTML(){},append(){},querySelector(){return null},close(){this.open=false},addEventListener(){}});
  globalThis.document={createElement:node};
  const $=s=>{if(!nodes.has(s))nodes.set(s,node());return nodes.get(s)};
  const root={...node(),querySelectorAll:()=>[],append(){},addEventListener:(name,fn)=>{listeners[name]=fn}};
  let doc=clone(initial.doc),current=initial.current,assets=new Map(),versions=[],selected=[];
  const response=(data,status=200)=>({ok:status<400,status,json:async()=>clone(data)});
  globalThis.fetch=async(url,options)=>{
    const path=new URL(url).pathname,body=options.body?JSON.parse(options.body):null;
    const req={path,method:options.method,body};requests.push(req);
    if(route){const custom=await route(req,response);if(custom)return custom;}
    if(path.includes('/projects/')){
      const key=path.split('/').at(-1);
      if(body){const saved={...clone(body.project),revision:body.expected_revision+1};server.set(key,saved);return response(saved);}
      return response(server.get(key)||initial);
    }
    if(path.includes('/results/'))return response({results:results.get(path.split('/').at(-1))||[]});
    if(path.endsWith('/batch-features'))return response({selection_version:2});
    if(path.endsWith('/compile'))return response({ready:true});
    return response({});
  };
  const ctx={root,$,esc:String,notify:m=>notices.push(m),showDialog:(title,html)=>{const dialog=$('[data-dialog]');dialog.open=true;dialog.innerHTML=html;},tokenMap:()=>new Map(),checkpoint(){},doc:()=>doc,current:()=>current,assets:()=>assets,versions:()=>versions,selectedShotIds:()=>selected,replace:(d,c,a)=>{doc=d;current=c;assets=a;versions=[];selected=[];},setResults:items=>{versions=mergeDirectorResults(versions,items)},recordResult:item=>{versions=mergeDirectorResults(versions,[item])},clearResults(){versions=[];selected=[];},resetHistory(){},render(){}};
  const service=makeDirectorServices(ctx);
  const click=dataset=>listeners.click({target:{closest:()=>({dataset,disabled:false})},preventDefault(){},stopImmediatePropagation(){}});
  return {ctx,service,click,notices,requests,messages,doc:()=>doc,versions:()=>versions,displayed:()=>selectDirectorResult(versions.filter(r=>r.shot_id===current),doc.shots.find(s=>s.id===current)?.adoptedResultId),local,session,server,$,intervals,timers,setCurrent:v=>{current=v},setSelected:v=>{selected=v},input:target=>listeners.input({target})};
}
