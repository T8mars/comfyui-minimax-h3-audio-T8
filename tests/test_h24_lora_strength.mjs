import test from 'node:test';
import assert from 'node:assert/strict';
import {createSamplingDialog} from '../web/director/sampling_ui.mjs';

function fixture(mode, strength) {
    const row = {id:'row',name:'user.safetensors',strength,enabled:true};
    const sampling = mode === 'single'
        ? {mode, lora_mode:'manual', loras:[row]}
        : {mode, low_loras:[row],high_loras:[],hyperflow_file:'hf.safetensors',variant:'single8'};
    const current = {id:'shot',samplingInherit:true};
    const project = {generation:{},sampling,shots:[current]};
    const errors = [], listeners = {}, applied = [];
    const body = {innerHTML:'',insertAdjacentHTML:(_where,text)=>errors.push(text)};
    const field = {setAttribute(){},focus(){},insertAdjacentHTML:(_where,text)=>errors.push(text)};
    const dialog = {
        querySelector:selector=>selector==='[data-model-settings]'?body:field,
        addEventListener:(name,handler)=>{listeners[name]=handler;},
        showModal(){}, close(){},
    };
    const ui = createSamplingDialog({
        root:{querySelector:()=>dialog},doc:()=>project,shot:()=>current,
        catalog:()=>({hyperflow:[{value:'hf.safetensors',label:'HF'}],lora:[{value:'user.safetensors',label:'User'}]}),
        apply:value=>applied.push(value),notify(){},
    });
    ui.open();
    return {ui,body,errors,applied,listeners};
}
for(const mode of ['single','two_pass','hyperflow'])for(const strength of [3,-3,12,-12,101,-101,1e6]) {
    test(`${mode} commits selected strength ${strength}`,()=>{
        const f=fixture(mode,strength);
        assert.doesNotMatch(f.body.innerHTML, /min="-2"|max="2"/);
        f.ui.commit();
        assert.equal(f.errors.length,0);
        assert.equal(f.applied.length,1);
        const rows=f.applied[0].global[mode==='single'?'loras':'low_loras'];
        assert.equal(rows[0].strength,strength);
    });
}
for(const strength of [NaN,Infinity,-Infinity,null,'3']) {
    test(`dialog retains invalid-strength feedback: ${strength}`,()=>{
        const f=fixture('single',strength);
        f.ui.commit();
        assert.equal(f.applied.length,0);
        assert.equal(f.errors.length,1);
        assert.match(f.errors[0], /有限数字/);
    });
}
test('cancel discards the draft even with a large finite strength',()=>{
    const f=fixture('two_pass',101);
    f.listeners.cancel({});
    f.ui.commit();
    assert.equal(f.applied.length,0);
});

// Run the actual legacy page handlers, not a rewritten clamp stand-in.
const {readFileSync} = await import('node:fs');
const {runInNewContext} = await import('node:vm');
const page = readFileSync(new URL('../web/director/index.html',import.meta.url),'utf8');
function legacyFixture(scalar, value) {
    const source = page.split(/\r?\n/).find(line => scalar
        ? line.includes("else if(key==='lora_strength')")
        : line.startsWith("root.addEventListener('change',e=>{const el=e.target;const i=Number(el.dataset.loraName"));
    assert.ok(source,'real page handler exists');
    const cfg = {loras:[{name:'user.safetensors',strength:1,enabled:true}],lora_strength:1};
    const messages=[],state={checkpoint:0,persist:0};
    let handler;
    runInNewContext(source,{
        root:{addEventListener:(_name,callback)=>{handler=callback;}},
        generationCfg:()=>cfg,doc:{shots:[]},
        checkpoint:()=>state.checkpoint++,persist:()=>state.persist++,
        renderModelSettings(){},render(){},notify:message=>messages.push(message),
    });
    const target = {
        value, dataset:scalar?{generationField:'lora_strength'}:{loraStrength:'0'},
        matches:selector=>scalar?selector==='[data-generation-field]':
            selector==='[data-lora-strength]'||selector==='[data-lora-name],[data-lora-strength],[data-lora-enabled]',
    };
    handler({target,stopImmediatePropagation(){}});
    return {cfg,state,messages};
}
for(const scalar of [false,true])for(const strength of ['3','-3','101','-101']) {
    test(`real legacy ${scalar?'scalar':'row'} never silently clamps ${strength}`,()=>{
        const f=legacyFixture(scalar,strength);
        assert.equal(scalar?f.cfg.lora_strength:f.cfg.loras[0].strength,Number(strength));
        assert.equal(f.state.persist,1);
    });
}
for(const scalar of [false,true])for(const strength of ['','NaN','Infinity','-Infinity']) {
    test(`real legacy ${scalar?'scalar':'row'} rejects ${JSON.stringify(strength)} before mutation`,()=>{
        const f=legacyFixture(scalar,strength);
        assert.equal(scalar?f.cfg.lora_strength:f.cfg.loras[0].strength,1);
        assert.equal(f.state.persist,0);
        assert.equal(f.state.checkpoint,0);
        assert.match(f.messages[0],/有限数字/);
    });
}
