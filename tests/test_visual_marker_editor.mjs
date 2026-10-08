import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { normalizedBox, rgbSHA, setDraft, validateSpec } from "../web/visual_marker_editor.mjs";

const spec = () => ({markers:[
    {marker_id:"A",kind:"actor",role_id:"A",color:"cyan",description:"person",xyxy:[.1,.1,.4,.9]},
    {marker_id:"T",kind:"target",color:"cyan",description:"floor",xyxy:[.7,.6,.9,.9]}],
    relations:[{actor_marker_id:"A",target_marker_id:"T",action:"walk once"}]});

test("reverse drag uses actual source dimensions and bounded nonzero box",()=>{
    assert.deepEqual(normalizedBox([80,45],[20,10],100,50),[.2,.2,.8,.9]);
    assert.deepEqual(normalizedBox([-10,-10],[120,80],100,50),[0,0,1,1]);
    assert.throws(()=>normalizedBox([10,10],[10,20],100,50));
    assert.throws(()=>normalizedBox([NaN,10],[20,20],100,50));
    assert.throws(()=>normalizedBox([10],[20,20],100,50));
});
test("same color/shared target legal and handdrawing coordinates optional",()=>{
    const value=spec(); value.markers.push({...value.markers[0],marker_id:"B",role_id:"B"});
    value.relations.push({...value.relations[0],actor_marker_id:"B"});
    assert.equal(validateSpec(value),value);
    for (const marker of value.markers) delete marker.xyxy;
    assert.equal(validateSpec(value,"provided_marked"),value);
    assert.throws(()=>validateSpec(value));
    const malformed=spec(); malformed.markers[0].marker_id=1;
    assert.throws(()=>validateSpec(malformed));
});
test("only real serialized widgets changed, no queue or mode mutation",()=>{
    let dirties=0;
    const node={widgets:[{name:"mode",value:"render_rectangles"},{name:"markers_json",value:"old"},
        {name:"expected_source_sha256",value:""}],setDirtyCanvas:()=>{dirties++;},
        queuePrompt:()=>{throw Error("must not queue");}};
    const sha="a".repeat(64);
    const result=setDraft(node,spec(),sha);
    assert.deepEqual(JSON.parse(result),spec()); assert.equal(node.widgets[2].value,sha);
    assert.equal(node.widgets[0].value,"render_rectangles"); assert.equal(dirties,1);
    const reopened=JSON.parse(JSON.stringify(node.widgets));
    assert.equal(reopened[1].value,result); assert.equal(reopened[2].value,sha);
    assert.throws(()=>setDraft(node,spec(),"bad"));
});
test("browser preview SHA matches actual float32 RGB tensor byte order",async()=>{
    const pixels={width:2,height:1,data:new Uint8ClampedArray([0,223,255,255,63,128,192,255])};
    const rgb=new Float32Array([0,223/255,1,63/255,128/255,192/255]);
    const expected=createHash("sha256").update(new Uint8Array(rgb.buffer)).digest("hex");
    assert.equal(await rgbSHA(pixels),expected);
});
