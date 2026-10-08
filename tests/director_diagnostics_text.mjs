import test from 'node:test';
import assert from 'node:assert/strict';
import {candidateDiagnosticText} from '../web/director/candidate_diagnostics.mjs';

test('candidate explanation distinguishes stale components, timing plan and cache proof',()=>{
    const report = {candidate:{status:'stale',changed_components:['assets','references']},
        time_plan:{requested_seconds:4,requested_frames:96,aligned_frames:107,generated_seconds:107/24,fps:24,delivery_trim_frames:96},
        reference_slots:[{asset_id:'video-id',role:'ref_video',native:'<Video 1>'},
            {asset_id:'video-id',role:'ref_video_audio',native:'<Audio 1>'}],
        reference_aliases:[{asset_id:'video-id',alias:'@ref1',native:'<Video 1>'}]};
    const before = structuredClone(report), text = candidateDiagnosticText(report);
    assert.match(text,/候选已失效/); assert.match(text,/素材的内容 SHA；参考绑定／顺序/);
    assert.match(text,/请求 4 秒／96 帧；对齐 107 帧／4.458333 秒/);
    assert.match(text,/<Video 1> · ref_video · 素材 video-id · @ref1/);
    assert.match(text,/<Audio 1> · ref_video_audio · 素材 video-id\n/);
    assert.match(text,/不是实测源 PTS/); assert.match(text,/不是 MODEL／Stage 缓存命中证明/);
    assert.deepEqual(report,before);
});
test('legacy and inconsistent receipts never invent exact change attribution',()=>{
    assert.match(candidateDiagnosticText({status:'stale',changed_components:null,component_snapshot:'legacy_missing'}),/不猜历史差异/);
    assert.match(candidateDiagnosticText({status:'active',changed_components:null,component_snapshot:'inconsistent_receipt_declaration'}),/不能准确归因/);
});
test('arbitrary strings stay literal, not executed or inserted as HTML',()=>{
    const text = candidateDiagnosticText({candidate:{status:'pending',changed_components:[]},
        reference_slots:[{native:'<script>not executable</script>',role:'ref_image',asset_id:'safe-id'}]});
    assert.match(text,/<script>not executable<\/script>/);
});
