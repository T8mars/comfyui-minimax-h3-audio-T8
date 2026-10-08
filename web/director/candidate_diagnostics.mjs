// Text only: no DOM, requests, mutable project state or runtime cache inference.
const components = {schema:'输入格式',global:'全片设定',draft:'本镜原作者稿／用途',
    references:'参考绑定／顺序',assets:'被引用素材的内容 SHA',evidence:'源证据／确认',
    intent:'创作意图',dependencies:'明确的跨镜依赖',bindings:'有效规则／顺序',options:'候选编译选项'};
const states = {legacy:'无候选，仍用原作者稿',pending:'待审，尚未采用',active:'已显式采用，输入仍一致',
    stale:'候选已失效，请核对后重新编译'};
const seconds = value => Number.isFinite(value) ? String(Number(value.toFixed(6))) : '未知';

export function candidateDiagnosticText(report) {
    const candidate = report.candidate || report;
    const lines = ['候选：' + (states[candidate.status] || '状态未知')];
    if (Array.isArray(candidate.changed_components)) {
        lines.push(candidate.changed_components.length ? '改变的输入：' + candidate.changed_components.map(key => components[key] || key).join('；') : '已记录的输入未变化。');
    } else if (candidate.component_snapshot === 'inconsistent_receipt_declaration') {
        lines.push('总签名与分字段声明不一致，不能准确归因；不自动修复或重签。');
    } else {
        lines.push('旧候选没有分字段快照，不猜历史差异。');
    }
    const timing = report.time_plan;
    if (timing) {
        lines.push(`计划时长：请求 ${seconds(timing.requested_seconds)} 秒／${timing.requested_frames} 帧；对齐 ${timing.aligned_frames} 帧／${seconds(timing.generated_seconds)} 秒（${timing.fps} fps）；交付裁齐 ${timing.delivery_trim_frames} 帧。`);
        lines.push('以上来自当前草稿编译计划，不是实测源 PTS 或生成结果。');
    }
    if (Array.isArray(report.reference_slots)) {
        lines.push('生效参考编号（计划）：');
        if (!report.reference_slots.length) lines.push('本镜没有生效参考槽。');
        for (const row of report.reference_slots) {
            const aliases = (report.reference_aliases || []).filter(item => item.asset_id === row.asset_id && item.native === row.native).map(item => item.alias);
            lines.push(`${row.native || '未知编号'} · ${row.role || '未知用途'} · 素材 ${row.asset_id || '未知'}${aliases.length ? ' · ' + aliases.join(' / ') : ''}`);
        }
    }
    lines.push('未采用、未保存、未排队；这不是 MODEL／Stage 缓存命中证明。');
    return lines.join('\n');
}
