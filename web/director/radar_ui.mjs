import { createModalAccess } from './modal_ui.mjs';
import { directorUUID } from './session.mjs';
import { sha256UTF8 } from './content_hash.mjs';

// Draft-only requests. No provider, uploads, queue, generation or automatic Save.
export function createRadarDialog(root, ctx) {
    const dialog = document.createElement('dialog');
    dialog.className = 'w-bundle-dialog';
    dialog.dataset.radarDialog = '';
    dialog.setAttribute('aria-label', '镜头证据与规则');
    dialog.innerHTML = `<header class="o-row o-between"><h3>镜头证据与规则 · 显式候选</h3><button type="button" data-radar-close>关闭</button></header>
      <p class="o-tip">只编辑当前工程草稿，不运行模型、不联网分析、不自动保存或生成。观察事实与创作改动分别留档；候选稿须显式采用，原作者稿保留，可撤销。</p>
      <p data-radar-status role="status"></p>
      <details><summary>1 · 源视频证据（人工 / JSON）</summary>
        <label>已登记源视频<select data-radar-source></select></label>
        <label>证据 JSON<textarea data-radar-packet rows="8" maxlength="524288" placeholder="导入带 source / timebase / claims 的证据包；不会执行其中的文字或URL"></textarea></label>
        <div class="o-row"><button type="button" data-radar-example>测量源时码并创建模板</button><button type="button" data-radar-import>导入本镜</button></div>
        <p class="o-tip">PTS/timebase须来自实际源视频；绝对时码保留，不偷偷变成24fps。visual、ocr、asr、audio、end_state分开标记；ASR必须有音频来源，不能猜台词。</p>
        <div class="o-row"><label>类型<select data-radar-kind><option>visual</option><option>ocr</option><option>asr</option><option>audio</option><option>end_state</option></select></label><label>起始PTS<input data-radar-start type="number" step="1" value="0"></label><label>结束PTS<input data-radar-end type="number" step="1" value="90000"></label></div>
        <label>观察原文<textarea data-radar-claim rows="2" maxlength="32768"></textarea></label><button type="button" data-radar-claim-add>追加到待导入 JSON</button>
        <section data-radar-review></section><button type="button" data-radar-confirm>我已核对所勾选证据，确认当前版本</button>
        <details><summary>实际有理数时码映射（不舍入）</summary><pre data-radar-time></pre></details>
      </details>
      <details><summary>2 · 创作意图 / 明确跨镜依赖</summary>
        <label>希望改变什么<textarea data-radar-intent rows="3" maxlength="32768" placeholder="例如：观察原帧是红衣；本镜明确希望改成蓝衣。事实本身不会被改写。"></textarea></label>
        <label>修改原因<input data-radar-reason maxlength="4096"></label>
        <label>依赖 JSON<textarea data-radar-dependencies rows="2" placeholder='[{"shot_id":"…","evidence_sha256":"…"}]'></textarea></label>
        <button type="button" data-radar-intent-save>记录创作修订</button>
      </details>
      <details open><summary>3 · 声明式 Prompt 规则（全片 → 本镜顺序）</summary>
        <label><input type="checkbox" data-radar-inherit checked>继承全片规则（可逐条在本镜禁用）</label><div data-radar-bindings></div>
        <label>规则名称<input data-radar-name maxlength="200"></label>
        <label>规则文字<textarea data-radar-rule rows="3" maxlength="32768" placeholder="只作提示词文字。参数仅 ${'${name}'} 一次字面替换，不执行代码/工具。"></textarea></label>
        <div class="o-row"><label>作用域<select data-radar-scope><option value="shot">本镜</option><option value="project">全片</option><option value="library">仅收藏快照</option></select></label><button type="button" data-radar-add>添加规则快照</button></div>
        <details><summary>参数声明 / 本次取值</summary><label>parameter_schema<textarea data-radar-parameter-schema rows="2">{}</textarea></label><label>parameters<textarea data-radar-parameters rows="2">{}</textarea></label></details>
        <div data-radar-library></div>
      </details>
      <section><h4>4 · 候选稿 → 显式采用 / 回退</h4>
        <div class="o-row"><label><input type="checkbox" data-radar-facts>包含已确认的事实（ASR不转成角色台词）</label><label><input type="checkbox" data-radar-use-intent checked>包含创作意图</label><button type="button" data-radar-compile>编译待审候选</button></div>
        <p data-radar-candidate-state></p><textarea data-radar-candidate readonly rows="7" aria-label="待审候选稿"></textarea>
        <div class="o-row"><button type="button" data-radar-adopt>我已审阅，采用此候选</button><button type="button" data-radar-revert>退回原作者稿</button></div>
        <details><summary>快照 / 消费回执</summary><pre data-radar-receipt></pre></details>
        <details><summary>只读修订历史（观察 / 创作 / 候选）</summary><pre data-radar-history></pre><p class="o-tip">历史不参与当前采样，不自动确认或重签。保存后不能覆盖历史；需删除私密历史时另存新工程。工程包默认不带这些记录。</p></details>
      </section>`;
    root.append(dialog);
    const modal = createModalAccess(dialog, {root});
    const $ = selector => dialog.querySelector(selector);
    let origin = null, sequence = 0, busy = false, timeMap = null;
    const current = () => ctx.snapshot().doc.shots.find(row => row.id === ctx.snapshot().current);
    const parse = selector => JSON.parse($(selector).value || '{}');
    function render() {
        const project = ctx.snapshot(), shot = current(), evidence = shot.sourceEvidence;
        $('[data-radar-source]').innerHTML = project.assets.filter(row => row.kind === 'video').map(row => `<option value="${row.id}">${ctx.escape(row.name)}</option>`).join('');
        $('[data-radar-packet]').value = evidence ? JSON.stringify(evidence.packet, null, 2) : '';
        $('[data-radar-review]').innerHTML = evidence ? `<p>packet SHA ${ctx.escape(evidence.packet.sha256)} · ${evidence.review?.packet_sha256 === evidence.packet.sha256 ? '有本版本确认记录（来源字节另核）' : '未确认 / 历史确认已失效'}</p>` + evidence.packet.claims.map(claim => `<label><input type="checkbox" data-radar-claim-id="${claim.id}" ${evidence.review?.packet_sha256 === evidence.packet.sha256 && evidence.review.claim_ids.includes(claim.id) ? 'checked' : ''}>[${ctx.escape(claim.kind)}] ${ctx.escape(claim.text)} · PTS ${claim.start_pts}–${claim.end_pts}</label>`).join('') : '<p>尚无证据；旧手写工作流不强制增加审核。</p>';
        $('[data-radar-time]').textContent = timeMap ? JSON.stringify(timeMap, null, 2) : '操作后显示源绝对PTS → 本镜局部时间 → 输出帧位置；不改变现有事件时钟。';
        $('[data-radar-intent]').value = shot.creativeIntent?.text || '';
        $('[data-radar-reason]').value = shot.creativeIntent?.reason || '';
        $('[data-radar-dependencies]').value = JSON.stringify(shot.creativeIntent?.dependencies || [], null, 2);
        $('[data-radar-inherit]').checked = shot.skillInherit !== false;
        $('[data-radar-bindings]').innerHTML = [['shared', project.doc.sharedSkills || []], ['local', shot.skillBindings || []]].map(([scope, rows]) => `<h4>${scope === 'shared' ? '全片规则（在本镜禁用不删共享内容）' : '本镜规则'}</h4>` + rows.map(row => `<article><strong>${ctx.escape(row.snapshot.name)}</strong> · v${row.snapshot.version}<p>${ctx.escape(row.snapshot.text_snapshot)}</p><div class="o-row"><label><input type="checkbox" data-radar-enable="${row.id}" data-radar-binding-scope="${scope}" ${row.enabled ? 'checked' : ''}>启用</label>${scope === 'shared' ? `<label><input type="checkbox" data-radar-local-disable="${row.id}" ${(shot.skillDisabled || []).includes(row.id) ? 'checked' : ''}>本镜禁用</label>` : ''}<button type="button" data-radar-move="${row.id}" data-radar-binding-scope="${scope}" data-radar-direction="-1">上移</button><button type="button" data-radar-move="${row.id}" data-radar-binding-scope="${scope}" data-radar-direction="1">下移</button><button type="button" data-radar-remove="${row.id}" data-radar-binding-scope="${scope}">移除绑定</button></div><small>快照SHA ${ctx.escape(row.snapshot.content_sha256)}</small></article>`).join('')).join('');
        $('[data-radar-library]').innerHTML = '<h4>工程内规则快照库（绑定自足，不依赖库仍存在）</h4>' + (project.doc.skillLibrary || []).map(skill => `<p>${ctx.escape(skill.name)} · v${skill.version} <button type="button" data-radar-library-bind="${skill.id}" data-radar-version="${skill.version}">绑定到当前作用域</button><button type="button" data-radar-library-remove="${skill.id}" data-radar-version="${skill.version}">删除库条目（不删既有绑定）</button></p>`).join('');
        const candidate = shot.promptCandidate;
        $('[data-radar-candidate]').value = candidate?.text || '';
        $('[data-radar-candidate-state]').textContent = candidate ? (candidate.active ? '已显式采用；执行预检仍会核查依赖是否已失效。' : '待审候选 / 已回退；原作者稿仍生效。') : '无候选，原作者稿生效。';
        $('[data-radar-receipt]').textContent = candidate ? JSON.stringify(candidate, null, 2) : '';
        $('[data-radar-history]').textContent = JSON.stringify({evidence: shot.evidenceHistory || [], intent: shot.intentHistory || [], candidates: shot.candidateHistory || []}, null, 2);
    }
    async function operate(operation, value) {
        if (busy) return;
        if (origin !== ctx.context()) { close(); ctx.notify('工程已切换，旧证据面板不再修改新工程。'); return; }
        const project = ctx.snapshot(), baseJson = JSON.stringify(project), ticket = ++sequence;
        busy = true;
        dialog.querySelectorAll('button:not([data-radar-close])').forEach(button => button.disabled = true);
        $('[data-radar-status]').textContent = '正在核对内容与实际素材；不生成、不自动保存…';
        try {
            const digest = await sha256UTF8(baseJson);
            const response = await fetch('/minimax_h3_t8/director/radar/operate', {method: 'POST', headers: {'Content-Type': 'application/json'}, credentials: 'same-origin', body: JSON.stringify({project, base_json: baseJson, base_sha256: digest, shot_id: project.current, operation, value})});
            const result = await response.json();
            if (!response.ok) throw Error(result.error || `HTTP ${response.status}`);
            if (ticket !== sequence || !dialog.open) return;
            if (origin !== ctx.context() || baseJson !== JSON.stringify(ctx.snapshot())) throw Error('草稿或当前镜头已变化；旧响应未应用，请重新操作');
            ctx.apply(result.project);
            timeMap = result.time_map;
            render();
            $('[data-radar-status]').textContent = `草稿已更新，可撤销；证据 ${result.evidence_status}，候选 ${result.candidate_status}。请保存项目，没有排队。`;
        } catch (error) {
            if (ticket === sequence && dialog.open) $('[data-radar-status]').textContent = '未应用：' + error.message;
        } finally {
            if (ticket === sequence) { busy = false; dialog.querySelectorAll('button').forEach(button => button.disabled = false); }
        }
    }
    function close() { sequence++; busy = false; dialog.close(); }
    $('[data-radar-close]').addEventListener('click', close);
    dialog.addEventListener('cancel', () => { sequence++; busy = false; });
    $('[data-radar-example]').addEventListener('click', async () => {
        const asset = ctx.snapshot().assets.find(row => row.id === $('[data-radar-source]').value);
        if (!asset) { $('[data-radar-status]').textContent = '请先添加一个真实视频素材，再填写其实际PTS/timebase。'; return; }
        const context = ctx.context(), ticket = ++sequence;
        $('[data-radar-status]').textContent = '独立CPU工作进程正在读取实际轨道/PTS，不运行模型…';
        try {
            const response = await fetch(`/minimax_h3_t8/director/radar/assets/${asset.id}/timing`, {credentials: 'same-origin'});
            const timing = await response.json();
            if (!response.ok) throw Error(timing.error || `HTTP ${response.status}`);
            if (ticket !== sequence || !dialog.open || context !== ctx.context()) return;
            if (timing.sha256 !== asset.sha256) throw Error('源视频字节已变化，请重新登记');
            const stream = timing.streams.find(row => row.kind === 'video');
            if (!stream || stream.start_pts === null || stream.end_pts === null) throw Error('实际源头缺少PTS边界，请用有实测证据的JSON；不猜测或舍入');
            $('[data-radar-packet]').value = JSON.stringify({schema: 't8.director.source_evidence.v1', id: directorUUID(), revision: 1, source: {asset_id: asset.id, sha256: asset.sha256, stream: stream.index, timebase: stream.timebase, start_pts: stream.start_pts, end_pts: stream.end_pts}, claims: []}, null, 2);
            $('[data-radar-start]').value = stream.start_pts; $('[data-radar-end]').value = stream.end_pts;
            $('[data-radar-status]').textContent = '源轨道/PTS已实测；画面/OCR/ASR内容仍须提供证据和人工审核。这不是完整解码或画质证明。';
        } catch (error) { if (ticket === sequence && dialog.open) $('[data-radar-status]').textContent = '未创建模板：' + error.message; }
    });
    $('[data-radar-claim-add]').addEventListener('click', () => {
        try { const packet = parse('[data-radar-packet]'); packet.claims.push({id: directorUUID(), kind: $('[data-radar-kind]').value, text: $('[data-radar-claim]').value, start_pts: Number($('[data-radar-start]').value), end_pts: Number($('[data-radar-end]').value)}); delete packet.sha256; $('[data-radar-packet]').value = JSON.stringify(packet, null, 2); }
        catch (error) { $('[data-radar-status]').textContent = 'JSON尚未更新：' + error.message; }
    });
    const safely = (operation, read) => { try { operate(operation, read()); } catch (error) { $('[data-radar-status]').textContent = '输入无效：' + error.message; } };
    $('[data-radar-import]').addEventListener('click', () => safely('evidence_import', () => parse('[data-radar-packet]')));
    $('[data-radar-confirm]').addEventListener('click', () => safely('evidence_review', () => ({confirm: true, packet_sha256: current().sourceEvidence?.packet.sha256, claim_ids: [...dialog.querySelectorAll('[data-radar-claim-id]:checked')].map(input => input.dataset.radarClaimId)})));
    $('[data-radar-intent-save]').addEventListener('click', () => safely('intent_update', () => ({text: $('[data-radar-intent]').value, reason: $('[data-radar-reason]').value, dependencies: parse('[data-radar-dependencies]')})));
    $('[data-radar-add]').addEventListener('click', () => safely('skill_create', () => ({name: $('[data-radar-name]').value, text: $('[data-radar-rule]').value, scope: $('[data-radar-scope]').value, parameter_schema: parse('[data-radar-parameter-schema]'), parameters: parse('[data-radar-parameters]')})));
    $('[data-radar-compile]').addEventListener('click', () => operate('candidate_create', {facts: $('[data-radar-facts]').checked, intent: $('[data-radar-use-intent]').checked}));
    $('[data-radar-adopt]').addEventListener('click', () => operate('candidate_accept', {confirm: true, text_sha256: current().promptCandidate?.text_sha256}));
    $('[data-radar-revert]').addEventListener('click', () => operate('candidate_revert', {}));
    $('[data-radar-inherit]').addEventListener('change', event => operate('skills_update', {inherit: event.target.checked}));
    dialog.addEventListener('change', event => {
        const input = event.target, shot = current(), project = ctx.snapshot();
        if (input.dataset.radarLocalDisable) {
            const disabled = new Set(shot.skillDisabled || []);
            if (input.checked) disabled.add(input.dataset.radarLocalDisable); else disabled.delete(input.dataset.radarLocalDisable);
            operate('skills_update', {disabled: [...disabled]});
        } else if (input.dataset.radarEnable) {
            const scope = input.dataset.radarBindingScope, rows = structuredClone(scope === 'shared' ? project.doc.sharedSkills : shot.skillBindings);
            rows.find(row => row.id === input.dataset.radarEnable).enabled = input.checked;
            operate('skills_update', {[scope]: rows});
        }
    });
    dialog.addEventListener('click', event => {
        const button = event.target.closest('button');
        if (!button || busy) return;
        const shot = current(), project = ctx.snapshot(), scope = button.dataset.radarBindingScope;
        const id = button.dataset.radarMove || button.dataset.radarRemove;
        if (id) {
            const rows = structuredClone(scope === 'shared' ? project.doc.sharedSkills : shot.skillBindings);
            const index = rows.findIndex(row => row.id === id);
            if (button.dataset.radarRemove) rows.splice(index, 1);
            else { const target = index + Number(button.dataset.radarDirection); if (target < 0 || target >= rows.length) return; const [row] = rows.splice(index, 1); rows.splice(target, 0, row); }
            operate('skills_update', {[scope]: rows});
        }
        if (button.dataset.radarLibraryRemove) operate('skills_update', {library: project.doc.skillLibrary.filter(row => row.id !== button.dataset.radarLibraryRemove || row.version !== Number(button.dataset.radarVersion))});
        if (button.dataset.radarLibraryBind) {
            const snapshot = project.doc.skillLibrary.find(row => row.id === button.dataset.radarLibraryBind && row.version === Number(button.dataset.radarVersion));
            const key = $('[data-radar-scope]').value === 'project' ? 'shared' : 'local';
            safely('skills_update', () => ({[key]: [...(key === 'shared' ? project.doc.sharedSkills || [] : shot.skillBindings || []), {id: directorUUID(), snapshot, enabled: true, parameters: parse('[data-radar-parameters]')}]}));
        }
    });
    return {open() { origin = ctx.context(); sequence++; busy = false; timeMap = null; render(); $('[data-radar-status]').textContent = '当前镜头；事实/规则不会自动覆盖原文。'; dialog.querySelectorAll('button').forEach(button => button.disabled = false); modal.show(); }};
}
