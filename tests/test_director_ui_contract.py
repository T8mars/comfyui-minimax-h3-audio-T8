from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_director_ui_keeps_the_frozen_beginner_and_advanced_controls():
    html = (ROOT / "web" / "director" / "index.html").read_text(encoding="utf-8")
    session = (ROOT / "web" / "director" / "session.mjs").read_text(encoding="utf-8")
    for marker in (
        "data-action=\"delete-shot\"",
        "data-delete-asset",
        "data-shared-ratio",
        "data-writing=\"simple\"",
        "data-writing=\"advanced\"",
        "data-audio-use=\"record\"",
        "data-action=\"global-add\"",
        "data-action=\"generate-all\"",
        "data-d3-inherit",
        "data-global-d3-toggle",
        "data-generation-field",
        "resolution_mp",
    ):
        assert marker in html, marker
    assert "data-d3-package" in session
    assert 'request("d3/package"' in session
    assert 'request("models")' in session
    assert 'generateAll' in session


def test_director_ui_is_local_and_does_not_embed_remote_runtime_or_media():
    html = (ROOT / "web" / "director" / "index.html").read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html
    assert "data-picker" in html


def test_director_scope_toolbar_and_compact_preview_contract():
    html = (ROOT / "web/director/index.html").read_text(encoding="utf-8")
    session = (ROOT / "web/director/session.mjs").read_text(encoding="utf-8")
    layout = (ROOT / "web/director/workbench.css").read_text(encoding="utf-8")
    assert 'first.closest(\'label\').insertAdjacentHTML(\'beforebegin\'' in html
    assert "el.disabled=false" in html
    assert "const d=effectiveD3(s);" in html
    assert "s.d3=clone(effectiveD3(s))" in html
    assert "doc.d3=clone(d3cfg(s))" in html
    assert 'data-model-zone open' not in html
    assert '<dialog data-model-zone' in html
    assert html.count('<div data-model-settings>') == 1
    assert 'data-action="model-settings" aria-haspopup="dialog"' in session
    assert 'data-local-zone open' in html
    assert 'class="o-media-workbench"' in html
    assert 'height:clamp(180px,25vh,260px);min-height:0;aspect-ratio:auto' in html
    assert 'data-action="toggle-preview"' in html
    assert 'root.dataset.previewCollapsed=String(hidden)' in html
    assert 'position:absolute;inset:0;width:100%;height:100%;max-width:100%;max-height:100%' in html
    assert 'grid-template-columns:var(--w-rail) minmax(350px,var(--w-write-fr,53fr)) minmax(320px,var(--w-preview-fr,47fr))' in layout
    assert 'height:100dvh' in layout
    assert html.index('<div class="o-project">') < html.index('</header>')
    assert '$(".o-project").insertAdjacentHTML("afterend"' not in session
    assert '失败节点：' in session
    assert '任务详情与原始错误' in session
    assert 'esc(failure.exception_message' in session
    assert "shots:[makeShot('镜头 1','text')]" in html
    assert "doc.shots[2].sound='record'" not in html
    assert 'request("compile", { project: p, shot_id: p.current })' in session
    assert 'sequence !== compileRequest' in session
    assert 'snapshot !== JSON.stringify(envelope())' in session
    assert '全部生成前检查 · 尚未提交任务' in session
    assert session.index('const batch = envelope()') < session.index('await request("compile", { project: batch })') < session.index('await request("batches", { batch_id: batchId, project: batch, seed, shot_ids, seed_map })')
    assert 'request("batches/" + encodeURIComponent(batchId) + "/continue"' in session
    assert 'if (row.state !== "not_submitted"' in session


def test_director_d4_frontend_keeps_reconnect_and_large_asset_feedback_contract():
    session = (ROOT / "web" / "director" / "session.mjs").read_text(encoding="utf-8")
    host = (ROOT / "web" / "director.js").read_text(encoding="utf-8")
    assert 'xhr.upload.onprogress' in session
    assert '服务端未确认注册' in session
    assert 'window.addEventListener("offline"' in session
    assert '恢复连接后手动重试，不会自动补交' in session
    assert 'window.addEventListener("online", connectionNotice)' in session
    assert 'xhr.onabort' in session
    assert 'button.dataset.service = "cancel-upload"' in session
    assert 't8director.activeJob:' in session
    assert 'setTimeout(() => watchJob' in session
    assert 'if (result.deleted_from_queue || result.interrupted)' in session
    assert '取消请求未确认；保留任务身份并继续跟踪' in session
    assert 'cancelledJobId' not in session  # A click/failed request cannot turn unknown into cancelled.
    assert 'unknownPolls >= 20' in session
    assert '已连接当前 Core' in host
    assert '页面加载失败，返回画布后可重试' in host


def test_director_uses_a_dedicated_left_sidebar_without_covering_canvas_controls():
    host = (ROOT / "web" / "director.js").read_text(encoding="utf-8")
    assert "registerSidebarTab" in host
    assert 'id: "t8-obsidian-director"' in host
    assert 'title: "导演台"' in host
    assert 'type: "custom"' in host
    assert "renderDirectorSidebar" in host
    assert 'dataset.action = "open-t8-director"' in host
    assert 'id = "t8-director-open"' not in host
    assert "position:fixed;bottom:" not in host


def test_parallel_workbench_moves_existing_controls_without_replacing_services():
    module = (ROOT / "web/director/workbench.mjs").read_text(encoding="utf-8")
    html = (ROOT / "web/director/index.html").read_text(encoding="utf-8")
    assert 'createDirectorWorkbench' in html
    assert 'workspace.prepend(shots, editor)' in module
    for selector in ('[data-global-zone]', '[data-writing-zone]', '[data-simple-zone]',
                     '[data-local-zone]', '[data-events]', '[data-inspector]',
                     '[data-action="generate-all"]', '[data-service="save"]'):
        assert selector in module
    for sync in ('syncInspector', 'syncAssets', 'syncShots', 'syncPreview', 'syncSummary'):
        assert f'workbench.{sync}()' in html
    assert 'fetch(' not in module
    assert 'localStorage' not in module
    assert 'sessionStorage' not in module


def test_parallel_workbench_preserves_global_upload_and_selected_asset_actions():
    module = (ROOT / "web/director/workbench.mjs").read_text(encoding="utf-8")
    html = (ROOT / "web/director/index.html").read_text(encoding="utf-8")
    assert "ctx.importFiles(e.dataTransfer.files,{target:'global',shot:shot().id})" in module
    assert "selectedActions.replaceChildren(selected.querySelector('.o-quick'))" in module
    assert "'.o-dragline,.o-delete-asset'" in module
    assert "selected.dataset.workbenchActions='detached'" in module
    assert "if(card.dataset.workbenchActions==='detached')continue" in html


def test_parallel_workbench_responsive_and_accessible_contract():
    module = (ROOT / "web/director/workbench.mjs").read_text(encoding="utf-8")
    css = (ROOT / "web/director/workbench.css").read_text(encoding="utf-8")
    for marker in ('@media(max-width:1199px)', '@media(max-width:800px)',
                   'object-fit:contain', ':focus-visible',
                   '.o-shot[aria-pressed="true"]', '.o-footer{position:fixed;bottom:0'):
        assert marker in css
    assert "mobileShots.setAttribute('aria-expanded','false')" in module
    assert "root.dataset.shotsOpen='false'" in module
    assert "document.addEventListener('click'" in module
    assert '},true);' in module  # closes menus before legacy root capture handlers
    assert "e.key==='Escape'" in module


def test_parallel_workbench_reports_only_actual_results_and_asset_counts():
    module = (ROOT / "web/director/workbench.mjs").read_text(encoding="utf-8")
    assert 'videos(results().get(s.id)).length>0' in module
    assert "'（'+cards.length+'）'" in module
    assert 'const d=ctx.effectiveD3(s)' in module
    assert 'http://' not in module and 'https://' not in module
