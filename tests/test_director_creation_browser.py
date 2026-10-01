"""Real isolated Chromium; mocked Core API, no model inference or user profile."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from playwright.sync_api import sync_playwright, expect


ROOT = Path(__file__).resolve().parents[1]


def test_versions_seed_scope_and_background_completion():
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def guess_type(self, path):
            return 'text/javascript' if path.endswith('.mjs') else super().guess_type(path)

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(ROOT / 'web/director')))
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1366, 'height': 768})
            errors, submitted, records = [], [], []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.route('**/models', lambda route: route.fulfill(json={'unet': [], 'clip': [], 'video_vae': [], 'audio_vae': [], 'lora': []}))
            page.route('**/results/**', lambda route: route.fulfill(json={'results': records}))
            page.route('**/view?**', lambda route: route.fulfill(status=404))

            def queue(route):
                submitted.append(route.request.post_data_json)
                route.fulfill(json={'prompt_id': 'new-result', 'recipe': 'test'})

            def status(route):
                route.fulfill(json={'state': 'success', 'outputs': {'12': {'images': [{'filename': 'new.mp4', 'type': 'output'}]}}})

            page.route('**/generate', queue)
            page.route('**/jobs/new-result', status)
            page.goto(f'http://127.0.0.1:{server.server_port}/index.html', wait_until='networkidle')
            page.wait_for_selector('[data-shot-seed]')
            project = page.evaluate("""() => JSON.parse(localStorage.getItem(Object.keys(localStorage).find(key=>key.startsWith('t8director.draft:'))))""")
            shot = project['doc']['shots'][0]
            for index in (1, 2):
                records.append({'shot_id': shot['id'], 'prompt_id': f'version-{index}', 'state': 'success', 'submitted_at': index,
                                'outputs': {'12': {'images': [{'filename': f'{index}.mp4', 'type': 'output'}]}}})
            page.evaluate("project=>window.postMessage({type:'t8-director:init',project},location.origin)", project)
            page.wait_for_selector('.o-output-player')
            expect(page.locator('[data-result-version]')).to_have_value('version-2')
            page.locator('[data-result-version]').select_option('version-1')
            page.locator('[data-adopt-version]').click()
            expect(page.locator('[data-adopt-version]')).to_have_text('已采用')
            player = page.locator('.o-output-player')
            player.evaluate("node=>node.dataset.testIdentity='original-player'")
            page.locator('[data-shot-seed]').fill('42')
            page.locator('[data-shot-seed]').press('Tab')
            page.locator('[data-field="simplePrompt"]').fill('A quiet room.')
            page.locator('[data-service="new-variation"]').click()
            expect(page.locator('[data-result-version] option')).to_have_count(4)
            assert submitted[0]['seed'] != 42
            assert submitted[0]['project']['doc']['shots'][0]['seed'] == submitted[0]['seed']
            expect(page.locator('[data-result-version]')).to_have_value('version-1')
            expect(player).to_have_attribute('data-test-identity', 'original-player')
            page.reload(wait_until='networkidle')
            expect(page.locator('[data-result-version]')).to_have_value('version-1')
            expect(page.locator('[data-adopt-version]')).to_have_text('已采用')
            page.set_viewport_size({'width': 1100, 'height': 768})
            expect(page.locator('.w-compact-save')).to_be_visible()
            for width, height in ((1366, 768), (1920, 1080), (2560, 1440)):
                page.set_viewport_size({'width': width, 'height': height})
                assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')

            # Actual controls, not a synthetic call to the reordering helper.
            page.locator('[data-action="new"]').click()
            page.locator('[data-action="new"]').click()
            cards = page.locator('.o-shotbar [data-shot]')
            expect(cards).to_have_count(3)
            original = cards.evaluate_all('nodes=>nodes.map(node=>node.dataset.shot)')
            page.locator('[data-shot-seed]').fill('314')
            page.locator('[data-shot-seed]').press('Tab')
            page.locator('.w-shot-actions summary').click()
            page.locator('[data-shot-up]').click()
            expect(cards.nth(1)).to_have_attribute('data-shot', original[2])
            expect(cards.nth(2)).to_have_attribute('data-shot', original[1])
            page.locator(f'[data-shot="{original[2]}"]').drag_to(page.locator(f'[data-shot="{original[0]}"]'))
            expect(cards.nth(0)).to_have_attribute('data-shot', original[2])
            page.locator(f'[data-shot-check="{original[0]}"]').check()
            page.locator(f'[data-shot-check="{original[2]}"]').check()
            batches, checks = [], []

            def preflight(route):
                checks.append(route.request.post_data_json)
                route.fulfill(json={'ready': True, 'errors': [], 'warnings': [], 'shots': []})

            def batch(route):
                batches.append(route.request.post_data_json)
                route.fulfill(json={'batch_id': batches[-1]['batch_id']})

            page.route('**/compile', preflight)
            page.route('**/batch-features', lambda route: route.fulfill(json={'selection_version': 2}))
            page.route('**/batches', batch)
            page.route('**/batches/*', lambda route: route.fulfill(json={'project_id': project['id'], 'complete': True, 'items': []}))
            page.locator('.w-batch-menu summary').click()
            page.locator('[data-service="generate-selected"]').click()
            page.wait_for_function("document.querySelector('[data-notice]').textContent.includes('身份已核对')")
            assert len(batches) == 1
            assert checks[-1]['shot_ids'] == [original[2], original[0]]
            assert batches[0]['shot_ids'] == [original[2], original[0]]
            assert len(batches[0]['project']['doc']['shots']) == 3
            assert batches[0]['seed_map'][original[2]] == 314
            assert batches[0]['project']['doc']['shots'][1]['adoptedResultId'] == 'version-1'
            expect(page.locator('[data-draft-status]')).to_contain_text('待完善')
            page.locator('.w-readiness summary').click()
            page.locator('[data-draft-field="source"]').click()
            expect(page.locator('[data-workbench-section="source"]')).to_be_visible()
            page.locator('[data-service="task-list"]').first.click()
            expect(page.locator('[data-task-drawer]')).to_be_visible()
            page.locator('[data-task-locate]').first.click()
            expect(page.locator('[data-workbench-heading]')).to_contain_text('第 02 镜')
            expect(page.locator('[data-dialog]')).not_to_be_visible()
            expect(page.locator('[data-task-drawer]')).not_to_be_visible()
            separator = page.get_by_role('separator', name='调整剧本与预览列宽')
            separator.focus()
            separator.press('End')
            expect(separator).to_have_attribute('aria-valuenow', '65')
            page.reload(wait_until='networkidle')
            expect(separator).to_have_attribute('aria-valuenow', '65')
            page.locator('.w-header-actions > details > summary').click()
            page.locator('summary').filter(has_text='布局').click()
            page.locator('[data-focus-layout="writing"]').click()
            expect(page.locator('.o-stage')).not_to_be_visible()
            expect(page.locator('.w-editor')).to_be_visible()
            page.locator('.w-header-actions > details > summary').click()
            page.locator('summary').filter(has_text='布局').click()
            page.locator('[data-focus-layout="review"]').click()
            expect(page.locator('.w-editor')).not_to_be_visible()
            expect(page.locator('.o-stage')).to_be_visible()
            page.locator('.w-header-actions > details > summary').click()
            page.locator('summary').filter(has_text='布局').click()
            page.locator('[data-reset-layout]').click()
            expect(separator).to_have_attribute('aria-valuenow', '53')
            for width, height in ((1366, 768), (1920, 1080), (2560, 1440)):
                page.set_viewport_size({'width': width, 'height': height})
                for zoom in (1, 1.25, 1.5):
                    page.evaluate('zoom=>document.documentElement.style.zoom=zoom', zoom)
                    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'), (width, zoom)
            page.evaluate('document.documentElement.style.zoom=1')
            page.locator(f'[data-shot="{original[0]}"]').click()
            page.locator('.w-shot-actions summary').click()
            page.locator('[data-action="duplicate"]').click()
            copied = page.evaluate("""() => {const p=JSON.parse(localStorage.getItem(Object.keys(localStorage).find(key=>key.startsWith('t8director.draft:'))));return p.doc.shots.find(s=>s.id===p.current);} """)
            assert 'adoptedResultId' not in copied
            # Large draft: names/filters must not renumber references or destroy text focus.
            page.evaluate("""() => {
                const key=Object.keys(localStorage).find(key=>key.startsWith('t8director.draft:'));
                const p=JSON.parse(localStorage.getItem(key)),base=p.doc.shots[0];
                p.assets=Array.from({length:30},(_,i)=>({id:crypto.randomUUID(),name:`素材-${i}.${i%3===0?'png':i%3===1?'mp4':'wav'}`,kind:['image','video','audio'][i%3],width:640,height:360,duration:4,sha256:'a'.repeat(64),server_path:`test/${i}`}));
                p.doc.sharedRefs=[p.assets[0].id];
                p.doc.shots=Array.from({length:50},(_,i)=>({...structuredClone(base),id:crypto.randomUUID(),name:`压力测试-${i+1}`,mode:'refs',sound:'native',writingMode:'simple',simplePrompt:'中文长提示词'.repeat(500),tray:p.assets.map(a=>a.id),refs:[p.assets[0].id],selected:p.assets[0].id,adoptedResultId:undefined}));
                p.doc.sampling={mode:'two_pass',output_mp:'0.6',low_loras:Array.from({length:20},(_,i)=>({id:`lora-${i}`,name:`style-${i}.safetensors`,enabled:false,strength:1})),high_loras:[]};
                p.current=p.doc.shots[0].id;
                window.postMessage({type:'t8-director:init',project:p},location.origin);
            }""")
            expect(page.locator('.o-shotbar [data-shot]')).to_have_count(50)
            expect(page.locator('[data-thumbs] [data-tray-id]')).to_have_count(30)
            ids_before = page.locator('[data-thumbs] [data-tray-id]').evaluate_all('nodes=>nodes.map(node=>node.dataset.trayId)')
            page.locator('[data-asset-filters-toggle]').click()
            page.locator('[data-asset-kind]').select_option('audio')
            expect(page.locator('[data-thumbs] [data-tray-id]:visible')).to_have_count(10)
            page.locator('[data-asset-search]').fill('素材-2.')
            expect(page.locator('[data-thumbs] [data-tray-id]:visible')).to_have_count(1)
            page.locator('[data-clear-asset-filter]').click()
            page.locator('[data-asset-scope]').select_option('shared')
            expect(page.locator('[data-thumbs] [data-tray-id]:visible')).to_have_count(1)
            assert page.locator('[data-thumbs] [data-tray-id]').evaluate_all('nodes=>nodes.map(node=>node.dataset.trayId)') == ids_before
            text = page.locator('[data-field="simplePrompt"]')
            expect(text).to_have_value('中文长提示词' * 500)
            text.focus()
            text.press('End')
            text.press('Delete')
            expect(page.locator('.o-shotbar [data-shot]')).to_have_count(50)
            expect(text).to_be_focused()
            page.locator('.w-header-actions [data-action="model-settings"]').click()
            expect(page.locator('[data-model-zone]')).to_be_visible()
            expect(page.locator('.o-sampling-row[data-stage="low_loras"]')).to_have_count(20)
            page.locator('[data-model-zone]').press('Escape')
            expect(page.locator('[data-model-zone]')).not_to_be_visible()
            page.set_viewport_size({'width': 1366, 'height': 768})
            evidence = ROOT / 'artifacts/development/director-subset-9988a2e16a/browser-large-draft.png'
            evidence.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(evidence))
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
