"""Actual Chromium interaction for the Director's transactional sampling dialog."""

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import os

from playwright.sync_api import sync_playwright, expect
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_split_workflow_export_downloads_without_queueing():
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def guess_type(self, path):
            return "text/javascript" if path.endswith(".mjs") else super().guess_type(path)

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(ROOT / "web" / "director")))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(accept_downloads=True)
            errors, sent = [], []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("request", lambda request: sent.append(request.url))
            page.route("**/models", lambda route: route.fulfill(json={
                "unet": [], "clip": [], "video_vae": [], "audio_vae": [],
                "lora": [], "hyperflow": [], "upscaler": [],
            }))
            page.route("**/d3/editable-split-workflow", lambda route: route.fulfill(json={
                "recipe": "hyperflow8_continuous_separate_stages_exp_v1",
                "workflow": {"version": 0.4, "nodes": [], "links": []},
            }))
            page.goto(f"http://127.0.0.1:{server.server_port}/index.html", wait_until="networkidle")
            page.locator('.w-header-actions > .w-menu > summary').click()
            page.locator('.w-tools-menu > summary').click()
            with page.expect_download() as event:
                page.locator('[data-service="split-workflow"]').click()
            download = event.value
            assert download.suggested_filename == "director-hyperflow-separate-full.workflow.json"
            assert '"version": 0.4' in Path(download.path()).read_text(encoding="utf-8")
            assert not any(url.endswith("/generate") for url in sent)
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_hyperflow_separate_variant_is_explicit_and_survives_reload():
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def guess_type(self, path):
            return "text/javascript" if path.endswith(".mjs") else super().guess_type(path)

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(ROOT / "web" / "director")))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1360, "height": 900})
            page.route("**/models", lambda route: route.fulfill(json={
                "unet": [], "clip": [], "video_vae": [], "audio_vae": [], "lora": [],
                "hyperflow": [{"value": "hyperflow/hf.safetensors", "label": "HyperFlow test"}],
                "upscaler": [],
            }))
            page.goto(f"http://127.0.0.1:{server.server_port}/index.html", wait_until="networkidle")
            page.locator('.w-header-actions [data-action="model-settings"]').click()
            page.locator('[data-sampling-mode="hyperflow"]').click()
            page.locator('[data-sampling-hyperflow-variant]').select_option("continuous4plus4separate")
            page.locator('[data-sampling-hyperflow-file]').select_option("hyperflow/hf.safetensors")
            expect(page.locator('[data-model-settings]')).to_contain_text("分阶段外置效果")
            page.locator('[data-sampling-eav-stage="head"][data-sampling-eav-field="mode"]').select_option("report_only")
            page.locator('[data-sampling-eav-stage="tail"][data-sampling-eav-field="mode"]').select_option("apply_exp")
            page.locator('[data-sampling-eav-stage="tail"][data-sampling-eav-field="tau"]').fill("")
            page.locator('[data-action="apply-model-settings"]').click()
            assert page.locator('[data-model-zone]').is_visible()
            assert page.locator('[data-sampling-eav-stage="tail"][data-sampling-eav-field="tau"]').get_attribute("aria-invalid") == "true"
            page.locator('[data-sampling-eav-stage="tail"][data-sampling-eav-field="tau"]').fill("2.5")
            page.locator('[data-sampling-relay-stage="head"][data-sampling-relay-field="mode"]').select_option("custom")
            page.locator('[data-action="apply-model-settings"]').click()
            assert page.locator('[data-sampling-relay-stage="head"][data-sampling-relay-field="global_prompt"]').get_attribute("aria-invalid") == "true"
            page.locator('[data-sampling-relay-stage="head"][data-sampling-relay-field="global_prompt"]').fill("Head text")
            page.locator('[data-sampling-relay-stage="head"][data-sampling-relay-field="local_prompts"]').fill("Head event A\nHead event B")
            page.locator('[data-sampling-relay-stage="tail"][data-sampling-relay-field="mode"]').select_option("custom")
            page.locator('[data-sampling-relay-stage="tail"][data-sampling-relay-field="global_prompt"]').fill("Tail text")
            page.locator('[data-sampling-relay-stage="tail"][data-sampling-relay-field="local_prompts"]').fill("Tail event A\nTail event B")
            page.locator('[data-action="apply-model-settings"]').click()
            assert "HF 分离 4+4" in page.locator('.w-header-actions [data-action="model-settings"]').inner_text()
            page.reload(wait_until="networkidle")
            page.locator('.w-header-actions [data-action="model-settings"]').click()
            assert page.locator('[data-sampling-hyperflow-variant]').input_value() == "continuous4plus4separate"
            assert page.locator('[data-sampling-eav-stage="head"][data-sampling-eav-field="mode"]').input_value() == "report_only"
            assert page.locator('[data-sampling-eav-stage="tail"][data-sampling-eav-field="mode"]').input_value() == "apply_exp"
            assert page.locator('[data-sampling-eav-stage="tail"][data-sampling-eav-field="tau"]').input_value() == "2.5"
            assert page.locator('[data-sampling-relay-stage="head"][data-sampling-relay-field="global_prompt"]').input_value() == "Head text"
            assert page.locator('[data-sampling-relay-stage="tail"][data-sampling-relay-field="global_prompt"]').input_value() == "Tail text"
            assert page.locator('[data-sampling-relay-stage="head"][data-sampling-relay-field="local_prompts"]').input_value() == "Head event A\nHead event B"
            assert page.locator('[data-sampling-relay-stage="tail"][data-sampling-relay-field="local_prompts"]').input_value() == "Tail event A\nTail event B"
            page.locator('[data-sampling-checkpoint-mode]').select_option("save")
            page.locator('[data-action="apply-model-settings"]').click()
            page.reload(wait_until="networkidle")
            page.locator('.w-header-actions [data-action="model-settings"]').click()
            assert page.locator('[data-sampling-checkpoint-mode]').input_value() == "save"
            page.locator('[data-sampling-checkpoint-mode]').select_option("resume_tail")
            page.locator('[data-action="apply-model-settings"]').click()
            assert page.locator('[data-model-zone]').is_visible()
            assert page.locator('[data-sampling-checkpoint-field="artifact_path"]').get_attribute("aria-invalid") == "true"
            page.locator('[data-sampling-checkpoint-field="artifact_path"]').fill("frozen/head.safetensors")
            page.locator('[data-sampling-checkpoint-field="artifact_sha256"]').fill("a" * 64)
            page.locator('[data-action="apply-model-settings"]').click()
            page.reload(wait_until="networkidle")
            page.locator('.w-header-actions [data-action="model-settings"]').click()
            assert page.locator('[data-sampling-checkpoint-mode]').input_value() == "resume_tail"
            assert page.locator('[data-sampling-eav-stage="head"]').count() == 0
            assert page.locator('[data-sampling-relay-stage="head"]').count() == 0
            assert page.locator('[data-sampling-relay-stage="tail"][data-sampling-relay-field="global_prompt"]').input_value() == "Tail text"
            assert page.locator('[data-sampling-checkpoint-field="artifact_path"]').input_value() == "frozen/head.safetensors"
            assert page.locator('[data-sampling-checkpoint-field="artifact_sha256"]').input_value() == "a" * 64
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_sampling_dialog_cancel_and_apply_do_not_change_legacy_until_committed():
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def guess_type(self, path):
            return "text/javascript" if path.endswith(".mjs") else super().guess_type(path)

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(ROOT / "web" / "director")))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1360, "height": 900})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/models", lambda route: route.fulfill(json={
                "unet": [], "clip": [], "video_vae": [], "audio_vae": [],
                "lora": [{"value": "style.safetensors", "label": "style.safetensors"}, {"value": "人物/汉服.safetensors", "label": "人物/汉服.safetensors"}],
                "upscaler": [{"value": "minimax_h3_latent_upscaler_3d_fp16.safetensors", "label": "3D upscaler"}],
            }))
            page.goto(f"http://127.0.0.1:{server.server_port}/index.html", wait_until="networkidle")
            delivered = page.evaluate("""async () => { const module = await import('./session.mjs'); return module.directorOutputVideos({ outputs: { '12': { images: [{ filename: 'result.mp4' }, { filename: 'still.png' }] } } }); }""")
            assert delivered == [{"filename": "result.mp4"}]
            assert page.locator('.w-header-actions [data-action="model-settings"]').count(), (errors, page.locator("body").inner_text()[:1000])
            page.locator('.w-header-actions [data-action="model-settings"]').click()
            page.locator('[data-sampling-mode="two_pass"]').click()
            assert page.locator(".o-sampling-columns .o-sampling-stage").count() == 2
            page.locator('[data-sampling-action="add"][data-stage="low_loras"]').click()
            search = page.locator('[data-sampling-search="low_loras"]')
            search.focus()
            search.evaluate("el => { window.imeSearch = el; el.dispatchEvent(new CompositionEvent('compositionstart', {bubbles:true})); el.value='汉'; el.dispatchEvent(new InputEvent('input', {bubbles:true,isComposing:true,inputType:'insertCompositionText',data:'汉'})); }")
            assert search.evaluate('el=>el===window.imeSearch && document.activeElement===el')
            assert page.locator('[data-stage="low_loras"] [data-sampling-field="name"] option').count() == 3
            # A synthetic composition event tests DOM continuity, not an OS IME.
            assert page.locator('[data-model-zone]').evaluate("el=>!el.dispatchEvent(new Event('cancel',{cancelable:true}))")
            search.evaluate("el=>{el.value='汉服';el.dispatchEvent(new CompositionEvent('compositionend',{bubbles:true,data:'汉服'}));}")
            assert search.evaluate('el=>el===window.imeSearch && document.activeElement===el')
            assert page.locator('[data-stage="low_loras"] [data-sampling-field="name"] option').count() == 2
            search.fill('style')
            assert search.evaluate('el=>el===window.imeSearch && document.activeElement===el')
            page.locator('[data-sampling-search="low_loras"]').fill("style")
            page.locator('[data-stage="low_loras"] [data-sampling-field="name"]').select_option("style.safetensors")
            page.locator('[data-stage="low_loras"] [data-sampling-field="enabled"]').check()
            page.locator('[data-action="cancel-model-settings"]').click()
            cancelled = page.evaluate("""() => { const key = Object.keys(localStorage).find(x => x.startsWith('t8director.draft:')); return key ? JSON.parse(localStorage.getItem(key)) : null; }""")
            assert cancelled["doc"]["sampling"]["mode"] == "single"
            page.locator('.w-header-actions [data-action="model-settings"]').click()
            assert page.locator('[data-sampling-mode="single"]').get_attribute("aria-pressed") == "true"
            page.locator('[data-sampling-mode="two_pass"]').click()
            page.locator('[data-sampling-action="add"][data-stage="low_loras"]').click()
            page.locator('[data-stage="low_loras"] [data-sampling-field="name"]').select_option("style.safetensors")
            page.locator('[data-stage="low_loras"] [data-sampling-field="enabled"]').check()
            page.locator('[data-sampling-action="copy-low"]').click()
            assert page.locator('[data-stage="high_loras"] [data-sampling-field="name"]').input_value() == "style.safetensors"
            page.locator('[data-action="apply-model-settings"]').click()
            applied = page.evaluate("""() => { const key = Object.keys(localStorage).find(x => x.startsWith('t8director.draft:')); return JSON.parse(localStorage.getItem(key)); }""")
            assert applied["version"] == 2 and applied["doc"]["sampling"]["mode"] == "two_pass"
            page.locator('.w-header-actions [data-action="model-settings"]').click()
            assert page.locator('[data-sampling-mode="two_pass"]').get_attribute("aria-pressed") == "true"
            assert page.locator('[data-stage="low_loras"] [data-sampling-field="name"]').input_value() == "style.safetensors"
            page.locator('[data-sampling-scope="local"]').click()
            page.locator('[data-sampling-mode="single"]').click()
            page.locator('[data-action="apply-model-settings"]').click()
            independent = page.evaluate("""() => { const key = Object.keys(localStorage).find(x => x.startsWith('t8director.draft:')); return JSON.parse(localStorage.getItem(key)); }""")
            assert independent["doc"]["sampling"]["mode"] == "two_pass"
            assert independent["doc"]["shots"][0]["samplingInherit"] is False
            assert independent["doc"]["shots"][0]["sampling"]["mode"] == "single"
            page.locator('.w-header-actions [data-action="model-settings"]').click()
            assert page.locator('[data-sampling-scope="local"]').get_attribute("aria-pressed") == "true"
            assert page.locator('[data-sampling-mode="single"]').get_attribute("aria-pressed") == "true"
            page.locator('[data-sampling-scope="global"]').click()
            assert page.locator('[data-sampling-mode="two_pass"]').get_attribute("aria-pressed") == "true"
            page.locator('[data-action="cancel-model-settings"]').click()
            project = page.evaluate("""() => { const key = Object.keys(localStorage).find(x => x.startsWith('t8director.draft:')); return JSON.parse(localStorage.getItem(key)); }""")
            shot_id = project["current"]
            page.route("**/results/*", lambda route: route.fulfill(json={"project_id": project["id"], "results": [{
                "shot_id": shot_id, "prompt_id": "e42bea1f-f961-4ef7-9c48-b181490b6f17", "state": "success",
                "recipe": "director_two_pass", "outputs": {"12": {"images": [{
                    "filename": "film.mp4", "subfolder": "T8_Director\\abc", "type": "output",
                }]}},
            }]}))
            page.evaluate("project => window.postMessage({ type: 't8-director:init', project }, location.origin)", project)
            page.wait_for_selector("video.o-output-player")
            assert page.locator('[data-view="output"]').get_attribute("aria-pressed") == "true"
            assert "film.mp4" in page.locator("video.o-output-player").get_attribute("src")
            assert "有成片" in page.locator('[data-shot]').first.inner_text()
            page.locator('[data-view="input"]').click()
            page.locator('[data-shot]').first.click()
            assert page.locator('[data-view="output"]').get_attribute("aria-pressed") == "true"
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()


def test_live_director_sampling_route_when_requested():
    url = os.getenv("T8_DIRECTOR_LIVE_URL")
    if not url:
        pytest.skip("Set T8_DIRECTOR_LIVE_URL for the running Core UI smoke")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1360, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url, wait_until="networkidle")
        page.locator('.w-header-actions [data-action="model-settings"]').click()
        zone = page.locator('[data-model-zone]')
        assert zone.get_attribute("open") is not None
        shared = zone.locator('[data-shared-models]')
        assert shared.get_attribute('open') is None
        shared.locator('summary').click()
        assert shared.get_attribute('open') is not None
        page.locator('[data-sampling-mode="two_pass"]').click()
        assert page.locator(".o-sampling-columns .o-sampling-stage").count() == 2
        page.locator('[data-action="cancel-model-settings"]').click()
        assert not errors, errors
        browser.close()


def test_live_director_recovers_completed_video_when_requested():
    url = os.getenv("T8_DIRECTOR_LIVE_URL")
    project_id = os.getenv('T8_DIRECTOR_LIVE_PROJECT_ID')
    if not url or not project_id:
        pytest.skip("Set T8_DIRECTOR_LIVE_URL and T8_DIRECTOR_LIVE_PROJECT_ID for an isolated existing result fixture")
    import json
    from urllib.request import urlopen
    from urllib.parse import urlsplit, urlunsplit, urlencode, parse_qs

    parsed = urlsplit(url)
    base = urlunsplit((parsed.scheme, parsed.netloc, parsed.path.removesuffix('/ui'), '', ''))
    with urlopen(base+'/projects/'+project_id, timeout=20) as response:
        project = json.load(response)
    with urlopen(base+'/results/'+project_id, timeout=20) as response:
        records = json.load(response)['results']
    shot = next(item for item in project['doc']['shots'] if item['id'] == project['current'])
    adopted = next(item for item in records if item['shot_id'] == shot['id'] and item['prompt_id'] == shot['adoptedResultId'])
    assert adopted['state'] == 'success'
    expected_files = [media['filename'] for output in adopted['outputs'].values()
                      for field in ('images', 'gifs', 'videos') for media in output.get(field, [])
                      if media.get('filename', '').lower().endswith(('.mp4', '.webm', '.mov', '.mkv'))]
    assert len(expected_files) == 1
    target = base+'/ui?'+urlencode({'project_id': project_id})
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1360, "height": 900})
        errors, result_reads, writes = [], [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on('request', lambda request: result_reads.append(request.url) if '/results/' in request.url else None)
        page.on('request', lambda request: writes.append(request.url) if request.method != 'GET' else None)
        page.goto(target, wait_until="networkidle")
        page.locator('[data-view="output"]').click()
        page.wait_for_selector("video.o-output-player", timeout=20000)
        player = page.locator('video.o-output-player')
        frozen_src = player.get_attribute('src')
        assert parse_qs(urlsplit(frozen_src).query)['filename'] == expected_files
        expect(page.locator(f'[data-shot="{shot["id"]}"]')).to_contain_text('有成片')
        page.wait_for_function("document.querySelector('video.o-output-player')?.readyState >= 2")
        assert player.evaluate('v=>v.videoWidth>0 && v.videoHeight>0 && v.duration>0')
        page.reload(wait_until='networkidle')
        page.locator('[data-view="output"]').click()
        expect(page.locator('video.o-output-player')).to_have_attribute('src', frozen_src)
        # Opt-in real wall-clock soak, not a mocked timer or a new generation.
        soak_seconds = int(os.getenv('T8_DIRECTOR_LIVE_SOAK_SECONDS', '0'))
        assert 0 <= soak_seconds <= 600
        if soak_seconds:
            import time
            player = page.locator('video.o-output-player')
            page.wait_for_function("document.querySelector('video.o-output-player')?.readyState >= 2")
            page.locator('[data-service="task-list"]').first.click()
            expect(page.locator('[data-task-state]')).to_contain_text('已更新')
            player.evaluate("""async v=>{
                window.directorSoakPlayer=v; window.directorSoakTicks=0;
                v.addEventListener('timeupdate',()=>window.directorSoakTicks++);
                v.muted=true;v.loop=true;await v.play();
            }""")
            prompt = page.locator('[data-field="simplePrompt"]')
            text = '中文长提示词' * 500
            prompt.fill(text)
            prompt.evaluate('el=>window.directorSoakEditor=el')
            initial_reads = len(result_reads)
            deadline = time.monotonic() + soak_seconds
            iteration = 0
            while time.monotonic() < deadline:
                page.wait_for_timeout(5100)
                expect(prompt).to_be_focused()
                assert prompt.evaluate('el=>el===window.directorSoakEditor')
                assert player.evaluate('v=>v===window.directorSoakPlayer && !v.paused && !v.error')
                prompt.press('End')
                prompt.press_sequentially('续')
                text += '续'
                iteration += 1
                expect(prompt).to_have_value(text)
            assert len(result_reads) >= initial_reads + max(1, iteration - 1)
            assert page.evaluate('window.directorSoakTicks') > iteration * 2
            player.evaluate('v=>v.pause()')
            page.locator('[data-task-close]').click()
            # Text stayed local: this test must never save or alter the server project.
            with urlopen(base+'/projects/'+project_id, timeout=20) as response:
                assert json.load(response) == project
        assert not writes, writes
        assert not errors, errors
        browser.close()
