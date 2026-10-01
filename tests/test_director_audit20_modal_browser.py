"""Audit D06/07/08/09 regressions: real Chromium/media, isolated HTTP boundary.

No Core, model or GPU is started. AX assertions are not screen-reader/OS IME
certification. Existing product regression tests remain unchanged.
"""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from playwright.sync_api import expect, sync_playwright

from test_director_film import video


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def audit_page(tmp_path):
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def guess_type(self, path):
            return 'text/javascript' if path.endswith('.mjs') else super().guess_type(path)

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(ROOT/'web/director')))
    Thread(target=server.serve_forever, daemon=True).start()
    media = tmp_path/'two-seconds.mp4'
    video(media, frames=48, audio=False, width=128, height=72)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1440, 'height': 900})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.route('**/models', lambda route: route.fulfill(json={}))
            page.route('**/results/**', lambda route: route.fulfill(json={'results': []}))
            def serve_media(route):
                data = media.read_bytes()
                byte_range = route.request.headers.get('range')
                if byte_range:
                    first, last = byte_range.removeprefix('bytes=').split('-')
                    first, last = int(first or 0), int(last) if last else len(data)-1
                    route.fulfill(status=206, body=data[first:last+1], content_type='video/mp4',
                                  headers={'Accept-Ranges': 'bytes', 'Content-Range': f'bytes {first}-{last}/{len(data)}'})
                else:
                    route.fulfill(body=data, content_type='video/mp4', headers={'Accept-Ranges': 'bytes'})

            page.route('**/audit-media.mp4', serve_media)
            page.goto(f'http://127.0.0.1:{server.server_port}/index.html', wait_until='networkidle')
            expect(page.locator('[data-field="simplePrompt"]')).to_be_visible()
            yield page
            assert not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(('opener', 'dialog', 'close', 'title'), [
    ('[data-storyboard-open]', '[data-storyboard-dialog]', '[data-storyboard-close]', '导入文本分镜'),
    ('[data-bundle-open]', '[data-bundle-dialog]', '[data-bundle-close]', '工程包 ZIP'),
])
def test_menu_modal_name_escape_and_explicit_close_return_visible_focus(audit_page, opener, dialog, close, title):
    page = audit_page
    for escape in (True, False):
        button = page.locator(opener)
        button.evaluate("e=>{for(let p=e.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true}")
        button.evaluate("e=>e.closest('details').querySelector(':scope > summary').dataset.auditReturn=''")
        button.click()
        expect(page.get_by_role('dialog', name=title, exact=True)).to_be_visible()
        assert title in page.locator(dialog).aria_snapshot()
        if escape:
            page.keyboard.press('Escape')
        else:
            page.locator(close).click()
        expect(page.locator(dialog)).not_to_be_visible()
        expect(page.locator('[data-audit-return]')).to_be_focused()


def test_media_dialog_names_focus_and_cancelled_late_prepare(audit_page):
    page = audit_page
    page.evaluate("""async()=>{
        const {createFilmViewer}=await import('./film_ui.mjs');
        const {createVersionComparison}=await import('./compare_ui.mjs');
        const {createFramePicker}=await import('./frame_ui.mjs');
        const root=document.createElement('section');root.id='audit-modals';document.body.append(root);
        root.innerHTML='<details><summary id="audit-modal-menu">Media tools</summary><button id="audit-film-open">Film</button><button id="audit-frame-open">Frame</button><button id="audit-compare-open">Compare</button></details>';
        const deferred=kind=>new Promise(resolve=>window['auditResolve_'+kind]=resolve);
        const film=createFilmViewer({root,prepare:()=>deferred('film'),escape:String,notify:()=>{},locate:()=>{}});
        const frame=createFramePicker(root,{context:()=>1,snapshot:()=>1,current:()=>1,prepare:()=>deferred('frame'),escape:String,notify:()=>{}});
        const compare=createVersionComparison({root,versions:()=>[{id:'a',state:'success'},{id:'b',state:'success'}],resultKey:r=>r.id,videos:()=>[{}],prepare:()=>deferred('compare'),escape:String});
        for(const [kind,viewer] of Object.entries({film,frame,compare}))root.querySelector('#audit-'+kind+'-open').onclick=()=>{viewer.open();queueMicrotask(()=>root.querySelector('details').open=false);};
    }""")
    for kind, title in [('film', '整片串播'), ('frame', '成片取帧 → 下一镜首帧'), ('compare', '本镜 A/B 对比')]:
        page.locator('#audit-modal-menu').click()
        page.locator(f'#audit-{kind}-open').click()
        modal = page.locator(f'#audit-modals [data-{kind}-dialog]')
        expect(page.get_by_role('dialog', name=title, exact=True)).to_be_visible()
        assert title in modal.aria_snapshot()
        page.keyboard.press('Escape')
        expect(modal).not_to_be_visible()
        expect(page.locator('#audit-modal-menu')).to_be_focused()
        page.evaluate('kind=>window["auditResolve_"+kind](null)', kind)
        # Flush the deferred completion; it must neither reopen nor populate media.
        page.evaluate('()=>Promise.resolve()')
        expect(modal).not_to_be_visible()
        assert modal.locator('video[src]').count() == 0
        expect(page.locator('#audit-modal-menu')).to_be_focused()


def test_modal_helper_preserves_owner_focus_other_dialog_and_cancel_rule(audit_page):
    page = audit_page
    page.evaluate("""async()=>{
        const {createModalAccess}=await import('./modal_ui.mjs');
        const root=document.createElement('section');root.id='audit-access';document.body.append(root);
        root.innerHTML='<button id="audit-access-open">Open</button><input id="audit-owner-input"><dialog id="audit-first"><h3 id="audit-title">First</h3><input></dialog><dialog id="audit-second"><h3 id="audit-title">Second</h3><input></dialog>';
        const a=root.querySelector('#audit-first'),b=root.querySelector('#audit-second');
        window.auditAccess={a,b,first:createModalAccess(a,{root}),second:createModalAccess(b,{root})};
        document.querySelector('#audit-access-open').onclick=()=>window.auditAccess.first.show();
    }""")
    page.locator('#audit-access-open').click()
    page.evaluate("()=>{const x=window.auditAccess;x.a.close();x.second.show();x.b.querySelector('input').focus()}")
    expect(page.locator('#audit-second input')).to_be_focused()
    assert page.locator('#audit-first').get_attribute('aria-labelledby') != page.locator('#audit-second').get_attribute('aria-labelledby')
    page.evaluate("()=>window.auditAccess.b.addEventListener('cancel',event=>event.preventDefault(),{once:true})")
    page.keyboard.press('Escape')
    expect(page.locator('#audit-second')).to_be_visible()
    page.evaluate("()=>{window.auditAccess.b.close();document.querySelector('#audit-owner-input').focus()}")
    expect(page.locator('#audit-second')).not_to_be_visible()
    expect(page.locator('#audit-owner-input')).to_be_focused()


def open_test_film(page, start=0, count=1):
    page.evaluate("""async({start,count})=>{
        const {createFilmViewer}=await import('./film_ui.mjs');
        const root=document.createElement('section');root.id='audit-film';document.body.append(root);
        const entries=Array.from({length:count},(_,i)=>({name:'clip '+i,in_frame:start,out_frame:start+12,seconds:.5,media:{fps:24,width:128,height:72},media_url:'/audit-media.mp4'}));
        const film=createFilmViewer({root,prepare:async()=>({ready:true,duration:.5*count,manifest_url:'/unused',entries}),escape:String,notify:()=>{},locate:()=>{}});
        await film.open();
    }""", {'start': start, 'count': count})
    page.wait_for_function("document.querySelector('#audit-film video').readyState>=2")
    return page.locator('#audit-film video')


@pytest.mark.parametrize('start_frame', [0, 6])
def test_last_clip_native_replay_and_seek_stay_in_frozen_range(audit_page, start_frame):
    page = audit_page
    player = open_test_film(page, start_frame)
    state = page.locator('#audit-film [data-film-state]')
    start, end = start_frame/24, (start_frame+12)/24
    page.locator('#audit-film [data-film-restart]').click()
    expect(state).to_contain_text('整片播放结束')
    assert player.evaluate('v=>v.paused')
    assert start <= player.evaluate('v=>v.currentTime') < end
    for _ in range(2):
        player.evaluate('v=>v.play()')
        expect(state).to_contain_text('第 1/1 镜')
        page.wait_for_function('start=>document.querySelector("#audit-film video").currentTime>start+.08', arg=start)
        expect(state).to_contain_text('整片播放结束')
        assert player.evaluate('v=>v.paused')
        assert start <= player.evaluate('v=>v.currentTime') < end
    player.evaluate('v=>{v.currentTime=1.8}')
    page.wait_for_function('end=>document.querySelector("#audit-film video").currentTime<end', arg=end)
    player.evaluate('(v,start)=>{v.currentTime=start+.15}', start)
    page.wait_for_function('start=>Math.abs(document.querySelector("#audit-film video").currentTime-start-.15)<.01', arg=start)
    player.evaluate('v=>v.play()')
    assert player.evaluate('v=>v.currentTime') >= start+.14
    expect(state).to_contain_text('整片播放结束')
    page.locator('#audit-film [data-film-close]').click()
    assert player.evaluate('v=>v.paused')


def test_multi_clip_advance_and_explicit_restart_keep_frozen_order(audit_page):
    page = audit_page
    player = open_test_film(page, 6, 2)
    page.locator('#audit-film [data-film-restart]').click()
    expect(page.locator('#audit-film [data-film-state]')).to_contain_text('第 2/2 镜')
    expect(page.locator('#audit-film [data-film-state]')).to_contain_text('整片播放结束')
    expect(page.locator('#audit-film [data-film-shot="1"]')).to_have_attribute('aria-current', 'true')
    assert .25 <= player.evaluate('v=>v.currentTime') < .75
    page.locator('#audit-film [data-film-restart]').click()
    expect(page.locator('#audit-film [data-film-state]')).to_contain_text('第 1/2 镜')
    page.locator('#audit-film [data-film-close]').click()
    assert player.evaluate('v=>v.paused')


def test_effective_layout_resize_media_and_narrow_footer(audit_page, tmp_path):
    page = audit_page
    writing = page.locator('button[data-focus-layout="writing"]')
    writing.evaluate("e=>{for(let p=e.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true}")
    writing.click()
    root = page.locator('#t8-obsidian-lab')
    expect(root).to_have_attribute('data-focus-layout', 'writing')
    page.set_viewport_size({'width': 640, 'height': 900})
    expect(root).to_have_attribute('data-focus-layout', 'parallel')
    expect(writing).to_have_attribute('aria-pressed', 'false')
    expect(writing).to_be_disabled()
    expect(page.locator('button[data-focus-layout="parallel"]')).to_have_attribute('aria-pressed', 'true')
    expect(page.locator('[data-column-width]')).to_be_disabled()
    page.evaluate("()=>{const v=document.createElement('video');v.id='audit-layout-video';v.src='/audit-media.mp4';v.muted=true;v.loop=true;document.querySelector('.o-stage').append(v)}")
    page.wait_for_function("document.querySelector('#audit-layout-video').readyState>=2")
    player = page.locator('#audit-layout-video')
    player.evaluate('v=>v.play()')
    page.set_viewport_size({'width': 639, 'height': 900})
    page.wait_for_function("innerWidth===639&&document.querySelector('#t8-obsidian-lab').dataset.focusLayout==='parallel'")
    assert not player.evaluate('v=>v.paused')
    page.set_viewport_size({'width': 1440, 'height': 900})
    expect(root).to_have_attribute('data-focus-layout', 'writing')
    page.wait_for_function("document.querySelector('#audit-layout-video').paused")
    expect(writing).to_have_attribute('aria-pressed', 'true')
    # Desktop preference survives fallback; return to parallel for layout controls.
    page.locator('button[data-focus-layout="parallel"]').evaluate('e=>e.click()')
    for width in (1920, 1440, 1024, 800, 640, 390, 320):
        page.set_viewport_size({'width': width, 'height': 900})
        page.wait_for_function('width=>innerWidth===width', arg=width)
        metrics = page.locator('.o-footer').evaluate("""footer=>({height:footer.offsetHeight,controls:[...footer.querySelectorAll('button,summary')].filter(e=>e.getClientRects().length).map(e=>({text:e.textContent,width:e.clientWidth,scroll:e.scrollWidth,left:e.getBoundingClientRect().left,right:e.getBoundingClientRect().right,height:e.clientHeight}))})""")
        assert len(metrics['controls']) >= 4
        for control in metrics['controls']:
            assert control['scroll'] <= control['width']+1, (width, control)
            assert control['width'] >= 44 and control['height'] >= 32, (width, control)
            assert control['left'] >= 0 and control['right'] <= width, (width, control)
        if width <= 480:
            assert metrics['height'] <= 180, metrics
        if width == 390:
            page.screenshot(path=str(tmp_path/'footer-390-fixed.png'))
