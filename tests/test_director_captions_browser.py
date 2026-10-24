"""Owned headless Chromium, actual tiny media and caption store; no user tabs."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
from urllib.parse import urlsplit
import uuid

from playwright.sync_api import sync_playwright, expect
import pytest

from test_director_film import setup
from h3_audio_t8_pkg.director_captions import (
    INPUT_SCHEMA, _compile, caption_file, load_captions, prepare_captions, track_from_speech_report,
)
from h3_audio_t8_pkg.director_film import film_media, load_film, prepare_film
from h3_audio_t8_pkg.director_project import file_sha

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def caption_page(tmp_path):
    store, output, project, records = setup(tmp_path)

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def guess_type(self, path):
            return 'text/javascript' if path.endswith('.mjs') else super().guess_type(path)

        def do_GET(self):
            # Chromium's download request is not a normal renderer fetch. Deliver
            # it over actual HTTP, rather than completing it with page.route.
            parts = urlsplit(self.path).path.split('/')
            if 'films' not in parts or 'captions' not in parts or 'file' not in parts:
                return super().do_GET()
            parts = parts[parts.index('films')+1:]
            try:
                owner, fid, _, cid, _, format = parts
                data = caption_file(store, owner, fid, cid, format).read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', ('text/vtt' if format == 'vtt' else 'application/x-subrip')+'; charset=utf-8')
                self.send_header('Content-Disposition', f'attachment; filename="director-captions.{format}"')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (ValueError, FileNotFoundError):
                self.send_error(400)

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(ROOT/'web/director')))
    Thread(target=server.serve_forever, daemon=True).start()
    state = {'films': [], 'saves': [], 'submissions': [], 'errors': [], 'lose_reply_once': False,
             'hold_save_reply': False, 'pending_save': None}
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={'width': 1366, 'height': 768}, accept_downloads=True)
            page.on('pageerror', lambda error: state['errors'].append(str(error)))
            page.route('**/models', lambda route: route.fulfill(json={}))
            page.route('**/results/**', lambda route: route.fulfill(json={'results': records}))
            page.route('**/generate', lambda route: (state['submissions'].append(route.request.url), route.fulfill(json={})))

            def prepare(route):
                film = prepare_film(store, route.request.post_data_json['project'], records, output)
                state['films'].append(film)
                route.fulfill(json=film)

            def media(route):
                parts = urlsplit(route.request.url).path.split('/')
                path = film_media(store, parts[-4], parts[-3], int(parts[-1]))
                route.fulfill(body=path.read_bytes(), content_type='video/mp4')

            def captions(route):
                parts = urlsplit(route.request.url).path.split('/')
                parts = parts[parts.index('films')+1:]
                owner, fid, cid = parts[0], parts[1], parts[3]
                try:
                    if cid == 'speech-track':
                        value = route.request.post_data_json
                        film = load_film(store, owner, fid)
                        entry = next(entry for entry in film['entries'] if entry['shot_id'] == value['shot_id'])
                        result = track_from_speech_report(entry, value['report'], confirmed_media_match=value.get('confirm_media_match'))
                        _compile(film, {'schema': INPUT_SCHEMA, 'film_sha256': film['sha256'], 'tracks': [result]})
                        route.fulfill(json=result)
                    elif len(parts) > 4:
                        route.continue_()
                    elif route.request.method == 'POST':
                        value = route.request.post_data_json
                        state['saves'].append({'id': cid, 'input': value})
                        result, created = prepare_captions(store, owner, fid, cid, value)
                        if state['hold_save_reply']:
                            state['hold_save_reply'] = False
                            state['pending_save'] = (route, result, created)
                        elif state['lose_reply_once']:
                            state['lose_reply_once'] = False
                            route.abort('connectionreset')
                        else:
                            route.fulfill(status=201 if created else 200, json=result)
                    else:
                        route.fulfill(json=load_captions(store, owner, fid, cid))
                except (ValueError, KeyError, StopIteration) as error:
                    route.fulfill(status=400, json={'error': str(error)})
                except FileNotFoundError:
                    route.fulfill(status=404, json={'error': 'missing caption fixture'})

            page.route('**/films/prepare', prepare)
            page.route('**/films/*/*/media/*', media)
            page.route('**/films/*/*/captions/**', captions)
            page.goto(f'http://127.0.0.1:{server.server_port}/index.html', wait_until='networkidle')
            page.evaluate("project=>window.postMessage({type:'t8-director:init',project},location.origin)", project)
            expect(page.locator('.o-shotbar [data-shot]')).to_have_count(2)
            page.locator('[data-film-open]').click()
            expect(page.locator('.w-film-player')).to_be_visible()
            page.wait_for_function("document.querySelector('.w-film-player').readyState>=1")
            yield page, state, store, project
            assert not state['submissions'] and not state['errors']
            browser.close()
    finally:
        server.shutdown()
        server.server_close()


def add_text(page, text, start='0', end='12'):
    page.locator('[data-caption-start]').fill(start)
    page.locator('[data-caption-end]').fill(end)
    page.locator('[data-caption-text]').fill(text)
    page.locator('[data-caption-add]').click()


def test_caption_full_ui_literal_download_and_no_prompt_adoption(caption_page, tmp_path):
    page, state, store, project = caption_page
    details = page.locator('.w-film-captions')
    assert details.get_attribute('open') is None
    details.locator('summary').first.click()
    assert page.locator('[data-caption-text]').input_value() == ''
    film = state['films'][0]
    original = {path: file_sha(path) for path in [store.root/'films'/f"{film['id']}.json",
                film_media(store, project['id'], film['id'], 0), film_media(store, project['id'], film['id'], 1)]}
    add_text(page, '你好 <script>window.badCaption=true</script>\n第二行', end='6')
    page.locator('[data-caption-shot]').select_option(film['entries'][1]['shot_id'])
    add_text(page, '下一镜对白', start='2')
    expect(page.locator('[data-caption-rows] li')).to_have_count(2)
    page.once('dialog', lambda dialog: dialog.dismiss())
    page.locator('[data-film-close]').click()
    expect(page.locator('[data-film-dialog]')).to_be_visible()
    assert len(state['saves']) == 0
    page.locator('[data-caption-save]').click()
    expect(page.locator('[data-caption-status]')).to_contain_text('字幕已保存')
    with page.expect_download() as downloaded:
        page.locator('[data-caption-status] a').filter(has_text='下载 SRT').click()
    data = Path(downloaded.value.path()).read_text(encoding='utf-8')
    assert '00:00:00,000 --> 00:00:00,250' in data
    assert '00:00:00,583 --> 00:00:01,000' in data
    assert '&lt;script&gt;window.badCaption=true&lt;/script&gt;' in data
    assert page.evaluate('window.badCaption') is None
    assert {path: file_sha(path) for path in original} == original
    page.locator('[data-film-last-caption]').click()
    expect(page.locator('[data-film-caption-status]')).to_contain_text('字幕已保存')
    evidence = tmp_path/'caption-browser-evidence'
    evidence.mkdir()
    page.screenshot(path=str(evidence/'b-caption-playlist-v2.png'))
    page.locator('[data-caption-status]').scroll_into_view_if_needed()
    page.screenshot(path=str(evidence/'b-caption-editor-v2.png'))
    page.locator('[data-film-close]').click()
    expect(page.locator('[data-film-dialog]')).not_to_be_visible()
    assert len(state['saves']) == 1


@pytest.mark.parametrize('storage_available', [True, False])
def test_ambiguous_caption_response_retry_reuses_id_and_new_draft_gets_new_id(caption_page, storage_available):
    page, state, store, _ = caption_page
    if not storage_available:
        page.evaluate("""() => {
            const get=Storage.prototype.getItem,set=Storage.prototype.setItem;
            Storage.prototype.getItem=function(key){if(key.startsWith('t8director.filmCaption:'))throw new DOMException('caption storage denied','SecurityError');return get.call(this,key);};
            Storage.prototype.setItem=function(key,value){if(key.startsWith('t8director.filmCaption:'))throw new DOMException('caption storage denied','SecurityError');return set.call(this,key,value);};
        }""")
    page.locator('.w-film-captions > summary').click()
    add_text(page, '网络中断仍只保存一次')
    state['lose_reply_once'] = True
    page.locator('[data-caption-save]').click()
    expect(page.locator('[data-caption-status]')).to_contain_text('字幕未确认保存')
    page.locator('[data-caption-save]').click()
    expect(page.locator('[data-caption-status]')).to_contain_text('字幕已保存')
    assert len(state['saves']) == 2
    assert state['saves'][0] == state['saves'][1]
    assert len(list((store.root/'film_captions').iterdir())) == 1
    page.locator('[data-film-last-caption]').click()
    expect(page.locator('[data-film-caption-status]')).to_contain_text('字幕已保存')
    add_text(page, '新增第二条', start='6')
    page.locator('[data-caption-save]').click()
    expect(page.locator('[data-caption-status]')).to_contain_text('字幕已保存 · 2 条')
    assert state['saves'][-1]['id'] != state['saves'][0]['id']
    assert len(list((store.root/'film_captions').iterdir())) == 2


def test_import_validation_and_speech_explicit_binding_preserve_original(caption_page):
    page, state, _, _ = caption_page
    page.locator('.w-film-captions > summary').click()
    add_text(page, '保留原稿')
    page.locator('.w-film-captions details > summary').click()
    page.once('dialog', lambda dialog: dialog.accept())
    page.locator('[data-caption-import]').set_input_files({'name': 'wrong-film.json', 'mimeType': 'application/json',
        'buffer': json.dumps({'schema': INPUT_SCHEMA, 'film_sha256': '0'*64, 'tracks': []}).encode()})
    expect(page.locator('[data-caption-status]')).to_contain_text('导入失败，原稿保留')
    expect(page.locator('[data-caption-rows]')).to_contain_text('保留原稿')
    report = {'schema': 'minimax_h3_t8_speech_timeline_v1', 'sample_rate': 48000,
              'timeline': [{'speaker_id': 'A', 'text': 'Speech 计划稿', 'start_sample': 0, 'end_sample': 24000}]}
    file = {'name': 'speech.json', 'mimeType': 'application/json', 'buffer': json.dumps(report).encode()}
    page.locator('[data-caption-speech]').set_input_files(file)
    expect(page.locator('[data-caption-status]')).to_contain_text('请先确认')
    expect(page.locator('[data-caption-rows]')).to_contain_text('保留原稿')
    page.locator('[data-caption-confirm]').check()
    page.once('dialog', lambda dialog: dialog.accept())
    page.locator('[data-caption-speech]').set_input_files(file)
    expect(page.locator('[data-caption-rows]')).to_contain_text('A: Speech 计划稿')
    page.locator('[data-caption-save]').click()
    expect(page.locator('[data-caption-status]')).to_contain_text('字幕已保存')
    assert state['saves'][0]['input']['tracks'][0]['clock'] == {'unit': 'samples', 'rate': 48000}
    page.locator('[data-caption-text]').fill('未添加文字')
    page.locator('[data-caption-save]').click()
    expect(page.locator('[data-caption-status]')).to_contain_text('还有未添加的字幕')
    assert len(state['saves']) == 1
    page.locator('[data-caption-draft]').click()
    expect(page.locator('[data-caption-status]')).to_contain_text('再下载完整字幕稿')
    expect(page.locator('[data-caption-text]')).to_have_value('未添加文字')


def test_escape_keeps_unsaved_draft_and_removal_allows_empty_captions(caption_page):
    page, state, _, _ = caption_page
    page.locator('.w-film-captions > summary').click()
    add_text(page, '待删除')
    page.once('dialog', lambda dialog: dialog.dismiss())
    page.keyboard.press('Escape')
    expect(page.locator('[data-film-dialog]')).to_be_visible()
    expect(page.locator('[data-caption-rows]')).to_contain_text('待删除')
    page.locator('[data-caption-remove]').click()
    expect(page.locator('[data-caption-rows] li')).to_have_count(0)
    page.locator('[data-caption-save]').click()
    expect(page.locator('[data-caption-status]')).to_contain_text('字幕已保存 · 0 条')
    assert state['saves'][0]['input']['tracks'] == []
    page.locator('[data-film-close]').click()
    expect(page.locator('[data-film-dialog]')).not_to_be_visible()


def test_caption_draft_json_roundtrip_preserves_added_multiline_text(caption_page):
    page, state, _, _ = caption_page
    page.locator('.w-film-captions > summary').click()
    add_text(page, '字幕稿第一行\n第二行 <b>按原文字保存</b>')
    with page.expect_download() as downloaded:
        page.locator('[data-caption-draft]').click()
    raw = Path(downloaded.value.path()).read_bytes()
    payload = json.loads(raw)
    assert payload['tracks'][0]['cues'][0]['text'] == '字幕稿第一行\n第二行 <b>按原文字保存</b>'
    page.once('dialog', lambda dialog: dialog.accept())
    page.locator('[data-caption-clear]').click()
    expect(page.locator('[data-caption-rows] li')).to_have_count(0)
    page.locator('.w-film-captions details > summary').click()
    page.locator('[data-caption-import]').set_input_files({'name': 'saved-draft.json', 'mimeType': 'application/json', 'buffer': b'\xef\xbb\xbf'+raw})
    expect(page.locator('[data-caption-rows] li')).to_have_count(1)
    page.locator('[data-caption-save]').click()
    expect(page.locator('[data-caption-status]')).to_contain_text('字幕已保存')
    assert state['saves'][0]['input'] == payload


def test_late_old_project_caption_reply_does_not_fill_new_project_window(caption_page):
    page, state, store, project = caption_page
    page.locator('.w-film-captions > summary').click()
    add_text(page, '旧项目的字幕')
    state['hold_save_reply'] = True
    page.locator('[data-caption-save]').click()
    expect(page.locator('[data-caption-save]')).to_be_disabled()
    assert state['pending_save'] is not None
    page.once('dialog', lambda dialog: dialog.accept())
    page.locator('[data-film-close]').click()
    expect(page.locator('[data-film-dialog]')).not_to_be_visible()
    project['id'] = str(uuid.uuid4())
    page.evaluate("project=>window.postMessage({type:'t8-director:init',project},location.origin)", project)
    page.wait_for_function('id=>new URL(location.href).searchParams.get("project_id")===id', arg=project['id'])
    page.locator('[data-film-open]').click()
    expect(page.locator('.w-film-player')).to_be_visible()
    route, result, created = state['pending_save']
    route.fulfill(status=201 if created else 200, json=result)
    page.locator('.w-film-captions > summary').click()
    expect(page.locator('[data-caption-rows] li')).to_have_count(0)
    expect(page.locator('[data-caption-status]')).to_have_text('')
    page.locator('[data-film-last-caption]').click()
    expect(page.locator('[data-film-caption-status]')).to_contain_text('当前项目没有可读取的字幕附件')
    assert result['project_id'] != project['id']
    assert len(list((store.root/'film_captions').iterdir())) == 1
