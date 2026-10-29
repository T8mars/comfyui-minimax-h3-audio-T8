"""Owned headless UI; real source/ASR receipts, injected transcription only."""
from urllib.parse import urlsplit

from playwright.sync_api import expect
import pytest

from test_director_captions_browser import caption_page as caption_page
from h3_audio_t8_pkg import director_caption_asr as asr
from h3_audio_t8_pkg import speech_verification as speech


@pytest.fixture
def recognition(request, tmp_path, monkeypatch):
    page, state, store, project = request.getfixturevalue('caption_page')
    model = tmp_path / 'chosen-model'
    model.mkdir()
    (model/'config.json').write_text('{}')
    (model/'model.bin').write_bytes(b'never loaded by UI fixture')
    monkeypatch.setattr(speech, 'resolve_asr_model_directory', lambda _: model)
    calls = {'decode': 0, 'cancel': 0, 'hold': False, 'jobs': []}
    def transcribe(audio, *_args, **kwargs):
        calls['decode'] += 1
        kwargs['cancel_check']()
        return {'text': 'hello world.', 'language': 'en',
                'words': [{'word':'hello', 'start':.1, 'end':.2},
                          {'word':'world.', 'start':.25, 'end':.4}],
                'untimed_words':[{'word':'missing','start':None,'end':None}]}
    monkeypatch.setattr(speech, 'transcribe_audio', transcribe)
    def api(route):
        parts = urlsplit(route.request.url).path.split('/')
        parts = parts[parts.index('films')+1:]
        owner, fid, job_id = parts[0], parts[1], parts[4]
        args = (store, owner, fid, job_id)
        if parts[-1] == 'cancel':
            calls['cancel'] += 1
            asr.cancel_asr(*args)
            route.fulfill(json=asr.run_asr(*args))
        elif route.request.method == 'POST':
            job, _ = asr.reserve_asr(*args, route.request.post_data_json)
            calls['jobs'].append(args)
            if not calls['hold']:
                asr.run_asr(*args)
            route.fulfill(status=202, json=job)
        else:
            job = asr.asr_status(*args)
            if job['state'] == 'success':
                job = {**job, **asr.recognized_track(*args)}
            route.fulfill(json=job)
    page.route('**/films/*/*/captions/asr/**', api)
    page.locator('.w-film-captions > summary').click()
    page.locator('[data-caption-asr] > summary').click()
    page.locator('[data-caption-asr-model]').fill(str(model))
    return page, state, store, calls


def test_actual_source_draft_can_be_edited_saved_and_not_repointed_on_reopen(recognition):
    page, state, store, calls = recognition
    page.locator('[data-caption-asr-start]').click()
    expect(page.locator('[data-caption-asr-status]')).to_contain_text('识别草稿已加入')
    expect(page.locator('[data-caption-rows] li')).to_have_count(1)
    expect(page.locator('[data-caption-rows]')).to_contain_text('hello world.')
    original = asr._path(store, calls['jobs'][0][-1]).read_bytes()
    page.locator('[data-caption-edit]').click()
    page.locator('[data-caption-edit-text]').fill('Edited <literal> transcript')
    page.locator('[data-caption-edit-start]').fill('0.125')
    page.locator('[data-caption-edit-apply]').click()
    page.locator('[data-caption-save]').click()
    expect(page.locator('[data-caption-status]')).to_contain_text('字幕已保存')
    saved = state['saves'][-1]['input']['tracks'][0]
    assert saved['source']['kind'] == 'recognized_audio'
    assert saved['cues'][0]['text'] == 'Edited <literal> transcript'
    assert saved['cues'][0]['start'] == 2000
    assert asr._path(store, calls['jobs'][0][-1]).read_bytes() == original
    page.locator('[data-film-close]').click()
    page.locator('[data-film-open]').click()
    page.locator('.w-film-captions > summary').click()
    page.locator('[data-caption-asr] > summary').click()
    page.locator('[data-caption-asr-last]').click()
    expect(page.locator('[data-caption-asr-status]')).to_contain_text('另一份冻结清单')
    expect(page.locator('[data-caption-rows] li')).to_have_count(0)
    assert calls['decode'] == 1


def test_cancel_keeps_existing_caption_draft_and_does_not_submit_h3(recognition):
    page, _, _, calls = recognition
    calls['hold'] = True
    page.locator('[data-caption-asr-start]').click()
    expect(page.locator('[data-caption-asr-cancel]')).to_be_enabled()
    page.locator('[data-caption-asr-cancel]').click()
    expect(page.locator('[data-caption-asr-status]')).to_contain_text('识别未完成')
    expect(page.locator('[data-caption-rows] li')).to_have_count(0)
    assert calls['decode'] == 0 and calls['cancel'] == 1
