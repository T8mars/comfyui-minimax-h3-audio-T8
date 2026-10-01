import uuid

import av
import numpy as np
import pytest

from test_director_film import setup, video
from h3_audio_t8_pkg.director_film import prepare_film
from h3_audio_t8_pkg.director_film_export import POLICY, reserve_export, run_export, export_file
from h3_audio_t8_pkg.director_project import file_sha


def test_real_mp4_export_preserves_order_frame_ranges_contain_and_originals(tmp_path):
    store, output, project, records = setup(tmp_path)
    video(output/'0.mp4', 70, tone=440)
    video(output/'1.mp4', 150, width=32, height=64, tone=880)
    project['doc']['shots'][0]['filmTrim'] = {'in_frame': 2, 'out_frame': 10}
    before = [file_sha(output/f'{i}.mp4') for i in range(2)]
    film = prepare_film(store, project, records, output)
    job_id = str(uuid.uuid4())
    options = {'policy': POLICY, 'confirm_encoding': True, 'width': 64, 'height': 64}
    job, created = reserve_export(store, project['id'], film['id'], job_id, options)
    assert created and job['state'] == 'queued'
    assert reserve_export(store, project['id'], film['id'], job_id, options)[1] is False
    result = run_export(store, project['id'], film['id'], job_id)
    assert result['state'] == 'success', result.get('error')
    path = export_file(store, project['id'], film['id'], job_id)
    with av.open(str(path)) as container:
        frames = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
    assert len(frames) == 20
    assert frames[0][:12].mean() < 4  # Landscape input letterboxed top/bottom.
    assert frames[0][22:42].mean() > 60
    assert frames[8][:, :12].mean() < 4  # Portrait input letterboxed left/right.
    assert frames[8][:, 22:42].mean() > 135
    assert result['media']['audio_rate'] == 48000 and result['media']['audio_channels'] == 2
    with av.open(str(path)) as container:
        signal = np.concatenate([frame.to_ndarray().mean(axis=0) for frame in container.decode(audio=0)])
    for start, expected in [(0.08, 440), (8/24+0.08, 880)]:
        segment = signal[int(start*48000):int((start+0.1)*48000)]
        spectrum = np.abs(np.fft.rfft(segment))
        frequency = np.fft.rfftfreq(len(segment), 1/48000)[spectrum.argmax()]
        assert abs(frequency-expected) < 15
    assert [file_sha(output/f'{i}.mp4') for i in range(2)] == before
    with pytest.raises(ValueError, match='其他规格'):
        reserve_export(store, project['id'], film['id'], job_id, {**options, 'width': 128})
    path.write_bytes(b'corrupt exported fixture')
    with pytest.raises(ValueError, match='变化'):
        export_file(store, project['id'], film['id'], job_id)


@pytest.mark.parametrize('change', [{'confirm_encoding': False}, {'width': 63}, {'height': True}, {'policy': 'unknown'}])
def test_export_requires_explicit_valid_encoding_policy(tmp_path, change):
    store, output, project, records = setup(tmp_path)
    film = prepare_film(store, project, records, output)
    options = {'policy': POLICY, 'confirm_encoding': True, 'width': 64, 'height': 64, **change}
    with pytest.raises(ValueError):
        reserve_export(store, project['id'], film['id'], str(uuid.uuid4()), options)


def test_silent_clip_requires_explicit_permission_then_real_export_works(tmp_path):
    store, output, project, records = setup(tmp_path)
    video(output/'1.mp4', audio=False)
    film = prepare_film(store, project, records, output)
    options = {'policy': POLICY, 'confirm_encoding': True, 'width': 64, 'height': 64}
    with pytest.raises(ValueError, match='静音'):
        reserve_export(store, project['id'], film['id'], str(uuid.uuid4()), options)
    job_id = str(uuid.uuid4())
    reserve_export(store, project['id'], film['id'], job_id, {**options, 'allow_silent': True})
    result = run_export(store, project['id'], film['id'], job_id)
    assert result['state'] == 'success', result.get('error')
    assert result['media']['frames'] == 24 and result['media']['has_audio']


def test_missing_explicit_ffmpeg_fails_without_replacing_any_media(tmp_path, monkeypatch):
    store, output, project, records = setup(tmp_path)
    before = file_sha(output/'0.mp4')
    film = prepare_film(store, project, records, output)
    job_id = str(uuid.uuid4())
    reserve_export(store, project['id'], film['id'], job_id, {'policy': POLICY, 'confirm_encoding': True, 'width': 64, 'height': 64})
    monkeypatch.setenv('T8_FFMPEG_PATH', str(tmp_path/'missing-ffmpeg.exe'))
    assert run_export(store, project['id'], film['id'], job_id)['state'] == 'error'
    assert file_sha(output/'0.mp4') == before
    assert not (store.root/'film_exports'/f'{job_id}.mp4').exists()
