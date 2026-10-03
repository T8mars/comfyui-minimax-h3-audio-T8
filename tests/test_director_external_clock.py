"""Actual isolated AV/PTS/CFR inspection and contradictory receipt rejection."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
from pathlib import Path

import av
from PIL import Image
import pytest

from h3_audio_t8_pkg.director_media import run_media, _result, MediaProcessError
from test_director_media_worker_ops import video

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('external_clock_worker', ROOT / 'h3_t8/director_media_worker.py')
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def test_actual_AV_and_silent_clock_are_measured_not_header_only(tmp_path):
    for tracks in (0, 1):
        path = tmp_path / f'{tracks}-tracks.mp4'
        video(path, audio_tracks=tracks)
        result = run_media('external', path.resolve())
        assert result['media']['frames'] == 12
        assert result['decoded_clock']['video']['fps'] == {'num': 24, 'den': 1}
        assert result['decoded_clock']['video']['first_time'] == {'num': 0, 'den': 1}
        assert result['decoded_clock']['fully_decoded'] is True
        assert result['source_timing']['fully_decoded'] is False
        assert result['decoded_clock']['normalization_performed'] is False
        assert (result['decoded_clock']['audio'] is None) == (tracks == 0)


def test_actual_multiple_audio_tracks_are_not_guessed(tmp_path):
    path = tmp_path / 'two-audio.mp4'
    video(path, audio_tracks=2)
    with pytest.raises(MediaProcessError, match='最多一音轨'):
        run_media('external', path.resolve())


def test_actual_nonzero_start_is_measured_and_export_rejected_without_conversion(tmp_path):
    import uuid
    from h3_audio_t8_pkg.director_project import ProjectStore, new_project, sha, file_sha
    from h3_audio_t8_pkg.director_external import register_external_take
    from h3_audio_t8_pkg.director_film import prepare_film, result_key
    from h3_audio_t8_pkg.director_film_export import reserve_export, POLICY
    store = ProjectStore(tmp_path / 'user', tmp_path / 'input')
    asset_id = str(uuid.uuid4())
    path = store.input_root / 't8_director' / asset_id / 'source.mp4'
    path.parent.mkdir(parents=True)
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=24)
        stream.width, stream.height, stream.pix_fmt = 64, 32, 'yuv420p'
        stream.options = {'threads': '1', 'bf': '0'}
        for index in range(12):
            frame = av.VideoFrame.from_image(Image.new('RGB', (64, 32), (50+index, 80, 110)))
            frame.pts, frame.time_base = index + 48, Fraction(1, 24)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    before = file_sha(path)
    asset = store.register_asset(path, asset_id, 'source.mp4')
    project = new_project()
    project['assets'] = [asset]
    project = store.save(project, 0)
    output = tmp_path / 'output'
    output.mkdir()
    record = register_external_take(store, {'project_id': project['id'], 'shot_id': project['current'],
        'asset_id': asset_id, 'take_id': str(uuid.uuid4()), 'expected_revision': 1,
        'project_sha256': sha(project), 'label': '实测偏移两秒', 'provenance_category': 'external'}, output)['record']
    assert record['decoded_clock']['video']['first_time'] == {'num': 2, 'den': 1}
    assert record['decoded_clock']['zero_origin_film_compatible'] is False
    assert record['decoded_clock']['normalization_performed'] is False
    project['doc']['shots'][0]['adoptedResultId'] = result_key(record)
    film = prepare_film(store, project, [record], output)
    assert film['ready'] and film['entries'][0]['decoded_clock'] == record['decoded_clock']
    with pytest.raises(ValueError, match='非零'):
        reserve_export(store, project['id'], film['id'], str(uuid.uuid4()),
            {'policy': POLICY, 'confirm_encoding': True, 'allow_silent': True, 'width': 64, 'height': 32})
    assert file_sha(path) == before


@pytest.mark.parametrize('vfr', [False, True])
def test_actual_fractional_rate_and_VFR_no_hidden_conversion(tmp_path, vfr):
    path = tmp_path / f'fractional-{vfr}.mp4'
    rate = Fraction(30000, 1001)
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=rate)
        stream.width, stream.height, stream.pix_fmt = 64, 32, 'yuv420p'
        stream.options = {'threads': '1', 'bf': '0'}
        for index in range(12):
            frame = av.VideoFrame.from_image(Image.new('RGB', (64, 32), (50+index, 80, 110)))
            frame.pts = index + int(vfr and index >= 5)
            frame.time_base = 1 / rate
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    if vfr:
        with pytest.raises(MediaProcessError, match='PTS|变帧率'):
            run_media('external', path.resolve())
    else:
        result = run_media('external', path.resolve())
        assert result['decoded_clock']['video']['fps'] == {'num': 30000, 'den': 1001}
        assert result['decoded_clock']['video']['last_time'] == {'num': 11011, 'den': 30000}


@pytest.mark.parametrize('damage', ['count', 'tolerance', 'rate', 'offset-policy', 'audio-count', 'digest', 'normalization', 'header-only'])
def test_contradictory_decoded_clock_cannot_be_validated(tmp_path, damage):
    path = tmp_path / 'source.mp4'
    video(path)
    value = deepcopy(run_media('external', path.resolve()))
    clock = value['decoded_clock']
    if damage == 'count':
        clock['video']['frames'] += 1
    elif damage == 'tolerance':
        clock['video']['grid_tolerance_seconds'] = {'num': 1, 'den': 1}
    elif damage == 'rate':
        clock['video']['fps'] = {'num': 25, 'den': 1}
    elif damage == 'offset-policy':
        clock['zero_origin_film_compatible'] = False
    elif damage == 'audio-count':
        clock['audio']['samples'] += 123
    elif damage == 'digest':
        clock['video']['frame_pts_sha256'] = 'short'
    elif damage == 'normalization':
        clock['normalization_performed'] = True
    else:
        clock['fully_decoded'] = False
    with pytest.raises(MediaProcessError):
        _result('external', value)
