"""Compiler -> actual trim -> codec -> batch contract on a numbered CPU clip."""
import json
import math

import av
import pytest
import torch

from h3_audio_t8_pkg.director_batch import _check_delivery, _delivery_contract
from h3_audio_t8_pkg.director_generation import build_director_generation_prompt
from h3_audio_t8_pkg.director_media import batch_media
from h3_audio_t8_pkg.director_project import new_project
from h3_audio_t8_pkg.h3_av_delivery import save_h3_av_safe
from h3_audio_t8_pkg.nodes import MiniMaxH3OutputTrimT8
from test_director_generation import _patch_director_models, _store


@pytest.mark.parametrize('seconds', [5., 5.1, 5.1667, 5.2])
def test_compiled_delivery_survives_actual_trim_save_and_batch(tmp_path, monkeypatch, seconds):
    _patch_director_models(monkeypatch)
    project = new_project()
    shot = project['doc']['shots'][0]
    shot.update(simplePrompt='One continuous shot.', manualDuration=seconds, autoDuration=False)
    built = build_director_generation_prompt(project, shot['id'], _store(tmp_path))
    expected = _delivery_contract({'built': built})
    length = built['shot']['time']['aligned_frames']
    assert length % 17 == 5 and length >= expected['frames']
    # Numbered scalar pixels expose a hidden shift/crop before lossy encoding.
    raw = torch.arange(length, dtype=torch.float32).reshape(-1, 1, 1, 1)
    rate = 48000
    waveform = torch.arange(round(length * rate / 24), dtype=torch.float32).reshape(1, 1, -1)
    waveform = waveform / waveform.shape[-1] * .1
    audio = {'waveform': waveform.clone(), 'sample_rate': rate}
    inputs = {k: v for k, v in built['prompt']['11']['inputs'].items() if k not in ('frames', 'audio')}
    frames, sound, report = MiniMaxH3OutputTrimT8.execute(raw, audio=audio, **inputs).result
    assert frames.shape[0] == expected['frames'] == math.ceil(seconds * 24)
    assert torch.equal(frames[:, 0, 0, 0], torch.arange(expected['frames']).float())
    assert torch.equal(sound['waveform'], waveform[..., :round(seconds * rate)])
    assert torch.equal(audio['waveform'], waveform)
    assert json.loads(report)['requested_duration_seconds'] == seconds
    images = (frames / length).expand(-1, 32, 32, 3).contiguous()
    path = tmp_path / 'delivery.mp4'
    result = save_h3_av_safe(images, sound, path,
                           preserve_audio_length=built['prompt']['12']['inputs']['preserve_audio_length'])
    assert result['audio']['encoded_samples'] == round(seconds * rate)
    evidence = batch_media(path)
    _check_delivery(evidence, {**expected, 'width': 32, 'height': 32})
    with av.open(str(path)) as media:
        decoded = list(media.decode(video=0))
        assert len(decoded) == expected['frames']
        assert [round(float(f.pts * f.time_base) * 24) for f in decoded] == list(range(expected['frames']))


def test_legacy_trim_keeps_rounding_and_explicit_count_only_changes_video():
    raw = torch.arange(141).reshape(-1, 1, 1, 1).float()
    audio = {'waveform': torch.arange(6000).reshape(1, 1, -1).float(), 'sample_rate': 1000}
    legacy = MiniMaxH3OutputTrimT8.execute(raw, 0., 5.1, 24., audio).result
    assert len(legacy[0]) == round(5.1 * 24)
    explicit = MiniMaxH3OutputTrimT8.execute(raw, 0., 5.1, 24., audio, frame_count=123).result
    assert len(explicit[0]) == 123
    assert torch.equal(legacy[1]['waveform'], explicit[1]['waveform'])


@pytest.mark.parametrize('count', [True, -1, 1.5, '123'])
def test_explicit_count_requires_a_real_nonnegative_integer(count):
    raw = torch.zeros(141, 1, 1, 1)
    with pytest.raises(ValueError, match='frame_count'):
        MiniMaxH3OutputTrimT8.execute(raw, 0., 5.1, 24., frame_count=count)
