"""A shorter repair decode must not discard untouched source PCM."""
import pytest
import torch

from h3_audio_t8_pkg.lanpaint_av_advanced import composite_lanpaint_av


def inputs():
    frames = torch.full((5, 32, 32, 3), .4)
    source = {'waveform': torch.linspace(-.2, .2, 6667).repeat(1, 2, 1), 'sample_rate': 32000}
    repaired = {'waveform': torch.full((1, 2, 3200), .1), 'sample_rate': 32000}
    return frames, source, repaired


@pytest.mark.parametrize('intervals', ['[]', '[{"start":0.01,"end":0.05}]'])
def test_shorter_repair_keeps_unedited_source_tail_and_output_length(intervals):
    frames, source, repaired = inputs()
    _, audio, _ = composite_lanpaint_av(frames, frames, source, repaired,
        intervals, 1, .01, torch.zeros(frames.shape[:3]))
    assert audio['waveform'].shape == source['waveform'].shape
    assert torch.equal(audio['waveform'][..., 3200:], source['waveform'][..., 3200:])
    if intervals == '[]':
        assert torch.equal(audio['waveform'], source['waveform'])


def test_requested_repair_past_available_candidate_reports_missing_samples():
    frames, source, repaired = inputs()
    with pytest.raises(ValueError, match='(?i)repaired audio|candidate audio'):
        composite_lanpaint_av(frames, frames, source, repaired,
            '[{"start":0.15,"end":0.2}]', 1, 0., torch.zeros(frames.shape[:3]))


def test_crossfade_cannot_read_nonexistent_repaired_tail():
    frames, source, repaired = inputs()
    with pytest.raises(ValueError, match='crossfade'):
        composite_lanpaint_av(frames, frames, source, repaired,
            '[{"start":0.08,"end":0.1}]', 1, .02, torch.zeros(frames.shape[:3]))


def test_resampled_short_repair_preserves_source_rate_and_tail():
    frames, source, repaired = inputs()
    repaired = {'waveform': repaired['waveform'][..., :1600], 'sample_rate': 16000}
    _, audio, _ = composite_lanpaint_av(frames, frames, source, repaired,
        '[{"start":0.01,"end":0.05}]', 1, 0., torch.zeros(frames.shape[:3]))
    assert audio['sample_rate'] == 32000
    assert audio['waveform'].shape == source['waveform'].shape
    assert torch.equal(audio['waveform'][..., 3200:], source['waveform'][..., 3200:])
