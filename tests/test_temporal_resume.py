"""Literal restart without CLIP/lift/noise, plus real completed-source gate."""
from dataclasses import replace
import json

import pytest
import torch

from h3_audio_t8_pkg.modular_sampling.temporal_resume import save_resume, load_resume
from h3_audio_t8_pkg.nodes_temporal_resume import (
    preserve_completed_audio, MiniMaxH3TemporalV5ResumeSaveEXPT8, MiniMaxH3TemporalV5ResumeLoadEXPT8,
)
from h3_audio_t8_pkg.modular_sampling.temporal_chunked_v5 import sample_scoped_v5_window
from h3_audio_t8_pkg.modular_sampling.results import sample_stage
from test_temporal_dialogue_bank import run, scoped_harness as _bank_fixture
from test_modular_results import inputs


scoped_harness = _bank_fixture


def test_literal_capsule_resumes_remaining_only_without_encoding_lift_or_noise(scoped_harness, tmp_path):
    h = scoped_harness
    _, _, first, _ = run(h, 0)
    expected, _, _, _ = run(h, 1, first)
    path, digest, _ = save_resume(first, h[1], h[8], h[9], h[5], h[7], tmp_path)
    before = dict(h[0])
    clip_calls = len(h[6].tokenize_calls)
    source, lifted, prepared, plan, bank, previous, report = load_resume(tmp_path, path, digest, 0)
    assert json.loads(report)['status'].startswith('literal_resume_loaded_no_sampling')
    assert source is not h[1] and bank is not h[7]
    output, *_ = sample_scoped_v5_window(object(), source, lifted, prepared, plan,
        h[2], h[3], h[4], bank, 1, previous)
    for a, b in zip(output['samples'].unbind(), expected['samples'].unbind(), strict=True):
        assert torch.equal(a, b)
    assert h[0] == {**before, 'sample':before['sample']+1, 'sampler_bind':before['sampler_bind']+1}
    assert len(h[6].tokenize_calls) == clip_calls
    with pytest.raises(ValueError, match='expected window'):
        load_resume(tmp_path, path, digest, 1)
    with pytest.raises(ValueError, match='manifest SHA'):
        load_resume(tmp_path, path, 'f'*64, 0)


def test_public_capsule_confirm_and_selected_fingerprint(scoped_harness, tmp_path, monkeypatch):
    import h3_audio_t8_pkg.nodes_temporal_resume as nodes
    monkeypatch.setattr(nodes, '_root', lambda:tmp_path)
    h = scoped_harness
    _, _, first, _ = run(h, 0)
    args = first, h[1], h[8], h[9], h[5], h[7]
    unsaved = MiniMaxH3TemporalV5ResumeSaveEXPT8.execute(*args).result
    assert unsaved[2:4] == ('','') and not list(tmp_path.iterdir())
    saved = MiniMaxH3TemporalV5ResumeSaveEXPT8.execute(*args, True).result
    restored = MiniMaxH3TemporalV5ResumeLoadEXPT8.execute(saved[2], saved[3], 0).result
    assert restored[5].dialogue_receipt == first.dialogue_receipt
    assert isinstance(MiniMaxH3TemporalV5ResumeLoadEXPT8.fingerprint_inputs(saved[2], saved[3]), tuple)


def test_completed_audio_uses_observed_terminal_stage_but_rejects_actual_partial():
    full = sample_stage(*inputs('high_4_8'))[2]
    passed, report = preserve_completed_audio(full.denoised_output, full)
    assert passed['samples'].tensors[1] is full.output['samples'].tensors[1]
    assert not json.loads(report)['quality_accepted']
    partial = sample_stage(*inputs('low_0_4'))[2]
    with pytest.raises(ValueError, match='terminal-zero'):
        preserve_completed_audio(partial.denoised_output, partial)
    with pytest.raises(ValueError, match='actual completed Stage Result'):
        preserve_completed_audio(partial.denoised_output, partial.output)
    receipt = full.verify()
    receipt['verified_recipe_completion'] = False
    from h3_audio_t8_pkg.modular_sampling.results import canonical, sha
    receipt.pop('receipt_sha256')
    receipt['receipt_sha256'] = sha(receipt)
    with pytest.raises(ValueError, match='terminal-zero'):
        preserve_completed_audio(full.output, replace(full, receipt_json=canonical(receipt)))
