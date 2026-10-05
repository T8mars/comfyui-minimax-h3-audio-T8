"""H16 literal Full/Cold, no earlier work or cross-format prefix claim."""
from dataclasses import replace

import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as legacy
from h3_audio_t8_pkg.modular_sampling.temporal_h16_resume import (
    prepare_bundle,select_bundle_window,save_h16_resume,load_h16_resume,
)
from h3_audio_t8_pkg.modular_sampling.temporal_h16 import sample_scoped_h16
from h3_audio_t8_pkg.nodes_temporal_h16_resume import (
    MiniMaxH3TemporalH16BundlePrepareEXPT8,MiniMaxH3TemporalH16BundleWindowEXPT8,
    MiniMaxH3TemporalH16ResumeSaveEXPT8,MiniMaxH3TemporalH16ResumeLoadEXPT8,
)
from test_temporal_h16 import make_case
from test_modular_h16_stages import _native_like_piece,_no_op_lift
from helpers import FakeClip


def setup(monkeypatch):
    import h3_audio_t8_pkg.modular_sampling.temporal_h16 as scoped
    monkeypatch.setattr(scoped,'rebind_dual_clock_sampler',lambda _m,_p,s:s)
    monkeypatch.setattr(legacy,'sample_piece',_native_like_piece)
    return make_case(monkeypatch,FakeClip())


def run(bundle,index,noise,previous=None):
    source,lifted,spec,context,plan,bank = select_bundle_window(bundle,index)
    return sample_scoped_h16(object(),source,lifted,spec,context,plan,noise,object(),torch.tensor([.3,0.]),bank,previous)


def test_H16_literal_capsule_restores_all_pieces_conditions_noise_without_earlier_work(monkeypatch,tmp_path):
    _,plan,bank,noise,context = setup(monkeypatch)
    calls = []
    def lift(*args,**kwargs):
        calls.append('lift')
        return _no_op_lift(*args,**kwargs)
    monkeypatch.setattr(legacy,'learned_upscale_h3_av_latent',lift)
    bundle = prepare_bundle(context,plan,bank)
    assert calls == ['lift','lift']
    _,first,_ = run(bundle,0,noise)
    expected,_,_ = run(bundle,1,noise,first)
    path,digest,_ = save_h16_resume(bundle,first,tmp_path)
    restored,previous,_ = load_h16_resume(tmp_path,path,digest,0)
    actual,_,_ = run(restored,1,noise,previous)
    assert calls == ['lift','lift'] and restored.bank is not bank
    for left,right in zip(expected['samples'].unbind(),actual['samples'].unbind(),strict=True):
        assert torch.equal(left,right)
    assert previous.dialogue_receipt == first.dialogue_receipt
    with pytest.raises(ValueError,match='expected window'):
        load_h16_resume(tmp_path,path,digest,1)
    with pytest.raises(ValueError,match='manifest SHA'):
        load_h16_resume(tmp_path,path,'f'*64,0)


def test_H16_public_confirm_selection_and_exact_fingerprint(monkeypatch,tmp_path):
    import h3_audio_t8_pkg.nodes_temporal_h16_resume as nodes
    monkeypatch.setattr(nodes,'_root',lambda:tmp_path)
    _,plan,bank,noise,context = setup(monkeypatch)
    bundle = MiniMaxH3TemporalH16BundlePrepareEXPT8.execute(context,plan,bank).result[0]
    piece = MiniMaxH3TemporalH16BundleWindowEXPT8.execute(bundle,0).result
    assert piece[3] is context
    _,first,_ = run(bundle,0,noise)
    result = MiniMaxH3TemporalH16ResumeSaveEXPT8.execute(bundle,first).result
    assert result[2:4] == ('','') and not list(tmp_path.iterdir())
    result = MiniMaxH3TemporalH16ResumeSaveEXPT8.execute(bundle,first,True).result
    restored,previous,_ = MiniMaxH3TemporalH16ResumeLoadEXPT8.execute(result[2],result[3],0).result
    assert previous.output_identity == first.output_identity and restored.identity == bundle.identity
    assert isinstance(MiniMaxH3TemporalH16ResumeLoadEXPT8.fingerprint_inputs(result[2],result[3]),tuple)


def test_H16_literal_bundle_mutation_and_v5_file_are_not_accepted(monkeypatch,tmp_path):
    _,plan,bank,noise,context = setup(monkeypatch)
    bundle = prepare_bundle(context,plan,bank)
    with pytest.raises(ValueError,match='window index'):
        select_bundle_window(bundle,True)
    with pytest.raises(ValueError,match='bundle changed'):
        select_bundle_window(replace(bundle,identity={}),0)
    _,first,_ = run(bundle,0,noise)
    path,digest,_ = save_h16_resume(bundle,first,tmp_path)
    (tmp_path/path).write_text('{}',encoding='utf8')
    with pytest.raises(ValueError,match='manifest SHA'):
        load_h16_resume(tmp_path,path,digest,0)
