"""Public schemas and actual scoped chain consumption, not trained quality."""
import asyncio
from dataclasses import asdict
import json

import torch

import h3_audio_t8_pkg
from h3_audio_t8_pkg.nodes import MiniMaxH3AudioConditioningT8
from h3_audio_t8_pkg.nodes_temporal_dialogue import (
    NODES, MiniMaxH3TemporalDialoguePlanEXPT8, MiniMaxH3TemporalNativeRecipeEXPT8,
    MiniMaxH3TemporalV5JointPass2EXPT8, MiniMaxH3TemporalV5WindowEXPT8,
    MiniMaxH3TemporalV5WindowSaveEXPT8, MiniMaxH3TemporalV5WindowLoadEXPT8,
)
from helpers import FakeClip, FakeVideoVAE, FakeAudioVAE
from test_temporal_dialogue_bank import run, scoped_harness as _bank_fixture
from test_temporal_dialogue_scope import user_plan


scoped_harness = _bank_fixture


def test_appends_after_all_632_released_nodes_and_capture_does_not_mutate_schema():
    classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    ids = [cls.define_schema().node_id for cls in classes]
    assert len(ids) == len(set(ids)) == 632+len(NODES)
    assert classes[632:] == NODES
    before = asdict(MiniMaxH3AudioConditioningT8.define_schema().get_v1_info(MiniMaxH3AudioConditioningT8))
    new = asdict(MiniMaxH3TemporalNativeRecipeEXPT8.define_schema().get_v1_info(MiniMaxH3TemporalNativeRecipeEXPT8))
    assert new['input'] == before['input'] and new['input_order'] == before['input_order']
    assert new['output'][:6] == before['output'] and len(new['output']) == 7
    assert asdict(MiniMaxH3AudioConditioningT8.define_schema().get_v1_info(MiniMaxH3AudioConditioningT8)) == before


def test_public_plan_full_LOW_prompt_contains_each_explicit_event_once_and_native_recipe():
    raw = [dict(event_id=e.event_id, speaker=e.speaker, utterance=e.utterance,
                start_seconds=e.requested_start_seconds, end_seconds=e.requested_end_seconds)
           for e in user_plan().events]
    plan, prompt, report = MiniMaxH3TemporalDialoguePlanEXPT8.execute('one room', json.dumps(raw), '[]', 288).result
    assert plan.total_frames == 294 and all(prompt.count(e.utterance) == 1 for e in plan.events)
    assert not json.loads(report)['hard_time_isolation']
    values = MiniMaxH3TemporalNativeRecipeEXPT8.execute(clip=FakeClip(), video_vae=FakeVideoVAE(),
        audio_vae=FakeAudioVAE(), prompt=prompt, width=64, height=64, length=294, audio_mode='native').result
    assert len(values) == 7 and values[-1].frame_count == 294


def test_public_integrated_chain_matches_separated_actual_AV_exactly(scoped_harness):
    h = scoped_harness
    _, _, first, _ = run(h, 0)
    expected, _, _, _ = run(h, 1, first)
    calls_before = dict(h[0])
    result = MiniMaxH3TemporalV5JointPass2EXPT8.execute(
        object(), h[1], h[5], h[2], h[3], h[4], h[7]).result
    actual, base, scoped, _lifted, _prepared, report = result
    assert scoped.base is base and base.index == 1
    assert h[0]['lift'] == calls_before['lift']+1 and h[0]['noise'] == calls_before['noise']+1
    assert h[0]['sample'] == calls_before['sample']+2
    for left, right in zip(actual['samples'].unbind(), expected['samples'].unbind(), strict=True):
        assert torch.equal(left, right)
    assert json.loads(report)['audio_output'] == 'refined_joint_audio'


def test_public_save_load_remains_explicit_and_requires_scoped_receipt(scoped_harness, monkeypatch, tmp_path):
    import h3_audio_t8_pkg.nodes_temporal_dialogue as nodes
    monkeypatch.setattr(nodes, '_store_root', lambda: tmp_path)
    h = scoped_harness
    actual = MiniMaxH3TemporalV5WindowEXPT8.execute(object(), h[1], h[8], h[9], h[5], h[7],
        h[2], h[3], h[4], 0).result
    result = actual[2]
    unchanged = MiniMaxH3TemporalV5WindowSaveEXPT8.execute(result, h[1], h[8], h[9], h[5], h[7]).result
    assert unchanged[3:5] == ('', '') and not list(tmp_path.iterdir())
    saved = MiniMaxH3TemporalV5WindowSaveEXPT8.execute(result, h[1], h[8], h[9], h[5], h[7], True).result
    restored = MiniMaxH3TemporalV5WindowLoadEXPT8.execute(h[1], h[8], h[9], h[5], h[7],
        0, saved[3], saved[4]).result
    assert restored[2].dialogue_receipt == result.dialogue_receipt
    assert restored[1].output_identity == result.base.output_identity
    assert isinstance(MiniMaxH3TemporalV5WindowLoadEXPT8.fingerprint_inputs(saved[3], saved[4]), tuple)
