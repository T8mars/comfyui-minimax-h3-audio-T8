"""Non-face RGB/MASK/PCM roundtrips through existing local-rework consumers.

Procedural props test temporal and adoption contracts, not diffusion quality.
"""
from copy import deepcopy
import json

import pytest
import torch
from comfy_extras.nodes_mask import ImageCompositeMasked

from h3_audio_t8_pkg import source_conform as source
from h3_audio_t8_pkg import person_removal
from h3_audio_t8_pkg import face_refine_window_advanced as window
from h3_audio_t8_pkg import face_refine_window_studio_advanced as studio
from h3_audio_t8_pkg.face_refine_advanced import source_proxy_sha256


def scene(kind):
    raw = torch.full((112, 32, 64, 3), .15)
    masks = torch.zeros(raw.shape[:3])
    for frame in range(len(raw)):
        x = 8 + frame // 4 % 36
        width = 1 if kind == 'fine_edge' else 6
        if kind == 'occlusion' and 20 <= frame < 30:
            continue
        raw[frame, 11:20, x:x+width] = raw.new_tensor([.7, .2, .1])
        masks[frame, 11:20, x:x+width] = 1.
    pcm = torch.arange(112000, dtype=torch.float32) / 48000
    audio = {'sample_rate': 48000, 'waveform': torch.stack([
        .1 * torch.sin(2*torch.pi*pcm*431), .1 * torch.sin(2*torch.pi*pcm*877)])[None]}
    conformed, track, _, _, mapping, _ = source.conform(raw, 48., 64, 32, 56, source_audio=audio)
    _, lineage = source.derive_control(raw, 'manual_mask', masks)
    mask, _, report = person_removal.prepare_mask(conformed, masks, 0, 1, mapping, lineage)
    assert report['frame_map_report']['lineage_status'] == 'owned_preprocess_verified'
    return conformed, mask, track


def plan(frames, ranges='2-10,30-40'):
    return window.build_face_refine_window_plan(frames, 24., ranges, 'frames_inclusive',
        2, 2, 22, 56, 1., 'reject', 'reject', True)[0]


def candidate(frames, masks, planned, index=0, audio=None):
    render, selected_audio, mapping, *_ = window.extract_face_refine_window(
        frames, planned, index, 'reject', audio)
    local_mask = torch.stack([masks[row['source_frame']] if row['source_frame'] is not None
                             else torch.zeros_like(masks[0]) for row in mapping['frame_map']])
    # Actual Core compositing; no face detector/tracker or model selection.
    changed = ImageCompositeMasked.execute(render, torch.full_like(render, .75),
        0, 0, False, local_mask).result[0]
    return changed, local_mask, mapping, selected_audio


@pytest.mark.parametrize('kind', ['prop', 'occlusion', 'fine_edge'])
def test_nonface_preview_reject_subset_and_durable_adoption(kind, tmp_path, monkeypatch):
    import h3_audio_t8_pkg.long_video_delivery as delivery
    monkeypatch.setattr(delivery.folder_paths, 'get_output_directory', lambda: str(tmp_path))
    frames, masks, audio = scene(kind)
    original, original_pcm = frames.clone(), audio['waveform'].clone()
    planned = plan(frames)
    changed, local_mask, mapping, selected_audio = candidate(frames, masks, planned, audio=audio)
    if kind == 'occlusion':
        assert torch.count_nonzero(masks[10:15]) == 0
        assert torch.count_nonzero(masks[15]) > 0
    assert torch.count_nonzero(masks) > 0
    assert selected_audio['sample_rate'] == 32000
    for decision in ('preview_only', 'reject'):
        result = window.apply_face_refine_manual_review(frames, changed, local_mask,
            mapping, decision, '', False, 0)
        assert torch.equal(result[1], original)
        assert not torch.count_nonzero(result[2])
    adopted = window.apply_face_refine_manual_review(frames, changed, local_mask,
        mapping, 'accept_selected', '4-10', True, 0)
    assert json.loads(adopted[-1])['source_identity'] == 'full_content'
    assert torch.equal(adopted[1][:4], original[:4])
    assert torch.equal(adopted[1][11:], original[11:])
    outside = (adopted[2] == 0).unsqueeze(-1).expand_as(frames)
    assert torch.equal(adopted[1][outside], original[outside])
    if kind == 'occlusion':
        assert torch.equal(adopted[1][10:15], original[10:15])
    studio.prepare_face_refine_window_studio(planned, 'nonface')
    committed = studio.commit_face_refine_window_studio(frames, changed, local_mask,
        mapping, planned, 'nonface', 'accept_selected', '4-10', True, 0)
    assert committed[4] is True
    other, other_mask, other_map, _ = candidate(frames, masks, planned, 1, audio)
    studio.commit_face_refine_window_studio(frames, other, other_mask, other_map,
        planned, 'nonface', 'reject', '', False, 0)
    reopened = studio.load_face_refine_window_studio(studio.studio_chain_id('nonface'))
    assert [row['state'] for row in reopened['windows']] == ['accepted', 'rejected']
    composed = studio.compose_face_refine_window_studio(frames, planned, 'nonface')
    assert composed[2] and torch.equal(composed[0], adopted[1])
    assert torch.equal(audio['waveform'], original_pcm)
    assert torch.equal(frames, original)


def test_fine_pixel_change_invisible_to_scene_proxy_invalidates_new_plan_and_studio(tmp_path, monkeypatch):
    import h3_audio_t8_pkg.long_video_delivery as delivery
    monkeypatch.setattr(delivery.folder_paths, 'get_output_directory', lambda: str(tmp_path))
    frames, masks, audio = scene('fine_edge')
    planned = plan(frames)
    changed_source = frames.clone()
    changed_source[0, 0, 0] = .9
    assert source_proxy_sha256(frames) == source_proxy_sha256(changed_source)
    with pytest.raises(ValueError, match='content differs'):
        window.extract_face_refine_window(changed_source, planned, 0, 'reject')
    changed, mask, mapping, _ = candidate(frames, masks, planned, audio=audio)
    rejected = window.apply_face_refine_manual_review(changed_source, changed, mask,
        mapping, 'accept_selected', '', True, 0)
    assert json.loads(rejected[-1])['status'] == 'rejected_contract'
    assert torch.equal(rejected[1], changed_source)
    studio.prepare_face_refine_window_studio(planned, 'fine-source')
    with pytest.raises(ValueError, match='content differs'):
        studio.compose_face_refine_window_studio(changed_source, planned, 'fine-source')


def test_second_rework_binds_the_adopted_ancestor_and_keeps_legacy_plans_readable():
    frames, masks, _ = scene('prop')
    planned = plan(frames)
    changed, mask, mapping, _ = candidate(frames, masks, planned)
    accepted = window.apply_face_refine_manual_review(frames, changed, mask,
        mapping, 'accept_selected', '', True, 0)[1]
    with pytest.raises(ValueError, match='source-bound'):
        window.extract_face_refine_window(accepted, planned, 0, 'reject')
    second = plan(accepted, '4-10')
    assert second['source']['content_sha256'] != planned['source']['content_sha256']
    changed2, mask2, mapping2, _ = candidate(accepted, masks, second)
    rejected2 = window.apply_face_refine_manual_review(accepted, changed2, mask2,
        mapping2, 'reject', '', False, 0)
    assert torch.equal(rejected2[1], accepted)
    legacy = deepcopy(planned)
    del legacy['source']['content_sha256']
    del legacy['plan_sha256']
    legacy = window._signed(legacy, 'plan_sha256')
    changed, mask, mapping, _ = candidate(frames, masks, legacy)
    report = window.apply_face_refine_manual_review(frames, changed, mask,
        mapping, 'preview_only', '', False, 0)[-1]
    assert json.loads(report)['source_identity'] == 'legacy_proxy_only'


def test_wrong_mask_source_frame_count_and_nonfinite_candidate_do_not_adopt():
    frames, masks, _ = scene('prop')
    planned = plan(frames)
    changed, mask, mapping, _ = candidate(frames, masks, planned)
    bad = changed.clone()
    bad[0, 0, 0, 0] = float('nan')
    for image, selected_mask in ((bad, mask), (changed[:-1], mask[:-1]), (changed, mask[:-1])):
        result = window.apply_face_refine_manual_review(frames, image, selected_mask,
            mapping, 'accept_selected', '', True, 0)
        assert json.loads(result[-1])['status'] == 'rejected_contract'
        assert torch.equal(result[1], frames)
