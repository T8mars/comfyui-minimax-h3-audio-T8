import json

import pytest
import torch

from h3_audio_t8_pkg.h07_continuation import continuation_card
from h3_audio_t8_pkg.h07_reports import reference_summary, continuity_lint
from h3_audio_t8_pkg.freevideo_quality.profiles import AUDIO_RUNTIME
from h3_audio_t8_pkg.res_history_exp import RESHistory, save_checkpoint
from test_h07_audio_version import completed
from test_freevideo_split import mid


def test_quality_completed_read_is_not_high_sampling_or_identity_approval():
    low, high = completed(AUDIO_RUNTIME), completed(AUDIO_RUNTIME, "HIGH")
    before = low.audio.clone()
    card = continuation_card(completed_quality_stage=low)
    assert card["completed_NFE"] == 8 and card["additional_NFE"] == 3 and card["read_extra_NFE"] == 0
    assert not card["sampling_started"] and not card["cache_reuse_authorized"]
    assert card["next_node"] == "MiniMaxH3FreeVideoCommunityHIGH3EXPT8"
    assert continuation_card(completed_quality_stage=high)["additional_NFE"] == 0
    assert torch.equal(low.audio, before)
    report = reference_summary('{"pictures":{},"videos":{},"audios":{}}', completed_quality_stage=high)
    assert report["completion"]["actual_NFE"] == 3
    with pytest.raises(ValueError, match="exactly one"):
        continuation_card(completed_quality_stage=low, freevideo_mid=mid())
    with pytest.raises(ValueError):
        continuation_card(completed_quality_stage={"path": "movie.mp4"})


def test_actual_mid_receipt_requires_remaining4_and_noisy_audio():
    state = mid()
    card = continuation_card(freevideo_mid=state)
    assert not card["complete_AV"] and card["completed_NFE"] == card["additional_NFE"] == 4
    assert card["next_node"] == "MiniMaxH3FreeVideoSplitHIGHEXPT8"
    state.audio.add_(1)
    with pytest.raises(ValueError, match="content"):
        continuation_card(freevideo_mid=state)


def test_actual_res_boundary_card_keeps_solver_history_and_sha(tmp_path):
    full = torch.linspace(1., 0., 9)
    x = torch.ones(1, 8)
    state = RESHistory(4, x, x * .8, full[4])
    saved = save_checkpoint(tmp_path, "post4.h3res.safetensors", state, full,
        original_noise=x * .1, original_latent_image=x * 0,
        run_contract={"test": "actual_tiny_file_not_GPU_model_qualification"})
    card = continuation_card(res_checkpoint_path="post4.h3res.safetensors",
        res_checkpoint_sha256=saved["file_sha256"], checkpoint_root=tmp_path)
    assert card["additional_NFE"] == 4 and not card["complete_AV"] and card["read_extra_NFE"] == 0
    assert card["restore_mode"] == "resume" and not card["cache_reuse_authorized"]
    with pytest.raises(ValueError, match="SHA"):
        continuation_card(res_checkpoint_path="post4.h3res.safetensors",
            res_checkpoint_sha256="f" * 64, checkpoint_root=tmp_path)


def test_malformed_explicit_role_and_transfer_records_have_clear_errors():
    media = '{"audios":{"1":"ref_audio_0"}}'
    with pytest.raises(ValueError, match="dictionary"):
        reference_summary(media, json.dumps([dict(role_id="A", tags=["Audio 1"], expected_labels=[])]))
    with pytest.raises(ValueError, match="bounded text"):
        reference_summary(media, json.dumps([dict(role_id="A", tags=["Audio 1"], Subject=1)]))
    with pytest.raises(ValueError, match="from/to"):
        continuity_lint("[]", '[{"t":1,"id":"cup_a"}]')
