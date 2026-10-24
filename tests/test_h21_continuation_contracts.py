"""H21-A4 continuation integrity, real Core layout/math and mixed tail clocks.

Tiny random Core forwards are execution probes, not trained continuation quality.
No VAE seam/schedule/audio math is replaced and no GPU is initialized here.
"""
from copy import deepcopy
import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from comfy.ldm.minimax import model as native
from comfy.ldm.modules import attention

from h3_audio_t8_pkg import long_video as long
from h3_audio_t8_pkg import long_video_delivery as delivery
from h3_audio_t8_pkg.conditioning import build_packed_layout
from h3_audio_t8_pkg.core import empty_av_latent
from h3_audio_t8_pkg.native_latent_timeline_advanced import concat_native_h3_av_continuation
from test_fast_h3_v2_core_sampler import model
from test_native_latent_timeline_advanced import _continuation_proofs, _latent


def context(count=39):
    generator = torch.Generator().manual_seed(211004)
    return {"schema": 1, "empty": False,
            "video_tail": torch.randn((1, 24, long.CONTEXT_FRAME_STEPS[count], 4, 4), generator=generator),
            "audio_tail": torch.randn((1, 32, 2, round(count*5/3)), generator=generator),
            "metadata": {"source_segment_index": 0, "max_context_frames": count, "audio_overhang": 1/3}}


@pytest.mark.parametrize("overhang", [float("nan"), float("inf"), -0.1, 1.])
def test_context_rejects_invalid_audio_grid_phase_before_conditioning(overhang):
    value = context()
    value["metadata"]["audio_overhang"] = overhang
    with pytest.raises(ValueError, match="overhang|phase|finite"):
        long._validate_context(value, 1, 22, 64, 64)


@pytest.mark.parametrize("field", ["video_tail", "audio_tail"])
def test_context_rejects_nonfinite_actual_tail(field):
    value = context()
    value[field].flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        long._validate_context(value, 1, 22, 64, 64)


def test_direct_motion_marker_cannot_poison_real_core_position_ids():
    keyframes, ref = long._motion_context_blocks(context(), 22, True)
    layout = build_packed_layout(2, 17, 4, 4, 93, keyframes=keyframes, refs=[ref], frame_count=56)
    assert torch.isfinite(layout.position_ids).all()
    ref[long.MOTION_AUDIO_END_FRAME] = float("nan")
    with pytest.raises(ValueError, match="finite|audio.*frame|marker"):
        long.repair_long_video_layout(layout, keyframes, [ref], 56)
    assert torch.isfinite(layout.position_ids).all()


@pytest.mark.parametrize("writer", ["legacy", "candidate"])
@pytest.mark.parametrize("part", [0, 1])
def test_nonfinite_sampled_tail_is_not_published_as_a_native_checkpoint(monkeypatch, tmp_path, writer, part):
    monkeypatch.setattr(long.folder_paths, "get_output_directory", lambda: str(tmp_path))
    value = empty_av_latent(64, 64, 124)[0]
    value["samples"].unbind()[part].flatten()[-1] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        if writer == "legacy":
            long.save_context_state(value, "integrity", 0, "arbitrary user model", "arbitrary stack")
        else:
            delivery._write_context_candidate(value, tmp_path/"candidate.context.safetensors",
                                              "integrity", 0, "arbitrary user model", "arbitrary stack")
    assert not list(tmp_path.rglob("*.safetensors"))


@pytest.mark.parametrize("loader", ["legacy", "accepted"])
def test_native_checkpoint_nan_phase_is_not_blessed_by_tensor_checksums(monkeypatch, tmp_path, loader):
    monkeypatch.setattr(long.folder_paths, "get_output_directory", lambda: str(tmp_path))
    value = empty_av_latent(64, 64, 124)[0]
    path, _ = long.save_context_state(value, "integrity", 0, "user model", "user stack")
    with safe_open(path, framework="pt", device="cpu") as handle:
        state = {key: handle.get_tensor(key) for key in handle.keys()}
        metadata = dict(handle.metadata())
    metadata["audio_overhang"] = "nan"
    save_file(state, path, metadata)
    with pytest.raises(ValueError, match="overhang|phase|finite"):
        if loader == "legacy":
            long.load_context_state("integrity", 1)
        else:
            delivery._load_accepted_context_file(path, "integrity", 0, 1)


@pytest.mark.parametrize("count", [5, 22, 39])
def test_native_tail_actual_core_forward_uses_original_rows_without_reencoding(monkeypatch, count):
    monkeypatch.setattr(native, "optimized_attention", attention.attention_pytorch)
    source = context()
    snapshot = deepcopy(source)
    assert long._validate_context(source, 1, count, 64, 64)
    keyframes, audio_ref = long._motion_context_blocks(source, count, True)
    expected_video = source["video_tail"][:, :, -long.CONTEXT_FRAME_STEPS[count]:]
    assert torch.equal(torch.cat([row["latent"] for row in keyframes], dim=2), expected_video)
    # Selecting fewer temporal rows legitimately calls .contiguous(); native
    # ancestry means exact latent values, not a required shared-storage view.
    assert audio_ref[long.MOTION_AUDIO_END_FRAME] == pytest.approx(count + (1/3)/(5/3))
    assert torch.equal(audio_ref["audio_latent"], source["audio_tail"][..., -round(count*5/3):])
    layout = build_packed_layout(2, 17, 4, 4, 93, keyframes=keyframes, refs=[audio_ref], frame_count=56)
    long.repair_long_video_layout(layout, keyframes, [audio_ref], 56)
    diffusion = model().model.diffusion_model
    generator = torch.Generator().manual_seed(211005)
    x = [torch.randn((1, 24, 17, 4, 4), generator=generator),
         torch.randn((1, 32, 2, 93), generator=generator)]
    payload = {"layout": layout, "frame_count": 56,
               "cond_video_latents": [row["latent"] for row in keyframes],
               "cond_audio_latents": [audio_ref["audio_latent"]]}
    with torch.no_grad():
        result = diffusion(x, torch.tensor([500.]), torch.randn((1, 2, 8), generator=generator),
                           transformer_options={}, minimax_payload=payload)
    assert [v.shape for v in result] == [v.shape for v in x]
    assert all(torch.isfinite(value).all() for value in result)
    assert torch.equal(source["video_tail"], snapshot["video_tail"])
    assert torch.equal(source["audio_tail"], snapshot["audio_tail"])
    assert source["metadata"] == snapshot["metadata"]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("field", ["timeline_start_seconds", "timeline_end_seconds", "trim_start_seconds"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_native_concat_nonfinite_clock_cannot_bypass_exact_boundary_proof(field, nonfinite):
    planner_json, conditioning = _continuation_proofs(segment_index=1, timeline_frames=124)
    planner = json.loads(planner_json)
    planner[field] = nonfinite
    first, second = _latent(124, 1.), _latent(124, 2.)
    before = [value.clone() for source in (first, second) for value in source["samples"].unbind()]
    with pytest.raises(ValueError, match="finite|clock|seconds"):
        concat_native_h3_av_continuation(first, second, json.dumps(planner), conditioning)
    actual = [value for source in (first, second) for value in source["samples"].unbind()]
    assert all(torch.equal(value, old) for value, old in zip(actual, before, strict=True))
    valid = concat_native_h3_av_continuation(first, second, planner_json, conditioning)
    assert valid[1:5] == (226, 226, 0, 377)
    assert not torch.cuda.is_initialized()
