"""N10 tiny real-Core row sentinels, not a claim of mixed-reference AV quality."""
import torch
from torch import nn

from comfy.model_base import MiniMaxH3
from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout, pack_audio


def test_native_extra_conds_keeps_guide_then_two_distinct_reference_audio_rows():
    guide = torch.full((1, 3, 2, 2), 11.)
    ref = torch.full((1, 3, 2, 3), 22.)
    video_ref = torch.full((1, 3, 2, 4), 33.)
    keyframes = [{"resolved_frame_index": 0, "audio_latent": guide}]
    refs = [{"kind": "audio", "ref_audio_t": 3, "audio_latent": ref},
            {"kind": "video_audio", "ref_audio_t": 4, "audio_latent": video_ref,
             "latent_t": 1, "latent_h": 4, "latent_w": 6,
             "latent": torch.zeros(1, 2, 1, 4, 6)}]
    base = object.__new__(MiniMaxH3)
    base.concat_keys, base.latent_shapes = (), None
    payload = MiniMaxH3.extra_conds(base, minimax_keyframes=keyframes, minimax_refs=refs)["minimax_payload"].cond
    payload["audio_cond_noise_aug"] = 1.
    model = MiniMaxH3Model(hidden_size=8, num_layers=0, token_refiner_num_layers=0,
        num_attention_heads=1, attention_head_dim=8, ffn_hidden_size=8,
        latents_dim=2, audio_latents_dim=3, text_dim=5, timestep_input_dim=8,
        time_embed_hidden_size=8, time_embed_dim=8, dtype=torch.float32, device="cpu", operations=nn)
    expected = torch.cat([pack_audio(guide), pack_audio(ref), pack_audio(video_ref)])
    actual = model._cond_audio_rows(payload, "cpu")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    layout = PackedLayout(4, 1, 4, 6, 2, keyframes=keyframes, refs=refs)
    assert sum(b - a for a, b, kind in layout.segments if kind in {"cond_audio", "ref_audio"}) == len(expected)
    # Also exercise real packing/projections, not only list lengths.
    video, audio, context = torch.zeros(1, 2, 1, 4, 6), torch.full((1, 3, 2, 2), 44.), torch.zeros(1, 4, 8)
    with torch.no_grad():
        packed = model._embed_and_pack(video, audio, context, layout, payload, {})
        audio_rows = torch.cat([packed[a:b] for a, b, kind in layout.segments
                                if kind in {"cond_audio", "ref_audio", "audio"}])
        projected = model.audio_patch_proj(torch.cat([expected, pack_audio(audio)]))
    torch.testing.assert_close(audio_rows, projected, rtol=0, atol=0)


def test_t8_reference_audio_metadata_equals_actual_encoded_time():
    from h3_audio_t8_pkg.conditioning import build_conditioning
    from helpers import FakeAudioVAE, FakeClip, FakeVideoVAE, make_audio

    cond, *_ = build_conditioning(clip=FakeClip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
        prompt="<Audio 1> is a reference", width=128, height=128, length=124,
        task_type="Hybrid", audio_mode="native", first_frame=torch.zeros(1, 128, 128, 3),
        ref_audios={"ref_audio_1": make_audio(1), "ref_audio_2": make_audio(2)})
    refs = cond[0][1]["minimax_refs"]
    audio_refs = [r for r in refs if r.get("audio_latent") is not None]
    assert len(audio_refs) == 2
    assert all(r["ref_audio_t"] == r["audio_latent"].shape[-1] for r in audio_refs)
    assert audio_refs[0]["audio_latent"].shape[-1] != audio_refs[1]["audio_latent"].shape[-1]
