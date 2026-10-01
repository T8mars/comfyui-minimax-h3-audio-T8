"""Compare the live AV manifest auditor with the unchanged formal mirror.

The mirror is loaded only as test code, with the live dependency modules, so
the one changed validation branch is isolated from unrelated package drift.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import types

import comfy.nested_tensor
import pytest
import torch

from h3_audio_t8_pkg.native_latent_timeline_advanced import (
    audit_native_h3_av_latent_resume_manifest as current_audit,
)
from test_native_latent_checkpoint_advanced import _latent


ROOT = Path(__file__).resolve().parents[1]
FORMAL_MIRROR = ROOT / "native_latent_timeline_advanced.py"
LIVE_RUNTIME = ROOT / "h3_t8/native_latent_timeline_advanced.py"
FORMAL_SHA256 = "084bf105c01df1768c1baecc9da2770113bdd88312fb50a24d626d3d9f3c6958"


def _formal_audit(monkeypatch):
    alias = "_t8_formal_mask_legacy_parity"
    package = types.ModuleType(alias)
    package.__path__ = [str(ROOT)]
    monkeypatch.setitem(sys.modules, alias, package)
    for dependency in ("core", "long_video"):
        monkeypatch.setitem(
            sys.modules, f"{alias}.{dependency}",
            importlib.import_module(f"h3_audio_t8_pkg.{dependency}"),
        )
    spec = importlib.util.spec_from_file_location(
        f"{alias}.native_latent_timeline_advanced", FORMAL_MIRROR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module.audit_native_h3_av_latent_resume_manifest


def test_live_runtime_diff_is_only_the_broadcast_video_mask_allowance():
    formal = FORMAL_MIRROR.read_bytes()
    assert hashlib.sha256(formal).hexdigest() == FORMAL_SHA256
    old = FORMAL_MIRROR.read_text(encoding="utf-8")
    before_mask_parts = """    if len(parts) != 2 or parts[0].ndim != 5 or parts[1].ndim != 4:
"""
    after_mask_parts = """    if len(parts) != 2 or parts[0].ndim not in (4, 5) or parts[1].ndim != 4:
"""
    before = """    masks = _mask_parts(av_latent)
    if masks is not None:
        if masks[0].shape != video.shape or masks[1].shape != audio.shape:
"""
    after = """    masks = _mask_parts(av_latent)
    if masks is not None:
        # Preserve both the latent-time one-channel mask and Core's native
        # SetLatentNoiseMask [F,1,H,W] form. Core interpolates its frame and
        # spatial dimensions for sampling; the LATENT retains source shape.
        broadcast_video_shape = (video.shape[0], 1, *video.shape[2:])
        native_pixel_mask = (masks[0].ndim == 4 and masks[0].shape[1] == 1
                             and all(size > 0 for size in masks[0].shape))
        if ((masks[0].shape not in (video.shape, broadcast_video_shape)
                and not native_pixel_mask)
                or masks[1].shape != audio.shape):
"""
    assert old.count(before_mask_parts) == 1
    assert old.count(before) == 1
    assert LIVE_RUNTIME.read_text(encoding="utf-8") == (
        old.replace(before_mask_parts, after_mask_parts).replace(before, after))


@pytest.mark.parametrize("frames,masked", ((22, False), (22, True),
                                            (124, False), (124, True)))
def test_legacy_no_mask_and_full_mask_manifest_are_byte_identical(
    monkeypatch, frames, masked,
):
    formal_audit = _formal_audit(monkeypatch)
    latent = _latent(frames=frames, mask=masked)
    arguments = {"checkpoint_id": f"old_{frames}_{masked}", "hash_chunk_megabytes": 1}
    formal = formal_audit(latent, **arguments)
    current = current_audit(latent, **arguments)
    assert current == formal
    assert current[3].encode("utf-8") == formal[3].encode("utf-8")
    assert current_audit(latent, expected_manifest_json=formal[3], **arguments) == (
        formal_audit(latent, expected_manifest_json=formal[3], **arguments))


def test_one_channel_mask_is_additive_not_a_legacy_reinterpretation(monkeypatch):
    formal_audit = _formal_audit(monkeypatch)
    latent = _latent(frames=22, mask=True)
    video_mask, audio_mask = latent["noise_mask"].unbind()
    compact = video_mask[:, :1].clone()
    latent["noise_mask"] = comfy.nested_tensor.NestedTensor((compact, audio_mask))
    with pytest.raises(ValueError, match="noise_mask shapes"):
        formal_audit(latent, checkpoint_id="new_broadcast")
    status, verified, digest, manifest = current_audit(
        latent, checkpoint_id="new_broadcast")
    assert status == "BASELINE_CREATED" and verified is False and len(digest) == 64
    assert current_audit(latent, checkpoint_id="new_broadcast",
                         expected_manifest_json=manifest)[0] == "MATCH"
    widened = _latent(frames=22, mask=True)
    widened["noise_mask"] = comfy.nested_tensor.NestedTensor((
        compact.expand_as(video_mask).clone(), audio_mask))
    assert current_audit(widened, checkpoint_id="new_broadcast")[2] != digest
    assert torch.equal(latent["samples"].unbind()[1], widened["samples"].unbind()[1])


def test_pixel_frame_mask_is_additive_and_kept_distinct_from_latent_time(monkeypatch):
    formal_audit = _formal_audit(monkeypatch)
    latent = _latent(frames=22, mask=True)
    video_mask, audio_mask = latent["noise_mask"].unbind()
    pixel_mask = torch.linspace(0, 1, 11 * 13).reshape(1, 1, 11, 13)
    latent["noise_mask"] = comfy.nested_tensor.NestedTensor((pixel_mask, audio_mask))
    with pytest.raises(ValueError, match="invalid nested AV noise_mask"):
        formal_audit(latent, checkpoint_id="native_pixel_mask")
    status, verified, digest, manifest = current_audit(
        latent, checkpoint_id="native_pixel_mask")
    assert status == "BASELINE_CREATED" and verified is False and len(digest) == 64
    assert current_audit(latent, checkpoint_id="native_pixel_mask",
                         expected_manifest_json=manifest)[0] == "MATCH"
    latent_time = _latent(frames=22, mask=True)
    latent_time["noise_mask"] = comfy.nested_tensor.NestedTensor((
        video_mask[:, :1].clone(), audio_mask))
    assert current_audit(latent_time, checkpoint_id="native_pixel_mask")[2] != digest
