"""Chunked v1 explicit segment freeze and fresh-source continuation."""
from dataclasses import replace
import json

import comfy.nested_tensor
import pytest
import torch

from h3_audio_t8_pkg import chunked_two_pass_upscale_advanced as old
from h3_audio_t8_pkg.modular_sampling.chunked_source import slice_chunked_source
from h3_audio_t8_pkg.modular_sampling.chunked_stages import (
    lift_chunked_segment, prepare_chunked_pass2, sample_chunked_pass2,
)
from h3_audio_t8_pkg.modular_sampling.chunked_v1_storage import (
    load_segment, save_segment, verify_segment,
)
from h3_audio_t8_pkg.modular_sampling.chunked_v1_storage_nodes import (
    MiniMaxH3ChunkedV1SegmentLoadEXPT8, MiniMaxH3ChunkedV1SegmentSaveEXPT8,
)
from test_chunked_two_pass_global_noise_advanced import _CountingCoordinateNoise, _plan
from test_modular_chunked_source import _latent
from test_modular_chunked_stages import _fake_lift, _fake_spatial


def _three_segment_source():
    video = torch.arange(1 * 24 * 17 * 2 * 2, dtype=torch.float32).reshape(1, 24, 17, 2, 2)
    # Native 56-frame AV requires round(56/24*40)=93 audio latent tokens.
    audio = torch.arange(1 * 32 * 2 * 93, dtype=torch.float32).reshape(1, 32, 2, 93)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def _prepare(source):
    plan = _plan(old.build_chunked_two_pass_plan, strategy="full_frame_safe")
    noise = _CountingCoordinateNoise()
    context, _ = prepare_chunked_pass2(source, plan, noise)
    return plan, noise, context


def _segment(source, plan, noise, context, index, previous=None):
    chunk, spec, _ = slice_chunked_source(source, plan, index)
    lifted, _ = lift_chunked_segment(chunk, spec, context, plan)
    output, result, _ = sample_chunked_pass2(
        object(), [], chunk, lifted, spec, context, plan,
        noise, object(), torch.tensor([1.0, 0.0]), previous,
    )
    return chunk, spec, output, result


def test_frozen_v1_segment_reloads_with_current_full_audio_and_only_samples_later_segments(
        monkeypatch, tmp_path):
    monkeypatch.setattr(old, "learned_upscale_h3_av_latent", _fake_lift)
    monkeypatch.setattr(old, "_spatial_resample", _fake_spatial)
    source = _three_segment_source()
    plan, noise, context = _prepare(source)
    assert slice_chunked_source(source, plan, 0)[1].count >= 3
    segment0, spec0, _first, result0 = _segment(source, plan, noise, context, 0)
    _s1, _p1, expected1, result1 = _segment(source, plan, noise, context, 1, result0)
    _s2, _p2, expected2, _result2 = _segment(source, plan, noise, context, 2, result1)
    _output, _result, path, digest, manifest_json = save_segment(
        result0, segment0, spec0, context, plan, tmp_path,
    )
    manifest = json.loads(manifest_json)
    assert manifest["automatic_cache_reuse"] is False
    assert manifest["execution_identity_certified"] is False

    fresh_source = _three_segment_source()
    fresh_plan, fresh_noise, fresh_context = _prepare(fresh_source)
    assert fresh_context.original_audio is not context.original_audio
    fresh_segment0, fresh_spec0, _ = slice_chunked_source(fresh_source, fresh_plan, 0)
    loaded_output, loaded, report_json = load_segment(
        fresh_segment0, fresh_spec0, fresh_context, fresh_plan, tmp_path, path, digest,
    )
    assert loaded_output["samples"].tensors[1] is fresh_context.original_audio
    assert loaded.output_latent["samples"].tensors[1] is fresh_context.original_audio
    assert json.loads(report_json)["execution_identity_certified"] is False
    _s1, _p1, actual1, loaded1 = _segment(
        fresh_source, fresh_plan, fresh_noise, fresh_context, 1, loaded,
    )
    _s2, _p2, actual2, _loaded2 = _segment(
        fresh_source, fresh_plan, fresh_noise, fresh_context, 2, loaded1,
    )
    for got, want in ((actual1, expected1), (actual2, expected2)):
        for current, expected in zip(got["samples"].unbind(), want["samples"].unbind(), strict=True):
            assert torch.equal(current, expected)


def test_frozen_v1_segment_rejects_wrong_sha_path_source_index_and_tensor_change(monkeypatch, tmp_path):
    monkeypatch.setattr(old, "learned_upscale_h3_av_latent", _fake_lift)
    monkeypatch.setattr(old, "_spatial_resample", _fake_spatial)
    source = _latent()
    plan, noise, context = _prepare(source)
    segment, spec, _output, result = _segment(source, plan, noise, context, 0)
    _output, _result, path, digest, _manifest = save_segment(
        result, segment, spec, context, plan, tmp_path,
    )
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_segment(segment, spec, context, plan, tmp_path, path, "0" * 64)
    with pytest.raises(ValueError, match="traversal|escapes|relative"):
        load_segment(segment, spec, context, plan, tmp_path, "../manifest.json", digest)
    next_segment, next_spec, _ = slice_chunked_source(source, plan, 1)
    with pytest.raises(ValueError, match="does not match"):
        load_segment(next_segment, next_spec, context, plan, tmp_path, path, digest)
    altered = _latent()
    altered["samples"].tensors[0][0, 0, 0, 0, 0] += 1
    changed_plan, _changed_noise, changed_context = _prepare(altered)
    changed_segment, changed_spec, _ = slice_chunked_source(altered, changed_plan, 0)
    with pytest.raises(ValueError, match="does not match"):
        load_segment(changed_segment, changed_spec, changed_context, changed_plan,
                     tmp_path, path, digest)
    video = result.output_latent["samples"].tensors[0]
    video[0, 0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="content differs"):
        verify_segment(result, segment, spec, context, plan)
    bad = replace(result, output_identity={})
    with pytest.raises(ValueError, match="content differs"):
        save_segment(bad, segment, spec, context, plan, tmp_path)
    unexpected = replace(result, output_latent={**result.output_latent, "extra": "not portable"})
    with pytest.raises(ValueError, match="unsupported latent fields"):
        save_segment(unexpected, segment, spec, context, plan, tmp_path)
    state = tmp_path / path.replace("manifest.json", "state.safetensors")
    with state.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="size or SHA"):
        load_segment(segment, spec, context, plan, tmp_path, path, digest)


def test_v1_storage_public_nodes_default_to_no_write_and_require_explicit_sha(monkeypatch, tmp_path):
    monkeypatch.setattr(old, "learned_upscale_h3_av_latent", _fake_lift)
    monkeypatch.setattr(old, "_spatial_resample", _fake_spatial)
    import h3_audio_t8_pkg.modular_sampling.chunked_v1_storage_nodes as nodes
    monkeypatch.setattr(nodes, "_store_root", lambda: tmp_path / "private-store")
    source = _latent()
    plan, noise, context = _prepare(source)
    segment, spec, _output, result = _segment(source, plan, noise, context, 0)
    save_schema = MiniMaxH3ChunkedV1SegmentSaveEXPT8.define_schema()
    load_schema = MiniMaxH3ChunkedV1SegmentLoadEXPT8.define_schema()
    assert save_schema.node_id == "MiniMaxH3ChunkedV1SegmentSaveEXPT8"
    assert load_schema.node_id == "MiniMaxH3ChunkedV1SegmentLoadEXPT8"
    assert "confirm_save" in [item.id for item in save_schema.inputs]
    outcome = MiniMaxH3ChunkedV1SegmentSaveEXPT8.execute(
        result, segment, spec, context, plan,
    ).result
    assert outcome[2:4] == ("", "")
    assert not (tmp_path / "private-store").exists()
    saved = MiniMaxH3ChunkedV1SegmentSaveEXPT8.execute(
        result, segment, spec, context, plan, confirm_save=True,
    ).result
    assert saved[2] and len(saved[3]) == 64
    loaded = MiniMaxH3ChunkedV1SegmentLoadEXPT8.execute(
        segment, spec, context, plan, saved[2], saved[3],
    ).result
    assert loaded[1].output_identity == result.output_identity
