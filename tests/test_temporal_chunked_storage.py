"""Literal sidecar load/resume without weakening released v5 cache checks."""
from dataclasses import replace
import json

import pytest
import torch

from h3_audio_t8_pkg.modular_sampling.temporal_chunked_storage import (
    load_scoped_window, save_scoped_window, verify_scoped_window,
)
from h3_audio_t8_pkg.modular_sampling.results import canonical
from h3_audio_t8_pkg.modular_sampling.storage import file_sha
from test_temporal_dialogue_bank import run, scoped_harness as _bank_scoped_harness


scoped_harness = _bank_scoped_harness


def save(h, result, root):
    return save_scoped_window(result, h[1], h[8], h[9], h[5], h[7], root)


def load(h, root, path, digest, index=0):
    return load_scoped_window(h[1], h[8], h[9], h[5], h[7], root, path, digest, index)


def test_scoped_literal_window_cold_load_only_runs_remaining_window(scoped_harness, tmp_path):
    h = scoped_harness
    _, _, first, _ = run(h, 0)
    expected, _, _, _ = run(h, 1, first)
    _, _, _, path, digest, saved = save(h, first, tmp_path)
    assert json.loads(saved)["automatic_cache_reuse"] is False
    before = h[0]["sample"]
    _, base, scoped, report = load(h, tmp_path, path, digest)
    assert h[0]["sample"] == before and scoped.base is base
    actual, _, _, _ = run(h, 1, scoped)
    assert h[0]["sample"] == before + 1
    for left, right in zip(actual["samples"].unbind(), expected["samples"].unbind(), strict=True):
        assert torch.equal(left, right)
    assert not json.loads(report)["quality_accepted"]


def test_scoped_freeze_rejects_wrong_bank_or_audio_receipt(scoped_harness, tmp_path):
    h = scoped_harness
    _, _, first, _ = run(h, 0)
    for changed in (replace(first, bank_sha256="a" * 64),
                    replace(first, dialogue_receipt=replace(first.dialogue_receipt,
                                                           published_audio_sha256="b" * 64))):
        with pytest.raises(ValueError):
            save(h, changed, tmp_path)
    assert not list(tmp_path.iterdir())


def test_scoped_load_rejects_old_format_wrong_sha_or_wrong_index(scoped_harness, tmp_path):
    h = scoped_harness
    _, _, first, _ = run(h, 0)
    _, _, _, path, digest, saved = save(h, first, tmp_path)
    old = json.loads(saved)
    with pytest.raises(ValueError, match="scope.json"):
        load(h, tmp_path, old["base_artifact_path"], old["base_artifact_sha256"])
    with pytest.raises(ValueError, match="SHA"):
        load(h, tmp_path, path, "f" * 64)
    with pytest.raises(ValueError, match="ownership"):
        load(h, tmp_path, path, digest, 1)


def test_scoped_load_keeps_base_file_sha_and_source_pins(scoped_harness, tmp_path):
    h = scoped_harness
    _, _, first, _ = run(h, 0)
    _, _, _, path, digest, saved = save(h, first, tmp_path)
    manifest = json.loads(saved)
    base = tmp_path / manifest["base_artifact_path"]
    original = base.read_bytes()
    with base.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="SHA"):
        load(h, tmp_path, path, digest)
    # Same-byte restoration is a fixture action, never a production reader bypass.
    base.write_bytes(original)
    manifest["implementation"]["temporal_dialogue_scope.py"] = "f" * 64
    (tmp_path / path).write_text(canonical(manifest), encoding="utf8")
    with pytest.raises(ValueError, match="implementation"):
        load(h, tmp_path, path, file_sha(tmp_path / path))


def test_scoped_freeze_content_verification_does_not_certify_execution(scoped_harness):
    h = scoped_harness
    _, _, first, _ = run(h, 0)
    binding = verify_scoped_window(first, h[1], h[8], h[9], h[5], h[7])
    assert binding["bank_sha256"] == h[7].sha256
    first.base.output_latent["samples"].tensors[0][0, 0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="content"):
        verify_scoped_window(first, h[1], h[8], h[9], h[5], h[7])
