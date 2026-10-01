"""The V2-only LOW witness can exceed generic Stage manifest JSON limits."""

from contextlib import nullcontext
import hashlib
import json
from types import SimpleNamespace

import pytest

from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_frozen_low as frozen


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def test_large_bundle_uses_bounded_compressed_envelope(monkeypatch, tmp_path):
    directory = tmp_path / "selected-low"
    directory.mkdir()
    manifest = directory / "manifest.json"
    manifest.write_bytes(b"selected manifest")
    state = directory / "state.safetensors"
    state.write_bytes(b"selected tensors")
    manifest_sha = _sha(manifest.read_bytes())

    class DummyResult:
        def verify(self):
            return {"receipt_sha256": "a" * 64}

    recipe_value = {"model_pass1": {"sha256": "1" * 64},
                    "model_pass2": {"sha256": "2" * 64}, "large": "X" * 10000}
    stage_value = DummyResult().verify()
    attestation_value = {"stage": "low_0_4"}
    condition_value = {"portable_identity": True}
    monkeypatch.setattr(frozen, "StageResult", DummyResult)
    monkeypatch.setattr(frozen, "_witnesses", lambda *_args: (
        recipe_value, stage_value, attestation_value, condition_value))
    monkeypatch.setattr(frozen, "_artifact", lambda *_args: (
        tmp_path, manifest, manifest_sha, DummyResult()))
    monkeypatch.setattr(frozen.storage, "fingerprint_stage", lambda *_args: (
        manifest_sha, _sha(state.read_bytes())))
    monkeypatch.setattr(frozen.storage, "_lease", lambda *_args: nullcontext())
    monkeypatch.setattr(frozen, "implementation_sha256", lambda: "3" * 64)
    monkeypatch.setattr(frozen, "MAX_INLINE_JSON", 1024)
    recipe = SimpleNamespace(sha256="4" * 64)
    attestation = SimpleNamespace(sha256="5" * 64)
    condition = SimpleNamespace(sha256="6" * 64)
    relative, digest, _ = frozen.save_current_low_bundle(
        recipe, DummyResult(), attestation, condition, "selected-low/manifest.json",
        manifest_sha, tmp_path)
    sidecar = tmp_path / relative
    envelope = json.loads(sidecar.read_text(encoding="utf-8"))
    assert envelope["schema"] == frozen.ENVELOPE_SCHEMA
    assert envelope["raw_bytes"] > frozen.MAX_INLINE_JSON
    assert envelope["compressed_bytes"] < envelope["raw_bytes"]
    assert digest == _sha(sidecar.read_bytes())
    loaded = frozen.load_current_low_bundle(
        "selected-low/manifest.json", manifest_sha, DummyResult(), tmp_path)
    assert loaded[0].sha256 == recipe.sha256
    assert loaded[4].sha256 == attestation.sha256
    assert loaded[5].sha256 == condition.sha256
    assert loaded[6] == digest

    blob = sidecar.with_name(frozen.BLOB)
    original = blob.read_bytes()
    blob.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="compressed blob changed"):
        frozen.load_current_low_bundle(
            "selected-low/manifest.json", manifest_sha, DummyResult(), tmp_path)
