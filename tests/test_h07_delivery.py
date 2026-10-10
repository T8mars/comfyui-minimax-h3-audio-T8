import hashlib
import json

import pytest

from h3_audio_t8_pkg.h07_delivery import delivery_documents


def records(tmp_path):
    movie, workflow = tmp_path / "private-person-secret.mp4", tmp_path / "private-token.json"
    # Tiny bytes only exercise binding/privacy. Not real media/decode/GPU evidence.
    movie.write_bytes(b"fixture-not-a-playable-movie")
    workflow.write_text('{"path":"F:/PRIVATE/cache","token":"hf_secret","prompt":"PRIVATE"}', encoding="utf8")
    return dict(video_path=movie, video_sha256=hashlib.sha256(movie.read_bytes()).hexdigest(),
        workflow_path=workflow, workflow_sha256=hashlib.sha256(workflow.read_bytes()).hexdigest(),
        project_id="P", shot_id="S", take_id="T", recipe_id="R")


def test_public_allowlist_never_copies_workflow_or_filename_secrets(tmp_path):
    args = records(tmp_path)
    sources = [dict(role_id="A", origin="original", path=str(args["video_path"]), sha256=args["video_sha256"])]
    private, public = delivery_documents(**args, audio_sources_json=json.dumps(sources))
    encoded = json.dumps(public)
    assert "private-person" not in encoded and "hf_secret" not in encoded and "PRIVATE" not in encoded
    assert "path" not in public["video"] and "path" in private["video"]
    assert public["audio_sources"][0]["origin"] == "original" and public["producer"]["actual_NFE"] == "unknown"
    assert not public["external_upload_started"] and not public["cache_reuse_authorized"]


def test_changed_take_wrong_origin_and_two_producers_do_not_get_bound(tmp_path):
    args = records(tmp_path)
    with pytest.raises(ValueError, match="SHA"):
        delivery_documents(**{**args, "video_sha256": "0"*64})
    with pytest.raises(ValueError, match="origin"):
        delivery_documents(**args, audio_sources_json='[{"origin":"TTS-replacement"}]')
    with pytest.raises(ValueError, match="one actual"):
        delivery_documents(**args, completed_quality_stage={}, completed_modular_stage={})
