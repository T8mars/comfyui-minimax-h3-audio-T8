from __future__ import annotations

import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import audit_skin_finish_blind_temporal as audit  # noqa: E402


def test_blind_temporal_audit_is_mapping_blind_pinned_and_non_automatic():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert audit.EXPECTED_REVIEW_ID == "b3aad4e0d57b"
    assert audit.EXPECTED_PUBLIC_SHA256 == (
        "E13D6C760326D32E9DBB7B409CA6BA39CDFE8E3698FB40631840CE5915C22A80"
    )
    assert '"private_key_accessed": False' in source
    assert '"automatic_selection": False' in source
    assert 'review / "media" / "A.mp4"' in source
    assert 'review / "media" / "B.mp4"' in source
    assert "custom review paths require --expected-review-id" in source
    assert "private_key.json" not in source
    assert "build_review" not in source
    assert "0.01" in source
