"""Externally verified native-checkpoint provenance bridge for H16 resume."""
from __future__ import annotations

import json

from .chunked_v5_native_source import verified_native_source
from .results import canonical


def verified_h16_native_source(av_latent, resume_verified, checkpoint_id,
                               content_sha256, file_sha256, manifest_json, report_json):
    """Return original AV content only after exact external checkpoint review.

    The native Load's volatile provenance marker is not part of the frozen H16
    window's source. Reuse the strict content/receipt audit already exercised
    by v5, without changing its module or any existing stored-window identity.
    """
    source, bridge_report = verified_native_source(
        av_latent, resume_verified, checkpoint_id, content_sha256,
        file_sha256, manifest_json, report_json,
    )
    return source, canonical({
        "status": "verified_native_source_for_h16",
        "checkpoint_id": checkpoint_id,
        "content_sha256": content_sha256,
        "file_sha256": file_sha256,
        "underlying_content_audit": json.loads(bridge_report),
        "sampling_executed": False,
    })
