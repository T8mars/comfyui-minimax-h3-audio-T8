"""Explicit NativeCheckpointLoad-to-v5 source bridge, outside window identity."""
from __future__ import annotations

import json

from ..native_latent_checkpoint_advanced import CHECKPOINT_SCHEMA, _VOLATILE_CHECKPOINT_KEY
from ..native_latent_timeline_advanced import audit_native_h3_av_latent_resume_manifest
from .results import canonical


def verified_native_source(av_latent, resume_verified, checkpoint_id,
                           content_sha256, file_sha256, manifest_json, report_json):
    """Remove only the verified Load provenance before v5 identity binding."""
    if type(av_latent) is not dict or set(av_latent) != {
            "samples", _VOLATILE_CHECKPOINT_KEY,
            *(key for key in ("noise_mask", "batch_index", "type") if key in av_latent)}:
        raise ValueError("v5 native source has missing or unsupported metadata")
    marker = av_latent[_VOLATILE_CHECKPOINT_KEY]
    try:
        report = json.loads(report_json)
    except (TypeError, ValueError) as exc:
        raise ValueError("v5 native source has invalid Load report JSON") from exc
    if (type(marker) is not dict or report != marker or
            marker.get("schema") != CHECKPOINT_SCHEMA or
            marker.get("status") != "MATCH_EXTERNAL" or
            marker.get("resume_verified") is not True or resume_verified is not True or
            marker.get("external_manifest_verified") is not True or
            marker.get("expected_file_sha256_supplied") is not True or
            marker.get("checkpoint_id") != checkpoint_id or
            marker.get("content_sha256") != content_sha256 or
            marker.get("file_sha256") != file_sha256 or
            marker.get("sampling_executed") is not False):
        raise ValueError("v5 native source is not an exact externally verified checkpoint")
    source = {key: value for key, value in av_latent.items()
              if key != _VOLATILE_CHECKPOINT_KEY}
    _status, verified, actual_sha, _manifest = audit_native_h3_av_latent_resume_manifest(
        source, checkpoint_id=checkpoint_id, expected_manifest_json=manifest_json,
        mismatch_policy="error",
    )
    if not verified or actual_sha != content_sha256:
        raise ValueError("v5 native source content differs from the external manifest")
    return source, canonical({"status": "verified_native_source_for_v5",
                              "checkpoint_id": checkpoint_id,
                              "content_sha256": content_sha256,
                              "file_sha256": file_sha256,
                              "provenance_removed": _VOLATILE_CHECKPOINT_KEY,
                              "sampling_executed": False})
