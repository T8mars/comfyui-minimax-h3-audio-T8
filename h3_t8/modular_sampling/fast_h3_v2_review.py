"""Durable, explicit review gate for a saved FastH3 V2 split candidate.

The local sidecar detects accidental substitution or stale artifacts. It is
not a cryptographic signature against an owner able to rewrite every file.
The existing delivery module remains the sole manifest/accept transaction.
"""

import hashlib
import json
from pathlib import Path

from .. import long_video_delivery as delivery
from . import fast_h3_v2_candidate as writer
from . import fast_h3_v2_color_candidate as color_writer
from .fast_h3_v2_color import SCHEMA as COLOR_SCHEMA
from .fast_h3_v2_color import implementation_sha256 as color_implementation
from ..long_video_dual_color import COLOR_MATCH_MODES
from .fast_h3_v2_job_binding import SCHEMA as BINDING_SCHEMA
from .fast_h3_v2_job_binding import implementation_sha256 as binding_implementation
from .fast_h3_v2_media import SCHEMA as MEDIA_SCHEMA
from .fast_h3_v2_media import implementation_sha256 as media_implementation
from .results import canonical


def _checked_payload(payload, expected_schema, expected_sha256, label):
    if not isinstance(payload, dict) or payload.get("schema") != expected_schema:
        raise ValueError(f"V2 {label} has an unknown schema")
    actual = hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"V2 {label} fingerprint differs from its saved receipt")
    return payload


def verify_saved_current_candidate(candidate_json_path):
    """Check the original candidate and all durable source links, read-only."""
    candidate, root, movie = delivery._load_candidate(str(candidate_json_path))
    path = delivery._resolve_inside(root, candidate_json_path)
    sidecar_path = path.parent / writer.SIDECAR
    if not sidecar_path.is_file() or not sidecar_path.resolve().is_relative_to(root.resolve()):
        raise ValueError("V2 candidate is missing its source provenance sidecar")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("V2 candidate source provenance is unreadable") from error
    if not isinstance(sidecar, dict) or sidecar.get("schema") not in (writer.SCHEMA, color_writer.SCHEMA):
        raise ValueError("V2 candidate source provenance schema is unsupported")
    if sidecar.get("accepted") is not False:
        raise ValueError("V2 candidate sidecar must record an initially unaccepted write")
    colored = sidecar["schema"] == color_writer.SCHEMA
    expected_sidecar = {
        "candidate_id": candidate["candidate_id"],
        "chain_id": candidate["chain_id"],
        "segment_index": candidate["index"],
        "candidate_json_sha256": delivery._sha256_file(path),
        "video_sha256": candidate["video_sha256"],
        "context_sha256": candidate["context_sha256"],
        "writer_implementation_sha256": hashlib.sha256(Path(
            color_writer.__file__ if colored else writer.__file__).read_bytes()).hexdigest(),
    }
    if any(sidecar.get(key) != value for key, value in expected_sidecar.items()):
        raise ValueError("V2 candidate/source sidecar identity or implementation changed")
    binding = _checked_payload(sidecar.get("job_binding"), BINDING_SCHEMA,
                               sidecar.get("job_binding_sha256"), "job binding")
    media = _checked_payload(sidecar.get("media_receipt"), MEDIA_SCHEMA,
                             sidecar.get("media_receipt_sha256"), "media")
    if colored:
        color = _checked_payload(sidecar.get("color_receipt"), COLOR_SCHEMA,
                                 sidecar.get("color_receipt_sha256"), "external Color Match")
        color_report = color.get("color_report")
        if (color.get("implementation_sha256") != color_implementation()
                or color.get("source_media_receipt_sha256") != sidecar["media_receipt_sha256"]
                or color.get("source_frames") != media.get("frames")
                or color.get("audio") != media.get("audio")
                or color.get("chain_id") != candidate["chain_id"]
                or color.get("segment_index") != candidate["index"]
                or type(color.get("enabled")) is not bool
                or color.get("mode") not in COLOR_MATCH_MODES
                or type(color_report) is not dict
                or color_report.get("enabled") != color["enabled"]
                or color_report.get("mode") != color["mode"]
                or color_report.get("audio_touched") is not False
                or color_report.get("latent_touched") is not False):
            raise ValueError("V2 candidate external Color Match origin changed")
        expected_parent_sha = None if binding.get("parent") is None else binding["parent"].get("accepted_video_sha256")
        if color.get("parent_video_sha256") != expected_parent_sha:
            raise ValueError("V2 candidate Color Match predecessor differs from current job")
        if candidate["index"] == 1 and color["enabled"]:
            if (color_report.get("predecessor_candidate_id") != binding["parent"].get("parent_candidate_id")
                    or color_report.get("predecessor_video_sha256") != expected_parent_sha):
                raise ValueError("V2 candidate Color Match used another accepted predecessor")
        if candidate["index"] == 0 or not color["enabled"]:
            expected_status = ("first_segment_identity" if candidate["index"] == 0 and color["enabled"]
                               else "disabled")
            if color.get("frames") != media.get("frames") or color_report.get("status") != expected_status:
                raise ValueError("V2 candidate Color Match identity branch changed decoded RGB")
    if (binding.get("implementation_sha256") != binding_implementation()
            or media.get("implementation_sha256") != media_implementation()
            or media.get("portable_identity") is not True):
        raise ValueError("V2 candidate current job/media source is stale or nonportable")
    segment = candidate["index"]
    parent = binding.get("parent")
    if segment not in (0, 1) or (segment == 0) != (parent is None):
        raise ValueError("V2 candidate parent/segment source is invalid")
    expected_parent_id = "" if parent is None else parent.get("parent_candidate_id")
    expected_revision = 0 if parent is None else parent.get("parent_revision")
    expected_start = 0 if parent is None else parent.get("timeline_start_frame")
    expected_frames = 124 if segment == 0 else 68
    expected_trim_start = 0 if segment == 0 else 22
    trim = media.get("trim")
    if not isinstance(trim, dict):
        raise ValueError("V2 candidate media trim receipt is missing")
    expected = {
        "chain_id": binding.get("chain_id"),
        "parent_candidate_id": expected_parent_id,
        "parent_manifest_revision": expected_revision,
        "timeline_start_frame": expected_start,
        "timeline_end_frame": 124 if segment == 0 else 192,
        "frame_count": expected_frames,
        "model_id": binding.get("model_id"),
        "sampling_summary": binding.get("sampling_summary"),
        "seed": binding.get("seed"),
        "is_final_segment": segment == 1,
    }
    if any(candidate.get(key) != value for key, value in expected.items()):
        raise ValueError("V2 candidate fields differ from the saved current job")
    if ((segment == 0) != bool(candidate["context_sha256"])
            or binding.get("segment_index") != segment
            or binding.get("current_recipe_sha256") != candidate["sampling_summary"]
            or media.get("current_recipe_sha256") != candidate["sampling_summary"]
            or media.get("current_job_binding_sha256") != sidecar["job_binding_sha256"]
            or media.get("high_stage_receipt_sha256") != sidecar.get("high_stage_receipt_sha256")
            or trim.get("fps") != 24
            or trim.get("start_frame") != expected_trim_start
            or trim.get("frame_count") != expected_frames):
        raise ValueError("V2 candidate media/current job linkage changed")
    return candidate, str(movie), str(sidecar_path)


def review_accept_current_candidate(candidate_json_path, accept_candidate=False):
    if type(accept_candidate) is not bool:
        raise ValueError("V2 candidate acceptance must be an explicit boolean")
    candidate, _movie, sidecar = verify_saved_current_candidate(candidate_json_path)
    movie, accepted, manifest, old_report = delivery.accept_long_video_candidate(
        str(candidate_json_path), accept_candidate, "reject_existing", True)
    transaction = json.loads(old_report)
    if accepted:
        if (transaction.get("candidate_id") != candidate["candidate_id"]
                or transaction.get("index") != candidate["index"]
                or transaction.get("accepted") is not True):
            raise ValueError("V2 accepted manifest report differs from reviewed candidate")
        parent_id = candidate["candidate_id"]
        revision = transaction["manifest_revision"]
        job_sha256 = candidate["sampling_summary"]
    else:
        parent_id, revision, job_sha256 = "", 0, ""
    report = {"schema": writer.SCHEMA,
              "status": "accepted" if accepted else "verified_preview_only",
              "accepted": accepted, "chain_id": candidate["chain_id"],
              "candidate_id": candidate["candidate_id"],
              "candidate_json_path": str(Path(candidate_json_path).resolve()),
              "source_provenance_path": sidecar,
              "manifest_path": manifest,
              "manifest_revision": revision,
              "current_job_sha256": job_sha256,
              "boundary": "Explicit local-integrity review, unchanged reject-existing manifest transaction; no compose"}
    return (movie, accepted, manifest, candidate["chain_id"], parent_id,
            revision, job_sha256, json.dumps(report, ensure_ascii=False, sort_keys=True))


def verify_current_accepted_chain(chain_id):
    """Read-only two-segment source gate before the unchanged composer runs."""
    manifest, source = delivery.load_delivery_manifest(chain_id)
    segments = manifest["segments"]
    if len(segments) != 2 or segments[0]["is_final_segment"] or not segments[1]["is_final_segment"]:
        raise ValueError("Current V2 compose requires exactly two accepted 124+68 segments")
    root = delivery.long_video_chain_root(chain_id)
    delivery._verify_accepted_files(manifest, root)
    original_candidates = []
    for index, accepted in enumerate(segments):
        path = delivery._resolve_inside(root, root / "candidates" /
            f"segment_{index:05d}" / accepted["candidate_id"] / "candidate.json")
        candidate, _movie, sidecar = verify_saved_current_candidate(path)
        fields = ("index", "candidate_id", "parent_candidate_id",
                  "frame_count", "timeline_start_frame", "timeline_end_frame",
                  "video_sha256", "context_sha256", "model_id", "sampling_summary")
        if (candidate["chain_id"] != chain_id
                or any(candidate.get(key) != accepted.get(key) for key in fields)):
            raise ValueError("Accepted V2 segment differs from its source-bound candidate")
        original_candidates.append({"candidate_id": candidate["candidate_id"],
            "candidate_json_sha256": delivery._sha256_file(path),
            "source_provenance_path": sidecar})
    if (segments[0]["frame_count"] != 124 or segments[1]["frame_count"] != 68
            or segments[0]["timeline_start_frame"] != 0
            or segments[0]["timeline_end_frame"] != 124
            or segments[1]["timeline_start_frame"] != 124
            or segments[1]["timeline_end_frame"] != 192
            or segments[1]["parent_candidate_id"] != segments[0]["candidate_id"]
            or segments[1]["sampling_summary"] != segments[0]["sampling_summary"]):
        raise ValueError("Accepted V2 chain is not one current 124+68-frame job")
    report = {"schema": writer.SCHEMA, "status": "accepted_chain_source_verified",
              "chain_id": chain_id, "manifest_revision": manifest["revision"],
              "manifest_source": source, "current_job_sha256": segments[0]["sampling_summary"],
              "frame_count": 192, "candidates": original_candidates,
              "boundary": "Read-only local source gate; original composer and human quality review remain separate"}
    return chain_id, json.dumps(report, ensure_ascii=False, sort_keys=True)
