"""Write only an unaccepted V2 candidate from typed current media provenance.

This opt-in owner prevents free text job IDs or arbitrary frame/audio sockets
from being presented as a certified split run. Existing delivery owns the
actual MP4/context encoding, checksums, and candidate transaction unchanged.
"""

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

from .. import long_video_delivery as delivery
from .continuation import chain_guard
from .fast_h3_v2_media import FastH3V2MediaReceipt


SCHEMA = "t8.modular-sampling.fasth3-v2-candidate-origin.v1"
SIDECAR = "source-provenance.json"


def save_current_candidate(media_receipt, candidate_id=""):
    if type(media_receipt) is not FastH3V2MediaReceipt:
        raise ValueError("Current V2 candidate needs typed HIGH media provenance")
    media = media_receipt.verify()
    if media.get("portable_identity") is not True:
        raise ValueError("Current V2 candidate media identity is not portable")
    binding = media_receipt.binding.verify()
    segment = binding["segment_index"]
    parent = binding["parent"]
    if (segment == 0) != (parent is None):
        raise ValueError("Current V2 candidate parent/segment contract changed")
    root = delivery.long_video_chain_root(binding["chain_id"])
    lock = nullcontext() if segment == 0 else chain_guard(root)
    with lock:
        media_receipt.verify()
        parent_id = "" if parent is None else parent["parent_candidate_id"]
        parent_revision = 0 if parent is None else parent["parent_revision"]
        start_frame = 0 if parent is None else parent["timeline_start_frame"]
        path, video_path, _ = delivery.save_long_video_candidate(
            frames=media_receipt.frames, audio=media_receipt.audio,
            av_latent=media_receipt.high_result.output,
            chain_id=binding["chain_id"], segment_index=segment,
            timeline_start_seconds=start_frame / 24., save_context=(segment == 0),
            parent_candidate_id=parent_id, parent_manifest_revision=parent_revision,
            candidate_id=candidate_id, model_id=binding["model_id"],
            sampling_summary=binding["sampling_summary"], prompt="",
            seed=binding["seed"], fps=24, bit_depth=8, crf=18)
        descriptor, verified_video = delivery.load_long_video_candidate_descriptor(path)
        expected = {"chain_id": binding["chain_id"], "index": segment,
                    "parent_candidate_id": parent_id,
                    "parent_manifest_revision": parent_revision,
                    "timeline_start_frame": start_frame,
                    "frame_count": media["trim"]["frame_count"],
                    "model_id": binding["model_id"],
                    "sampling_summary": binding["sampling_summary"],
                    "seed": binding["seed"], "is_final_segment": segment == 1}
        if (verified_video != video_path
                or any(descriptor.get(name) != value for name, value in expected.items())):
            raise ValueError("Written V2 candidate differs from typed execution/media origin")
        media_receipt.verify()
        candidate_path = Path(path).resolve()
        sidecar_path = candidate_path.parent / SIDECAR
        if sidecar_path.exists():
            raise FileExistsError("Current V2 candidate provenance sidecar already exists")
        sidecar = {"schema": SCHEMA, "candidate_id": descriptor["candidate_id"],
                   "chain_id": binding["chain_id"], "segment_index": segment,
                   "candidate_json_sha256": delivery._sha256_file(candidate_path),
                   "video_sha256": descriptor["video_sha256"],
                   "context_sha256": descriptor["context_sha256"],
                   "job_binding_sha256": media_receipt.binding.sha256,
                   "job_binding": json.loads(media_receipt.binding.payload_json),
                   "media_receipt_sha256": media_receipt.sha256,
                   "media_receipt": json.loads(media_receipt.payload_json),
                   "high_stage_receipt_sha256": media["high_stage_receipt_sha256"],
                   "writer_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "accepted": False}
        delivery.atomic_write_long_video_json(sidecar_path, sidecar)
        report = {"schema": SCHEMA, "status": "candidate_saved_unaccepted",
                  "candidate_json_path": path, "candidate_video_path": video_path,
                  "source_provenance_path": str(sidecar_path),
                  "current_job_sha256": binding["sampling_summary"],
                  "media_receipt_sha256": media_receipt.sha256,
                  "accepted": False,
                  "boundary": "Existing candidate writer only; separate explicit review/accept and compose required"}
        return path, video_path, str(sidecar_path), json.dumps(report, ensure_ascii=False, sort_keys=True)
