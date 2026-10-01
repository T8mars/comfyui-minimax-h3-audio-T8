"""Versioned, unaccepted V2 candidate writer for externally color-matched media.

The original raw-media writer and its sidecar implementation SHA stay intact.
"""

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

from .. import long_video_delivery as delivery
from .continuation import chain_guard
from .fast_h3_v2_color import FastH3V2ColoredMediaReceipt
from .fast_h3_v2_candidate import SIDECAR


SCHEMA = "t8.modular-sampling.fasth3-v2-colored-candidate-origin.v1"


def save_colored_candidate(color_receipt, candidate_id=""):
    if type(color_receipt) is not FastH3V2ColoredMediaReceipt:
        raise ValueError("Colored V2 candidate needs the typed external Color Match receipt")
    color = color_receipt.verify()
    source = color_receipt.source
    media = source.verify()
    binding = source.binding.verify()
    segment, parent = binding["segment_index"], binding["parent"]
    if (segment == 0) != (parent is None):
        raise ValueError("Colored V2 candidate parent/segment contract changed")
    root = delivery.long_video_chain_root(binding["chain_id"])
    with (nullcontext() if segment == 0 else chain_guard(root)):
        color_receipt.verify()
        parent_id = "" if parent is None else parent["parent_candidate_id"]
        parent_revision = 0 if parent is None else parent["parent_revision"]
        start_frame = 0 if parent is None else parent["timeline_start_frame"]
        path, video_path, _ = delivery.save_long_video_candidate(
            frames=color_receipt.frames, audio=color_receipt.audio,
            av_latent=source.high_result.output,
            chain_id=binding["chain_id"], segment_index=segment,
            timeline_start_seconds=start_frame / 24., save_context=segment == 0,
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
        if verified_video != video_path or any(descriptor.get(key) != value for key, value in expected.items()):
            raise ValueError("Written colored V2 candidate differs from typed current job")
        color_receipt.verify()
        candidate_path = Path(path).resolve()
        sidecar_path = candidate_path.parent / SIDECAR
        if sidecar_path.exists():
            raise FileExistsError("Colored V2 candidate provenance sidecar already exists")
        sidecar = {"schema": SCHEMA, "candidate_id": descriptor["candidate_id"],
                   "chain_id": binding["chain_id"], "segment_index": segment,
                   "candidate_json_sha256": delivery._sha256_file(candidate_path),
                   "video_sha256": descriptor["video_sha256"],
                   "context_sha256": descriptor["context_sha256"],
                   "job_binding_sha256": source.binding.sha256,
                   "job_binding": json.loads(source.binding.payload_json),
                   "media_receipt_sha256": source.sha256,
                   "media_receipt": json.loads(source.payload_json),
                   "color_receipt_sha256": color_receipt.sha256,
                   "color_receipt": json.loads(color_receipt.payload_json),
                   "high_stage_receipt_sha256": media["high_stage_receipt_sha256"],
                   "writer_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "accepted": False}
        delivery.atomic_write_long_video_json(sidecar_path, sidecar)
        report = {"schema": SCHEMA, "status": "colored_candidate_saved_unaccepted",
                  "candidate_json_path": path, "candidate_video_path": video_path,
                  "source_provenance_path": str(sidecar_path),
                  "current_job_sha256": binding["sampling_summary"],
                  "media_receipt_sha256": source.sha256,
                  "color_receipt_sha256": color_receipt.sha256,
                  "mode": color["mode"], "accepted": False,
                  "boundary": "Original candidate encoder, RGB-only external Color Match; separate explicit accept"}
        return path, video_path, str(sidecar_path), json.dumps(report, ensure_ascii=False, sort_keys=True)
