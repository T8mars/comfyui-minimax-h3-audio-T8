"""Opt-in, source-bound external Color Match for the current split V2 media.

The old dual-loop algorithm owns the RGB transformation. This wrapper does not
change audio, HIGH latent, or the pre-color decode/trim receipt.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from .. import long_video_color_match_advanced, long_video_delivery, long_video_dual_color
from .. import long_video_motion_color
from .fast_h3_v2_media import FastH3V2MediaReceipt
from .results import _input_identity, canonical


SCHEMA = "t8.modular-sampling.fasth3-v2-colored-media.v1"


def implementation_sha256():
    sources = (Path(__file__), Path(long_video_dual_color.__file__),
               Path(long_video_color_match_advanced.__file__), Path(long_video_motion_color.__file__))
    return hashlib.sha256(canonical([hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in sources]).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FastH3V2ColoredMediaReceipt:
    payload_json: str
    sha256: str
    source: FastH3V2MediaReceipt
    frames: torch.Tensor
    audio: object

    def verify(self):
        if type(self.source) is not FastH3V2MediaReceipt:
            raise ValueError("Color Match lost its typed decoded HIGH media")
        media = self.source.verify()
        binding = self.source.binding.verify()
        payload = json.loads(self.payload_json)
        if (payload.get("schema") != SCHEMA or canonical(payload) != self.payload_json
                or hashlib.sha256(self.payload_json.encode("utf-8")).hexdigest() != self.sha256
                or payload.get("implementation_sha256") != implementation_sha256()):
            raise ValueError("V2 Color Match receipt or implementation changed")
        if media.get("portable_identity") is not True:
            raise ValueError("V2 Color Match requires portable source media")
        segment, parent = binding["segment_index"], binding["parent"]
        if (segment not in (0, 1) or (segment == 0) != (parent is None)
                or payload.get("segment_index") != segment
                or payload.get("chain_id") != binding["chain_id"]
                or payload.get("source_media_receipt_sha256") != self.source.sha256
                or payload.get("source_frames") != media["frames"]
                or payload.get("parent_video_sha256") != (
                    None if parent is None else parent["accepted_video_sha256"])):
            raise ValueError("V2 Color Match source or accepted parent changed")
        if (self.audio is not self.source.audio
                or payload.get("audio") != media["audio"]
                or payload.get("frames") != _input_identity(self.frames)
                or not isinstance(self.frames, torch.Tensor)
                or self.frames.shape != self.source.frames.shape
                or not bool(torch.isfinite(self.frames).all())):
            raise ValueError("V2 Color Match changed audio or lost its RGB output")
        report = payload.get("color_report")
        if (type(report) is not dict
                or type(payload.get("enabled")) is not bool
                or payload.get("mode") not in long_video_dual_color.COLOR_MATCH_MODES
                or report.get("mode") != payload.get("mode")
                or report.get("enabled") != payload.get("enabled")
                or report.get("audio_touched") is not False
                or report.get("latent_touched") is not False):
            raise ValueError("V2 Color Match report no longer describes an RGB-only operation")
        if (segment == 0 or not payload["enabled"]):
            expected_status = "first_segment_identity" if segment == 0 and payload["enabled"] else "disabled"
            if (self.frames is not self.source.frames or payload["frames"] != media["frames"]
                    or report.get("status") != expected_status):
                raise ValueError("V2 Color Match identity branch changed decoded RGB")
        if segment == 1 and payload["enabled"]:
            if (report.get("predecessor_candidate_id") != parent["parent_candidate_id"]
                    or report.get("predecessor_video_sha256") != parent["accepted_video_sha256"]):
                raise ValueError("V2 Color Match used another accepted predecessor")
        return payload


def color_match_current_media(media_receipt, enabled=True, mode="bounded_motion_color_exp"):
    if type(media_receipt) is not FastH3V2MediaReceipt:
        raise ValueError("V2 external Color Match requires typed completed-HIGH media")
    if type(enabled) is not bool or mode not in long_video_dual_color.COLOR_MATCH_MODES:
        raise ValueError("V2 external Color Match settings are unsupported")
    media = media_receipt.verify()
    binding = media_receipt.binding.verify()
    if media.get("portable_identity") is not True:
        raise ValueError("V2 external Color Match requires portable source media")
    segment, parent = binding["segment_index"], binding["parent"]
    frames, color_report = long_video_dual_color.correct_dual_segment_color(
        media_receipt.frames, long_video_delivery.long_video_chain_root(binding["chain_id"]),
        binding["chain_id"], segment, "" if parent is None else parent["parent_candidate_id"],
        enabled, mode)
    media_receipt.verify()
    if (not isinstance(frames, torch.Tensor) or frames.shape != media_receipt.frames.shape
            or not bool(torch.isfinite(frames).all())):
        raise ValueError("V2 external Color Match returned invalid RGB frames")
    payload = {"schema": SCHEMA, "implementation_sha256": implementation_sha256(),
               "chain_id": binding["chain_id"], "segment_index": segment,
               "source_media_receipt_sha256": media_receipt.sha256,
               "source_frames": media["frames"], "frames": _input_identity(frames),
               "audio": media["audio"], "enabled": enabled, "mode": mode,
               "parent_video_sha256": None if parent is None else parent["accepted_video_sha256"],
               "color_report": color_report,
               "boundary": "Original accepted-video RGB Color Match only; no audio/latent/sample/edit/accept"}
    encoded = canonical(payload)
    receipt = FastH3V2ColoredMediaReceipt(encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                                          media_receipt, frames, media_receipt.audio)
    receipt.verify()
    report = {"schema": SCHEMA, "status": "colored_media_prepared",
              "segment_index": segment, "color_receipt_sha256": receipt.sha256,
              "mode": mode, "enabled": enabled, "color_report": color_report,
              "boundary": payload["boundary"]}
    return frames, media_receipt.audio, receipt, json.dumps(report, ensure_ascii=False, sort_keys=True)
