"""One native AV decode plus exact segment trim from a completed V2 HIGH.

The existing audio_ops functions own all numerical behavior. The receipt
retains the source and returned objects so a later candidate writer can reject
substituted frames/audio without decoding or sampling again.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import uuid

from .. import audio_ops, nodes_long_video_dual_model
from ..audio_ops import decode_av_latent, trim_av_output
from ..nodes_long_video_dual_model import _component_identity
from .fast_h3_v2_job_binding import FastH3V2CurrentJobBinding
from .results import StageResult, _input_identity, canonical


SCHEMA = "t8.modular-sampling.fasth3-v2-current-media.v1"


def implementation_sha256():
    sources = {"media_provenance": __file__, "decode_trim": audio_ops.__file__,
               "component_identity": nodes_long_video_dual_model.__file__}
    files = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
             for name, path in sources.items()}
    return hashlib.sha256(canonical(files).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FastH3V2MediaReceipt:
    payload_json: str
    sha256: str
    binding: FastH3V2CurrentJobBinding
    high_result: StageResult
    frames: object
    audio: object

    def verify(self):
        payload = json.loads(self.payload_json)
        if payload.get("schema") != SCHEMA or canonical(payload) != self.payload_json:
            raise ValueError("Unknown or noncanonical V2 media receipt")
        if hashlib.sha256(self.payload_json.encode("utf-8")).hexdigest() != self.sha256:
            raise ValueError("V2 media receipt fingerprint changed")
        if payload.get("implementation_sha256") != implementation_sha256():
            raise ValueError("V2 media preparation implementation changed")
        if type(self.binding) is not FastH3V2CurrentJobBinding or type(self.high_result) is not StageResult:
            raise ValueError("V2 media lost its typed current job or HIGH stage")
        binding, high = self.binding.verify(), self.high_result.verify()
        handoff = self.binding.handoff.verify()
        if (payload["current_job_binding_sha256"] != self.binding.sha256
                or payload["current_recipe_sha256"] != binding["current_recipe_sha256"]
                or payload["high_stage_receipt_sha256"] != high["receipt_sha256"]
                or payload["high_stage_receipt_sha256"] != handoff["high_stage_receipt_sha256"]
                or payload["high_output"] != high["outputs"]["output"]
                or payload["high_output"] != handoff["high_output"]
                or payload["frames"] != _input_identity(self.frames)
                or payload["audio"] != _input_identity(self.audio)):
            raise ValueError("V2 candidate media differs from completed current HIGH")
        return payload


def prepare_current_media(job_binding, high_result, video_vae, audio_vae,
                          start_seconds, duration_seconds, fps=24.0):
    if type(job_binding) is not FastH3V2CurrentJobBinding or type(high_result) is not StageResult:
        raise ValueError("Current V2 media needs typed job binding and completed HIGH stage")
    binding = job_binding.verify()
    high = high_result.verify()
    handoff = job_binding.handoff.verify()
    if (high["receipt_sha256"] != handoff["high_stage_receipt_sha256"]
            or high["outputs"]["output"] != handoff["high_output"]):
        raise ValueError("Media source is not the completed authenticated HIGH output")
    segment = binding["segment_index"]
    start_frame = 0 if segment == 0 else job_binding.recipe.verify()["window"]["continuation_context_frames"]
    frame_count = 124 if segment == 0 else 68
    if (not math.isclose(float(fps), 24.0, abs_tol=1e-9)
            or not math.isclose(float(start_seconds), start_frame / 24.0, abs_tol=1e-9)
            or not math.isclose(float(duration_seconds), frame_count / 24.0, abs_tol=1e-9)):
        raise ValueError("Current V2 media trim must be the exact native 124/68-frame window")
    try:
        components = {"video_vae": _component_identity(video_vae),
                      "audio_vae": _component_identity(audio_vae)}
        portable = all(value.get("portable_cache_reuse") is not False for value in components.values())
    except Exception as error:
        components = {"unverified": f"{type(error).__name__}: {error}",
                      "execution_nonce": uuid.uuid4().hex}
        portable = False
    # These are the unchanged public AV Decode and Output Trim implementations.
    decoded_frames, decoded_audio, _, _ = decode_av_latent(high_result.output, video_vae, audio_vae)
    frames, audio, trim_report = trim_av_output(
        decoded_frames, start_seconds, duration_seconds, decoded_audio, fps)
    try:
        inputs = {"high_output": _input_identity(high_result.output),
                  "decoded_frames": _input_identity(decoded_frames),
                  "decoded_audio": _input_identity(decoded_audio)}
        outputs = {"frames": _input_identity(frames), "audio": _input_identity(audio)}
    except Exception as error:
        inputs = {"unverified": f"{type(error).__name__}: {error}",
                  "execution_nonce": uuid.uuid4().hex}
        outputs = {"unverified": True}
        portable = False
    recipe_components = job_binding.recipe.verify()["components"]
    if components != {name: recipe_components[name] for name in ("video_vae", "audio_vae")}:
        portable = False
    if (frames.shape[0] != frame_count
            or frames.shape[1:3] != (job_binding.recipe.verify()["geometry"]["height"],
                                     job_binding.recipe.verify()["geometry"]["width"])):
        raise ValueError("Current V2 decoded media has wrong frame count or canvas")
    payload = {"schema": SCHEMA, "implementation_sha256": implementation_sha256(),
               "portable_identity": portable,
               "current_job_binding_sha256": job_binding.sha256,
               "current_recipe_sha256": binding["current_recipe_sha256"],
               "high_stage_receipt_sha256": high["receipt_sha256"],
               "high_output": high["outputs"]["output"],
               "components": components, "inputs": inputs,
               "trim": {"fps": 24, "start_frame": start_frame, "frame_count": frame_count,
                        "report_sha256": hashlib.sha256(trim_report.encode("utf-8")).hexdigest()},
               "frames": outputs.get("frames"), "audio": outputs.get("audio"),
               "boundary": "One native decode/trim only; no candidate save, acceptance or composition"}
    encoded = canonical(payload)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    receipt = FastH3V2MediaReceipt(encoded, digest, job_binding, high_result, frames, audio)
    if portable:
        receipt.verify()
    report = {"schema": SCHEMA, "status": "current_media_prepared" if portable else "execution_only_unverified",
              "segment_index": segment, "frames": frame_count, "media_receipt_sha256": digest,
              "boundary": payload["boundary"]}
    return frames, audio, receipt, json.dumps(report, ensure_ascii=False, sort_keys=True)
