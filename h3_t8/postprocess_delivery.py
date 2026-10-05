"""Full-length pixel postprocessing against a durable, content-bound H3 master.

Reuses the established isolated encoder, media inspector, original packet mux
and no-replace publisher. No sampling, audio decoding/reencoding for delivery,
retiming, source overwrite or automatic perceptual acceptance.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

import torch

from . import long_video_delivery as delivery
from .director_project import atomic_json
from .skin_finish_p1 import _file_video_source_path
from .video_outpaint_delivery import delivery_report_path, publish_with_delivery_report, read_delivery_report
from .video_outpaint_media import inspect_outpaint_source, _packet_copy, _validate_final_file
from .video_outpaint_plan import canonical


STATE_SCHEMA = "t8.h3.postprocess.job/v1"
FINAL_SCHEMA = "t8.h3.postprocess.final_file/v1"


def state_path(output):
    return Path(output).with_suffix(".postprocess-state.json")


def _seal(data):
    result = {key: value for key, value in data.items() if key != "state_sha256"}
    return {**result, "state_sha256": hashlib.sha256(canonical(result).encode()).hexdigest()}


def read_postprocess_state(path, *, verify_media=True):
    """A running/orphan state is not success; hashes are bindings, not signatures."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 8 * 1024**2:
        raise ValueError("invalid postprocess state file")
    path = path.resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf8"))
    if (type(value) is not dict or value.get("schema") != STATE_SCHEMA
            or value.get("state_sha256") != _seal(value)["state_sha256"]
            or state_path(value.get("output_path", "")).resolve() != path
            or value.get("state") not in {"running", "postprocess_complete", "postprocess_failed", "postprocess_cancelled"}):
        raise ValueError("postprocess state integrity/path mismatch")
    if verify_media:
        master = Path(value["master"]["path"]).resolve(strict=True)
        if delivery._sha256_file(master) != value["master"]["sha256"]:
            raise ValueError("master has changed; recovery must not use another file")
        if value["state"] == "postprocess_complete":
            result = read_delivery_report(value["output_path"], report_kind="postprocess")
            if (result.get("job_id") != value["id"] or result.get("sha256") != value.get("output_sha256")
                    or result.get("source_sha256") != value["master"]["sha256"]):
                raise ValueError("postprocess state and delivered media do not match")
    return value


def _save_state(path, data, *, initial=False):
    value = _seal(data)
    if initial:
        # Exclusive reservation: never overwrite another job, including a crash
        # receipt without a video. Later updates belong only to this same job.
        with Path(path).open("x", encoding="utf8", newline="\n") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
    else:
        prior = read_postprocess_state(path, verify_media=False)
        if (prior["id"], prior["master"]["sha256"], prior["output_path"]) != (
                value["id"], value["master"]["sha256"], value["output_path"]):
            raise ValueError("refusing to overwrite another postprocess job")
        atomic_json(path, value)
    return value


def _master_available(path, expected):
    try:
        return path.is_file() and delivery._sha256_file(path) == expected
    except OSError:
        return False


def finalize_postprocess(master_video, processed_frames, output_path, *, crf=18,
                         expected_master_sha256="", interrupt_check=None):
    """Return a durable terminal state. Failed candidates never replace master.

    IMAGE is explicitly caller-provided full-length postprocessed RGB, not a
    claim that arbitrary pixels originated from this master. The supplied
    templates read this very saved master before Core crop/pad/resize.
    """
    target = Path(output_path).resolve()
    source_path = _file_video_source_path(master_video)
    sidecar = delivery_report_path(target, report_kind="postprocess")
    job_path = state_path(target)
    if (target.suffix.lower() != ".mp4" or target == source_path
            or any(path.exists() for path in (target, sidecar, job_path))):
        raise FileExistsError("postprocess needs a new MP4 and receipt path, never the master")
    if interrupt_check is None:
        from comfy.model_management import throw_exception_if_processing_interrupted
        interrupt_check = throw_exception_if_processing_interrupted
    interrupt_check()
    master = inspect_outpaint_source(master_video, interrupt_check=interrupt_check)
    if expected_master_sha256:
        if (type(expected_master_sha256) is not str or len(expected_master_sha256) != 64
                or any(char not in "0123456789abcdefABCDEF" for char in expected_master_sha256)
                or master["sha256"] != expected_master_sha256.lower()):
            raise ValueError("selected master differs from expected SHA256")
    # The existing isolated raw encoder starts at zero. Do not silently retime
    # nonzero external media or pass a VFR clip through a fixed24fps output.
    cfr = master["cfr"]
    if Fraction(cfr["first_pts"]) * Fraction(cfr["time_base"]) != 0:
        raise ValueError("postprocess requires a zero-origin24fps master; no implicit PTS reset")
    target.parent.mkdir(parents=True, exist_ok=True)
    job = {"schema": STATE_SCHEMA, "id": str(uuid.uuid4()), "state": "running",
           "stage": "validate_pixels", "master": master, "master_available": True,
           "output_path": str(target), "state_path": str(job_path), "created_at": time.time(),
           "audio_regenerated": False, "automatic_accept": False, "perceptual_acceptance": False}
    _save_state(job_path, job, initial=True)
    published = None
    try:
        if (not isinstance(processed_frames, torch.Tensor) or processed_frames.ndim != 4
                or processed_frames.shape[3] not in (3, 4)):
            raise ValueError("processed_frames must be full-length IMAGE [frames,height,width,RGB/RGBA]")
        count, height, width, _ = processed_frames.shape
        if (count != master["frames"] or min(count, height, width) <= 0 or width % 2 or height % 2):
            raise ValueError("postprocess cannot trim time; use the same frame count and even nonempty dimensions")
        if type(crf) is not int or not 0 <= crf <= 51:
            raise ValueError("crf must be an integer from0 to51")
        rgb_hash = hashlib.sha256()

        def chunks():
            for frame in processed_frames:
                interrupt_check()
                rgb = frame[..., :3].detach().float().cpu()
                if not torch.isfinite(rgb).all():
                    raise ValueError("nonfinite postprocessed IMAGE")
                raw = (rgb.clamp(0, 1) * 255).round().to(torch.uint8).contiguous().numpy().tobytes()
                rgb_hash.update(raw)
                yield raw

        job["stage"] = "encode_video_only"
        _save_state(job_path, job)
        with tempfile.TemporaryDirectory(prefix=".h3-postprocess-", dir=target.parent) as directory:
            work = Path(directory)
            video_only, muxed = work / "video.mp4", work / "muxed.mp4"
            delivery._encode_rgb_frames_isolated(video_only, chunks, frame_count=count,
                width=width, height=height, fps=24, bit_depth=8, crf=crf)
            delivery._strict_validate_mp4(video_only, require_audio=False)
            job["stage"] = "copy_master_audio_packets"
            _save_state(job_path, job)
            mux = _packet_copy(video_only, source_path, muxed, interrupt_check=interrupt_check)
            job["stage"] = "verify_video_clock_audio_packets_pcm"
            _save_state(job_path, job)
            evidence = _validate_final_file(muxed, frame_count=count, width=width, height=height,
                rate=Fraction(24), source_audio_packets=master["audio_packets"],
                source_audio_pcm=master["audio_pcm"], interrupt_check=interrupt_check)
            final_clock = evidence["cfr"]
            if Fraction(final_clock["first_pts"]) * Fraction(final_clock["time_base"]) != 0:
                raise RuntimeError("postprocess shifted the video origin")
            if delivery._sha256_file(source_path) != master["sha256"]:
                raise ValueError("master changed during postprocess; not publishing")
            interrupt_check()
            with muxed.open("r+b") as stream:
                os.fsync(stream.fileno())
            report = {"schema": FINAL_SCHEMA, "status": "postprocess_complete", "job_id": job["id"],
                      "path": str(target), "sha256": delivery._sha256_file(muxed),
                      "source_path": str(source_path), "source_sha256": master["sha256"],
                      "master": master, "media": evidence, "source_rgb8_sha256": rgb_hash.hexdigest(),
                      "encoder_policy": delivery.ISOLATED_VIDEO_ENCODER_POLICY, "crf": crf,
                      "packet_mux": mux, "audio_regenerated": False, "automatic_accept": False,
                      "perceptual_acceptance": False, "pixel_provenance": "caller_declared_full_length_postprocessed_IMAGE",
                      "atomic_no_replace_publish": True, "source_overwritten": False}
            published = publish_with_delivery_report(muxed, target, report,
                interrupt_check=interrupt_check, report_kind="postprocess")
    except BaseException as error:
        job.update(state="postprocess_failed" if isinstance(error, Exception) else "postprocess_cancelled",
                   error=str(error)[:4000], error_type=type(error).__name__, finished_at=time.time(),
                   master_available=_master_available(source_path, master["sha256"]),
                   output_published=False)
        _save_state(job_path, job)
        if not isinstance(error, Exception):
            raise
        return read_postprocess_state(job_path, verify_media=False)
    # A state-write error after publication is propagated, never relabelled as
    # failed encoding. The verified delivery sidecar remains authoritative.
    job.update(state="postprocess_complete", stage="published", output_sha256=published["sha256"],
               output_published=True, finished_at=time.time(), delivery_report_path=published["delivery_report_path"])
    _save_state(job_path, job)
    return read_postprocess_state(job_path)
