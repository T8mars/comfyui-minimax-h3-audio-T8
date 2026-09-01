from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
import uuid

import numpy as np
import torch

import folder_paths
from safetensors import safe_open
from safetensors.torch import save_file

from .core import AUDIO_LATENT_FPS, FPS, nested_av_parts, validate_audio
from .long_video import (
    CONTEXT_FRAME_STEPS,
    FRAME_RESCALE,
    LONG_VIDEO_SCHEMA,
    STATE_FOLDER,
    pixel_frames_from_latent_t,
    sanitize_chain_id,
)


DELIVERY_SCHEMA = 1
MANIFEST_SCHEMA = 2
LEGACY_MANIFEST_SCHEMA = 1
MANIFEST_FORMAT = "minimax_h3_t8_accepted_manifest"
MANIFEST_NAME = "manifest.json"
MANIFEST_BACKUP_NAME = "manifest.json.bak"
LOCK_NAME = "manifest.lock"
ADVISORY_LOCK_NAME = "manifest.lock.v2"
ADVISORY_LOCK_KIND = "t8_os_advisory_v2"
_RETRYABLE_FFMPEG_NATIVE_EXIT_CODES = {
    0xC0000005,  # Windows access violation.
    0xC0000093,  # Windows floating-point underflow observed in FFmpeg/libavcodec.
}
STRICT_AV_DECODE_POLICY = "ffmpeg_xerror_threads1_before_atomic_publish_v1"
ISOLATED_VIDEO_ENCODER_POLICY = (
    "ffmpeg_rawvideo_pipe_libx264_all_intra_baseline_threads1_subprocess_v3"
)


class UnsupportedManifestSchemaError(ValueError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _interruption_check() -> None:
    try:
        import comfy.model_management

        comfy.model_management.throw_exception_if_processing_interrupted()
    except ImportError:
        return


def _cleanup_temporary(path: Path, active_error: BaseException | None = None) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as cleanup_error:
        message = f"temporary file cleanup failed for {path}: {cleanup_error}"
        if active_error is not None:
            add_note = getattr(active_error, "add_note", None)
            if callable(add_note):
                add_note(message)
            return
        raise


def _run_isolated_ffmpeg(
    args: list[str], log_path: Path, operation: str = "AAC mux"
) -> None:
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    for attempt in range(2):
        with log_path.open("ab" if attempt else "wb") as log:
            if attempt:
                log.write(b"\n--- retry after transient native FFmpeg crash ---\n")
                log.flush()
            process = subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
            )
            try:
                while process.poll() is None:
                    _interruption_check()
                    time.sleep(0.1)
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                raise
        returncode = int(process.returncode or 0)
        if returncode == 0:
            return
        normalized_returncode = returncode & 0xFFFFFFFF
        if attempt == 0 and normalized_returncode in _RETRYABLE_FFMPEG_NATIVE_EXIT_CODES:
            _interruption_check()
            continue
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(
            f"FFmpeg {operation} failed with exit code "
            f"{returncode} after {attempt + 1} attempt(s):\n{tail}"
        )


def _run_isolated_ffmpeg_with_input(
    args: list[str],
    log_path: Path,
    chunks_factory,
    *,
    expected_chunks: int,
    expected_chunk_bytes: int,
    operation: str,
) -> None:
    """Run FFmpeg with bounded raw input while keeping native encoder faults isolated."""
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    for attempt in range(2):
        pipe_error: BrokenPipeError | None = None
        with log_path.open("ab" if attempt else "wb") as log:
            if attempt:
                log.write(b"\n--- retry after transient native FFmpeg crash ---\n")
                log.flush()
            process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
            )
            try:
                if process.stdin is None:
                    raise RuntimeError("FFmpeg raw-video stdin pipe was not created")
                written_chunks = 0
                try:
                    for chunk in chunks_factory():
                        _interruption_check()
                        payload = memoryview(chunk)
                        if payload.nbytes != int(expected_chunk_bytes):
                            raise ValueError(
                                "raw-video frame byte size does not match the declared geometry: "
                                f"{payload.nbytes} != {expected_chunk_bytes}"
                            )
                        written = process.stdin.write(payload)
                        if written != payload.nbytes:
                            raise BrokenPipeError(
                                f"FFmpeg raw-video pipe accepted {written} of {payload.nbytes} bytes"
                            )
                        written_chunks += 1
                except BrokenPipeError as error:
                    pipe_error = error
                finally:
                    try:
                        process.stdin.close()
                    except BrokenPipeError as error:
                        pipe_error = pipe_error or error
                if pipe_error is None and written_chunks != int(expected_chunks):
                    raise ValueError(
                        f"raw-video source yielded {written_chunks} frames; "
                        f"expected {expected_chunks}"
                    )
                while process.poll() is None:
                    _interruption_check()
                    time.sleep(0.1)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                raise
        returncode = int(process.returncode or 0)
        if returncode == 0 and pipe_error is None:
            return
        normalized_returncode = returncode & 0xFFFFFFFF
        if attempt == 0 and normalized_returncode in _RETRYABLE_FFMPEG_NATIVE_EXIT_CODES:
            _interruption_check()
            continue
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        pipe_detail = f"\nraw-video pipe error: {pipe_error}" if pipe_error else ""
        raise RuntimeError(
            f"FFmpeg {operation} failed with exit code "
            f"{returncode} after {attempt + 1} attempt(s):\n{tail}{pipe_detail}"
        )


def _encode_rgb_frames_isolated(
    output_path: Path,
    chunks_factory,
    *,
    frame_count: int,
    width: int,
    height: int,
    fps: int,
    bit_depth: int,
    crf: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for isolated H.264 long-video encoding")
    if int(bit_depth) not in {8, 10}:
        raise ValueError("bit_depth must be 8 or 10")
    input_pixel_format = "rgb48le" if int(bit_depth) == 10 else "rgb24"
    output_pixel_format = "yuv420p10le" if int(bit_depth) == 10 else "yuv420p"
    bytes_per_frame = int(width) * int(height) * (6 if int(bit_depth) == 10 else 3)
    descriptor, log_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".video-encode.log.tmp", dir=output_path.parent
    )
    os.close(descriptor)
    log_path = Path(log_name)
    try:
        _run_isolated_ffmpeg_with_input(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                input_pixel_format,
                "-video_size",
                f"{int(width)}x{int(height)}",
                "-framerate",
                str(int(fps)),
                "-i",
                "pipe:0",
                "-frames:v",
                str(int(frame_count)),
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-x264-params",
                (
                    "threads=1:lookahead_threads=1:sliced_threads=0:ref=1:bframes=0:"
                    "keyint=1:min-keyint=1:scenecut=0:cabac=0"
                ),
                "-profile:v",
                "baseline",
                "-preset",
                "medium",
                "-crf",
                str(int(crf)),
                "-pix_fmt",
                output_pixel_format,
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                str(output_path),
            ],
            log_path,
            chunks_factory,
            expected_chunks=int(frame_count),
            expected_chunk_bytes=bytes_per_frame,
            operation="isolated H.264 encode",
        )
    finally:
        _cleanup_temporary(log_path, sys.exc_info()[1])


def _strict_validate_mp4(path: Path, *, require_audio: bool = True) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for strict long-video candidate validation")
    descriptor, log_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".strict-decode.log.tmp", dir=path.parent
    )
    os.close(descriptor)
    log_path = Path(log_name)
    try:
        args = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-xerror",
            "-threads",
            "1",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:v:0",
        ]
        if require_audio:
            args.extend(["-map", "0:a:0"])
        args.extend(["-f", "null", "-"])
        _run_isolated_ffmpeg(
            args,
            log_path,
            operation="strict AV decode" if require_audio else "strict video decode",
        )
    finally:
        _cleanup_temporary(log_path, sys.exc_info()[1])


def _write_planar_audio_raw(path: Path, audio_array: np.ndarray) -> None:
    if audio_array.ndim != 2 or int(audio_array.shape[0]) != 2:
        raise ValueError("normalized audio must be planar stereo [2,samples]")
    interleaved = np.ascontiguousarray(audio_array.T, dtype="<f4")
    with path.open("wb") as handle:
        handle.write(interleaved.tobytes())
        handle.flush()
        os.fsync(handle.fileno())


def _mux_video_with_raw_audio(
    video_path: Path,
    raw_audio_path: Path,
    output_path: Path,
    *,
    sample_rate: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg is required for crash-isolated AAC encoding in MiniMax H3 long-video delivery"
        )
    descriptor, log_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".ffmpeg.log.tmp", dir=output_path.parent
    )
    os.close(descriptor)
    log_path = Path(log_name)
    try:
        _run_isolated_ffmpeg(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(video_path),
                "-f",
                "f32le",
                "-ar",
                str(int(sample_rate)),
                "-ac",
                "2",
                "-i",
                str(raw_audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                # Preserve the manifest's exact logical sample boundary after AAC frame
                # quantization. The decoder trims this bounded codec padding locally.
                "-af",
                "apad=pad_len=1024",
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                str(output_path),
            ],
            log_path,
        )
    finally:
        _cleanup_temporary(log_path, sys.exc_info()[1])


def _chain_root(chain_id: str) -> Path:
    safe_chain = sanitize_chain_id(chain_id)
    output_root = Path(folder_paths.get_output_directory()).resolve()
    root = (output_root / STATE_FOLDER / safe_chain).resolve()
    if output_root != root and output_root not in root.parents:
        raise ValueError("Resolved long-video chain directory escaped the ComfyUI output folder")
    return root


def long_video_chain_root(chain_id: str) -> Path:
    """Return the validated output directory for one long-video chain."""
    return _chain_root(chain_id)


def _resolve_inside(root: Path, value: str | os.PathLike) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Resolved path escaped the long-video chain directory: {path}")
    return path


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _safe_token(value: str, *, fallback_prefix: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "").strip())
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", normalized).strip("._-")
    if not normalized:
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        normalized = f"{fallback_prefix}_{stamp}_{uuid.uuid4().hex[:8]}"
    if len(normalized) > 80:
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
        normalized = f"{normalized[:64]}_{digest}"
    return normalized


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping, *, keep_backup: bool = False) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    if keep_backup and path.is_file():
        try:
            current = path.read_bytes()
            parsed = json.loads(current.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("Existing manifest root is not an object")
            _validate_manifest(parsed, str(payload.get("chain_id", "")))
            _atomic_write_bytes(path.with_name(MANIFEST_BACKUP_NAME), current)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            # Never overwrite a known-good backup with a corrupt primary.
            pass
    _atomic_write_bytes(path, encoded)


def atomic_write_long_video_json(path: Path, payload: Mapping) -> None:
    """Atomically persist auxiliary state inside a validated chain directory."""
    resolved = Path(path).resolve()
    output_root = Path(folder_paths.get_output_directory()).resolve()
    if output_root != resolved and output_root not in resolved.parents:
        raise ValueError("Long-video auxiliary JSON escaped the ComfyUI output folder")
    _atomic_write_json(resolved, payload)


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        # Access denied means the process exists but is protected from inspection.
        return ctypes.get_last_error() == 5
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _legacy_manifest_lock_is_active(lock_path: Path, stale_seconds: float) -> bool:
    """Protect an in-flight pre-v2 lock during a rolling code upgrade.

    New builds use a persistent OS advisory lock file. The old O_EXCL lock was deleted on
    release, so a surviving legacy file either belongs to an older process that is still
    accepting a segment or is crash residue. We never remove it here: old builds retain their
    own cleanup semantics, and leaving crash residue is safer for rollback compatibility.
    """
    try:
        stat = lock_path.stat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return time.time() - stat.st_mtime <= stale_seconds
    try:
        pid = int(payload.get("pid", -1))
    except (TypeError, ValueError):
        return time.time() - stat.st_mtime <= stale_seconds
    # Never steal from a live legacy writer merely because a large file copy exceeded the old
    # age threshold. A reused PID may conservatively delay availability, but cannot corrupt data.
    return _process_is_alive(pid)


def _open_advisory_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    if os.fstat(descriptor).st_size == 0:
        handle.write(b"\0")
        handle.flush()
        os.fsync(descriptor)
    return handle


def _try_advisory_lock(handle) -> bool:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _release_advisory_lock(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _manifest_lock(root: Path, *, timeout_seconds: float = 5.0, stale_seconds: float = 300.0):
    """Serialize manifest commits across threads and processes.

    The operating system owns the actual lock, so an interpreter crash or forced termination
    releases it without waiting for an age-based lock-file deletion. The persistent v2 path is
    intentionally distinct from the legacy O_EXCL file so downgrading does not leave an old build
    blocked by a file it cannot understand.
    """
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ADVISORY_LOCK_NAME
    legacy_lock_path = root / LOCK_NAME
    deadline = time.monotonic() + timeout_seconds
    handle = _open_advisory_lock(lock_path)
    acquired = False
    try:
        while not acquired:
            acquired = _try_advisory_lock(handle)
            if acquired and _legacy_manifest_lock_is_active(legacy_lock_path, stale_seconds):
                _release_advisory_lock(handle)
                acquired = False
            if acquired:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Long-video manifest is busy: {lock_path}. Retry after the other queue finishes."
                )
            time.sleep(0.05)

        payload = json.dumps(
            {
                "lock_kind": ADVISORY_LOCK_KIND,
                "pid": os.getpid(),
                "token": uuid.uuid4().hex,
                "acquired_unix": time.time(),
            },
            ensure_ascii=False,
        ).encode("utf-8")
        handle.seek(0)
        handle.write(payload)
        handle.truncate(len(payload))
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        if acquired:
            _release_advisory_lock(handle)
        handle.close()


def _normalize_manifest_schema(payload: dict) -> dict:
    try:
        manifest_schema = int(payload.get("schema", -1))
    except (TypeError, ValueError) as error:
        raise ValueError("Long-video manifest schema is invalid") from error
    if manifest_schema == LEGACY_MANIFEST_SCHEMA:
        migrated = dict(payload)
        segments = migrated.get("segments")
        first_accepted = (
            segments[0].get("accepted_unix")
            if isinstance(segments, list) and segments and isinstance(segments[0], dict)
            else None
        )
        migrated.update(
            {
                "schema": MANIFEST_SCHEMA,
                "format": MANIFEST_FORMAT,
                "created_unix": float(
                    first_accepted
                    if isinstance(first_accepted, (int, float))
                    else migrated.get("updated_unix", 0.0)
                ),
                "migrated_from_schema": LEGACY_MANIFEST_SCHEMA,
            }
        )
        return migrated
    if manifest_schema != MANIFEST_SCHEMA:
        raise UnsupportedManifestSchemaError(
            f"Unsupported long-video manifest schema {payload.get('schema')}; expected "
            f"{MANIFEST_SCHEMA} or migratable legacy schema {LEGACY_MANIFEST_SCHEMA}"
        )
    return dict(payload)


def _validate_manifest(payload: object, chain_id: str) -> dict:
    safe_chain = sanitize_chain_id(chain_id)
    if not isinstance(payload, dict):
        raise ValueError("Long-video manifest root must be an object")
    payload = _normalize_manifest_schema(payload)
    if payload.get("format") != MANIFEST_FORMAT:
        raise ValueError("Long-video manifest format marker is invalid")
    if not isinstance(payload.get("created_unix"), (int, float)):
        raise ValueError("Long-video manifest created_unix is invalid")
    if payload.get("chain_id") != safe_chain:
        raise ValueError("Long-video manifest chain_id does not match the requested chain")
    if not isinstance(payload.get("revision"), int) or payload["revision"] < 0:
        raise ValueError("Long-video manifest revision is invalid")
    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Long-video manifest segments must be a list")
    for expected_index, segment in enumerate(segments):
        if not isinstance(segment, dict) or int(segment.get("index", -1)) != expected_index:
            raise ValueError("Long-video manifest segments must be contiguous from index 0")
        if bool(segment.get("is_final_segment")) and expected_index != len(segments) - 1:
            raise ValueError("Only the last accepted segment may be marked final")
        required = {
            "candidate_id", "video_path", "video_sha256", "frame_count", "fps",
            "width", "height", "sample_rate", "audio_start_sample", "audio_end_sample",
            "timeline_start_frame", "timeline_end_frame", "model_id", "sampling_summary",
        }
        missing = sorted(required - set(segment))
        if missing:
            raise ValueError(
                f"Accepted segment {expected_index} is missing: {', '.join(missing)}"
            )
        if int(segment["timeline_end_frame"]) - int(segment["timeline_start_frame"]) != int(
            segment["frame_count"]
        ):
            raise ValueError(f"Accepted segment {expected_index} has inconsistent frame boundaries")
        if int(segment["audio_end_sample"]) <= int(segment["audio_start_sample"]):
            raise ValueError(f"Accepted segment {expected_index} has invalid audio boundaries")
        if expected_index == 0:
            if int(segment["timeline_start_frame"]) != 0 or int(segment["audio_start_sample"]) != 0:
                raise ValueError("The first accepted segment must start at timeline zero")
        else:
            previous = segments[expected_index - 1]
            if int(segment["timeline_start_frame"]) != int(previous["timeline_end_frame"]):
                raise ValueError("Accepted segment video boundaries are not contiguous")
            if int(segment["audio_start_sample"]) != int(previous["audio_end_sample"]):
                raise ValueError("Accepted segment audio boundaries are not contiguous")
    invalidated = payload.get("invalidated", [])
    if not isinstance(invalidated, list):
        raise ValueError("Long-video manifest invalidated history must be a list")
    return payload


def _new_manifest(chain_id: str) -> dict:
    now = time.time()
    return {
        "schema": MANIFEST_SCHEMA,
        "format": MANIFEST_FORMAT,
        "chain_id": sanitize_chain_id(chain_id),
        "revision": 0,
        "segments": [],
        "invalidated": [],
        "created_unix": now,
        "updated_unix": now,
    }


def load_delivery_manifest(chain_id: str, *, allow_new: bool = False) -> tuple[dict, str]:
    root = _chain_root(chain_id)
    manifest_path = root / MANIFEST_NAME
    backup_path = root / MANIFEST_BACKUP_NAME
    if not manifest_path.is_file():
        if backup_path.is_file():
            try:
                payload = json.loads(backup_path.read_text(encoding="utf-8"))
                return _validate_manifest(payload, chain_id), "backup"
            except UnsupportedManifestSchemaError:
                # A missing primary must not turn an unknown backup into a new empty chain.
                raise
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                if allow_new:
                    raise ValueError(
                        "Long-video manifest primary is missing and its backup is corrupt; "
                        "refusing to create an empty replacement chain"
                    ) from error
                raise ValueError(
                    f"Long-video manifest primary is missing and its backup is corrupt: {error}"
                ) from error
        if allow_new:
            return _new_manifest(chain_id), "new"
        raise FileNotFoundError(f"No accepted long-video manifest exists at {manifest_path}")

    primary_error = None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return _validate_manifest(payload, chain_id), "primary"
    except UnsupportedManifestSchemaError:
        # A syntactically valid manifest from another schema is not corruption. Falling back to
        # an older backup here could silently discard accepted segments during a downgrade.
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        primary_error = error

    if backup_path.is_file():
        try:
            payload = json.loads(backup_path.read_text(encoding="utf-8"))
            return _validate_manifest(payload, chain_id), "backup"
        except UnsupportedManifestSchemaError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            pass
    raise ValueError(
        f"Long-video manifest is corrupt and no valid backup is available: {primary_error}"
    ) from primary_error


def manifest_fingerprint(chain_id: str, segment_index: int | None = None) -> str:
    root = _chain_root(chain_id)
    parts = [sanitize_chain_id(chain_id), str(segment_index)]
    for name in (MANIFEST_NAME, MANIFEST_BACKUP_NAME):
        path = root / name
        if path.is_file():
            stat = path.stat()
            parts.extend((name, str(stat.st_mtime_ns), str(stat.st_size)))
        else:
            parts.extend((name, "missing"))
    return ":".join(parts)


def _write_context_candidate(
    av_latent: dict,
    target: Path,
    chain_id: str,
    segment_index: int,
    model_id: str,
    sampling_summary: str,
) -> dict:
    video, audio = nested_av_parts(av_latent)
    total_frames = pixel_frames_from_latent_t(int(video.shape[2]))
    supported = [
        (frames, steps)
        for frames, steps in sorted(CONTEXT_FRAME_STEPS.items(), reverse=True)
        if steps <= int(video.shape[2])
    ]
    if not supported:
        raise ValueError("The sampled video latent is too short to save H3 motion context")
    max_context_frames, video_steps = supported[0]
    audio_steps = min(
        int(audio.shape[-1]), round(max_context_frames / FPS * AUDIO_LATENT_FPS)
    )
    if audio_steps < 1:
        raise ValueError("The sampled AV latent has no usable audio tail")

    video_tail = video[:1, :, -video_steps:].detach().cpu().contiguous()
    audio_tail = audio[:1, :, :, -audio_steps:].detach().cpu().contiguous()
    audio_overhang = int(audio.shape[-1]) - FRAME_RESCALE * total_frames
    if not 0.0 <= audio_overhang < 1.0:
        audio_overhang = 0.0
    metadata = {
        "schema": str(LONG_VIDEO_SCHEMA),
        "chain_id": sanitize_chain_id(chain_id),
        "source_segment_index": str(int(segment_index)),
        "model_id": str(model_id or "unknown"),
        "sampling_summary": str(sampling_summary or "unknown"),
        "fps": str(FPS),
        "source_total_frames": str(total_frames),
        "max_context_frames": str(max_context_frames),
        "video_shape": json.dumps(list(video_tail.shape)),
        "audio_shape": json.dumps(list(audio_tail.shape)),
        "video_dtype": str(video_tail.dtype),
        "audio_dtype": str(audio_tail.dtype),
        "audio_overhang": repr(float(audio_overhang)),
        "video_sha256": _tensor_sha256(video_tail),
        "audio_sha256": _tensor_sha256(audio_tail),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file({"video_tail": video_tail, "audio_tail": audio_tail}, str(temporary), metadata)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(target),
        "sha256": _sha256_file(target),
        "max_context_frames": max_context_frames,
        "video_shape": list(video_tail.shape),
        "audio_shape": list(audio_tail.shape),
    }


def _load_accepted_context_file(
    path: Path, chain_id: str, source_index: int, target_index: int
) -> tuple[dict, dict]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        keys = set(handle.keys())
        if keys != {"video_tail", "audio_tail"}:
            raise ValueError(f"Invalid H3 T8 context tensor keys in {path}: {sorted(keys)}")
        video_tail = handle.get_tensor("video_tail")
        audio_tail = handle.get_tensor("audio_tail")
    required = {
        "schema", "chain_id", "source_segment_index", "fps", "source_total_frames",
        "max_context_frames", "video_shape", "audio_shape", "video_dtype", "audio_dtype",
        "audio_overhang", "video_sha256", "audio_sha256",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"H3 T8 context metadata is incomplete: {', '.join(missing)}")
    safe_chain = sanitize_chain_id(chain_id)
    if int(metadata["schema"]) != LONG_VIDEO_SCHEMA:
        raise ValueError("Accepted context schema does not match this H3 T8 build")
    if metadata["chain_id"] != safe_chain or int(metadata["source_segment_index"]) != source_index:
        raise ValueError("Accepted context chain/segment metadata does not match the manifest")
    if int(metadata["fps"]) != FPS:
        raise ValueError(f"Accepted context fps must be {FPS}")
    if json.loads(metadata["video_shape"]) != list(video_tail.shape):
        raise ValueError("Accepted context video shape metadata does not match the tensor")
    if json.loads(metadata["audio_shape"]) != list(audio_tail.shape):
        raise ValueError("Accepted context audio shape metadata does not match the tensor")
    if metadata["video_dtype"] != str(video_tail.dtype):
        raise ValueError("Accepted context video dtype metadata does not match the tensor")
    if metadata["audio_dtype"] != str(audio_tail.dtype):
        raise ValueError("Accepted context audio dtype metadata does not match the tensor")
    max_context_frames = int(metadata["max_context_frames"])
    expected_steps = CONTEXT_FRAME_STEPS.get(max_context_frames)
    if expected_steps is None or int(video_tail.shape[2]) != expected_steps:
        raise ValueError("Accepted context is not a supported 5/22/39-frame H3 window")
    if metadata["video_sha256"] != _tensor_sha256(video_tail):
        raise ValueError("Accepted context video tensor checksum failed")
    if metadata["audio_sha256"] != _tensor_sha256(audio_tail):
        raise ValueError("Accepted context audio tensor checksum failed")
    parsed = {
        "schema": LONG_VIDEO_SCHEMA,
        "chain_id": safe_chain,
        "source_segment_index": source_index,
        "target_segment_index": target_index,
        "fps": FPS,
        "source_total_frames": int(metadata["source_total_frames"]),
        "max_context_frames": max_context_frames,
        "audio_overhang": float(metadata["audio_overhang"]),
        "model_id": metadata.get("model_id", "unknown"),
        "sampling_summary": metadata.get("sampling_summary", "unknown"),
        "path": str(path),
    }
    context = {
        "schema": LONG_VIDEO_SCHEMA,
        "empty": False,
        "video_tail": video_tail,
        "audio_tail": audio_tail,
        "metadata": parsed,
    }
    return context, parsed | {
        "video_shape": list(video_tail.shape),
        "audio_shape": list(audio_tail.shape),
        "checksums_valid": True,
    }


def _normalize_audio(audio: Mapping, target_samples: int) -> tuple[np.ndarray, dict]:
    waveform, sample_rate = validate_audio(audio)
    source = waveform[0].detach().float().cpu()
    if source.shape[0] == 1:
        source = source.expand(2, -1)
    elif source.shape[0] != 2:
        source = source.mean(dim=0, keepdim=True).expand(2, -1)
    original_samples = int(source.shape[-1])
    padded_samples = max(0, target_samples - original_samples)
    trimmed_samples = max(0, original_samples - target_samples)
    if padded_samples:
        source = torch.nn.functional.pad(source, (0, padded_samples))
    source = source[:, :target_samples].clamp(-1.0, 1.0).contiguous()
    return source.numpy().astype(np.float32, copy=False), {
        "sample_rate": int(sample_rate),
        "input_samples": original_samples,
        "written_samples": target_samples,
        "padded_samples": padded_samples,
        "trimmed_samples": trimmed_samples,
        "channels": 2,
    }


def _write_mp4_atomic(
    path: Path,
    frames: torch.Tensor,
    audio: Mapping,
    *,
    fps: int,
    target_audio_samples: int,
    bit_depth: int,
    crf: int,
) -> dict:
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise ValueError("frames must be a connected IMAGE value [N,H,W,C]")
    if int(frames.shape[0]) < 1 or int(frames.shape[-1]) < 3:
        raise ValueError("frames must contain at least one RGB image")
    if int(frames.shape[1]) % 2 or int(frames.shape[2]) % 2:
        raise ValueError("H.264 output width and height must be even")
    if int(fps) != FPS:
        raise ValueError(f"MiniMax H3 long-video delivery currently requires {FPS} fps")
    if int(bit_depth) not in {8, 10}:
        raise ValueError("bit_depth must be 8 or 10")
    audio_array, audio_report = _normalize_audio(audio, int(target_audio_samples))
    sample_rate = int(audio_report["sample_rate"])

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".mp4.tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    descriptor, video_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".video.mp4.tmp", dir=path.parent
    )
    os.close(descriptor)
    video_temporary = Path(video_name)
    descriptor, audio_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".audio.f32.tmp", dir=path.parent
    )
    os.close(descriptor)
    audio_temporary = Path(audio_name)
    try:
        def frame_chunks():
            for image in frames:
                image = image[..., :3].detach().float().clamp(0.0, 1.0).cpu()
                if bit_depth == 10:
                    array = (image * 65535.0).round().numpy().astype("<u2", copy=False)
                else:
                    array = (image * 255.0).round().numpy().astype(np.uint8, copy=False)
                yield np.ascontiguousarray(array).tobytes()

        # Both native encoders run outside the long-lived ComfyUI process. The raw RGB pipe
        # bounds memory to one generated frame and avoids the damaged in-process PyAV/libx264
        # packet observed during the real 32-second MV stage gate.
        _encode_rgb_frames_isolated(
            video_temporary,
            frame_chunks,
            frame_count=int(frames.shape[0]),
            width=int(frames.shape[2]),
            height=int(frames.shape[1]),
            fps=int(fps),
            bit_depth=int(bit_depth),
            crf=int(crf),
        )
        # Localize corruption before AAC mux/stream-copy so an encoder fault and a mux fault
        # cannot collapse into the same report. Single-threaded strict decode is independent
        # from the isolated single-threaded libx264 process that produced this file.
        _strict_validate_mp4(video_temporary, require_audio=False)
        _write_planar_audio_raw(audio_temporary, audio_array)
        _mux_video_with_raw_audio(
            video_temporary,
            audio_temporary,
            temporary,
            sample_rate=sample_rate,
        )
        # A successfully muxed MP4 can still contain a damaged H.264 packet. Do not publish
        # or accept it based on container metadata and hashes alone: decode every AV packet
        # in an isolated single-threaded FFmpeg process while the file is still temporary.
        _strict_validate_mp4(temporary)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        active_error = sys.exc_info()[1]
        _cleanup_temporary(temporary, active_error)
        _cleanup_temporary(video_temporary, active_error)
        _cleanup_temporary(audio_temporary, active_error)
    return audio_report | {
        "path": str(path),
        "sha256": _sha256_file(path),
        "frame_count": int(frames.shape[0]),
        "width": int(frames.shape[2]),
        "height": int(frames.shape[1]),
        "fps": int(fps),
        "bit_depth": int(bit_depth),
        "crf": int(crf),
        "video_encoder_process": "isolated_ffmpeg_subprocess",
        "video_encoder_policy": ISOLATED_VIDEO_ENCODER_POLICY,
        "video_only_strict_decode_validated": True,
        "audio_encoder_process": "isolated_ffmpeg_subprocess",
        "strict_decode_validated": True,
        "strict_decode_policy": STRICT_AV_DECODE_POLICY,
    }


def save_long_video_candidate(
    frames: torch.Tensor,
    audio: Mapping,
    av_latent: dict,
    chain_id: str,
    segment_index: int,
    timeline_start_seconds: float,
    save_context: bool,
    parent_candidate_id: str = "",
    parent_manifest_revision: int = 0,
    candidate_id: str = "",
    model_id: str = "unknown",
    sampling_summary: str = "dual_clock_euler/native_flow",
    prompt: str = "",
    seed: int = 0,
    fps: int = FPS,
    bit_depth: int = 8,
    crf: int = 18,
) -> tuple[str, str, str]:
    safe_chain = sanitize_chain_id(chain_id)
    segment_index = int(segment_index)
    if segment_index < 0:
        raise ValueError("segment_index cannot be negative")
    if int(fps) != FPS:
        raise ValueError(f"MiniMax H3 long-video delivery currently requires {FPS} fps")
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise ValueError("frames must be IMAGE [N,H,W,C]")
    frame_count = int(frames.shape[0])
    if frame_count < 1:
        raise ValueError("candidate frames cannot be empty")
    start_frame_float = float(timeline_start_seconds) * FPS
    timeline_start_frame = round(start_frame_float)
    if not math.isclose(start_frame_float, timeline_start_frame, abs_tol=1e-5):
        raise ValueError("timeline_start_seconds must land on an exact 24fps frame boundary")
    if segment_index == 0 and timeline_start_frame != 0:
        raise ValueError("segment 0 must start at timeline zero")
    if segment_index > 0 and not str(parent_candidate_id or "").strip():
        raise ValueError(
            "Continuation candidates must record the immediately previous accepted candidate id"
        )

    _, sample_rate = validate_audio(audio)
    timeline_end_frame = timeline_start_frame + frame_count
    audio_start_sample = round(timeline_start_frame * sample_rate / FPS)
    audio_end_sample = round(timeline_end_frame * sample_rate / FPS)
    target_audio_samples = audio_end_sample - audio_start_sample
    if target_audio_samples < 1:
        raise ValueError("candidate audio boundary calculation produced an empty segment")

    root = _chain_root(safe_chain)
    token = _safe_token(candidate_id, fallback_prefix=f"seg{segment_index:05d}")
    candidate_dir = root / "candidates" / f"segment_{segment_index:05d}" / token
    candidate_json = candidate_dir / "candidate.json"
    candidate_video = candidate_dir / "candidate.mp4"
    candidate_context = candidate_dir / "candidate.context.safetensors"
    if candidate_json.exists() or candidate_video.exists() or candidate_context.exists():
        raise FileExistsError(
            f"Candidate id already exists for segment {segment_index}: {token}. Use a new id."
        )

    created: list[Path] = []
    try:
        video_report = _write_mp4_atomic(
            candidate_video,
            frames,
            audio,
            fps=fps,
            target_audio_samples=target_audio_samples,
            bit_depth=bit_depth,
            crf=crf,
        )
        created.append(candidate_video)
        context_report = None
        if save_context:
            context_report = _write_context_candidate(
                av_latent,
                candidate_context,
                safe_chain,
                segment_index,
                model_id,
                sampling_summary,
            )
            created.append(candidate_context)

        descriptor = {
            "schema": DELIVERY_SCHEMA,
            "status": "candidate",
            "chain_id": safe_chain,
            "index": segment_index,
            "candidate_id": token,
            "parent_candidate_id": str(parent_candidate_id or ""),
            "parent_manifest_revision": int(parent_manifest_revision),
            "video_path": _relative_path(candidate_video, root),
            "video_sha256": video_report["sha256"],
            "context_path": (
                _relative_path(candidate_context, root) if context_report is not None else ""
            ),
            "context_sha256": context_report["sha256"] if context_report is not None else "",
            "frame_count": frame_count,
            "fps": int(fps),
            "width": int(frames.shape[2]),
            "height": int(frames.shape[1]),
            "sample_rate": int(sample_rate),
            "audio_samples": target_audio_samples,
            "audio_start_sample": audio_start_sample,
            "audio_end_sample": audio_end_sample,
            "timeline_start_frame": timeline_start_frame,
            "timeline_end_frame": timeline_end_frame,
            "is_final_segment": not bool(save_context),
            "model_id": str(model_id or "unknown"),
            "sampling_summary": str(sampling_summary or "unknown"),
            "prompt": str(prompt or ""),
            "seed": int(seed),
            "bit_depth": int(bit_depth),
            "crf": int(crf),
            "video_encoder_process": str(video_report["video_encoder_process"]),
            "video_encoder_policy": str(video_report["video_encoder_policy"]),
            "video_only_strict_decode_validated": bool(
                video_report["video_only_strict_decode_validated"]
            ),
            "strict_decode_validated": bool(video_report["strict_decode_validated"]),
            "strict_decode_policy": str(video_report["strict_decode_policy"]),
            "created_unix": time.time(),
            "audio_adjustment": {
                key: video_report[key]
                for key in ("input_samples", "written_samples", "padded_samples", "trimmed_samples")
            },
        }
        _atomic_write_json(candidate_json, descriptor)
        created.append(candidate_json)
    except BaseException:
        for path in reversed(created):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise

    report = descriptor | {
        "candidate_json_path": str(candidate_json),
        "candidate_video_path": str(candidate_video),
        "context_saved": context_report is not None,
        "atomic_files": True,
        "accepted": False,
    }
    return str(candidate_json), str(candidate_video), json.dumps(report, ensure_ascii=False, indent=2)


def _load_candidate(candidate_json_path: str) -> tuple[dict, Path, Path]:
    candidate_json = Path(str(candidate_json_path or "")).resolve()
    if not candidate_json.is_file():
        raise FileNotFoundError(f"Long-video candidate descriptor does not exist: {candidate_json}")
    try:
        candidate = json.loads(candidate_json.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Long-video candidate descriptor is invalid JSON: {error}") from error
    if not isinstance(candidate, dict) or int(candidate.get("schema", -1)) != DELIVERY_SCHEMA:
        raise ValueError("Long-video candidate descriptor schema is unsupported")
    if candidate.get("status") != "candidate":
        raise ValueError("Long-video candidate descriptor status is invalid")
    required = {
        "chain_id", "index", "candidate_id", "parent_candidate_id",
        "parent_manifest_revision", "video_path", "video_sha256", "context_path",
        "context_sha256", "frame_count", "fps", "width", "height", "sample_rate",
        "audio_samples", "audio_start_sample", "audio_end_sample", "timeline_start_frame",
        "timeline_end_frame", "is_final_segment", "model_id", "sampling_summary",
        "strict_decode_validated", "strict_decode_policy",
    }
    missing = sorted(required - set(candidate))
    if missing:
        raise ValueError("Long-video candidate descriptor is missing: " + ", ".join(missing))
    if candidate["strict_decode_validated"] is not True:
        raise ValueError("Long-video candidate did not pass strict AV decode validation")
    if candidate["strict_decode_policy"] != STRICT_AV_DECODE_POLICY:
        raise ValueError("Long-video candidate strict AV decode policy is unsupported")
    if int(candidate["index"]) < 0:
        raise ValueError("Long-video candidate index cannot be negative")
    if int(candidate["timeline_end_frame"]) - int(candidate["timeline_start_frame"]) != int(
        candidate["frame_count"]
    ):
        raise ValueError("Long-video candidate frame boundaries are inconsistent")
    if int(candidate["audio_end_sample"]) - int(candidate["audio_start_sample"]) != int(
        candidate["audio_samples"]
    ):
        raise ValueError("Long-video candidate audio boundaries are inconsistent")
    root = _chain_root(candidate.get("chain_id", ""))
    candidate_json = _resolve_inside(root, candidate_json)
    expected_dir = root / "candidates" / f"segment_{int(candidate.get('index', -1)):05d}"
    if expected_dir != candidate_json.parent.parent:
        raise ValueError("Candidate descriptor is not in its expected chain/segment directory")
    if candidate_json.parent.name != _safe_token(candidate["candidate_id"], fallback_prefix="candidate"):
        raise ValueError("Candidate descriptor directory does not match candidate_id")
    video_path = _resolve_inside(root, candidate.get("video_path", ""))
    if video_path.parent != candidate_json.parent:
        raise ValueError("Candidate video must be stored beside its descriptor")
    if not video_path.is_file() or _sha256_file(video_path) != candidate.get("video_sha256"):
        raise ValueError("Candidate video is missing or its SHA-256 checksum failed")
    context_value = candidate.get("context_path", "")
    if bool(candidate.get("is_final_segment")):
        if context_value:
            raise ValueError("A final candidate must not contain a continuation context")
    else:
        context_path = _resolve_inside(root, context_value)
        if context_path.parent != candidate_json.parent:
            raise ValueError("Candidate context must be stored beside its descriptor")
        if not context_path.is_file() or _sha256_file(context_path) != candidate.get("context_sha256"):
            raise ValueError("Candidate continuation context is missing or failed SHA-256")
    return candidate, root, video_path


def load_long_video_candidate_descriptor(candidate_json_path: str) -> tuple[dict, str]:
    """Validate a candidate and return its descriptor plus absolute preview path."""
    candidate, _root, video_path = _load_candidate(candidate_json_path)
    return dict(candidate), str(video_path)


def _copy_atomic(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        source_hash = _sha256_file(source)
        if _sha256_file(temporary) != source_hash:
            raise OSError(f"Atomic copy checksum failed for {source}")
        os.replace(temporary, target)
        return source_hash
    finally:
        if temporary.exists():
            temporary.unlink()


def _accepted_asset_matches(path: Path, expected_sha256: str) -> bool:
    if not path.is_file():
        return False
    try:
        return _sha256_file(path) == str(expected_sha256)
    except OSError:
        return False


def _assert_accepted_destination_compatible(path: Path, expected_sha256: str) -> None:
    """Refuse to overwrite an accepted path that already belongs to different bytes.

    Accepted filenames contain the normalized candidate id. Reusing an id (or two ids that
    normalize to the same token) must not overwrite an immutable asset before the manifest
    transaction commits. An existing byte-identical orphan from an interrupted earlier attempt is
    safe and may be reused.
    """
    if not path.exists():
        return
    if not path.is_file() or not _accepted_asset_matches(path, expected_sha256):
        raise ValueError(
            f"Accepted destination collision at {path}. Use a new candidate_id; the existing "
            "accepted asset was left unchanged."
        )


def _assert_chain_identity(manifest: dict, candidate: dict) -> None:
    if not manifest["segments"]:
        return
    first = manifest["segments"][0]
    fields = ("fps", "width", "height", "sample_rate", "model_id", "sampling_summary")
    mismatches = [
        field for field in fields if candidate.get(field) != first.get(field)
    ]
    if mismatches:
        raise ValueError(
            "Candidate identity differs from the accepted chain: " + ", ".join(mismatches)
        )


def accept_long_video_candidate(
    candidate_json_path: str,
    accept_candidate: bool,
    replace_policy: str = "reject_existing",
    strict_chain_identity: bool = True,
) -> tuple[str, bool, str, str]:
    if replace_policy not in {"reject_existing", "replace_and_invalidate_following"}:
        raise ValueError("Unknown long-video replace policy")
    candidate, root, candidate_video = _load_candidate(candidate_json_path)
    if not accept_candidate:
        report = candidate | {
            "accepted": False,
            "reason": "accept_candidate is false; preview/review only",
            "candidate_video_path": str(candidate_video),
        }
        return str(candidate_video), False, "", json.dumps(report, ensure_ascii=False, indent=2)

    manifest_path = root / MANIFEST_NAME
    with _manifest_lock(root):
        manifest, manifest_source = load_delivery_manifest(candidate["chain_id"], allow_new=True)
        index = int(candidate["index"])
        segments = manifest["segments"]

        if index < len(segments):
            existing = segments[index]
            same_candidate_id = existing.get("candidate_id") == candidate.get("candidate_id")
            same_candidate_assets = (
                existing.get("video_sha256") == candidate.get("video_sha256")
                and existing.get("context_sha256", "")
                == candidate.get("context_sha256", "")
            )
            if same_candidate_id and not same_candidate_assets:
                raise ValueError(
                    f"Candidate id '{candidate['candidate_id']}' is already bound to different "
                    f"accepted assets for segment {index}. Use a new candidate_id for a replacement."
                )
            if same_candidate_id and same_candidate_assets:
                accepted_video = _resolve_inside(root, existing["video_path"])
                repaired_assets = []
                if not _accepted_asset_matches(accepted_video, existing["video_sha256"]):
                    repaired_hash = _copy_atomic(candidate_video, accepted_video)
                    if repaired_hash != existing["video_sha256"]:
                        raise OSError("Repaired accepted video does not match the manifest hash")
                    repaired_assets.append("video")
                accepted_context = None
                if existing.get("context_path"):
                    accepted_context = _resolve_inside(root, existing["context_path"])
                    source_context = _resolve_inside(root, candidate["context_path"])
                    if not _accepted_asset_matches(
                        accepted_context, existing.get("context_sha256", "")
                    ):
                        repaired_hash = _copy_atomic(source_context, accepted_context)
                        if repaired_hash != existing.get("context_sha256", ""):
                            raise OSError(
                                "Repaired accepted context does not match the manifest hash"
                            )
                        repaired_assets.append("context")
                report = {
                    "schema": DELIVERY_SCHEMA,
                    "chain_id": candidate["chain_id"],
                    "index": index,
                    "candidate_id": candidate["candidate_id"],
                    "accepted": True,
                    "idempotent": True,
                    "manifest_revision": manifest["revision"],
                    "manifest_source": manifest_source,
                    "manifest_path": str(manifest_path),
                    "accepted_video_path": str(accepted_video),
                    "accepted_context_path": (
                        str(accepted_context) if accepted_context is not None else ""
                    ),
                    "repaired_assets": repaired_assets,
                }
                return (
                    str(accepted_video), True, str(manifest_path),
                    json.dumps(report, ensure_ascii=False, indent=2),
                )
            if replace_policy == "reject_existing":
                raise ValueError(
                    f"Segment {index} is already accepted. Select replace_and_invalidate_following "
                    "only after intentionally reviewing this replacement."
                )
        elif index > len(segments):
            raise ValueError(
                f"Cannot accept segment {index}; accepted segment {len(segments)} is the next slot"
            )

        if int(candidate.get("parent_manifest_revision", -1)) != int(manifest["revision"]):
            raise ValueError(
                "Candidate was generated from a stale manifest revision; preview the current "
                "Accepted Context and re-generate this candidate."
            )

        if index > 0:
            if index - 1 >= len(segments):
                raise ValueError(f"Segment {index} has no accepted parent segment {index - 1}")
            expected_parent = segments[index - 1]["candidate_id"]
            if candidate.get("parent_candidate_id") != expected_parent:
                raise ValueError(
                    "Candidate was generated from a stale or different accepted parent; re-generate it "
                    "from the current Accepted Context node."
                )
        elif candidate.get("parent_candidate_id"):
            raise ValueError("Segment 0 candidate must not declare a parent candidate")

        if index == len(segments) and segments and bool(segments[-1].get("is_final_segment")):
            raise ValueError("The accepted chain already ends with a final segment")
        if strict_chain_identity:
            _assert_chain_identity(manifest, candidate)

        if index > 0:
            previous = segments[index - 1]
            if int(candidate["timeline_start_frame"]) != int(previous["timeline_end_frame"]):
                raise ValueError("Candidate video timeline does not continue the accepted parent")
            if int(candidate["audio_start_sample"]) != int(previous["audio_end_sample"]):
                raise ValueError("Candidate audio sample boundary does not continue the accepted parent")

        accepted_dir = root / "accepted"
        safe_candidate = _safe_token(candidate["candidate_id"], fallback_prefix="candidate")
        accepted_video = accepted_dir / f"segment_{index:05d}_{safe_candidate}.mp4"
        accepted_context = None
        context_hash = str(candidate.get("context_sha256", ""))
        if candidate.get("context_path"):
            source_context = _resolve_inside(root, candidate["context_path"])
            accepted_context = accepted_dir / (
                f"segment_{index:05d}_{safe_candidate}.context.safetensors"
            )
        _assert_accepted_destination_compatible(
            accepted_video, str(candidate["video_sha256"])
        )
        if accepted_context is not None:
            _assert_accepted_destination_compatible(accepted_context, context_hash)
        video_hash = _copy_atomic(candidate_video, accepted_video)
        if accepted_context is not None:
            context_hash = _copy_atomic(source_context, accepted_context)

        invalidated = list(manifest.get("invalidated", []))
        invalidated_now = []
        if index < len(segments):
            invalidated_now = [dict(item) for item in segments[index:]]
            for item in invalidated_now:
                item["invalidated_unix"] = time.time()
                item["invalidated_reason"] = (
                    f"segment {index} was replaced by candidate {candidate['candidate_id']}"
                )
            invalidated.extend(invalidated_now)
            segments = segments[:index]

        accepted_entry = {
            key: candidate[key]
            for key in (
                "index", "candidate_id", "parent_candidate_id", "frame_count", "fps",
                "width", "height", "sample_rate", "audio_samples", "audio_start_sample",
                "audio_end_sample", "timeline_start_frame", "timeline_end_frame",
                "is_final_segment", "model_id", "sampling_summary", "prompt", "seed",
                "bit_depth", "crf", "strict_decode_validated", "strict_decode_policy",
                "created_unix",
            )
        }
        accepted_entry.update(
            {
                "video_path": _relative_path(accepted_video, root),
                "video_sha256": video_hash,
                "video_encoder_process": str(
                    candidate.get("video_encoder_process", "legacy_unknown")
                ),
                "video_encoder_policy": str(
                    candidate.get("video_encoder_policy", "legacy_unknown")
                ),
                "video_only_strict_decode_validated": bool(
                    candidate.get("video_only_strict_decode_validated", False)
                ),
                "context_path": (
                    _relative_path(accepted_context, root) if accepted_context is not None else ""
                ),
                "context_sha256": context_hash,
                "accepted_unix": time.time(),
            }
        )
        segments.append(accepted_entry)
        manifest = dict(manifest)
        manifest.update(
            {
                "schema": MANIFEST_SCHEMA,
                "format": MANIFEST_FORMAT,
                "chain_id": candidate["chain_id"],
                "revision": int(manifest["revision"]) + 1,
                "segments": segments,
                "invalidated": invalidated,
                "updated_unix": time.time(),
            }
        )
        _validate_manifest(manifest, candidate["chain_id"])
        _atomic_write_json(manifest_path, manifest, keep_backup=True)

    report = {
        "schema": DELIVERY_SCHEMA,
        "chain_id": candidate["chain_id"],
        "index": index,
        "candidate_id": candidate["candidate_id"],
        "accepted": True,
        "idempotent": False,
        "manifest_revision": manifest["revision"],
        "manifest_source_before_write": manifest_source,
        "manifest_path": str(manifest_path),
        "accepted_video_path": str(accepted_video),
        "accepted_context_path": str(accepted_context) if accepted_context is not None else "",
        "invalidated_segment_count": len(invalidated_now),
        "candidate_files_retained": True,
    }
    return (
        str(accepted_video), True, str(manifest_path),
        json.dumps(report, ensure_ascii=False, indent=2),
    )


def load_accepted_context(chain_id: str, segment_index: int) -> tuple[dict, bool, str, int, str]:
    safe_chain = sanitize_chain_id(chain_id)
    segment_index = int(segment_index)
    if segment_index < 0:
        raise ValueError("segment_index cannot be negative")
    if segment_index == 0:
        try:
            manifest, manifest_source = load_delivery_manifest(safe_chain)
            manifest_revision = int(manifest["revision"])
        except FileNotFoundError:
            manifest_source = "new_chain"
            manifest_revision = 0
        context = {
            "schema": LONG_VIDEO_SCHEMA,
            "empty": True,
            "chain_id": safe_chain,
            "target_segment_index": 0,
        }
        report = context | {
            "accepted_candidate_id": "",
            "manifest_revision": manifest_revision,
            "manifest_source": manifest_source,
        }
        return (
            context,
            False,
            "",
            manifest_revision,
            json.dumps(report, ensure_ascii=False, indent=2),
        )

    manifest, source = load_delivery_manifest(safe_chain)
    source_index = segment_index - 1
    if source_index >= len(manifest["segments"]):
        raise FileNotFoundError(
            f"Segment {segment_index} needs accepted segment {source_index}, but the manifest "
            f"contains only {len(manifest['segments'])} accepted segment(s)"
        )
    entry = manifest["segments"][source_index]
    if bool(entry.get("is_final_segment")):
        raise ValueError("A final accepted segment cannot be used as continuation context")
    root = _chain_root(safe_chain)
    context_value = entry.get("context_path", "")
    if not context_value:
        raise ValueError(f"Accepted segment {source_index} has no continuation context")
    path = _resolve_inside(root, context_value)
    if not path.is_file() or _sha256_file(path) != entry.get("context_sha256"):
        raise ValueError("Accepted continuation context is missing or failed its file checksum")
    context, context_report = _load_accepted_context_file(
        path, safe_chain, source_index, segment_index
    )
    context["metadata"]["accepted_candidate_id"] = entry["candidate_id"]
    context["metadata"]["manifest_revision"] = manifest["revision"]
    report = context_report | {
        "accepted_candidate_id": entry["candidate_id"],
        "manifest_revision": manifest["revision"],
        "manifest_source": source,
        "file_checksum_valid": True,
    }
    return (
        context,
        True,
        entry["candidate_id"],
        manifest["revision"],
        json.dumps(report, ensure_ascii=False, indent=2),
    )


def apply_cosine_bridge(
    previous_last_sample: np.ndarray,
    current_audio: np.ndarray,
    bridge_samples: int,
) -> tuple[np.ndarray, dict]:
    if current_audio.ndim != 2:
        raise ValueError("current_audio must be [channels,samples]")
    previous = np.asarray(previous_last_sample, dtype=np.float32).reshape(-1)
    if previous.shape[0] != current_audio.shape[0]:
        raise ValueError("Audio bridge channel count does not match")
    count = min(max(0, int(bridge_samples)), int(current_audio.shape[1]))
    output = np.asarray(current_audio, dtype=np.float32).copy()
    before = float(np.max(np.abs(output[:, 0] - previous))) if output.shape[1] else 0.0
    clipped = 0
    if count > 0:
        if count == 1:
            weights = np.ones((1,), dtype=np.float32)
        else:
            weights = 0.5 * (
                1.0 + np.cos(np.linspace(0.0, np.pi, count, dtype=np.float32))
            )
        delta = previous - output[:, 0]
        output[:, :count] += delta[:, None] * weights[None, :]
        clipped = int(np.count_nonzero((output < -1.0) | (output > 1.0)))
        np.clip(output, -1.0, 1.0, out=output)
    after = float(np.max(np.abs(output[:, 0] - previous))) if output.shape[1] else 0.0
    return output, {
        "bridge_samples": count,
        "jump_before": before,
        "jump_after": after,
        "clipped_sample_values": clipped,
    }


def _decode_audio_exact(path: Path, sample_rate: int, expected_samples: int) -> tuple[np.ndarray, dict]:
    import av

    chunks: list[np.ndarray] = []
    with av.open(str(path), mode="r") as container:
        if not container.streams.audio:
            raise ValueError(f"Accepted segment has no audio stream: {path}")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=sample_rate)
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                array = converted.to_ndarray()
                if array.ndim != 2 or array.shape[0] != 2:
                    raise ValueError("Decoded accepted audio did not produce planar stereo")
                chunks.append(array.astype(np.float32, copy=False))
        for converted in resampler.resample(None):
            array = converted.to_ndarray()
            chunks.append(array.astype(np.float32, copy=False))
    decoded = np.concatenate(chunks, axis=1) if chunks else np.zeros((2, 0), dtype=np.float32)
    decoded_samples = int(decoded.shape[1])
    if decoded_samples < expected_samples:
        raise ValueError(
            f"Accepted audio decoded to {decoded_samples} samples, shorter than manifest "
            f"boundary {expected_samples}: {path}"
        )
    return decoded[:, :expected_samples].copy(), {
        "decoded_samples": decoded_samples,
        "used_samples": expected_samples,
        "discarded_codec_padding_samples": decoded_samples - expected_samples,
    }


def _verify_accepted_files(manifest: dict, root: Path) -> list[tuple[dict, Path]]:
    verified = []
    for segment in manifest["segments"]:
        path = _resolve_inside(root, segment["video_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Accepted segment file is missing: {path}")
        if _sha256_file(path) != segment["video_sha256"]:
            raise ValueError(f"Accepted segment file checksum failed: {path}")
        verified.append((segment, path))
    return verified


def compose_accepted_long_video(
    chain_id: str,
    filename_prefix: str = "H3_Long_Video",
    require_final_segment: bool = True,
    audio_seam_policy: str = "cosine_bridge",
    bridge_ms: float = 5.0,
    crf: int = 18,
) -> tuple[str, str]:
    if audio_seam_policy not in {"none", "cosine_bridge"}:
        raise ValueError("audio_seam_policy must be none or cosine_bridge")
    if not 0.0 <= float(bridge_ms) <= 50.0:
        raise ValueError("bridge_ms must be between 0 and 50")
    manifest, manifest_source = load_delivery_manifest(chain_id)
    if not manifest["segments"]:
        raise ValueError("The accepted long-video manifest has no segments")
    if require_final_segment and not bool(manifest["segments"][-1].get("is_final_segment")):
        raise ValueError("The last accepted segment is not marked final")
    root = _chain_root(chain_id)
    verified = _verify_accepted_files(manifest, root)
    first = manifest["segments"][0]
    fps = int(first["fps"])
    sample_rate = int(first["sample_rate"])
    width = int(first["width"])
    height = int(first["height"])
    total_frames = int(manifest["segments"][-1]["timeline_end_frame"])
    total_audio_samples = int(manifest["segments"][-1]["audio_end_sample"])

    safe_prefix = _safe_token(filename_prefix, fallback_prefix="H3_Long_Video")
    assembled_dir = root / "assembled"
    output_path = assembled_dir / (
        f"{safe_prefix}_r{manifest['revision']:04d}_{audio_seam_policy}.mp4"
    )
    assembled_dir.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".mp4.tmp", dir=assembled_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    descriptor, video_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".video.mp4.tmp", dir=assembled_dir
    )
    os.close(descriptor)
    video_temporary = Path(video_name)
    descriptor, audio_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".audio.f32.tmp", dir=assembled_dir
    )
    os.close(descriptor)
    audio_temporary = Path(audio_name)
    seam_reports = []
    segment_reports = []
    encoded_frames = 0
    encoded_audio_samples = 0
    previous_last = None
    try:
        def frame_chunks():
            for segment, path in verified:
                local_frames = 0
                import av

                with av.open(str(path), mode="r") as source:
                    if not source.streams.video:
                        raise ValueError(f"Accepted segment has no video stream: {path}")
                    for decoded_frame in source.decode(source.streams.video[0]):
                        if local_frames >= int(segment["frame_count"]):
                            raise ValueError(
                                f"Accepted segment {segment['index']} contains more video frames "
                                f"than the manifest declares ({segment['frame_count']})"
                            )
                        rgb = decoded_frame.to_ndarray(format="rgb24")
                        local_frames += 1
                        yield np.ascontiguousarray(rgb).tobytes()
                if local_frames != int(segment["frame_count"]):
                    raise ValueError(
                        f"Accepted segment {segment['index']} decoded {local_frames} video frames; "
                        f"manifest requires {segment['frame_count']}"
                    )

        # Decode one accepted frame at a time and pipe it to an isolated libx264 process. No
        # full-chain IMAGE tensor or multi-gigabyte raw-video staging file is created.
        _encode_rgb_frames_isolated(
            video_temporary,
            frame_chunks,
            frame_count=total_frames,
            width=width,
            height=height,
            fps=fps,
            bit_depth=8,
            crf=int(crf),
        )
        _strict_validate_mp4(video_temporary, require_audio=False)
        encoded_frames = total_frames

        # Audio is held one accepted segment at a time and streamed to a small raw temporary
        # file. Absolute manifest boundaries keep the total sample count invariant even when
        # samples-per-frame is fractional. AAC encoding/mux then runs outside this Python process.
        with audio_temporary.open("wb") as audio_handle:
            for segment, path in verified:
                expected = int(segment["audio_end_sample"]) - int(segment["audio_start_sample"])
                audio_array, decode_report = _decode_audio_exact(path, sample_rate, expected)
                seam_report = None
                if previous_last is not None:
                    if audio_seam_policy == "cosine_bridge":
                        bridge_samples = round(float(bridge_ms) * sample_rate / 1000.0)
                        audio_array, seam_report = apply_cosine_bridge(
                            previous_last, audio_array, bridge_samples
                        )
                    else:
                        jump = float(np.max(np.abs(audio_array[:, 0] - previous_last)))
                        seam_report = {
                            "bridge_samples": 0,
                            "jump_before": jump,
                            "jump_after": jump,
                            "clipped_sample_values": 0,
                        }
                    seam_report.update(
                        {
                            "boundary_before_segment": int(segment["index"]),
                            "absolute_sample": int(segment["audio_start_sample"]),
                        }
                    )
                    seam_reports.append(seam_report)
                previous_last = audio_array[:, -1].copy()
                interleaved = np.ascontiguousarray(audio_array.T, dtype="<f4")
                audio_handle.write(interleaved.tobytes())
                encoded_audio_samples += expected
                segment_reports.append(
                    {
                        "index": int(segment["index"]),
                        "frame_count": int(segment["frame_count"]),
                        "audio_samples": expected,
                        **decode_report,
                    }
                )
            audio_handle.flush()
            os.fsync(audio_handle.fileno())
        if encoded_frames != total_frames:
            raise RuntimeError(
                f"Composed {encoded_frames} frames but manifest absolute end is {total_frames}"
            )
        if encoded_audio_samples != total_audio_samples:
            raise RuntimeError(
                f"Composed {encoded_audio_samples} samples but manifest absolute end is "
                f"{total_audio_samples}"
            )
        _mux_video_with_raw_audio(
            video_temporary,
            audio_temporary,
            temporary,
            sample_rate=sample_rate,
        )
        _strict_validate_mp4(temporary)
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        active_error = sys.exc_info()[1]
        _cleanup_temporary(temporary, active_error)
        _cleanup_temporary(video_temporary, active_error)
        _cleanup_temporary(audio_temporary, active_error)

    report = {
        "schema": DELIVERY_SCHEMA,
        "chain_id": sanitize_chain_id(chain_id),
        "manifest_revision": manifest["revision"],
        "manifest_source": manifest_source,
        "segment_count": len(verified),
        "final_segment_present": bool(manifest["segments"][-1].get("is_final_segment")),
        "output_path": str(output_path),
        "output_sha256": _sha256_file(output_path),
        "fps": fps,
        "frame_count": total_frames,
        "video_duration_seconds": total_frames / fps,
        "sample_rate": sample_rate,
        "audio_samples": total_audio_samples,
        "audio_duration_seconds": total_audio_samples / sample_rate,
        "absolute_sample_accounting": True,
        "audio_seam_policy": audio_seam_policy,
        "bridge_ms": float(bridge_ms) if audio_seam_policy == "cosine_bridge" else 0.0,
        "video_reencoded_h264": True,
        "audio_reencoded_aac": True,
        "video_encoder_process": "isolated_ffmpeg_subprocess",
        "video_encoder_policy": ISOLATED_VIDEO_ENCODER_POLICY,
        "video_only_strict_decode_validated": True,
        "audio_encoder_process": "isolated_ffmpeg_subprocess",
        "strict_decode_validated": True,
        "strict_decode_policy": STRICT_AV_DECODE_POLICY,
        "streaming_memory_scope": (
            "one decoded video frame piped to isolated libx264 plus one segment PCM buffer; "
            "interleaved audio is staged on disk before isolated AAC mux"
        ),
        "segments": segment_reports,
        "seams": seam_reports,
    }
    return str(output_path), json.dumps(report, ensure_ascii=False, indent=2)
