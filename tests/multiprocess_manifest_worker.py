from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PACKAGE_ROOT.parents[1]
PACKAGE_NAME = "h3_audio_t8_pkg"


def _load_package() -> None:
    sys.path.insert(0, str(COMFY_ROOT))
    sys.path.insert(0, str(PACKAGE_ROOT / "tests"))
    if PACKAGE_NAME in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _wait_for(path: Path, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(0.02)


def _accept(args) -> int:
    from h3_audio_t8_pkg import long_video_delivery as delivery

    output_dir = str(Path(args.output_dir).resolve())
    delivery.folder_paths.get_output_directory = lambda: output_dir
    Path(args.ready).write_text(str(os.getpid()), encoding="utf-8")
    _wait_for(Path(args.start))
    try:
        output = delivery.accept_long_video_candidate(args.candidate, True)
        payload = {
            "ok": True,
            "pid": os.getpid(),
            "accepted_video": output[0],
            "report": json.loads(output[3]),
        }
    except Exception as error:  # noqa: BLE001 - the parent needs the exact cross-process failure.
        payload = {
            "ok": False,
            "pid": os.getpid(),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    _write_json(Path(args.result), payload)
    return 0


def _hold(args) -> int:
    from h3_audio_t8_pkg.long_video_delivery import _manifest_lock

    with _manifest_lock(Path(args.root), timeout_seconds=5.0):
        Path(args.ready).write_text(str(os.getpid()), encoding="utf-8")
        time.sleep(args.hold_seconds)
    return 0


def _counter(args) -> int:
    from h3_audio_t8_pkg.long_video_delivery import _manifest_lock

    root = Path(args.root)
    counter = Path(args.counter)
    Path(args.ready).write_text(str(os.getpid()), encoding="utf-8")
    _wait_for(Path(args.start))
    for _ in range(args.iterations):
        with _manifest_lock(root, timeout_seconds=30.0):
            value = int(counter.read_text(encoding="utf-8"))
            time.sleep(0.001)
            counter.write_text(str(value + 1), encoding="utf-8")
    _write_json(Path(args.result), {"ok": True, "pid": os.getpid()})
    return 0


def _accept_killpoint(args) -> int:
    """Pause inside an accepted-manifest transaction until the parent kills this process."""
    from h3_audio_t8_pkg import long_video_delivery as delivery

    output_dir = str(Path(args.output_dir).resolve())
    delivery.folder_paths.get_output_directory = lambda: output_dir
    ready = Path(args.ready)

    if args.break_at == "after-video-copy":
        original_copy = delivery._copy_atomic
        copy_count = 0

        def copy_then_pause(source, target):
            nonlocal copy_count
            result = original_copy(source, target)
            copy_count += 1
            if copy_count == 1:
                ready.write_text(str(os.getpid()), encoding="utf-8")
                time.sleep(args.hold_seconds)
            return result

        delivery._copy_atomic = copy_then_pause
    else:
        original_write = delivery._atomic_write_bytes

        def write_then_pause(path, data):
            result = original_write(path, data)
            if path.name == delivery.MANIFEST_BACKUP_NAME:
                ready.write_text(str(os.getpid()), encoding="utf-8")
                time.sleep(args.hold_seconds)
            return result

        delivery._atomic_write_bytes = write_then_pause

    delivery.accept_long_video_candidate(args.candidate, True)
    raise RuntimeError(f"Parent did not terminate worker at {args.break_at}")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    accept = subparsers.add_parser("accept")
    accept.add_argument("--output-dir", required=True)
    accept.add_argument("--candidate", required=True)
    accept.add_argument("--start", required=True)
    accept.add_argument("--ready", required=True)
    accept.add_argument("--result", required=True)

    hold = subparsers.add_parser("hold")
    hold.add_argument("--root", required=True)
    hold.add_argument("--ready", required=True)
    hold.add_argument("--hold-seconds", type=float, default=60.0)

    counter = subparsers.add_parser("counter")
    counter.add_argument("--root", required=True)
    counter.add_argument("--counter", required=True)
    counter.add_argument("--start", required=True)
    counter.add_argument("--ready", required=True)
    counter.add_argument("--result", required=True)
    counter.add_argument("--iterations", type=int, required=True)

    killpoint = subparsers.add_parser("accept-killpoint")
    killpoint.add_argument("--output-dir", required=True)
    killpoint.add_argument("--candidate", required=True)
    killpoint.add_argument("--ready", required=True)
    killpoint.add_argument(
        "--break-at",
        choices=("after-video-copy", "after-backup-write"),
        required=True,
    )
    killpoint.add_argument("--hold-seconds", type=float, default=60.0)

    args = parser.parse_args()
    _load_package()
    if args.mode == "accept":
        return _accept(args)
    if args.mode == "hold":
        return _hold(args)
    if args.mode == "accept-killpoint":
        return _accept_killpoint(args)
    return _counter(args)


if __name__ == "__main__":
    raise SystemExit(main())
