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


class Runtime:
    def __init__(self, prompt_id: str):
        self.prompt_id = prompt_id

    def current_prompt_id(self):
        return self.prompt_id

    @staticmethod
    def current_client_id():
        return "multiprocess-client"

    @staticmethod
    def queue_prompt(_prompt, _client_id):
        raise RuntimeError("This worker does not advance segments")

    def prompt_location(self, prompt_id):
        return "running" if prompt_id == self.prompt_id else "missing"

    @staticmethod
    def history_record(_prompt_id):
        return None

    @staticmethod
    def cancel_prompt(prompt_id):
        return {
            "prompt_id": prompt_id,
            "deleted_from_queue": False,
            "interrupt_signalled": False,
        }

    @staticmethod
    def request_release(_release_policy):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chain-id", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--ready", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    args = parser.parse_args()
    _load_package()

    from h3_audio_t8_pkg import long_video_background as background
    from h3_audio_t8_pkg import long_video_delivery as delivery

    output_dir = str(Path(args.output_dir).resolve())
    delivery.folder_paths.get_output_directory = lambda: output_dir
    runtime = Runtime(args.prompt_id)
    manager = background.BackgroundJobManager(runtime, start_monitors=False)
    try:
        state = manager.attach_prompt(
            args.chain_id,
            {"1": {"class_type": "Test", "inputs": {"text": "private prompt body"}}},
            "99",
            0,
            0.0,
            "clear_execution_cache",
        )
        result = {"ok": True, "state": state, "pid": os.getpid()}
    except Exception as error:  # noqa: BLE001 - preserve exact cross-process rejection evidence.
        result = {
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "pid": os.getpid(),
        }
    _write_json(Path(args.result), result)
    Path(args.ready).write_text(str(os.getpid()), encoding="utf-8")
    if result["ok"] and args.hold_seconds > 0:
        time.sleep(args.hold_seconds)
    manager.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
