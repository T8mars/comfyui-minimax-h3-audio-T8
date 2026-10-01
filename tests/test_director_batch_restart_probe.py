"""Controller contract only: no Core startup, HTTP requests or GPU execution."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


def _tool():
    path = Path(__file__).resolve().parents[1] / "tools/director_batch_restart_probe.py"
    spec = importlib.util.spec_from_file_location("director_restart_probe_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_is_plan_only_and_does_not_start_core():
    module = _tool()
    run = subprocess.run([sys.executable, str(Path(module.__file__))], check=True,
                         capture_output=True, text=True, timeout=30)
    result = json.loads(run.stdout)
    assert result["status"] == "plan_only"
    assert result["gpu_started"] is False


def test_project_is_two_independent_fixed_shots_without_mutating_template():
    from h3_audio_t8_pkg.director_project import new_project

    source = new_project()
    before = deepcopy(source)
    project = _tool().configure_project(source, 1.0, .2)
    assert source == before
    first, second = project["doc"]["shots"]
    assert first["id"] != second["id"]
    assert first["simplePrompt"] != second["simplePrompt"]
    assert project["current"] == first["id"]
    assert first["duration"] == second["duration"] == 1.0
    assert project["doc"]["generation"]["unet"] == "minimax_h3_fl2va_int8_convrot.safetensors"


def test_request_rejects_nonowned_port_before_any_network():
    class NonOwner:
        def assert_port_owner(self):
            raise RuntimeError("not owned")
    with pytest.raises(RuntimeError, match="not owned"):
        _tool().call(NonOwner(), "GET", "/queue")


def test_v2_probe_has_incomplete_unselected_shot_and_exact_seed_order():
    from h3_audio_t8_pkg.director_project import new_project
    from h3_audio_t8_pkg.director_batch import batch_selection
    module = _tool()
    original = module.configure_project(new_project(), 1, .2)
    project, ids, seeds = module.selected_probe_project(original)
    assert len(original['doc']['shots']) == 2 and len(project['doc']['shots']) == 3
    shots, resolved = batch_selection(project, 100, ids, seeds)
    assert [row['id'] for row in shots] == list(reversed(ids))
    assert list(resolved.values()) == [0, 4294967295]
    frozen = {'schema': 't8.minimax_h3.director_batch.v2', 'project': project,
              'items': [{'shot_id': row['id'], 'seed': resolved[row['id']], 'built': {'seed': resolved[row['id']]}} for row in shots]}
    module.check_selection_frozen(frozen, project, seeds)
    frozen['items'][1]['built']['seed'] = 1
    with pytest.raises(RuntimeError, match='exact seed'):
        module.check_selection_frozen(frozen, project, seeds)


def test_cleanup_attempts_all_owned_servers_despite_one_failure():
    seen = []
    class Owned:
        def __init__(self, number):
            self.number = number
            self.stop_receipt = {"pid": number}
        def stop(self):
            seen.append(self.number)
            if self.number == 2:
                raise RuntimeError("owned child still alive")
    records, errors = _tool().stop_servers([Owned(1), Owned(2)])
    assert seen == [2, 1]
    assert records == [{"pid": 2}, {"pid": 1}]
    assert errors == ["RuntimeError: owned child still alive"]


def test_real_core_argument_parser_accepts_shared_directories_and_vhs_whitelist(tmp_path, monkeypatch):
    import comfy.cli_args

    module = _tool()
    monkeypatch.syspath_prepend(str(module.ROOT / "tools"))
    import run_progressive_pilot as pilot

    core = Path(comfy.cli_args.__file__).resolve().parents[1]
    monkeypatch.setattr(pilot, "CORE", core)
    shared = tmp_path / "state"
    for folder in ("user", "output", "input"):
        (shared / folder).mkdir(parents=True)
    command = module.build_server_command(pilot.server_command, shared, tmp_path / "core-1", 8871, False)
    args = comfy.cli_args.parser.parse_args(command[4:])
    assert args.port == 8871
    assert args.whitelist_custom_nodes == [pilot.PROJECT.name, "progressive_probe_extension", "ComfyUI-VideoHelperSuite"]
    assert args.extra_model_paths_config == [[str(tmp_path / "core-1/paths.json")]]
    assert args.user_directory == str(shared / "user")
    assert args.output_directory == str(shared / "output")
    assert args.input_directory == str(shared / "input")
    assert args.disable_comfy_compiler is True
    assert args.use_pytorch_cross_attention is True


def test_fault_probe_restores_original_receipt_when_http_check_fails(tmp_path, monkeypatch):
    module = _tool()
    receipt = tmp_path / "receipt.json"
    original = b'{"fingerprint":"original"}'
    receipt.write_bytes(original)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"owned fake bytes; this test does not decode media")
    media = [{"file": str(video), "sha256": module.digest(video)}]
    def interrupted(*_args, **_kwargs):
        raise TimeoutError("owned request timed out")
    monkeypatch.setattr(module, "call", interrupted)
    with pytest.raises(TimeoutError):
        module.rejection_probes(object(), "batch", receipt, media, tmp_path)
    assert receipt.read_bytes() == original
    assert module.digest(video) == media[0]["sha256"]


@pytest.mark.parametrize("field,value", [("frames", 4), ("width", 64), ("seconds", 2)])
def test_media_delivery_checks_actual_frames_shape_and_audio_not_just_container(tmp_path, field, value):
    from test_director_batch import _write_av

    module = _tool()
    video = tmp_path / "av.mp4"
    _write_av(video)
    time = {"delivery_trim_frames": 3, "requested_seconds": .125}
    canvas = {"width": 32, "height": 32}
    row = {"media": [{"file": video.name, "sha256": module.digest(video)}]}
    valid = {"shot": {"time": time, "canvas": canvas}}
    assert module.verify_media(tmp_path, row, valid)[0]["frames"] == 3
    if field == "frames":
        time["delivery_trim_frames"] = value
    elif field == "width":
        canvas["width"] = value
    else:
        time["requested_seconds"] = value
    with pytest.raises(RuntimeError, match="mismatched AV"):
        module.verify_media(tmp_path, row, valid)
