"""Real Core worker/executor, 124-frame tiny AV; no trained model/server/GPU.

Real Core VAE.encode wraps a miniature native video Encoder; text/audio and
HyperFlow adapter weights are explicitly synthetic. No model function or
module hook is replaced by the private observer.
"""
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import uuid

import comfy.sample
from comfy.cli_args import args as core_args
from comfy.model_patcher import ModelPatcher
from comfy.sd import VAE
import execution
import folder_paths
import nodes
from PIL import Image
import pytest
import torch

from h3_audio_t8_pkg import hyperflow_runtime_advanced as runtime
from h3_audio_t8_pkg import multikeyframe_advanced as multi
from h3_audio_t8_pkg.modular_sampling import hyperflow as stages
from h3_audio_t8_pkg.modular_sampling import hyperflow_effects as effects
from h3_audio_t8_pkg.modular_sampling import hyperflow_fresh as fresh
from h3_audio_t8_pkg.modular_sampling.hyperflow_identity import model_identity
from h3_audio_t8_pkg.nodes_director_keyframes import MiniMaxH3DirectorKeyframeConditioningEXPT8
from helpers import FakeAudioVAE
from test_h21_control_regions import RealTinyEncoderOwner
from test_h21_director_keyframe_runtime import CountingNativeLikeFakeClip
from test_h21_hyperflow_keyframe_identity import real_base
from test_h21_hyperflow_keyframe_runtime import check_original

HERE = Path(__file__).resolve().parent / 'fixtures/h21_observers'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


profile = load("_D_worker_original_profiler", HERE / "observer.py")
adapter = load("_D_worker_actual_adapter", HERE / "worker_adapter.py")
router_module = load("_D_worker_actual_router", HERE / "worker_router.py")


def make_vae():
    # VAE({}) gives the actual Core protocol's original defaults but no model.
    # Install the explicitly labelled tiny architecture through a real CPU
    # ModelPatcher. encode(), crop(), range conversion and transfer stay Core.
    vae = VAE(sd={}, device=torch.device("cpu"), dtype=torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21102195)
        vae.first_stage_model = RealTinyEncoderOwner().core
    vae.device = vae.output_device = torch.device("cpu")
    vae.vae_dtype = torch.float32
    vae.latent_dim, vae.latent_channels = 3, 24
    vae.downscale_ratio = (lambda count: max(1, (count-1)//17*5+2) if count > 1 else 1, 16, 16)
    vae.upscale_ratio = (lambda count: max(1, (count-2)//5*17+5), 16, 16)
    vae.patcher = ModelPatcher(vae.first_stage_model, vae.device, vae.device)
    return vae


def build(base, weights, vae, first, middle, last, *, nonuniform):
    selected = fresh.install(base, weights)[0]
    plan = multi.append_keyframe_plan(None, middle, "frame", 59, .92 if nonuniform else .999, "stretch", True)
    return MiniMaxH3DirectorKeyframeConditioningEXPT8.execute(model=selected,
        clip=CountingNativeLikeFakeClip(), video_vae=vae, audio_vae=FakeAudioVAE(),
        prompt="A person moves.", width=32, height=32, length=124, task_type="FL2VA",
        audio_mode="native", audio_denoise_strength=.35, add_source_as_reference=False,
        prompt_primary_audio_ordinal=0, strict_prompt_tags=True, ref_image_size="match",
        reference_video_policy="official_2_to_15s", first_frame_noise_aug=.999,
        last_frame_noise_aug=.999, reference_visual_noise_aug=.999,
        first_frame=first, last_frame=last, keyframe_plan=plan).result


def sample(low, high, *, cancel=False):
    callbacks = []
    head = effects.bind_head(low[0], low[2], low[1], [], 4)
    before_head = model_identity(head[0])

    def progress(step, _p, _x, total):
        callbacks.append((step, total))
        if cancel and step == 1:
            raise RuntimeError("owned124 actual sampler interruption")

    boundary = stages.sample_head(head[0], low[2], low[1], [],
        comfy.sample.prepare_noise(low[2]["samples"], 21102195), seed=21102195, callback=progress)
    assert model_identity(head[0]) == before_head
    tail = effects.bind_tail(boundary, high[0], high[1], [])
    before_tail = model_identity(tail[0])
    output, report = stages.sample_tail(boundary, tail[0], tail[1], [], seed=21102195, callback=progress)
    assert model_identity(tail[0]) == before_tail
    return output, boundary.verify(), json.loads(report), callbacks, (before_head, before_tail)


class CapturedServer:
    def __init__(self):
        self.client_id = self.last_node_id = self.last_prompt_id = None
        self.workflow_metadata, self.sockets_metadata, self.events = {}, {}, []

    def send_sync(self, event, data, sid=None):
        self.events.append((event, data, sid))


class FiniteQueue:
    def __init__(self, jobs):
        self.jobs, self.history = list(jobs), []

    def get(self, timeout):
        if not self.jobs:
            raise SystemExit(0)
        return self.jobs.pop(0)

    def task_done(self, item_id, history, *, status, process_item):
        self.history.append(dict(item_id=item_id, result=history, status=status))

    def get_tasks_remaining(self):
        return len(self.jobs)

    def get_flags(self):
        return {}


def actual_main(tmp_path):
    # Ordinary import of the actual whole Core file, never AST extraction or
    # substitute worker. Explicit flags avoid all user prestartup/agent hooks.
    core_args.disable_all_custom_nodes = True
    core_args.whitelist_custom_nodes = []
    core_args.cache_none = True
    core_args.enable_agent = core_args.enable_manager = False
    for label in ("input", "output", "temp", "user"):
        path = tmp_path / label
        path.mkdir(exist_ok=True)
        setattr(core_args, label + "_directory", str(path))
        getattr(folder_paths, "set_" + label + "_directory")(str(path))
    old_argv = sys.argv
    sys.argv = [str(Path(folder_paths.__file__).parent / "main.py")]
    try:
        return load("_D_actual_Core_main_worker", Path(folder_paths.__file__).parent / "main.py")
    finally:
        sys.argv = old_argv


def scenario(monkeypatch, tmp_path, *, nonuniform, cancel=False, use_bootstrap=False):
    from comfy.ldm.minimax import model as native
    from app.assets.manager import NoAssets
    main = actual_main(tmp_path)
    base, weights, original = real_base(monkeypatch)
    vae = make_vae()
    result_store, observed = {}, []
    input_root = Path(folder_paths.get_input_directory())
    files, assets = [], {}
    for index, fill in enumerate((185, 127, 65)):
        path = input_root / f"actual-source-{index}.png"
        image = Image.new("RGB", (32, 32), (fill, fill//2, 255-fill))
        image.save(path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(path)
        assets[str(path.resolve())] = dict(sha256=sha, asset_ids=[("first", "middle", "last")[index]])
    deps = SimpleNamespace(native=native, hyperflow=runtime, advanced=multi,
        profiler_base=profile.ConsumptionProfiler, tensor_record=profile.tensor_record,
        module_call_code=torch.nn.Module._call_impl.__code__, load_image_code=nodes.LoadImage.load_image.__code__,
        keyframe_encode_code=multi._encode_keyframe.__code__, vae_encode_code=VAE.encode.__code__,
        actual_frame_index_key=multi.ACTUAL_FRAME_INDEX)

    class ActualTinyOutput:
        RETURN_TYPES, FUNCTION, OUTPUT_NODE = ("LATENT",), "execute", True

        @classmethod
        def INPUT_TYPES(cls):
            return dict(required={"first": ("IMAGE",), "middle": ("IMAGE",), "last": ("IMAGE",),
                "mode": ("STRING",), "cancel": ("BOOLEAN",)})

        def execute(self, first, middle, last, mode, cancel):
            low = build(base, weights, vae, first, middle, last, nonuniform=nonuniform)
            high = build(base, weights, vae, first, middle, last, nonuniform=nonuniform)
            result = sample(low, high, cancel=cancel)
            result_store[mode] = result
            return {"ui": {"text": [json.dumps(dict(portable=result[1]["execution"]["portable_identity"],
                callbacks=result[3], frame_count=124, synthetic_text_audio_and_HF=True))]}, "result": (result[0],)}

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "H21PrivateActual124", ActualTinyOutput)
    selected_id = str(uuid.uuid4())
    selected = dict(name="actual124_observed", middle_frame_index=59, middle_enabled=True,
        middle_asset_sha256=assets[str(files[1].resolve())]["sha256"])

    def begin(frame):
        if frame.f_locals["prompt_id"] != selected_id:
            return None
        return adapter.RequestObservation(selected, selected_id, deps=deps, assets=assets)

    def end(state, frame, returned, aborted):
        report = state.finish(frame, aborted=aborted)
        report["execute_return_is_None"] = returned is None
        observed.append(report)

    router = router_module.OwnedWorkerRouter(main.prompt_worker, execution.PromptExecutor.execute,
        begin=begin, observe=lambda state, frame, event, value: state.event(frame, event, value), end=end)
    jobs, graphs = [], []
    for ordinal, mode in enumerate(("baseline", "observed", "next_plain")):
        graph = {str(i+1): dict(class_type="LoadImage", inputs=dict(image=path.name)) for i, path in enumerate(files)}
        graph["4"] = dict(class_type="H21PrivateActual124", inputs=dict(first=["1", 0], middle=["2", 0],
            last=["3", 0], mode=mode, cancel=bool(cancel and mode == "observed")))
        prompt_id = selected_id if mode == "observed" else str(uuid.uuid4())
        valid = asyncio.run(execution.validate_prompt(prompt_id, graph, None))
        assert valid[0], valid
        jobs.append(((ordinal, prompt_id, graph, {}, ["4"], {}), ordinal))
        graphs.append(graph)
    queue, server = FiniteQueue(jobs), CapturedServer()
    bootstrap_output = None
    if use_bootstrap:
        # Same real modules/worker, plus the real explicit bootstrap selection.
        # This tiny package/manifest is a labelled test fixture, not a full
        # frozen GPU source-root or a replacement for its forthcoming guard.
        bootstrap = load("h3_audio_t8_pkg._h21_D_bootstrap_fixture", HERE / "worker_bootstrap.py")
        for suffix, module in (("profiler", profile), ("router", router_module), ("adapter", adapter)):
            monkeypatch.setitem(sys.modules, "h3_audio_t8_pkg.h21_d_" + suffix, module)
        monkeypatch.setitem(sys.modules, "__main__", main)
        package_file = tmp_path / "private-package/__init__.py"
        package_file.parent.mkdir()
        package_file.write_text("# Explicit tiny bootstrap fixture, not GPU source inventory\n", encoding="utf8")
        prompt_file = tmp_path / "prompts/observed.json"
        prompt_file.parent.mkdir()
        prompt_file.write_text(json.dumps(graphs[1]), encoding="utf8")
        prompt_sha = bootstrap.digest(prompt_file)
        manifest = dict(run_root=str(tmp_path), package_root=str(package_file.parent), python=sys.executable,
            core_root=str(Path(folder_paths.__file__).parent),
            core_files={"main.py": dict(sha256=bootstrap.digest(Path(main.__file__)))},
            cases=[dict(name=selected["name"], prompt="prompts/observed.json")],
            sealed_files={"prompts/observed.json": dict(sha256=prompt_sha)}, resources=dict(assets={
                name: dict(server_path=path.name, sha256=assets[str(path.resolve())]["sha256"])
                for name, path in zip(("first", "middle", "last"), files, strict=True)}))
        bootstrap.write_new(tmp_path / "manifest.json", manifest)
        contract = dict(enabled=True, manifest_sha256=bootstrap.digest(tmp_path / "manifest.json"),
            output_relative="observations/actual", cases=[dict(selected, prompt_sha256=prompt_sha)])
        bootstrap.write_new(tmp_path / "execution-binding.json", dict(
            schema="t8.h21.D.private-execution-binding.v1", activation_contract=contract))
        bootstrap.write_new(tmp_path / "D-worker-observer.json", dict(contract,
            schema="t8.h21.D.private-worker-activation.v1", activation_binding="execution-binding.json",
            activation_binding_sha256=bootstrap.digest(tmp_path / "execution-binding.json")))
        router = bootstrap.install(package_file)
        assert router is not None
        bootstrap_output = tmp_path / "observations/actual"
    hooks = {name: (dict(module._forward_pre_hooks), dict(module._forward_hooks))
             for name, module in base.model.diffusion_model.named_modules()}
    assets_disabled = NoAssets(core_args)
    worker = threading.Thread(target=main.prompt_worker, args=(queue, server, assets_disabled), daemon=True)
    try:
        if not use_bootstrap:
            assert router.arm() is True
        worker.start()
        worker.join(240)
        assert not worker.is_alive(), "Actual tiny worker did not terminate; no user process killed"
    finally:
        router.close()
    assert sys.getprofile() is None and threading.getprofile() is None
    assert router.report()["diagnostics"] == [], router.report()
    if use_bootstrap:
        observed = [json.loads(path.read_bytes()) for path in bootstrap_output.glob("prompt-*.json")]
        status = json.loads((bootstrap_output / "bootstrap.json").read_bytes())
        assert status["activated"] is True
        assert status["actual_consumption_qualified"] is False
    assert len(queue.history) == 3
    assert [row["status"].status_str for row in queue.history] == (["success", "error", "success"] if cancel else ["success"]*3)
    assert len(observed) == 1
    report = observed[0]
    (tmp_path / "actual-worker-report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    if use_bootstrap:
        assert report["actual_request_return_value_is_None"] is True
        assert report["actual_owned_worker_router"]["diagnostics"] == []
        assert report["actual_owned_worker_router"]["worker_ident"] == report["request_thread_ident"]
        assert report["actual_owned_worker_router"]["future_default_restored"] is True
        assert report["actual_process_PID"] == os.getpid()
    else:
        assert report["execute_return_is_None"] is True
    assert report["actual_Core_executor_success_observed"] is (not cancel)
    assert report["actual_sampler_completion_or_trained_GPU_qualified"] is False
    assert report["diagnostics"] == []
    assert report["actual_VAE_encode_entries"] == 6
    assert len(report["actual_keyframe_encodings"]) == 6
    assert all(item["actual_original_VAE_join"] for item in report["actual_keyframe_encodings"])
    assert report["complete_actual_forward_projection_coverage"] is True
    assert report["complete_selected_middle_RGB_latent_projection_joins"] is True
    rows = [row for owner in report["networks"] for row in owner["report"]["records"]]
    assert len(rows) == (2 if cancel else 8)
    assert [row["HF_absolute_interval"] for row in rows] == list(range(len(rows)))
    assert all([item["frame_index"] for item in row["keyframes"]] == [0, 59, 123] for row in rows)
    assert all([item["frame_index"] for item in row["actual_keyframe_projection_segments"]] == [0, 59, 123] for row in rows)
    assert all(row["video_shape"][2] == 37 for row in rows), "Actual124 latent not old22 fixture"
    baseline = result_store["baseline"]
    for mode in ("next_plain",) if cancel else ("observed", "next_plain"):
        actual = result_store[mode]
        assert actual[4] == baseline[4]
        assert actual[1]["execution"] == baseline[1]["execution"]
        assert actual[2]["execution"] == baseline[2]["execution"]
        assert actual[1]["execution"]["portable_identity"] is True
        assert actual[2]["execution"]["portable_identity"] is True
        assert actual[3] == baseline[3] == [(index, 8) for index in range(8)]
        assert all(torch.equal(a, b) for a, b in zip(actual[0]["samples"].unbind(), baseline[0]["samples"].unbind(), strict=True))
    assert hooks == {name: (dict(module._forward_pre_hooks), dict(module._forward_hooks))
                     for name, module in base.model.diffusion_model.named_modules()}
    check_original(base, original)
    assert runtime.active_step() is None and runtime._ACTIVE_FORWARD.get() is None
    assert not torch.cuda.is_initialized()
    return report


@pytest.mark.parametrize("nonuniform", [False, True])
def test_actual_worker124_middle59_encoder_projection_HF_identity_output_and_next_plain(monkeypatch, tmp_path, nonuniform):
    scenario(monkeypatch, tmp_path, nonuniform=nonuniform)


def test_actual_worker124_handled_sampler_error_is_not_success_and_next_plain_recovers(monkeypatch, tmp_path):
    scenario(monkeypatch, tmp_path, nonuniform=False, cancel=True)


def test_actual_bootstrap_routes_only_SHA_selected_prompt_into_real_Core124_worker(monkeypatch, tmp_path):
    scenario(monkeypatch, tmp_path, nonuniform=False, use_bootstrap=True)
