"""Portable combined-node progress: actual tiny Core, not trained/GPU/WS."""
import ast
import asyncio
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import uuid

import comfy.model_management as mm
import comfy.utils
from comfy.ldm.minimax import model as native
from comfy.cli_args import args
import execution
import nodes
import pytest
import torch

from h3_audio_t8_pkg import nodes_hyperflow_advanced as original
from h3_audio_t8_pkg import hyperflow_runtime_advanced as runtime
from h3_audio_t8_pkg.modular_sampling.hyperflow_identity import model_identity
from test_modular_hyperflow import inputs
from h21_progress_worker_fixture import actual_main, FiniteQueue, CapturedServer, OwnedWorkerRouter

# Public implementation is the default. An explicit source is useful for the
# pre-promotion A/B check, but neither an environment variable nor a private
# artifact directory is needed by the installed public regression.
CANDIDATE = Path(os.environ.get("H21_HF_PROGRESS_SOURCE", original.__file__))
if CANDIDATE.resolve() == Path(original.__file__).resolve():
    candidate = original
else:
    spec = importlib.util.spec_from_file_location("h3_audio_t8_pkg._progress_test_candidate", CANDIDATE)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
NEW = candidate.MiniMaxH3HyperFlowSplitT8Advanced


class CallbacklessSplitReference(original.MiniMaxH3HyperFlowSplitT8Advanced):
    """Captured pre-progress execute body, not a second invocation of NEW.

    Schema/import/default preservation is independently pinned by the complete
    outside-body AST below. The reference delegates to the actual unchanged
    sampler with no callback, preserving the old numerical/control boundary.
    """

    @classmethod
    def execute(cls, model_low, model_high, positive, av_latent, seed, split_interval, cfg,
                negative=None, positive_high=None, negative_high=None):
        from h3_audio_t8_pkg.hyperflow_two_pass_advanced import sample_hyperflow_split
        from comfy_api.latest import io
        return io.NodeOutput(*sample_hyperflow_split(
            model_low, model_high, positive, av_latent, seed=seed,
            split_interval=split_interval, cfg=cfg, negative=negative,
            positive_high=positive_high, negative_high=negative_high,
        ))


OLD = CallbacklessSplitReference


def stable_report(text):
    value = json.loads(text)
    del value["stage_seconds"], value["total_seconds"]
    return value


def same_av(left, right):
    assert left.keys() == right.keys()
    assert all(torch.equal(a, b) for a, b in zip(left["samples"].unbind(), right["samples"].unbind(), strict=True))


def run_node(cls, fixture, split):
    low, high, latent, positive = fixture
    return cls.execute(low, high, positive, latent, 7, split, 1.).result


def observe(frame, event, state):
    if event != "call":
        return
    if frame.f_code is comfy.utils.ProgressBar.update_absolute.__code__:
        state["updates"].append((frame.f_locals["value"], frame.f_locals["total"], frame.f_locals["preview"]))
    if frame.f_code is native.MiniMaxH3Model.forward.__code__:
        active = runtime.active_step()
        state["forwards"].append(None if active is None else active.absolute_index)


def direct(cls, fixture, split):
    state = dict(updates=[], forwards=[])
    previous = sys.getprofile()
    assert previous is None
    try:
        sys.setprofile(lambda frame, event, value: observe(frame, event, state))
        result = run_node(cls, fixture, split)
    finally:
        sys.setprofile(previous)
    return result, state


def test_exact_old_node_schema_ABI_and_no_other_runtime_AST_change():
    assert NEW.GET_NODE_INFO_V1() == OLD.GET_NODE_INFO_V1()
    assert inspect.signature(NEW.execute) == inspect.signature(OLD.execute)
    # Captured from the public pre-progress module, with only this execute
    # body masked. A fixed preimage catches schema/default/other-node changes
    # even after public itself is upgraded; comparing NEW with NEW would not.
    tree = ast.parse(CANDIDATE.read_text(encoding="utf8"))
    cls, = [row for row in tree.body if isinstance(row, ast.ClassDef)
            and row.name == "MiniMaxH3HyperFlowSplitT8Advanced"]
    method, = [row for row in cls.body if isinstance(row, ast.FunctionDef) and row.name == "execute"]
    method.body = [ast.Pass()]
    outside = ast.dump(tree, include_attributes=False).encode("utf8")
    assert hashlib.sha256(outside).hexdigest() == "d0d1e8603ba46c2009909d390e158d35c126ea0fae530e57193144501bd9434f"


@pytest.mark.parametrize("split", [1, 4, 7])
def test_real_split_progress_absolute_AV_callbacks_and_identity_unchanged(monkeypatch, split):
    fixture = inputs(monkeypatch, distinct=True)
    low, high, _, _ = fixture
    identities = (model_identity(low), model_identity(high))
    old, old_events = direct(OLD, fixture, split)
    selected, events = direct(NEW, fixture, split)
    plain, next_events = direct(OLD, fixture, split)
    assert events["updates"] == [(value, 8, None) for value in range(1, 9)]
    assert old_events["updates"] == next_events["updates"] == []
    assert events["forwards"] == old_events["forwards"] == next_events["forwards"] == list(range(8))
    same_av(old[0], selected[0])
    same_av(old[0], plain[0])
    assert stable_report(old[1]) == stable_report(selected[1]) == stable_report(plain[1])
    callbacks = stable_report(selected[1])["observed_sampler_callbacks"]
    assert [row["absolute_interval"] for row in callbacks["LOW"]] == list(range(split))
    assert [row["absolute_interval"] for row in callbacks["HIGH"]] == list(range(split, 8))
    assert identities == (model_identity(low), model_identity(high))
    assert runtime.active_step() is runtime._ACTIVE_FORWARD.get() is None
    assert not low.object_patches_backup and not high.object_patches_backup


@pytest.mark.parametrize("failure", [None, "error", "cancel"])
def test_actual_worker_Core_progress_messages_and_failure_nextplain(monkeypatch, tmp_path, failure):
    from app.assets.manager import NoAssets
    main = actual_main(tmp_path)
    fixture = inputs(monkeypatch, distinct=True)
    low, high, _, _ = fixture
    before = (model_identity(low), model_identity(high))
    results, requests = {}, {}
    labels = ("baseline", "selected", "next_plain")
    ids = {label: str(uuid.uuid4()) for label in labels}

    class Source:
        RETURN_TYPES, FUNCTION = ("MODEL", "MODEL", "CONDITIONING", "LATENT"), "get"

        @classmethod
        def INPUT_TYPES(cls):
            return dict(required={})

        def get(self):
            return low, high, fixture[3], fixture[2]

    class Sink:
        RETURN_TYPES, FUNCTION, OUTPUT_NODE = (), "save", True

        @classmethod
        def INPUT_TYPES(cls):
            return dict(required=dict(value=("LATENT",), report=("STRING",), label=("STRING",)))

        def save(self, value, report, label):
            results[label] = (value, report)
            return {"ui": {"text": [report]}, "result": ()}

    for key, cls in (("H21ProgressInput", Source), ("H21ProgressOutput", Sink),
                     ("H21OriginalSplit", OLD), ("H21ProgressSplit", NEW)):
        monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, key, cls)

    class Server(CapturedServer):
        def send_sync(self, event, data, sid=None):
            super().send_sync(event, data, sid)
            if (event == "progress" and data["prompt_id"] == ids["selected"]
                    and failure is not None and data["value"] == 1):
                # Real callback boundary, not a fake model failure. Core's
                # actual sampler must stop and restore even for BaseException.
                if failure == "cancel":
                    raise mm.InterruptProcessingException()
                raise RuntimeError("owned progress callback failure")

    def begin(frame):
        state = dict(updates=[], forwards=[])
        requests[frame.f_locals["prompt_id"]] = state
        return state

    def end(state, frame, returned, aborted):
        state.update(success=frame.f_locals["self"].success, returned_None=returned is None, aborted=aborted)

    router = OwnedWorkerRouter(main.prompt_worker, execution.PromptExecutor.execute,
        begin=begin, observe=lambda state, frame, event, returned: observe(frame, event, state), end=end)
    jobs = []
    for ordinal, label in enumerate(labels):
        graph = {
            "1": dict(class_type="H21ProgressInput", inputs={}),
            "2": dict(class_type="H21ProgressSplit" if label == "selected" else "H21OriginalSplit",
                inputs=dict(model_low=["1", 0], model_high=["1", 1], positive=["1", 2],
                    av_latent=["1", 3], seed=7, split_interval=4, cfg=1.)),
            "3": dict(class_type="H21ProgressOutput", inputs=dict(value=["2", 0], report=["2", 1], label=label)),
        }
        valid = asyncio.run(execution.validate_prompt(ids[label], graph, None))
        assert valid[0], valid
        jobs.append(((ordinal, ids[label], graph, {}, ["3"], {}), ordinal))
    queue, server = FiniteQueue(jobs), Server()
    previous = comfy.utils.PROGRESS_BAR_HOOK
    worker = threading.Thread(target=main.prompt_worker, args=(queue, server, NoAssets(args)), daemon=True)
    try:
        main.hijack_progress(server)
        assert router.arm()
        worker.start()
        worker.join(240)
        assert not worker.is_alive(), "Own actual worker did not finish; no other process stopped"
    finally:
        router.close()
        comfy.utils.set_progress_bar_global_hook(previous)
    assert router.report()["diagnostics"] == [], router.report()
    assert sys.getprofile() is threading.getprofile() is None
    assert len(queue.history) == 3
    assert [row["status"].completed for row in queue.history] == [True, failure is None, True]
    assert [row["status"].status_str for row in queue.history] == ["success", "success" if failure is None else "error", "success"]
    same_av(results["baseline"][0], results["next_plain"][0])
    messages = [data for event, data, _ in server.events if event == "progress"]
    assert messages and all(row["prompt_id"] == ids["selected"] and row["node"] == "2" and row["max"] == 8 for row in messages)
    values = [row["value"] for row in messages]
    assert values == sorted(set(values)) and values[0] == 1
    assert all(1 <= value <= 8 for value in values)
    selected = requests[ids["selected"]]
    assert selected["returned_None"] and selected["success"] is (failure is None)
    assert requests[ids["baseline"]]["updates"] == requests[ids["next_plain"]]["updates"] == []
    assert requests[ids["baseline"]]["forwards"] == requests[ids["next_plain"]]["forwards"] == list(range(8))
    if failure is None:
        assert values[-1] == 8
        assert selected["updates"] == [(value, 8, None) for value in range(1, 9)]
        assert selected["forwards"] == list(range(8))
        same_av(results["baseline"][0], results["selected"][0])
        assert stable_report(results["baseline"][1]) == stable_report(results["selected"][1]) == stable_report(results["next_plain"][1])
    else:
        assert values == [1] and selected["updates"] == [(1, 8, None)]
        assert selected["forwards"] == [0]
        assert "selected" not in results
    assert before == (model_identity(low), model_identity(high))
    assert runtime.active_step() is runtime._ACTIVE_FORWARD.get() is None
    assert not low.object_patches_backup and not high.object_patches_backup
    (tmp_path / "actual-Core-progress.json").write_text(json.dumps(dict(
        messages=messages, requests=requests, actual_network_WebSocket_observed=False,
        Core_throttle_unchanged=True, router=router.report()), indent=2), encoding="utf8")
    assert not torch.cuda.is_initialized()
