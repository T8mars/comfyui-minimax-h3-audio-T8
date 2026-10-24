"""Portable tiny-test worker fixtures; no service, model files or GPU.

The router is the original stdlib-only owned-thread observation fixture,
relocated from the frozen research tree. It never replaces Core functions or
claims trained-model consumption. The actual whole Core main/worker is loaded
from folder_paths, not a machine-specific or development-artifact path.
"""
import importlib.util
import logging
from pathlib import Path
import sys
import threading

import folder_paths
from comfy.cli_args import args


def actual_main(tmp_path):
    args.disable_all_custom_nodes = True
    args.whitelist_custom_nodes = []
    args.cache_none = True
    args.enable_agent = args.enable_manager = False
    for label in ("input", "output", "temp", "user"):
        path = tmp_path / label
        path.mkdir(exist_ok=True)
        setattr(args, label + "_directory", str(path))
        getattr(folder_paths, "set_" + label + "_directory")(str(path))
    path = Path(folder_paths.__file__).parent / "main.py"
    old_argv = sys.argv
    old_streams = sys.stdout, sys.stderr
    root_logger = logging.getLogger()
    old_level, old_handlers = root_logger.level, tuple(root_logger.handlers)
    sys.argv = [str(path)]
    try:
        spec = importlib.util.spec_from_file_location("_portable_actual_Core_main_worker", path)
        result = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = result
        spec.loader.exec_module(result)
        return result
    finally:
        sys.argv = old_argv
        # Core main imports its real logger setup. Preserve the test process's
        # incoming capture/level rather than leaking INFO into later tests or
        # weakening their no-warning assertions to compensate for this fixture.
        sys.stdout, sys.stderr = old_streams
        root_logger.setLevel(old_level)
        for handler in tuple(root_logger.handlers):
            if handler not in old_handlers:
                root_logger.removeHandler(handler)
                handler.close()


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


class OwnedWorkerRouter:
    def __init__(self, worker, request, *, begin, observe, end):
        self.worker, self.worker_code = worker, worker.__code__
        self.request_code = request.__code__
        self.begin, self.observe, self.end = begin, observe, end
        self.active = None
        self.request_frames = []
        self.worker_ident = None
        self.default_profile = self.current_profile = None
        self.close_requested = False
        self.status = "not_armed"
        self.diagnostics = []
        self.default_dispatch = self._route
        self.worker_dispatch = self._worker_event

    def arm(self):
        if self.status != "not_armed":
            raise RuntimeError("Private profile router may be armed only once")
        self.default_profile = threading.getprofile()
        self.current_profile = sys.getprofile()
        if self.default_profile is not None or self.current_profile is not None:
            self.status = "existing_profiler_retained"
            return False
        self.status = "armed_for_future_owned_worker_only"
        threading.setprofile(self.default_dispatch)
        return True

    def _restore_default_if_owned(self):
        if threading.getprofile() is self.default_dispatch:
            threading.setprofile(self.default_profile)

    def _diagnose(self, where, error):
        self.diagnostics.append(dict(where=where, type=type(error).__name__, message=str(error)))

    def _route(self, frame, event, returned):
        current = threading.current_thread()
        if self.close_requested:
            self._restore_default_if_owned()
            sys.setprofile(self.default_profile)
            return
        actual = getattr(current, "_target", None)
        if actual is not self.worker or getattr(actual, "__code__", None) is not self.worker_code:
            sys.setprofile(self.default_profile)
            return
        if self.worker_ident is not None and self.worker_ident != threading.get_ident():
            self.diagnostics.append(dict(where="route", type="ownership", message="Another thread attempted the same owned worker; not observed"))
            sys.setprofile(self.default_profile)
            return
        self.worker_ident = threading.get_ident()
        self.status = "owned_worker_routed"
        self._restore_default_if_owned()
        sys.setprofile(self.worker_dispatch)
        self._worker_event(frame, event, returned)

    def _finish(self, frame, returned, *, aborted):
        state, self.active = self.active, None
        if state is not None:
            try:
                self.end(state, frame, returned, aborted)
            except Exception as error:
                self._diagnose("request_end", error)

    def _worker_event(self, frame, event, returned):
        if threading.get_ident() != self.worker_ident:
            sys.setprofile(self.default_profile)
            return
        if self.close_requested:
            self._finish(frame, returned, aborted=True)
            self.request_frames.clear()
            if sys.getprofile() is self.worker_dispatch:
                sys.setprofile(self.default_profile)
            self.status = "closed_on_owned_worker"
            return
        if frame.f_code is self.request_code:
            if event == "call":
                self.request_frames.append(frame)
                if len(self.request_frames) > 1:
                    self.diagnostics.append(dict(where="request_begin", type="nested", message="Nested request not conflated with its parent"))
                    return
                try:
                    self.active = self.begin(frame)
                except Exception as error:
                    self._diagnose("request_begin", error)
            elif event == "return":
                if not self.request_frames or self.request_frames[-1] is not frame:
                    self.diagnostics.append(dict(where="request_end", type="coverage", message="Request-frame stack differs; no completion invented"))
                    return
                self.request_frames.pop()
                if not self.request_frames:
                    self._finish(frame, returned, aborted=False)
            return
        if len(self.request_frames) > 1:
            return
        if self.active is not None:
            try:
                self.observe(self.active, frame, event, returned)
            except Exception as error:
                self._diagnose("request_observation", error)
        if frame.f_code is self.worker_code and event == "return":
            self._finish(frame, returned, aborted=True)
            self.request_frames.clear()
            if sys.getprofile() is self.worker_dispatch:
                sys.setprofile(self.default_profile)
            self.status = "owned_worker_returned"

    def close(self):
        """Restore future-thread default now; existing worker at next event.

        No setprofile_all_threads, injected exception or worker wakeup/kill.
        A blocked worker may keep its owned callback until another event or
        process shutdown, which is explicitly not immediate cross-thread reset.
        """
        self.close_requested = True
        self._restore_default_if_owned()
        if self.worker_ident == threading.get_ident() and sys.getprofile() is self.worker_dispatch:
            self._finish(None, None, aborted=True)
            self.request_frames.clear()
            sys.setprofile(self.default_profile)
            self.status = "closed_on_owned_worker"

    def report(self):
        return dict(status=self.status, worker_ident=self.worker_ident, active_request=self.active is not None,
            request_depth=len(self.request_frames), diagnostics=list(self.diagnostics),
            future_default_restored=threading.getprofile() is self.default_profile,
            no_MODEL_or_module_hooks=True, no_function_replacement=True,
            cross_thread_immediate_restore_claimed=False,
            actual_Core_or_consumption_or_GPU_qualified=False)
