"""Opt-in owned-thread profile routing. Stdlib only; no function replacement.

This is not the Core bootstrap or a consumption certificate. It routes a future
thread only when its actual target object/code is the explicitly supplied owned
worker. Actual Core code/path/SHA, prompt identity, model consumption and all
success qualifications belong to the separate private adapter/guard.
"""
from __future__ import annotations

import sys
import threading


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
        # Keep stable bound-method identities: accessing a method anew would
        # create a different object for an `is` ownership comparison.
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
        # CPython installs threading's default profile inside the new thread's
        # bootstrap before Thread.run. Read that actual thread's selected
        # target; names or a lookalike function are not an ownership proof.
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
        # Close the default installation window. Other threads that already
        # inherited this callback still immediately take the restore branch.
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
            # Defensive, never attach an accidental second execution thread.
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
            request_depth=len(self.request_frames),
            diagnostics=list(self.diagnostics), future_default_restored=threading.getprofile() is self.default_profile,
            no_MODEL_or_module_hooks=True, no_function_replacement=True,
            cross_thread_immediate_restore_claimed=False,
            actual_Core_or_consumption_or_GPU_qualified=False)
