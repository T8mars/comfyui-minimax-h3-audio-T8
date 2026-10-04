"""Cancel only this invocation's process tree; launch gate closes spawn races."""
from contextlib import contextmanager
import ctypes
import os
import signal
import subprocess


def file_change_time_ns(path):
    from ctypes import wintypes as w
    class BasicInfo(ctypes.Structure):
        _fields_ = [(n, ctypes.c_int64) for n in ("created", "accessed", "written", "changed")] + [("attributes", w.DWORD)]
    lib = ctypes.WinDLL("kernel32", use_last_error=True)
    lib.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, ctypes.c_void_p, w.DWORD, w.DWORD, w.HANDLE]
    lib.CreateFileW.restype = w.HANDLE
    lib.GetFileInformationByHandleEx.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
    lib.CloseHandle.argtypes = [w.HANDLE]
    handle = lib.CreateFileW(str(path), 0x80, 7, None, 3, 0x02000000, None)
    if handle == ctypes.c_void_p(-1).value:
        return None
    try:
        info = BasicInfo()
        if not lib.GetFileInformationByHandleEx(handle, 0, ctypes.byref(info), ctypes.sizeof(info)):
            return None
        return info.changed * 100
    finally:
        lib.CloseHandle(handle)


class WindowsJob:
    def __init__(self):
        from ctypes import wintypes as w
        class Basic(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", w.DWORD), ("minimum", ctypes.c_size_t), ("maximum", ctypes.c_size_t),
                        ("active_processes", w.DWORD), ("affinity", ctypes.c_size_t),
                        ("priority", w.DWORD), ("scheduling", w.DWORD)]
        class Counters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", Counters), ("process_memory", ctypes.c_size_t),
                        ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
        lib = self.lib = ctypes.WinDLL("kernel32", use_last_error=True)
        lib.CreateJobObjectW.argtypes, lib.CreateJobObjectW.restype = [ctypes.c_void_p, w.LPCWSTR], w.HANDLE
        lib.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        lib.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        lib.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        lib.CloseHandle.argtypes = [w.HANDLE]
        self.handle = lib.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limit = Extended()
        limit.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not lib.SetInformationJobObject(self.handle, 9, ctypes.byref(limit), ctypes.sizeof(limit)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, process):
        if not self.lib.AssignProcessToJobObject(self.handle, process._handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            handle, self.handle = self.handle, None
            if not self.lib.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())


@contextmanager
def owned_process(command, environment, directory, log):
    job, process = None, None
    try:
        if os.name == "nt":
            job = WindowsJob()
        process = subprocess.Popen(command, env=environment, cwd=str(directory), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                                   start_new_session=os.name != "nt")
        if job:
            try:
                job.assign(process)
            except BaseException:
                # This child is not in the Job yet and is still behind the gate.
                process.terminate()
                process.wait(timeout=10)
                raise
        # Worker has not imported engine/Torch or spawned compiler children yet.
        (directory / "launch-gate").touch(exist_ok=False)
        yield process
    finally:
        if job:
            job.close()  # Own Job only; closes remaining compiler children too.
        elif process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process is not None and process.poll() is None:
            process.wait(timeout=10)

