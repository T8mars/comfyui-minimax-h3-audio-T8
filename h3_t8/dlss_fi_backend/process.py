"""Windows kill-on-close isolation for the file FI entry point.

No GPU import or automatic execution. A trusted Python task is held at a stdin
gate until assigned to a Windows Job Object. Codec hangs, cancellation, parent
exit and worker grandchildren are bounded by one owned process tree. This is
not yet connected to the public node or a qualification of arbitrary media.
"""
from __future__ import annotations

from collections import deque
import ctypes
from ctypes import wintypes
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


class IsolatedTaskError(RuntimeError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(f"Isolated task {receipt['status']}: {receipt.get('stderr_tail', '')[-1200:]}")


class WindowsJob:
    def __init__(self):
        if os.name != 'nt':
            raise RuntimeError('This isolated FI prototype requires Windows')
        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        api = self.api
        api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        api.CreateJobObjectW.restype = wintypes.HANDLE
        api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        api.SetInformationJobObject.restype = wintypes.BOOL
        api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        api.AssignProcessToJobObject.restype = wintypes.BOOL
        api.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        api.QueryInformationJobObject.restype = wintypes.BOOL
        api.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        api.TerminateJobObject.restype = wintypes.BOOL
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        api.CloseHandle.restype = wintypes.BOOL

        class Basic(ctypes.Structure):
            _fields_ = [('PerProcessUserTimeLimit', ctypes.c_int64), ('PerJobUserTimeLimit', ctypes.c_int64),
                ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD), ('SchedulingClass', wintypes.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ('ReadOperationCount', 'WriteOperationCount',
                'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]

        class Extended(ctypes.Structure):
            _fields_ = [('BasicLimitInformation', Basic), ('IoInfo', IO), ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t), ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]

        self.handle = api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, process):
        if not self.api.AssignProcessToJobObject(self.handle, wintypes.HANDLE(int(process._handle))):
            raise ctypes.WinError(ctypes.get_last_error())

    def active(self):
        class Accounting(ctypes.Structure):
            _fields_ = [(name, ctypes.c_int64) for name in ('TotalUserTime', 'TotalKernelTime',
                'ThisPeriodTotalUserTime', 'ThisPeriodTotalKernelTime')] + [
                (name, wintypes.DWORD) for name in ('TotalPageFaultCount', 'TotalProcesses', 'ActiveProcesses', 'TotalTerminatedProcesses')]
        value = Accounting()
        if not self.api.QueryInformationJobObject(self.handle, 1, ctypes.byref(value), ctypes.sizeof(value), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return value.ActiveProcesses

    def terminate(self):
        if not self.api.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


# -I -S avoids user startup hooks before the gate. -B prevents a trusted
# worker importing siblings from adding __pycache__ to a frozen source tree;
# -I ignores PYTHONDONTWRITEBYTECODE, so an environment flag is insufficient.
# Site packages are enabled
# only after ownership is established; no arbitrary task runs before assignment.
BOOTSTRAP = (
    "import sys; gate=sys.stdin.buffer.readline(); "
    "assert gate==b'T8FI-GO\\n', 'missing ownership gate'; "
    "import site; site.main(); import runpy; "
    "sys.argv=sys.argv[1:]; sys.path.insert(0,__import__('os').path.dirname(sys.argv[0])); "
    "runpy.run_path(sys.argv[0],run_name='__main__')"
)


def _settled_active_processes(job, *, timeout=.25, cancel=None, check=None):
    """Job accounting can lag a signalled process handle by a scheduler tick.

    Wait only briefly, retaining ownership/guards. A live descendant is still
    reported and killed by the existing finally block, never called success.
    """
    deadline = time.monotonic()+timeout
    active = job.active()
    while active and time.monotonic() < deadline:
        if cancel is not None and cancel.is_set():
            raise InterruptedError('Cancelled during process-exit confirmation')
        if check is not None:
            check()
        time.sleep(min(.01,max(0,deadline-time.monotonic())))
        active = job.active()
    return active


def run_isolated(script, arguments=(), *, timeout=180, cancel=None, check=None):
    """Only caller-constructed trusted tasks; no shell, inherited job or UI window."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise ValueError('Use a finite 0..3600 second task deadline')
    script = Path(script).resolve(strict=True)
    if not script.is_file() or script.suffix != '.py' or not isinstance(arguments, (tuple, list)) or any(not isinstance(a, str) for a in arguments):
        raise ValueError('Trusted Python script and explicit argument vector required')
    cancel = cancel if cancel is not None else threading.Event()
    if cancel.is_set():
        raise IsolatedTaskError({'status': 'cancelled_before_start', 'pid': None, 'active_after_cleanup': 0})
    job, process, readers = WindowsJob(), None, []
    stdout, stderr = deque(maxlen=16), deque(maxlen=16)
    receipt = {'status': 'incomplete', 'pid': None, 'job_assigned_before_task': False}
    started = time.monotonic()
    try:
        process = subprocess.Popen([sys.executable, '-I', '-S', '-B', '-u', '-c', BOOTSTRAP, str(script), *arguments],
            cwd=script.parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW, bufsize=0, close_fds=True)
        receipt['pid'] = process.pid
        job.assign(process)
        receipt['job_assigned_before_task'] = True

        def drain(pipe, target):
            try:
                while block := pipe.read(8192):
                    target.append(block)
            except (OSError, ValueError):
                pass

        for pipe, target in ((process.stdout, stdout), (process.stderr, stderr)):
            thread = threading.Thread(target=drain, args=(pipe, target), daemon=True)
            thread.start()
            readers.append(thread)
        process.stdin.write(b'T8FI-GO\n')
        process.stdin.close()
        while process.poll() is None:
            if cancel.is_set():
                receipt['status'] = 'cancelled'
                break
            if time.monotonic()-started >= timeout:
                receipt['status'] = 'timeout'
                break
            if check is not None:
                check()
            time.sleep(.02)
        if receipt['status'] == 'incomplete':
            receipt['status'] = 'complete' if process.returncode == 0 else 'child_failed'
            if _settled_active_processes(job, timeout=min(.25,max(0,timeout-(time.monotonic()-started))),
                                         cancel=cancel, check=check):
                receipt['status'] = 'child_left_descendants'
    except BaseException as error:
        receipt.update(status='controller_failed', error=f'{type(error).__name__}: {error}')
    finally:
        try:
            if job.active():
                job.terminate()
            if process is not None:
                if not receipt['job_assigned_before_task'] and process.poll() is None:
                    process.kill()  # Only our gated process; no task/children started.
                process.wait(timeout=10)
                receipt['exit_code'] = process.returncode
            limit = time.monotonic()+5
            while job.active() and time.monotonic() < limit:
                time.sleep(.02)
            receipt['active_after_cleanup'] = job.active()
            if receipt['active_after_cleanup']:
                receipt['status'] = 'cleanup_failed'
        finally:
            job.close()
            for thread in readers:
                thread.join(timeout=2)
            if process is not None:
                for pipe in (process.stdin, process.stdout, process.stderr):
                    if pipe is not None:
                        pipe.close()
            receipt.update(wall_seconds=time.monotonic()-started,
                stdout_tail=b''.join(stdout).decode('utf8', 'replace'), stderr_tail=b''.join(stderr).decode('utf8', 'replace'))
    if receipt['status'] != 'complete':
        raise IsolatedTaskError(receipt)
    return receipt
