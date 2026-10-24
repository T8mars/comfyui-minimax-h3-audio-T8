"""Atomic status publication, including real Windows reader sharing conflicts."""
import ctypes
from ctypes import wintypes
import errno
import json
import os
from pathlib import Path

import pytest

from h3_audio_t8_pkg import director_project as project


def _windows_error(code):
    error = PermissionError(errno.EACCES, 'explicit Windows replace conflict')
    error.winerror = code
    return error


@pytest.mark.parametrize('code', [5, 32, 33])
def test_atomic_json_retries_only_same_complete_temporary_payload(tmp_path, monkeypatch, code):
    target = tmp_path / 'job.json'
    target.write_text('{"state":"running"}', encoding='utf8')
    value = {'state': 'success', 'unicode': '完整原稿🙂'}
    expected = project.canonical(value).encode('utf8')
    replace, calls, delays = project.os.replace, [], []
    failure = _windows_error(code)

    def transient(source, destination):
        calls.append((Path(source), Path(destination), Path(source).read_bytes()))
        assert target.read_text(encoding='utf8') == '{"state":"running"}'
        if len(calls) < 3:
            raise failure
        return replace(source, destination)

    monkeypatch.setattr(project.os, 'replace', transient)
    monkeypatch.setattr(project.time, 'sleep', delays.append)
    project.atomic_json(target, value)
    assert len(calls) == 3 and len(delays) == 2
    assert len({row[0] for row in calls}) == 1
    assert all(row[1] == target and row[2] == expected for row in calls)
    assert target.read_bytes() == expected
    assert not list(tmp_path.glob('*.tmp'))


def test_atomic_json_permanent_windows_denial_is_bounded_and_preserves_original(tmp_path, monkeypatch):
    target = tmp_path / 'job.json'
    original = b'{"state":"running"}'
    target.write_bytes(original)
    failure, calls, delays = _windows_error(5), [], []

    def denied(source, destination):
        calls.append((Path(source), Path(destination), Path(source).read_bytes()))
        raise failure

    monkeypatch.setattr(project.os, 'replace', denied)
    monkeypatch.setattr(project.time, 'sleep', delays.append)
    with pytest.raises(PermissionError) as caught:
        project.atomic_json(target, {'state': 'error'})
    assert caught.value is failure
    assert len(calls) == 6 and len(delays) == 5 and sum(delays) <= 0.25
    assert len({row[0] for row in calls}) == 1
    assert all(row[2] == project.canonical({'state': 'error'}).encode() for row in calls)
    assert target.read_bytes() == original and not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('failure', [PermissionError(errno.EACCES, 'permanent unspecified ACL'),
    OSError(errno.EIO, 'storage error'), _windows_error(2)])
def test_atomic_json_other_errors_are_not_retried_or_hidden(tmp_path, monkeypatch, failure):
    target = tmp_path / 'job.json'
    target.write_text('original', encoding='utf8')
    calls, delays = [], []

    def denied(source, destination):
        calls.append((source, destination))
        raise failure

    monkeypatch.setattr(project.os, 'replace', denied)
    monkeypatch.setattr(project.time, 'sleep', delays.append)
    with pytest.raises(OSError) as caught:
        project.atomic_json(target, {'state': 'success'})
    assert caught.value is failure and len(calls) == 1 and delays == []
    assert target.read_text(encoding='utf8') == 'original' and not list(tmp_path.glob('*.tmp'))


def test_atomic_json_serialization_error_never_replaces_the_original(tmp_path, monkeypatch):
    target = tmp_path / 'job.json'
    target.write_text('original', encoding='utf8')
    calls = []
    monkeypatch.setattr(project.os, 'replace', lambda *args: calls.append(args))
    with pytest.raises(ValueError):
        project.atomic_json(target, {'invalid': float('nan')})
    assert calls == [] and target.read_text(encoding='utf8') == 'original'
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.skipif(os.name != 'nt', reason='Real Windows delete-sharing contract')
def test_atomic_json_real_windows_reader_release_retries_atomic_replace(tmp_path, monkeypatch):
    target = tmp_path / 'job.json'
    target.write_text('{"state":"running"}', encoding='utf8')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    # Match a short Python reader: allow reads/writes but deliberately not DELETE.
    handle = kernel.CreateFileW(str(target), 0x80000000, 1 | 2, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value, ctypes.get_last_error()
    replace, observed, source_bytes = project.os.replace, [], []

    def actual_conflict_then_release(source, destination):
        nonlocal handle
        source_bytes.append(Path(source).read_bytes())
        try:
            return replace(source, destination)
        except OSError as error:
            observed.append(error.winerror)
            assert kernel.CloseHandle(handle)
            handle = None
            raise

    monkeypatch.setattr(project.os, 'replace', actual_conflict_then_release)
    try:
        project.atomic_json(target, {'state': 'success'})
    finally:
        if handle is not None:
            kernel.CloseHandle(handle)
    assert len(observed) == 1 and observed[0] in (5, 32, 33)
    assert source_bytes == [b'{"state":"success"}'] * 2
    assert json.loads(target.read_text(encoding='utf8')) == {'state': 'success'}
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.skipif(os.name != 'nt', reason='Real Windows delete-sharing contract')
def test_atomic_json_real_windows_reader_kept_open_has_finite_failure_without_data_loss(tmp_path, monkeypatch):
    target = tmp_path / 'job.json'
    original = b'{"state":"running"}'
    target.write_bytes(original)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(target), 0x80000000, 1 | 2, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value, ctypes.get_last_error()
    replace, observed, delays = project.os.replace, [], []

    def denied(source, destination):
        try:
            return replace(source, destination)
        except OSError as error:
            observed.append(error)
            raise

    monkeypatch.setattr(project.os, 'replace', denied)
    monkeypatch.setattr(project.time, 'sleep', delays.append)
    try:
        with pytest.raises(PermissionError) as caught:
            project.atomic_json(target, {'state': 'success'})
    finally:
        assert kernel.CloseHandle(handle)
    assert len(observed) == 6 and caught.value is observed[-1]
    assert all(error.winerror in (5, 32, 33) for error in observed)
    assert len(delays) == 5 and sum(delays) <= 0.25
    assert target.read_bytes() == original and not list(tmp_path.glob('*.tmp'))
