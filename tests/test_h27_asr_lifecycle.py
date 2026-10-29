"""Real owned processes/IPC and contended locks; no model or GPU needed."""
from pathlib import Path
import time

import numpy as np
import pytest

from h3_audio_t8_pkg import asr_process
from h3_audio_t8_pkg import speech_verification as speech


@pytest.fixture
def peer(monkeypatch):
    monkeypatch.setattr(asr_process, 'WORKER', Path(__file__).parent / 'fixtures/h27_asr_worker.py')


def test_reuse_keeps_same_process_and_close_reaps_it(peer):
    worker = asr_process.ASRProcess('normal', 2)
    pid = worker.process.pid
    try:
        for _ in range(2):
            segments, info = worker.transcribe(np.zeros(8000, dtype=np.float32), language='en')
            assert list(segments)[0].words[0].start == .1
            assert info.language == 'en' and worker.process.pid == pid and worker.alive
    finally:
        worker.close()
    assert worker.process.poll() is not None and not worker.root.exists()
    worker.close()


@pytest.mark.parametrize('mode', ['loadhang', 'hang'])
def test_cancel_during_load_or_native_iterator_reaps_child_and_cleans(peer, mode):
    started = time.monotonic()
    def cancel():
        if time.monotonic() - started > .6:
            raise InterruptedError('explicit cancellation')
    worker = asr_process.ASRProcess(mode, 2, cancel)
    with pytest.raises(InterruptedError, match='explicit cancellation'):
        worker.transcribe(np.zeros(8000, dtype=np.float32), language='en')
    assert time.monotonic() - started < 5
    assert worker.process.poll() is not None and not worker.root.exists()
    # A new task works after cancellation, without borrowing another child.
    following = asr_process.ASRProcess('normal', 2)
    try:
        segments, _ = following.transcribe(np.zeros(8000, dtype=np.float32))
        assert list(segments)[0].text == 'hello'
    finally:
        following.close()


@pytest.mark.parametrize('mode,message', [('error', 'iterator failure'), ('exit', 'exited')])
def test_failure_has_no_success_receipt_and_no_living_worker(peer, mode, message):
    worker = asr_process.ASRProcess(mode, 2)
    with pytest.raises(RuntimeError, match=message):
        worker.transcribe(np.zeros(8000, dtype=np.float32))
    assert worker.process.poll() is not None and not worker.root.exists()


def test_cancel_waiting_for_asr_lock_does_not_release_another_owner():
    speech._ASR_LOCK.acquire()
    started = time.monotonic()
    def cancel():
        if time.monotonic() - started > .1:
            raise InterruptedError('waiting cancelled')
    token = speech._ASR_CANCEL_CHECK.set(cancel)
    try:
        with pytest.raises(InterruptedError, match='waiting cancelled'):
            with speech._asr_guard():
                pytest.fail('must not acquire the busy lock')
        assert speech._ASR_LOCK.locked()
    finally:
        speech._ASR_LOCK.release()
        speech._ASR_CANCEL_CHECK.reset(token)
    with speech._asr_guard():
        assert speech._ASR_LOCK.locked()
    assert not speech._ASR_LOCK.locked()
