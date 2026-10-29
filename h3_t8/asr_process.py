"""Cancelable CPU ASR with a task-owned process and reusable model lifetime.

No thread timeout pretending that native inference has stopped. Cancellation
kills and reaps only this child. The normal path has no fixed time limit.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import uuid

import numpy as np


WORKER = Path(__file__).with_name('asr_process_worker.py')


class ASRProcess:
    def __init__(self, model_directory, cpu_threads, cancel_check=lambda: None):
        self.cpu_threads = int(cpu_threads)
        self.cancel_check = cancel_check
        self._temporary = tempfile.TemporaryDirectory(prefix='t8-asr-')
        self.root = Path(self._temporary.name).resolve()
        self._log = (self.root / 'worker.log').open('w+b')
        self.process = None
        try:
            self.process = subprocess.Popen(
                [sys.executable, '-I', '-u', str(WORKER.resolve()), str(self.root),
                 str(Path(model_directory).resolve()), str(self.cpu_threads), str(os.getpid())],
                stdin=subprocess.DEVNULL, stdout=self._log, stderr=self._log,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except BaseException:
            self.close()
            raise

    @property
    def alive(self):
        return self.process is not None and self.process.poll() is None

    def _wait(self, path):
        while True:
            self.cancel_check()
            if path.exists():
                result = json.loads(path.read_text(encoding='utf-8'))
                if result.get('ok') is not True:
                    raise RuntimeError('ASR worker failed: ' + str(result.get('error', 'missing result')))
                return result
            fatal = self.root / 'fatal.json'
            if fatal.exists():
                raise RuntimeError('ASR worker failed: ' + str(json.loads(fatal.read_text(encoding='utf-8')).get('error')))
            if not self.alive:
                self._log.seek(0, os.SEEK_END)
                self._log.seek(max(0, self._log.tell() - 4000))
                raise RuntimeError(f'ASR worker exited ({self.process.returncode}): ' +
                                   self._log.read().decode('utf-8', errors='replace'))
            time.sleep(.025)

    def transcribe(self, array, **options):
        try:
            self._wait(self.root / 'ready.json')
            self.cancel_check()
            token = uuid.uuid4().hex
            source = self.root / (token + '.npy')
            np.save(source, array, allow_pickle=False)
            request = {'id': token, 'audio_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                       'options': options}
            staged = self.root / 'request.pending'
            staged.write_text(json.dumps(request, allow_nan=False), encoding='utf-8')
            os.replace(staged, self.root / 'request.json')
            result = self._wait(self.root / (token + '.json'))
            self.cancel_check()
            if result.get('id') != token or result.get('audio_sha256') != request['audio_sha256']:
                raise ValueError('ASR worker result belongs to a different input')
            segments = [SimpleNamespace(text=segment['text'], words=[SimpleNamespace(**word)
                        for word in segment['words']]) for segment in result['segments']]
            info = SimpleNamespace(**result['info'])
            source.unlink()
            (self.root / (token + '.json')).unlink()
            return iter(segments), info
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
        if not self._log.closed:
            self._log.close()
        # Only the exact TemporaryDirectory created above is ever removed.
        if self.root == Path(self._temporary.name).resolve() and not self.root.is_symlink():
            self._temporary.cleanup()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass  # Module globals can already be cleared during interpreter exit.
