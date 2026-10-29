"""Standalone faster-whisper CPU worker. Never imports Core or this package."""
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time


def publish(path, value):
    staged = path.with_suffix('.pending')
    staged.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(staged, path)


def watch_parent(parent):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x00100000, False, parent)
        if not handle:
            os._exit(1)
        try:
            if kernel.WaitForSingleObject(handle, 0xFFFFFFFF) == 0:
                os._exit(0)
        finally:
            kernel.CloseHandle(handle)
    else:
        while os.getppid() == parent:
            time.sleep(.25)
        os._exit(0)


def main():
    root, model_directory, threads, parent = sys.argv[1:]
    root = Path(root).resolve(strict=True)
    threading.Thread(target=watch_parent, args=(int(parent),), daemon=True).start()
    try:
        import numpy as np
        from faster_whisper import WhisperModel
        model = WhisperModel(model_directory, device='cpu', compute_type='int8',
                             cpu_threads=int(threads), local_files_only=True)
        publish(root / 'ready.json', {'ok': True})
        previous = None
        while True:
            request_path = root / 'request.json'
            if not request_path.exists():
                time.sleep(.025)
                continue
            request = json.loads(request_path.read_text(encoding='utf-8'))
            token = request['id']
            if not isinstance(token, str) or len(token) != 32 or any(c not in '0123456789abcdef' for c in token):
                raise ValueError('Invalid owned ASR request ID')
            if token == previous:
                time.sleep(.025)
                continue
            previous = token
            source = root / (token + '.npy')
            if hashlib.sha256(source.read_bytes()).hexdigest() != request['audio_sha256']:
                raise ValueError('ASR input bytes changed before consumption')
            try:
                array = np.load(source, allow_pickle=False)
                segments, info = model.transcribe(array, **request['options'])
                rows = []
                for segment in segments:
                    rows.append({'text': segment.text, 'words': [
                        {'word': word.word, 'start': word.start, 'end': word.end,
                         'probability': getattr(word, 'probability', None)} for word in segment.words or []]})
                publish(root / (token + '.json'), {'ok': True, 'id': token,
                    'audio_sha256': request['audio_sha256'], 'segments': rows,
                    'info': {'language': info.language, 'language_probability': info.language_probability,
                             'duration': info.duration}})
            except Exception as error:
                publish(root / (token + '.json'), {'ok': False, 'id': token, 'error': str(error)})
                return 1
    except Exception as error:
        publish(root / 'fatal.json', {'ok': False, 'error': str(error)})
        return 1


if __name__ == '__main__':
    sys.exit(main())
