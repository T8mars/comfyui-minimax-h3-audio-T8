"""Fault-injection IPC peer. It is not an ASR engine or quality evidence."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

root, mode, _, _ = sys.argv[1:]
root, mode = Path(root), Path(mode).name


def publish(path, value):
    stage = path.with_suffix('.pending')
    stage.write_text(json.dumps(value), encoding='utf-8')
    os.replace(stage, path)


if mode != 'loadhang':
    publish(root / 'ready.json', {'ok': True})
previous = None
while True:
    path = root / 'request.json'
    if mode == 'loadhang' or not path.exists():
        time.sleep(.01)
        continue
    request = json.loads(path.read_text(encoding='utf-8'))
    if request['id'] == previous:
        time.sleep(.01)
        continue
    previous = request['id']
    if mode == 'hang':
        while True:
            time.sleep(.01)
    if mode == 'exit':
        os._exit(7)
    if mode == 'error':
        publish(root / (previous + '.json'), {'ok': False, 'error': 'injected iterator failure'})
        continue
    source = root / (previous + '.npy')
    publish(root / (previous + '.json'), {'ok': True, 'id': previous,
        'audio_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'segments': [{'text': 'hello', 'words': [{'word': 'hello', 'start': .1, 'end': .3,
                                               'probability': .8}]}],
        'info': {'language': 'en', 'language_probability': .9, 'duration': .5}})
