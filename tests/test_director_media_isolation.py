"""Real subprocess fault containment, independent of Core and model execution.

The repository's pytest conftest may import the plugin/Torch. Separate -I Python
probes below explicitly verify the facade and worker without that root import.
Fake workers are installed only by monkeypatching the private module constant.
To avoid pytest's parent-package setup as well, run this file with
``--noconftest --confcutdir=tests --rootdir=tests --import-mode=importlib``.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pytest


ROOT = Path(__file__).resolve().parents[1]
FACADE = ROOT/'h3_t8/director_media.py'
WORKER = ROOT/'h3_t8/director_media_worker.py'
INSPECT = {'width': 64, 'height': 32, 'frames': 12, 'fps': 24.0, 'duration': .5,
           'video_codec': 'h264', 'has_audio': True, 'audio_samples': 25600,
           'audio_rate': 48000, 'audio_channels': 1}
BATCH = {'width': 64, 'height': 32, 'frames': 12, 'audio_frames': 25,
         'audio_samples': 25600, 'audio_sample_rate': 48000}
METADATA = {'kind': 'video', 'width': 64, 'height': 32, 'duration': .5, 'has_audio': True}


def isolated(script, *arguments, timeout=30):
    return subprocess.run([sys.executable, '-I', '-X', 'faulthandler', '-c', script, *map(str, arguments)],
                          capture_output=True, text=True, timeout=timeout, check=True,
                          creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)


@pytest.fixture
def media_module(monkeypatch):
    name = 'isolated_director_media_'+uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, FACADE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source_media(tmp_path):
    path = tmp_path/'source with spaces.mp4'
    isolated("""
import av, numpy as np, sys
with av.open(sys.argv[1], 'w') as container:
    video=container.add_stream('libx264',rate=24)
    video.width=64;video.height=32;video.pix_fmt='yuv420p'
    video.options={'threads':'1','bf':'0'}
    audio=container.add_stream('aac',rate=48000);audio.layout='mono'
    for number in range(12):
        frame=av.VideoFrame.from_ndarray(np.full((32,64,3),60+number*5,dtype=np.uint8),format='rgb24')
        for packet in video.encode(frame):container.mux(packet)
    for packet in video.encode():container.mux(packet)
    frame=av.AudioFrame.from_ndarray(np.zeros((1,24000),dtype=np.float32),format='fltp',layout='mono')
    frame.sample_rate=48000
    for packet in audio.encode(frame):container.mux(packet)
    for packet in audio.encode():container.mux(packet)
""", path)
    return path


def fake_worker(tmp_path, monkeypatch, module, body):
    path = tmp_path/('worker_'+uuid.uuid4().hex+'.py')
    path.write_text('import json, os, sys, time\nrequest=json.load(sys.stdin)\n'+body+'\n', encoding='utf-8')
    monkeypatch.setattr(module, 'WORKER_PATH', path)
    return path


def reply_worker(tmp_path, monkeypatch, module, result):
    payload = json.dumps({'ok': True, 'result': result})
    return fake_worker(tmp_path, monkeypatch, module, f'print({payload!r})')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_worker_not_reading_stdin_cannot_bypass_timeout(media_module, tmp_path, monkeypatch):
    source = tmp_path/'source.mp4'
    source.write_bytes(b'preserve source')
    worker_path = tmp_path/'not_reading.py'
    worker_path.write_text('import time\ntime.sleep(3)\n', encoding='utf-8')
    monkeypatch.setattr(media_module, 'WORKER_PATH', worker_path)
    # Exercise transport near the wire limit, independently of platform path
    # length limits. A Windows PIPE write can block before communicate's wait.
    monkeypatch.setattr(media_module, '_request', lambda *args: {'padding': 'x'*62000})
    started = time.monotonic()
    with pytest.raises(ValueError):
        media_module.run_media('inspect', source, timeout=.2)
    assert time.monotonic()-started < 2
    assert source.read_bytes() == b'preserve source'


@pytest.mark.parametrize('body', [
    'os._exit(23)',
    pytest.param("""
if os.name == 'nt':
    import ctypes
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel.TerminateProcess(kernel.GetCurrentProcess(), 0xC0000005)
else:
    os._exit(139)
""", id='forced-native-fault-exit-status'),
    'print(json.dumps({"ok":True,"result":{}}),flush=True);os._exit(23)',
    'pass',
    'print("not json")',
    'print("x"*70000)',
    'print("{}\\n{}")',
    'print("[]")',
    'print("null")',
    'print(json.dumps({"ok":"true","result":{}}))',
    'print(json.dumps({"result":{}}))',
    'print(json.dumps({"ok":True,"result":[]}))',
    'print(json.dumps({"ok":True,"result":None}))',
    'print(json.dumps({"ok":False,"error":"controlled decoder failure"}))',
])
def test_abnormal_exit_and_invalid_protocol_fail_closed_parent_survives(media_module, tmp_path, monkeypatch, body):
    source = tmp_path/'untouched.bin'
    source.write_bytes(b'original input bytes')
    before = sha(source)
    parent_pid = __import__('os').getpid()
    fake_worker(tmp_path, monkeypatch, media_module, body)
    with pytest.raises(ValueError):
        media_module.run_media('inspect', source, timeout=5)
    assert __import__('os').getpid() == parent_pid and sha(source) == before
    reply_worker(tmp_path, monkeypatch, media_module, INSPECT)
    assert media_module.inspect_media(source)['frames'] == 12


def test_timeout_reaps_real_child_and_parent_can_run_next_request(media_module, tmp_path, monkeypatch):
    source = tmp_path/'input.bin'
    source.write_bytes(b'unchanged')
    fake_worker(tmp_path, monkeypatch, media_module, 'time.sleep(30)')
    children = []
    real_popen = subprocess.Popen

    def capture(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, 'Popen', capture)
    started = time.monotonic()
    with pytest.raises(ValueError):
        media_module.run_media('inspect', source, timeout=.5)
    assert time.monotonic()-started < 6
    assert len(children) == 1 and children[0].poll() is not None
    reply_worker(tmp_path, monkeypatch, media_module, INSPECT)
    assert media_module.inspect_media(source)['width'] == 64


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf'), True, 'invalid'])
def test_invalid_timeout_rejected_without_starting_process(media_module, tmp_path, monkeypatch, timeout):
    source = tmp_path/'input.bin'
    source.write_bytes(b'input')

    def forbidden(*_args, **_kwargs):
        pytest.fail('invalid timeout must not start a media worker')

    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    with pytest.raises((ValueError, TypeError)):
        media_module.run_media('inspect', source, timeout=timeout)


@pytest.mark.parametrize(('operation', 'options'), [
    ('unknown', {}), ('validate', {'kind': 'image'}),
    ('frame', {'frame_number': True}), ('frame', {'frame_number': -1}),
    ('frame', {'frame_number': 1.5}), ('inspect', {'width': -1}),
])
def test_invalid_operations_or_parameters_cannot_modify_input(media_module, source_media, tmp_path, operation, options):
    before = sha(source_media)
    if operation == 'frame':
        options = {**options, 'output_png': tmp_path/'invalid-frame.png'}
    with pytest.raises((ValueError, TypeError)):
        media_module.run_media(operation, source_media, timeout=5, **options)
    assert sha(source_media) == before


@pytest.mark.parametrize(('method', 'baseline', 'field', 'value'), [
    ('inspect_media', INSPECT, 'width', 0),
    ('inspect_media', INSPECT, 'height', True),
    ('inspect_media', INSPECT, 'frames', -1),
    ('inspect_media', INSPECT, 'fps', float('nan')),
    ('inspect_media', INSPECT, 'duration', float('inf')),
    pytest.param('inspect_media', INSPECT, 'duration', 10**1000, id='duration-overflow'),
    ('inspect_media', INSPECT, 'audio_rate', 0),
    ('inspect_media', INSPECT, 'has_audio', 'yes'),
    ('batch_media', BATCH, 'audio_samples', -1),
    ('batch_media', BATCH, 'audio_frames', True),
    ('metadata', METADATA, 'width', 0),
    ('metadata', METADATA, 'duration', float('nan')),
    ('metadata', METADATA, 'kind', []),
    ('metadata', METADATA, 'kind', {}),
])
def test_parent_rejects_invalid_operation_schema(media_module, tmp_path, monkeypatch, method, baseline, field, value):
    source = tmp_path/'input.bin'
    source.write_bytes(b'input')
    result = deepcopy(baseline)
    result[field] = value
    reply_worker(tmp_path, monkeypatch, media_module, result)
    with pytest.raises(ValueError):
        getattr(media_module, method)(source)


@pytest.mark.parametrize('result', [{}, {'valid': False}, {'valid': 'true'}])
def test_parent_rejects_unproven_validate_success(media_module, tmp_path, monkeypatch, result):
    source = tmp_path/'input.bin'
    source.write_bytes(b'input')
    reply_worker(tmp_path, monkeypatch, media_module, result)
    with pytest.raises(ValueError):
        media_module.validate_media(source, 'video')


@pytest.mark.parametrize(('method', 'result'), [
    ('metadata', {'kind': 'audio', 'width': 0, 'height': 0, 'duration': .5, 'has_audio': True}),
    ('inspect_media', {**INSPECT, 'has_audio': False, 'audio_samples': 0, 'audio_channels': 0, 'audio_rate': None}),
])
def test_audio_only_metadata_and_silent_video_are_valid_contracts(media_module, tmp_path, monkeypatch, method, result):
    source = tmp_path/'input.bin'
    source.write_bytes(b'input')
    reply_worker(tmp_path, monkeypatch, media_module, result)
    assert getattr(media_module, method)(source) == result


def test_busy_worker_slot_times_out_without_launch_and_recovers(media_module, tmp_path, monkeypatch):
    source = tmp_path/'input.bin'
    source.write_bytes(b'input')
    reply_worker(tmp_path, monkeypatch, media_module, INSPECT)
    assert media_module._WORKERS.acquire(blocking=False)
    assert media_module._WORKERS.acquire(blocking=False)
    real_popen = subprocess.Popen

    def forbidden(*_args, **_kwargs):
        pytest.fail('a timed-out semaphore wait must not start a child')

    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    try:
        with pytest.raises(ValueError):
            media_module.run_media('inspect', source, timeout=.1)
    finally:
        media_module._WORKERS.release()
        media_module._WORKERS.release()
    monkeypatch.setattr(subprocess, 'Popen', real_popen)
    assert media_module.inspect_media(source)['frames'] == 12


def test_launch_failure_releases_slot_and_keeps_original(media_module, tmp_path, monkeypatch):
    source = tmp_path/'input.bin'
    source.write_bytes(b'original')
    before = sha(source)
    real_popen = subprocess.Popen

    def unavailable(*_args, **_kwargs):
        raise OSError('controlled executable unavailable')

    monkeypatch.setattr(subprocess, 'Popen', unavailable)
    with pytest.raises(ValueError):
        media_module.run_media('inspect', source)
    monkeypatch.setattr(subprocess, 'Popen', real_popen)
    reply_worker(tmp_path, monkeypatch, media_module, INSPECT)
    assert media_module.inspect_media(source)['frames'] == 12 and sha(source) == before


def test_one_failed_concurrent_request_does_not_poison_the_other(media_module, tmp_path, monkeypatch):
    bad, good = tmp_path/'bad.bin', tmp_path/'good.bin'
    bad.write_bytes(b'bad unchanged')
    good.write_bytes(b'good unchanged')
    before = sha(bad), sha(good)
    fake_worker(tmp_path, monkeypatch, media_module, f"""
from pathlib import Path
if Path(request['path']).name == 'bad.bin':os._exit(23)
time.sleep(.1)
print({json.dumps({'ok': True, 'result': INSPECT})!r})
""")
    with ThreadPoolExecutor(max_workers=2) as pool:
        failed = pool.submit(media_module.inspect_media, bad)
        passed = pool.submit(media_module.inspect_media, good)
        with pytest.raises(ValueError):
            failed.result()
        assert passed.result()['frames'] == 12
    assert (sha(bad), sha(good)) == before


def test_real_worker_rejects_corrupt_input_and_wrong_kind_without_mutation(media_module, source_media, tmp_path):
    corrupt = tmp_path/'corrupt.mp4'
    corrupt.write_bytes(b'not an MP4; preserve these exact bytes')
    before = sha(source_media), sha(corrupt)
    with pytest.raises(ValueError):
        media_module.inspect_media(corrupt)
    with pytest.raises(ValueError):
        media_module.validate_media(source_media, 'audio')
    assert (sha(source_media), sha(corrupt)) == before
    assert media_module.inspect_media(source_media)['frames'] == 12


def test_frame_output_cannot_overwrite_existing_file(media_module, source_media, tmp_path):
    destination = tmp_path/'occupied.png'
    destination.write_bytes(b'original destination')
    before = sha(source_media), sha(destination)
    with pytest.raises(ValueError):
        media_module.run_media('frame', source_media, frame_number=0, output_png=destination)
    assert (sha(source_media), sha(destination)) == before


@pytest.mark.parametrize('case', ['missing', 'unreadable', 'wrong-size', 'wrong-format', 'invalid-dimension'])
def test_frame_return_checks_actual_png_and_declared_dimensions(media_module, tmp_path, monkeypatch, case):
    source = tmp_path/'input.bin'
    source.write_bytes(b'input')
    body = 'from PIL import Image\n'
    if case == 'unreadable':
        body += "from pathlib import Path\nPath(request['output_png']).write_bytes(b'not a PNG')\n"
    elif case != 'missing':
        body += "Image.new('RGB',(64,32),(20,30,40)).save(request['output_png'],format="+repr('JPEG' if case == 'wrong-format' else 'PNG')+')\n'
    dimensions = {'width': 65 if case == 'wrong-size' else 64, 'height': True if case == 'invalid-dimension' else 32}
    body += f'print({json.dumps({"ok": True, "result": dimensions})!r})'
    fake_worker(tmp_path, monkeypatch, media_module, body)
    with pytest.raises(ValueError):
        media_module.extract_image(source, 0)


def test_parallel_frame_requests_get_separate_tempfiles_and_detached_images(media_module, tmp_path, monkeypatch):
    source = tmp_path/'input.bin'
    source.write_bytes(b'input')
    markers = tmp_path/'markers'
    markers.mkdir()
    fake_worker(tmp_path, monkeypatch, media_module, f"""
from pathlib import Path
from PIL import Image
Image.new('RGB',(64,32),(request['frame_number'],30,40)).save(request['output_png'],format='PNG')
(Path({str(markers)!r})/(str(os.getpid())+'.json')).write_text(json.dumps({{'output':request['output_png'],'frame':request['frame_number']}}))
print(json.dumps({{'ok':True,'result':{{'width':64,'height':32}}}}))
""")
    with ThreadPoolExecutor(max_workers=3) as pool:
        images = list(pool.map(lambda number: media_module.extract_image(source, number), [3, 7, 11]))
    records = [json.loads(path.read_text()) for path in markers.glob('*.json')]
    assert len(records) == 3
    assert len({Path(record['output']).parent for record in records}) == 3
    assert all(not Path(record['output']).exists() for record in records)
    assert [image.getpixel((0, 0))[0] for image in images] == [3, 7, 11]
    assert all(image.size == (64, 32) and len(image.tobytes()) == 64*32*3 for image in images)


def test_real_workers_inspect_metadata_validate_batch_frame_and_preserve_source(media_module, source_media):
    before = sha(source_media)
    info = media_module.inspect_media(source_media)
    assert (info['width'], info['height'], info['frames'], info['fps']) == (64, 32, 12, 24)
    assert info['has_audio'] and info['audio_samples'] > 0 and info['audio_rate'] == 48000
    batch = media_module.batch_media(source_media)
    assert batch['frames'] == 12 and batch['audio_samples'] > 0
    metadata = media_module.metadata(source_media)
    assert metadata['kind'] == 'video' and metadata['has_audio'] and metadata['duration'] > 0
    media_module.validate_media(source_media, 'video')
    frames = [media_module.extract_image(source_media, number) for number in [0, 5, 11]]
    assert all(image.size == (64, 32) for image in frames)
    assert len({image.tobytes() for image in frames}) == 3
    assert sha(source_media) == before


def test_real_concurrent_workers_do_not_share_process_or_mutate_media(media_module, source_media, monkeypatch):
    before = sha(source_media)
    children = []
    real_popen = subprocess.Popen

    def capture(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        children.append((process, args, kwargs))
        return process

    monkeypatch.setattr(subprocess, 'Popen', capture)
    with ThreadPoolExecutor(max_workers=3) as pool:
        values = list(pool.map(lambda _number: media_module.inspect_media(source_media), range(6)))
    assert all(value == values[0] for value in values)
    assert len(children) == len({child.pid for child, _args, _kwargs in children}) == 6
    for child, arguments, options in children:
        assert child.poll() == 0
        assert '-I' in arguments[0] and '-X' in arguments[0] and 'faulthandler' in arguments[0]
        assert not options.get('shell', False)
        if sys.platform == 'win32':
            assert options.get('creationflags', 0) & subprocess.CREATE_NO_WINDOW
    assert sha(source_media) == before


def test_standalone_facade_and_worker_do_not_import_torch_or_core(source_media):
    result = isolated("""
import importlib.util,json,sys
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
facade=load('media_facade_probe',sys.argv[1])
result=facade.inspect_media(sys.argv[3])
assert 'av' not in sys.modules,'parent imported native PyAV'
parent_clean=not any(name=='torch' or name.startswith(('torch.','comfy.')) or name in {'comfy','server','nodes','folder_paths'} for name in sys.modules)
worker=load('media_worker_probe',sys.argv[2])
direct=worker.dispatch({'op':'inspect','path':sys.argv[3]})
worker_clean=not any(name=='torch' or name.startswith(('torch.','comfy.')) or name in {'comfy','server','nodes','folder_paths'} for name in sys.modules)
print(json.dumps({'parent_clean':parent_clean,'worker_clean':worker_clean,'frames':result['frames'],'direct_frames':direct['frames']}))
""", FACADE, WORKER, source_media)
    report = json.loads(result.stdout)
    assert report == {'parent_clean': True, 'worker_clean': True, 'frames': 12, 'direct_frames': 12}
