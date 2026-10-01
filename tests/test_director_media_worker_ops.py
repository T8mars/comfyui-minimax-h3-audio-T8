"""Actual CPU operations; isolated pytest also uses --rootdir=tests --confcutdir=tests --import-mode=importlib --noconftest."""
import hashlib
import importlib.util
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
import wave

import av
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('director_media_worker_ops', ROOT/'h3_t8/director_media_worker.py')
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def video(path, *, audio_tracks=1):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=24)
        stream.width, stream.height, stream.pix_fmt = 64, 32, 'yuv420p'
        stream.options = {'threads': '1', 'bf': '0'}
        audio = [container.add_stream('aac', rate=48000) for _ in range(audio_tracks)]
        for sound in audio:
            sound.layout = 'mono'
        for index in range(12):
            frame = av.VideoFrame.from_image(Image.new('RGB', (64, 32), (50+index, 70, 90)))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        for sound in audio:
            frame = av.AudioFrame(format='fltp', layout='mono', samples=24000)
            frame.sample_rate = 48000
            frame.planes[0].update(bytes(frame.planes[0].buffer_size))
            for packet in sound.encode(frame):
                container.mux(packet)
            for packet in sound.encode():
                container.mux(packet)


@pytest.fixture
def clip(tmp_path):
    path = tmp_path/'source.mp4'
    video(path)
    return path


def test_worker_inspect_batch_metadata_are_exact_and_do_not_modify_source(clip):
    before = digest(clip)
    inspected = worker.dispatch({'op': 'inspect', 'path': str(clip)})
    batched = worker.dispatch({'op': 'batch', 'path': str(clip)})
    metadata = worker.dispatch({'op': 'metadata', 'path': str(clip)})
    with av.open(str(clip)) as container:
        decoded = list(container.decode(audio=0))
        samples, audio_frames = sum(frame.samples for frame in decoded), len(decoded)
    assert samples >= 24000 and audio_frames > 0
    assert inspected == {'width': 64, 'height': 32, 'frames': 12, 'fps': 24.0,
                         'duration': 0.5, 'video_codec': 'h264', 'has_audio': True,
                         'audio_samples': samples, 'audio_rate': 48000, 'audio_channels': 1}
    assert batched == {'width': 64, 'height': 32, 'frames': 12, 'audio_frames': audio_frames,
                       'audio_samples': samples, 'audio_sample_rate': 48000}
    assert metadata == {'kind': 'video', 'width': 64, 'height': 32, 'duration': 0.5, 'has_audio': True}
    assert digest(clip) == before


def test_worker_audio_metadata_and_kind_validation(tmp_path):
    path = tmp_path/'voice.wav'
    with wave.open(str(path), 'wb') as output:
        output.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        output.writeframes(struct.pack('<h', 100)*8000)
    assert worker.dispatch({'op': 'metadata', 'path': str(path)}) == {
        'kind': 'audio', 'width': 0, 'height': 0, 'duration': 0.5, 'has_audio': True}
    assert worker.dispatch({'op': 'validate', 'path': str(path), 'kind': 'audio'}) == {'valid': True}
    for operation in ('inspect', 'batch'):
        with pytest.raises(ValueError):
            worker.dispatch({'op': operation, 'path': str(path)})
    with pytest.raises(ValueError, match='真实种类'):
        worker.dispatch({'op': 'validate', 'path': str(path), 'kind': 'video'})


def test_worker_silent_video_preserves_optional_inspect_mandatory_batch(tmp_path):
    path = tmp_path/'silent.mp4'
    video(path, audio_tracks=0)
    evidence = worker.dispatch({'op': 'inspect', 'path': str(path)})
    assert evidence['has_audio'] is False
    assert (evidence['audio_samples'], evidence['audio_rate'], evidence['audio_channels']) == (0, None, 0)
    with pytest.raises(ValueError, match='缺少音频'):
        worker.dispatch({'op': 'batch', 'path': str(path)})
    assert worker.dispatch({'op': 'validate', 'path': str(path), 'kind': 'video'}) == {'valid': True}


@pytest.mark.parametrize('empty_last', [False, True])
def test_worker_validate_decodes_both_audio_tracks(tmp_path, monkeypatch, empty_last):
    path = tmp_path/'two-audio.mp4'
    video(path, audio_tracks=2)
    before = digest(path)
    original = av.open
    seen = {}

    class Tracked:
        def __init__(self):
            self.container = original(str(path))
            self.streams = self.container.streams

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.container.close()

        def demux(self, streams):
            seen['ids'] = [stream.index for stream in streams]
            for packet in self.container.demux(streams):
                seen[packet.stream.index] = seen.get(packet.stream.index, 0)+1
                yield SimpleNamespace(stream=packet.stream, decode=lambda: []) if empty_last and packet.stream.index == 2 else packet

    monkeypatch.setattr(av, 'open', lambda _: Tracked())
    if empty_last:
        with pytest.raises(ValueError, match='轨道没有完整'):
            worker.dispatch({'op': 'validate', 'path': str(path), 'kind': 'video'})
    else:
        assert worker.dispatch({'op': 'validate', 'path': str(path), 'kind': 'video'}) == {'valid': True}
    assert seen['ids'] == [0, 1, 2]
    assert all(seen[index] > 0 for index in (0, 1, 2))
    assert digest(path) == before


def test_worker_frame_exact_pixels_and_exclusive_output(clip, tmp_path):
    before = digest(clip)
    with av.open(str(clip)) as container:
        expected = [frame.to_image().tobytes() for frame in container.decode(video=0)]
    for index in (0, 5, 11):
        output = tmp_path/f'{index}.png'
        request = {'op': 'frame', 'path': str(clip), 'frame_number': index, 'output_png': str(output)}
        assert worker.dispatch(request) == {'width': 64, 'height': 32}
        with Image.open(output) as image:
            assert image.size == (64, 32)
            assert image.tobytes() == expected[index]
        with pytest.raises(ValueError, match='拒绝覆盖'):
            worker.dispatch(request)
    assert digest(clip) == before
    missing = tmp_path/'missing-frame.png'
    with pytest.raises(ValueError, match='目标帧'):
        worker.dispatch({'op': 'frame', 'path': str(clip), 'frame_number': 12, 'output_png': str(missing)})
    assert not missing.exists()


@pytest.mark.parametrize('op', ['inspect', 'batch', 'metadata', 'validate', 'frame'])
def test_worker_invalid_media_fails_without_touching_source(tmp_path, op):
    path = tmp_path/'broken.mp4'
    path.write_bytes(b'not a media file')
    before = digest(path)
    request = {'op': op, 'path': str(path)}
    if op == 'validate':
        request['kind'] = 'video'
    if op == 'frame':
        request.update(frame_number=0, output_png=str(tmp_path/'bad.png'))
    with pytest.raises((ValueError, av.FFmpegError)):
        worker.dispatch(request)
    assert digest(path) == before
    assert not (tmp_path/'bad.png').exists()


@pytest.mark.parametrize('payload', [None, [], {'op': 'eval'}, {'op': 'inspect'},
                                  {'op': 'inspect', 'path': 'relative.mp4'},
                                  {'op': 'inspect', 'path': 'file:///tmp/a.mp4'},
                                  {'op': 'inspect', 'path': 'https://example.invalid/a.mp4'}])
def test_worker_rejects_untrusted_request_shapes(payload):
    with pytest.raises(ValueError):
        worker.dispatch(payload)


@pytest.mark.parametrize('number', [True, False, -1, 1.5, '1', None, 2**53])
def test_worker_rejects_non_integer_frame_requests(clip, tmp_path, number):
    with pytest.raises(ValueError, match='整数帧号'):
        worker.dispatch({'op': 'frame', 'path': str(clip), 'frame_number': number,
                         'output_png': str(tmp_path/'frame.png')})


def test_worker_module_is_not_a_package_loader():
    source = (ROOT/'h3_t8/director_media_worker.py').read_text(encoding='utf-8')
    assert 'import torch' not in source and 'from .' not in source
    # The joint suite legitimately loads the plugin through its conftest.
    # Prove worker import isolation in the same fresh -I process used at runtime,
    # rather than attributing unrelated modules in this pytest process to it.
    probe = '''
import importlib.util, sys
spec = importlib.util.spec_from_file_location('isolated_worker', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not any(name == 'torch' or name == 'server' or name.startswith('comfy') or name == 'h3_audio_t8_pkg' for name in sys.modules)
print('worker_import_isolated')
'''
    result = subprocess.run([sys.executable, '-I', '-c', probe, str(ROOT/'h3_t8/director_media_worker.py')],
                            check=True, capture_output=True, text=True, timeout=15,
                            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
    assert result.stdout.strip() == 'worker_import_isolated'


@pytest.mark.parametrize('mode', ['sar', 'rotate_tag', 'rotate_matrix'])
def test_worker_frame_retains_no_sar_or_rotation_transform_policy(clip, tmp_path, monkeypatch, mode):
    original = av.open

    class Container:
        def __init__(self):
            self.real = original(str(clip))
            stream = self.real.streams.video[0]
            proxy = SimpleNamespace(codec_context=stream.codec_context,
                                    sample_aspect_ratio=2 if mode == 'sar' else 1,
                                    metadata={'rotate': 90 if mode == 'rotate_tag' else 0})
            self.streams = SimpleNamespace(video=[proxy])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.real.close()

        def decode(self, _stream):
            for decoded in self.real.decode(video=0):
                yield SimpleNamespace(rotation=90, to_image=decoded.to_image)

    monkeypatch.setattr(av, 'open', lambda _: Container())
    output = tmp_path/'frame.png'
    with pytest.raises(ValueError, match='非方形|旋转'):
        worker.dispatch({'op': 'frame', 'path': str(clip), 'frame_number': 0, 'output_png': str(output)})
    assert not output.exists()


@pytest.mark.parametrize('name', ['relative.png', 'wrong.txt', 'missing/frame.png'])
def test_worker_rejects_invalid_output_paths(clip, tmp_path, name):
    output = name if name == 'relative.png' else str(tmp_path/name)
    with pytest.raises(ValueError, match='PNG'):
        worker.dispatch({'op': 'frame', 'path': str(clip), 'frame_number': 0, 'output_png': output})
