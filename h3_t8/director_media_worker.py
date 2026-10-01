"""One-request CPU media worker; standalone entrypoint, never import the project.

Run by absolute file path with ``python -I -X faulthandler``. Native decoder
failure is intentionally not retried or converted into a successful result.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

MAX_REQUEST_BYTES = 64 * 1024


def _input_path(value):
    if not isinstance(value, str) or not value or '\x00' in value or '://' in value:
        raise ValueError('媒体路径必须是本地绝对文件路径，不接受 URI')
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise ValueError('媒体路径必须是已存在的绝对文件路径')
    return path.resolve(strict=True)


def _output_path(value):
    if not isinstance(value, str) or not value or '\x00' in value or '://' in value:
        raise ValueError('PNG 输出必须是绝对文件路径')
    path = Path(value)
    if not path.is_absolute() or path.suffix.lower() != '.png' or not path.parent.is_dir():
        raise ValueError('PNG 输出必须位于已存在目录内，使用绝对 .png 文件路径')
    if path.exists() or path.is_symlink():
        raise ValueError('PNG 输出已存在，拒绝覆盖')
    return path.parent.resolve(strict=True) / path.name


def inspect(path):
    """Exact former director_film.inspect_media decoding and measurements."""
    import av

    with av.open(str(path)) as container:
        video = next(iter(container.streams.video), None)
        if video is None or not video.average_rate:
            raise ValueError('缺少可读取的视频帧率')
        fps = float(video.average_rate)
        if not math.isfinite(fps) or not 1 <= fps <= 240:
            raise ValueError('视频帧率无效')
        video.codec_context.thread_count = 1
        count, first_time, last_time = 0, None, None
        width, height = video.width, video.height
        for frame in container.decode(video):
            if frame.width != width or frame.height != height:
                raise ValueError('视频中途改变画幅')
            if frame.time is not None:
                if last_time is not None and frame.time <= last_time:
                    raise ValueError('视频时间戳不递增')
                if first_time is None:
                    first_time = frame.time
                last_time = frame.time
            count += 1
        if count < 1 or width < 1 or height < 1:
            raise ValueError('视频没有完整可解码画面')
        duration = count / fps
        measured = last_time - first_time + 1 / fps if last_time is not None else duration
        if abs(measured - duration) > 1 / fps + 0.002:
            raise ValueError('暂不支持变帧率成片；请先转换为恒定帧率')
        evidence = {'width': width, 'height': height, 'frames': count, 'fps': fps,
                    'duration': duration, 'video_codec': video.codec_context.name}
    with av.open(str(path)) as container:
        audio = next(iter(container.streams.audio), None)
        samples = 0
        if audio is not None:
            audio.codec_context.thread_count = 1
            samples = sum(frame.samples for frame in container.decode(audio))
            if samples <= 0 or not audio.rate:
                raise ValueError('音频轨道无法完整解码')
        evidence.update(has_audio=audio is not None, audio_samples=samples,
                        audio_rate=audio.rate if audio else None,
                        audio_channels=len(audio.layout.channels) if audio else 0)
    return evidence


def batch(path):
    """Decode mandatory AV; file SHA and delivery comparison stay in parent."""
    import av

    with av.open(str(path)) as container:
        video = next((stream for stream in container.streams if stream.type == 'video'), None)
        if video is None:
            raise ValueError('没有视频轨道')
        width, height = video.width, video.height
        frames = sum(1 for _ in container.decode(video))
        if frames == 0 or width <= 0 or height <= 0:
            raise ValueError('没有完整的可解码视频帧')
    with av.open(str(path)) as container:
        audio = next((stream for stream in container.streams if stream.type == 'audio'), None)
        if audio is None:
            raise ValueError('导演台成片缺少音频轨道')
        audio_frames = 0
        audio_samples = 0
        for frame in container.decode(audio):
            audio_frames += 1
            audio_samples += frame.samples
        if audio_frames == 0:
            raise ValueError('音频轨道无法解码')
        audio_sample_rate = audio.rate
    return {'frames': frames, 'width': width, 'height': height,
            'audio_frames': audio_frames, 'audio_samples': audio_samples,
            'audio_sample_rate': audio_sample_rate}


def metadata(path):
    """Original AV fallback from asset registration, without decoding policy changes."""
    import av

    asset = {'width': 0, 'height': 0}
    with av.open(str(path)) as media:
        video = list(media.streams.video)
        audio = list(media.streams.audio)
        if not video and not audio:
            raise ValueError('素材不是可读取的图片、视频或音频')
        asset['kind'] = 'video' if video else 'audio'
        asset['has_audio'] = bool(audio)
        if video:
            asset.update(width=video[0].width, height=video[0].height)
        stream = (video or audio)[0]
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration
            else float(media.duration or 0) / av.time_base
        )
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('素材时长不可读取，请转为标准 MP4/WAV 后上传')
        asset['duration'] = duration
    return asset


def validate(path, kind):
    """Original bundle validation: decode every audio/video stream, not only first."""
    import av

    with av.open(str(path)) as container:
        video_streams, audio_streams = list(container.streams.video), list(container.streams.audio)
        if (kind == 'audio' and (video_streams or not audio_streams)) or (kind == 'video' and not video_streams):
            raise ValueError('素材真实种类不符')
        streams = video_streams + audio_streams
        for stream in streams:
            stream.codec_context.thread_count = 1
        counts = [0] * len(streams)
        stream_ids = {stream.index: i for i, stream in enumerate(streams)}
        for packet in container.demux(streams):
            counts[stream_ids[packet.stream.index]] += len(packet.decode())
        if not all(counts):
            raise ValueError('素材轨道没有完整可解码内容')
    return {'valid': True}


def frame(path, frame_number, output_png):
    """Original decoded frame selection; write only the new parent-owned PNG."""
    import av

    image = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        if stream.sample_aspect_ratio and float(stream.sample_aspect_ratio) != 1:
            raise ValueError('非方形像素视频请先转换，不能静默改变画面比例')
        if float(stream.metadata.get('rotate', 0)) % 360:
            raise ValueError('带旋转元数据的视频请先转换后取帧')
        for number, decoded in enumerate(container.decode(stream)):
            if getattr(decoded, 'rotation', 0) % 360:
                raise ValueError('带旋转矩阵的视频请先转换后取帧')
            if number == frame_number:
                image = decoded.to_image()
                break
    if image is None:
        raise ValueError('目标帧无法完整解码或尺寸不一致')
    try:
        with output_png.open('xb') as output:
            image.save(output, format='PNG')
        return {'width': image.width, 'height': image.height}
    finally:
        image.close()


_KEYS = {
    'inspect': {'op', 'path'}, 'batch': {'op', 'path'}, 'metadata': {'op', 'path'},
    'validate': {'op', 'path', 'kind'},
    'frame': {'op', 'path', 'frame_number', 'output_png'},
}


def dispatch(request):
    if not isinstance(request, dict):
        raise ValueError('媒体请求必须是 JSON 对象')
    op = request.get('op')
    if not isinstance(op, str) or op not in _KEYS or set(request) != _KEYS[op]:
        raise ValueError('媒体请求操作或字段无效')
    path = _input_path(request['path'])
    if op == 'validate':
        if request['kind'] not in ('audio', 'video'):
            raise ValueError('素材种类无效')
        return validate(path, request['kind'])
    if op == 'frame':
        number = request['frame_number']
        if type(number) is not int or number < 0 or number > 2**53-1:
            raise ValueError('请选择有效非负整数帧号')
        return frame(path, number, _output_path(request['output_png']))
    return {'inspect': inspect, 'batch': batch, 'metadata': metadata}[op](path)


def main():
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError('媒体请求超过 64 KiB')
        request = json.loads(raw.decode('utf-8'))
        result = dispatch(request)
        encoded = json.dumps({'ok': True, 'result': result}, ensure_ascii=False, allow_nan=False)
        code = 0
    except Exception as error:
        encoded = json.dumps({'ok': False, 'error': str(error)}, ensure_ascii=False, allow_nan=False)
        code = 1
    sys.stdout.buffer.write(encoded.encode('utf-8') + b'\n')
    sys.stdout.buffer.flush()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
