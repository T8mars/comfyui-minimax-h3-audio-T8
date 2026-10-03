"""One-request CPU media worker; standalone entrypoint, never import the project.

Run by absolute file path with ``python -I -X faulthandler``. Native decoder
failure is intentionally not retried or converted into a successful result.
"""
from __future__ import annotations

import json
import hashlib
from fractions import Fraction
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


def source_timing(path):
    """Header PTS/timebases only; not a decoded-frame or factual-quality proof."""
    import av

    rows = []
    with av.open(str(path)) as container:
        for stream in container.streams:
            if stream.type not in ('video', 'audio'):
                continue
            if len(rows) >= 64 or not stream.time_base or stream.time_base <= 0:
                raise ValueError('源媒体轨道过量或缺少有效 timebase')
            start = stream.start_time
            duration = stream.duration
            rows.append({'index': stream.index, 'kind': stream.type,
                         'timebase': {'num': stream.time_base.numerator, 'den': stream.time_base.denominator},
                         'start_pts': start, 'duration_pts': duration,
                         'end_pts': start + duration if start is not None and duration is not None else None,
                         'frames': stream.frames, 'width': stream.width if stream.type == 'video' else 0,
                         'height': stream.height if stream.type == 'video' else 0,
                         'sample_rate': stream.rate if stream.type == 'audio' else None})
    if not rows:
        raise ValueError('没有可用的源视频/音频轨道')
    return {'streams': rows, 'fully_decoded': False}


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


def external(path):
    import av

    def rational(value):
        value = Fraction(value)
        if abs(value.numerator) > 2**53 - 1 or value.denominator > 2**53 - 1:
            raise ValueError('实测时间坐标超出页面可精确表示的整数范围')
        return {'num': value.numerator, 'den': value.denominator}

    headers = source_timing(path)
    videos = [row for row in headers['streams'] if row['kind'] == 'video']
    audios = [row for row in headers['streams'] if row['kind'] == 'audio']
    if len(videos) != 1 or len(audios) > 1:
        raise ValueError('外部take须唯一视频、最多一音轨；请先明确选择轨道并另存，不自动猜轨')
    # Old inspect math/defaults remain untouched. It fully decodes AV; this
    # extra operation verifies every decoded timestamp rather than using only
    # first/last duration as a variable-frame-rate certificate.
    media = inspect(path)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        rate = Fraction(stream.average_rate)
        if not 1 <= float(rate) <= 240 or not math.isfinite(float(rate)):
            raise ValueError('外部take实测帧率无效')
        if stream.sample_aspect_ratio and Fraction(stream.sample_aspect_ratio) != 1:
            raise ValueError('外部take非方形像素，需显式另存转换版，原片保留')
        if float(stream.metadata.get('rotate', 0)) % 360:
            raise ValueError('外部take有旋转元数据，需显式另存转换版，原片保留')
        first = last = tolerance = None
        count, pts_digest = 0, hashlib.sha256()
        for decoded in container.decode(stream):
            if (decoded.pts is None or not decoded.time_base or decoded.time_base <= 0
                    or getattr(decoded, 'rotation', 0) % 360):
                raise ValueError('外部take缺少实测逐帧PTS或带旋转矩阵，不能猜坐标')
            current = decoded.pts * Fraction(decoded.time_base)
            if first is None:
                first = current
                tolerance = Fraction(decoded.time_base) / 2
            if (last is not None and current <= last
                    or abs(current - first - count / rate) > tolerance):
                raise ValueError('逐帧PTS不是实测CFR网格；请显式转换新素材，不静默补帧')
            if (decoded.width, decoded.height) != (media['width'], media['height']):
                raise ValueError('外部take中途改变画幅')
            # A bounded digest, not an unbounded million-frame JSON list.
            pts_digest.update(f'{current.numerator}/{current.denominator}\n'.encode('ascii'))
            last, count = current, count + 1
            if count > 5_000_000:
                raise ValueError('外部take超过五百万帧，请拆分登记')
        if count != media['frames'] or first is None:
            raise ValueError('两次完整解码的帧数不一致，未记录成功')
        video_clock = {'stream_index': stream.index, 'frames': count,
                       'fps': rational(rate), 'first_time': rational(first),
                       'last_time': rational(last), 'end_time': rational(last + 1 / rate),
                       'grid_tolerance_seconds': rational(tolerance),
                       'frame_pts_sha256': pts_digest.hexdigest()}
    audio_clock = None
    if audios:
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            stream.codec_context.thread_count = 1
            first = end = None
            samples, pts_digest = 0, hashlib.sha256()
            for decoded in container.decode(stream):
                if (decoded.pts is None or not decoded.time_base or decoded.time_base <= 0
                        or decoded.sample_rate != media['audio_rate']
                        or len(decoded.layout.channels) != media['audio_channels']):
                    raise ValueError('外部take音频缺PTS或中途改变采样格式')
                current = decoded.pts * Fraction(decoded.time_base)
                if end is not None and abs(current - end) > Fraction(1, decoded.sample_rate):
                    raise ValueError('音频PTS不连续，需显式另存处理，不自动补静音或对齐')
                if first is None:
                    first = current
                end = current + Fraction(decoded.samples, decoded.sample_rate)
                samples += decoded.samples
                pts_digest.update(f'{current.numerator}/{current.denominator}:{decoded.samples}\n'.encode('ascii'))
            if samples != media['audio_samples'] or first is None:
                raise ValueError('两次完整解码的音频样本数不一致')
            audio_clock = {'stream_index': stream.index, 'samples': samples,
                           'sample_rate': media['audio_rate'], 'channels': media['audio_channels'],
                           'first_time': rational(first), 'end_time': rational(end),
                           'audio_pts_sha256': pts_digest.hexdigest()}
    return {'media': media, 'source_timing': headers,
            'decoded_clock': {'fully_decoded': True, 'video': video_clock, 'audio': audio_clock,
                'normalization_performed': False,
                'zero_origin_film_compatible': video_clock['first_time']['num'] == 0
                    and (audio_clock is None or audio_clock['first_time']['num'] == 0)}}


def external_tail(path, start, end, include_audio, output):
    """One bounded integer RGB/PCM window from the completely verified movie."""
    import av
    import numpy as np

    def digest(file):
        value = hashlib.sha256()
        with file.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024**2), b''):
                value.update(chunk)
        return value.hexdigest()

    original_sha = digest(path)
    evidence = external(path)
    media, clock = evidence['media'], evidence['decoded_clock']
    count = end - start
    if (clock['video']['fps'] != {'num': 24, 'den': 1} or not clock['zero_origin_film_compatible']
            or not 0 <= start < end <= media['frames'] or count not in (5, 22, 39)):
        raise ValueError('续拍尾部须为零起点24fps CFR的精确5/22/39帧窗口')
    if count * media['width'] * media['height'] * 3 > 512 * 1024**2:
        raise ValueError('续拍RGB尾部超过512MiB，请显式处理新素材，未自动缩放')
    pixels = []
    with av.open(str(path)) as container:
        container.streams.video[0].codec_context.thread_count = 1
        for index, decoded in enumerate(container.decode(video=0)):
            if start <= index < end:
                pixels.append(decoded.to_ndarray(format='rgb24'))
            if index + 1 == end:
                break
    if len(pixels) != count:
        raise ValueError('未解出完整续拍RGB尾部')
    frames = np.stack(pixels)
    audio = np.empty((0, 0), dtype=np.float32)
    interval = None
    if include_audio:
        audio_clock = clock['audio']
        if not audio_clock or audio_clock['channels'] != 2:
            raise ValueError('续拍音频须实际双声道，未自动复制或混音')
        rate = audio_clock['sample_rate']
        if start * rate % 24 or end * rate % 24:
            raise ValueError('音频窗不是整数原始样本边界，不舍入')
        first, last = start * rate // 24, end * rate // 24
        if last > audio_clock['samples'] or (last-first)*2*4 > 64*1024**2:
            raise ValueError('音频尾部不完整或超过64MiB')
        chunks, offset = [], 0
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            stream.codec_context.thread_count = 1
            converter = av.AudioResampler(format='fltp', layout=stream.layout, rate=rate)
            for decoded in container.decode(stream):
                for converted in converter.resample(decoded):
                    samples = converted.to_ndarray()
                    if samples.dtype != np.float32 or samples.shape != (2, converted.samples):
                        raise ValueError('音频PCM解码布局异常')
                    left, right = max(first-offset, 0), min(last-offset, converted.samples)
                    if left < right:
                        chunks.append(samples[:, left:right].copy())
                    offset += converted.samples
                if offset >= last:
                    break
        audio = np.concatenate(chunks, axis=1) if chunks else audio
        if audio.shape != (2, last-first) or not np.isfinite(audio).all():
            raise ValueError('实际PCM尾部样本数或有限值无效')
        interval = [first, last]
    if digest(path) != original_sha:
        raise ValueError('外片在尾部解码时改变，未交付')
    # Server-created new output only; no object/pickle data and no overwrite.
    with output.open('xb') as handle:
        np.savez(handle, frames=frames, audio=audio)
    return {'evidence': evidence, 'media_sha256': original_sha,
            'frame_interval': [start, end], 'sample_interval': interval,
            'frames_shape': list(frames.shape), 'audio_shape': list(audio.shape),
            'output_sha256': digest(output),
            'audio_format': 'float32_planar_no_rate_or_channel_change' if include_audio else None}


_KEYS = {
    'inspect': {'op', 'path'}, 'batch': {'op', 'path'}, 'metadata': {'op', 'path'},
    'source_timing': {'op', 'path'},
    'external': {'op', 'path'},
    'external_tail': {'op', 'path', 'start_frame', 'end_frame', 'include_audio', 'output_npz'},
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
    if op == 'external_tail':
        start, end, include = (request[key] for key in ('start_frame', 'end_frame', 'include_audio'))
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= 5_000_000
                or end-start not in (5, 22, 39) or type(include) is not bool):
            raise ValueError('外片尾部请求须精确整数帧窗口和显式音频政策')
        output = Path(request['output_npz'])
        if (not output.is_absolute() or output.suffix.lower() != '.npz' or not output.parent.is_dir()
                or output.exists() or output.is_symlink()):
            raise ValueError('尾部输出须已存在目录内的新绝对NPZ路径')
        return external_tail(path, start, end, include, output.parent.resolve(strict=True)/output.name)
    if op == 'validate':
        if request['kind'] not in ('audio', 'video'):
            raise ValueError('素材种类无效')
        return validate(path, request['kind'])
    if op == 'frame':
        number = request['frame_number']
        if type(number) is not int or number < 0 or number > 2**53-1:
            raise ValueError('请选择有效非负整数帧号')
        return frame(path, number, _output_path(request['output_png']))
    return {'inspect': inspect, 'batch': batch, 'metadata': metadata, 'source_timing': source_timing, 'external': external}[op](path)


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
