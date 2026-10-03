"""Bound Director media inspection to disposable CPU-only Python workers.

Native decoder faults cannot be caught by a Python try/except in Core. Each
operation gets its own process; failure is explicit and is never retried or
treated as a successful receipt. This does not claim to repair FFmpeg itself.
"""
from __future__ import annotations

import json
from fractions import Fraction
import re
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading


WORKER_PATH = Path(__file__).with_name('director_media_worker.py')
MAX_JSON = 65536
_WORKERS = threading.BoundedSemaphore(2)


class MediaProcessError(ValueError):
    """The media check failed; no successful output or receipt may be inferred."""


def _source(path):
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError('媒体路径必须是本机绝对文件路径')
    value = Path(path)
    if not value.is_absolute() or not value.is_file():
        raise ValueError('媒体文件不存在或不是本机绝对路径')
    return value.resolve()


def _request(operation, path, options):
    if not isinstance(operation, str) or operation not in {'inspect', 'batch', 'metadata', 'validate', 'frame', 'source_timing', 'external', 'external_tail'}:
        raise ValueError('不支持的媒体检查操作')
    expected = {'kind'} if operation == 'validate' else {'frame_number', 'output_png'} if operation == 'frame' else set()
    if operation == 'external_tail':
        expected = {'start_frame', 'end_frame', 'include_audio', 'output_npz'}
    if set(options) != expected:
        raise ValueError('媒体检查参数不完整或包含未知字段')
    source = _source(path)
    if operation == 'external_tail':
        start, end = options['start_frame'], options['end_frame']
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= 5_000_000
                or end-start not in (5, 22, 39) or type(options['include_audio']) is not bool):
            raise ValueError('尾部须精确5/22/39整数帧窗口和显式音频政策')
        output = Path(options['output_npz'])
        if (not output.is_absolute() or output.suffix.lower() != '.npz' or not output.parent.is_dir()
                or output.exists() or output.is_symlink()):
            raise ValueError('尾部输出必须是独立的新绝对NPZ路径')
        options = {**options, 'output_npz': str(output)}
    if operation == 'validate' and options['kind'] not in ('audio', 'video'):
        raise ValueError('素材真实种类不符')
    if operation == 'frame':
        number = options['frame_number']
        if type(number) is not int or not 0 <= number <= 2**53 - 1:
            raise ValueError('请选择非负整数帧')
        if not isinstance(options['output_png'], (str, os.PathLike)):
            raise ValueError('取帧临时输出必须是独立的新 PNG 文件')
        output = Path(options['output_png'])
        if not output.is_absolute() or output.suffix.lower() != '.png' or not output.parent.is_dir() or output.exists():
            raise ValueError('取帧临时输出必须是独立的新 PNG 文件')
        options = {**options, 'output_png': str(output)}
    return {'op': operation, 'path': str(source), **options}


def _external_clock(value):
    validate_result = _result
    if not isinstance(value, dict) or set(value) != {'media', 'source_timing', 'decoded_clock'}:
        raise ValueError('外片解码证据字段不完整')
    media = validate_result('inspect', value['media'])
    headers = validate_result('source_timing', value['source_timing'])
    clock = value['decoded_clock']
    fields = {'fully_decoded', 'video', 'audio', 'normalization_performed', 'zero_origin_film_compatible'}
    if (not isinstance(clock, dict) or set(clock) != fields or clock['fully_decoded'] is not True
            or clock['normalization_performed'] is not False):
        raise ValueError('外片必须完整解码，不能声明未执行的时间转换')
    videos = [row for row in headers['streams'] if row['kind'] == 'video']
    audios = [row for row in headers['streams'] if row['kind'] == 'audio']
    if len(videos) != 1 or len(audios) != int(media['has_audio']):
        raise ValueError('外片实测视频/音轨选择不唯一')

    def rational(raw, *, positive=False):
        if (not isinstance(raw, dict) or set(raw) != {'num', 'den'}
                or any(type(raw[key]) is not int for key in raw)
                or not abs(raw['num']) <= 2**53 - 1 or not 0 < raw['den'] <= 2**53 - 1):
            raise ValueError('外片实测有理数坐标无效')
        result = Fraction(raw['num'], raw['den'])
        if positive and result <= 0:
            raise ValueError('外片实测帧率/timebase必须正数')
        if (result.numerator, result.denominator) != (raw['num'], raw['den']):
            raise ValueError('外片有理数须使用约分后的唯一表示')
        return result

    def digest(raw):
        if not isinstance(raw, str) or not re.fullmatch('[0-9a-f]{64}', raw):
            raise ValueError('外片实测PTS缺少完整SHA')

    video = clock['video']
    if (not isinstance(video, dict) or set(video) != {'stream_index', 'frames', 'fps',
            'first_time', 'last_time', 'end_time', 'grid_tolerance_seconds', 'frame_pts_sha256'}
            or type(video['stream_index']) is not int or video['stream_index'] != videos[0]['index']
            or type(video['frames']) is not int or not 1 <= video['frames'] <= 5_000_000
            or video['frames'] != media['frames']
            or (media['width'], media['height']) != (videos[0]['width'], videos[0]['height'])):
        raise ValueError('外片逐帧证据与媒体轨道矛盾')
    fps = rational(video['fps'], positive=True)
    first, last, end = (rational(video[key]) for key in ('first_time', 'last_time', 'end_time'))
    tolerance = rational(video['grid_tolerance_seconds'], positive=True)
    header_tb = Fraction(videos[0]['timebase']['num'], videos[0]['timebase']['den'])
    if (not math.isfinite(float(fps)) or not 1 <= float(fps) <= 240 or float(fps) != media['fps']
            or tolerance != header_tb / 2 or last < first or end != last + 1 / fps
            or abs(last - first - (video['frames'] - 1) / fps) > tolerance):
        raise ValueError('外片CFR坐标/精度/末端与实测帧数不一致')
    digest(video['frame_pts_sha256'])
    audio = clock['audio']
    if media['has_audio']:
        if (not isinstance(audio, dict) or set(audio) != {'stream_index', 'samples', 'sample_rate',
                'channels', 'first_time', 'end_time', 'audio_pts_sha256'}
                or any(type(audio[key]) is not int for key in ('stream_index', 'samples', 'sample_rate', 'channels'))
                or audio['stream_index'] != audios[0]['index']
                or any(audio[key] != media['audio_' + suffix] for key, suffix in
                       (('samples', 'samples'), ('sample_rate', 'rate'), ('channels', 'channels')))
                or audio['sample_rate'] != audios[0]['sample_rate']):
            raise ValueError('外片音轨与解码样本证据不一致')
        afirst, aend = rational(audio['first_time']), rational(audio['end_time'])
        if aend <= afirst or abs(aend - afirst - Fraction(audio['samples'], audio['sample_rate'])) > Fraction(1, audio['sample_rate']):
            raise ValueError('外片音频PTS范围与完整样本数矛盾')
        digest(audio['audio_pts_sha256'])
    elif audio is not None:
        raise ValueError('无音轨外片不能声明音频上下文')
    zero = first == 0 and (audio is None or rational(audio['first_time']) == 0)
    if type(clock['zero_origin_film_compatible']) is not bool or clock['zero_origin_film_compatible'] != zero:
        raise ValueError('外片起点政策与实测偏移矛盾')
    return value


def _result(operation, value):
    if operation == 'external_tail':
        try:
            fields = {'evidence', 'media_sha256', 'frame_interval', 'sample_interval',
                      'frames_shape', 'audio_shape', 'output_sha256', 'audio_format'}
            if type(value) is not dict or set(value) != fields:
                raise ValueError('尾部回执字段不完整')
            evidence = _external_clock(value['evidence'])
            if (type(value['frame_interval']) is not list or len(value['frame_interval']) != 2
                    or any(type(dim) is not int or dim < 0 for key in ('frames_shape', 'audio_shape')
                           for dim in value[key])):
                raise ValueError('尾部窗口/数组维度必须是实际整数')
            start, end = value['frame_interval']
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= evidence['media']['frames']
                    or end-start not in (5, 22, 39)
                    or evidence['decoded_clock']['video']['fps'] != {'num': 24, 'den': 1}
                    or not evidence['decoded_clock']['zero_origin_film_compatible']
                    or value['frames_shape'] != [end-start, evidence['media']['height'], evidence['media']['width'], 3]):
                raise ValueError('尾部图像窗口/几何/时钟无效')
            for key in ('media_sha256', 'output_sha256'):
                if not isinstance(value[key], str) or not re.fullmatch('[0-9a-f]{64}', value[key]):
                    raise ValueError('尾部缺完整内容SHA')
            if value['audio_format'] is None:
                if value['sample_interval'] is not None or value['audio_shape'] != [0, 0]:
                    raise ValueError('Video-only尾部不能包含音频')
            else:
                clock = evidence['decoded_clock']['audio']
                if (value['audio_format'] != 'float32_planar_no_rate_or_channel_change' or not clock
                        or clock['channels'] != 2 or start*clock['sample_rate'] % 24 or end*clock['sample_rate'] % 24):
                    raise ValueError('尾部音频政策/声道/采样边界无效')
                interval = [start*clock['sample_rate']//24, end*clock['sample_rate']//24]
                if (type(value['sample_interval']) is not list or any(type(x) is not int for x in value['sample_interval'])
                        or value['sample_interval'] != interval or interval[1] > clock['samples']
                        or value['audio_shape'] != [2, interval[1]-interval[0]]):
                    raise ValueError('尾部实际PCM样本范围不符')
            return value
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OverflowError) as error:
            raise MediaProcessError('外片尾部回执无效：' + str(error)) from error
    if operation == 'external':
        try:
            return _external_clock(value)
        except (ValueError, KeyError, TypeError, ZeroDivisionError, OverflowError) as error:
            raise MediaProcessError('外片完整解码/时间证据无效：' + str(error)) from error
    schemas = {
        'inspect': {'width', 'height', 'frames', 'fps', 'duration', 'video_codec', 'has_audio', 'audio_samples', 'audio_rate', 'audio_channels'},
        'batch': {'width', 'height', 'frames', 'audio_frames', 'audio_samples', 'audio_sample_rate'},
        'metadata': {'kind', 'width', 'height', 'duration', 'has_audio'},
        'validate': {'valid'}, 'frame': {'width', 'height'},
        'source_timing': {'streams', 'fully_decoded'},
    }
    if not isinstance(value, dict) or set(value) != schemas[operation]:
        raise MediaProcessError('媒体工作进程返回字段不完整，未记录成功')

    def integer(name, minimum=1):
        return type(value.get(name)) is int and value[name] >= minimum

    def number(name):
        try:
            return type(value.get(name)) in (float, int) and math.isfinite(value[name]) and value[name] > 0
        except OverflowError:
            return False

    valid = True
    if operation in {'inspect', 'batch', 'frame'}:
        valid = integer('width') and integer('height')
    if operation in {'inspect', 'batch'}:
        valid = valid and integer('frames')
    if operation == 'inspect':
        valid = (valid and number('fps') and 1 <= value['fps'] <= 240 and number('duration')
                 and isinstance(value['video_codec'], str) and bool(value['video_codec'])
                 and type(value['has_audio']) is bool and integer('audio_samples', 0) and integer('audio_channels', 0))
        if value['has_audio']:
            valid = valid and integer('audio_rate') and integer('audio_samples') and integer('audio_channels')
        else:
            valid = valid and value['audio_rate'] is None and value['audio_samples'] == value['audio_channels'] == 0
    elif operation == 'batch':
        valid = valid and all(integer(key) for key in ('audio_frames', 'audio_samples', 'audio_sample_rate'))
    elif operation == 'metadata':
        valid = (value['kind'] in ('audio', 'video') and integer('width', 0) and integer('height', 0)
                 and number('duration') and type(value['has_audio']) is bool)
        if value['kind'] == 'video':
            valid = valid and integer('width') and integer('height')
        elif value['kind'] == 'audio':
            valid = valid and value['width'] == value['height'] == 0 and value['has_audio'] is True
    elif operation == 'validate':
        valid = value['valid'] is True
    elif operation == 'source_timing':
        rows = value['streams']
        valid = value['fully_decoded'] is False and isinstance(rows, list) and 1 <= len(rows) <= 64
        indices = []
        if valid:
            for row in rows:
                if (not isinstance(row, dict) or set(row) != {'index', 'kind', 'timebase', 'start_pts', 'duration_pts', 'end_pts', 'frames', 'width', 'height', 'sample_rate'}
                        or type(row['index']) is not int or row['index'] < 0 or row['kind'] not in ('video', 'audio')):
                    valid = False
                    break
                indices.append(row['index'])
                tb = row['timebase']
                if not isinstance(tb, dict) or set(tb) != {'num', 'den'} or not all(type(v) is int and 0 < v <= 2**53-1 for v in tb.values()):
                    valid = False
                    break
                if not all(row[key] is None or (type(row[key]) is int and -(2**53-1) <= row[key] <= 2**53-1) for key in ('start_pts', 'duration_pts', 'end_pts')):
                    valid = False
                    break
                if row['duration_pts'] is not None and row['duration_pts'] <= 0:
                    valid = False
                    break
                if (row['start_pts'] is not None and row['duration_pts'] is not None
                        and row['end_pts'] != row['start_pts'] + row['duration_pts']):
                    valid = False
                    break
                if not all(type(row[key]) is int and row[key] >= 0 for key in ('frames', 'width', 'height')):
                    valid = False
                    break
                if row['kind'] == 'video' and (row['width'] < 1 or row['height'] < 1 or row['sample_rate'] is not None):
                    valid = False
                    break
                if row['kind'] == 'audio' and (type(row['sample_rate']) is not int or row['sample_rate'] <= 0 or row['width'] or row['height']):
                    valid = False
                    break
        valid = valid and len(indices) == len(set(indices))
    if not valid:
        raise MediaProcessError('媒体工作进程返回无效数据，未记录成功')
    return value


def run_media(operation, path, *, timeout=1800, **options):
    """One bounded subprocess, with no shared decoder, Core imports or retries."""
    if type(timeout) not in (int, float) or not 0 < timeout <= threading.TIMEOUT_MAX or not math.isfinite(timeout):
        raise ValueError('媒体检查超时必须是正数')
    request = _request(operation, path, options)
    encoded = json.dumps(request, ensure_ascii=False, allow_nan=False).encode('utf-8')
    if len(encoded) > MAX_JSON:
        raise ValueError('媒体检查参数过大')
    command = [sys.executable, '-I', '-X', 'faulthandler', str(WORKER_PATH.resolve())]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    # -I ignores PYTHONPATH/user-site: the worker imports installed av/PIL only,
    # not a package __init__, ComfyUI, torch, plugin hook or test conftest.
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    if not _WORKERS.acquire(timeout=timeout):
        raise MediaProcessError('媒体检查等待超时，原片保留；没有自动重试')
    try:
        # Regular files also avoid Windows' blocking PIPE stdin write: a child
        # stuck before reading JSON must still obey the process timeout.
        with tempfile.TemporaryFile() as incoming, tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            incoming.write(encoded)
            incoming.seek(0)
            try:
                process = subprocess.Popen(command, stdin=incoming, stdout=output, stderr=errors,
                                           env=env, creationflags=flags)
            except OSError as error:
                raise MediaProcessError('无法启动独立媒体检查，原片保留：' + str(error)) from error
            try:
                process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                process.kill()
                process.communicate()
                raise MediaProcessError('独立媒体检查超时并已结束，原片保留；没有自动重试') from error
            except BaseException:
                if process.poll() is None:
                    process.kill()
                process.communicate()
                raise
            output.seek(0)
            raw = output.read(MAX_JSON + 1)
            errors.seek(0, os.SEEK_END)
            errors.seek(max(0, errors.tell() - 8192))
            diagnostic = errors.read(8192).decode('utf-8', errors='replace').strip()
            try:
                response = json.loads(raw) if len(raw) <= MAX_JSON else None
            except (ValueError, UnicodeError):
                response = None
            if process.returncode != 0:
                detail = response.get('error') if isinstance(response, dict) and response.get('ok') is False else None
                detail = detail if isinstance(detail, str) else diagnostic or '没有完整错误回执'
                raise MediaProcessError(f'独立媒体检查失败（退出码 {process.returncode}）：{detail[:2000]}；原片保留，没有自动重试')
            if not isinstance(response, dict) or response.get('ok') is not True or set(response) != {'ok', 'result'}:
                raise MediaProcessError('独立媒体检查未返回完整成功回执，原片保留')
            return _result(operation, response['result'])
    finally:
        _WORKERS.release()


def inspect_media(path):
    return run_media('inspect', path)


def batch_media(path):
    return run_media('batch', path)


def metadata(path):
    return run_media('metadata', path)


def source_timing(path):
    return run_media('source_timing', path, timeout=60)


def validate_media(path, kind):
    return run_media('validate', path, kind=kind)


def extract_image(path, frame_number):
    from PIL import Image

    with tempfile.TemporaryDirectory(prefix='t8-director-frame-') as temporary:
        output = Path(temporary) / 'frame.png'
        result = run_media('frame', path, frame_number=frame_number, output_png=output)
        if not output.is_file():
            raise MediaProcessError('取帧工作进程未交付 PNG，原片保留')
        try:
            with Image.open(output) as image:
                if image.format != 'PNG' or image.size != (result['width'], result['height']):
                    raise MediaProcessError('取帧 PNG 格式或尺寸与回执不一致')
                image.load()
                return image.copy()
        except OSError as error:
            raise MediaProcessError('取帧 PNG 无法完整读取，未登记成功，原片保留') from error
