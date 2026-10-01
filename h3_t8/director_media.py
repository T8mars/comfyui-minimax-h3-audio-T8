"""Bound Director media inspection to disposable CPU-only Python workers.

Native decoder faults cannot be caught by a Python try/except in Core. Each
operation gets its own process; failure is explicit and is never retried or
treated as a successful receipt. This does not claim to repair FFmpeg itself.
"""
from __future__ import annotations

import json
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
    if not isinstance(operation, str) or operation not in {'inspect', 'batch', 'metadata', 'validate', 'frame'}:
        raise ValueError('不支持的媒体检查操作')
    expected = {'kind'} if operation == 'validate' else {'frame_number', 'output_png'} if operation == 'frame' else set()
    if set(options) != expected:
        raise ValueError('媒体检查参数不完整或包含未知字段')
    source = _source(path)
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


def _result(operation, value):
    schemas = {
        'inspect': {'width', 'height', 'frames', 'fps', 'duration', 'video_codec', 'has_audio', 'audio_samples', 'audio_rate', 'audio_channels'},
        'batch': {'width', 'height', 'frames', 'audio_frames', 'audio_samples', 'audio_sample_rate'},
        'metadata': {'kind', 'width', 'height', 'duration', 'has_audio'},
        'validate': {'valid'}, 'frame': {'width', 'height'},
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
