"""Decoder-only owned process; no engine imports or audio decoding in ComfyUI."""
from __future__ import annotations

import os
from pathlib import Path
import time
import uuid

from . import contract as c
from ..freevideo_exp.runtime import canonical, digest, tensor_record, write_json_new
from ..freevideo_exp.supervision import owned_process


def environment(config, mode):
    if mode not in ('eager', 'compile'):
        raise ValueError('Choose eager or explicit compile; no automatic fallback')
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'FREEVIDEO_RUNTIME_LOCK_FD',
            'FREEVIDEO_RUNTIME_LOCK_HANDLE', 'FREEVIDEO_LOCK_PATH', 'FREEVIDEO_MODEL_ROOT'):
        env.pop(key, None)
    cache = Path(config['home']) / mode
    env.update(FREEVIDEO_HOME=config['home'], FREEVIDEO_VDN_ROOT=config['vdn_root'],
        FREEVIDEO_VAE_COMPILE='1' if mode == 'compile' else '0',
        TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), TRITON_CACHE_DIR=str(cache / 'triton'),
        TORCHINDUCTOR_COMPILE_THREADS='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    return env


def accept_output(stage, family, binding, request, result, frames):
    """Validate exact output clock and unchanged source before returning any result."""
    import numpy as np
    import torch
    _, current = c.stage_input(stage, family)
    geometry = binding['geometry']
    shape = (geometry['frames'], geometry['height'], geometry['width'], 3)
    if current != binding or request['binding'] != binding:
        raise ValueError('Completed source Stage changed while decoding')
    if (result.get('schema') != 't8-freevideo-decoder-result-v1'
            or result.get('binding') != binding or result.get('mode') != request['mode']
            or result.get('compiled_blocks') is not (request['mode'] == 'compile')
            or result.get('source_inventory_sha256') != request['inventory_sha256']
            or type(result.get('sampling_NFE_added')) is not int or result['sampling_NFE_added'] != 0
            or result.get('audio_sent_to_worker') is not False
            or result.get('decoder_blocks') != 36 or result.get('geometry') != geometry
            or result.get('RGB_shape') != list(shape) or result.get('RGB_dtype') != 'uint8'
            or result.get('normalization') != c.NORMALIZATION):
        raise ValueError('Decoder completion contract does not match its actual request')
    if not isinstance(frames, np.ndarray) or frames.dtype != np.uint8 or frames.shape != shape:
        raise ValueError('Decoder RGB bytes/geometry mismatch')
    images = torch.from_numpy(frames.copy()).to(torch.float32).div_(255.)
    audio = {'samples': stage.audio.detach().cpu().clone()}
    if tensor_record(audio['samples']) != binding['audio']:
        raise ValueError('Decoder must pass through the original normalized audio latent')
    return images, audio


def decode(stage, config_path, family, mode='eager', *, interrupt=None, progress=None):
    import numpy as np
    from safetensors.torch import save_file
    video, binding = c.stage_input(stage, family)
    if not isinstance(config_path, str) or not config_path.strip():
        raise ValueError('Prepare and select the independent decoder_config JSON explicitly')
    path = Path(config_path).resolve(strict=True)
    config_sha = digest(path)
    config = c.config_for(path, full=True)
    env = environment(config, mode)
    run = Path(config['home']) / 'decoder-runs' / uuid.uuid4().hex
    run.mkdir(parents=True, exist_ok=False)
    # Never transfer audio, text, prompt/effects, a model or pickle to this child.
    save_file({'video': video.detach().cpu().contiguous()}, str(run / 'input.safetensors'))
    request = dict(schema='t8-freevideo-decoder-request-v1', config_path=str(path), config_sha256=config_sha,
        input_sha256=digest(run / 'input.safetensors'), binding=binding, mode=mode,
        inventory_sha256=config['inventory_sha256'], sampling_NFE_added=0, audio_sent_to_worker=False)
    write_json_new(run / 'request.json', request)
    command = [config['python'], '-I', '-B', '-X', 'utf8', str(Path(__file__).with_name('worker.py')), str(run)]
    with (run / 'worker.log').open('x', encoding='utf8') as log, owned_process(command, env, run, log) as process:
        while process.poll() is None:
            if interrupt:
                interrupt()
            if progress and (run / 'progress.json').is_file():
                try:
                    progress(c.read_json(run / 'progress.json')['completed'], 2)
                except (OSError, ValueError, KeyError):
                    pass
            time.sleep(.2)
        if process.returncode:
            raise RuntimeError(f'Decoder worker failed; original Stage unchanged, evidence at {run}\n'
                + (run / 'worker.log').read_text(encoding='utf8')[-5000:])
    # The own Job and compiler children are cleaned up before acceptance.
    if digest(path) != config_sha or c.config_for(path, full=True) != config:
        raise ValueError('Decoder config changed during execution')
    result = c.read_json(run / 'result.json')
    if (c.read_json(run / 'request.json') != request or result.get('request_sha256') != digest(run / 'request.json')
            or digest(run / 'input.safetensors') != request['input_sha256']
            or result.get('RGB_file_sha256') != digest(run / 'rgb.npy')):
        raise ValueError('Decoder request/input/RGB receipt mismatch')
    with (run / 'rgb.npy').open('rb') as stream:
        frames = np.load(stream, allow_pickle=False)
    images, audio = accept_output(stage, family, binding, request, result, frames)
    report = dict(result, run_directory=str(run), config_sha256=config_sha,
        audio_latent_unchanged=True, audio_decoded_by_this_worker=False,
        original_Stage_modified=False, human_quality_accepted=False)
    write_json_new(run / 'accepted.json', report)
    if progress:
        progress(2, 2)
    return images, audio, canonical(report)
