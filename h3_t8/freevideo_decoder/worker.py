"""Pinned upstream video-only decoder, isolated behind the owned launch gate."""
from __future__ import annotations

import gc
import os
from pathlib import Path
import sys
import time


def decoder_execution_context(torch):
    """Match upstream no_grad; prefetch threads must write normal staging buffers."""
    return torch.no_grad()


def main():
    root = Path(sys.argv[1]).resolve(strict=True)
    tick = time.monotonic()
    while not (root / 'launch-gate').is_file():
        if time.monotonic() - tick > 30:
            raise RuntimeError('Decoder child was not assigned to its own supervisor')
        time.sleep(.05)
    # h3_t8 is a namespace package, not the plugin root; this never loads Core.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from h3_t8.freevideo_decoder import contract as c
    from h3_t8.freevideo_exp.runtime import digest, tensor_record, write_json_new, verify_files
    request = c.read_json(root / 'request.json')
    if (request.get('schema') != 't8-freevideo-decoder-request-v1'
            or request.get('mode') not in ('eager', 'compile') or request.get('sampling_NFE_added') != 0
            or request.get('audio_sent_to_worker') is not False
            or digest(request['config_path']) != request['config_sha256']
            or digest(root / 'input.safetensors') != request['input_sha256']):
        raise ValueError('Decoder request, immutable config or input identity mismatch')
    config = c.config_for(request['config_path'], full=True)
    if config['inventory_sha256'] != request['inventory_sha256']:
        raise ValueError('Decoder worker source inventory mismatch')
    expected_flag = '1' if request['mode'] == 'compile' else '0'
    if (os.environ.get('FREEVIDEO_VAE_COMPILE') != expected_flag
            or os.environ.get('FREEVIDEO_HOME') != config['home']):
        raise ValueError('Decoder mode/cache environment changed')
    sys.path.insert(0, config['source_root'])
    from freevideo_engine.paths import add_vdn
    add_vdn()
    import numpy as np
    import torch
    from safetensors.torch import load_file
    import diffusers
    from freevideo_engine.vae_weights import load_video_decoder
    from freevideo_engine.vae_tiles import TileDecoder, compile_blocks
    from freevideo_engine.streamed_weights import SafetensorLayers
    from freevideo_engine.system import system_memory
    from freevideo_engine.locking import runtime_lock
    if not Path(diffusers.__file__).resolve().is_relative_to(Path(config['vdn_root']) / 'diffusers/src'):
        raise ValueError('Foreign Diffusers won isolated decoder import')
    if any(name == 'comfy' or name.startswith('comfy.') for name in sys.modules):
        raise ValueError('Core must not be imported into the decoder child')
    torch.set_num_threads(8)
    tensors = load_file(str(root / 'input.safetensors'), device='cpu')
    if set(tensors) != {'video'} or tensor_record(tensors['video']) != request['binding']['video']:
        raise ValueError('Decoder child receives exactly the completed normalized video tensor')
    canvas = request['binding']['geometry']
    if tuple(tensors['video'].shape) != (1, 24, (canvas['frames'] - 5) // 17 * 5 + 2,
            canvas['height'] // 16, canvas['width'] // 16):
        raise ValueError('Decoder temporal/spatial latent grid mismatch')
    if not torch.cuda.is_available():
        raise RuntimeError('The pinned FreeVideo decoder requires an available CUDA device')
    vae_config = c.read_json(Path(config['base']) / 'vae/config.json')
    if vae_config.get('decoder_num_layers') != 36:
        raise ValueError('Unexpected decoder block count')
    # This is a startup floor, not a large-canvas memory/speed guarantee.
    free_gpu, total_gpu = torch.cuda.mem_get_info()
    memory = system_memory()
    if free_gpu < 3 * 1024**3 or memory['available_bytes'] < 8 * 1024**3:
        raise RuntimeError('Decoder startup requires 3 GiB free VRAM and 8 GiB available host memory')
    vae = tiles = weights = None
    started = time.perf_counter()
    report = {}
    with runtime_lock(), decoder_execution_context(torch):
        try:
            write_json_new(root / 'progress.json', dict(completed=0, phase='load_video_vae'))
            weights = SafetensorLayers((Path(config['base']) / 'vae').glob('*.safetensors'),
                [f'decoder.transformer_blocks.{index}.' for index in range(36)])
            vae, preparation = load_video_decoder(config['base'], resident_blocks=0,
                weight_source=weights, linear_fp16=False)
            # Constructor-computed RoPE buffers are real CPU buffers, not checkpoint keys.
            for name, child in vae.decoder.named_children():
                if name != 'transformer_blocks':
                    child.to('cuda')
            vae.decoder.register_tokens.data = vae.decoder.register_tokens.data.to('cuda')
            vae.post_quant_conv.to('cuda')
            tiles = TileDecoder(vae.decoder, prefetch=True, weight_source=weights, resident_blocks=0)
            tiles.install(vae)
            compiled = compile_blocks(vae.decoder)
            if compiled is not (request['mode'] == 'compile'):
                raise ValueError('Actual compiled decoder blocks do not match explicit mode')
            torch.cuda.synchronize()
            load_seconds = time.perf_counter() - started
            decode_start = time.perf_counter()
            normalized = tensors['video'].to('cuda')
            latents = c.denormalize(normalized, vae_config)
            with torch.autocast(device_type='cuda', dtype=torch.float16, cache_enabled=False):
                decoded = vae.decode(latents, return_dict=False)[0].cpu()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - decode_start
            if tuple(decoded.shape) != (1, 3, canvas['frames'], canvas['height'], canvas['width']):
                raise ValueError('Actual decoded RGB frame count/axes differ from the completed Stage')
            frames = c.pixel_frames(decoded)
            with (root / 'rgb.npy').open('xb') as stream:
                np.save(stream, frames.numpy(), allow_pickle=False)
            report = dict(schema='t8-freevideo-decoder-result-v1', request_sha256=digest(root / 'request.json'),
                binding=request['binding'], mode=request['mode'], compiled_blocks=compiled,
                decoder_blocks=36, geometry=canvas, RGB_shape=list(frames.shape), RGB_dtype='uint8',
                RGB_file_sha256=digest(root / 'rgb.npy'), sampling_NFE_added=0, audio_sent_to_worker=False,
                normalization=c.NORMALIZATION, source_inventory_sha256=config['inventory_sha256'],
                vae_load_seconds=load_seconds, video_decode_seconds=seconds, preparation=preparation,
                offload=tiles.offloader.stats(), initial_free_VRAM_bytes=free_gpu, total_VRAM_bytes=total_gpu,
                host_memory=memory, torch=str(torch.__version__), cuda=torch.version.cuda,
                device=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                freevideo_revision=c.FREEVIDEO_REVISION, vdn_revision=c.VDN_REVISION, vae_revision=c.VAE_REVISION,
                no_sampler_or_audio_decoder_called=True, human_quality_accepted=False)
        finally:
            if tiles is not None:
                tiles.close()
            if weights is not None:
                weights.release_host_views()
            vae = tiles = weights = None
            gc.collect()
            torch.cuda.empty_cache()
    # Do not publish success on decode, cleanup or immutable-input failures.
    verify_files(config['files'])
    if (c.read_json(root / 'request.json') != request
            or digest(request['config_path']) != request['config_sha256']
            or digest(root / 'input.safetensors') != request['input_sha256']):
        raise ValueError('Decoder request/config/input changed during execution')
    write_json_new(root / 'result.json', report)


if __name__ == '__main__':
    main()
