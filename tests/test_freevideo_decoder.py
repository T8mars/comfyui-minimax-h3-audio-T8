"""New video-only decoder contracts; CPU fixtures do not certify GPU quality."""
import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import h3_audio_t8_pkg
from h3_audio_t8_pkg.freevideo_decoder import contract as c, runtime as r
from h3_audio_t8_pkg.freevideo_decoder.nodes import NODES
from h3_audio_t8_pkg.freevideo_exp.runtime import canonical, digest, tensor_record
from test_freevideo_exp import stage as legacy_stage
from test_freevideo_quality import stage as quality_stage
from test_freevideo_split import mid


def config_fixture(directory):
    base = directory / 'base'
    rows = [dict(path=str(base / 'vae' / name), bytes=size, sha256=sha, kind='original_video_vae')
            for name, (size, sha) in c.ASSETS.items()]
    rows += [dict(path=str(base / 'decoder-assets.json'), bytes=123, sha256='f' * 64, kind='assets_proof')]
    rows += [dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path), kind='decoder_adapter')
             for path in Path(c.__file__).parent.glob('*.py')]
    return dict(schema=c.SCHEMA, freevideo_revision=c.FREEVIDEO_REVISION, vdn_revision=c.VDN_REVISION,
        vae_revision=c.VAE_REVISION, normalization=c.NORMALIZATION, python=str(directory / 'python.exe'),
        source_root=str(directory / 'source'), vdn_root=str(directory / 'vdn'), base=str(base), home=str(directory / 'home'),
        files=rows, inventory_sha256=hashlib.sha256(canonical(rows).encode()).hexdigest(),
        vae_config_sha256=c.ASSETS['config.json'][1], assets_proof_sha256='f' * 64,
        pixel_mean=list(c.PIXEL_MEAN), pixel_std=list(c.PIXEL_STD),
        parameter_policy='original_FP32_streamed_no_Linear_compute_cache')


def test_metadata_config_requires_independent_complete_pins(tmp_path):
    config = config_fixture(tmp_path)
    assert c.validate_config(config) is config
    for field, wrong in [('schema', 't8-freevideo-runtime-v2'), ('freevideo_revision', 'old'),
            ('normalization', 'already_unnormalized'), ('python', 'python.exe'),
            ('parameter_policy', 'fp16_linear'), ('pixel_mean', [0, 0, 0]), ('vae_config_sha256', '0' * 64)]:
        with pytest.raises(ValueError):
            c.validate_config(dict(config, **{field: wrong}))
    # A metadata-valid fixture has NO weights; full execution cannot accept it.
    path = tmp_path / 'config.json'
    path.write_text(canonical(config), encoding='utf8')
    with pytest.raises(FileNotFoundError):
        c.config_for(path, full=True)


def test_inventory_rehash_cannot_authorize_wrong_weight_or_missing_adapter(tmp_path):
    config = config_fixture(tmp_path)
    for changed in ([dict(row, sha256='0' * 64) if row['kind'] == 'original_video_vae' else row for row in config['files']],
            config['files'][:-1], [*config['files'], config['files'][0]], [None]):
        value = dict(config, files=changed, inventory_sha256=hashlib.sha256(canonical(changed).encode()).hexdigest())
        with pytest.raises(ValueError):
            c.validate_config(value)


def test_bounded_json_rejects_nonfinite_and_non_object(tmp_path):
    path = tmp_path / 'value.json'
    for value in ('{"x":NaN}', '[]', '{"x":Infinity}'):
        path.write_text(value, encoding='utf8')
        with pytest.raises(ValueError):
            c.read_json(path)


def test_exactly_once_native_mean_std_and_no_input_mutation():
    from comfy.ldm.minimax.vae import LATENTS_MEAN, LATENTS_STD
    original = torch.linspace(-2, 2, 24 * 2 * 3 * 4).reshape(1, 24, 2, 3, 4)
    before = original.clone()
    config = dict(_class_name='AutoencoderKLMiniMaxH3', latent_channels=24,
        latents_mean=list(LATENTS_MEAN), latents_std=list(LATENTS_STD))
    actual = c.denormalize(original, config)
    expected = original * torch.tensor(LATENTS_STD).view(1, 24, 1, 1, 1) + torch.tensor(LATENTS_MEAN).view(1, 24, 1, 1, 1)
    assert torch.equal(actual, expected) and torch.equal(original, before)
    assert not torch.equal(c.denormalize(actual, config), actual)
    for bad in (dict(config, latents_std=[0.] * 24), dict(config, latents_mean=[float('nan')] * 24)):
        with pytest.raises(ValueError):
            c.denormalize(original, bad)
    with pytest.raises(ValueError):
        c.denormalize(original.half(), config)


def test_pixel_axes_clamp_round_exact_chunked_and_no_mutation():
    source = torch.linspace(-9, 9, 1 * 3 * 19 * 2 * 4).reshape(1, 3, 19, 2, 4).half()
    original = source.clone()
    actual = c.pixel_frames(source)
    expected = (source.float() * torch.tensor(c.PIXEL_STD).view(1, 3, 1, 1, 1)
        + torch.tensor(c.PIXEL_MEAN).view(1, 3, 1, 1, 1)).clamp(0, 1)
    expected = (expected[0].permute(1, 2, 3, 0) * 255).round().to(torch.uint8)
    assert actual.shape == (19, 2, 4, 3) and torch.equal(actual, expected)
    assert torch.equal(source, original)
    source[0, 0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError):
        c.pixel_frames(source)


def test_accept_completed_quality_high_and_single_but_not_light_low_mid_or_legacy():
    for stage in (quality_stage('light', 'HIGH'), quality_stage('medium'), quality_stage('high'), quality_stage('max')):
        video, binding = c.stage_input(stage, 'quality')
        assert video is stage.video and binding['audio'] == tensor_record(stage.audio)
        assert binding['sampling_NFE_added'] == 0
    for stage in (quality_stage(), mid(), legacy_stage('HIGH')):
        with pytest.raises(ValueError):
            c.stage_input(stage, 'quality')


def test_legacy_accepts_only_completed_high_and_cold_source_mutation_rejected():
    stage = legacy_stage('HIGH')
    _, binding = c.stage_input(stage, 'legacy')
    assert binding['family'] == 'legacy'
    for wrong in (legacy_stage(), mid(), quality_stage('light', 'HIGH')):
        with pytest.raises(ValueError):
            c.stage_input(wrong, 'legacy')
    stage.audio.add_(1)
    with pytest.raises(ValueError, match='content'):
        c.stage_input(stage, 'legacy')


def output_fixture():
    stage = quality_stage('medium')
    _, binding = c.stage_input(stage, 'quality')
    request = dict(binding=binding, mode='eager', inventory_sha256='7' * 64)
    shape = [39, 256, 256, 3]
    result = dict(schema='t8-freevideo-decoder-result-v1', binding=binding, mode='eager', compiled_blocks=False,
        source_inventory_sha256='7' * 64, sampling_NFE_added=0, audio_sent_to_worker=False,
        decoder_blocks=36, geometry=binding['geometry'], RGB_shape=shape, RGB_dtype='uint8', normalization=c.NORMALIZATION)
    return stage, binding, request, result, np.full(shape, 127, dtype=np.uint8)


def test_return_audio_exact_clone_independent_of_rgb_and_geometry_complete():
    stage, binding, request, result, frames = output_fixture()
    video_before, audio_before = stage.video.clone(), stage.audio.clone()
    images, audio = r.accept_output(stage, 'quality', binding, request, result, frames)
    assert images.shape == (39, 256, 256, 3) and images.dtype == torch.float32
    assert torch.equal(audio['samples'], stage.audio) and audio['samples'].data_ptr() != stage.audio.data_ptr()
    images.add_(1)
    audio['samples'].add_(1)
    assert torch.equal(stage.video, video_before) and torch.equal(stage.audio, audio_before)


def test_output_forgery_wrong_clock_nfe_audio_or_compile_not_accepted():
    stage, binding, request, result, frames = output_fixture()
    for bad in (dict(sampling_NFE_added=1), dict(sampling_NFE_added=False), dict(audio_sent_to_worker=True),
            dict(compiled_blocks=True), dict(RGB_shape=[38, 256, 256, 3]), dict(normalization='double'),
            dict(source_inventory_sha256='0' * 64)):
        with pytest.raises(ValueError):
            r.accept_output(stage, 'quality', binding, request, dict(result, **bad), frames)
    with pytest.raises(ValueError):
        r.accept_output(stage, 'quality', binding, request, result, frames[:-1])
    stage.video.add_(1)
    with pytest.raises(ValueError, match='content'):
        r.accept_output(stage, 'quality', binding, request, result, frames)


def test_explicit_mode_isolated_caches_and_no_parent_environment_mutation(tmp_path, monkeypatch):
    config = config_fixture(tmp_path)
    monkeypatch.setenv('PYTHONPATH', 'foreign')
    monkeypatch.setenv('FREEVIDEO_VAE_COMPILE', '1')
    eager, compiled = r.environment(config, 'eager'), r.environment(config, 'compile')
    assert 'PYTHONPATH' not in eager and eager['FREEVIDEO_VAE_COMPILE'] == '0'
    assert compiled['FREEVIDEO_VAE_COMPILE'] == '1'
    assert eager['TORCHINDUCTOR_CACHE_DIR'] != compiled['TORCHINDUCTOR_CACHE_DIR']
    import os
    assert os.environ['PYTHONPATH'] == 'foreign' and os.environ['FREEVIDEO_VAE_COMPILE'] == '1'
    with pytest.raises(ValueError):
        r.environment(config, 'auto')


def test_actual_worker_context_keeps_cross_thread_staging_mutable_and_no_grad():
    from concurrent.futures import ThreadPoolExecutor
    from h3_audio_t8_pkg.freevideo_decoder.worker import decoder_execution_context
    grad_before = torch.is_grad_enabled()
    with decoder_execution_context(torch):
        assert not torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        target = torch.empty(9, dtype=torch.float32)
        source = torch.arange(9, dtype=torch.float32)
        assert not torch.is_inference(target)
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(target.copy_, source).result()
        assert torch.equal(target, source)
    assert torch.is_grad_enabled() == grad_before
    # Preserve the actual failed behavior as a negative regression, no CUDA.
    with torch.inference_mode():
        forbidden = torch.empty(9, dtype=torch.float32)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match='Inplace update to inference tensor outside InferenceMode'):
            executor.submit(forbidden.copy_, source).result()


def test_new_two_typed_nodes_append_without_importing_engine_and_eager_default():
    import sys
    ids = [cls.GET_SCHEMA().node_id for cls in asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())]
    declared = json.loads((Path(__file__).parents[1] / 'features.json').read_text(encoding='utf8'))['nodes']
    assert ids == declared and len(ids) == len(set(ids)) == 678
    assert ids[676:] == [node.__name__ for node in NODES]
    for node, expected in zip(NODES, ['H3_T8_FREEVIDEO_QUALITY_STAGE', 'H3_T8_FREEVIDEO_STAGE']):
        schema = node.GET_SCHEMA().get_v1_info(node)
        assert schema.input['required']['completed_stage'][0] == expected
        assert schema.input['required']['decode_mode'][1]['default'] == 'eager'
        assert schema.output == ['IMAGE', 'LATENT', 'STRING']
    assert not any(key == 'freevideo_engine' or key.startswith('freevideo_engine.') for key in sys.modules)
