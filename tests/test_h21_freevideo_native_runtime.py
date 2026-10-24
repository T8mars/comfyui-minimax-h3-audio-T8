"""Selected-width native export and actual CPU sampler contracts, not CUDA Engine."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from h3_audio_t8_pkg.freevideo_exp import effects as shared_effects, runtime as shared_runtime
from h3_audio_t8_pkg.freevideo_h21 import effects, runtime, source


def test_native_selected_width_real_export_and_paired_identity():
    canvas = dict(width=256, height=256, frames=39)
    embeds = torch.arange(24).float().reshape(1, 3, 8)
    tags = torch.ones(3, dtype=torch.long)
    condition = [[embeds, dict(minimax_token_tags=tags)]]
    original = embeds.clone()
    with pytest.raises(ValueError, match="5120"):
        shared_runtime.conditioning_transport(condition, canvas)
    tensors, metadata = shared_runtime.conditioning_transport(condition, canvas, text_dim=8)
    assert torch.equal(tensors["embeds"], embeds[0]) and metadata["task"] == "t2va"
    assert torch.equal(embeds, original)
    with pytest.raises(ValueError, match="7"):
        shared_runtime.conditioning_transport(condition, canvas, text_dim=7)
    value = dict(schema="t8-freevideo-relay-v1", mode="apply_exp", query_route="video_only_paper", query_chunk_rows=32,
        geometry=canvas, events=[dict(text_key_start=1, text_key_end=3, midpoint=1., window=2., sigma=1.)],
        embeddings=shared_runtime.tensor_record(embeds), tags_sha256=hashlib.sha256(tags.view(torch.uint8).numpy().tobytes()).hexdigest(),
        conditions_sha256=shared_effects.condition_identity(condition, canvas, text_dim=8))
    value["sha256"] = hashlib.sha256(source.canonical(value).encode()).hexdigest()
    model = runtime.Model("unused", "unused", effects_json=source.canonical(dict(relay=value)))
    condition[0][1][shared_effects.RELAY_KEY] = value
    assert effects.transport(model, condition, canvas, text_dim=8)["relay"] == value
    embeds[0, 1, 0] += 1
    with pytest.raises(ValueError, match="conditioning|identity|bound"):
        effects.transport(model, condition, canvas, text_dim=8)


def test_shared_default_identity_retains_original_two_argument_delegate(monkeypatch):
    calls = []
    def legacy_two_argument(condition, canvas):
        calls.append((condition, canvas))
        return dict(embeds=torch.ones(2, 5120), tags=torch.ones(2, dtype=torch.long)), dict(task="t2va")
    monkeypatch.setattr(shared_runtime, "conditioning_transport", legacy_two_argument)
    result = shared_effects.condition_identity("native-default", "physical-default")
    assert len(result) == 64 and calls == [("native-default", "physical-default")]


@pytest.mark.parametrize("mode", ["t2va", "ref2va_audio", "i2va", "cancel_retry"])
def test_actual_selected_width_validators_original_sampler_and_exception_restore(tmp_path, mode):
    python, vdn, root = (os.environ.get(key) for key in
                        ("T8_H21_FREEVIDEO_PYTHON", "T8_H21_FREEVIDEO_VDN", "T8_H21_FREEVIDEO_SOURCE"))
    if not python or not vdn or not root:
        pytest.skip("Explicit existing fixed source/VDN Python required; no CUDA emulation or download")
    output = tmp_path / "actual-native-width.json"
    script = r'''
import faulthandler, json, pathlib, sys
faulthandler.enable()
faulthandler.dump_traceback_later(45, repeat=True)
print('native-width-stage: before_torch', flush=True)
repo, selected_source, vdn, mode, output = sys.argv[1:]
sys.path[:0] = [repo, selected_source, str(pathlib.Path(vdn) / 'diffusers/src'), vdn]
import torch
print('native-width-stage: torch_imported', flush=True)
from h3_t8.freevideo_exp.runtime import conditioning_transport
from h3_t8.freevideo_h21.source import Source
from h3_t8.freevideo_h21.plans import plan
from h3_t8.freevideo_h21.assets import model_structure
from h3_t8.freevideo_h21.native_contract import selected_width
from h3_t8.freevideo_h21.tiny_worker import TinyVDNCPU
from h3_t8.freevideo_h21.execution import execute
from freevideo_engine import conditioning, keyframes, media_conditioning
print('native-width-stage: producer_modules_imported', flush=True)
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
source = Source(selected_source)
task = 't2va' if mode == 'cancel_retry' else mode
selected = plan(source, 'light', 256, 256, 39, task=task)
owner = TinyVDNCPU(source, selected)
print('native-width-stage: actual_tiny_model_constructed', flush=True)
config_root = pathlib.Path(output).parent / 'actual-tiny-model'
(config_root / 'transformer').mkdir(parents=True)
(config_root / 'transformer/config.json').write_text(json.dumps(dict(owner.transformer.config)), encoding='utf8')
dimension = model_structure(config_root)['text_dim']
assert dimension == 8
embeds = torch.arange(24).float().reshape(1, 3, dimension) / 20
tags = torch.tensor([0, 1, 1]) if task == 'i2va' else torch.ones(3, dtype=torch.long)
extra = dict(minimax_token_tags=tags)
if task == 'i2va':
    extra['minimax_keyframes'] = [dict(resolved_frame_index=0, latent=torch.randn(1, 24, 1, 16, 16))]
if task == 'ref2va_audio':
    extra['minimax_refs'] = [dict(kind='audio', audio_latent=torch.randn(1, 32, 2, 2))]
export, meta = conditioning_transport([[embeds, extra]], dict(width=256, height=256, frames=39), text_dim=dimension)
value = dict(prompt_embeds=export['embeds'], text_token_tags=export['tags'], task=meta['task'])
if meta['keyframes']:
    value.update(keyframe_anchors=[r['anchor'] for r in meta['keyframes']], condition_latents=[export[r['tensor']] for r in meta['keyframes']])
if meta['refs']:
    value['references'] = [dict(kind=r['kind'], audio_latent=export[r['audio_tensor']]) for r in meta['refs']]
original = (conditioning.to_cache, keyframes.validate_conditioning, media_conditioning.to_cache, media_conditioning.validate_conditioning)
def assert_restored():
    current = (conditioning.to_cache, keyframes.validate_conditioning, media_conditioning.to_cache, media_conditioning.validate_conditioning)
    assert all(a is b for a, b in zip(current, original))
with selected_width(5120):
    assert_restored()
try:
    media_conditioning.describe(value, 256, 256, 39)
except ValueError as old_error:
    assert '5120' in str(old_error)
else:
    raise AssertionError('original fixed Source width counterexample did not fail')
try:
    with selected_width(dimension):
        broken = dict(value, prompt_embeds=value['prompt_embeds'][:, :-1])
        media_conditioning.describe(broken, 256, 256, 39)
except ValueError as error:
    assert str(dimension) in str(error)
else:
    raise AssertionError('actual selected-width mismatch was accepted')
assert_restored()
cancelled = False
if mode == 'cancel_retry':
    def stop(completed, *args):
        if completed == 2:
            raise InterruptedError('actual-native-width-cancel')
    try:
        with selected_width(dimension):
            media_conditioning.describe(value, 256, 256, 39)
            execute(owner, value, 42, selected, progress=stop)
    except InterruptedError:
        cancelled = True
    assert cancelled
    assert_restored()
with selected_width(dimension):
    information = media_conditioning.describe(value, 256, 256, 39)
    video, audio, actual = execute(owner, value, 42, selected)
assert_restored()
assert actual['completed_callbacks'] == 11 and all(p['measurements']['actual_transformer_calls'] == n
    for p, n in zip(actual['passes'], [8, 3]))
assert torch.isfinite(video).all() and torch.isfinite(audio).all()
assert actual['high_audio_preserved']
if mode == 'cancel_retry':
    fresh = TinyVDNCPU(source, selected)
    with selected_width(dimension):
        fresh_video, fresh_audio, fresh_actual = execute(fresh, value, 42, selected)
    assert torch.equal(video, fresh_video) and torch.equal(audio, fresh_audio)
    assert_restored()
source.verify()
assert not torch.cuda.is_initialized()
faulthandler.cancel_dump_traceback_later()
pathlib.Path(output).write_text(json.dumps(dict(text_dim=dimension, task=task, information=information,
    actual=actual, cancelled=cancelled, original_four_functions_restored=True, CUDA_initialized=False,
    qualification='actual selected config + Source validation + original samplers + real tiny model CPU; not CUDA Engine constructor')), encoding='utf8')
'''
    child = subprocess.run([python, "-I", "-B", "-X", "utf8", "-c", script, str(Path(__file__).parents[1]),
        root, vdn, mode, str(output)], capture_output=True, text=True, encoding="utf8",
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", FREEVIDEO_VDN_ROOT=vdn, OMP_NUM_THREADS="1"), timeout=240)
    assert child.returncode == 0, child.stdout + child.stderr
    report = json.loads(output.read_text(encoding="utf8"))
    assert report["text_dim"] == 8 and report["original_four_functions_restored"]
    assert report["actual"]["completed_callbacks"] == 11 and not report["CUDA_initialized"]
