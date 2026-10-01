"""Real old-save -> current-load/cold-only-remainder; exact legacy boundaries."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from h3_audio_t8_pkg.modular_sampling import chunked_v5, chunked_v5_storage as storage
from h3_audio_t8_pkg.modular_sampling import chunked_v5_storage_compat as compat
from h3_audio_t8_pkg.modular_sampling.results import _input_identity, canonical
from test_modular_chunked_v5 import make_v5_harness
from v5_storage_preimages import historical_module, reviewed_preimage_bytes


PROFILES = (compat.FULL_REPORT_PROFILE, compat.STABLE_PROFILE, compat.CURRENT_PROFILE)
TELEMETRY = ('memory_before', 'memory_after_release', 'gpu_weights_released',
             'cpu_cache_cleared', 'model.cache_hit', 'geometry.memory_warning')


def make_case(monkeypatch, tmp_path, profile, *, telemetry=False):
    calls, source, positive, noise, sampler, sigmas, plan = make_v5_harness(monkeypatch)
    lifted, lift, _ = chunked_v5.lift_standard_joint(source, plan)
    prepared, _ = chunked_v5.prepare_standard_joint(source, lifted, lift, plan, noise)
    if telemetry:
        report = {'model': {'sha256': 'a' * 64, 'cache_hit': False},
                  'geometry': {'output_width': 64, 'memory_warning': 'first'},
                  'memory_before': {'free_mib': 100}, 'memory_after_release': {'free_mib': 200},
                  'gpu_weights_released': True, 'cpu_cache_cleared': False, 'status': 'ok'}
        prepared = replace(prepared, lift=replace(lift, upscale_report=report))
    _, first, _ = chunked_v5.sample_standard_window(
        object(), positive, source, lifted, prepared, plan, noise, sampler, sigmas, 0)
    saver = storage if profile == compat.CURRENT_PROFILE else historical_module(tmp_path, profile)
    saved = saver.save_window(first, source, lifted, prepared, plan, tmp_path)
    return calls, source, positive, noise, sampler, sigmas, plan, lifted, prepared, first, saved


@pytest.mark.parametrize('profile', PROFILES)
def test_actual_completed_formats_load_without_sampling_rewriting_or_certifying(
        monkeypatch, tmp_path, profile):
    calls, source, positive, noise, sampler, sigmas, plan, lifted, prepared, first, saved = make_case(
        monkeypatch, tmp_path, profile)
    _, _, path, digest, _ = saved
    manifest_path = tmp_path / path
    original = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in manifest_path.parent.iterdir()}
    expected, _, _ = chunked_v5.sample_standard_window(
        object(), positive, source, lifted, prepared, plan, noise, sampler, sigmas, 1, first)
    before = dict(calls)
    output, loaded, report = storage.load_window(source, lifted, prepared, plan, tmp_path, path, digest, 0)
    assert calls == before and _input_identity(output) == _input_identity(first.output_latent)
    record = json.loads(report)
    assert record['implementation_profile'] == profile
    assert record['historical_literal_format'] is (profile != compat.CURRENT_PROFILE)
    assert record['automatic_cache_reuse'] is record['execution_identity_certified'] is False
    actual, _, _ = chunked_v5.sample_standard_window(
        object(), positive, source, lifted, prepared, plan, noise, sampler, sigmas, 1, loaded)
    assert calls['sample'] == before['sample'] + 1
    for got, want in zip(actual['samples'].unbind(), expected['samples'].unbind(), strict=True):
        assert torch.equal(got, want)
    worker = Path(__file__).with_name('chunked_v5_storage_worker.py')
    child = subprocess.run([sys.executable, str(worker), str(tmp_path), path, digest],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=120, check=False)
    assert child.returncode == 0, child.stdout + child.stderr
    cold = json.loads(child.stdout.strip().splitlines()[-1])
    assert cold['loaded_index'] == 0 and cold['sample_count'] == 1
    assert cold['video_identity'] == _input_identity(expected['samples'].tensors[0])
    assert cold['audio_identity'] == _input_identity(expected['samples'].tensors[1])
    assert original == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in manifest_path.parent.iterdir()}


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('field', TELEMETRY)
def test_each_telemetry_field_retains_its_actual_historical_policy(monkeypatch, tmp_path, profile, field):
    calls, source, _, _, _, _, plan, lifted, prepared, _, saved = make_case(
        monkeypatch, tmp_path, profile, telemetry=True)
    report = json.loads(json.dumps(prepared.lift.upscale_report))
    if '.' in field:
        owner, key = field.split('.')
        report[owner][key] = True if key == 'cache_hit' else None
    else:
        report[field] = not report[field] if type(report[field]) is bool else {'free_mib': 1}
    changed = replace(prepared, lift=replace(prepared.lift, upscale_report=report))
    before = dict(calls)
    args = (source, lifted, changed, plan, tmp_path, saved[2], saved[3], 0)
    if profile == compat.FULL_REPORT_PROFILE:
        with pytest.raises(ValueError, match='does not match current source, plan or noise'):
            storage.load_window(*args)
    else:
        output, _, _ = storage.load_window(*args)
        assert _input_identity(output) == _input_identity(saved[0])
    assert calls == before


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('field', ['model', 'video_noise', 'audio_noise', 'noise_report', 'noise_seed'])
def test_true_preparation_contract_changes_never_gain_historical_admission(
        monkeypatch, tmp_path, profile, field):
    calls, source, _, _, _, _, plan, lifted, prepared, _, saved = make_case(
        monkeypatch, tmp_path, profile, telemetry=True)
    if field == 'model':
        report = json.loads(json.dumps(prepared.lift.upscale_report))
        report['model']['sha256'] = 'b' * 64
        changed = replace(prepared, lift=replace(prepared.lift, upscale_report=report))
    elif field in {'video_noise', 'audio_noise'}:
        value = getattr(prepared, field).clone()
        value.reshape(-1)[0] += 1
        changed = replace(prepared, **{field: value})
    elif field == 'noise_report':
        changed = replace(prepared, noise_report={**prepared.noise_report, 'unexpected': 1})
    else:
        changed = replace(prepared, noise_seed=prepared.noise_seed + 1)
    before = dict(calls)
    with pytest.raises(ValueError, match='does not match current source, plan or noise'):
        storage.load_window(source, lifted, changed, plan, tmp_path, saved[2], saved[3], 0)
    assert calls == before


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('mutation', ['unknown_storage', 'changed_owner', 'missing_owner',
                                    'extra_owner', 'fake_helper', 'bad_digest', 'not_dict'])
def test_unknown_or_changed_implementation_is_not_a_compatibility_shortcut(
        monkeypatch, tmp_path, profile, mutation):
    calls, source, _, _, _, _, plan, lifted, prepared, _, saved = make_case(monkeypatch, tmp_path, profile)
    manifest = json.loads(saved[4])
    identity = manifest['implementation']
    if mutation == 'unknown_storage':
        identity[compat.STORAGE_SOURCE] = '0' * 64
    elif mutation == 'changed_owner':
        identity['chunked_v5.py'] = '0' * 64
    elif mutation == 'missing_owner':
        del identity['chunked_v5_relay.py']
    elif mutation == 'extra_owner':
        identity['unknown.py'] = '0' * 64
    elif mutation == 'fake_helper':
        identity[compat.COMPAT_SOURCE] = '0' * 64
    elif mutation == 'bad_digest':
        identity[compat.STORAGE_SOURCE] = []
    else:
        manifest['implementation'] = []
    manifest_path = tmp_path / saved[2]
    manifest_path.write_text(canonical(manifest), encoding='utf8')
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    before = dict(calls)
    with pytest.raises(ValueError, match='Unknown, stale or incomplete'):
        storage.load_window(source, lifted, prepared, plan, tmp_path, saved[2], digest, 0)
    assert calls == before


@pytest.mark.parametrize('profile', (compat.FULL_REPORT_PROFILE, compat.STABLE_PROFILE))
def test_historical_formats_still_reject_original_file_content_and_path_failures(monkeypatch, tmp_path, profile):
    calls, source, _, _, _, _, plan, lifted, prepared, _, saved = make_case(monkeypatch, tmp_path, profile)
    before = dict(calls)
    with pytest.raises(ValueError, match='SHA mismatch'):
        storage.load_window(source, lifted, prepared, plan, tmp_path, saved[2], '0' * 64, 0)
    with pytest.raises(ValueError, match='traversal|escapes|relative'):
        storage.load_window(source, lifted, prepared, plan, tmp_path, '../manifest.json', saved[3], 0)
    with pytest.raises(ValueError, match='does not match'):
        storage.load_window(source, lifted, prepared, plan, tmp_path, saved[2], saved[3], 1)
    state = (tmp_path / saved[2]).parent / 'state.safetensors'
    state.write_bytes(state.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='size or SHA'):
        storage.load_window(source, lifted, prepared, plan, tmp_path, saved[2], saved[3], 0)
    assert calls == before


@pytest.mark.parametrize('profile', (compat.FULL_REPORT_PROFILE, compat.STABLE_PROFILE))
def test_review_projection_recovers_actual_entire_original_source(profile):
    expected = compat.LEGACY_FULL_REPORT_SHA if profile == compat.FULL_REPORT_PROFILE else compat.HISTORICAL_STABLE_SHA
    assert hashlib.sha256(reviewed_preimage_bytes(profile)).hexdigest() == expected
