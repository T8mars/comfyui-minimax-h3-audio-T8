"""Explicit dormant aliases never hide missing/changed live runtime or new calls."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from tools.audit_modular_new_sampling import guard, reviewed_layout_sites

ROOT = Path(__file__).resolve().parents[1]
NAME = 'tests/fixtures/modular_reviewed_runtime_layout_v1.json'


def fixture(tmp_path):
    target = tmp_path / NAME
    target.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / NAME, target)
    review = json.loads(target.read_text(encoding='utf8'))
    for item in review['replacements'].values():
        path = tmp_path / item['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / item['path'], path)
    baseline = json.loads((ROOT / 'tests/fixtures/modular_new_sampling_baseline_v10.json').read_text(encoding='utf8'))
    return review, baseline


def test_exact_nine_runtime_modules_qualify21_absent_aliases_without_mutating_v10(tmp_path):
    review, baseline = fixture(tmp_path)
    original = deepcopy(baseline)
    effective, omitted = reviewed_layout_sites(tmp_path, baseline)
    assert baseline == original and len(baseline['sites']) == 118
    assert len(effective) == 97 and len(omitted) == 21
    assert omitted == review['retired_sites']
    assert all(site['path'].startswith('h3_t8/') for site in effective)


def test_without_review_no_missing_site_is_silently_ignored(tmp_path):
    baseline = {'sites': [{'path': 'old.py', 'symbol': 'old', 'callee': 'x.sample', 'call_sha256': '0'*64}]}
    effective, omitted = reviewed_layout_sites(tmp_path, baseline)
    assert effective == baseline['sites'] and omitted == []


@pytest.mark.parametrize('damage', ['missing', 'changed_AST', 'invalid_receipt', 'baseline_lost_site'])
def test_missing_changed_runtime_or_removed_reviewed_baseline_fail_closed(tmp_path, damage):
    review, baseline = fixture(tmp_path)
    path = tmp_path / next(iter(review['replacements'].values()))['path']
    if damage == 'missing':
        path.unlink()
    elif damage == 'changed_AST':
        with path.open('a', encoding='utf8') as output:
            output.write('\ndef unrelated_change():\n return 42\n')
    elif damage == 'invalid_receipt':
        receipt = tmp_path / NAME
        receipt.write_text('{}', encoding='utf8')
    else:
        baseline['sites'].remove(review['retired_sites'][0])
    with pytest.raises(ValueError):
        reviewed_layout_sites(tmp_path, baseline)


def test_reintroduced_root_calls_are_scanned_not_granted_layout_exception(tmp_path):
    review, baseline = fixture(tmp_path)
    for name in review['replacements']:
        (tmp_path / name).write_text('def run_two_pass(sampler):\n return sampler.sample(999)\n', encoding='utf8')
    effective, omitted = reviewed_layout_sites(tmp_path, baseline)
    assert effective == baseline['sites'] and omitted == []
    checked = guard(tmp_path, baseline)
    assert checked['status'] == 'fail' and checked['new_sites'] >= 9


def test_mechanical_line_endings_do_not_change_full_AST_qualification(tmp_path):
    review, baseline = fixture(tmp_path)
    for item in review['replacements'].values():
        path = tmp_path / item['path']
        path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
    effective, omitted = reviewed_layout_sites(tmp_path, baseline)
    assert len(effective) == 97 and len(omitted) == 21
