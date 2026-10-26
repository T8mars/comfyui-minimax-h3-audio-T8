"""Test harness isolation only; never changes a production media path."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tools.run_candidate_cpu_regression import isolated_tmp_path_plugin, source_snapshot


def test_independent_tmp_path_is_unique_recorded_and_retained(tmp_path):
    plugin = isolated_tmp_path_plugin(pytest, tmp_path)
    request = SimpleNamespace(node=SimpleNamespace(nodeid='synthetic/fixture-boundary'))
    create = plugin.tmp_path.__wrapped__
    first, second = create(plugin, request), create(plugin, request)
    assert first != second and first.is_dir() and second.is_dir()
    receipt = json.loads((tmp_path / 'temporary-boundary.json').read_text(encoding='utf8'))
    root = Path(receipt['root'])
    assert receipt['retained'] and receipt['test_assertions_and_runtime_unchanged']
    assert [row['directory'] for row in receipt['directories']] == [str(first), str(second)]
    assert all(row['test'] == request.node.nodeid for row in receipt['directories'])
    assert first.resolve().parent == second.resolve().parent == root.resolve()
    assert not first.is_symlink() and not second.is_symlink()
    (first / 'failure-evidence.json').write_text('{}', encoding='utf8')
    assert first.is_dir() and (first / 'failure-evidence.json').is_file()


def test_source_freeze_includes_actual_frontend_modules_and_styles(tmp_path):
    for name in ('pyproject.toml', 'meta.json', 'features.json', '.comfyignore'):
        (tmp_path / name).write_text('{}', encoding='utf8')
    root = tmp_path / 'web'
    root.mkdir()
    for name in ('session.mjs', 'workbench.css', 'plain.js'):
        (root / name).write_text('before', encoding='utf8')
    before = source_snapshot(tmp_path)
    assert {'web/session.mjs', 'web/workbench.css', 'web/plain.js'} <= before.keys()
    (root / 'session.mjs').write_text('changed', encoding='utf8')
    (root / 'workbench.css').write_text('changed', encoding='utf8')
    after = source_snapshot(tmp_path)
    assert {key for key in before if before[key] != after[key]} == {'web/session.mjs', 'web/workbench.css'}
