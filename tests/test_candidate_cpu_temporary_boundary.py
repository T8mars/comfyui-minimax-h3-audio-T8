"""Test harness isolation only; never changes a production media path."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tools.run_candidate_cpu_regression import isolated_tmp_path_plugin


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
