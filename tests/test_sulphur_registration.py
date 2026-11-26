"""Compare against the actual immutable pre-Sulphur release, not a new ledger."""
import asyncio
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import h3_audio_t8_pkg as current

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'be3a3e9219feb63848b3415f1dd1d1571b812203'


def test_all_804_actual_preceding_release_node_schemas_are_unchanged(tmp_path):
    try:
        raw = subprocess.check_output(['git', 'archive', '--format=zip', BASELINE],
                                      cwd=ROOT, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        pytest.skip('The immutable pre-Sulphur v1.106.3 Git object is unavailable')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for name in archive.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
        archive.extractall(tmp_path)
    spec = importlib.util.spec_from_file_location('sulphur_previous_release', tmp_path/'__init__.py',
                                                   submodule_search_locations=[str(tmp_path)])
    previous = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = previous
    spec.loader.exec_module(previous)
    old = asyncio.run(previous.comfy_entrypoint().get_node_list())
    actual = asyncio.run(current.comfy_entrypoint().get_node_list())
    assert len(old) == 804 and len(actual) == 806
    assert [n.define_schema().node_id for n in actual[:804]] == [n.define_schema().node_id for n in old]
    for left, right in zip(old, actual):
        assert left.GET_NODE_INFO_V1() == right.GET_NODE_INFO_V1(), left.define_schema().node_id


def test_every_preceding_saved_workflow_byte_is_preserved():
    paths = subprocess.check_output(['git', 'diff', '--name-only', BASELINE, '--', 'examples/workflows'],
                                    cwd=ROOT, text=True)
    assert all(name.startswith('examples/workflows/99-sulphur3/') for name in paths.splitlines())
