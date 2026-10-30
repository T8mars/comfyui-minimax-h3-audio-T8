"""Full released schemas and reviewed workflows compared against Git baseline."""
import asyncio
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

import h3_audio_t8_pkg as current

ROOT=Path(__file__).parents[1]
BASELINE='728db7ed9249cedd99c304c5c865f30e285c8d36'


def test_all_779_original_schemas_are_unchanged_and_new_node_is_append_only(tmp_path):
    try:
        archive=subprocess.check_output(['git','archive','--format=zip',BASELINE],cwd=ROOT,stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        pytest.skip('The immutable v1.104.0 Git object is unavailable in this source checkout')
    with zipfile.ZipFile(io.BytesIO(archive)) as package:
        for name in package.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
        package.extractall(tmp_path)
    spec=importlib.util.spec_from_file_location('pika_released_schema_reference',tmp_path/'__init__.py',
                                              submodule_search_locations=[str(tmp_path)])
    old=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=old
    spec.loader.exec_module(old)
    previous=asyncio.run(old.comfy_entrypoint().get_node_list())
    actual=asyncio.run(current.comfy_entrypoint().get_node_list())
    assert len(previous)==779 and len(actual)==780
    assert [n.define_schema().node_id for n in actual[:779]] == [n.define_schema().node_id for n in previous]
    for reference,candidate in zip(previous,actual):
        assert json.dumps(reference.GET_NODE_INFO_V1(),sort_keys=True)==json.dumps(candidate.GET_NODE_INFO_V1(),sort_keys=True)
    assert actual[-1].define_schema().node_id=='MiniMaxH3PikaTwoPassSamplerEXPT8'
    assert actual[-1].GET_NODE_INFO_V1()['output']==['LATENT','LATENT','STRING']


def test_all_ten_accepted_release_workflow_bytes_are_preserved():
    paths=[*(ROOT/'examples/workflows/92-open-regenerate').glob('CompleteRecipe*.json'),
           *(ROOT/'examples/workflows/93-director-reviewed').glob('*.json'),
           *(ROOT/'examples/workflows/94-pika-flex').glob('PikaFlex*.json')]
    # Index metadata is not an executable graph.
    paths=[p for p in paths if p.name!='index.json']
    assert len(paths)==10
    for path in paths:
        relative=path.relative_to(ROOT)
        try:
            expected=subprocess.check_output(['git','show',f'{BASELINE}:{relative.as_posix()}'],cwd=ROOT,stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            pytest.skip('The immutable v1.104.0 Git object is unavailable in this source checkout')
        assert hashlib.sha256((ROOT/relative).read_bytes()).digest()==hashlib.sha256(expected).digest()
