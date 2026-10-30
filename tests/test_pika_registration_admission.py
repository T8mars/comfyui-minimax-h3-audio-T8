"""Finite successor keeps old reviews, rejects missing files and forged parents."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest
from tools import audit_modular_new_sampling as gate


ROOT = Path(__file__).resolve().parents[1]
NAME = 'tests/fixtures/pika_flex_source.json'


def tree(destination):
    old = json.loads((ROOT/'tests/fixtures/h27_reviewed_sampling_source.json').read_text('utf-8'))
    successor = json.loads((ROOT/NAME).read_text('utf-8'))
    paths = {*old['files'], *successor['files'], NAME, 'tests/fixtures/h27_reviewed_sampling_source.json',
             'tests/fixtures/h24_reviewed_sampling_source.json', 'tests/fixtures/h21_reviewed_sampling_1_102.json'}
    for name in paths:
        target = destination/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    return old, successor


def test_original_h27_parent_and_all_other_source_bytes_are_unchanged():
    old = json.loads((ROOT/'tests/fixtures/h27_reviewed_sampling_source.json').read_text('utf-8'))
    successor = json.loads((ROOT/NAME).read_text('utf-8'))
    assert all(old['files'][name] == digest for name,digest in successor['previous_files'].items())
    for name,digest in old['files'].items():
        if name not in successor['previous_files']:
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    assert gate.guard(ROOT,json.loads(gate.BASELINE.read_text('utf-8')))['status'] == 'pass'


@pytest.mark.parametrize('name', ['__init__.py','h3_t8/pika_flex.py','h3_t8/nodes_pika_flex.py',
                                  'tools/audit_modular_new_sampling.py'])
def test_mutated_or_missing_successor_file_fails_the_original_sha_check(tmp_path,name):
    tree(tmp_path)
    target=tmp_path/name
    target.write_bytes(target.read_bytes()+b'\n# unreviewed\n')
    with pytest.raises(ValueError,match='source/workflow changed'):
        gate._h27_review(tmp_path)
    target.unlink()
    with pytest.raises(ValueError,match='source/workflow changed'):
        gate._h27_review(tmp_path)


def test_successor_cannot_relabel_a_different_parent_or_add_unbounded_files(tmp_path):
    _old, successor=tree(tmp_path)
    changed=deepcopy(successor)
    changed['previous_files']['__init__.py']='0'*64
    (tmp_path/NAME).write_text(json.dumps(changed),'utf-8')
    with pytest.raises(ValueError,match='different H27 source binding'):
        gate._h27_review(tmp_path)
    changed=deepcopy(successor)
    changed['files']['h3_t8/other_sampler.py']='0'*64
    (tmp_path/NAME).write_text(json.dumps(changed),'utf-8')
    with pytest.raises(ValueError,match='bounded Pika successor'):
        gate._h27_review(tmp_path)
