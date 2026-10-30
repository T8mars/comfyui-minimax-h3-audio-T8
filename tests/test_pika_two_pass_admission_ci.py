"""Bound only this new bridge; reject unknown calls and media relabelling."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from tools import audit_modular_new_sampling as gate
import test_h27_sampling_admission_ci as predecessor

ROOT = Path(__file__).parents[1]
REVIEW = 'tests/fixtures/pika_two_pass_source.json'


def test_current_source_only_admits_the_exact_seven_sites_and_retains_future_gate():
    baseline=json.loads(gate.BASELINE.read_text('utf8'))
    report=gate.guard(ROOT,baseline)
    assert (report['status'],report['new_sites'],report['removed_sites']) == ('pass',0,0)
    live=gate.discover(ROOT)
    changed=deepcopy(live)
    next(row for row in changed if row['path']=='h3_t8/pika_two_pass.py')['call_sha256']='0'*64
    with patch.object(gate,'discover',return_value=changed):
        assert gate.guard(ROOT,baseline)['status']=='fail'
    unknown={**live[-1],'path':'h3_t8/unknown_next_sampler.py','symbol':'future.sample'}
    with patch.object(gate,'discover',return_value=live+[unknown]):
        assert gate.guard(ROOT,baseline)['new_sites']==1


@pytest.mark.parametrize('mutation',['source','parent','acceptance','extra_site'])
def test_source_parent_or_quality_mutation_cannot_borrow_acceptance(mutation):
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        helper = predecessor.H27ReviewedSource()
        helper.setUp()
        helper.copy_root(root)
        review=json.loads((root/REVIEW).read_text('utf8'))
        if mutation=='source':
            with (root/'h3_t8/pika_two_pass.py').open('a') as f:
                f.write('\n# unreviewed change\n')
        elif mutation=='parent':
            review['previous_files']['__init__.py']='0'*64
        elif mutation=='acceptance':
            review['human_quality_accepted']=True
        else:
            review['reviewed_sites'].append({**review['reviewed_sites'][0],'symbol':'another.sample'})
        (root/REVIEW).write_text(json.dumps(review),'utf8')
        with pytest.raises(ValueError,match='review|source/workflow changed'):
            gate._h27_review(root)
