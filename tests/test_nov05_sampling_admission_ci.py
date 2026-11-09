"""Keep reviewed November sources bounded; exercise rejection, not media approval."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from tools import audit_modular_new_sampling as gate  # noqa: E402
import test_h27_sampling_admission_ci as historical  # noqa: E402


class NovemberAdmission(unittest.TestCase):
    fixture = 'tests/fixtures/nov05_reviewed_sampling_source.json'

    def test_reuse_call_multiplicity_and_future_calls_are_not_admitted(self):
        baseline = json.loads(gate.BASELINE.read_text('utf8'))
        current = gate.discover(ROOT)
        self.assertEqual(gate.guard(ROOT, baseline)['status'], 'pass')
        reuse = [x for x in current if x['path'] == 'h3_t8/modular_sampling/reuse.py']
        self.assertEqual(len(reuse), 3)
        for changed in [current + [deepcopy(reuse[0])], [x for x in current if x is not reuse[0]],
                        current + [dict(reuse[0], path='h3_t8/unreviewed_future.py')]]:
            with self.subTest(count=len(changed)), patch.object(gate, 'discover', return_value=changed):
                self.assertEqual(gate.guard(ROOT, baseline)['status'], 'fail')

    def test_changed_source_missing_workflow_and_bad_parent_fail(self):
        for mutation in ['source', 'workflow', 'parent', 'new_call', 'registration']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                helper = historical.H27ReviewedSource(); helper.setUp(); helper.copy_root(root)
                fixture = root / self.fixture
                review = json.loads(fixture.read_text('utf8'))
                if mutation == 'source':
                    p = root / 'h3_t8/modular_sampling/reuse.py'
                    p.write_bytes(p.read_bytes() + b'\n# changed implementation\n')
                elif mutation == 'workflow':
                    (root / 'examples/workflows/97-nov09-reviewed/Cardigan_Model_Refine_5s.json').unlink()
                elif mutation == 'parent':
                    review['previous_files']['__init__.py'] = '0' * 64
                elif mutation == 'registration':
                    review['registered_nodes'] += 1
                else:
                    review['reviewed_sites'].append(dict(review['reviewed_sites'][-1], callee='future.sample'))
                fixture.write_text(json.dumps(review), 'utf8')
                with self.assertRaisesRegex(ValueError, 'source/workflow changed|November source review'):
                    gate._h27_review(root)

    def test_rehashed_index_cannot_borrow_a_different_accepted_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = historical.H27ReviewedSource(); helper.setUp(); helper.copy_root(root)
            fixture = root / self.fixture
            review = json.loads(fixture.read_text('utf8'))
            path = root / review['workflows_index']
            index = json.loads(path.read_text('utf8'))
            index['workflows'][0]['source_media_sha256'] = ['0' * 64]
            path.write_text(json.dumps(index), 'utf8')
            review['files'][review['workflows_index']] = hashlib.sha256(path.read_bytes()).hexdigest()
            fixture.write_text(json.dumps(review), 'utf8')
            with self.assertRaisesRegex(ValueError, 'different actual media'):
                gate._h27_review(root)


if __name__ == '__main__':
    unittest.main()
