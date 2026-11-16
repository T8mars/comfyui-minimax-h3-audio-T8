"""Finite development successor; old human approvals cannot approve new media."""
from copy import deepcopy
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


class Nov12Source(unittest.TestCase):
    fixture = 'tests/fixtures/nov12_sampling_source.json'

    def test_current_delta_and_unchanged_old_ledgers_pass(self):
        baseline = json.loads(gate.BASELINE.read_text('utf8'))
        self.assertEqual(gate.guard(ROOT, baseline)['status'], 'pass')
        review = json.loads((ROOT / self.fixture).read_text('utf8'))
        self.assertFalse(review['human_quality_accepted'])
        self.assertEqual(review['accepted_media_sha256'], [])
        self.assertEqual(review['human_feedback_sha256'], [])

    def test_new_delegated_call_multiplicity_and_unreviewed_sources_fail(self):
        baseline = json.loads(gate.BASELINE.read_text('utf8'))
        live = gate.discover(ROOT)
        selected = next(r for r in live if r['path'] == 'h3_t8/prompt_edit_guidance.py')
        for rows in [live + [deepcopy(selected)], [r for r in live if r is not selected],
                     live + [dict(selected, path='h3_t8/unknown_future.py')]]:
            with self.subTest(count=len(rows)), patch.object(gate, 'discover', return_value=rows):
                self.assertEqual(gate.guard(ROOT, baseline)['status'], 'fail')

    def test_mutated_helper_or_detached_binding_fails(self):
        for mutation in ('source', 'detached', 'parent', 'prefix', 'order', 'media', 'call'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                helper = historical.H27ReviewedSource()
                helper.setUp()
                helper.copy_root(root)
                path = root / self.fixture
                review = json.loads(path.read_text('utf8'))
                if mutation == 'source':
                    selected = root / 'h3_t8/lora_windows.py'
                    selected.write_bytes(selected.read_bytes() + b'\n# unreviewed\n')
                elif mutation == 'detached':
                    review['files'].pop('h3_t8/lora_windows.py')
                elif mutation == 'parent':
                    review['previous_files']['h3_t8/director_hyperflow.py'] = '0' * 64
                elif mutation == 'prefix':
                    review['preserved_prefix_sha256'] = '0' * 64
                elif mutation == 'order':
                    review['appended_nodes'].reverse()
                elif mutation == 'media':
                    review['human_quality_accepted'] = True
                    review['accepted_media_sha256'] = ['0' * 64]
                else:
                    review['reviewed_sites'].append(deepcopy(review['reviewed_sites'][-1]))
                path.write_text(json.dumps(review), 'utf8')
                with self.assertRaisesRegex(ValueError, 'source/workflow changed|bounded Nov12'):
                    gate._h27_review(root)


if __name__ == '__main__':
    unittest.main()
