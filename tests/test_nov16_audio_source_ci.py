import json
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import audit_modular_new_sampling as gate
import test_h27_sampling_admission_ci as historical


ROOT = Path(__file__).resolve().parents[1]


class Nov16AudioSource(unittest.TestCase):
    def test_finite_audio_successor_does_not_approve_media_and_binds_both_delegates(self):
        baseline = json.loads(gate.BASELINE.read_text('utf8'))
        self.assertEqual(gate.guard(ROOT, baseline)['status'], 'pass')
        review = json.loads((ROOT / 'tests/fixtures/nov16_audio_handoff_source.json').read_text('utf8'))
        self.assertFalse(review['human_quality_accepted'])
        self.assertEqual(len(review['reviewed_sites']), 2)

    def test_changed_helper_and_wrong_predecessor_or_append_order_fail(self):
        for kind in ('helper', 'parent', 'node'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                old = historical.H27ReviewedSource()
                old.setUp()
                old.copy_root(root)
                path = root / 'tests/fixtures/nov16_audio_handoff_source.json'
                review = json.loads(path.read_text('utf8'))
                if kind == 'helper':
                    selected = root / 'h3_t8/two_pass_audio_continuation.py'
                    selected.write_bytes(selected.read_bytes() + b'\n# unreviewed\n')
                elif kind == 'parent':
                    review['previous_files']['__init__.py'] = '0' * 64
                else:
                    review['appended_nodes'] = ['SomeOtherNode']
                path.write_text(json.dumps(review), 'utf8')
                with self.assertRaisesRegex(ValueError, 'source/workflow changed|bounded Nov16'):
                    gate._h27_review(root)


if __name__ == '__main__':
    unittest.main()
