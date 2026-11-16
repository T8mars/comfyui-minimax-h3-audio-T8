"""Stdlib-only finite release binding, including negative source mutations."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_modular_new_sampling as gate  # noqa: E402
import test_h27_sampling_admission_ci as historical  # noqa: E402


class Release1106Source(unittest.TestCase):
    def test_exact_count_and_no_new_runtime_or_media_admission(self):
        review = json.loads((ROOT / 'tests/fixtures/release_1_106_sampling_source.json').read_text('utf8'))
        self.assertEqual(review['runtime_sites_added'], 0)
        self.assertFalse(review['human_quality_accepted'])
        result = gate.guard(ROOT, json.loads(gate.BASELINE.read_text('utf8')))
        self.assertEqual((result['status'], result['current_sites'], result['new_sites'], result['removed_sites']),
                         ('pass', 128, 0, 0))

    def test_source_parent_extra_runtime_and_borrowed_media_fail(self):
        for kind in ('source', 'workflow', 'parent', 'extra', 'media'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                old = historical.H27ReviewedSource()
                old.setUp()
                old.copy_root(root)
                path = root / 'tests/fixtures/release_1_106_sampling_source.json'
                review = json.loads(path.read_text('utf8'))
                if kind in {'source', 'workflow'}:
                    selected = root / ('tests/test_h09_sampling_admission_ci.py' if kind == 'source'
                                       else '.github/workflows/modular-sampling-admission.yml')
                    selected.write_bytes(selected.read_bytes() + b'\n# unreviewed\n')
                elif kind == 'parent':
                    review['previous_files']['tools/audit_modular_new_sampling.py'] = '0' * 64
                elif kind == 'extra':
                    review['files']['h3_t8/unreviewed_sampler.py'] = '0' * 64
                else:
                    review['human_quality_accepted'] = True
                path.write_text(json.dumps(review), 'utf8')
                with self.assertRaisesRegex(ValueError, 'source/workflow changed|bounded release1106'):
                    gate._h27_review(root)


if __name__ == '__main__':
    unittest.main()
