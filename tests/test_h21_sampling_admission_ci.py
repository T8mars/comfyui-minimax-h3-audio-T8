"""Dependency-free exact release bindings; unknown source is still rejected."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_modular_new_sampling as gate  # noqa: E402


class H21ReleaseAdmission(unittest.TestCase):
    def setUp(self):
        self.baseline = json.loads(gate.BASELINE.read_text(encoding='utf8'))
        self.review = json.loads((ROOT / 'tests/fixtures/h21_reviewed_sampling_1_102.json').read_text(encoding='utf8'))

    def test_live_reviewed_delta_and_future_call_rejection(self):
        live = gate.discover(ROOT)
        result = gate.guard(ROOT, self.baseline)
        self.assertEqual(result['status'], 'pass', result['issues'])
        self.assertEqual((result['new_sites'], result['removed_sites']), (0, 0))
        changed = deepcopy(live)
        selected = next(row for row in changed if row['path'] == self.review['reviewed_sites'][0]['path'])
        selected['call_sha256'] = '0' * 64
        with patch.object(gate, 'discover', return_value=changed):
            self.assertEqual(gate.guard(ROOT, self.baseline)['status'], 'fail')
        unknown = dict(live[-1], path='h3_t8/not_reviewed.py', symbol='Unexpected.sample')
        with patch.object(gate, 'discover', return_value=live + [unknown]):
            self.assertEqual(gate.guard(ROOT, self.baseline)['new_sites'], 1)

    def test_missing_or_changed_full_source_never_borrows_release_acceptance(self):
        old, _ = gate.reviewed_layout_sites(ROOT, self.baseline)
        old = gate.reviewed_model_choice_sites(ROOT, old)
        with tempfile.TemporaryDirectory(prefix='h21-source-admission-') as temporary:
            root = Path(temporary)
            for relative in [*self.review['files'], 'tests/fixtures/h21_reviewed_sampling_1_102.json']:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, target)
            self.assertEqual(len(gate.reviewed_h21_sites(root, old)), len(old) + 3)
            relative = next(iter(self.review['files']))
            target = root / relative
            original = target.read_bytes()
            target.write_bytes(original + b'\n# unreviewed mutation\n')
            with self.assertRaisesRegex(ValueError, 'source/workflow changed'):
                gate.reviewed_h21_sites(root, old)
            target.write_bytes(original)
            with self.assertRaisesRegex(ValueError, 'different baseline'):
                gate.reviewed_h21_sites(root, [])


if __name__ == '__main__':
    unittest.main()
