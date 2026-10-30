"""Exact reviewed successor identities; no Core/GPU dependency or blanket admission."""
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


class H24ReviewedSource(unittest.TestCase):
    def setUp(self):
        self.baseline = json.loads(gate.BASELINE.read_text(encoding='utf8'))
        self.path = 'tests/fixtures/h24_reviewed_sampling_source.json'
        self.review = json.loads((ROOT / self.path).read_text(encoding='utf8'))

    def copy_root(self, root):
        h27_path = 'tests/fixtures/h27_reviewed_sampling_source.json'
        h27 = json.loads((ROOT / h27_path).read_text(encoding='utf8'))
        pika_path = 'tests/fixtures/pika_flex_source.json'
        pika = json.loads((ROOT / pika_path).read_text(encoding='utf8'))
        for name in [*self.review['files'], self.path, 'tests/fixtures/h21_reviewed_sampling_1_102.json',
                     *h27['files'], h27_path, *pika['files'], pika_path]:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)

    def test_exact_current_delta_passes_and_unknown_or_mutated_calls_fail(self):
        self.assertEqual(gate.guard(ROOT, self.baseline)['status'], 'pass')
        live = gate.discover(ROOT)
        changed = deepcopy(live)
        selected = next(row for row in changed if row['path'] == self.review['reviewed_sites'][0]['path'])
        selected['call_sha256'] = '0' * 64
        with patch.object(gate, 'discover', return_value=changed):
            self.assertEqual(gate.guard(ROOT, self.baseline)['status'], 'fail')
        unknown = dict(live[-1], path='h3_t8/unreviewed_future.py', symbol='Future.sample')
        with patch.object(gate, 'discover', return_value=live + [unknown]):
            self.assertEqual(gate.guard(ROOT, self.baseline)['new_sites'], 1)

    def test_changed_and_missing_sources_fail_without_reusing_human_acceptance(self):
        with tempfile.TemporaryDirectory(prefix='h24-source-review-') as directory:
            root = Path(directory)
            self.copy_root(root)
            self.assertIsNotNone(gate._h24_review(root))
            target = root / 'h3_t8/director_generation.py'
            original = target.read_bytes()
            target.write_bytes(original + b'\n# unreviewed change\n')
            with self.assertRaisesRegex(ValueError, 'source/workflow changed'):
                gate._h24_review(root)
            target.write_bytes(original)
            target.unlink()
            with self.assertRaisesRegex(ValueError, 'source/workflow changed'):
                gate._h24_review(root)

    def test_previous_hash_cannot_relabel_a_different_release(self):
        with tempfile.TemporaryDirectory(prefix='h24-parent-review-') as directory:
            root = Path(directory)
            self.copy_root(root)
            changed = deepcopy(self.review)
            changed['previous_files']['h3_t8/director_generation.py'] = '0' * 64
            (root / self.path).write_text(json.dumps(changed), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'different H21 source binding'):
                gate._h24_review(root)

    def test_reviewed_bytes_cannot_be_applied_to_an_unrelated_call_baseline(self):
        with self.assertRaisesRegex(ValueError, 'different baseline'):
            gate.reviewed_h24_sites(ROOT, [])


if __name__ == '__main__':
    unittest.main()
