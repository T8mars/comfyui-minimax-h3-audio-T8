"""Selected human-reviewed successor; reject unknown calls and changed assets."""
from copy import deepcopy
import hashlib
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


class H27ReviewedSource(unittest.TestCase):
    path = 'tests/fixtures/h27_reviewed_sampling_source.json'

    def setUp(self):
        self.review = json.loads((ROOT / self.path).read_text('utf8'))

    def copy_root(self, root):
        pika_path = 'tests/fixtures/pika_flex_source.json'
        pika = json.loads((ROOT / pika_path).read_text('utf8'))
        dual_path = 'tests/fixtures/pika_two_pass_source.json'
        dual = json.loads((ROOT / dual_path).read_text('utf8'))
        for name in [*self.review['files'], self.path, 'tests/fixtures/h21_reviewed_sampling_1_102.json',
                     'tests/fixtures/h24_reviewed_sampling_source.json', *pika['files'], pika_path,
                     *dual['files'], dual_path]:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)

    def test_current_exact_source_passes_unknown_and_mutated_calls_fail(self):
        baseline = json.loads(gate.BASELINE.read_text('utf8'))
        live = gate.discover(ROOT)
        with patch.object(gate, 'discover', return_value=live):
            result = gate.guard(ROOT, baseline)
            self.assertEqual((result['status'], result['new_sites'], result['removed_sites']), ('pass', 0, 0))
        changed = deepcopy(live)
        next(row for row in changed if row['path'] == 'h3_t8/director_generation.py')['call_sha256'] = '0' * 64
        with patch.object(gate, 'discover', return_value=changed):
            self.assertEqual(gate.guard(ROOT, baseline)['status'], 'fail')
        unknown = dict(live[-1], path='h3_t8/future_unreviewed.py', symbol='Future.sample')
        with patch.object(gate, 'discover', return_value=live + [unknown]):
            self.assertEqual(gate.guard(ROOT, baseline)['new_sites'], 1)

    def test_full_sources_and_workflows_are_required(self):
        with tempfile.TemporaryDirectory(prefix='h27-reviewed-') as directory:
            root = Path(directory)
            self.copy_root(root)
            self.assertIsNotNone(gate._h27_review(root))
            for name in ['h3_t8/director_generation.py', 'h3_t8/h3_memory_advanced.py',
                         'h3_t8/open_regenerate_recipe.py', self.review['workflow_files'][0]]:
                with self.subTest(name=name):
                    path = root / name
                    data = path.read_bytes()
                    path.write_bytes(data + b'\n# unreviewed\n')
                    with self.assertRaisesRegex(ValueError, 'source/workflow changed'):
                        gate._h27_review(root)
                    path.unlink()
                    with self.assertRaisesRegex(ValueError, 'source/workflow changed'):
                        gate._h27_review(root)
                    path.write_bytes(data)

    def test_prior_binding_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory(prefix='h27-parent-') as directory:
            root = Path(directory)
            self.copy_root(root)
            review = deepcopy(self.review)
            review['previous_files']['h3_t8/director_generation.py'] = '0' * 64
            (root / self.path).write_text(json.dumps(review), 'utf8')
            with self.assertRaisesRegex(ValueError, 'different prior source binding'):
                gate._h27_review(root)

    def test_rehashed_index_cannot_borrow_other_media(self):
        with tempfile.TemporaryDirectory(prefix='h27-media-') as directory:
            root = Path(directory)
            self.copy_root(root)
            name = 'examples/workflows/93-director-reviewed/index.json'
            path = root / name
            data = json.loads(path.read_text('utf8'))
            data['workflows'][0]['source_media_sha256'] = ['0' * 64]
            path.write_text(json.dumps(data), 'utf8')
            review = deepcopy(self.review)
            review['files'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root / self.path).write_text(json.dumps(review), 'utf8')
            with self.assertRaisesRegex(ValueError, 'different human media'):
                gate._h27_review(root)

    def test_out_of_project_binding_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='h27-path-') as directory:
            root = Path(directory)
            self.copy_root(root)
            review = deepcopy(self.review)
            review['files']['../outside.py'] = '0' * 64
            (root / self.path).write_text(json.dumps(review), 'utf8')
            with self.assertRaisesRegex(ValueError, 'source/workflow changed'):
                gate._h27_review(root)

    def test_unrelated_call_baseline_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'different baseline'):
            gate.reviewed_h27_sites(ROOT, [])


if __name__ == '__main__':
    unittest.main()
