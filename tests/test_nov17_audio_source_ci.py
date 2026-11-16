import json
from pathlib import Path
import tempfile
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import audit_modular_new_sampling as gate
import test_h27_sampling_admission_ci as historical


ROOT = Path(__file__).resolve().parents[1]


def _assert_current_audio_clock_source():
    baseline = json.loads(gate.BASELINE.read_text('utf-8'))
    result = gate.guard(ROOT, baseline)
    assert result['status'] == 'pass' and result['new_sites'] == result['removed_sites'] == 0
    current = json.loads((ROOT / 'tests/fixtures/nov17_audio_clock_source.json').read_text('utf-8'))
    assert current['human_quality_accepted'] is False
    assert len(current['reviewed_sites']) == 1


def _assert_invalid_clock_successor(kind):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        previous = historical.H27ReviewedSource()
        previous.setUp()
        previous.copy_root(root)
        path = root / 'tests/fixtures/nov17_audio_clock_source.json'
        review = json.loads(path.read_text('utf-8'))
        if kind == 'helper':
            source = root / 'h3_t8/two_pass_audio_continuation.py'
            source.write_bytes(source.read_bytes() + b'\n# unreviewed\n')
        elif kind == 'parent':
            review['previous_files']['h3_t8/two_pass_audio_continuation.py'] = '0' * 64
        elif kind == 'callsite':
            review['reviewed_sites'][0]['callee'] = 'other_sampler'
        else:
            review['files']['h3_t8/other_sampler.py'] = '0' * 64
        path.write_text(json.dumps(review), 'utf-8')
        with unittest.TestCase().assertRaisesRegex(ValueError, 'source/workflow changed|bounded Nov17'):
            gate._h27_review(root)


class Nov17AudioSource(unittest.TestCase):
    def test_current_source(self):
        _assert_current_audio_clock_source()

    def test_all_four_negative_mutations(self):
        for kind in ('helper', 'parent', 'callsite', 'extra_file'):
            with self.subTest(kind=kind):
                _assert_invalid_clock_successor(kind)


if __name__ == '__main__':
    unittest.main()
