"""Finite compatibility successor and preserved unknown-source rejection."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from tools import audit_modular_new_sampling as gate  # noqa: E402
import test_h27_sampling_admission_ci as predecessor  # noqa: E402


class CoreCompatAdmission(unittest.TestCase):
    def test_exact_source_rejects_mutation_missing_prior_and_quality_relabelling(self):
        path='tests/fixtures/core_compat_20261031_source.json'
        for mutation in ['source','missing','parent','quality','extra_site']:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                helper=predecessor.H27ReviewedSource()
                helper.setUp()
                helper.copy_root(root)
                self.assertIsNotNone(gate._h27_review(root))
                review=deepcopy(json.loads((root/path).read_text('utf8')))
                target=root/'h3_t8/long_video_dual_identity.py'
                if mutation=='source':
                    target.write_bytes(target.read_bytes()+b'\n# changed execution identity\n')
                elif mutation=='missing':
                    target.unlink()
                elif mutation=='parent':
                    review['previous_files']['h3_t8/long_video_dual_identity.py']='0'*64
                elif mutation=='quality':
                    review['human_quality_accepted']=True
                else:
                    review['sampling_site_delta']=[{'callee':'future.sample'}]
                (root/path).write_text(json.dumps(review),'utf8')
                with self.assertRaisesRegex(ValueError,'source/workflow changed|compatibility source review'):
                    gate._h27_review(root)


if __name__=='__main__':
    unittest.main()
