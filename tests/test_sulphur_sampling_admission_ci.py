"""Stdlib source/graph tripwires, separate from runtime and media acceptance."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
from tools import audit_modular_new_sampling as gate
from test_h27_sampling_admission_ci import H27ReviewedSource


class SulphurSource(unittest.TestCase):
    def test_exact_source_and_existing_sampling_sites(self):
        result = gate.guard(ROOT, json.loads(gate.BASELINE.read_text('utf-8')))
        self.assertEqual((result['status'], result['current_sites'], result['new_sites'], result['removed_sites']),
                         ('pass', 128, 0, 0), result['issues'])

    def test_changed_source_workflow_parent_and_false_quality_fail(self):
        for fault in ['network', 'workflow', 'parent', 'quality', 'nodes', 'extra_source']:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                helper = H27ReviewedSource()
                helper.setUp()
                helper.copy_root(root)
                path = root/'tests/fixtures/sulphur_corrector_source.json'
                record = json.loads(path.read_text('utf-8'))
                if fault in {'network', 'workflow'}:
                    target = root/('h3_t8/sulphur_corrector_network.py' if fault == 'network'
                                   else 'examples/workflows/99-sulphur3/S3_Corrector_T2VA_EXP_API.json')
                    target.write_bytes(target.read_bytes() + b'\n ')
                elif fault == 'parent':
                    record['previous_files']['__init__.py'] = '0' * 64
                elif fault == 'quality':
                    record['human_quality_accepted'] = True
                elif fault == 'nodes':
                    record['registered_nodes'] += 1
                else:
                    record['files']['h3_t8/future_unknown_corrector.py'] = '0' * 64
                path.write_text(json.dumps(record), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'source/workflow changed|bounded Sulphur'):
                    gate._h27_review(root)

    def test_example_api_ui_are_the_same_recipe_and_not_pika_or_lora(self):
        directory = ROOT/'examples/workflows/99-sulphur3'
        workflow = json.loads((directory/'S3_Corrector_T2VA_EXP.json').read_text('utf-8'))
        api = json.loads((directory/'S3_Corrector_T2VA_EXP_API.json').read_text('utf-8'))
        nodes = {node['id']: node for node in workflow['nodes']}
        self.assertEqual(len(nodes), 19)
        for edge in workflow['links']:
            number, source, slot, target, target_slot, kind = edge
            self.assertEqual(nodes[source]['outputs'][slot]['type'], kind)
            self.assertIn(number, nodes[source]['outputs'][slot]['links'])
            self.assertEqual(nodes[target]['inputs'][target_slot]['link'], number)
            target_name = nodes[target]['inputs'][target_slot]['name']
            self.assertEqual(api[str(target)]['inputs'][target_name], [str(source), slot])
        for number, node in nodes.items():
            if node['type'] == 'MarkdownNote':
                continue
            self.assertEqual(api[str(number)]['class_type'], node['type'])
            for key, value in node.get('widgets_values_named', {}).items():
                if key != 'control_after_generate':
                    self.assertEqual(api[str(number)]['inputs'][key], value)
        self.assertEqual(api['18']['inputs']['steps'], 20)
        self.assertEqual(api['19']['inputs']['sampler_name'], 'euler')
        self.assertEqual(api['10']['class_type'], 'BasicGuider')
        self.assertFalse(any('Pika' in x['type'] or 'LoRA' in x['type'] for x in nodes.values()))
        self.assertEqual(api['9']['inputs']['corrector'], ['2', 0])
        self.assertEqual(api['12']['inputs']['sigmas'], ['18', 0])


if __name__ == '__main__':
    unittest.main()
