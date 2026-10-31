"""Multi-bridge Core validation/dispatch and native AV software compatibility.

Fixtures use random adapter and tiny H3 weights; no video quality qualification.
"""
import asyncio
import json

import pytest
import torch

import execution
import nodes as core_nodes
from h3_audio_t8_pkg import nodes_semantic_bridge as bridge
from h3_audio_t8_pkg import semantic_bridge as sb
from test_semantic_bridge_profiles_stack import native, saved_models  # noqa: F401
from test_semantic_bridge_native_pipeline import run_bridge_pipeline


async def core_run(cls, **values):
    cls.VALIDATE_CLASS()
    data, ui, subgraph, pending = await execution.get_output_data(
        'stack-compat', cls.__name__, cls(), {key: [value] for key, value in values.items()})
    assert not subgraph and not pending
    return [item[0] for item in data]


@pytest.mark.parametrize('names', [('original', 'bunny'), ('original', 'comic'),
                                  ('comic', 'original')])
@pytest.mark.parametrize('task', ['t2va', 'ref2va'])
def test_linked_multibridge_graph_Core_dispatch_and_native_AV(saved_models, monkeypatch, names, task):  # noqa: F811
    class NativeConditioning:
        @classmethod
        def INPUT_TYPES(cls):
            return {'required': {}}
        RETURN_TYPES = ('CONDITIONING',)
        FUNCTION = 'run'

        def run(self):
            return (native(),)

    for cls in [bridge.MiniMaxH3SemanticBridgeAutoConfigT8,
                bridge.MiniMaxH3SemanticBridgeComposeT8,
                bridge.MiniMaxH3SemanticBridgeApplyT8, NativeConditioning]:
        monkeypatch.setitem(core_nodes.NODE_CLASS_MAPPINGS, cls.__name__, cls)
    graph = {
        '1': {'class_type': 'MiniMaxH3SemanticBridgeAutoConfigT8',
              'inputs': {'model_name': names[0], 'enabled': True, 'device': 'cpu'}},
        '2': {'class_type': 'MiniMaxH3SemanticBridgeAutoConfigT8',
              'inputs': {'model_name': names[1], 'enabled': True, 'device': 'cpu'}},
        '3': {'class_type': 'MiniMaxH3SemanticBridgeComposeT8',
              'inputs': {'first': ['1', 0], 'second': ['2', 0], 'enabled': True}},
        '4': {'class_type': 'NativeConditioning', 'inputs': {}},
        '5': {'class_type': 'MiniMaxH3SemanticBridgeApplyT8',
              'inputs': {'conditioning': ['4', 0], 'semantic_bridge': ['3', 0]}},
    }
    assert asyncio.run(execution.validate_inputs('stack-compat', graph, '5', {}))[0] is True
    first = asyncio.run(core_run(bridge.MiniMaxH3SemanticBridgeAutoConfigT8,
                                **graph['1']['inputs']))[0]
    second = asyncio.run(core_run(bridge.MiniMaxH3SemanticBridgeAutoConfigT8,
                                 **graph['2']['inputs']))[0]
    stack, composition = asyncio.run(core_run(bridge.MiniMaxH3SemanticBridgeComposeT8,
                                             first=first, second=second, enabled=True))
    assert json.loads(composition)['identity'] == stack.identity()
    assert [item.alpha for item in stack.bridges] == [first.alpha, second.alpha]
    source = native()
    actual, report = asyncio.run(core_run(bridge.MiniMaxH3SemanticBridgeApplyT8,
                                         conditioning=source, semantic_bridge=stack))
    expected, _ = sb.apply_bridge(source, stack)
    assert torch.equal(actual[0][0], expected[0][0])
    assert sb.RECEIPT_KEY not in source[0][1]
    assert len(json.loads(report)['items'][0]['stages']) == 2
    full_report, output = run_bridge_pipeline(stack, task)
    assert len(full_report['items'][0]['stages']) == 2
    assert not torch.cuda.is_initialized()
