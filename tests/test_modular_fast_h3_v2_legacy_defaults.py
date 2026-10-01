"""Exact legacy widget tails, negative graph checks, and actual port defaults."""
from copy import deepcopy
import inspect

import pytest

from legacy_v2_frontend import REVIEWED_OPTIONAL_TAILS, assert_legacy_v2_frontend
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_continuation as ports
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_continuation_nodes as nodes
from h3_audio_t8_pkg.modular_sampling import fast_h3_v2_job as job
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg import progressive_continuation_relay as legacy_relay
from test_modular_fast_h3_v2_continuation_ports import contexts_for
from test_progressive_continuation import accepted as accepted_fixture
from test_modular_fast_h3_v2_job import inputs as recipe_inputs_fixture

accepted = accepted_fixture
recipe_inputs = recipe_inputs_fixture


def graphs(kind):
    count, field, default = REVIEWED_OPTIONAL_TAILS[kind]
    old = {'id': 'old-id', 'nodes': [{'id': 1, 'type': kind,
        'widgets_values': list(range(count)), 'inputs': [{'name': 'original', 'link': 7}],
        'flags': {}, 'mode': 0}], 'links': [[7, 2, 0, 1, 0, 'ORIGINAL']]}
    new = deepcopy(old)
    new['id'] = 'new-id'
    new['nodes'][0]['widgets_values'].append(default)
    info = {kind: {'input': {'optional': {field: ['STRING' if type(default) is str else 'INT',
                                                {'default': default}]}}}}
    return old, new, info


@pytest.mark.parametrize('kind', REVIEWED_OPTIONAL_TAILS)
def test_only_reviewed_type_exact_default_tail_and_random_uuid_are_projected(kind):
    old, new, info = graphs(kind)
    before = deepcopy((old, new, info))
    assert_legacy_v2_frontend(old, new, info)
    assert (old, new, info) == before


@pytest.mark.parametrize('kind', REVIEWED_OPTIONAL_TAILS)
@pytest.mark.parametrize('change', ['nondefault', 'extra', 'old_widget', 'edge',
                                   'socket', 'mode', 'schema_default', 'new_optional',
                                   'missing_tail', 'duplicate_id'])
def test_other_graph_or_schema_changes_are_not_hidden(kind, change):
    old, new, info = graphs(kind)
    node = new['nodes'][0]
    if change == 'nondefault':
        node['widgets_values'][-1] = 'old_fixed_124' if type(node['widgets_values'][-1]) is str else 124
    elif change == 'extra':
        node['widgets_values'].append('unreviewed')
    elif change == 'old_widget':
        node['widgets_values'][0] = 'changed'
    elif change == 'edge':
        new['links'][0][1] = 99
    elif change == 'socket':
        node['inputs'][0]['name'] = 'changed'
    elif change == 'mode':
        node['mode'] = 4
    elif change == 'schema_default':
        next(iter(info[kind]['input']['optional'].values()))[1]['default'] = 'changed'
    elif change == 'new_optional':
        info[kind]['input']['optional']['unreviewed'] = ['INT', {'default': 0}]
    elif change == 'missing_tail':
        node['widgets_values'].pop()
    else:
        new['nodes'].append(deepcopy(node))
    with pytest.raises(AssertionError):
        assert_legacy_v2_frontend(old, new, info)


@pytest.mark.parametrize('kind', ['MiniMaxH3FastH3V2AcceptedRelayProjectEXPT8',
                                 'MiniMaxH3FastH3V2CurrentRecipeEXPT8'])
def test_numeric_equal_float_default_is_not_an_int_widget(kind):
    old, new, info = graphs(kind)
    new['nodes'][0]['widgets_values'][-1] = float(new['nodes'][0]['widgets_values'][-1])
    with pytest.raises(AssertionError):
        assert_legacy_v2_frontend(old, new, info)


def test_omitted_and_explicit_original_render_defaults_have_exact_port_outputs(accepted):
    contexts = contexts_for(accepted)
    for port, cls in ((ports.plan_accepted_window, nodes.MiniMaxH3FastH3V2AcceptedWindowEXPT8),
                      (ports.accepted_delivery_window, nodes.MiniMaxH3FastH3V2AcceptedDeliveryWindowEXPT8)):
        assert inspect.signature(port).parameters['render_policy'].default == 'compact_remainder'
        assert inspect.signature(cls.execute).parameters['render_policy'].default == 'compact_remainder'
        assert port(contexts, 192) == port(contexts, 192, 'compact_remainder')
        assert cls.execute(contexts, 192).result == cls.execute(contexts, 192, 'compact_remainder').result
    assert ports.plan_accepted_window(contexts, 192)[:2] == (90, 68)
    assert ports.plan_accepted_window(contexts, 192, 'old_fixed_124')[:2] == (124, 68)


def test_omitted_recipe_default_preserves_original_payload_and_sha(recipe_inputs):
    omitted = job.make_current_recipe(**recipe_inputs)
    explicit = job.make_current_recipe(**recipe_inputs, continuation_render_frames=90)
    assert omitted == explicit
    assert 'continuation_render_frames' not in omitted[0].verify()['window']
    assert job.make_current_recipe(**recipe_inputs, continuation_render_frames=124)[1] != omitted[1]


def test_omitted_zero_relay_end_keeps_original_projection_and_source_plan(accepted):
    contexts = contexts_for(accepted)
    plan = relay.build_prompt_relay_plan('Scene.', 'Walk.\nStop.\nTurn.', 193,
        'auto_equal', '', 'paper_v1', .1, False, False)[0]
    before = deepcopy(plan)
    omitted = ports.project_accepted_relay(contexts, plan, 90)
    assert omitted == ports.project_accepted_relay(contexts, plan, 90, 0)
    assert omitted[0] == legacy_relay.project_for_source(contexts.source, plan, 90)
    cls = nodes.MiniMaxH3FastH3V2AcceptedRelayProjectEXPT8
    assert inspect.signature(cls.execute).parameters['accepted_end_frame'].default == 0
    assert inspect.signature(ports.project_accepted_relay).parameters['accepted_end_frame'].default == 0
    assert cls.execute(contexts, plan, 90).result == cls.execute(contexts, plan, 90, 0).result
    assert plan == before
