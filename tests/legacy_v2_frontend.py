"""Exact saved V2 graph comparison with four reviewed optional widget tails.

Historical graphs are never rewritten. Only the named, type-exact current
schema default appended after the exact historical widget count is projected.
All other graph data (including original widgets and named edges) stays exact.
Execution-default equivalence is tested separately against the actual ports.
"""
from copy import deepcopy


REVIEWED_OPTIONAL_TAILS = {
    'MiniMaxH3FastH3V2AcceptedWindowEXPT8': (1, 'render_policy', 'compact_remainder'),
    'MiniMaxH3FastH3V2AcceptedDeliveryWindowEXPT8': (1, 'render_policy', 'compact_remainder'),
    'MiniMaxH3FastH3V2AcceptedRelayProjectEXPT8': (1, 'accepted_end_frame', 0),
    'MiniMaxH3FastH3V2CurrentRecipeEXPT8': (7, 'continuation_render_frames', 90),
}


def assert_legacy_v2_frontend(saved, current, info):
    old, new = deepcopy(saved), deepcopy(current)
    # Existing builders generate a new workflow UUID; no other field is ignored.
    old.pop('id', None)
    new.pop('id', None)
    old_nodes = {node['id']: node for node in old['nodes']}
    assert len(old_nodes) == len(old['nodes'])
    assert len({node['id'] for node in new['nodes']}) == len(new['nodes'])
    for node in new['nodes']:
        previous = old_nodes.get(node['id'])
        reviewed = REVIEWED_OPTIONAL_TAILS.get(node['type'])
        if previous is None or reviewed is None:
            continue
        assert previous['type'] == node['type']
        count, field, default = reviewed
        optional = info[node['type']]['input']['optional']
        assert list(optional) == [field]
        actual_default = optional[field][1]['default']
        assert type(actual_default) is type(default) and actual_default == default
        before, after = previous.get('widgets_values'), node.get('widgets_values')
        if type(before) is list and len(before) == count:
            assert type(after) is list and len(after) == count + 1
            assert type(after[-1]) is type(default) and after[-1] == default
            node['widgets_values'] = after[:-1]
    assert old == new
