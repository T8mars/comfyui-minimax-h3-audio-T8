"""Reviewed scalar pairs and real file-backed menu additions only."""
from copy import deepcopy

import pytest

from legacy_schema_review import ABSENT, SCALAR_REVIEWS, assert_reviewed_legacy_schemas, parent


def scalar_case(node_id):
    old = {'description': '', 'input': {'required': {
        'chunk_tokens': ['INT', {'default': 256, 'min': 1, 'max': 65536}],
        'model_name': ['COMBO', {'options': ['old.safetensors']}],
        'alpha': ['FLOAT', {'default': .1}],
    }}}
    new = deepcopy(old)
    for path, before, after in SCALAR_REVIEWS[node_id]:
        if before is ABSENT:
            parent(old, path).pop(path[-1], None)
        else:
            parent(old, path)[path[-1]] = before
        parent(new, path)[path[-1]] = after
    return [{'id': node_id, 'info': old}], {node_id: new}


@pytest.mark.parametrize('node_id', SCALAR_REVIEWS)
def test_only_exact_reviewed_scalar_pairs_pass_without_changing_snapshot_or_live(node_id, tmp_path):
    entries, info = scalar_case(node_id)
    before = deepcopy((entries, info))
    assert len(assert_reviewed_legacy_schemas(entries, info, tmp_path)) == len(SCALAR_REVIEWS[node_id])
    assert (entries, info) == before


@pytest.mark.parametrize('node_id', SCALAR_REVIEWS)
@pytest.mark.parametrize('side', ['old', 'new'])
def test_close_but_unreviewed_description_is_rejected(node_id, side, tmp_path):
    entries, info = scalar_case(node_id)
    target = entries[0]['info'] if side == 'old' else info[node_id]
    target['description'] += ' unreviewed'
    with pytest.raises(AssertionError):
        assert_reviewed_legacy_schemas(entries, info, tmp_path)


@pytest.mark.parametrize('change', ['alpha_default', 'chunk_default', 'min_below_zero',
                                   'float_min', 'unreviewed_tooltip', 'socket', 'extra', 'missing_node'])
def test_defaults_or_other_unreviewed_schema_changes_are_not_projected(change, tmp_path):
    node_id = 'MiniMaxH3SemanticBridgeConfigT8'
    entries, info = scalar_case(node_id)
    fields = info[node_id]['input']['required']
    if change == 'alpha_default':
        fields['alpha'][1]['default'] = 1.
    elif change == 'chunk_default':
        fields['chunk_tokens'][1]['default'] = 0
    elif change == 'min_below_zero':
        fields['chunk_tokens'][1]['min'] = -1
    elif change == 'float_min':
        fields['chunk_tokens'][1]['min'] = 0.
    elif change == 'unreviewed_tooltip':
        fields['chunk_tokens'][1]['tooltip'] += 'changed'
    elif change == 'socket':
        fields['chunk_tokens'][0] = 'FLOAT'
    elif change == 'extra':
        info[node_id]['hidden_changed'] = True
    else:
        del info[node_id]
    with pytest.raises(AssertionError):
        assert_reviewed_legacy_schemas(entries, info, tmp_path)


@pytest.mark.parametrize('change', ['valid', 'missing_file', 'removed', 'reordered', 'duplicate',
                                   'escape', 'generic_model_name', 'new_default'])
def test_only_ordered_physical_assets_of_the_exact_declared_menu_owner_are_reviewed(change, tmp_path):
    node_id = 'MiniMaxH3SemanticBridgeConfigT8'
    entries, info = scalar_case(node_id)
    old_options = ['old.safetensors', 'second.safetensors']
    entries[0]['info']['input']['required']['model_name'][1]['options'] = old_options
    options = ['old.safetensors', 'new.safetensors', 'second.safetensors']
    root = tmp_path / 'semantic_bridge'
    root.mkdir()
    (root / 'new.safetensors').write_bytes(b'tiny menu-presence fixture, not a model')
    if change == 'missing_file':
        options[1] = 'absent.safetensors'
    elif change == 'removed':
        options = ['new.safetensors', 'second.safetensors']
    elif change == 'reordered':
        options = ['second.safetensors', 'old.safetensors', 'new.safetensors']
    elif change == 'duplicate':
        options.append('new.safetensors')
    elif change == 'escape':
        (tmp_path / 'outside.safetensors').write_bytes(b'outside')
        options[1] = '../outside.safetensors'
    elif change == 'generic_model_name':
        entries[0]['id'] = 'ForeignModelLoader'
        info = {'ForeignModelLoader': deepcopy(entries[0]['info'])}
        node_id = 'ForeignModelLoader'
    elif change == 'new_default':
        info[node_id]['input']['required']['model_name'][1]['default'] = 'new.safetensors'
    info[node_id]['input']['required']['model_name'][1]['options'] = options
    if change == 'valid':
        changes = assert_reviewed_legacy_schemas(entries, info, tmp_path)
        assert changes[-1]['category'] == 'semantic_bridge' and changes[-1]['added'] == ['new.safetensors']
    else:
        with pytest.raises((AssertionError, ValueError)):
            assert_reviewed_legacy_schemas(entries, info, tmp_path)
