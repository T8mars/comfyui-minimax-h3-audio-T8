from copy import deepcopy
import uuid

import pytest

from h3_audio_t8_pkg.director_project import compile_project, new_project, validate_project
from h3_audio_t8_pkg.director_radar import (
    CANDIDATE_COMPONENTS, _inputs, apply_operation, candidate_diagnostics, candidate_status,
)


def candidate_project():
    project = new_project()
    shot = project['doc']['shots'][0]
    shot['simplePrompt'] = '你好，今天真不错。'
    return apply_operation(project, shot['id'], 'candidate_create', {'facts': False, 'intent': True})


def test_new_digests_are_complete_and_diagnostics_do_not_modify_or_adopt():
    project = candidate_project()
    before = deepcopy(project)
    shot = project['doc']['shots'][0]
    assert set(shot['promptCandidate']['receipt']['input_component_sha256']) == CANDIDATE_COMPONENTS
    result = candidate_diagnostics(project, shot)
    assert result['status'] == 'pending' and result['inputs_hash_match']
    assert result['changed_components'] == []
    assert result['saved'] is result['queued'] is result['adopted'] is False
    assert result['cache_hit_certified'] is False
    assert project == before


@pytest.mark.parametrize('kind', ['global', 'draft', 'references'])
def test_exact_changed_input_component_without_changing_stale_compilation_policy(kind):
    project = candidate_project()
    shot = project['doc']['shots'][0]
    candidate = shot['promptCandidate']
    project = apply_operation(project, shot['id'], 'candidate_accept',
                              {'confirm': True, 'text_sha256': candidate['text_sha256']})
    shot = project['doc']['shots'][0]
    if kind == 'global':
        project['doc']['global'] = '新全局稿'
    elif kind == 'draft':
        shot['simplePrompt'] = '修改原作者稿'
    else:
        # The map order is part of the existing aggregate key, even before
        # compilation points out a missing asset. Diagnostics is not a loader.
        shot['refs'] = [str(uuid.uuid4())]
    result = candidate_diagnostics(project, shot)
    assert result['status'] == candidate_status(project, shot) == 'stale'
    # Adding a referenced ID also changes the actual referenced-asset SHA
    # projection (including the missing/null entry); both consumed inputs must
    # be explained. Do not hide the second change to make a one-field test pass.
    expected = ['assets', 'references'] if kind == 'references' else [kind]
    assert result['changed_components'] == expected
    assert result['inputs_hash_match'] is False
    assert result['cache_hit_certified'] is False


def test_legacy_candidates_stay_accepted_and_aggregate_stale_without_invented_diffs():
    project = candidate_project()
    shot = project['doc']['shots'][0]
    del shot['promptCandidate']['receipt']['input_component_sha256']
    validate_project(project)
    project = apply_operation(project, shot['id'], 'candidate_accept',
        {'confirm': True, 'text_sha256': shot['promptCandidate']['text_sha256']})
    shot = project['doc']['shots'][0]
    before = deepcopy(project)
    assert candidate_diagnostics(project, shot)['status'] == 'active'
    assert compile_project(project) == compile_project(before)
    assert project == before
    project['doc']['global'] = '用户改动'
    result = candidate_diagnostics(project, shot)
    assert result['status'] == 'stale' and result['changed_components'] is None
    assert result['component_snapshot'] == 'legacy_missing'


def test_future_third_shot_edit_does_not_rewrite_first_two_candidate_receipts():
    project = new_project()
    template = project['doc']['shots'][0]
    project['doc']['shots'] += [{**deepcopy(template), 'id': str(uuid.uuid4())} for _ in range(2)]
    ids = [shot['id'] for shot in project['doc']['shots']]
    for sid in ids[:2]:
        project = apply_operation(project, sid, 'candidate_create', {'facts': False, 'intent': True})
        shot = next(row for row in project['doc']['shots'] if row['id'] == sid)
        project = apply_operation(project, sid, 'candidate_accept',
                                  {'confirm': True, 'text_sha256': shot['promptCandidate']['text_sha256']})
    before = deepcopy(project['doc']['shots'][:2])
    project['doc']['shots'][2]['simplePrompt'] = '只改未来第三镜'
    assert project['doc']['shots'][:2] == before
    for shot in project['doc']['shots'][:2]:
        assert candidate_diagnostics(project, shot)['status'] == 'active'
        assert candidate_diagnostics(project, shot)['changed_components'] == []


@pytest.mark.parametrize('bad', ['unknown_component', 'missing_component', 'bad_digest', 'null'])
def test_incomplete_or_malformed_component_receipts_rejected(bad):
    project = candidate_project()
    values = project['doc']['shots'][0]['promptCandidate']['receipt']['input_component_sha256']
    if bad == 'unknown_component':
        values['invented'] = 'a' * 64
    elif bad == 'missing_component':
        del values['global']
    elif bad == 'null':
        project['doc']['shots'][0]['promptCandidate']['receipt']['input_component_sha256'] = None
    else:
        values['global'] = 'not-a-digest'
    with pytest.raises(ValueError):
        validate_project(project)


def test_inconsistent_component_declaration_never_certifies_exact_reason_or_cache():
    project = candidate_project()
    shot = project['doc']['shots'][0]
    before_inputs = _inputs(project, shot, shot['promptCandidate']['options'])
    shot['promptCandidate']['receipt']['input_component_sha256']['global'] = 'a' * 64
    result = candidate_diagnostics(project, shot)
    assert result['inputs_hash_match'] is True
    assert result['component_snapshot'] == 'inconsistent_receipt_declaration'
    assert result['changed_components'] is None and not result['cache_hit_certified']
    assert _inputs(project, shot, shot['promptCandidate']['options']) == before_inputs
