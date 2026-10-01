"""Real reservation/receipt code, only graph construction and Core queue substituted."""
import asyncio
import uuid

import pytest

from h3_audio_t8_pkg import director_routes
from h3_audio_t8_pkg.director_project import ProjectStore, new_project


@pytest.mark.parametrize('error', [ValueError('bad prompt'), FileNotFoundError('missing model'), TypeError('bad type'), KeyError('missing field')])
def test_only_pre_reservation_validation_confirms_no_submission(tmp_path, monkeypatch, error):
    store = ProjectStore(tmp_path/'user', tmp_path/'input')
    project = new_project()
    request_id = str(uuid.uuid4())

    def reject(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(director_routes, 'build_director_generation_prompt', reject)
    response, status = asyncio.run(director_routes._submit_director_request_locked(
        store, project, project['current'], 1, request_id))
    assert status == 400 and response['submission_state'] == 'not_submitted'
    assert not (store.root/'requests'/f'{request_id}.json').exists()


def test_queue_value_error_is_ambiguous_and_never_released(tmp_path, monkeypatch):
    store = ProjectStore(tmp_path/'user', tmp_path/'input')
    project = new_project()
    request_id = str(uuid.uuid4())

    async def queue(*_args, **_kwargs):
        raise ValueError('accepted but acknowledgement failed')

    monkeypatch.setattr(director_routes, 'queue_director_prompt', queue)
    with pytest.raises(ValueError, match='acknowledgement'):
        asyncio.run(director_routes._submit_director_request_locked(
            store, project, project['current'], 1, request_id, built={'prompt': {}}))
    response, status = asyncio.run(director_routes._submit_director_request_locked(
        store, project, project['current'], 1, request_id, built={'prompt': {}}))
    assert status == 409 and response['prompt_id']
    assert 'submission_state' not in response
