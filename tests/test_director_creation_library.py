from copy import deepcopy
import uuid

import pytest

from h3_audio_t8_pkg.director_project import new_project, validate_project, ProjectStore
from h3_audio_t8_pkg.director_creation import TEMPLATE_FIELDS, remap_library
from h3_audio_t8_pkg.director_bundle import portable_project, prepare_bundle, build_bundle, inspect_bundle, import_bundle
from test_director_d1 import asset
from test_director_bundle import upload


def collection(project, aid):
    shot = project['doc']['shots'][0]
    snapshot = {key: deepcopy(value) for key, value in shot.items() if key in TEMPLATE_FIELDS}
    snapshot.update(tray=[aid], refs=[aid], mode='refs', simplePrompt='@image1 微笑')
    return {'version': 1, 'items': [
        {'id': str(uuid.uuid4()), 'name': '片段', 'kind': 'snippet', 'text': '@image1 微笑', 'bindings': {'@image1': aid}},
        {'id': str(uuid.uuid4()), 'name': '模板', 'kind': 'template', 'shot': snapshot, 'bindings': {'@image1': aid}},
        {'id': str(uuid.uuid4()), 'name': '配方', 'kind': 'recipe', 'sampling': {'mode': 'two_pass', 'output_mp': '0.6',
         'low_loras': [{'name': 'low.safetensors', 'strength': 0.6, 'enabled': True}],
         'high_loras': [{'name': 'high.safetensors', 'strength': 0.3, 'enabled': True}]}, 'd3': {}, 'ratio': '16:9', 'duration': 4},
        {'id': str(uuid.uuid4()), 'name': '素材组', 'kind': 'group', 'asset_ids': [aid], 'note': '只复用素材，不保证锁定身份'},
    ]}


def test_creation_collections_roundtrip_store_and_real_bundle_new_identities(tmp_path):
    store = ProjectStore(tmp_path/'user', tmp_path/'input')
    media = asset(store)
    project = new_project()
    project['assets'] = [media]
    project['doc']['creationLibrary'] = collection(project, media['id'])
    validated = validate_project(project)
    assert validated == project
    saved = store.save(project, 0)
    assert store.load(project['id'])['doc']['creationLibrary'] == project['doc']['creationLibrary']
    output = tmp_path/'output'
    output.mkdir()
    plan = prepare_bundle(store, saved, [], output, include_results=False)
    assert plan['ready'], plan
    archive = build_bundle(store, project['id'], plan['id'])
    target = ProjectStore(tmp_path/'fresh-user', tmp_path/'fresh-input')
    output2 = tmp_path/'fresh-output'
    output2.mkdir()
    upload_id = upload(target, archive)
    assert inspect_bundle(target, upload_id)['ready']
    new_id = str(uuid.uuid4())
    import_bundle(target, output2, upload_id, new_id)
    restored = target.load(new_id)
    items = restored['doc']['creationLibrary']['items']
    source = project['doc']['creationLibrary']['items']
    aid = restored['assets'][0]['id']
    assert aid != media['id']
    assert all(item['id'] != old['id'] for item, old in zip(items, source))
    assert items[0]['bindings'] == items[1]['bindings'] == {'@image1': aid}
    assert items[1]['shot']['tray'] == items[1]['shot']['refs'] == items[3]['asset_ids'] == [aid]
    assert items[1]['shot']['simplePrompt'] == '@image1 微笑'
    assert items[2]['sampling'] == source[2]['sampling']
    assert store.load(project['id'])['doc']['creationLibrary'] == project['doc']['creationLibrary']


@pytest.mark.parametrize('mutate', [
    lambda lib: lib.update(version=True),
    lambda lib: lib.update(version=2),
    lambda lib: lib['items'][0].update(kind=[]),
    lambda lib: lib['items'][0].update(kind={}),
    lambda lib: lib['items'].append(deepcopy(lib['items'][0])),
    lambda lib: lib['items'][0].update(id='invalid'),
    lambda lib: lib['items'][0].update(name=''),
    lambda lib: lib['items'][0].update(text=''),
    lambda lib: lib['items'][0].update(bindings={'@image1': '../outside'}),
    lambda lib: lib['items'][0].update(bindings={'secret': str(uuid.uuid4())}),
    lambda lib: lib['items'][0].update(secret='not a declared field'),
    lambda lib: lib['items'][1]['shot'].update(adoptedResultId='old-result'),
    lambda lib: lib['items'][1]['shot'].update(duration=-1),
    lambda lib: lib['items'][2].update(sampling=None),
    lambda lib: lib['items'][2].update(duration=float('nan')),
    lambda lib: lib['items'][2].update(ratio='nonsense'),
    lambda lib: lib['items'][3].update(asset_ids=[]),
    lambda lib: lib['items'][3]['asset_ids'].append(lib['items'][3]['asset_ids'][0]),
])
def test_invalid_collection_fields_fail_before_save(mutate):
    project = new_project()
    library = collection(project, str(uuid.uuid4()))
    mutate(library)
    project['doc']['creationLibrary'] = library
    with pytest.raises(ValueError):
        validate_project(project)


def test_library_portability_strips_nested_metadata_and_rejects_private_model_paths():
    project = new_project()
    library = collection(project, str(uuid.uuid4()))
    project['doc']['creationLibrary'] = library
    library['items'][2]['sampling']['low_loras'][0]['private_data'] = 'SECRET'
    clean = portable_project(project)
    assert 'private_data' not in clean['doc']['creationLibrary']['items'][2]['sampling']['low_loras'][0]
    library['items'][2]['sampling']['upscaler'] = 'G:/private/model.safetensors'
    with pytest.raises(ValueError, match='绝对路径'):
        portable_project(project)


def test_missing_library_asset_remains_explicit_missing_identity_after_bundle_remap():
    aid = str(uuid.uuid4())
    library = collection(new_project(), aid)
    namespace = uuid.uuid4()
    def fresh(role, key):
        return str(uuid.uuid5(namespace, role+':'+key))
    remap_library(library, {}, fresh)
    mapped = fresh('missing-collection-asset', aid)
    assert mapped != aid
    assert library['items'][0]['bindings']['@image1'] == mapped
    assert library['items'][1]['shot']['refs'] == library['items'][3]['asset_ids'] == [mapped]
