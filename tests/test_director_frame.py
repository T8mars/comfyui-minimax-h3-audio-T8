from copy import deepcopy
import uuid

import av
import numpy as np
from PIL import Image
import pytest

from h3_audio_t8_pkg.director_film import prepare_film, film_media
from h3_audio_t8_pkg.director_frame import extract_frame
from h3_audio_t8_pkg.director_project import ProjectStore, atomic_json, contained, file_sha, validate_project
from test_director_film import setup


def test_frame_exact_decoded_png_retries_and_original_project_unchanged(tmp_path):
    store, output, project, records = setup(tmp_path)
    original = deepcopy(project)
    manifest = prepare_film(store, project, records, output)
    source = film_media(store, project['id'], manifest['id'], 0)
    digest = file_sha(source)
    with av.open(str(source)) as container:
        expected = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
    for number in (0, 5, 11):
        result = extract_frame(store, project['id'], manifest['id'], 0, number)
        asset = result['asset']
        path = contained(store.input_root, asset['server_path'])
        with Image.open(path) as image:
            assert image.size == (64, 32)
            np.testing.assert_array_equal(np.asarray(image), expected[number])
        assert asset['kind'] == 'image' and result['seconds'] == number/24
        assert asset['source_frame'] == {'media_sha256': digest, 'frame': number, 'fps': 24}
        assert store.asset(asset['id'], verify=True)['source_frame'] == asset['source_frame']
        assert extract_frame(store, project['id'], manifest['id'], 0, number) == result
    assert project == original and file_sha(source) == digest
    assert len(list((store.root/'assets').glob('*.json'))) == 3


def test_frame_provenance_survives_save_reload_and_new_store_zip(tmp_path):
    from h3_audio_t8_pkg.director_bundle import prepare_bundle, build_bundle, inspect_bundle, import_bundle
    from test_director_bundle import upload

    store, output, project, records = setup(tmp_path)
    manifest = prepare_film(store, project, records, output)
    extracted = extract_frame(store, project['id'], manifest['id'], 0, 5)['asset']
    project['assets'] = [extracted]
    project['doc']['shots'][1].update(mode='first', first=extracted['id'], tray=[extracted['id']])
    store.save(project, 0)
    project = store.load(project['id'])
    assert project['assets'][0]['source_frame'] == extracted['source_frame']
    plan = prepare_bundle(store, project, records, output, include_results=False)
    archive = build_bundle(store, project['id'], plan['id'])
    fresh = ProjectStore(tmp_path/'new-user', tmp_path/'new-input')
    upload_id = upload(fresh, archive)
    inspect_bundle(fresh, upload_id)
    new_id = str(uuid.uuid4())
    import_bundle(fresh, tmp_path/'new-output', upload_id, new_id)
    migrated = fresh.load(new_id)['assets'][0]
    assert migrated['id'] != extracted['id']
    assert migrated['sha256'] == extracted['sha256']
    assert fresh.asset(migrated['id'], verify=True)['source_frame'] == extracted['source_frame']
    assert migrated['source_frame'] == extracted['source_frame']


def test_legacy_frame_backfill_requires_exact_pixels_and_conflicts_never_overwrite(tmp_path):
    store, output, project, records = setup(tmp_path)
    manifest = prepare_film(store, project, records, output)
    result = extract_frame(store, project['id'], manifest['id'], 0, 5)
    legacy = deepcopy(result['asset'])
    legacy.pop('source_frame')
    receipt = contained(store.root, f"assets/{legacy['id']}.json")
    atomic_json(receipt, legacy)
    assert extract_frame(store, project['id'], manifest['id'], 0, 5) == result
    conflict = deepcopy(result['asset'])
    conflict['source_frame']['frame'] = 6
    atomic_json(receipt, conflict)
    before = file_sha(receipt)
    with pytest.raises(ValueError, match='来源不一致'):
        extract_frame(store, project['id'], manifest['id'], 0, 5)
    assert file_sha(receipt) == before
    png = contained(store.input_root, legacy['server_path'])
    Image.new('RGB', (64, 32), (255, 0, 255)).save(png)
    legacy.update(sha256=file_sha(png), size=png.stat().st_size)
    atomic_json(receipt, legacy)
    with pytest.raises(ValueError, match='与来源帧不符'):
        extract_frame(store, project['id'], manifest['id'], 0, 5)
    assert 'source_frame' not in store.asset(legacy['id'])


@pytest.mark.parametrize('field,value', [('media_sha256', 'bad'), ('frame', True), ('frame', -1),
                                       ('frame', 1.5), ('fps', 0), ('fps', float('nan')),
                                       ('fps', True), ('private_path', 'C:/private')])
def test_frame_source_rejects_malformed_or_undeclared_metadata(tmp_path, field, value):
    store, output, project, records = setup(tmp_path)
    manifest = prepare_film(store, project, records, output)
    extracted = extract_frame(store, project['id'], manifest['id'], 0, 5)['asset']
    extracted['source_frame'][field] = value
    project['assets'] = [extracted]
    with pytest.raises(ValueError, match='取帧来源'):
        validate_project(project)


@pytest.mark.parametrize('number', [-1, 12, True, 1.5, '0', None])
def test_frame_rejects_invalid_number_without_input_writes(tmp_path, number):
    store, output, project, records = setup(tmp_path)
    manifest = prepare_film(store, project, records, output)
    with pytest.raises(ValueError):
        extract_frame(store, project['id'], manifest['id'], 0, number)
    assert not list(store.input_root.rglob('*.png'))


def test_frame_rejects_foreign_owner_corrupt_cache_and_corrupt_previous_asset(tmp_path):
    store, output, project, records = setup(tmp_path)
    manifest = prepare_film(store, project, records, output)
    with pytest.raises(ValueError):
        extract_frame(store, str(uuid.uuid4()), manifest['id'], 0, 0)
    first = extract_frame(store, project['id'], manifest['id'], 0, 0)
    png = contained(store.input_root, first['asset']['server_path'])
    png.write_bytes(b'changed')  # Owned test fixture only.
    with pytest.raises(ValueError, match='字节改变'):
        extract_frame(store, project['id'], manifest['id'], 0, 0)
    source = film_media(store, project['id'], manifest['id'], 0)
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='冻结成片'):
        extract_frame(store, project['id'], manifest['id'], 0, 1)
