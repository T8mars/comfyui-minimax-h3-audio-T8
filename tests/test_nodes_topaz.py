import json

import pytest
from comfy_api.latest import InputImpl

from h3_audio_t8_pkg import nodes_topaz as nodes
from h3_audio_t8_pkg import topaz_runtime
from tests.test_topaz_contract import runtime  # noqa: F401


def test_optional_nodes_have_no_license_checkbox_or_implicit_install():
    classes = nodes.TOPAZ_NODE_CLASSES
    assert len(classes) == 4
    for cls in classes:
        schema = cls.define_schema()
        assert schema.is_experimental and schema.is_output_node
        ids = [item.id for item in schema.inputs]
        assert not any('license' in key or 'download' in key for key in ids)
    defaults = {item.id: item.default for item in classes[0].define_schema().inputs}
    assert set(defaults.values()) == {''}


def test_regular_model_id_is_a_dropdown_with_future_model_escape_hatch():
    inputs = {item.id: item for item in nodes.MiniMaxH3TopazVideoEXPT8.define_schema().inputs}
    assert inputs['model_id'].options == nodes.TOPAZ_REGULAR_MODEL_IDS
    assert inputs['model_id'].default == 'iris-3'
    assert inputs['custom_model_id'].default == ''
    assert inputs['output_profile'].options == ['delivery_h264', 'delivery_hevc_main10', 'lossless_master']
    assert inputs['gpu_device_index'].default == 0


def test_empty_environment_cannot_treat_current_directory_as_install():
    with pytest.raises(ValueError, match='Topaz'):
        nodes.MiniMaxH3TopazEnvironmentEXPT8.execute('', '', '')


def test_file_video_is_forwarded_without_parent_decode(monkeypatch, runtime, tmp_path):  # noqa: F811
    source = tmp_path / 'source.mkv'
    source.write_bytes(b'fixture-not-decoded')
    video = InputImpl.VideoFromFile(str(source))
    monkeypatch.setattr(video, 'get_dimensions', lambda: pytest.fail('Parent tried to decode media'))
    monkeypatch.setattr(nodes.folder_paths, 'get_output_directory', lambda: str(tmp_path))
    observed = {}
    def run(env, path, job, **settings):
        observed.update(settings)
        assert path == source and env.install == runtime.install
        return tmp_path / 'result.mkv', {'status': 'fixture_only'}
    monkeypatch.setattr(topaz_runtime, 'run_regular', run)
    handle = {'schema': 't8_official_topaz_paths_v1', 'install': str(runtime.install),
        'definitions': str(runtime.definitions), 'data': str(runtime.data)}
    result = nodes.MiniMaxH3TopazVideoEXPT8.execute(handle, video, 'iris-3', '2x',
        parameters_json='{"noise":0.3}', gpu_device_index=2)
    assert observed['width'] is None and observed['height'] is None
    assert observed['scale'] == 2 and observed['parameters'] == {'noise': .3}
    assert observed['device'] == 2
    assert result.result[1] is video
    assert json.loads(result.result[3])['status'] == 'fixture_only'


def test_custom_output_directory_is_forwarded_as_task_parent(monkeypatch, runtime, tmp_path):  # noqa: F811
    source = tmp_path / 'source.mkv'
    source.write_bytes(b'fixture-not-decoded')
    video = InputImpl.VideoFromFile(str(source))
    selected = tmp_path / 'other-drive' / 'Topaz-Masters'
    monkeypatch.setattr(nodes.folder_paths, 'get_output_directory',
        lambda: pytest.fail('Default output directory should not be used'))
    observed = {}
    def run(env, path, job, **settings):
        observed['job'] = job
        return tmp_path / 'result.mkv', {'status': 'fixture_only'}
    monkeypatch.setattr(topaz_runtime, 'run_regular', run)
    handle = {'schema': 't8_official_topaz_paths_v1', 'install': str(runtime.install),
        'definitions': str(runtime.definitions), 'data': str(runtime.data)}
    nodes.MiniMaxH3TopazVideoEXPT8.execute(handle, video, 'iris-3', '2x',
        output_directory=str(selected))
    assert observed['job'].parent == selected.resolve()


def test_custom_model_id_overrides_dropdown_for_future_official_model(monkeypatch, runtime, tmp_path):  # noqa: F811
    source = tmp_path / 'source.mkv'
    source.write_bytes(b'fixture-not-decoded')
    video = InputImpl.VideoFromFile(str(source))
    monkeypatch.setattr(nodes.folder_paths, 'get_output_directory', lambda: str(tmp_path))
    observed = {}
    def run(env, path, job, **settings):
        observed.update(settings)
        return tmp_path / 'result.mkv', {'status': 'fixture_only'}
    monkeypatch.setattr(topaz_runtime, 'run_regular', run)
    handle = {'schema': 't8_official_topaz_paths_v1', 'install': str(runtime.install),
        'definitions': str(runtime.definitions), 'data': str(runtime.data)}
    (runtime.definitions / 'future-9.json').write_text(
        (runtime.definitions / 'iris-3.json').read_text())
    nodes.MiniMaxH3TopazVideoEXPT8.execute(handle, video, 'iris-3', '2x',
        custom_model_id='future-9')
    assert observed['model_id'] == 'future-9'


def test_interpolation_forwards_main10_and_gpu_without_parent_decode(monkeypatch, runtime, tmp_path):  # noqa: F811
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'fixture-not-decoded')
    video = InputImpl.VideoFromFile(str(source))
    monkeypatch.setattr(video, 'get_dimensions', lambda: pytest.fail('Parent tried to decode media'))
    monkeypatch.setattr(nodes.folder_paths, 'get_output_directory', lambda: str(tmp_path))
    observed = {}

    def run(env, path, job, **settings):
        observed.update(settings)
        assert path == source and env.install == runtime.install
        return tmp_path / 'result.mp4', {'status': 'fixture_only'}

    monkeypatch.setattr(topaz_runtime, 'run_interpolation', run)
    handle = {'schema': 't8_official_topaz_paths_v1', 'install': str(runtime.install),
        'definitions': str(runtime.definitions), 'data': str(runtime.data)}
    result = nodes.MiniMaxH3TopazFrameInterpolationEXPT8.execute(
        handle, video, 'apo-8', '2x', output_profile='delivery_hevc_main10', gpu_device_index=4)
    assert observed['multiplier'] == 2 and observed['output_profile'] == 'delivery_hevc_main10'
    assert observed['device'] == 4
    assert result.result[1] is video


@pytest.mark.parametrize('value', ['relative/path', '/', 3])
def test_invalid_custom_output_directory_is_rejected(monkeypatch, runtime, tmp_path, value):  # noqa: F811
    source = tmp_path / 'source.mkv'
    source.write_bytes(b'fixture')
    video = InputImpl.VideoFromFile(str(source))
    handle = {'schema': 't8_official_topaz_paths_v1', 'install': str(runtime.install),
        'definitions': str(runtime.definitions), 'data': str(runtime.data)}
    with pytest.raises(ValueError, match='output directory'):
        nodes.MiniMaxH3TopazVideoEXPT8.execute(handle, video, 'iris-3', '2x',
            output_directory=value)
