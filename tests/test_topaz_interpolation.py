from copy import deepcopy
from fractions import Fraction
import json

import pytest

from h3_audio_t8_pkg import nodes_topaz
from h3_audio_t8_pkg.topaz_contract import interpolation_command, interpolation_filter
from h3_audio_t8_pkg.topaz_media import compare_interpolated_video
from tests.test_topaz_contract import runtime  # noqa: F401


def prepare_fi(installation):
    definition = {'backends': {'onnx': {}}, 'changesFPS': 1, 'modelType': 2,
        'shortName': 'apo', 'version': '8', 'displayName': 'Apollo'}
    (installation.definitions / 'apo-8.json').write_text(json.dumps(definition))


def test_interpolation_command_multiplies_fps_without_slow_motion(runtime, tmp_path):  # noqa: F811
    prepare_fi(runtime)
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'fixture')
    command = interpolation_command(runtime, source, tmp_path / 'result.mp4',
        'apo-8', Fraction(60000, 1001), duplicate_threshold=.01)
    filters = command[command.index('-vf') + 1]
    assert 'tvai_fi=model=apo-8' in filters
    assert 'fps=60000/1001' in filters
    assert 'slowmo=1' in filters
    assert 'download=0' in filters
    assert command[command.index('-c:v') + 1] == 'h264_nvenc'
    assert command[command.index('-c:a') + 1] == 'copy'
    assert '-r' not in command


def test_interpolation_can_automatically_encode_incompatible_audio_to_aac(runtime, tmp_path):  # noqa: F811
    prepare_fi(runtime)
    source = tmp_path / 'source.avi'
    source.write_bytes(b'fixture')
    command = interpolation_command(runtime, source, tmp_path / 'result.mp4',
        'apo-8', 48, audio_mode='aac')
    assert command[command.index('-c:a') + 1] == 'aac'
    assert command[command.index('-b:a') + 1] == '192k'
    with pytest.raises(ValueError, match='MP4'):
        interpolation_command(runtime, source, tmp_path / 'result.mkv', 'apo-8', 48)


def test_interpolation_main10_uses_explicit_hevc_profile(runtime, tmp_path):  # noqa: F811
    prepare_fi(runtime)
    source = tmp_path / 'source.mkv'
    source.write_bytes(b'fixture')
    command = interpolation_command(runtime, source, tmp_path / 'result.mp4',
        'apo-8', 48, output_profile='delivery_hevc_main10', device=2)
    assert command[command.index('-c:v') + 1] == 'hevc_nvenc'
    assert command[command.index('-profile:v') + 1] == 'main10'
    assert command[command.index('-pix_fmt') + 1] == 'p010le'
    assert command[command.index('-tag:v') + 1] == 'hvc1'
    assert 'device=2' in command[command.index('-vf') + 1]
    assert 'format=p010le' in command[command.index('-vf') + 1]


def test_regular_model_cannot_be_used_for_interpolation(runtime):  # noqa: F811
    with pytest.raises(ValueError, match='interpolation'):
        interpolation_filter(runtime, 'iris-3', 48)


def test_interpolation_audit_accepts_only_exact_fps_and_endpoint_convention():
    source = {'width': 1024, 'height': 512, 'frames': 73, 'fps': '24', 'origin': '0', 'bit_depth': 10}
    output = {'width': 1024, 'height': 512, 'frames': 145, 'fps': '48', 'origin': '0', 'bit_depth': 10}
    assert compare_interpolated_video(source, output, 2)['status'] == 'fps_multiplied_duration_preserved'
    for change in ({'fps': '47'}, {'frames': 140}, {'width': 1000}):
        broken = deepcopy(output)
        broken.update(change)
        with pytest.raises(ValueError):
            compare_interpolated_video(source, broken, 2)


def test_node_exposes_separate_interpolation_not_mixed_with_upscale():
    ids = [cls.define_schema().node_id for cls in nodes_topaz.TOPAZ_NODE_CLASSES]
    assert ids == ['MiniMaxH3TopazEnvironmentEXPT8', 'MiniMaxH3TopazSourcePrepareEXPT8',
        'MiniMaxH3TopazVideoEXPT8', 'MiniMaxH3TopazFrameInterpolationEXPT8']
    schema = nodes_topaz.MiniMaxH3TopazFrameInterpolationEXPT8.GET_NODE_INFO_V1()['input']
    required = schema['required']
    assert required['model_id'][1]['options'] == nodes_topaz.TOPAZ_FI_MODEL_IDS
    assert required['multiplier'][1]['options'] == ['2x', '4x']
    assert schema['optional']['output_profile'][1]['options'] == ['delivery_h264', 'delivery_hevc_main10']
    assert schema['optional']['gpu_device_index'][1]['default'] == 0


def test_parameter_modes_are_clear_and_json_is_only_an_override():
    helper = nodes_topaz._parameter_overrides
    assert helper('model_defaults', 8, 0, .5, .5, .5, .5, .5, 0, 0, 0, True, 0, '{}') == {}
    assert helper('auto_estimate', 12, 0, .5, .5, .5, .5, .5, 0, 0, 0, True, 0,
        '{}') == {'estimate': 12}
    manual = helper('manual', 8, -.2, .3, .4, .5, .6, .7, .01, .2, 1.5, True,
        .1, '{"noise":0.9}')
    assert manual['preblur'] == -.2 and manual['noise'] == .9
    assert manual['gsize'] == 1.5 and manual['kcolor'] == 1
    fixed = helper('manual', 8, -.2, .3, .4, .5, .6, .7, .01, .2, 1.5, True,
        .1, '{}', supported_model_parameters=set())
    assert not {'preblur', 'noise', 'details', 'halo', 'blur', 'compression'} & fixed.keys()
    assert fixed['estimate'] == 0 and fixed['kcolor'] == 1
