"""H3 sound candidate graph validation with an actual synthetic source file.

This fixture checks native decode and graph contracts, not trained audio quality.
"""
import asyncio
import json

import av
import numpy as np

from tools import build_nov05_core_examples as builder


def test_h3_sfx_actual_source_and_native_export_keep_audio_only_recipe(tmp_path, monkeypatch):
    import execution
    import folder_paths
    import nodes
    import torch

    monkeypatch.setattr(folder_paths, 'input_directory', str(tmp_path))
    source = tmp_path / 'TaoMate_Source.mp4'
    with av.open(str(source), 'w') as container:
        stream = container.add_stream('libx264', rate=24)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = 'yuv420p'
        for index in range(5):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), index * 30, np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    info = builder.load_live_info()
    graph, workflow, audit, _selected = builder.build('H3_SFX_Generate_EXP', info)
    validation = asyncio.run(execution.validate_prompt('h3-sfx-example-source', graph, None))
    assert validation[0] and not validation[3] and set(validation[2]) == {'15'}, validation
    assert audit['nodes'] == len(graph) and len(workflow['nodes']) == len(graph) + 1
    assert '不提取或分离人声' in workflow['nodes'][-1]['widgets_values'][0]

    video = nodes.NODE_CLASS_MAPPINGS['LoadVideo'].execute('TaoMate_Source.mp4').result[0]
    decoded = nodes.NODE_CLASS_MAPPINGS['MiniMaxH3SourceClockEXPT8'].execute(video).result
    assert decoded[0].shape == (5, 32, 32, 3) and decoded[2] == 24
    assert graph['9']['inputs']['steps'] == 40
    assert graph['9']['inputs']['sampler_name'] == 'dual_clock_euler'
    assert graph['45']['inputs']['value'] == 0
    assert json.loads(graph['46']['inputs']['audio_intervals']) == [{'start': 0, 'end': 124 / 24}]
    assert graph['46']['inputs']['require_lanpaint_sampler'] is False
    assert 'source_audio' not in graph['42']['inputs']  # fresh audio uses generated silence, not source voice
    assert graph['14']['inputs']['frames'] == ['42', 0]  # source RGB, never the reconstructed video
    assert graph['14']['inputs']['audio'] == ['13', 1]
    assert graph['14']['inputs']['frame_count'] == 120 and graph['14']['inputs']['fps'] == 24

    # The native frontend adds this empty transient preview widget. It is not
    # a sampling parameter; the actual Core validator must still accept it.
    graph['40']['inputs']['video-preview'] = ''
    exported = asyncio.run(execution.validate_prompt('h3-sfx-native-preview', graph, None))
    assert exported[0] and not exported[3] and set(exported[2]) == {'15'}, exported
    source.unlink()
    missing = asyncio.run(execution.validate_prompt('h3-sfx-missing-source', graph, None))
    assert not missing[0] and '40' in missing[3], missing
    assert not torch.cuda.is_initialized()
