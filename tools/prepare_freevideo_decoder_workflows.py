"""Create-only decoder examples from real local node schemas, never Queue."""
import argparse
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def template(info, family):
    from tools.api_to_frontend_workflow import convert
    from tools.validate_freevideo_quality_canvas import check_serialized
    if family not in ('quality', 'legacy'):
        raise ValueError('Choose the real completed Stage family')
    loader = ('MiniMaxH3FreeVideoQualityStageLoadEXPT8' if family == 'quality'
        else 'MiniMaxH3FreeVideoStageLoadEXPT8')
    decoder = ('MiniMaxH3FreeVideoQualityVideoDecodeEXPT8' if family == 'quality'
        else 'MiniMaxH3FreeVideoVideoDecodeEXPT8')
    parameters = dict(manifest_path='', manifest_sha256='')
    if family == 'legacy':
        parameters['role'] = 'HIGH'
    api = {'1': dict(class_type=loader, inputs=parameters),
        '2': dict(class_type='VAELoader', inputs=dict(vae_name='minimax_h3_audio_vae_fp32.safetensors')),
        '3': dict(class_type=decoder, inputs=dict(completed_stage=['1', 1], decoder_config='', decode_mode='eager')),
        '4': dict(class_type='VAEDecodeAudio', inputs=dict(samples=['3', 1], vae=['2', 0])),
        '5': dict(class_type='MiniMaxH3OutputTrimT8', inputs=dict(frames=['3', 0], audio=['4', 0],
            start_seconds=0., duration_seconds=5., fps=24.)),
        '6': dict(class_type='MiniMaxH3SafeAVSaveT8Advanced', inputs=dict(images=['5', 0], audio=['5', 1],
            filename_prefix='FreeVideo/' + ('Quality' if family == 'quality' else 'Legacy') + '_Decode', crf=18))}
    graph = convert(api, info, 'FreeVideo Decoder · ' + family)
    for row in graph['nodes']:
        source = api[str(row['id'])]['inputs']
        row['widgets_values_named'] = {name: source[name]
            for section in ('required', 'optional')
            for name in info[row['type']]['input'].get(section, {})
            if name in source and not isinstance(source[name], list)}
        row['size'][0] = 440
        row['pos'] = {1: [0, 0], 2: [0, 300], 3: [490, 0], 4: [490, 310],
            5: [980, 0], 6: [1470, 0]}[row['id']]
    note = ('新增独立视频解码EXP，不替换旧Core VAE／8+2／真4+4／Quality四档。'
        '\n先准备官方三分片视频VAE和独立decoder_config；本图不自动下载或安装。'
        '\nStage Load填写已完成manifest绝对路径及SHA。Quality只HIGH／SINGLE；旧版只HIGH。LOW／MID不可用。'
        '\n默认eager；compile需手动选择，首次编译可能更慢、有融合舍入差，不保证提速或逐位相同。'
        '\n视频解码0新增采样；第二输出为原音频latent，接正常音频VAE。不向视频子进程传音频。'
        '\n当前裁齐是0起点24fps5秒；其他时长请按真实完成Stage修改，不伪造完整帧。'
        '\n本机唯一Quality HIGH eager／compile代表已机械验证，最终画质听感待人审；不扩大为全部配方通过。')
    graph['nodes'].append(dict(id=7, type='Note', pos=[0, -300], size=[1100, 240], flags={},
        order=6, mode=0, inputs=[], outputs=[], properties={}, widgets_values=[note],
        title='使用前必读 · 原完整阶段 · 0采样'))
    graph['last_node_id'] = 7
    serial = check_serialized(graph, api)
    assert serial['nodes'] == 6 and serial['links'] == 7
    return graph, api, serial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True, help='Existing local ComfyUI origin; only reads object_info')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    parsed = urllib.parse.urlparse(args.server)
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
            or parsed.path not in ('', '/') or parsed.query or parsed.fragment or parsed.username or parsed.password):
        raise ValueError('Use an explicit local ComfyUI origin without credentials or paths')
    if args.output.exists():
        raise FileExistsError('Never overwrite old examples or user workflows')
    with urllib.request.urlopen(args.server.rstrip('/') + '/object_info', timeout=30) as response:
        info = json.load(response)
    candidates = [(name, template(info, family)) for name, family in
        [('FVQ_Decode_EXP.json', 'quality'), ('FV_Decode_EXP.json', 'legacy')]]
    args.output.mkdir(parents=True, exist_ok=False)
    for name, (graph, _, serial) in candidates:
        with (args.output / name).open('x', encoding='utf8', newline='\n') as stream:
            json.dump(graph, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(dict(path=str(args.output / name), serialization=serial,
            blank_user_Stage_and_decoder_paths=True, queued=False, GPU_or_human_qualified=False)))


if __name__ == '__main__':
    main()
