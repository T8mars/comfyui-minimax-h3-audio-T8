"""Add only an explicit anime full/cold export pair; preserve all 84 old graphs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_modular_sampling_compat import write_new  # noqa: E402
from tools.build_modular_h16_storage_workflow import Draft  # noqa: E402
from tools import build_formal_face_refine_storage_workflows as storage  # noqa: E402

DESTINATION = storage.face.DESTINATION / 'isolated-delivery-exp'
WRITER = 'MiniMaxH3SaveVideoIsolatedEXPT8'
FILES = {mode: f'S24_Face_anime_combined_{mode}_Isolated_Delivery_EXP.json' for mode in storage.MODES}


def graph_for(mode):
    if mode not in FILES:
        raise ValueError('Only the explicit anime full-save/cold-delivery pair is supported')
    source = storage.face.DESTINATION / storage.FILES['anime', 'combined', mode]
    original_bytes = source.read_bytes()
    original_frontend = json.loads(original_bytes)
    rebuilt, original_api = storage.graph_for('anime', 'combined', mode)
    if original_frontend != rebuilt:
        raise ValueError('The original public anime graph differs from its unchanged builder')
    draft = Draft(original_frontend)
    writers = [node for node in draft.nodes.values() if node['type'] == 'SaveVideo']
    if len(writers) != 1:
        raise ValueError('Exactly one original Face writer is required')
    writer = writers[0]
    key = str(writer['id'])
    before = original_api[key]
    if (set(before['inputs']) != {'video', 'filename_prefix', 'format', 'codec'}
            or before['inputs']['format'] != 'mp4' or before['inputs']['codec'] != 'h264'
            or any(item.get('links') for item in writer['outputs'])):
        raise ValueError('The original mp4/h264 writer contract or consumer edges changed')
    widget_values = [before['inputs']['filename_prefix'], 600, 16., 2.]
    writer.update(type=WRITER, title='Explicit isolated H.264/AAC delivery · sampling unchanged',
        size=[470, 300], widgets_values=widget_values,
        inputs=[deepcopy(writer['inputs'][0]), *[
            {'name': name, 'type': dtype, 'link': None, 'widget': {'name': name}}
            for name, dtype in (('filename_prefix', 'STRING'), ('timeout_seconds', 'INT'),
                                ('max_staging_gib', 'FLOAT'), ('min_free_disk_gib', 'FLOAT'))]],
        outputs=[{'name': name, 'type': dtype, 'links': []}
                 for name, dtype in (('video', 'VIDEO'), ('video_path', 'STRING'), ('report_json', 'STRING'))],
        properties={'Node name for S&R': WRITER, 'cnr_id': 'minimax-h3-audio-T8'})
    preview = draft.make('PreviewAny', 'Export audit · not automatic quality acceptance',
        [('source', '*')], [('STRING', 'STRING')], [], [writer['pos'][0] + 600, writer['pos'][1]])
    draft.connect((writer['id'], 2), preview, 'source', 'STRING')
    frontend = draft.graph
    frontend['last_node_id'] = max(draft.nodes)
    frontend['last_link_id'] = max(draft.links)
    frontend['id'] = str(uuid.uuid5(uuid.NAMESPACE_URL,
        't8/S24/anime/isolated-delivery/' + mode + '/' + hashlib.sha256(original_bytes).hexdigest()))
    frontend['extra']['t8_split_example'].update(
        delivery='explicit_isolated_h264', status='experimental_isolated_delivery_not_quality_accepted')
    api = deepcopy(original_api)
    api[key] = {'class_type': WRITER, 'inputs': {
        'video': deepcopy(before['inputs']['video']),
        **dict(zip(('filename_prefix', 'timeout_seconds', 'max_staging_gib', 'min_free_disk_gib'),
                   widget_values, strict=True))}}
    api[str(preview['id'])] = {'class_type': 'PreviewAny', 'inputs': {'source': [key, 2]}}
    if any(api[node] != value for node, value in original_api.items() if node != key):
        raise ValueError('Explicit delivery changed another sampler, source, audio, effect or audit')
    if source.read_bytes() != original_bytes:
        raise ValueError('Original Face workflow changed during export')
    return frontend, api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DESTINATION)
    args = parser.parse_args()
    output = args.output.resolve()
    if (output != DESTINATION.resolve() and not output.is_relative_to(ROOT / 'artifacts/development')):
        raise ValueError('Only the additive example directory or an owned private draft is supported')
    for mode, filename in FILES.items():
        frontend, _api = graph_for(mode)
        write_new(output / filename, frontend)
        print(json.dumps({'file': filename, 'nodes': len(frontend['nodes']), 'links': len(frontend['links'])}))


if __name__ == '__main__':
    main()
