"""Add external visible-MASK delivery to ALL seven existing Face families.

Old source graphs are read-only. Original steps, audio, review/confirmation and
stage identities remain; cold templates retain blank exact receipt fields.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import build_formal_face_refine_storage_workflows as storage  # noqa: E402
from tools.audit_modular_sampling_compat import write_new  # noqa: E402
from tools.build_modular_h16_storage_workflow import Draft  # noqa: E402
from tools.modular_frontend_layout import spread_frontend_columns  # noqa: E402

DESTINATION = ROOT / 'examples/workflows/65-radar-visible-face-mask'
FILES = {(variant, mode): f'R07_Face_{variant}_{mode}_Visible_MASK_EXP.json'
         for variant in storage.face.VARIANTS for mode in storage.MODES}
MASK_OUTPUTS = {
    'MiniMaxH3FaceRefineStitchAuditT8Advanced': (0, 1),
    'MiniMaxH3FaceRefineParityStitchT8Advanced': (0, 1),
    'MiniMaxH3FaceRefineQualityGateT8Advanced': (0, 1),
    'MiniMaxH3FaceRefineManualReviewT8Advanced': (1, 2),
    'MiniMaxH3FaceRefineWindowStudioComposeT8Advanced': (0, 1),
}


def graph_for(variant, mode, *, effect='none'):
    if (variant, mode) not in FILES:
        raise ValueError('Unknown visible face workflow family')
    original, old_api = storage.graph_for(variant, effect, mode)
    draft, api = Draft(original), deepcopy(old_api)
    video = [node for node in draft.nodes.values() if node['type'] == 'CreateVideo']
    source = [node for node in draft.nodes.values() if node['type'] == 'GetVideoComponents']
    delivery = [node for node in draft.nodes.values() if node['type'] == 'SaveVideo']
    if len(video) != 1 or len(source) != 1 or len(delivery) != 1:
        raise ValueError('One original whole-source video/audio and one final delivery required')
    video, source, delivery = video[0], source[0], delivery[0]
    candidate = storage._source(draft, video, 'images')
    audio = storage._source(draft, video, 'audio')
    producer = draft.nodes[candidate[0]]
    mask_producer = producer
    mask_candidate_slot = candidate[1]
    if producer['type'] == 'MiniMaxH3FaceRefineManual512RelativeBaselineT8Advanced':
        # This existing guard returns candidate_frames unchanged. Keep it in
        # the image path; use the exact upstream Stitch MASK, not its report.
        parent = storage._source(draft, producer, 'candidate_frames')
        mask_producer, mask_candidate_slot = draft.nodes[parent[0]], parent[1]
        if candidate[1] != 0 or mask_producer['type'] != 'MiniMaxH3FaceRefineParityStitchT8Advanced':
            raise ValueError('Unrecognized parity baseline passthrough')
    if producer['type'] == 'MiniMaxH3MultiFaceCompositeT8Advanced':
        support, support_name, support_type = (producer['id'], 1), 'multiface_composite', 'H3_T8_MULTIFACE_COMPOSITE'
    else:
        if mask_producer['type'] not in MASK_OUTPUTS or mask_candidate_slot != MASK_OUTPUTS[mask_producer['type']][0]:
            raise ValueError('Final image producer must expose its exact full-source changed/accepted mask')
        support, support_name, support_type = (mask_producer['id'], MASK_OUTPUTS[mask_producer['type']][1]), 'changed_alpha', 'MASK'
    image = draft.make('LoadImage', 'REPLACE with full-coordinate visible-face RGB MASK image',
        [('image', 'COMBO')], [('IMAGE', 'IMAGE'), ('MASK', 'MASK')],
        ['REPLACE_visible_face_full_frame_MASK.png', 'image'], (5900, -500))
    red = draft.make('ImageToMask', 'Explicit RED channel: white allows existing changes / black protects',
        [('image', 'IMAGE'), ('channel', 'COMBO')], [('MASK', 'MASK')], ['red'], (6450, -500))
    bind = draft.make('MiniMaxH3VisibleFaceMaskBindEXPT8',
        'Bind full source + visible MASK (explicit static one-frame broadcast)',
        [('source_images', 'IMAGE'), ('visible_mask', 'MASK'), ('fps', 'FLOAT'),
         ('source_start_frame', 'INT'), ('broadcast_single_mask', 'BOOLEAN')],
        [('visible_face_mask', 'T8_VISIBLE_FACE_COMPOSITE_MASK'), ('report_json', 'STRING')],
        [24., 0, True], (7000, -500))
    finish = draft.make('MiniMaxH3VisibleFaceCompositeEXPT8', 'Restrict FINAL already-blended result; no alpha squared',
        [('visible_face_mask', 'T8_VISIBLE_FACE_COMPOSITE_MASK'), ('candidate_images', 'IMAGE'),
         ('changed_alpha', 'MASK'), ('audio', 'AUDIO'), ('multiface_composite', 'H3_T8_MULTIFACE_COMPOSITE')],
        [('images', 'IMAGE'), ('audio', 'AUDIO'), ('effective_mask', 'MASK'), ('report_json', 'STRING')],
        [], (7500, 350))
    for edge, target, name, dtype in [((image['id'], 0), red, 'image', 'IMAGE'),
        ((red['id'], 0), bind, 'visible_mask', 'MASK'), ((source['id'], 0), bind, 'source_images', 'IMAGE'),
        ((source['id'], 2), bind, 'fps', 'FLOAT'), ((bind['id'], 0), finish, 'visible_face_mask', 'T8_VISIBLE_FACE_COMPOSITE_MASK'),
        (candidate, finish, 'candidate_images', 'IMAGE'), (support, finish, support_name, support_type),
        (audio, finish, 'audio', 'AUDIO')]:
        draft.connect(edge, target, name, dtype)
    draft.disconnect(video, 'images')
    draft.disconnect(video, 'audio')
    draft.connect((finish['id'], 0), video, 'images', 'IMAGE')
    draft.connect((finish['id'], 1), video, 'audio', 'AUDIO')
    api[str(image['id'])] = {'class_type': 'LoadImage', 'inputs': {'image': image['widgets_values'][0]}}
    api[str(red['id'])] = {'class_type': 'ImageToMask', 'inputs': {'image': [str(image['id']), 0], 'channel': 'red'}}
    api[str(bind['id'])] = {'class_type': bind['type'], 'inputs': {'source_images': [str(source['id']), 0],
        'visible_mask': [str(red['id']), 0], 'fps': [str(source['id']), 2], 'source_start_frame': 0, 'broadcast_single_mask': True}}
    api[str(finish['id'])] = {'class_type': finish['type'], 'inputs': {'visible_face_mask': [str(bind['id']), 0],
        'candidate_images': [str(candidate[0]), candidate[1]], support_name: [str(support[0]), support[1]],
        'audio': [str(audio[0]), audio[1]]}}
    api[str(video['id'])]['inputs'].update(images=[str(finish['id']), 0], audio=[str(finish['id']), 1])
    prefix = f'MiniMaxH3/R07_VisibleFace/{variant}_{mode}'
    # Change only this NEW template's explicit output prefix, not the old file.
    api[str(delivery['id'])]['inputs']['filename_prefix'] = prefix
    delivery['widgets_values'][0] = prefix
    roots = [delivery['id']] + [node['id'] for node in draft.nodes.values() if node['type'] == 'MiniMaxH3StageSaveEXPT8']
    graph = draft.prune(roots)
    if set(api) != {str(node['id']) for node in graph['nodes']}:
        raise ValueError('Visible MASK template lost an original dependency')
    graph.setdefault('extra', {})['t8_visible_face_mask'] = {
        'schema': 't8.visible-face-workflow.v1', 'family': variant, 'storage': mode,
        'status': 'explicit_source_mask_requires_input_and_final_review_not_quality_accepted',
        'single_image_broadcast_explicit': True, 'audio_original_source_edge': list(audio),
        'original_sampler_unchanged': True, 'original_accept_confirmation_unchanged': True}
    spread_frontend_columns(graph)
    return graph, api


def main():
    for (variant, mode), filename in FILES.items():
        frontend, api = graph_for(variant, mode)
        path = DESTINATION / filename
        if path.exists():
            if json.loads(path.read_text(encoding='utf8')) != frontend:
                raise ValueError('Existing graph differs; preserve it and explicitly review a new file')
        else:
            write_new(path, frontend)
        print(json.dumps({'file': filename, 'nodes': len(frontend['nodes']), 'api_nodes': len(api)}))


if __name__ == '__main__':
    main()
