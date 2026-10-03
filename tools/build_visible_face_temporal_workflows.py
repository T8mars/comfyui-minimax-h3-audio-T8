"""Two NEW opt-in temporal MASK / explicit low-denoise Face templates.

Never rewrites the original fourteen static-mask or older source workflows.
MASK media, exact source/clock and completed Stage path/SHA remain user inputs.
Existing per-stage Relay stays external, cold is actual delivery only.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import build_visible_face_mask_workflows as visible  # noqa: E402
from tools.audit_modular_sampling_compat import write_new  # noqa: E402
from tools.build_modular_h16_storage_workflow import Draft  # noqa: E402
from tools.modular_frontend_layout import spread_frontend_columns  # noqa: E402

DESTINATION = visible.DESTINATION
FILES = {mode: f'R07_Face_standard_{mode}_Temporal_MASK_Light005_Relay_EXP.json'
         for mode in visible.storage.MODES}


def graph_for(mode):
    if mode not in FILES:
        raise ValueError('Explicit full_save or cold_delivery required')
    base, api = visible.graph_for('standard', mode, effect='relay')
    draft = Draft(base)
    image = next(n for n in draft.nodes.values() if n['type'] == 'LoadImage')
    red = next(n for n in draft.nodes.values() if n['type'] == 'ImageToMask')
    bind = next(n for n in draft.nodes.values() if n['type'] == 'MiniMaxH3VisibleFaceMaskBindEXPT8')
    delivery = next(n for n in draft.nodes.values() if n['type'] == 'SaveVideo')
    load = draft.make('LoadVideo', 'REPLACE with lossless full-source 24fps per-frame visible MASK',
        [('file', 'COMBO')], [('VIDEO', 'VIDEO')], ['REPLACE_visible_face_per_frame_24fps.mkv'], (5900, -500))
    components = draft.make('GetVideoComponents', 'MASK RGB only; MASK audio is not used',
        [('video', 'VIDEO')], [('images', 'IMAGE'), ('audio', 'AUDIO'), ('fps', 'FLOAT')], [], (6450, -500))
    draft.disconnect(red, 'image')
    draft.connect((load['id'], 0), components, 'video', 'VIDEO')
    draft.connect((components['id'], 0), red, 'image', 'IMAGE')
    api.pop(str(image['id']))
    api[str(load['id'])] = {'class_type': 'LoadVideo', 'inputs': {'file': load['widgets_values'][0]}}
    api[str(components['id'])] = {'class_type': 'GetVideoComponents', 'inputs': {'video': [str(load['id']), 0]}}
    api[str(red['id'])]['inputs']['image'] = [str(components['id']), 0]
    api[str(bind['id'])]['inputs']['broadcast_single_mask'] = False
    bind['widgets_values'][-1] = False
    bind['title'] = 'Full per-frame MASK; do NOT broadcast static half-opacity masks across moving faces'
    samplers = [(key, n) for key, n in api.items() if n['class_type'] == 'MiniMaxH3FaceRefineSamplerT8Advanced']
    if mode == 'full_save':
        if len(samplers) != 1 or samplers[0][1]['inputs']['steps'] != 12:
            raise ValueError('One original twelve-step Standard recipe expected')
        key, sampler = samplers[0]
        sampler['inputs']['denoise'] = .05
        frontend = draft.nodes[int(key)]
        # Original explicit widget order is checked rather than guessed.
        if frontend['widgets_values'][:2] != [12, .45]:
            raise ValueError('Original Face sampler widgets changed')
        frontend['widgets_values'][1] = .05
        frontend['title'] = 'OPT-IN light repair denoise0.05, original12 steps; NOT a new default'
    elif samplers:
        raise ValueError('Cold delivery must not contain a Face sampler')
    prefix = 'MiniMaxH3/R07_TemporalLight/' + mode
    api[str(delivery['id'])]['inputs']['filename_prefix'] = prefix
    delivery['widgets_values'][0] = prefix
    roots = [delivery['id']] + [n['id'] for n in draft.nodes.values() if n['type'] == 'MiniMaxH3StageSaveEXPT8']
    graph = draft.prune(roots)
    if set(api) != {str(n['id']) for n in graph['nodes']}:
        raise ValueError('Temporal graph lost a required source dependency')
    graph['extra']['t8_visible_face_mask'].update(
        single_image_broadcast_explicit=False, temporal_external_mask=True,
        original_sampler_unchanged=False, explicit_opt_in_denoise=.05, original_steps=12,
        existing_external_relay_preserved=True,
        status='specific_light_recipe_and_temporal_MASK_human_review_pending_not_general_occlusion_model',
        mask_requirements='Complete original-source frame coordinates/24fps; lossless RGB red0..1, no guessed retime or segmentation')
    spread_frontend_columns(graph)
    return graph, api


def main():
    for mode, filename in FILES.items():
        frontend, api = graph_for(mode)
        path = DESTINATION / filename
        if path.exists():
            if json.loads(path.read_text(encoding='utf8')) != frontend:
                raise ValueError('Existing opt-in file differs; never overwrite it')
        else:
            write_new(path, frontend)
        print(json.dumps({'file': filename, 'nodes': len(api), 'status': 'new_opt_in_template_not_human_approved'}))


if __name__ == '__main__':
    main()
