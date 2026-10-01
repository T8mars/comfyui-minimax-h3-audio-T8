"""Build private freeze/cold-resume pairs for every sampled Audio Refine source.

Freeze the *last video-generating* native AV output, not necessarily the first
SamplerCustomAdvanced: learned/PDD spatial dual passes have two video samplers
before their audio-only tail. The old examples remain read-only.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.build_modular_h16_storage_workflow import Draft
from tools.build_modular_audio_refine_resume_workflows import (
    CHECKPOINT_LOAD, CHECKPOINT_SAVE, SOURCE as RELAY_SOURCE,
    pair_from_current_source,
)
from tools.build_modular_audio_refine_workflows import FAMILIES, sources, split_frontend

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'artifacts/development/modular-sampling-m4-audio-refine-all-resume-20260924/candidate-v7'
CHECKPOINT_GUARD = 'MiniMaxH3AudioRefineFrozenAVFrameGuardEXPT8'
CONTEXT_COMMIT_GATE = 'MiniMaxH3AudioRefineContextCommitGateEXPT8'


def _single(draft, types):
    found = [node for node in draft.nodes.values() if node['type'] in types]
    if len(found) != 1:
        raise ValueError(f'Expected one Audio Refine {types}, found {len(found)}')
    return found[0]


def _source(draft, node, field):
    item = next(item for item in node['inputs'] if item['name'] == field)
    if item['link'] is None:
        raise ValueError(f'Audio Refine {node["type"]}.{field} is not connected')
    link = draft.links[item['link']]
    return draft.nodes[link[1]], link[2], link[5]


def _last_video_sampler(draft):
    setup = _single(draft, FAMILIES)
    source, slot, dtype = _source(draft, setup, 'av_latent')
    if source['type'] != 'SamplerCustomAdvanced' or slot != 0 or dtype != 'LATENT':
        raise ValueError('Audio Refine tail no longer consumes final video sampler AV')
    return source


def _note(draft, title, message, pos):
    return draft.make('MarkdownNote', title, [], [], [message], pos)


def _bind_int_widget_to_guard(draft, target, field, guard, slot):
    item = next((item for item in target['inputs'] if item['name'] == field), None)
    if item is None:
        item = {'name': field, 'type': 'INT', 'link': None,
                'widget': {'name': field}}
        target['inputs'].append(item)
    elif item['type'] != 'INT':
        raise ValueError(f'Cold Audio Refine {target["type"]}.{field} is no longer INT')
    else:
        draft.disconnect(target, field)
        item.setdefault('widget', {'name': field})
    draft.connect((guard['id'], slot), target, field, 'INT')


def freeze_frontend(source):
    draft = Draft(source)
    video_sampler = _last_video_sampler(draft)
    checkpoint_source = (video_sampler['id'], 0)
    eav = [node for node in draft.nodes.values()
           if node['type'] == 'MiniMaxH3EnhanceAVideoT8Advanced']
    if eav:
        if len(eav) != 1:
            raise ValueError('Expected exactly one first-video EAV runtime')
        audit = _single(draft, {'MiniMaxH3EnhanceAVideoAuditT8Advanced'})
        av_source, av_slot, av_type = _source(draft, audit, 'av_latent')
        runtime_source, runtime_slot, _ = _source(draft, audit, 'runtime')
        if ((av_source['id'], av_slot, av_type) !=
                (video_sampler['id'], 0, 'LATENT') or
                (runtime_source['id'], runtime_slot) != (eav[0]['id'], 1)):
            raise ValueError('First-video EAV audit is not paired with frozen AV')
        # Saving the audit output makes its effect check a required predecessor.
        checkpoint_source = (audit['id'], 0)
    save = draft.make(
        CHECKPOINT_SAVE, 'STEP 1 · freeze final video AV before audio refinement',
        [('av_latent', 'LATENT')],
        [('av_latent', 'LATENT'), ('status', 'STRING'),
         ('checkpoint_path', 'STRING'), ('file_sha256', 'STRING'),
         ('manifest_json', 'STRING'), ('report_json', 'STRING')],
        ['audio_refine_firstpass', 'audio_refine_firstpass', False, True, 8],
        (2300, 50))
    draft.connect(checkpoint_source, save, 'av_latent', 'LATENT')
    roots = [save['id']]
    context_saves = [node for node in draft.nodes.values()
                     if node['type'] == 'MiniMaxH3LongVideoContextSaveT8']
    if context_saves:
        if len(context_saves) != 1:
            raise ValueError('Expected exactly one long-video context saver')
        context_save = context_saves[0]
        delivery, delivery_slot, av_type = _source(draft, context_save, 'av_latent')
        original, original_slot, original_type = _source(
            draft, delivery, 'original_continuation_av_latent')
        planner, planned_slot, planned_type = _source(draft, context_save, 'save_context')
        if (delivery['type'] != 'MiniMaxH3AudioRefineLongVideoDeliveryT8Advanced' or
                (delivery_slot, av_type) != (0, 'LATENT') or
                (original['id'], original_slot, original_type) !=
                (video_sampler['id'], 0, 'LATENT') or
                (planner['type'], planned_slot, planned_type) !=
                ('MiniMaxH3LongVideoPlannerT8', 8, 'BOOLEAN')):
            raise ValueError('Long-video continuation AV/context plan changed')
        commit = draft.make(
            CONTEXT_COMMIT_GATE, 'Commit long context only after verified freeze',
            [('checkpoint_status', 'STRING'), ('planned_save_context', 'BOOLEAN')],
            [('save_context', 'BOOLEAN')], [], (2600, 260))
        draft.connect((save['id'], 1), commit, 'checkpoint_status', 'STRING')
        draft.connect((planner['id'], planned_slot), commit,
                      'planned_save_context', 'BOOLEAN')
        draft.disconnect(context_save, 'av_latent')
        draft.connect((save['id'], 0), context_save, 'av_latent', 'LATENT')
        draft.disconnect(context_save, 'save_context')
        draft.connect((commit['id'], 0), context_save, 'save_context', 'BOOLEAN')
        roots.append(context_save['id'])
    note = _note(draft, 'Freeze final video AV · keep all three values',
                 'Set confirm_save=true only after reviewing the final video pass. '
                 'Copy checkpoint_path, file_sha256 and manifest_json to STEP 2. '
                 'For learned/PDD double-video routes this checkpoint is after both video samplers. '
                 'Long-video context writes only after verified checkpoint Save.',
                 (2250, -360))
    return draft.prune((*roots, note['id']))


def _reaches(draft, start, target):
    pending, seen = [start], set()
    while pending:
        current = pending.pop()
        if current == target:
            return True
        if current in seen:
            continue
        seen.add(current)
        pending.extend(link[3] for link in draft.graph['links'] if link[1] == current)
    return False


def resume_frontend(source):
    long_relay = any(node['type'] == 'MiniMaxH3PromptRelayLongVideoConditioningT8Advanced'
                     for node in source['nodes'])
    draft = Draft(split_frontend(source, abstain_safe=True,
                                 external_refine_relay=long_relay))
    video_sampler = _last_video_sampler(draft)
    source_gate = _single(draft, {'MiniMaxH3AudioRefineQualityGateT8Advanced'})
    source_gate_values = source_gate.get('widgets_values', [])
    if (len(source_gate_values) < 2 or
            type(source_gate_values[1]) is not int or source_gate_values[1] < 1):
        raise ValueError('Audio Refine source video_frame_count is not an explicit integer')
    expected_video_frame_count = source_gate_values[1]
    load = draft.make(
        CHECKPOINT_LOAD, 'STEP 2 · verify frozen final-video AV before audio tail',
        [], [('av_latent', 'LATENT'), ('status', 'STRING'),
             ('resume_verified', 'BOOLEAN'), ('checkpoint_id', 'STRING'),
             ('content_sha256', 'STRING'), ('file_sha256', 'STRING'),
             ('manifest_json', 'STRING'), ('report_json', 'STRING'),
             ('video_width', 'INT'), ('video_height', 'INT')],
        ['', 'PASTE_EXACT_SAVE_MANIFEST_JSON', '0' * 64, 8], (300, -360))
    guard = draft.make(
        CHECKPOINT_GUARD, 'STEP 2 · reject wrong-length frozen video',
        [('av_latent', 'LATENT')],
        [('av_latent', 'LATENT'), ('video_width', 'INT'),
         ('video_height', 'INT'), ('video_frame_count', 'INT')],
        [expected_video_frame_count], (650, -360))
    draft.connect((load['id'], 0), guard, 'av_latent', 'LATENT')
    consumers = [link for link in list(draft.graph['links'])
                 if link[1:3] == [video_sampler['id'], 0]]
    if len(consumers) < 2:
        raise ValueError('Final-video AV consumers changed')
    for link in consumers:
        if link[5] != 'LATENT':
            raise ValueError('Final-video AV consumer is not a LATENT input')
        target = draft.nodes[link[3]]
        field = target['inputs'][link[4]]['name']
        draft.disconnect(target, field)
        draft.connect((guard['id'], 0), target, field, 'LATENT')
    # The original EAV audit is a prerequisite of *freezing* the video AV.
    # A cold resume already authenticates that saved output, so routing it
    # through the original audit again would rebuild an unrelated EAV runtime.
    if any(node['type'] == 'MiniMaxH3EnhanceAVideoT8Advanced'
           for node in draft.nodes.values()):
        eav_audit = _single(draft, {'MiniMaxH3EnhanceAVideoAuditT8Advanced'})
        for link in [link for link in list(draft.graph['links'])
                     if link[1:3] == [eav_audit['id'], 0]]:
            if link[5] != 'LATENT':
                raise ValueError('EAV audit AV consumer is not LATENT')
            target = draft.nodes[link[3]]
            field = target['inputs'][link[4]]['name']
            draft.disconnect(target, field)
            draft.connect((guard['id'], 0), target, field, 'LATENT')
    # Every standalone refinement prompt must match the authenticated frozen
    # AV. Original widget values remain in the frontend graph for old schema
    # compatibility, but connected values come only from the frame guard.
    for node in list(draft.nodes.values()):
        if node['type'] != 'MiniMaxH3AudioConditioningT8':
            continue
        for field, slot in (('width', 1), ('height', 2), ('length', 3)):
            _bind_int_widget_to_guard(draft, node, field, guard, slot)
    gate = _single(draft, {'MiniMaxH3AudioRefineQualityGateT8Advanced'})
    _bind_int_widget_to_guard(draft, gate, 'video_frame_count', guard, 3)
    final_saves = [node for node in draft.nodes.values()
                   if node['type'] == 'SaveVideo' and _reaches(draft, gate['id'], node['id'])]
    if len(final_saves) != 1:
        raise ValueError('Audio Refine final Quality Gate delivery changed')
    note = _note(draft, 'Cold resume · audio tail only',
                 'Paste all three verified Save values. Placeholder path/manifest/SHA fails '
                 'before the audio tail. This is explicit checkpoint reuse, not automatic cache.',
                 (300, -680))
    graph = draft.prune((gate['id'], final_saves[0]['id'], note['id']))
    remaining = {node['id'] for node in graph['nodes']}
    kinds = [node['type'] for node in graph['nodes']]
    if (video_sampler['id'] in remaining or kinds.count('SamplerCustomAdvanced') != 1 or
            kinds.count(CHECKPOINT_LOAD) != 1 or kinds.count(CHECKPOINT_GUARD) != 1 or
            kinds.count(CHECKPOINT_SAVE) or
            'MiniMaxH3EnhanceAVideoT8Advanced' in kinds or
            'MiniMaxH3EnhanceAVideoAuditT8Advanced' in kinds or
            kinds.count('MiniMaxH3AudioRefineQualityGateT8Advanced') != 1):
        samplers = [node['id'] for node in graph['nodes']
                    if node['type'] == 'SamplerCustomAdvanced']
        raise ValueError(f'Cold Audio Refine graph can still sample video or lost its quality gate: '
                         f'video={video_sampler["id"]}, samplers={samplers}, '
                         f'load={kinds.count(CHECKPOINT_LOAD)}, gate={kinds.count("MiniMaxH3AudioRefineQualityGateT8Advanced")}')
    return graph


def pairs_from_current_sources():
    result = {}
    for path in sources():
        if path == RELAY_SOURCE:
            result[path] = pair_from_current_source()
        else:
            source = json.loads(path.read_text(encoding='utf-8'))
            result[path] = freeze_frontend(source), resume_frontend(source)
    return result
