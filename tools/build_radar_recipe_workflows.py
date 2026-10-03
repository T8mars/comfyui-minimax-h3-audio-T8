"""Additive generic Union2 native40 and CastSolo full/resume templates.

Public placeholders intentionally cannot Queue until the user selects inputs.
No private asset, previous failed Turbo4+4 recipe or implied human acceptance.
"""
import argparse
import json
from pathlib import Path

from tools import build_modular_fast_h3_v2_workflow as shared
from tools.audit_modular_sampling_compat import write_new
from tools.modular_frontend_layout import spread_frontend_columns

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'examples/workflows/69-radar-native-recipes'
FILES = {'union_full': 'Union2_Native40_Full_Save_EXP.json',
         'union_cold': 'Union2_Native40_Completed_Cold_Delivery_EXP.json',
         'cast_full': 'MV_CastSolo_AB_Full_EXP.json',
         'cast_resume': 'MV_CastSolo_AB_Same_Output_Resume_EXP.json'}


def node(kind, **inputs):
    return {'class_type': kind, 'inputs': inputs}


def union_graph(cold=False):
    graph = {
        '7': node('VAELoader', vae_name='minimax_h3_video_vae_fp16.safetensors'),
        '8': node('VAELoader', vae_name='minimax_h3_audio_vae_fp32.safetensors'),
    }
    if cold:
        graph['60'] = node('MiniMaxH3StageLoadEXPT8', artifact_path='SELECT_SAVED_STAGE/manifest.json',
                           artifact_sha256='0' * 64, expected_stage='native_low')
        source = ['60', 1]
    else:
        graph.update({
            '1': node('UNETLoader', unet_name='minimax_h3_fl2va_int8_convrot.safetensors', weight_dtype='default'),
            '6': node('CLIPLoader', clip_name='qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', type='minimax', device='default'),
            '9': node('MiniMaxH3AudioConditioningT8', clip=['6', 0], video_vae=['7', 0], audio_vae=['8', 0],
                prompt='One adult makes a small natural hand gesture in a continuous stable shot. Quiet synchronized ambience.',
                width=448, height=256, length=124, task_type='T2VA', audio_mode='native', audio_denoise_strength=1.,
                add_source_as_reference=False, prompt_primary_audio_ordinal=0, strict_prompt_tags=True,
                ref_image_size='match', reference_video_policy='official_2_to_15s'),
            '100': node('LoadVideo', file='SELECT_SOURCE_VIDEO.mp4'),
            '101': node('GetVideoComponents', video=['100', 0]),
            '107': node('SolidMask', value=0., width=448, height=256),
            '108': node('SolidMask', value=1., width=192, height=192),
            '109': node('MaskComposite', destination=['107', 0], source=['108', 0], x=128, y=64, operation='add'),
            '110': node('MiniMaxH3FunUnion2LoaderEXPT8', control_net_name='minimax_h3_fun_controlnet_union2_full_2688.safetensors'),
            '111': node('MiniMaxH3FunUnion2ApplyEXPT8', model=['1', 0], positive=['9', 0], union_control=['110', 0],
                vae=['7', 0], source_video=['101', 0], regen_mask=['109', 0], width=448, height=256, length=124,
                strength=1., start_percent=0., end_percent=1., broadcast_single_mask=True),
            '90': node('MiniMaxH3DualClockSamplerT8', model=['111', 0], av_latent=['9', 1], steps=40,
                shift_video=12., shift_audio=3., sampler_name='dual_clock_euler', scheduler='native_flow'),
            '10': node('MiniMaxH3NativeStageBindEXPT8', model=['90', 0], sampler=['90', 1], sigmas=['90', 2],
                av_latent=['9', 1], stage='native_low'),
            '11': node('RandomNoise', noise_seed=2609032101),
            '12': node('BasicGuider', model=['10', 0], conditioning=['111', 1]),
            '13': node('MiniMaxH3StageSamplerEXPT8', noise=['11', 0], guider=['12', 0], sampler=['10', 1],
                sigmas=['10', 2], latent_image=['9', 1], stage_context=['10', 3]),
            '50': node('MiniMaxH3StageSaveEXPT8', stage_result=['13', 2], prefix='Union2/native40'),
            '201': node('PreviewAny', source=['110', 1]), '202': node('PreviewAny', source=['111', 2]),
            '213': node('PreviewAny', source=['13', 3]),
        })
        source = ['50', 0]
    graph.update({
        '14': node('MiniMaxH3AVDecodeT8', av_latent=source, video_vae=['7', 0], audio_vae=['8', 0]),
        '208': node('MiniMaxH3OutputTrimT8', frames=['14', 0], audio=['14', 1], start_seconds=0., duration_seconds=5., fps=24.),
        '16': node('MiniMaxH3SafeAVSaveT8Advanced', images=['208', 0], audio=['208', 1],
            filename_prefix='MiniMaxH3/Union2/native40_' + ('cold' if cold else 'full'), crf=18),
    })
    if cold:
        graph['201'] = node('PreviewAny', source=['60', 2])
    return graph


def cast_graph():
    return {
        '1': node('UNETLoader', unet_name='minimax_h3_ref2va_int8_convrot.safetensors', weight_dtype='default'),
        '2': node('MiniMaxH3LoRACompatibilityLoaderT8Advanced', model=['1', 0],
            lora_name='minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors', strength_model=1.),
        '3': node('CLIPLoader', clip_name='qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', type='minimax', device='default'),
        '4': node('VAELoader', vae_name='minimax_h3_video_vae_fp16.safetensors'),
        '5': node('VAELoader', vae_name='minimax_h3_audio_vae_fp32.safetensors'),
        '6': node('LoadImage', image='SELECT_PERFORMER_A.png'), '7': node('LoadImage', image='SELECT_PERFORMER_B.png'),
        '8': node('LoadAudio', audio='SELECT_FULL_SONG.wav'), '9': node('LoadAudio', audio='SELECT_ALIGNED_VOCAL.wav'),
        '10': node('MiniMaxH3MVVocalLockScenePlannerV2T8Advanced', full_song=['8', 0], vocal_lock_audio=['9', 0],
            min_scene_seconds=5., target_scene_seconds=5.166666666666667, max_scene_seconds=6.,
            analysis_hop_ms=50, vocal_active_ratio=.12, manual_boundaries_json='[5.166666666666667]'),
        '11': node('MiniMaxH3MVCastSoloPlanEXPT8', scene_plan=['10', 0],
            performer_a_description='the adult performer in reference A', performer_b_description='the adult performer in reference B',
            assignments_json='[{"scene_index":0,"performer_id":"A","exact_vocal_text":""},{"scene_index":1,"performer_id":"B","exact_vocal_text":""}]',
            global_creative_prompt='A quiet studio solo vocal performance. One performer at a time, plain background.',
            visual_style='natural facial detail, diffuse studio light', scene_directions_json='',
            vocal_content_type='singing', vocal_language='Chinese'),
        '12': node('MiniMaxH3MVCastSoloRendererEXPT8', model=['2', 0], clip=['3', 0], video_vae=['4', 0], audio_vae=['5', 0],
            reference_a=['6', 0], reference_b=['7', 0], full_song=['8', 0], vocal_lock_audio=['9', 0], cast_solo_plan=['11', 0],
            chain_id='SELECT_NEW_CHAIN_FOR_THIS_AB_REQUEST', width=448, height=256, base_seed=2610022101, steps=4,
            shift_video=12., shift_audio=3., sampler_name='euler', scheduler='simple', resume_existing=True,
            filename_prefix='CastSolo/AB', bit_depth=8, crf=18,
            model_id='minimax_h3_ref2va_int8_convrot+official_ref2v_turbo4_v0.1'),
        '20': node('PreviewAny', source=['11', 1]), '21': node('PreviewAny', source=['12', 5]),
        '22': node('PreviewAny', source=['2', 1]),
    }


def graph_for(variant):
    if variant not in FILES:
        raise ValueError('Choose a named native recipe template')
    return union_graph(variant == 'union_cold') if variant.startswith('union_') else cast_graph()


def load_info():
    info = shared.load_live_info()
    import nodes
    from comfy_extras.nodes_video import LoadVideo, GetVideoComponents
    from comfy_extras.nodes_audio import LoadAudio
    from comfy_extras.nodes_mask import SolidMask, MaskComposite
    selected = {cls.__name__: cls for cls in (LoadVideo, GetVideoComponents, LoadAudio, SolidMask, MaskComposite)}
    selected['LoadImage'] = nodes.LoadImage
    nodes.NODE_CLASS_MAPPINGS.update(selected)
    info.update({key: shared.native_info(key, cls) for key, cls in selected.items()})
    return json.loads(json.dumps(info))


def build_candidate(variant, info):
    graph, selected = shared.selected_frontend_schema(graph_for(variant), info)
    workflow = shared.convert(graph, selected, 'RADAR native recipe / ' + variant + ' EXP')
    text = ('本地EXP，不认证人审／任意素材。先选自己的模型、源视频／参考图／对齐音轨；SELECT占位不应Queue。'
        '\nUnion2 full：原生40步CFG1／12-3／无Turbo。源必须显式448×256、124帧24fps，MASK白重绘黑保留、单帧复制显式true。'
        '更改画布须同时对齐源、MASK、条件与Apply；此控制不保证黑区RGB或生成音轨精确不变。'
        'Cold只填完成StageSave真实path／完整SHA，零采样；不接MODEL／CLIP／Union，不改旧存档。失败的Turbo4+4不作推荐模板。'
        '\nCastSolo：A/B独立一张参考，两条实际对齐音轨，场景索引／角色必须覆盖实际scene_plan，空歌词不猜原文。'
        '默认示意两镜10⅓秒；换歌曲须同步改边界及assignments。首次用新chain；Resume保留同一output根、chain、素材、设置与producer。'
        'Resume图与Full执行节点和参数完全相同，仅操作说明不同；已完成返回master／零采样，未完成只按原resume规则补缺失场景。'
        '它不是任意移址或免MODEL检查的解码图；不能把多人同时同框当交替独唱。'
        '\n保留旧图／默认，不自动采用、审核、下载或触发队列。机械资格见专题文档。')
    note_id = workflow['last_node_id'] + 1
    workflow['nodes'].append({'id': note_id, 'type': 'MarkdownNote', 'title': '使用前必读 / Full vs exact resume',
        'pos': [0, -700], 'size': [1150, 630], 'flags': {}, 'order': len(graph), 'mode': 0,
        'inputs': [], 'outputs': [], 'properties': {}, 'widgets_values': [text + '\n当前图：' + variant]})
    workflow['last_node_id'] = note_id
    workflow.setdefault('extra', {})['t8_radar_recipe'] = {'schema': 't8.radar.recipe-example.v1',
        'variant': variant, 'human_quality_accepted': False, 'public_inputs_are_placeholders': True}
    spread_frontend_columns(workflow)
    return graph, workflow, shared.audit_candidate(graph, workflow, selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DESTINATION)
    options = parser.parse_args()
    dest = options.output_dir.resolve()
    if not dest.is_relative_to(ROOT) or any((dest / name).exists() for name in FILES.values()):
        parser.error('Use new files within the project, never overwrite old workflows')
    info = load_info()
    rows = []
    for variant, filename in FILES.items():
        _, workflow, report = build_candidate(variant, info)
        write_new(dest / filename, workflow)
        rows.append({'variant': variant, 'serialization': report, 'queue_requires_user_inputs': True})
    print(json.dumps({'destination': str(dest), 'templates': rows}))


if __name__ == '__main__':
    main()
