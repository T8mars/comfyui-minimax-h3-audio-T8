"""Existing loaders only: LMS weights and domain-specific Orbit/Wallpaper LoRA.

These T8 joint-AV adaptation graphs do not claim upstream audio-off parity,
R64, loop quality or the author's multifactor Taomate3/LMS workflow.
"""
import argparse
import json
from pathlib import Path

from tools import build_modular_fast_h3_v2_workflow as shared
from tools.audit_modular_sampling_compat import write_new
from tools.modular_frontend_layout import spread_frontend_columns
from tools.build_radar_recipe_workflows import node, load_info

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'examples/workflows/70-radar-model-compatibility'
WEIGHTS = {'orbit': 'minimax_h3_flf2v_lora_v1.safetensors',
           'wallpaper': 'minimax_h3_live_wallpaper_v1.0_ref2va_r32.safetensors',
           'lms': 'h3_upscaler_lms_v0.1.safetensors'}
FILES = {family + '_' + mode: family.title() + '_T8_Compatibility_' + mode + '_EXP.json'
         for family in WEIGHTS for mode in ('baseline', 'candidate')}


def model_weight_choice(family):
    """Explicit subfolder choice, never change an existing node's default."""
    return str(Path('zz_radar') / WEIGHTS[family])


def graph_for(variant):
    if variant not in FILES:
        raise ValueError('Choose a named domain-specific model comparison')
    family, mode = variant.split('_')
    if family == 'lms':
        return {
            '1': node('MiniMaxH3HyperFlowCurveTailLoadEXPT8', artifact_path='SELECT_COMPLETED_CURVE_TAIL/curve-tail.safetensors', artifact_sha256='0' * 64),
            '2': node('MiniMaxH3LearnedLatentUpscaleT8Advanced', av_latent=['1', 0],
                model_name=model_weight_choice('lms') if mode == 'candidate' else 'minimax_h3_latent_upscaler_3d_fp16.safetensors',
                size_mode='scale_by', scale_by=2., target_megapixels=1., target_width=1024, target_height=1024,
                aspect_policy='preserve_source', max_anisotropy=1.05, precision='fp16', release_policy='clear_after'),
            '7': node('VAELoader', vae_name='minimax_h3_video_vae_fp16.safetensors'),
            '8': node('VAELoader', vae_name='minimax_h3_audio_vae_fp32.safetensors'),
            '14': node('MiniMaxH3AVDecodeT8', av_latent=['2', 0], video_vae=['7', 0], audio_vae=['8', 0]),
            '16': node('MiniMaxH3SafeAVSaveT8Advanced', images=['14', 0], audio=['14', 1],
                filename_prefix='MiniMaxH3/RADAR/LMS_' + mode, crf=18),
            '20': node('PreviewAny', source=['2', 3]), '21': node('PreviewAny', source=['1', 2]),
        }
    orbit = family == 'orbit'
    graph = {
        '1': node('UNETLoader', unet_name='minimax_h3_fl2va_pruned_int8_convrot.safetensors' if orbit else 'minimax_h3_ref2va_int8_convrot.safetensors', weight_dtype='default'),
        '2': node('MiniMaxH3LoRACompatibilityLoaderT8Advanced', model=['1' if orbit else '3', 0],
            lora_name=model_weight_choice(family) if mode == 'candidate' else 'disabled', strength_model=1.),
        '6': node('CLIPLoader', clip_name='qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', type='minimax', device='default'),
        '7': node('VAELoader', vae_name='minimax_h3_video_vae_fp16.safetensors'),
        '8': node('VAELoader', vae_name='minimax_h3_audio_vae_fp32.safetensors'),
        '200': node('LoadImage', image='SELECT_ONE_SOURCE_REFERENCE.png'),
        '201': node('ImageScale', image=['200', 0], upscale_method='lanczos', width=768 if orbit else 512,
            height=768 if orbit else 512, crop='center'),
        '9': node('MiniMaxH3AudioConditioningT8', clip=['6', 0], video_vae=['7', 0], audio_vae=['8', 0],
            prompt=('A frozen scene with one adult. A single uninterrupted camera circuit travels around the subject and returns to the original viewpoint. Keep the person and environment stationary; only perspective and parallax change. No cuts or added objects.' if orbit else
                'live_wallpaper: Animate <Picture 1> with a fixed camera and gentle movement in existing hair and clothing, while preserving the original subject and composition.'),
            width=768 if orbit else 512, height=768 if orbit else 512, length=73 if orbit else 124,
            task_type='FL2VA' if orbit else 'Ref2VA', audio_mode='native', audio_denoise_strength=1.,
            add_source_as_reference=False, prompt_primary_audio_ordinal=0, strict_prompt_tags=True,
            ref_image_size='match', reference_video_policy='official_2_to_15s'),
        '90': node('MiniMaxH3DualClockSamplerT8', model=['2', 0], av_latent=['9', 1], steps=28 if orbit else 4,
            shift_video=12., shift_audio=3., sampler_name='dual_clock_euler' if orbit else 'euler', scheduler='native_flow' if orbit else 'simple'),
        '11': node('RandomNoise', noise_seed=2610031101 if orbit else 2610031102),
        '12': node('BasicGuider', model=['90', 0], conditioning=['9', 0]),
        '13': node('SamplerCustomAdvanced', noise=['11', 0], guider=['12', 0], sampler=['90', 1], sigmas=['90', 2], latent_image=['9', 1]),
        '14': node('MiniMaxH3AVDecodeT8', av_latent=['13', 0], video_vae=['7', 0], audio_vae=['8', 0]),
        '20': node('PreviewAny', source=['2', 1]), '21': node('PreviewAny', source=['9', 5]),
    }
    if orbit:
        graph['9']['inputs'].update(first_frame=['201', 0], last_frame=['201', 0])
        graph['15'] = node('CreateVideo', images=['14', 0], fps=24., bit_depth=8)
        graph['16'] = node('SaveVideo', video=['15', 0], filename_prefix='MiniMaxH3/RADAR/Orbit_' + mode,
            format='mp4', **{'format.codec': 'auto'})
    else:
        graph['3'] = node('MiniMaxH3LoRACompatibilityLoaderT8Advanced', model=['1', 0],
            lora_name='minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors', strength_model=1.)
        graph['9']['inputs']['ref_images.ref_image_0'] = ['201', 0]
        graph['16'] = node('MiniMaxH3SafeAVSaveT8Advanced', images=['14', 0], audio=['14', 1],
            filename_prefix='MiniMaxH3/RADAR/Wallpaper_R32_' + mode, crf=18)
    return graph


def build_candidate(variant, info):
    graph, selected = shared.selected_frontend_schema(graph_for(variant), info)
    workflow = shared.convert(graph, selected, 'RADAR existing loader / ' + variant + ' EXP')
    family = variant.split('_')[0]
    text = ('仅现有加载节点；EXP、人审未验。先选自己的文件；SELECT占位不可Queue。Baseline与Candidate保持同素材／seed／配方，'
        '只换该模型权重或LoRA disabled→strength1。不同家族不能混为一个通用加速开关。'
        '\nLMS是完整3D放大器权重，不是LoRA。读同一完成curveTAIL→2x→完整AV；保留原音频，不跑DiT。不能因tiny CPU成功宣称16MP或更快。'
        '\nOrbit：pruned FL2VA、768方／73帧／28步／CFG1、同一经显式center resize的首尾图、无Turbo。'
        '这是T8 joint-AV计算后静音视频交付，内层仍有audio stream；不是作者audio-off推理或逐字prompt的严格复现，不保证360°／闭环／冻结脸。'
        '\nWallpaper仅R32／Ref2VA／一张参考，live_wallpaper触发；本图是既有Ref2V Turbo4单变量兼容小样，'
        '不是作者R64＋Taomate3＋LMS多因素图。增加参考时要同步原生顺序／Picture tags；R32相机和身份效果需实片确认。'
        '\n不改旧默认／原图／采样器，不自动下载、审核、采用或Queue。详见docs/RADAR_MODEL_COMPATIBILITY_EXP.md。')
    note_id = workflow['last_node_id'] + 1
    workflow['nodes'].append({'id': note_id, 'type': 'MarkdownNote', 'title': family + ' / 配方与资格边界',
        'pos': [0, -700], 'size': [1150, 660], 'flags': {}, 'order': len(graph), 'mode': 0,
        'inputs': [], 'outputs': [], 'properties': {}, 'widgets_values': [text]})
    workflow['last_node_id'] = note_id
    workflow.setdefault('extra', {})['t8_radar_model'] = {'schema': 't8.radar.model-example.v1', 'variant': variant,
        'human_quality_accepted': False, 'author_parity_claimed': False}
    spread_frontend_columns(workflow)
    return graph, workflow, shared.audit_candidate(graph, workflow, selected)


def model_info():
    info = load_info()
    import nodes
    info['ImageScale'] = shared.native_info('ImageScale', nodes.ImageScale)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DESTINATION)
    options = parser.parse_args()
    dest = options.output_dir.resolve()
    if not dest.is_relative_to(ROOT) or any((dest / name).exists() for name in FILES.values()):
        parser.error('Use new files within this project, never overwrite old workflows')
    info = model_info()
    rows = []
    for variant, filename in FILES.items():
        _, workflow, report = build_candidate(variant, info)
        write_new(dest / filename, workflow)
        rows.append({'variant': variant, 'serialization': report})
    print(json.dumps({'destination': str(dest), 'templates': rows}))


if __name__ == '__main__':
    main()
