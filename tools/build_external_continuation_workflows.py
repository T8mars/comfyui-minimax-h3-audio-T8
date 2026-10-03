"""Four opt-in external motion FullSave/ColdDelivery examples; no private assets."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
from tools import build_modular_fast_h3_v2_workflow as shared  # noqa: E402
from tools.api_to_frontend_workflow import convert  # noqa: E402
from tools.audit_progressive_workflows import audit_candidate  # noqa: E402
from tools.audit_modular_sampling_compat import write_new  # noqa: E402
from tools.modular_frontend_layout import spread_frontend_columns  # noqa: E402

DESTINATION = ROOT/'examples/workflows/67-radar-external-continuation'
FILES = {(route,mode): f'R06_External_{route}_{mode}_EXP.json'
         for route in ('EAV_Only','Relay_EAV') for mode in ('Full_Save','Cold_Delivery')}


def node(kind, **inputs):
    return {'class_type':kind,'inputs':inputs}


def graph_for(route, mode):
    if (route,mode) not in FILES:
        raise ValueError('Choose an explicit external EAV-only/Relay-EAV full-save or cold-delivery example')
    graph = {
        '4':node('VAELoader',vae_name='minimax_h3_video_vae_fp16.safetensors'),
        '5':node('VAELoader',vae_name='minimax_h3_audio_vae_fp32.safetensors'),
        '14':node('MiniMaxH3AVDecodeT8',av_latent=['50' if mode=='Cold_Delivery' else '31',0],
                  video_vae=['4',0],audio_vae=['5',0]),
        '15':node('MiniMaxH3OutputTrimT8',frames=['14',0],audio=['14',1],
                  start_seconds=22/24,duration_seconds=102/24,fps=24.),
        '16':node('MiniMaxH3SafeAVSaveT8Advanced',images=['15',0],audio=['15',1],
                  filename_prefix='MiniMaxH3/External/'+route+'/'+mode,crf=18),
        '21':node('PreviewAny',source=['15',2])}
    if mode=='Cold_Delivery':
        return {**graph,'50':node('MiniMaxH3StageLoadEXPT8',
            artifact_path='REPLACE_WITH_ACTUAL_COMPLETED_EXTERNAL_STAGE/manifest.json',
            artifact_sha256='REPLACE_WITH_ACTUAL_STAGE_SHA256',expected_stage='native_high'),
            '22':node('PreviewAny',source=['50',4])}
    graph.update({
        '1':node('UNETLoader',unet_name='minimax_h3_fl2va_int8_convrot.safetensors',weight_dtype='default'),
        '2':node('MiniMaxH3LoRACompatibilityLoaderT8Advanced',model=['1',0],
                 lora_name='minimax_h3_turbo_v4_step600_ema_comfyui_B.safetensors',strength_model=1.),
        '3':node('CLIPLoader',clip_name='qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',type='minimax',device='default'),
        '6':node('MiniMaxH3ExternalContinuationSourceEXPT8',project_id='',shot_id='',take_id='',
                 context_frames=22,audio_policy='video_only'),
        '7':node('MiniMaxH3ExternalContextEncodeEXPT8',external_source=['6',0],video_vae=['4',0],width=448,height=256),
        '8':node('MiniMaxH3ExternalContinuationConditioningEXPT8',external_context=['7',0],
            model=['2',0],clip=['3',0],video_vae=['4',0],audio_vae=['5',0],
            prompt='Continue the same subject and place with quiet natural motion, stable camera and lighting. Natural ambience, no cut or captions.',
            length=124,task_type='auto',audio_mode='native',audio_denoise_strength=.35,
            add_source_as_reference=True,prompt_primary_audio_ordinal=0,strict_prompt_tags=True,
            ref_image_size='match',reference_video_policy='official_2_to_15s',first_frame_reuse='segment0_only',
            persistent_identity_strategy='single_reference',persistent_identity_interval=1),
        '9':node('MiniMaxH3DualClockSamplerT8',model=['28',0],av_latent=['8',2],steps=4,shift_video=12.,
            shift_audio=3.,sampler_name='dual_clock_euler',scheduler='native_flow'),
        '10':node('MiniMaxH3NativeStageBindEXPT8',model=['9',0],sampler=['9',1],sigmas=['9',2],av_latent=['8',2],stage='native_high'),
        '11':node('RandomNoise',noise_seed=2609220606),
        '12':node('BasicGuider',model=['30',0],conditioning=['8',1]),
        '13':node('MiniMaxH3StageSamplerEXPT8',noise=['11',0],guider=['12',0],sampler=['10',1],
            sigmas=['10',2],latent_image=['8',2],stage_context=['10',3]),
        '28':node('MiniMaxH3ExternalMotionEffectsBindEXPT8',external_context=['7',0],model=['8',0],positive=['8',1],av_latent=['8',2]),
        '29':node('MiniMaxH3StageEAVConfigEXPT8',mode='report_only',tau=4.,start_video_progress=.15,
            end_video_progress=.9,max_workspace_mib=32,g_hard_limit=1.5),
        '30':node('MiniMaxH3ExternalStageEAVApplyEXPT8',model=['10',0],sigmas=['10',2],av_latent=['8',2],
            stage_context=['10',3],eav_config=['29',0],effect_scope=['28',1]),
        '31':node('MiniMaxH3StageEAVAuditEXPT8',av_latent=['13',0],runtime=['30',1]),
        '32':node('PreviewAny',source=['31',1]),
        '33':node('PreviewAny',source=['8',6]),
        '50':node('MiniMaxH3StageSaveEXPT8',stage_result=['13',2],prefix='External/'+route)})
    if route=='Relay_EAV':
        prompt = graph['8']['inputs'].pop('prompt')
        graph['8']['class_type']='MiniMaxH3ExternalRelayConditioningEXPT8'
        graph['8']['inputs'].update(projected_relay=['27',0],execution_mode='report_only',query_chunk_rows=256)
        graph.update({'26':node('MiniMaxH3PromptRelayPlanT8Advanced',global_prompt=prompt,
            local_prompts='The subject continues a calm natural movement.\nThe subject gently changes expression without a cut.',
            length=124,timing_mode='auto_equal',time_ranges='',math_profile='paper_v1',epsilon=.1,
            allow_gaps=False,allow_overlaps=False),
            '27':node('MiniMaxH3ExternalRelayWindowEXPT8',external_context=['7',0],global_plan=['26',0],
                length=124,generated_start_frame=22)})
    return graph


def build(route, mode, info):
    graph, selected = shared.selected_frontend_schema(graph_for(route,mode),info)
    workflow = convert(graph,selected,f'External RGB/PCM {route} {mode} / EXP')
    note_id=workflow['last_node_id']+1
    workflow['nodes'].append({'id':note_id,'type':'MarkdownNote','title':'先读 / External motion boundaries',
        'pos':[0,-650],'size':[1250,540],'flags':{},'order':len(graph),'mode':0,'inputs':[],'outputs':[],
        'properties':{},'widgets_values':['外片续拍RGB/PCM重编码EXP，不是原生采样祖先或原始latent恢复。'
        'Full先填写保存工程中已人工采用外片的project/shot/take ID；LOW/HIGH独立context/条件/模型。'
        '此图是单个完整4步阶段，不冒称双采。Relay与EAV在外部、默认report_only，apply须显式选择并检查真实audit。'
        '窗口generated_start_frame为独立Relay时钟，不自动使用外片结束或工程时间。'
        '默认124渲染/22context，显式裁掉前22→102新帧；改变context/length时同步trim。'
        'Cold只交付明确保存的固定Stage，必须填写实际manifest路径/SHA，不重采/不自动判断新设置相同。'
        '模型文件须存在。没有自动append/adopt/QualityGate；旧图不动，源码机械与媒体人审分别。']})
    workflow['last_node_id']=note_id
    workflow.setdefault('extra',{})['t8_external_example']={'schema':'t8.external-continuation.example.v1',
        'route':route,'mode':mode,'automatic_accept':False,'status':'opt_in_not_human_quality_accepted'}
    spread_frontend_columns(workflow)
    return graph,workflow,audit_candidate(graph,workflow,selected)


def main():
    info=shared.load_live_info()
    for (route,mode),filename in FILES.items():
        _graph,workflow,_audit=build(route,mode,info)
        write_new(DESTINATION/filename,workflow)


if __name__=='__main__':
    main()
