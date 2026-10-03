"""Two additive cast-solo nodes; old MV schemas and defaults remain intact."""
from comfy_api.latest import io

from .mv_cast_solo import TYPE, build_plan, render
from .mv_lipsync_advanced import MV_SCENE_PLAN_TYPE
from .nodes_mv_lipsync_advanced import _preview_video
from .sampling import SAMPLER_OPTIONS, SCHEDULER_OPTIONS

CATEGORY = 'T8/MiniMax H3/MV & Lip Sync/Cast Solo EXP'
CastIO = io.Custom(TYPE)


class MiniMaxH3MVCastSoloPlanEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3MVCastSoloPlanEXPT8', display_name='H3 MV · A/B独唱分镜＋显式歌词Cue',
                         category=CATEGORY, is_experimental=True,
                         description='Assign each existing VocalLock scene explicitly to A or B. One solo performer per scene, never a guessed duet. Uses original scene frame clock and one final original-song mux.',
                         inputs=[io.Custom(MV_SCENE_PLAN_TYPE).Input('scene_plan'),
                                 io.String.Input('performer_a_description', default='the performer in reference A', multiline=True),
                                 io.String.Input('performer_b_description', default='the performer in reference B', multiline=True),
                                 io.String.Input('assignments_json', default='[{"scene_index":0,"performer_id":"A","exact_vocal_text":""},{"scene_index":1,"performer_id":"B","exact_vocal_text":""}]', multiline=True,
                                                 tooltip='逐场景明确零基scene_index、A/B和原文。须覆盖scene_count；不自动循环角色、不猜歌词。'),
                                 io.String.Input('global_creative_prompt', default='A restrained studio music performance.', multiline=True),
                                 io.String.Input('visual_style', default='cinematic realism, natural light', multiline=True),
                                 io.String.Input('scene_directions_json', default='', multiline=True),
                                 io.Combo.Input('vocal_content_type', options=['singing', 'spoken_dialogue'], default='singing'),
                                 io.String.Input('vocal_language', default='English')],
                         outputs=[CastIO.Output('cast_solo_plan'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, **inputs):
        return io.NodeOutput(*build_plan(**inputs))


class MiniMaxH3MVCastSoloRendererEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3MVCastSoloRendererEXPT8', display_name='H3 MV · A/B逐镜独立参考生成',
                         category=CATEGORY, is_experimental=True, is_output_node=True,
                         description='Uses the actually selected A/B IMAGE for each independent Ref2VA scene, no previous performer tail. Separate resume identity; original full song muxed once. Mechanical internal assembly is not human quality approval.',
                         inputs=[io.Model.Input('model'), io.Clip.Input('clip'), io.Vae.Input('video_vae'), io.Vae.Input('audio_vae'),
                                 io.Image.Input('reference_a'), io.Image.Input('reference_b'),
                                 io.Audio.Input('full_song'), io.Audio.Input('vocal_lock_audio'), CastIO.Input('cast_solo_plan'),
                                 io.String.Input('chain_id', default='my_h3_mv_cast_solo_v4'),
                                 io.Int.Input('width', default=736, min=32, max=16384, step=32),
                                 io.Int.Input('height', default=416, min=32, max=16384, step=32),
                                 io.Int.Input('base_seed', default=123456789, min=0, max=0xFFFFFFFFFFFFFFFF),
                                 io.Int.Input('steps', default=4, min=1, max=1000),
                                 io.Float.Input('shift_video', default=12., min=.01, max=100., step=.01),
                                 io.Float.Input('shift_audio', default=3., min=.01, max=100., step=.01),
                                 io.Combo.Input('sampler_name', options=SAMPLER_OPTIONS, default='euler'),
                                 io.Combo.Input('scheduler', options=SCHEDULER_OPTIONS, default='simple'),
                                 io.Boolean.Input('resume_existing', default=True),
                                 io.String.Input('filename_prefix', default='H3_Local_MV_CastSolo_V4'),
                                 io.Combo.Input('bit_depth', options=[8, 10], default=8),
                                 io.Int.Input('crf', default=18, min=0, max=51),
                                 io.String.Input('model_id', default='minimax_h3_ref2va+official_ref2v_turbo4_v0.1',
                                                 tooltip='仅审计标签，不是模型字节资格。需实际安装对应模型。')],
                         outputs=[io.Video.Output('video'), io.String.Output('video_path'), io.String.Output('manifest_path'),
                                  io.Int.Output('completed_scenes'), io.String.Output('status'), io.String.Output('report_json')])

    @classmethod
    def execute(cls, **inputs):
        result = render(**inputs)
        video, preview = _preview_video(result[0])
        return io.NodeOutput(video, *result, ui=preview)


NODES = [MiniMaxH3MVCastSoloPlanEXPT8, MiniMaxH3MVCastSoloRendererEXPT8]
