from comfy_api.latest import io

from .local_sfx import prepare_local_sfx_candidates


class MiniMaxH3LocalSFXCandidatesT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="MiniMaxH3LocalSFXCandidatesT8",
            display_name="H3 Audio · Local SFX Candidates / 局部音效对比 (T8)",
            category="T8/MiniMax H3/Audio",
            description="Compare original, external and optional H3 PCM within one time window. "
                        "Optional voice stem stays unchanged; outside the window remains sample-exact. "
                        "target_rms_ceiling only reduces candidate level; it never boosts quiet noise. "
                        "RMS matching is not LUFS. This node does not generate audio or separate voices.",
            inputs=[io.Audio.Input("source_audio"), io.Audio.Input("external_sfx_audio"),
                    io.Float.Input("start_seconds",default=0.,min=0.,max=None,step=.01,round=.01),
                    io.Float.Input("end_seconds",default=1.,min=0.,max=None,step=.01,round=.01),
                    io.Combo.Input("selection",options=["original","external","h3"],default="original"),
                    io.Combo.Input("level_policy",options=["match_source_rms","target_rms","target_rms_ceiling"],default="match_source_rms"),
                    io.Float.Input("target_rms_dbfs",default=-24.,min=-120.,max=0.),
                    io.Float.Input("candidate_start_seconds",default=0.,min=0.,max=None,step=.01,round=.01),
                    io.Float.Input("fade_seconds",default=.01,min=0.,max=None,step=.001,round=.001),
                    io.Combo.Input("headroom_policy",options=["none","attenuate_candidate"],default="none"),
                    io.Audio.Input("protected_voice_audio",optional=True), io.Audio.Input("h3_sfx_audio",optional=True)],
            outputs=[io.Audio.Output("selected_audio"), io.Audio.Output("external_preview"),
                     io.Audio.Output("h3_preview"), io.String.Output("report_json")])

    @classmethod
    def execute(cls, source_audio, external_sfx_audio, start_seconds=0.,end_seconds=1.,
                selection="original",level_policy="match_source_rms",target_rms_dbfs=-24.,
                candidate_start_seconds=0.,fade_seconds=.01,headroom_policy="none",
                protected_voice_audio=None,h3_sfx_audio=None):
        return io.NodeOutput(*prepare_local_sfx_candidates(source_audio,external_sfx_audio,start_seconds,end_seconds,
            protected_voice_audio=protected_voice_audio,h3_sfx_audio=h3_sfx_audio,
            candidate_start_seconds=candidate_start_seconds,fade_seconds=fade_seconds,
            level_policy=level_policy,target_rms_dbfs=target_rms_dbfs,selection=selection,headroom_policy=headroom_policy))


NODES = [MiniMaxH3LocalSFXCandidatesT8]
