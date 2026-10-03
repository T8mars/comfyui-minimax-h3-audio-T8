"""Append-only complete detection/explicit decision/original-plan adapters."""
from comfy_api.latest import io
from . import face_observations as face
from .face_refine_advanced import local_face_detector_options
from .nodes_face_refine_advanced import MiniMaxH3FaceRefinePlanT8Advanced,MiniMaxH3FaceRefineStitchAuditT8Advanced
from .nodes_face_refine_parity_advanced import MiniMaxH3FaceRefineParityPlanT8Advanced,MiniMaxH3FaceRefineParityStitchT8Advanced
from .modular_sampling.nodes import MiniMaxH3StageSaveEXPT8
from .modular_sampling.face_nodes import MiniMaxH3FaceStageAuditEXPT8,MiniMaxH3FaceParityStageAuditEXPT8

CATEGORY = 'T8/MiniMax H3/Face Refine Experimental/Lazy'


class MiniMaxH3CompleteFaceObserveEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,display_name='H3 Face · Complete Source Observe ONCE (T8 EXP)',
            category=CATEGORY,is_experimental=True,
            description='All RGB frames through the existing local detector once. Exact source/model/code/clock '
                'binding. Missing model, errors or partial coverage are UNKNOWN, never no-face. Select '
                'Standard or Parity to preserve its original colour input. No segmentation/identity truth '
                'or sampling. Use matching Original Plan from Observation on the true lazy branch.',
            inputs=[io.Image.Input('source_images'),io.Combo.Input('route',options=list(face.ROUTES),default='standard'),
                    io.Float.Input('fps',default=24.,min=24.,max=24.),
                    io.Int.Input('source_start_frame',default=0,min=0,max=5_000_000),
                    io.Combo.Input('detector_mode',options=list(face.MODES),default='local_opencv_yunet'),
                    io.Combo.Input('detector_model',options=local_face_detector_options()),
                    io.Combo.Input('detector_device',options=['cpu','cuda_auto'],default='cpu'),
                    io.Float.Input('confidence',default=.35,min=.001,max=1.,step=.01)],
            outputs=[io.Custom(face.OBSERVATION_TYPE).Output('observation'),io.String.Output('report_json'),
                     io.String.Output('observation_sha256')])

    @classmethod
    def execute(cls, source_images, **request):
        value = face.observe(source_images,**request)
        return io.NodeOutput(value,value.contract_json,value.contract_sha256)

    @classmethod
    def fingerprint_inputs(cls, **_inputs):
        return float('nan')


class MiniMaxH3NoFaceDecisionEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,display_name='H3 Face · Explicit No-Face Decision (T8 EXP)',
            category=CATEGORY,is_experimental=True,
            description='Connect Boolean to Core If/Else Switch: true=original Face branch, false=original '
                'full RGB/AUDIO. UNKNOWN/negative-unconfirmed block delivery, not false. No-face requires '
                'manual confirmation and exact current observation SHA; not human truth or Stage completion.',
            inputs=[io.Custom(face.OBSERVATION_TYPE).Input('observation'),
                    io.Boolean.Input('confirm_no_face',default=False),
                    io.String.Input('confirmation_sha256',default='')],
            outputs=[io.Boolean.Output('face_branch'),io.String.Output('report_json')])

    @classmethod
    def execute(cls, observation, **request):
        return io.NodeOutput(*face.decision(observation,**request))

    @classmethod
    def fingerprint_inputs(cls, **_inputs):
        return float('nan')


def _schema(cls, original, route):
    inherited = original.define_schema()
    removed = {'frames','fps','detector_mode','detector_model','detector_device','confidence'}
    return io.Schema(node_id=cls.__name__,display_name='H3 Face · Original '+route.title()+' Plan from ONCE Observe (T8 EXP)',
        category=CATEGORY,is_experimental=True,
        description='Original unmodified planner/code/math/schema outputs; read the bound detector observations '
            'once in a call-local namespace, no module-global mutation or second YOLO. Use ONLY on selected '
            'face-found lazy branch; no-face/unknown never manufactures a plan. Original crop/manual ROI '
            'fallback parameters remain independently editable.',
        inputs=[io.Custom(face.OBSERVATION_TYPE).Input('observation'),
                *[item for item in inherited.inputs if item.id not in removed]],outputs=inherited.outputs)


class MiniMaxH3ObservedStandardPlanEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls,MiniMaxH3FaceRefinePlanT8Advanced,'standard')

    @classmethod
    def execute(cls, observation, **options):
        return io.NodeOutput(*face.plan_from_observation(observation,'standard',**options))

    @classmethod
    def fingerprint_inputs(cls, **_inputs):
        return float('nan')


class MiniMaxH3ObservedParityPlanEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(cls,MiniMaxH3FaceRefineParityPlanT8Advanced,'parity')

    @classmethod
    def execute(cls, observation, **options):
        return io.NodeOutput(*face.plan_from_observation(observation,'parity',**options))

    @classmethod
    def fingerprint_inputs(cls, **_inputs):
        return float('nan')


class MiniMaxH3FaceBranchStageSaveEXPT8(io.ComfyNode):
    """An output root must itself be lazy, not just the final image switch."""
    @classmethod
    def define_schema(cls):
        original = MiniMaxH3StageSaveEXPT8.define_schema()
        return io.Schema(node_id=cls.__name__,
            display_name='H3 Face · Save Stage ONLY on selected Face branch (T8 EXP)',
            category=CATEGORY,is_experimental=True,is_output_node=True,
            description='Connect the explicit Face Decision. True lazily requests a real completed '
                'StageResult and delegates to the unchanged original immutable saver. False does '
                'not request the sampler, write an artifact, invent a Stage or return a restore '
                'path/SHA. This gate is required for FullSave output roots as well as RGB/AUDIO switches.',
            inputs=[io.Boolean.Input('face_branch'),
                    io.Custom('T8_STAGE_RESULT').Input('stage_result',lazy=True),
                    *[item for item in original.inputs if item.id!='stage_result']],
            outputs=original.outputs)

    @classmethod
    def check_lazy_status(cls, face_branch, stage_result=None, prefix='stage'):
        if type(face_branch) is not bool:
            raise ValueError('Explicit Boolean Face Decision required')
        return ['stage_result'] if face_branch and stage_result is None else []

    @classmethod
    def execute(cls, face_branch, stage_result=None, prefix='stage'):
        from comfy_execution.graph import ExecutionBlocker
        if type(face_branch) is not bool:
            raise ValueError('Explicit Boolean Face Decision required')
        if face_branch:
            if stage_result is None:
                raise ValueError('Selected Face branch requires an actual completed StageResult')
            return MiniMaxH3StageSaveEXPT8.execute(stage_result,prefix)
        report = face.canonical({'schema':'t8.face-branch-stage-save.v1',
            'status':'unselected_face_branch_no_stage_saved','stage_saved':False,
            'artifact_path':None,'artifact_sha256':None,'automatic_accept':False,
            'sampling_requested_by_this_output':False})
        return io.NodeOutput(*(ExecutionBlocker(None) for _ in range(4)),report,ui={'text':[report]})

    @classmethod
    def fingerprint_inputs(cls, **_inputs):
        return float('nan')


def _dependent_schema(cls,original):
    # Every original define_schema creates a fresh instance. Change ONLY this
    # new node's root flag and identity; keep all original input/output fields.
    schema=original.define_schema()
    schema.node_id=cls.__name__
    schema.display_name='H3 Face · Selected dependency ONLY: '+schema.display_name
    schema.category=CATEGORY
    schema.is_output_node=False
    schema.description += (' This independent adapter calls the unchanged operation ONLY as a selected '
        'dependency, not an eager output root. Keep its audit in the true lazy delivery branch.')
    return schema


class MiniMaxH3FaceDependentStageAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _dependent_schema(cls,MiniMaxH3FaceStageAuditEXPT8)
    @classmethod
    def execute(cls,**inputs):
        return MiniMaxH3FaceStageAuditEXPT8.execute(**inputs)


class MiniMaxH3FaceDependentParityStageAuditEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _dependent_schema(cls,MiniMaxH3FaceParityStageAuditEXPT8)
    @classmethod
    def execute(cls,**inputs):
        return MiniMaxH3FaceParityStageAuditEXPT8.execute(**inputs)


class MiniMaxH3FaceDependentStitchEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _dependent_schema(cls,MiniMaxH3FaceRefineStitchAuditT8Advanced)
    @classmethod
    def execute(cls,**inputs):
        return MiniMaxH3FaceRefineStitchAuditT8Advanced.execute(**inputs)


class MiniMaxH3FaceDependentParityStitchEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _dependent_schema(cls,MiniMaxH3FaceRefineParityStitchT8Advanced)
    @classmethod
    def execute(cls,**inputs):
        return MiniMaxH3FaceRefineParityStitchT8Advanced.execute(**inputs)


NODES = [MiniMaxH3CompleteFaceObserveEXPT8,MiniMaxH3NoFaceDecisionEXPT8,
         MiniMaxH3ObservedStandardPlanEXPT8,MiniMaxH3ObservedParityPlanEXPT8,
         MiniMaxH3FaceBranchStageSaveEXPT8,MiniMaxH3FaceDependentStageAuditEXPT8,
         MiniMaxH3FaceDependentParityStageAuditEXPT8,MiniMaxH3FaceDependentStitchEXPT8,
         MiniMaxH3FaceDependentParityStitchEXPT8]
