"""Opt-in Standard/Anime/Parity lazy delivery, including lazy FullSave roots.

Only new graphs are emitted. Detector settings and original planner outputs,
sampling, final audio, Stage identities and manual approvals remain explicit.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import build_formal_face_refine_storage_workflows as storage  # noqa: E402
from tools.audit_modular_sampling_compat import write_new  # noqa: E402
from tools.build_modular_h16_storage_workflow import Draft  # noqa: E402
from tools.modular_frontend_layout import spread_frontend_columns  # noqa: E402

DESTINATION = ROOT/'examples/workflows/66-radar-no-face-lazy'
VARIANTS = ('standard','anime','parity')
FILES = {(family,mode):f'R08_Face_{family}_{mode}_Explicit_NoFace_Lazy_EXP.json'
         for family in VARIANTS for mode in storage.MODES}
BOUND_INPUTS = ('frames','fps','detector_mode','detector_model','detector_device','confidence')
PLANNERS = {'MiniMaxH3FaceRefinePlanT8Advanced':'MiniMaxH3ObservedStandardPlanEXPT8',
            'MiniMaxH3FaceRefineParityPlanT8Advanced':'MiniMaxH3ObservedParityPlanEXPT8'}
DEPENDENT_OPERATIONS = {
    'MiniMaxH3FaceStageAuditEXPT8':'MiniMaxH3FaceDependentStageAuditEXPT8',
    'MiniMaxH3FaceParityStageAuditEXPT8':'MiniMaxH3FaceDependentParityStageAuditEXPT8',
    'MiniMaxH3FaceRefineStitchAuditT8Advanced':'MiniMaxH3FaceDependentStitchEXPT8',
    'MiniMaxH3FaceRefineParityStitchT8Advanced':'MiniMaxH3FaceDependentParityStitchEXPT8'}


def _one(draft,kind):
    found=[node for node in draft.nodes.values() if node['type']==kind]
    if len(found)!=1:
        raise ValueError('Expected one full-source '+kind)
    return found[0]


def _replace_kind(node,kind,title):
    node.update(type=kind,title=title)
    node['properties'].update({'Node name for S&R':kind})


def graph_for(family,mode):
    if (family,mode) not in FILES:
        raise ValueError('No-face reuse supports explicit Standard/Anime/Parity, not fabricated SAM observations')
    original,old_api=storage.graph_for(family,'none',mode)
    draft,api=Draft(original),deepcopy(old_api)
    source,video,delivery=(_one(draft,kind) for kind in ('GetVideoComponents','CreateVideo','SaveVideo'))
    plans=[node for node in draft.nodes.values() if node['type'] in PLANNERS]
    if len(plans)!=1:
        raise ValueError('Expected one compatible original full-source Face planner')
    plan=plans[0]
    old_request=deepcopy(api[str(plan['id'])]['inputs'])
    if old_request['frames']!=[str(source['id']),0]:
        raise ValueError('Observer must use the actual complete original RGB')
    route='parity' if family=='parity' else 'standard'
    settings={name:old_request[name] for name in ('detector_mode','detector_model','detector_device','confidence')}
    if any(isinstance(value,list) for value in settings.values()):
        raise ValueError('Dynamic detector settings require an explicitly reviewed new template')
    observe=draft.make('MiniMaxH3CompleteFaceObserveEXPT8','ONE complete scan; unknown is never no-face',
        [('source_images','IMAGE'),('route','COMBO'),('fps','FLOAT'),('source_start_frame','INT'),
         ('detector_mode','COMBO'),('detector_model','COMBO'),('detector_device','COMBO'),('confidence','FLOAT')],
        [('observation','T8_COMPLETE_FACE_OBSERVATION'),('report_json','STRING'),('observation_sha256','STRING')],
        [route,24.,0,*settings.values()],(650,-650))
    decision=draft.make('MiniMaxH3NoFaceDecisionEXPT8','Manual no-face confirmation: paste this source OBS SHA independently',
        [('observation','T8_COMPLETE_FACE_OBSERVATION'),('confirm_no_face','BOOLEAN'),('confirmation_sha256','STRING')],
        [('face_branch','BOOLEAN'),('report_json','STRING')],[False,''],(1250,-650))
    draft.connect((source['id'],0),observe,'source_images','IMAGE')
    # Use the same original FPS edge where present. Parity had a literal24;
    # retain that exact original value instead of silently changing its clock.
    fps=old_request['fps']
    if isinstance(fps,list):
        draft.connect((int(fps[0]),fps[1]),observe,'fps','FLOAT')
    elif fps!=24.:
        raise ValueError('No implicit retiming')
    draft.connect((observe['id'],0),decision,'observation','T8_COMPLETE_FACE_OBSERVATION')
    for name in BOUND_INPUTS:
        if any(item['name']==name for item in plan['inputs']):
            draft.disconnect(plan,name)
    plan['inputs']=[{'name':'observation','type':'T8_COMPLETE_FACE_OBSERVATION','link':None}]
    # Both original native planners have exactly five bound widget values,
    # followed by their unchanged independent crop/tracking settings.
    if plan['widgets_values'][:5]!=[24.,*settings.values()]:
        raise ValueError('Original detector widget layout changed; update explicit adapter')
    plan['widgets_values']=plan['widgets_values'][5:]
    _replace_kind(plan,PLANNERS[plan['type']],'Original unchanged '+route+' Plan from complete ONCE observation')
    draft.connect((observe['id'],0),plan,'observation','T8_COMPLETE_FACE_OBSERVATION')
    api[str(observe['id'])]={'class_type':observe['type'],'inputs':{
        'source_images':[str(source['id']),0],'route':route,'fps':fps,'source_start_frame':0,**settings}}
    api[str(decision['id'])]={'class_type':decision['type'],'inputs':{
        'observation':[str(observe['id']),0],'confirm_no_face':False,'confirmation_sha256':''}}
    api[str(plan['id'])]={'class_type':plan['type'],'inputs':{
        'observation':[str(observe['id']),0],**{k:v for k,v in old_request.items() if k not in BOUND_INPUTS}}}
    roots=[delivery['id']]
    for node in list(draft.nodes.values()):
        if node['type']!='MiniMaxH3StageSaveEXPT8':
            continue
        stage_edge=storage._source(draft,node,'stage_result')
        draft.disconnect(node,'stage_result')
        node['inputs'].insert(0,{'name':'face_branch','type':'BOOLEAN','link':None,'widget':{'name':'face_branch'}})
        node['widgets_values'].insert(0,False)
        _replace_kind(node,'MiniMaxH3FaceBranchStageSaveEXPT8','Freeze actual Stage ONLY if Face branch is selected')
        draft.connect(stage_edge,node,'stage_result','T8_STAGE_RESULT')
        draft.connect((decision['id'],0),node,'face_branch','BOOLEAN')
        api[str(node['id'])]['class_type']=node['type']
        api[str(node['id'])]['inputs']['face_branch']=[str(decision['id']),0]
        roots.append(node['id'])
    for name,slot,dtype in (('images',0,'IMAGE'),('audio',1,'AUDIO')):
        candidate=storage._source(draft,video,name)
        switch=draft.make('ComfySwitchNode','TRUE original Face / FALSE exact original '+dtype+' (Core LAZY)',
            [('switch','BOOLEAN'),('on_false',dtype),('on_true',dtype)],[('output',dtype)],
            [False],(6100,slot*450))
        switch['properties']['cnr_id']='comfy-core'
        draft.connect((decision['id'],0),switch,'switch','BOOLEAN')
        draft.connect((source['id'],slot),switch,'on_false',dtype)
        draft.connect(candidate,switch,'on_true',dtype)
        draft.disconnect(video,name)
        draft.connect((switch['id'],0),video,name,dtype)
        api[str(switch['id'])]={'class_type':'ComfySwitchNode','inputs':{
            'switch':[str(decision['id']),0],'on_false':[str(source['id']),slot],
            'on_true':[str(candidate[0]),candidate[1]]}}
        api[str(video['id'])]['inputs'][name]=[str(switch['id']),0]
    # These light output roots expose the SHA/blocked reason BEFORE a user
    # elects a negative bypass. They never depend on any Face planner/model.
    for parent,slot,title in ((observe,1,'Inspect complete OBS status + SHA BEFORE confirming'),
                              (decision,1,'Inspect unknown/unconfirmed delivery blocker')):
        preview=draft.make('PreviewAny',title,[('source','*')],[],[],(1800,-650+slot*250))
        preview['properties']['cnr_id']='comfy-core'
        draft.connect((parent['id'],slot),preview,'source','STRING')
        api[str(preview['id'])]={'class_type':'PreviewAny','inputs':{'source':[str(parent['id']),slot]}}
        roots.append(preview['id'])
    prefix=f'MiniMaxH3/R08_NoFaceLazy/{family}_{mode}'
    delivery['widgets_values'][0]=prefix
    api[str(delivery['id'])]['inputs']['filename_prefix']=prefix
    frontend=draft.prune(roots)
    # Audit/Stitch are output roots in legacy schemas too. A final Switch
    # alone cannot make them lazy. Preserve their exact operation via explicit
    # new dependent-only adapters; never change old node root semantics.
    for node in frontend['nodes']:
        if node['type'] in DEPENDENT_OPERATIONS:
            _replace_kind(node,DEPENDENT_OPERATIONS[node['type']],
                          'Selected Face dependency ONLY · '+node['title'])
            api[str(node['id'])]['class_type']=node['type']
    if set(api)!={str(node['id']) for node in frontend['nodes']}:
        raise ValueError('Lazy template pruned a required original dependency')
    frontend.setdefault('extra',{})['t8_no_face_lazy']={
        'schema':'t8.no-face-lazy-workflow.v1','family':family,'storage':mode,
        'confirmation':'manual_exact_observation_SHA_never_automatically_wired',
        'no_face_is_human_truth':False,'false_audio_original_object':True,
        'full_save_output_roots_lazy':True,'original_sampling_and_approvals_unchanged':True,
        'status':'explicit_opt_in_requires_input_confirmation_and_final_quality_review'}
    spread_frontend_columns(frontend)
    return frontend,api


def main():
    for key,filename in FILES.items():
        frontend,api=graph_for(*key)
        path=DESTINATION/filename
        if path.exists():
            if json.loads(path.read_text(encoding='utf8'))!=frontend:
                raise ValueError('Existing new graph differs; preserve and review separately')
        else:
            write_new(path,frontend)
        print(json.dumps({'file':filename,'nodes':len(frontend['nodes']),'api_nodes':len(api)}))


if __name__=='__main__':
    main()
