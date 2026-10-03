"""Four additive graphs, explicit placeholders, genuine Core schema/edges."""
import asyncio
from copy import deepcopy
import json

import pytest
from tools import build_external_continuation_workflows as examples


@pytest.fixture(scope='module')
def current_info():
    return examples.shared.load_live_info()


@pytest.mark.parametrize('key',tuple(examples.FILES))
def test_saved_graphs_match_current_typed_builder_and_core_validation(key,current_info):
    graph,workflow,audit=examples.build(*key,current_info)
    saved_path=examples.DESTINATION/examples.FILES[key]
    saved=json.loads(saved_path.read_text(encoding='utf8'))
    assert {k:v for k,v in saved.items() if k!='id'}=={k:v for k,v in workflow.items() if k!='id'}
    assert audit['nodes']==len(graph)
    text=saved_path.read_text(encoding='utf8')
    assert all(private not in text for private in ('artifacts/development','7e026528','e8fdee4f','F:\\','G:\\'))
    for _,source,slot,target,target_slot,_type in workflow['links']:
        by_id={node['id']:node for node in workflow['nodes']}
        name=by_id[target]['inputs'][target_slot]['name']
        api_target=str(next(index for index,node in enumerate(graph,1) if index==target))
        api_source=list(graph)[source-1]
        api_target=list(graph)[int(api_target)-1]
        assert graph[api_target]['inputs'][name]==[api_source,slot]
    import execution
    valid,error,*_rest=asyncio.run(execution.validate_prompt('external-example',deepcopy(graph),None))
    assert valid is True and error is None


@pytest.mark.parametrize('key',tuple(examples.FILES))
def test_exact_external_effect_and_explicit_cold_roots_without_sampling(key):
    route,mode=key
    graph=examples.graph_for(*key)
    kinds={node['class_type'] for node in graph.values()}
    if mode=='Cold_Delivery':
        assert not kinds & {'UNETLoader','CLIPLoader','MiniMaxH3StageSamplerEXPT8',
            'MiniMaxH3DualClockSamplerT8','MiniMaxH3ExternalContextEncodeEXPT8'}
        assert graph['50']['inputs']['artifact_path'].startswith('REPLACE_')
        assert graph['50']['inputs']['artifact_sha256'].startswith('REPLACE_')
    else:
        assert graph['6']['inputs']['project_id']==graph['6']['inputs']['shot_id']==graph['6']['inputs']['take_id']==''
        assert graph['29']['inputs']['mode']=='report_only'
        assert graph['30']['inputs']['effect_scope']==['28',1]
        assert graph['12']['inputs']['model']==['30',0]
        assert graph['28']['inputs']['av_latent']==graph['13']['inputs']['latent_image']==['8',2]
        assert graph['9']['inputs']['steps']==4
        assert graph['8']['class_type']==('MiniMaxH3ExternalRelayConditioningEXPT8' if route=='Relay_EAV'
                                        else 'MiniMaxH3ExternalContinuationConditioningEXPT8')
    assert graph['15']['inputs']['start_seconds']==22/24 and graph['15']['inputs']['duration_seconds']==102/24


def test_unknown_example_variant_rejects():
    with pytest.raises(ValueError):
        examples.graph_for('unimplemented','Full_Save')
