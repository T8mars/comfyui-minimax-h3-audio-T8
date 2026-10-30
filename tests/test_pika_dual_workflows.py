"""Keep accepted browser bytes, explicit recipes and evidence bindings intact."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).parents[1]
DIRECTORY=ROOT/'examples/workflows/95-pika-dual'
INDEX=json.loads((DIRECTORY/'index.json').read_text('utf8'))


def test_eight_selected_workflows_have_two_independent_accepted_media_bindings():
    assert len(INDEX['workflows'])==8 and len(INDEX['reviews'])==2
    assert INDEX['preserved_registered_prefix']==779 and INDEX['registered_nodes']==780
    accepted=set()
    for review in INDEX['reviews']:
        assert review['picture']==review['sound']=='pass'
        assert len(review['binding'])==len(review['human_feedback_sha256'])==64
        for media in review['media'].values():
            accepted.update(media)
    for row in INDEX['workflows']:
        assert row['picture']==row['sound']=='pass'
        assert set(row['media_sha256'])<=accepted
        assert row['browser_roundtrip_parameters_and_links_preserved']


@pytest.mark.parametrize('row',INDEX['workflows'],ids=lambda row:row['case_id'])
def test_native_browser_bytes_links_and_common_base_are_preserved(row):
    path=ROOT/row['path']
    assert path.parent==DIRECTORY
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['workflow_sha256']
    graph=json.loads(path.read_text('utf8'))
    nodes={n['id']:n for n in graph['nodes']}
    assert len(nodes)==len(graph['nodes'])
    assert sum(n['type']=='UNETLoader' for n in nodes.values())==1
    for link,origin,slot,target,pin,kind in graph['links']:
        assert nodes[target]['inputs'][pin]['link']==link
        assert link in nodes[origin]['outputs'][slot]['links']
    if row['case_id']=='fl2va_regular_6plus3':
        assert row['NFE']==9 and row['dimensions']==[1024,576]
        assert not row['official_Pika_hires_recipe']
        assert not any('Regenerate' in n['type'] for n in nodes.values())
        splits=[n for n in nodes.values() if n['type']=='SplitSigmas']
        assert len(splits)==1 and splits[0]['widgets_values']==[3]
    elif row['case_id']=='fl2va_6plus3_2k':
        assert row['NFE']==9 and row['dimensions']==[2560,1440]
        assert any(n['type']=='MiniMaxH3OpenRegenerateRecipeEXPT8' for n in nodes.values())
    else:
        assert row['NFE']==6
