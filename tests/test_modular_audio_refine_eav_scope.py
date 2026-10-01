"""Audio Refine keeps external EAV on the original video-generating pass."""

import json
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg import enhance_a_video_advanced as eav
from tools.build_modular_audio_refine_workflows import split_frontend

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'examples/workflows/18-audio-refine/2026-08-29_H3_Audio_Refine_EAV_Turbo8_Advanced_EXP.json'


def test_separated_audio_refine_preserves_external_eav_on_first_pass_only():
    source = json.loads(SOURCE.read_text(encoding='utf-8'))
    draft = split_frontend(source, abstain_safe=True)
    nodes = {node['id']: node for node in draft['nodes']}
    links = draft['links']
    eav_nodes = [node for node in nodes.values()
                 if node['type'] == 'MiniMaxH3EnhanceAVideoT8Advanced']
    samplers = [node for node in nodes.values() if node['type'] == 'SamplerCustomAdvanced']
    assert len(eav_nodes) == 1 and len(samplers) == 2
    assert nodes[10]['type'] == nodes[23]['type'] == 'SamplerCustomAdvanced'
    assert any(link[1] == eav_nodes[0]['id'] and link[3] == 8 for link in links)
    assert any(link[1] == 8 and link[3] == 10 for link in links)
    assert not any(link[1] == eav_nodes[0]['id'] and link[3] == 23 for link in links)
    assert any(node['type'] == 'MiniMaxH3AudioRefineCompatStageBindEXPT8'
               for node in nodes.values())
    assert any(node['type'] == 'MiniMaxH3EnhanceAVideoAuditT8Advanced'
               for node in nodes.values())


def test_legacy_eav_refuses_audio_only_four_step_tail_and_unowned_masks():
    tail = torch.tensor([.9231, .8780, .8000, .6316, 0.0])
    with pytest.raises(ValueError, match='expects 9 sigma'):
        eav._validate_sigma_schedule(tail, 'turbo8_alpha8')
    with pytest.raises(RuntimeError, match='rejects video/audio denoise masks'):
        eav._runtime_route(
            x=None, timestep=None, context=None, payload={},
            denoise_mask=torch.zeros(1), audio_denoise_mask=torch.ones(1),
            start_progress=0., end_progress=1.)
