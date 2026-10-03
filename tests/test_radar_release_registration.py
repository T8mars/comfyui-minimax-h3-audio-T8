"""Append-only RADAR registration and published feature-list identity."""
import asyncio
import json
from pathlib import Path

import h3_audio_t8_pkg

ROOT = Path(__file__).resolve().parents[1]


def test_release_registry_preserves_entire_1881_prefix_and_metadata():
    classes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    ids = [cls.define_schema().node_id for cls in classes]
    original = json.loads((ROOT / 'tests/fixtures/radar_pre1881_registry_ids.json').read_text(encoding='utf8'))
    assert len(original) == 582
    assert len(ids) == len(set(ids)) == 619
    assert ids[:582] == original
    assert ids[582:584] == ['MiniMaxH3FunUnion2LoaderEXPT8', 'MiniMaxH3FunUnion2ApplyEXPT8']
    assert ids[-1] == 'MiniMaxH3HyperFlowCurveBaseLoaderEXPT8'
    assert json.loads((ROOT / 'features.json').read_text(encoding='utf8'))['nodes'] == ids
