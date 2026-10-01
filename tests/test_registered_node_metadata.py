"""Published metadata must enumerate the actual native registry, not an old count."""
import asyncio
import json
from pathlib import Path

import h3_audio_t8_pkg


def test_features_node_index_is_exact_actual_registry_order_without_duplicates():
    root = Path(__file__).resolve().parents[1]
    recorded = json.loads((root/"features.json").read_bytes())["nodes"]
    actual = [cls.define_schema().node_id for cls in
              asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())]
    assert len(recorded) == len(set(recorded))
    assert recorded == actual


def test_all_catalogued_public_stage_nodes_are_in_actual_metadata():
    from h3_audio_t8_pkg.modular_sampling.catalogue import ROUTES
    root = Path(__file__).resolve().parents[1]
    recorded = set(json.loads((root/"features.json").read_bytes())["nodes"])
    for route in ROUTES:
        assert set(route.public_nodes) <= recorded, route.id
