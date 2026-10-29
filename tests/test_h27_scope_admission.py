"""A rejected unpublished prototype must not enter the standard node registry."""
import asyncio

import h3_audio_t8_pkg


def test_standard_registry_preserves_mask_without_admitting_rejected_tara():
    nodes = asyncio.run(h3_audio_t8_pkg.comfy_entrypoint().get_node_list())
    ids = [node.define_schema().node_id for node in nodes]
    assert ids[774] == "MiniMaxH3AuthoredMaskSequenceEXPT8"
    assert "MiniMaxH3TaraMotionTimeEXPT8" not in ids
    assert len(ids) == len(set(ids))
