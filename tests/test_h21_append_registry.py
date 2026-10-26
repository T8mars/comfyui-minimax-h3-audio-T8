"""The live extension retains its complete prefix and finalizes actual schemas."""

import asyncio
import json
from pathlib import Path

import h3_audio_t8_pkg as package
from h3_audio_t8_pkg import nodes_reference_temporal_pack as packed
from h3_audio_t8_pkg import nodes_int8_row_chunk as quantized
from h3_audio_t8_pkg import nodes_ptd_advanced as ptd
from h3_audio_t8_pkg import nodes_reference_step_cache as cached
from h3_audio_t8_pkg.freevideo_h21 import nodes as freevideo
from h3_audio_t8_pkg import nodes_mc_seam_experimental as seam
from h3_audio_t8_pkg import nodes_ptd_continuous as continuous
from h3_audio_t8_pkg import nodes_av_loop as periodic
from h3_audio_t8_pkg import nodes_solar_structural as solar


ROOT = Path(__file__).resolve().parents[1]


def test_reference_pack_append_keeps_complete_original745_registered_prefix():
    previous = asyncio.run(package._ProjectionChunkExtension().get_node_list())
    selected = asyncio.run(package._ReferenceTemporalPackExtension().get_node_list())
    assert len(previous) == 745 and selected[:-2] == previous
    assert selected[-2:] == packed.NODES and len(selected) == 747


def test_int8_append_keeps_complete747_prefix_including_both_prior_features():
    previous = asyncio.run(package._ReferenceTemporalPackExtension().get_node_list())
    selected = asyncio.run(package._Int8RowChunkExtension().get_node_list())
    assert len(previous) == 747 and selected[:-2] == previous
    assert selected[-2:] == quantized.NODES and len(selected) == 749


def test_ptd_append_keeps_complete749_prefix_and_all_original_schemas():
    previous = asyncio.run(package._Int8RowChunkExtension().get_node_list())
    selected = asyncio.run(package._PTDExtension().get_node_list())
    assert len(previous) == 749 and selected[:-3] == previous
    assert selected[-3:] == ptd.NODES and len(selected) == 752


def test_reference_step_cache_append_keeps_complete752_prefix():
    previous = asyncio.run(package._PTDExtension().get_node_list())
    selected = asyncio.run(package._ReferenceStepCacheExtension().get_node_list())
    assert len(previous) == 752 and selected[:-2] == previous
    assert selected[-2:] == cached.NODES and len(selected) == 754


def test_freevideo_h21_append_keeps_complete754_prefix_and_old_family():
    previous = asyncio.run(package._ReferenceStepCacheExtension().get_node_list())
    selected = asyncio.run(package._FreeVideoH21Extension().get_node_list())
    assert len(previous) == 754 and selected[:-8] == previous
    assert selected[-8:] == freevideo.NODES and len(selected) == 762
    assert [cls.__name__ for cls in selected[620:632]] == [
        cls.__name__ for cls in package._freevideo_node_classes]


def test_mc_append_keeps_complete762_prefix_and_default_disabled_operators():
    previous = asyncio.run(package._FreeVideoH21Extension().get_node_list())
    selected = asyncio.run(package._MCSeamExtension().get_node_list())
    assert len(previous) == 762 and selected[:-3] == previous
    assert selected[-3:] == seam.NODES and len(selected) == 765
    for cls in seam.NODES:
        enabled, = [item for item in cls.define_schema().inputs if item.id == "enabled"]
        assert enabled.default is False


def test_continuous_ptd_append_keeps_complete765_prefix_and_old_full_sampler():
    previous = asyncio.run(package._MCSeamExtension().get_node_list())
    selected = asyncio.run(package._PTDContinuousExtension().get_node_list())
    assert len(previous) == 765 and selected[:-3] == previous
    assert selected[-3:] == continuous.NODES and len(selected) == 768
    assert selected[749:752] == ptd.NODES


def test_periodic_av_append_keeps_complete768_prefix_and_off_default():
    previous = asyncio.run(package._PTDContinuousExtension().get_node_list())
    selected = asyncio.run(package._PeriodicAVExtension().get_node_list())
    assert len(previous) == 768 and selected[:-3] == previous
    assert selected[-3:] == periodic.NODES and len(selected) == 771
    enabled, = [item for item in periodic.NODES[0].define_schema().inputs if item.id == "enabled"]
    assert enabled.default is False


def test_solar_append_keeps_complete771_prefix_and_off_default():
    previous = asyncio.run(package._PeriodicAVExtension().get_node_list())
    selected = asyncio.run(package._SolarStructuralExtension().get_node_list())
    assert len(previous) == 771 and selected[:-3] == previous
    assert selected[-3:] == solar.NODES and len(selected) == 774
    enabled, = [item for item in solar.NODES[1].define_schema().inputs if item.id == "enabled"]
    assert enabled.default is False


def test_live_registry_all_actual_schemas_are_unique_and_match_feature_metadata():
    classes = asyncio.run(package.comfy_entrypoint().get_node_list())
    ids = [cls.GET_SCHEMA().node_id for cls in classes]
    features = json.loads((ROOT / "features.json").read_text(encoding="utf-8"))
    metadata = json.loads((ROOT / "meta.json").read_text(encoding="utf-8"))
    assert len(ids) == len(set(ids))
    assert ids == features["nodes"]
    for value in (features, metadata):
        selected = value["h21_working_tree_development"]
        assert selected["registered_nodes"] == 768
        assert selected["registered_nodes"] == selected["preserved_registered_prefix"] + selected["appended_nodes"]
        assert selected["whole_scope_completed"] is False
        current = value["h24_working_tree_development"]
        assert current["registered_nodes"] == len(ids)
        assert current["preserved_registered_prefix"] == selected["registered_nodes"]
        assert current["registered_nodes"] == current["preserved_registered_prefix"] + current["appended_nodes"]
        reviewed = json.loads((ROOT / "examples/workflows/91-h24-reviewed/index.json").read_text(encoding="utf-8"))
        assert current["published"] is True and current["human_quality_accepted"] is True
        assert current["publication_kind"] == "github_source"
        assert current["production_deployed"] is False
        assert current["human_feedback_sha256"] == reviewed["human_feedback_sha256"]
        assert set(current["human_review_case_ids"]) == {row["case_id"] for row in reviewed["workflows"]}
    expected = asyncio.run(package._ReferenceTemporalPackExtension().get_node_list())
    assert classes[:747] == expected
    assert len(classes) == 774
