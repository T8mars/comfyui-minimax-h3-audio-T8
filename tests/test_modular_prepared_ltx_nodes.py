"""New S28 public graph and legacy prepared registration contracts."""

import asyncio
import json
import math
from pathlib import Path

import pytest
from comfy_api.latest import InputImpl

from h3_audio_t8_pkg import comfy_entrypoint
from h3_audio_t8_pkg import nodes_prepared_generation as legacy_nodes
from h3_audio_t8_pkg.modular_sampling import prepared_ltx_nodes as nodes
from h3_audio_t8_pkg.modular_sampling import prepared_ltx_stage as stage
from tools.build_modular_prepared_ltx_workflows import build_prompt, build_workflow

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "artifacts/development/modular-sampling-m4-prepared-ltx-20260923/candidate-v1"


def test_append_only_public_registration_and_stable_legacy_ids():
    classes = asyncio.run(comfy_entrypoint().get_node_list())
    for cls in nodes.NODES:
        assert classes.count(cls) == 1
        assert cls.define_schema().is_experimental
        assert math.isnan(cls.fingerprint_inputs())
    for cls in legacy_nodes.PREPARED_GENERATION_NODE_CLASSES:
        assert classes.count(cls) == 1
    assert nodes.MiniMaxH3PreparedLTXDecodeEXPT8.define_schema().is_output_node
    assert not nodes.MiniMaxH3PreparedLTXGenerateEXPT8.define_schema().is_output_node


@pytest.mark.parametrize("resume,stem", [
    (False, "Prepared_LTX_Generate_Decode_EXP"),
    (True, "Prepared_LTX_DecodeOnly_EXP"),
])
def test_saved_workflow_matches_current_nodes_and_does_not_use_combined_node(resume, stem):
    classes = [legacy_nodes.MiniMaxH3PreparedGenerationBundleEXPT8, *nodes.NODES]
    info = {cls.define_schema().node_id: cls.GET_NODE_INFO_V1() for cls in classes}
    for value in info.values():
        value["cnr_id"] = "minimax-h3-audio-T8"
    api = build_prompt(resume=resume)
    frontend = build_workflow(info, resume=resume)
    assert json.loads((CANDIDATE / (stem + ".api.json")).read_text(encoding="utf8")) == api
    assert json.loads((CANDIDATE / (stem + ".json")).read_text(encoding="utf8")) == frontend
    assert len(api) == 3 and len(frontend["nodes"]) == 4
    assert api["3"]["class_type"] == "MiniMaxH3PreparedLTXDecodeEXPT8"
    assert api["2"]["class_type"] == ("MiniMaxH3PreparedLTXLoadGenerationEXPT8" if resume
                                           else "MiniMaxH3PreparedLTXGenerateEXPT8")
    assert "MiniMaxH3PreparedVideoEXPT8" not in {node["class_type"] for node in api.values()}


def test_public_nodes_forward_independent_operations_without_old_combined_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes.folder_paths, "get_output_directory", lambda: str(tmp_path))
    bundle = {"synthetic": "bundle"}
    latent = tmp_path / "latent.safetensors"
    latent.write_bytes(b"synthetic")
    movie = tmp_path / stage.NAMESPACE / "fixture" / "movie.mp4"
    movie.parent.mkdir(parents=True)
    movie.write_bytes(b"synthetic media placeholder")
    receipt = {"latent_path": str(latent), "synthetic": True}
    observed = []

    def generation(value, **settings):
        observed.append(("generation", settings))
        assert value == bundle and value is not bundle
        return receipt, {"status": "synthetic_generation"}

    def load(value, **settings):
        observed.append(("load", settings))
        assert value == bundle and value is not bundle
        return receipt, {"status": "synthetic_load"}

    def decode(value, value_receipt, **settings):
        observed.append(("decode", settings))
        assert value == bundle and value is not bundle
        assert value_receipt == receipt and value_receipt is not receipt
        return movie, {"status": "synthetic_decode"}

    monkeypatch.setattr(stage, "run_generation", generation)
    monkeypatch.setattr(stage, "load_generation", load)
    monkeypatch.setattr(stage, "run_decode", decode)
    lease = str(tmp_path / "serial.lock")
    generated = nodes.MiniMaxH3PreparedLTXGenerateEXPT8.execute(bundle, 8301, "fixture", True, lease)
    loaded = nodes.MiniMaxH3PreparedLTXLoadGenerationEXPT8.execute(bundle, 8301, "fixture", "a" * 64)
    decoded = nodes.MiniMaxH3PreparedLTXDecodeEXPT8.execute(bundle, receipt, lease)
    assert generated.result[0] == loaded.result[0] == receipt
    assert isinstance(decoded.result[0], InputImpl.VideoFromFile)
    assert decoded.result[1] == str(movie) and decoded.ui is not None
    assert [name for name, _ in observed] == ["generation", "load", "decode"]
    assert observed[0][1]["lease_path"] == Path(lease)
    assert observed[1][1]["expected_sha256"] == "a" * 64


def test_missing_gpu_lease_rejected_before_stage_execution(monkeypatch):
    monkeypatch.setattr(stage, "run_generation", lambda *args, **kwargs: pytest.fail("worker started"))
    with pytest.raises(ValueError, match="absolute"):
        nodes.MiniMaxH3PreparedLTXGenerateEXPT8.execute({}, 8301, "fixture", True, "")
