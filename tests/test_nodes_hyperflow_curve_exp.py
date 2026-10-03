"""Public curve nodes: typed separate phases and real tiny execution/storage."""
import json
import math
from pathlib import Path

import pytest
import torch

from h3_audio_t8_pkg import nodes_hyperflow_curve_exp as nodes
from h3_audio_t8_pkg import hyperflow_curve_runtime_exp as runtime
from h3_audio_t8_pkg.modular_sampling import hyperflow_curve as stages
from test_hyperflow_curve_runtime_exp import prepare
from test_hyperflow_curve_stages import equal
from test_progressive_sampling_runtime import conditioning


@pytest.mark.parametrize("cls", nodes.NODES)
def test_public_schema_exact_unique_and_no_old_type(cls):
    schema = cls.define_schema().get_v1_info(cls)
    assert schema.name == cls.__name__
    assert "Curves Experimental" in schema.category
    assert schema.experimental is True
    assert len(nodes.NODES) == len({c.__name__ for c in nodes.NODES}) == 15
    assert nodes.BOUNDARY != "T8_HYPERFLOW_CONTINUOUS_BOUNDARY"
    assert nodes.RESULT != "T8_HYPERFLOW_COMPLETED_AV"


def test_asset_load_real_sha_no_sampling_and_not_model_certificate(tmp_path, monkeypatch):
    _, _, _, _, files = prepare(tmp_path, monkeypatch, portable=True)
    fit = nodes.MiniMaxH3HyperFlowCurveFitLoadEXPT8.execute("missing_curve_fit", str(files["fit"]))
    report = json.loads(fit[3])
    assert report["model_checked"] is False and report["quality_accepted"] is False
    assert fit[2] == nodes.fitting._file(files["fit"])["sha256"]
    assert nodes.MiniMaxH3HyperFlowCurveFitLoadEXPT8.fingerprint_inputs("missing_curve_fit", str(files["fit"])) == fit[2]
    with pytest.raises(ValueError, match="SHA mismatch"):
        nodes.MiniMaxH3HyperFlowCurveFitLoadEXPT8.execute("missing_curve_fit", str(files["fit"]), "0" * 64)
    with pytest.raises(ValueError, match="absolute"):
        nodes._fit_path("missing", "relative.safetensors")
    assert not torch.cuda.is_initialized()


def test_public_apply_and_independent_head_tail_store_load_actual_native(tmp_path, monkeypatch):
    import comfy.nested_tensor
    from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
    base, _, fit, _, files = prepare(tmp_path, monkeypatch, portable=True)
    monkeypatch.setattr(nodes, "_base", lambda name: files[name])
    monkeypatch.setattr(nodes, "_resolve", lambda _name: files["adapter"])
    monkeypatch.setattr(nodes, "_store_root", lambda: tmp_path / "store")
    apply = nodes.MiniMaxH3HyperFlowCurveModelApplyEXPT8
    low = apply.execute(base, fit, "base", "adapter")[0]
    high = apply.execute(base, fit, "base", "adapter")[0]
    source = {"samples": comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, 2, 4, 4), torch.zeros(1, 32, 2, 3)))}
    positive = conditioning()
    bound = nodes.MiniMaxH3HyperFlowCurveHeadEffectsBindEXPT8.execute(low, source, positive, [])
    boundary = nodes.MiniMaxH3HyperFlowCurveHeadStageEXPT8.execute(bound[0], source, Noise_RandomNoise(7), bound[1], bound[2])[0]
    audit = nodes.MiniMaxH3HyperFlowCurveHeadEffectsAuditEXPT8.execute(boundary)
    assert json.loads(audit[1])["phase"] == "head"
    saved = nodes.MiniMaxH3HyperFlowCurveHeadSaveEXPT8.execute(audit[0])
    loaded = nodes.MiniMaxH3HyperFlowCurveHeadLoadEXPT8.execute(saved[1], saved[2])
    assert json.loads(loaded[1])["sampling_calls"] == 0
    bound_tail = nodes.MiniMaxH3HyperFlowCurveTailEffectsBindEXPT8.execute(loaded[0], high, positive, [])
    def forbidden(*_args, **_kwargs):
        raise AssertionError("TAIL may not execute HEAD or draw fresh noise")
    monkeypatch.setattr(stages, "sample_head", forbidden)
    monkeypatch.setattr(nodes, "_generate_noise", forbidden)
    tail = nodes.MiniMaxH3HyperFlowCurveTailStageEXPT8.execute(loaded[0], bound_tail[0], bound_tail[1], bound_tail[2], seed=7)
    audit_tail = nodes.MiniMaxH3HyperFlowCurveTailEffectsAuditEXPT8.execute(tail[2])
    assert json.loads(audit_tail[2])["phase"] == "tail"
    saved_tail = nodes.MiniMaxH3HyperFlowCurveTailSaveEXPT8.execute(audit_tail[0])
    monkeypatch.setattr(stages, "sample_tail", forbidden)
    delivery = nodes.MiniMaxH3HyperFlowCurveTailLoadEXPT8.execute(saved_tail[1], saved_tail[2])
    equal(tail[0], delivery[0])
    assert json.loads(delivery[2])["sampling_calls"] == 0
    assert nodes.MiniMaxH3HyperFlowCurveHeadLoadEXPT8.fingerprint_inputs(saved[1], saved[2]) == saved[2]
    assert nodes.MiniMaxH3HyperFlowCurveTailLoadEXPT8.fingerprint_inputs(saved_tail[1], saved_tail[2]) == saved_tail[2]
    assert runtime.ACTIVE.get() is None and not torch.cuda.is_initialized()


def test_build_is_explicit_unique_cancelable_producer_no_sampling(tmp_path, monkeypatch):
    import comfy.model_management
    import folder_paths
    seen = []
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    monkeypatch.setattr(nodes, "_base", lambda name: Path(name))
    monkeypatch.setattr(nodes, "_resolve", lambda name: Path(name))
    def producer(base, teacher, adapter, output, **kwargs):
        seen.append((base, teacher, adapter, output, kwargs))
        raise RuntimeError("producer-only sentinel")
    monkeypatch.setattr(nodes.fitting, "build_fit", producer)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="producer-only"):
            nodes.MiniMaxH3HyperFlowCurveFitBuildEXPT8.execute("pruned", "full", "adapter", "cpu")
    assert seen[0][3] != seen[1][3]
    for base, teacher, adapter, path, kwargs in seen:
        assert (base, teacher, adapter) == tuple(map(Path, ("pruned", "full", "adapter")))
        assert path.parent == tmp_path / "MiniMaxH3" / "curve_fits" and not path.exists()
        assert kwargs["cancel"] is comfy.model_management.throw_exception_if_processing_interrupted
        assert kwargs["device"] == "cpu" and callable(kwargs["progress"])


def test_full_setup_has_no_partial_restart_selector():
    schema = nodes.MiniMaxH3HyperFlowCurveFullSamplerSetupEXPT8.define_schema().get_v1_info(nodes.MiniMaxH3HyperFlowCurveFullSamplerSetupEXPT8)
    assert set(schema.input["required"]) == {"model", "av_latent"}
    with pytest.raises(ValueError, match="dedicated curve HEAD"):
        nodes.MiniMaxH3HyperFlowCurveTailStageEXPT8.execute({}, None, [], [])


def test_asset_size_preflight_precedes_any_full_sha(tmp_path, monkeypatch):
    path = tmp_path / "wrong-base.safetensors"
    with path.open("wb") as stream:
        stream.truncate(16 * 1024 * 1024 + 1)
    def forbidden(_path):
        raise AssertionError("Oversized fit must be refused before full SHA I/O")
    monkeypatch.setattr(nodes.fitting, "_file", forbidden)
    with pytest.raises(ValueError, match="16MiB"):
        nodes.MiniMaxH3HyperFlowCurveFitLoadEXPT8.execute("missing", str(path))
    assert math.isnan(nodes.MiniMaxH3HyperFlowCurveFitLoadEXPT8.fingerprint_inputs("missing", str(path)))
