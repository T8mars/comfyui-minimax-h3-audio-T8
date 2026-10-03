"""Actual native fifty-projection fit; not a perceptual/cross-model certificate."""
import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from h3_audio_t8_pkg import hyperflow_curve_fit_exp as curve
from h3_audio_t8_pkg.hyperflow_weights_advanced import parse_metadata
from test_hyperflow_advanced import _metadata, _synthetic_source


def inputs(tmp_path, *, int8=False):
    generator = torch.Generator().manual_seed(314159)

    def rnd(*shape):
        return torch.randn(*shape, generator=generator) * 0.08

    base = {"adaln_t_table": rnd(1025, 8)}
    teacher = {}
    for name in curve.PROJECTIONS:
        base[name + ".weight"] = rnd(16, 8)
        base[name + ".bias"] = rnd(16) + 1
        teacher[name + ".weight"] = rnd(16, 4)
        teacher[name + ".bias"] = rnd(16) + 1
    for part in ("proj_in", "proj_out"):
        teacher["time_embedder." + part + ".weight"] = rnd(4, 4)
        teacher["time_embedder." + part + ".bias"] = rnd(4)
    if int8:
        for name in list(teacher):
            if name.endswith(".weight"):
                prefix = name[:-len(".weight")]
                integer = (teacher[name] * 100).round().to(torch.int8)
                teacher[name] = integer
                teacher[prefix + ".weight_scale"] = torch.tensor(0.01)
                teacher[prefix + ".comfy_quant"] = torch.tensor(
                    list(json.dumps({"format": "int8_tensorwise", "convrot": False}).encode()), dtype=torch.uint8)
    paths = [tmp_path / (name + ".safetensors") for name in ("pruned", "teacher", "original")]
    save_file(base, str(paths[0]))
    save_file(teacher, str(paths[1]))
    save_file(_synthetic_source(), str(paths[2]), metadata=_metadata())
    return paths


def test_lstsq_and_chunk_errors_match_direct_double_solution():
    generator = torch.Generator().manual_seed(9001)
    weight = torch.randn(19, 8, generator=generator)
    bias = torch.randn(19, generator=generator)
    target = torch.randn(1041, 19, generator=generator)
    coordinates, errors = curve.solve_projection(weight, bias, target, chunk=31)
    expected = torch.linalg.lstsq(weight.double(), (target - bias).double().T).solution.T.float()
    torch.testing.assert_close(coordinates, expected, rtol=1e-5, atol=2e-6)
    actual = (torch.nn.functional.linear(expected, weight, bias) - target).norm(dim=1) / target.norm(dim=1)
    torch.testing.assert_close(torch.tensor(errors), actual, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="Rank-deficient"):
        curve.solve_projection(torch.ones(19, 8), bias, target)


def test_full_51_fit_exact_inputs_two_times_errors_and_mutation(tmp_path):
    paths = inputs(tmp_path)
    before = [curve._file(p) for p in paths]
    seen = []
    result = curve.build_fit(*paths, tmp_path / "fit.safetensors", progress=lambda n, error: seen.append((n, error)))
    assert [n for n, _ in seen] == list(range(1, 52))
    assert result.generated.shape == (51, 16, 8)
    assert result.pinned.shape == (51, 1025, 8)
    t, r = curve.trained_times(parse_metadata(_metadata()))
    assert torch.equal(result.t, t) and torch.equal(result.r, r)
    assert not torch.equal(t, r)
    assert len(result.metadata["errors"]) == 51
    for error in result.metadata["errors"]:
        assert error["generated_after"] <= error["generated_before"] + 1e-5
        assert error["pinned_after"] <= error["pinned_before"] + 1e-5
        assert error["worst_after"] >= error["generated_after"]
    assert [curve._file(p) for p in paths] == before
    assert result.metadata["teacher"]["sha256"] == before[1]["sha256"]
    assert result.metadata["quality_verified"] is False
    result.verify()
    with pytest.raises(ValueError, match="exact base"):
        curve.load_fit(result.path, base_sha256="0" * 64, adapter_sha256=before[2]["sha256"])
    result.generated[0, 0, 0] += 1
    with pytest.raises(ValueError, match="tensor mutated"):
        result.verify()


def test_native_int8_dequantized_linear_is_real_math(tmp_path):
    paths = inputs(tmp_path, int8=True)
    with safe_open(str(paths[1]), framework="pt", device="cpu") as handle:
        weight, bias = curve.linear(handle, "blocks.0.adaln_proj.linear", "cpu")
        assert torch.equal(weight, handle.get_tensor("blocks.0.adaln_proj.linear.weight").float() * 0.01)
        assert torch.equal(bias, handle.get_tensor("blocks.0.adaln_proj.linear.bias"))
    result = curve.build_fit(*paths, tmp_path / "fit.safetensors")
    assert result.generated.shape == (51, 16, 8)
    result.verify()


def test_cancel_and_input_changed_never_publish(tmp_path):
    paths = inputs(tmp_path)
    output = tmp_path / "fit.safetensors"

    def cancel():
        raise RuntimeError("explicit cancellation")

    with pytest.raises(RuntimeError, match="cancellation"):
        curve.build_fit(*paths, output, cancel=cancel)
    assert not output.exists()

    def changed(n, error):
        if n == 51:
            with paths[2].open("ab") as stream:
                stream.write(b"new-source")

    with pytest.raises(ValueError, match="source file"):
        curve.build_fit(*paths, output, progress=changed)
    assert not output.exists()


def test_exclusive_existing_output_and_original_alias(tmp_path):
    paths = inputs(tmp_path)
    output = tmp_path / "fit.safetensors"
    output.touch()
    with pytest.raises(FileExistsError, match="already exists"):
        curve.build_fit(*paths, output)
    assert output.read_bytes() == b""
    with pytest.raises(FileExistsError):
        curve.build_fit(*paths, paths[0])


def test_nonfinite_shape_and_wrong_original_adapter_fail(tmp_path):
    paths = inputs(tmp_path)
    source = _synthetic_source()
    source.pop(next(key for key in source if "endpoint_time" in key))
    save_file(source, str(paths[2]), metadata=_metadata())
    with pytest.raises(ValueError, match="Incomplete HyperFlow"):
        curve.build_fit(*paths, tmp_path / "bad.safetensors")
    with pytest.raises(ValueError, match="eight-coordinate"):
        curve.solve_projection(torch.ones(16, 9), torch.ones(16), torch.ones(17, 16))


def test_real_core_time_embedder_matches_fit_teacher_contract(tmp_path):
    from comfy.ldm.minimax.model import TimeEmbedder
    from torch import nn

    paths = inputs(tmp_path)
    native = TimeEmbedder(4, 4, 4, dtype=torch.float32, device="cpu", operations=nn)
    with safe_open(str(paths[1]), framework="pt", device="cpu") as handle:
        native.load_state_dict({key.removeprefix("time_embedder."): handle.get_tensor(key)
                               for key in handle.keys() if key.startswith("time_embedder.")}, strict=True)
    t = torch.linspace(0, 1, 1025)
    half = 2
    frequency = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(half) / half)
    angle = t[:, None] * frequency
    encoded = torch.cat((angle.cos(), angle.sin()), dim=1)
    expected = native.proj_out(torch.nn.functional.silu(native.proj_in(encoded)))
    assert torch.equal(native(t), expected)
    assert not torch.cuda.is_initialized()


def test_atomic_concurrent_creator_preserved(tmp_path, monkeypatch):
    paths = inputs(tmp_path)
    output = tmp_path / "fit.safetensors"
    original_link = curve.os.link

    def concurrent(source, target):
        save_file({"other": torch.tensor([42])}, str(target))
        return original_link(source, target)

    monkeypatch.setattr(curve.os, "link", concurrent)
    with pytest.raises(FileExistsError):
        curve.build_fit(*paths, output)
    with safe_open(str(output), framework="pt") as handle:
        assert handle.keys() == ["other"]
        assert handle.get_tensor("other").item() == 42
    assert not list(tmp_path.glob("*.part"))
