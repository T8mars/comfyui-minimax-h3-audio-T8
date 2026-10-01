"""Core's file-path import is accepted, foreign implementations are not."""

import importlib.util
from pathlib import Path
import sys
import types

from comfy_extras import nodes_custom_sampler as builtin

from h3_audio_t8_pkg.modular_sampling.core_sampler_identity import (
    native_basic_guider_class,
    native_random_noise_class,
)
from h3_audio_t8_pkg.modular_sampling.noise import operator_identity


def _alias(monkeypatch):
    nodes = types.ModuleType("nodes")
    nodes.NODE_CLASS_MAPPINGS = {}
    monkeypatch.setitem(sys.modules, "nodes", nodes)
    path = Path(builtin.__file__).resolve()
    spec = importlib.util.spec_from_file_location("_t8_core_sampler_test_alias", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "RandomNoise", module.RandomNoise)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "BasicGuider", module.BasicGuider)
    return module, nodes


def test_exact_file_path_alias_is_a_portable_native_provider(monkeypatch):
    alias, _ = _alias(monkeypatch)
    assert alias.Noise_RandomNoise is not builtin.Noise_RandomNoise
    assert alias.Guider_Basic is not builtin.Guider_Basic
    assert native_random_noise_class() is alias.Noise_RandomNoise
    assert native_basic_guider_class() is alias.Guider_Basic
    assert operator_identity(alias.Noise_RandomNoise(17)) == {
        "portable": True, "provider": "Noise_RandomNoise", "seed": 17,
    }


def test_modified_or_foreign_core_mapping_is_not_trusted(monkeypatch):
    alias, nodes = _alias(monkeypatch)

    class ForeignRandomNoise:
        pass

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "RandomNoise", ForeignRandomNoise)
    assert native_random_noise_class() is None
    assert operator_identity(alias.Noise_RandomNoise(17))["portable"] is False

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "RandomNoise", alias.RandomNoise)
    monkeypatch.setattr(alias.Noise_RandomNoise, "generate_noise", lambda self, latent: latent)
    assert native_random_noise_class() is None
    assert operator_identity(alias.Noise_RandomNoise(17))["portable"] is False

    monkeypatch.setattr(alias.Guider_Basic, "set_conds", lambda self, positive: None)
    assert native_basic_guider_class() is None
