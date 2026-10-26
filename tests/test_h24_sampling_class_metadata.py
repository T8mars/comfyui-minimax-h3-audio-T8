"""Inert CPython class metadata must not invalidate native sampling identity."""
import sys

import pytest

from h3_audio_t8_pkg import long_video_dual_identity as identity
from h3_audio_t8_pkg import sampling
from h3_audio_t8_pkg.patch_stack_policy import UnverifiedModelStack
from test_fast_h3_v2_core_sampler import model
from test_face_refine_parity_advanced import _locked_av
from test_native_sigma_shift_identity import _configured


@pytest.fixture(params=("raw", "native_av", "core_factory"))
def selected_sampling(request):
    if request.param == "raw":
        return sampling.MiniMaxH3FlowSampling()
    if request.param == "native_av":
        patched, _, _ = sampling.setup_dual_clock_sampling(
            model(), _locked_av(), 8, 12., 3., "er_sde", "simple")
    else:
        patched = _configured()
    return patched.get_model_object("model_sampling")


@pytest.mark.parametrize("metadata", (
    {}, {"__firstlineno__": 36}, {"__static_attributes__": ()},
    {"__static_attributes__": ("sigma",)},
    {"__firstlineno__": 36, "__static_attributes__": ()},
    {"__firstlineno__": -1, "__static_attributes__": ("sigma", "shift")},
))
def test_metadata_does_not_change_sampling_identity(selected_sampling, metadata, monkeypatch):
    cls = type(selected_sampling)
    # Both keys are independently optional, including on a real 3.13 class.
    for name in ("__firstlineno__", "__static_attributes__"):
        monkeypatch.delattr(cls, name, raising=False)
    expected = identity._v2_sampling_identity(selected_sampling)
    for name, value in metadata.items():
        monkeypatch.setattr(cls, name, value, raising=False)
    assert identity._v2_sampling_identity(selected_sampling) == expected


class IntSubclass(int):
    pass


class TupleSubclass(tuple):
    pass


class StrSubclass(str):
    pass


@pytest.mark.parametrize(("name", "value"), (
    ("__firstlineno__", True), ("__firstlineno__", 1.5),
    ("__firstlineno__", "36"), ("__firstlineno__", IntSubclass(36)),
    ("__static_attributes__", []), ("__static_attributes__", TupleSubclass()),
    ("__static_attributes__", (1,)), ("__static_attributes__", (StrSubclass("sigma"),)),
))
def test_invalid_metadata_is_not_certified(selected_sampling, name, value, monkeypatch):
    monkeypatch.setattr(type(selected_sampling), name, value, raising=False)
    with pytest.raises(UnverifiedModelStack):
        identity._v2_sampling_identity(selected_sampling)


@pytest.mark.parametrize("name", ("__firstlineno__", "__static_attributes__"))
@pytest.mark.parametrize("descriptor", (False, True))
def test_metadata_is_read_without_calling_user_code(selected_sampling, name, descriptor, monkeypatch):
    called = []

    def unexpected(*args):
        called.append(args)
        raise AssertionError("Metadata must not execute a user callback")

    monkeypatch.setattr(type(selected_sampling), name,
                        property(unexpected) if descriptor else unexpected, raising=False)
    with pytest.raises(UnverifiedModelStack):
        identity._v2_sampling_identity(selected_sampling)
    assert called == []


def test_metadata_names_do_not_allow_real_execution_members(selected_sampling, monkeypatch):
    cls = type(selected_sampling)
    monkeypatch.setattr(cls, "__static_attributes__", ("sigma",), raising=False)
    monkeypatch.setattr(cls, "sigma", lambda self, value: value)
    with pytest.raises(UnverifiedModelStack):
        identity._v2_sampling_identity(selected_sampling)


def test_real_interpreter_metadata_is_supported(selected_sampling):
    if sys.version_info < (3, 13):
        pytest.skip("Actual CPython 3.13+ compiler metadata requires that interpreter")
    namespace = vars(type(selected_sampling))
    assert type(namespace["__firstlineno__"]) is int
    assert type(namespace["__static_attributes__"]) is tuple
    assert identity._v2_sampling_identity(selected_sampling)["protocol"] in {
        "t8_raw_audio_flow", "native_av_carrier", "core_minimax_h3_sigma_shift"}
