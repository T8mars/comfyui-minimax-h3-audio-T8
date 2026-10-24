"""Actual installed Sol definitions; CPU math is explicitly fallback evidence."""
import importlib
import inspect
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch
from comfy.ldm.modules import attention

from h3_audio_t8_pkg import relay_sol_backend as sol
from h3_audio_t8_pkg import prompt_relay_advanced as relay
from h3_audio_t8_pkg import enhance_a_video_advanced as eav
from test_prompt_relay_core_compat import model_fixture, bound_layout


@pytest.fixture
def installed_sol(monkeypatch):
    path = Path(inspect.getsourcefile(attention)).parents[3] / "custom_nodes/ComfyUI-sol-attn"
    if not (path / "nodes.py").is_file():
        pytest.skip("actual Sol node is not installed")
    package = ModuleType("test_actual_sol")
    package.__path__ = [str(path)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    # No extension __init__, global backend selection, or GPU work.
    return importlib.import_module(package.__name__ + ".nodes")


def test_selector_authentication_and_separate_bind_counters(installed_sol):
    patched = installed_sol.SolAttentionPatch().patch(model_fixture(), True, 1.3, min_tokens=256)[0]
    fn = patched.model_options["transformer_options"]["optimized_attention_override"]
    a, b = sol.capture_composed_backend(fn), sol.capture_composed_backend(fn)
    assert type(a) is sol.SolRelayBackend and type(b) is sol.SolRelayBackend
    assert a is not b and a.kernel is installed_sol.sol_attn
    assert a.report()["completed_calls"] == {}
    a.counters["fake_not_an_execution"] += 1
    assert b.report()["completed_calls"] == {}
    binding, _ = bound_layout("joint_av_exp")
    composed, _ = relay.patch_prompt_relay_model(patched, binding, 32)
    assert type(relay.prompt_relay_model_contract(composed)["attention_backend"]) is sol.SolRelayBackend


@pytest.mark.parametrize("mutation", ["foreign_kernel", "bad_tau", "bad_min", "unknown_previous"])
def test_changed_or_unknown_sol_is_not_authenticated(installed_sol, monkeypatch, mutation):
    params = dict(tau=1.3, min_tokens=256, strict=True)
    if mutation == "bad_tau":
        params["tau"] = float("nan")
    elif mutation == "bad_min":
        params["min_tokens"] = 0
    elif mutation == "unknown_previous":
        params["fallback_override"] = lambda *a, **k: pytest.fail("unknown fallback ran")
    else:
        monkeypatch.setattr(installed_sol, "sol_attn", lambda *a, **k: pytest.fail("foreign kernel ran"))
    fn = installed_sol._make_override(**params)
    # Strict authentication must fail, but the user-selected callable stays
    # delegated under the existing advisory policy, never certified or invoked
    # by inspection. Invalid arguments/kernel failures propagate if later used.
    assert sol.capture_sol_relay_backend(fn) is None
    backend = sol.capture_composed_backend(fn)
    assert type(backend) is sol.UserSelectedBackend
    assert backend.override is fn
    assert backend.report() == {
        "kind": "user_selected_unverified_delegate",
        "portable_cache_reuse": False,
        "completed_calls": {},
        "kernel_verified": False,
    }


@pytest.mark.parametrize("variant", ["bias", "short_query", "cpu"])
def test_unsupported_shape_keeps_bias_and_never_claims_sol(installed_sol, variant):
    backend = sol.capture_sol_relay_backend(installed_sol._make_override(1.3, 256, True))
    assert backend is not None
    q, k, v = [torch.randn(1, 2, 9, 128, dtype=torch.bfloat16) for _ in range(3)]
    mask = None
    if variant == "bias":
        mask = torch.zeros(9, 9, dtype=torch.bfloat16)
        mask[:, :4] = -5
    elif variant == "short_query":
        q = q[:, :, :3]
    saved = [x.clone() for x in (q, k, v)]
    actual = backend.attention(q, k, v, 2, mask=mask, skip_reshape=True, skip_output_reshape=True)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.testing.assert_close(actual, expected)
    assert sum(backend.counters.values()) == 1
    assert backend.counters["sol:completed"] == 0
    for before, after in zip(saved, (q, k, v)):
        torch.testing.assert_close(before, after)


def test_eav_only_captures_sol_without_discarding_its_settings(installed_sol):
    source = installed_sol.SolAttentionPatch().patch(model_fixture(), True, 1.7, min_tokens=256)[0]
    sigmas = torch.cat([torch.linspace(1., .05, 20), torch.zeros(1)])
    patched, runtime, _ = eav.build_eav_model(source, sigmas, mode="apply_exp", tau=4.,
        start_video_progress=0., end_video_progress=1., max_workspace_mib=32, g_hard_limit=3.)
    assert runtime.config["composed_attention_backend"]["configuration"]["tau"] == 1.7
    assert runtime.config["core_contract"]
    assert patched is not source
