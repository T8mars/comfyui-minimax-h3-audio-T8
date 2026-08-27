from __future__ import annotations

import json
from types import SimpleNamespace

from h3_audio_t8_pkg import residency_strategy_advanced as residency


def _planner(**kwargs):
    policy = {"schema": "t8.minimax_h3.vram_policy.v1", **kwargs}
    return policy, {"current_gate_pass": True, "runtime": {"gpu": {"device": "cpu"}}}


def test_report_only_and_presets_are_side_effect_free(monkeypatch):
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return _planner(**kwargs)

    monkeypatch.setattr(residency, "build_vram_policy", build)
    for strategy, reserve in (("report_only", 4.0), ("minimum_memory", 6.0), ("balanced", 4.0), ("faster", 2.0)):
        policy, value, gate, report_json = residency.build_h3_residency_strategy(strategy)
        report = json.loads(report_json)
        assert value == reserve and gate is True
        assert report["side_effects"] is False
        assert report["unload_all_models_called"] is False
        assert policy["clean_before_load"] is False
    assert calls[0]["mode"] == "report_only"
    assert all(not item["clean_before_load"] for item in calls)


def test_connected_model_reports_partial_residency_and_multiple_owner_warning(monkeypatch):
    monkeypatch.setattr(residency, "build_vram_policy", lambda **kwargs: _planner(**kwargs))
    diffusion = SimpleNamespace(manual_cast_dtype="fp8")
    model = SimpleNamespace(
        load_device="cuda:0",
        offload_device="cpu",
        model_size=lambda: 10 * 1024**2,
        loaded_size=lambda: 4 * 1024**2,
        lowvram_patch_counter=3,
        model=SimpleNamespace(diffusion_model=diffusion),
        model_options={
            "transformer_options": {
                "sol_morton": False,
                "t8_h3_lightx2v_sla_runtime_v1": {},
            }
        },
    )
    _policy, _reserve, _gate, report_json = residency.build_h3_residency_strategy(
        "balanced", model=model
    )
    report = json.loads(report_json)
    assert report["model"]["partially_loaded"] is True
    assert report["model"]["lowvram_patch_counter"] == 3
    assert report["conflicts"]
