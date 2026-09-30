# ruff: noqa: F811 -- pytest imports fixtures, then injects them by parameter name.
from dataclasses import replace
import json

import pytest

from test_long_video_dual_model_runner import rig  # noqa: F401
from test_semantic_bridge import weight_file, config  # noqa: F401
from h3_audio_t8_pkg import semantic_bridge as sb


def install(engine, configs):
    engine.bridge_configs = configs
    engine.bridge_identities = tuple(sb.preflight_bridge(item) for item in configs)


def test_independent_bridge_routes_and_resume_cache(rig, weight_file, monkeypatch, tmp_path):
    engine, run, calls, _, first, second = rig
    configs = (config(weight_file, alpha=.1), config(weight_file, alpha=.2))
    install(engine, configs)
    original, seen = engine._conditions, []

    def conditions(model, context, inputs, plan):
        seen.append((model, inputs.get("semantic_bridge")))
        return original(model, context, inputs, plan)
    monkeypatch.setattr(engine, "_conditions", conditions)
    run()
    assert seen == [(first, configs[0]), (second, configs[1])]
    before = len(calls)
    result = run()
    assert result["sampling_report"]["dual_model"]["high_reused"]
    assert not any(call[0] == "sample" for call in calls[before:])
    # HIGH template is reconstructed, but cached samplers never run twice.
    assert seen[-1] == (second, configs[1]) and len(seen) == 3
    records = list(tmp_path.rglob("high_output-*.json"))
    assert json.loads(records[0].read_text())["contract"]["semantic_bridges"] == list(engine.bridge_identities)
    install(engine, (configs[0], replace(configs[1], alpha=.3)))
    before = len(calls)
    result = run()
    assert not result["sampling_report"]["dual_model"]["high_reused"]
    assert sum(call[0] == "sample" for call in calls[before:]) == 2


def test_explicit_disabled_pass_does_not_read_missing_weights(rig, weight_file, monkeypatch):
    engine, run, *_ = rig
    disabled = sb.BridgeConfig("no-such-file", "", enabled=False)
    install(engine, (config(weight_file), disabled))
    original, seen = engine._conditions, []

    def conditions(model, context, inputs, plan):
        seen.append(inputs.get("semantic_bridge"))
        return original(model, context, inputs, plan)
    monkeypatch.setattr(engine, "_conditions", conditions)
    run()
    assert seen[1] is disabled and engine.bridge_identities[1] is None


def test_changed_bridge_file_is_checked_before_cached_stage(rig, weight_file, monkeypatch):
    engine, run, calls, *_ = rig
    install(engine, (config(weight_file), None))
    run()
    before = len(calls)
    monkeypatch.setattr(sb, "file_sha", lambda _: "replaced-content")
    with pytest.raises(ValueError, match="content changed"):
        run()
    assert len(calls) == before


def test_composite_pass_cache_binds_each_bridge_and_order(rig, weight_file, tmp_path):
    from safetensors.torch import save_file
    engine, run, calls, *_ = rig
    weights, _ = sb.read_weights(weight_file)
    weights["fc1.bias"] = weights["fc1.bias"] + .02
    second_path = tmp_path / "second-semantic-bridge.safetensors"
    save_file(weights, second_path)
    first, second = config(weight_file), config(second_path, alpha=.2)
    disabled = sb.BridgeConfig("missing", "", enabled=False)
    stack = sb.compose_bridges(first, second)
    install(engine, (stack, disabled))
    run()
    before = len(calls)
    result = run()
    assert result["sampling_report"]["dual_model"]["high_reused"]
    assert not any(call[0] == "sample" for call in calls[before:])
    records = list(tmp_path.rglob("high_output-*.json"))
    identity = json.loads(records[0].read_text())["contract"]["semantic_bridges"][0]
    assert identity == stack.identity()
    assert [item["alpha"] for item in identity["bridges"]] == [.1, .2]
    install(engine, (sb.compose_bridges(second, first), disabled))
    before = len(calls)
    result = run()
    assert not result["sampling_report"]["dual_model"]["high_reused"]
    assert sum(call[0] == "sample" for call in calls[before:]) == 2
