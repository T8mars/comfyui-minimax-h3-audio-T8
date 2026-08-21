from __future__ import annotations

from copy import deepcopy

import pytest

from h3_audio_t8_pkg.tools.build_eav_prompt_relay_probe_prompts import (
    GLOBAL_PROMPT,
    SEED,
    build_prompt,
)


def test_eav_prompt_relay_probe_is_a_strict_single_variable_pair():
    relay_only = build_prompt("disabled")
    relay_plus_eav = build_prompt("apply_exp")
    control = deepcopy(relay_only)
    treatment = deepcopy(relay_plus_eav)
    control["9"]["inputs"]["mode"] = "PAIR_MODE"
    treatment["9"]["inputs"]["mode"] = "PAIR_MODE"
    control["14"]["inputs"]["filename_prefix"] = "PAIR_OUTPUT"
    treatment["14"]["inputs"]["filename_prefix"] = "PAIR_OUTPUT"
    assert control == treatment


def test_eav_prompt_relay_probe_uses_low_load_stock20_contract():
    prompt = build_prompt("apply_exp")
    conditioning = prompt["6"]["inputs"]
    dual_clock = prompt["7"]["inputs"]
    composer = prompt["9"]
    assert [conditioning[key] for key in ("width", "height")] == [736, 416]
    assert conditioning["task_type"] == "T2VA"
    assert conditioning["execution_mode"] == "apply_exp"
    assert conditioning["query_chunk_rows"] == 256
    assert dual_clock["steps"] == 20
    assert dual_clock["model"] == ["6", 0]
    assert composer["class_type"] == (
        "MiniMaxH3EnhanceAVideoPromptRelayComposerT8Advanced"
    )
    assert composer["inputs"]["model"] == ["7", 0]
    assert composer["inputs"]["sigmas"] == ["7", 2]
    assert composer["inputs"]["sampling_profile"] == "stock20"
    assert prompt["10"]["inputs"]["model"] == ["9", 0]
    assert prompt["12"]["inputs"] == {
        "av_latent": ["11", 0],
        "runtime": ["9", 1],
    }
    assert prompt["13"]["inputs"]["av_latent"] == ["12", 0]
    assert prompt["8"]["inputs"]["noise_seed"] == SEED
    assert "你在干嘛呢，我在这里呀，看看效果如何" in GLOBAL_PROMPT


def test_eav_prompt_relay_probe_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unsupported"):
        build_prompt("report_only")
