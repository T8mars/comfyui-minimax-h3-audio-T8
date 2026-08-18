from __future__ import annotations

import pytest

from h3_audio_t8_pkg.tools.build_h3_speed_turbo8_validation import (
    build_turbo8_speed_prompt,
)


def test_turbo8_builder_has_explicit_lora_scope_and_eight_total_steps():
    prompt = build_turbo8_speed_prompt()
    assert prompt["2"]["class_type"] == "LoraLoaderModelOnly"
    assert prompt["2"]["inputs"]["model"] == ["1", 0]
    assert prompt["6"]["inputs"]["steps"] == 8
    assert prompt["8"]["inputs"]["model"] == ["2", 0]
    assert prompt["8"]["inputs"]["execution_scope"] == "turbo8_t2va_research_exp"
    assert prompt["7"]["inputs"]["task_type"] == "T2VA"
    assert prompt["7"]["inputs"]["audio_mode"] == "native"


def test_turbo8_builder_rejects_invalid_canvas_or_frame_grid():
    with pytest.raises(ValueError, match="multiples of 32"):
        build_turbo8_speed_prompt(width=1000)
    with pytest.raises(ValueError, match=r"17n\+5"):
        build_turbo8_speed_prompt(length=123)
