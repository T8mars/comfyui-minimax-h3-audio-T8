from __future__ import annotations

from h3_audio_t8_pkg.tools.validate_h3_speed_turbo8_output import (
    _turbo8_report_contract,
)


def test_turbo8_report_contract_requires_six_plus_two_nfe_and_no_lora_overclaim():
    report = {
        "execution_scope": "turbo8_t2va_research_exp",
        "resolved_task": "t2va",
        "steps": 8,
        "nfe": 8,
        "stages": [{"nfe": 6}, {"nfe": 2}],
        "weight_patch_contract": {
            "has_weight_patches": True,
            "lora_identity_verified_by_runtime": False,
        },
    }
    assert all(_turbo8_report_contract(report).values())
    report["stages"][1]["nfe"] = 3
    assert _turbo8_report_contract(report)["stage_nfe_exact"] is False
