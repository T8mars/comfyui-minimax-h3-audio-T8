from __future__ import annotations

from h3_audio_t8_pkg.tools.validate_h3_speed_reference_outputs import (
    _conditioning_contract,
)


def test_conditioning_contract_requires_every_stage_to_rebuild_expected_refs():
    report = {
        "stages": [
            {
                "conditioning_route": "full_conditioning_rebuild",
                "conditioning_report": (
                    "task=ref2va\naudio_mode=native\n"
                    "pictures=0, videos=1, audios=1"
                ),
            },
            {
                "conditioning_route": "full_conditioning_rebuild",
                "conditioning_report": (
                    "task=ref2va\naudio_mode=native\n"
                    "pictures=0, videos=1, audios=1"
                ),
            },
        ]
    }
    case = {
        "task": "ref2va",
        "conditioning_counts": "pictures=0, videos=1, audios=1",
    }
    assert _conditioning_contract(report, case) is True
    report["stages"][1]["conditioning_report"] = (
        "task=ref2va\npictures=0, videos=1, audios=0"
    )
    assert _conditioning_contract(report, case) is False
