from __future__ import annotations

import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import validate_skin_finish_surface_stream as validation  # noqa: E402


def test_dichromatic_stream_validation_is_pinned_single_run_and_low_load():
    source = Path(validation.__file__).read_text(encoding="utf-8")
    assert validation.DICHROMATIC_OUTPUT.name.endswith("-v1")
    assert validation.DICHROMATIC_PARAMETERS == {
        "amount": 0.90,
        "specular_strength": 0.85,
        "diffuse_radius_percent": 2.5,
        "specular_threshold_linear": 0.003,
        "specular_softness_linear": 0.025,
        "chroma_dilution_threshold": 0.001,
        "chroma_dilution_softness": 0.015,
        "minimum_diffuse_chroma": 0.006,
        "diffuse_chroma_softness": 0.035,
        "minimum_direction_cosine": 0.70,
        "maximum_surface_delta": 0.08,
        "minimum_texture_ratio": 0.82,
        "maximum_texture_ratio": 1.10,
    }
    assert 'choices=("surface", "dichromatic")' in source
    assert "attenuate_skin_specular_dichromatic" in source
    assert '"loads_h3": False' in source
    assert '"loads_sam": False' in source
    assert '"stress_or_repeat": False' in source
    assert "chunk_frames=2" in source
    assert "build_review" in source
    assert "stage_candidate_exists" in source
