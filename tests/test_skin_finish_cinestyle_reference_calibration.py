from __future__ import annotations

import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import calibrate_skin_finish_cinestyle_reference as reference  # noqa: E402


def test_cinestyle_reference_audit_is_pinned_external_and_low_load():
    source = Path(reference.__file__).read_text(encoding="utf-8")
    assert reference.UPSTREAM_COMMIT == (
        "e7d5facafd95c97190fcf54171960f25c21b3043"
    )
    assert reference.EXPECTED_UPSTREAM_FILE_SHA256 == (
        "1F8D8EBD44FEA4C75A0C77D2798173A525B2CCBFDEAFE60F0C82F74B3CB7FDF6"
    )
    assert '"loads_h3": False' in source
    assert '"loads_sam": False' in source
    assert '"runs_full_video": False' in source
    assert '"stress_or_repeat": False' in source
    assert '"cpu_only": True' in source
    assert '"code_vendored": False' in source
    assert "upstream._preferred_device = lambda _fallback: torch.device(\"cpu\")" in source
    assert "same_exact_t8_mask_for_all_rows" in source
