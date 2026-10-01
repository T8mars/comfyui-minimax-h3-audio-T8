"""Do not silently include missing providers or double-count qualified groups."""
from tools.serve_modular_m0_external_browser import PROVIDERS, selected
from tools.serve_modular_m0_embedded_browser import selected as embedded
from tools.serve_modular_m0_legacy_browser import selected as flat
from tools.serve_modular_m0_markdown_browser import selected as markdown


def test_external_scope_contains_exactly_nine_new_frozen_graphs():
    cases = selected()
    earlier = {case["relative"] for case in flat() + markdown() + embedded()}
    assert len(cases) == 9
    assert not earlier & {case["relative"] for case in cases}
    assert all(case["copy_name"].startswith("M0X_") and
               case["saved_name"].startswith("QA_M0X_") for case in cases)
    assert PROVIDERS == ("LanPaint", "ComfyUI-sol-attn", "comfyui-minimax-h3-blockcache-T8", "ComfyUI-MiniMaxH3")
    assert not any("RAVEN" in case["relative"] or "接线修正版" in case["relative"] for case in cases)
