"""Keep the Markdown-only browser scope separate from the earlier 51 graphs."""

from tools.serve_modular_m0_legacy_browser import selected as registered_selected
from tools.serve_modular_m0_markdown_browser import selected as markdown_selected


def test_frozen_markdown_only_scope_is_disjoint_and_has_stable_private_names():
    registered = registered_selected()
    markdown = markdown_selected()
    assert len(registered) == 51
    assert len(markdown) == 236
    assert not {case["relative"] for case in registered} & {
        case["relative"] for case in markdown}
    assert [case["index"] for case in markdown] == list(range(1, 237))
    assert len({case["copy_name"] for case in markdown}) == 236
    assert len({case["saved_name"] for case in markdown}) == 236
    assert all(case["copy_name"].startswith("M0M_") and
               case["saved_name"].startswith("QA_M0M_") for case in markdown)
