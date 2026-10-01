"""The M0 source-extension audit must not excuse changes to old node code."""

import pytest

from tools.audit_modular_m0_additive_source import additive_insert


OLD = (b"class Old:\n    def execute(self):\n        return 1\n\n\n"
       b"class Next:\n    pass\n")
NEW_CLASS = b"class Added(Old):\n    pass\n\n\n"


def check(old, current, *, added_json_import=False):
    return additive_insert(old, current, next_class="Next", new_class="Added",
                           parent_class="Old", added_json_import=added_json_import)


def test_only_specified_derived_class_may_be_inserted():
    current = OLD.replace(b"class Next:", NEW_CLASS + b"class Next:")
    assert check(OLD, current) == NEW_CLASS
    with pytest.raises(ValueError, match="Old source bytes changed"):
        check(OLD, current.replace(b"return 1", b"return 2"))
    with pytest.raises(ValueError, match="expected derived node class"):
        check(OLD, current.replace(b"class Added(Old)", b"class Added(Next)"))
    with pytest.raises(ValueError, match="expected derived node class"):
        check(OLD, current.replace(NEW_CLASS, NEW_CLASS + b"value = 3\n"))


def test_unique_json_import_is_the_only_other_allowed_addition():
    old = b"from __future__ import annotations\n\n" + OLD
    current = old.replace(b"class Next:", NEW_CLASS + b"class Next:")
    current = current.replace(b"\n\nclass Old", b"\n\nimport json\nclass Old")
    assert check(old, current, added_json_import=True) == NEW_CLASS
    with pytest.raises(ValueError, match="unique addition"):
        check(old, current + b"import json\n", added_json_import=True)
