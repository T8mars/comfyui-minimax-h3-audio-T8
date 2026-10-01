"""Debug-only variants authenticate; every executable mutation still fails."""
from types import CodeType

import pytest

from h3_audio_t8_pkg.source_code_identity import executable_code_equal


MODULE = compile('''def outer(x):
    def inner(y):
        return y + 2
    try:
        return inner(x)
    except TypeError:
        return None
''', '<exact-source-owner>', 'exec', dont_inherit=True)
BASE = next(value for value in MODULE.co_consts if type(value) is CodeType)


def without_ranges(code):
    return code.replace(co_linetable=b'', co_consts=tuple(
        without_ranges(value) if type(value) is CodeType else value for value in code.co_consts))


def test_same_source_with_recursive_debug_range_difference_only_is_accepted():
    changed = without_ranges(BASE)
    assert BASE != changed and BASE.co_code == changed.co_code
    assert executable_code_equal(BASE, changed)
    assert executable_code_equal(changed, BASE)


@pytest.mark.parametrize('field,value', [
    ('co_code', b''), ('co_argcount', 0), ('co_posonlyargcount', 1),
    ('co_kwonlyargcount', 1), ('co_stacksize', BASE.co_stacksize + 1),
    ('co_flags', BASE.co_flags ^ 0x100), ('co_names', BASE.co_names + ('foreign',)),
    ('co_varnames', ('renamed_x', *BASE.co_varnames[1:])),
    ('co_filename', '<foreign-source>'), ('co_name', 'foreign_outer'),
    ('co_qualname', 'foreign_owner.outer'), ('co_firstlineno', BASE.co_firstlineno + 1),
    ('co_exceptiontable', b'changed'), ('co_freevars', ('foreign_cell',)),
])
def test_any_non_debug_code_field_change_is_rejected_even_with_empty_debug_ranges(field, value):
    changed = without_ranges(BASE).replace(**{field: value})
    assert not executable_code_equal(BASE, changed)


def test_nested_executable_change_is_not_normalized_with_debug_tables():
    changed = without_ranges(BASE)
    values = list(changed.co_consts)
    index = next(index for index, value in enumerate(values) if type(value) is CodeType)
    child = values[index]
    values[index] = child.replace(co_consts=tuple(3 if type(value) is int and value == 2 else value
                                                 for value in child.co_consts))
    assert not executable_code_equal(BASE, changed.replace(co_consts=tuple(values)))


@pytest.mark.parametrize('first,second', [(1, True), (1, 1.), (0., -0.), (b'1', '1'), (2., 2j)])
def test_typed_or_signed_constant_mutations_do_not_share_authentication(first, second):
    left = BASE.replace(co_consts=(first,))
    right = BASE.replace(co_consts=(second,))
    assert not executable_code_equal(left, right)


def test_unknown_constant_equality_is_never_executed():
    class Foreign:
        def __eq__(self, other):
            pytest.fail('User constant equality was executed')
    foreign = BASE.replace(co_consts=(Foreign(),))
    assert not executable_code_equal(foreign, foreign)
    assert not executable_code_equal(BASE, object())
