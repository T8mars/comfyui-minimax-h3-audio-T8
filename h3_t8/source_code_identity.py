"""Exact executable source comparison, excluding only debug line/column tables.

Identical .pyc source can have different debug ranges from a fresh compile.
Instructions, typed constants, exception tables, flags, source locations,
argument layouts and closure names remain exact. Never execute live constants.
"""
import struct
from types import CodeType


_FIELDS = (
    "co_argcount", "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals",
    "co_stacksize", "co_flags", "co_code", "co_names", "co_varnames",
    "co_filename", "co_name", "co_qualname", "co_firstlineno",
    "co_exceptiontable", "co_freevars", "co_cellvars",
)


def _constant_key(value):
    kind = type(value)
    if kind is CodeType:
        return ("code", _code_key(value))
    if kind is tuple:
        return ("tuple", tuple(_constant_key(item) for item in value))
    if kind is frozenset:
        return ("frozenset", frozenset(_constant_key(item) for item in value))
    if kind is float:
        return ("float", struct.pack(">d", value))
    if kind is complex:
        return ("complex", struct.pack(">dd", value.real, value.imag))
    if value is None or value is Ellipsis or kind in (bool, int, str, bytes):
        return (kind.__name__, value)
    raise ValueError("Unknown code constant cannot be source authenticated")


def _code_key(code):
    return (tuple(getattr(code, field, None) for field in _FIELDS),
            tuple(_constant_key(value) for value in code.co_consts))


def executable_code_equal(left, right):
    if type(left) is not CodeType or type(right) is not CodeType:
        return False
    try:
        return _code_key(left) == _code_key(right)
    except (ValueError, TypeError):
        return False
