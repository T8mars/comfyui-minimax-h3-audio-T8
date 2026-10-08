"""Inspect Sol's actual lookup/derived caches without changing global state."""
import inspect
from types import FunctionType

import torch
from comfy.ldm.minimax.model import PackedLayout

from .patch_stack_policy import UnverifiedModelStack


def _require(condition, message):
    if not condition:
        raise UnverifiedModelStack("RES Sol cache " + message)


def _pair(actual, expected):
    _require(type(actual) is tuple and len(actual) == 2, "permutation pair is malformed")
    for item, reference in zip(actual, expected, strict=True):
        _require(type(item) is torch.Tensor and item.dtype == torch.int64
                 and item.shape == reference.shape and item.is_contiguous()
                 and torch.equal(item.detach().cpu(), reference), "permutation content differs from the native derivation")


def _inspect_sol_caches(sol):
    """Verify lookup metadata and native-derived permutations, not every layout.

    Existing cached lookup addresses and unrelated prior layouts are deliberately
    excluded from the portable digest. Their invariants are checked, not erased.
    Full positional coordinates must separately bind the actual condition payload.
    """
    _require(all(type(getattr(sol, name)) is dict for name in (
        "_SPANS", "_PERM_CACHE", "_DEVICE_CACHE", "_SOL_KERNEL_PARAMETERS")), "cache owner changed")
    for key, entry in sol._SPANS.items():
        _require(type(key) is int and type(entry) is tuple and len(entry) == 4, "layout entry is malformed")
        layout, bounds, audio, span = entry
        _require(type(layout) is PackedLayout and key == id(layout.position_ids), "layout lookup points to a foreign owner")
        signature = layout.signature
        _require(type(signature) is tuple and len(signature) == 5
                 and all(type(value) is int and value >= 0 for value in signature), "layout signature changed")
        _require(type(layout.segments) is list and layout.segments
                 and layout.segments[-1][2] == "video", "layout segment table is not native")
        offset = 0
        for start, stop, kind in layout.segments:
            _require(type(start) is int and type(stop) is int and start == offset and stop >= start
                     and kind in ("text", "cond", "cond_audio", "ref_img", "ref_audio", "audio", "video"),
                     "layout segments no longer form the actual contiguous sequence")
            offset = stop
        expected_bounds = next(((a, b) for a, b, kind in layout.segments if kind == "video"), None)
        expected_audio = next(((a, b) for a, b, kind in layout.segments if kind == "audio"), None)
        grid = (signature[1], signature[2] // 2, signature[3] // 2)
        expected_span = ((*expected_bounds, grid)
                         if grid[0] * grid[1] * grid[2] == expected_bounds[1] - expected_bounds[0] else None)
        _require(bounds == expected_bounds and audio == expected_audio and span == expected_span,
                 "cached sink/video/Morton bounds disagree with their actual layout")
        _require(type(layout.position_ids) is torch.Tensor and layout.position_ids.dtype == torch.float64
                 and layout.position_ids.shape == (offset, 3) and layout.seq_len == offset,
                 "cached layout position rows do not match their sequence")
    # Only an authenticated pure CPU helper is evaluated on a private empty
    # cache. No original cache write, model forward, or attention launch occurs.
    isolated = dict(vars(sol), _PERM_CACHE={})
    native_perm = FunctionType(sol.morton_perm.__code__, isolated, argdefs=sol.morton_perm.__defaults__)
    expected = {}

    def reference(grid, curve):
        _require(type(grid) is tuple and len(grid) == 3
                 and all(type(value) is int and value > 0 for value in grid)
                 and curve in ("3d", "2d_frame"), "Morton cache key is malformed")
        rows = grid[0] * grid[1] * grid[2]
        _require(rows <= 1048576, "permutation exceeds this bounded identity adapter; not a sampling prohibition")
        key = grid, curve
        if key not in expected:
            expected[key] = native_perm(grid, torch.device("cpu"), curve)
        return expected[key]

    for key, value in sol._PERM_CACHE.items():
        _require(type(key) is tuple and len(key) == 2, "Morton key changed")
        _pair(value, reference(*key))
    for key, value in sol._DEVICE_CACHE.items():
        _require(type(key) is tuple and len(key) == 4 and type(key[2]) is str
                 and type(key[3]) is int and 0 <= key[3] < sol.BLOCK_SIZE, "device permutation key changed")
        perm, inverse = reference(key[0], key[1])
        if key[3]:
            perm = torch.roll(perm, key[3])
            inverse = torch.argsort(perm)
        _pair(value, (perm, inverse))
        _require(all(str(item.device) == key[2] for item in value), "device cache tensors belong to another device")
    kernel = sol._ck.sol_attn
    key = id(kernel)
    if key in sol._SOL_KERNEL_PARAMETERS:
        params = inspect.signature(kernel).parameters.values()
        expected_params = frozenset(parameter.name for parameter in params
            if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY))
        _require(sol._SOL_KERNEL_PARAMETERS[key] == expected_params, "selected kernel signature cache is stale or modified")
    return {"lookup_metadata_verified": True, "derived_permutation_content_verified": True,
            "selected_kernel_parameter_cache_verified": True,
            "full_position_coordinates_verified": False,
            "cache_addresses_in_portable_digest": False, "actual_global_caches_changed": False,
            "attention_kernel_called": False, "permutation_CPU_check_limit": 1048576}


def inspect_sol_caches(sol):
    """A malformed/foreign cache does not authorize deleting it or banning runs."""
    try:
        return _inspect_sol_caches(sol)
    except UnverifiedModelStack:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise UnverifiedModelStack("RES Sol cache identity unavailable: " + str(error)) from error
