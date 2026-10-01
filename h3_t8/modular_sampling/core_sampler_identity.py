"""Authenticate Core's built-in sampler objects across its two import names.

Core registers ``nodes_custom_sampler`` from a file path, while extensions may
import ``comfy_extras.nodes_custom_sampler``. Both are the same installed file
but Python gives their classes different identities. Only exact registered
built-ins with the original implementation are accepted as portable providers.
"""

from pathlib import Path
import sys

from comfy_extras import nodes_custom_sampler as reference


def _code(value):
    function = getattr(value, "__func__", value)
    return getattr(function, "__code__", None)


def _same_file_and_methods(actual, expected, methods):
    if not isinstance(actual, type) or actual.__qualname__ != expected.__qualname__:
        return False
    try:
        if Path(sys.modules[actual.__module__].__file__).resolve() != \
                Path(sys.modules[expected.__module__].__file__).resolve():
            return False
    except (AttributeError, KeyError, OSError, TypeError):
        return False
    return all(_code(getattr(actual, method, None)) == _code(getattr(expected, method, None))
               and _code(getattr(actual, method, None)) is not None for method in methods)


def exact_registered_provider(registry_name, provider_name, owner_methods, provider_methods):
    """Return a native provider class, or None for an override/unknown stack."""
    nodes = sys.modules.get("nodes")
    if nodes is None or not isinstance(getattr(nodes, "NODE_CLASS_MAPPINGS", None), dict):
        return None
    expected_owner = getattr(reference, registry_name)
    expected_provider = getattr(reference, provider_name)
    owner = nodes.NODE_CLASS_MAPPINGS.get(registry_name)
    if not _same_file_and_methods(owner, expected_owner, owner_methods):
        return None
    module = sys.modules.get(owner.__module__)
    provider = getattr(module, provider_name, None)
    if not _same_file_and_methods(provider, expected_provider, provider_methods):
        return None
    if provider.__bases__ != expected_provider.__bases__:
        return None
    return provider


def native_random_noise_class():
    return exact_registered_provider(
        "RandomNoise", "Noise_RandomNoise", ("define_schema", "execute", "get_noise"),
        ("__init__", "generate_noise"),
    )


def native_basic_guider_class():
    return exact_registered_provider(
        "BasicGuider", "Guider_Basic", ("define_schema", "execute", "get_guider"),
        ("set_conds",),
    )
