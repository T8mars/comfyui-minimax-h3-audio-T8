if __package__:
    from .nodes import comfy_entrypoint
else:  # Allows direct test collection from a hyphenated custom-node directory.
    import importlib.util
    from pathlib import Path
    import sys
    import types

    _package_name = "_minimax_h3_audio_t8_direct"
    _package_root = Path(__file__).resolve().parent
    _package = types.ModuleType(_package_name)
    _package.__path__ = [str(_package_root)]
    sys.modules.setdefault(_package_name, _package)
    _spec = importlib.util.spec_from_file_location(f"{_package_name}.nodes", _package_root / "nodes.py")
    _nodes = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _nodes
    assert _spec.loader is not None
    _spec.loader.exec_module(_nodes)
    comfy_entrypoint = _nodes.comfy_entrypoint


WEB_DIRECTORY = "./web"


__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
