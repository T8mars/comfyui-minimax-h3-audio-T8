from pathlib import Path

# Keep the public module namespace unchanged while storing implementation files
# below h3_t8. Search it first so leftovers from an older install cannot win.
_package_root = Path(__file__).resolve().parent
_runtime_root = _package_root / "h3_t8"

if __package__:
    __path__ = [str(_runtime_root), str(_package_root)]
    from .nodes import MiniMaxH3AudioT8Extension as _BaseExtension
    from .hyperflow_long_video_exp.nodes import MiniMaxH3HyperFlowLongVideoEXPT8 as _HyperFlowLongVideoNode
    from .hyperflow_long_video_exp.single8_node import MiniMaxH3HyperFlowSingle8LongVideoEXPT8 as _HyperFlowSingle8LongVideoNode
    from .nodes_hyper_vae_2x import HYPER_VAE_2X_NODE_CLASSES as _hyper_vae_2x_node_classes
    from .nodes_semantic_bridge import SEMANTIC_BRIDGE_EXTRA_NODE_CLASSES as _semantic_bridge_extra_node_classes
else:  # Allows direct test collection from a hyphenated custom-node directory.
    import importlib.util
    import sys
    import types

    _package_name = "_minimax_h3_audio_t8_direct"
    _package = types.ModuleType(_package_name)
    _package.__path__ = [str(_runtime_root), str(_package_root)]
    sys.modules.setdefault(_package_name, _package)
    _spec = importlib.util.spec_from_file_location(f"{_package_name}.nodes", _runtime_root / "nodes.py")
    _nodes = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _nodes
    assert _spec.loader is not None
    _spec.loader.exec_module(_nodes)
    _BaseExtension = _nodes.MiniMaxH3AudioT8Extension
    from importlib import import_module
    _HyperFlowLongVideoNode = import_module(
        f"{_package_name}.hyperflow_long_video_exp.nodes"
    ).MiniMaxH3HyperFlowLongVideoEXPT8
    _HyperFlowSingle8LongVideoNode = import_module(
        f"{_package_name}.hyperflow_long_video_exp.single8_node"
    ).MiniMaxH3HyperFlowSingle8LongVideoEXPT8
    _hyper_vae_2x_node_classes = import_module(
        f"{_package_name}.nodes_hyper_vae_2x"
    ).HYPER_VAE_2X_NODE_CLASSES
    _semantic_bridge_extra_node_classes = import_module(
        f"{_package_name}.nodes_semantic_bridge"
    ).SEMANTIC_BRIDGE_EXTRA_NODE_CLASSES


class _HyperFlowLongVideoExtension(_BaseExtension):
    async def get_node_list(self):
        return [*(await super().get_node_list()), _HyperFlowLongVideoNode,
                _HyperFlowSingle8LongVideoNode, *_hyper_vae_2x_node_classes,
                *_semantic_bridge_extra_node_classes]


def comfy_entrypoint():
    return _HyperFlowLongVideoExtension()


WEB_DIRECTORY = "./web"


__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
