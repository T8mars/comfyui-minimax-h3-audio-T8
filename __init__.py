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
    from .modular_sampling import node_classes as _modular_node_classes
    from .nodes_hyper_vae_2x import HYPER_VAE_2X_NODE_CLASSES as _hyper_vae_2x_node_classes
    from .modular_sampling.audio_refine_effect_nodes import NODES as _audio_refine_effect_node_classes
    from .modular_sampling.ltx_rgb_source_nodes import NODES as _ltx_rgb_source_node_classes
    from .modular_sampling.video_io_nodes import NODES as _serial_video_io_node_classes
    from .modular_sampling.ltx_effect_nodes import NODES as _ltx_effect_node_classes
    from .modular_sampling.ltx_relay_nodes import NODES as _ltx_relay_node_classes
    from .nodes_veda_sparse_exp import VEDA_SPARSE_NODE_CLASSES as _veda_sparse_node_classes
    from .nodes_veda_heuristic_exp import VEDA_HEURISTIC_NODE_CLASSES as _veda_heuristic_node_classes
    from .modular_sampling.prepared_ltx_effects_nodes import NODES as _prepared_ltx_effect_node_classes
    from .modular_sampling.prepared_ltx_relay_cache_nodes import NODES as _prepared_ltx_relay_cache_node_classes
    from .nodes_semantic_bridge import SEMANTIC_BRIDGE_EXTRA_NODE_CLASSES as _semantic_bridge_extra_node_classes
    from .modular_sampling.ltx_load_policy_nodes import NODES as _ltx_load_policy_node_classes
    from .modular_sampling.face_source_nodes import NODES as _face_source_node_classes
    from .modular_sampling.rf_audio_clock_nodes import NODES as _rf_audio_clock_node_classes
    from .nodes_h3_fun_union2 import NODES as _union2_node_classes
    from .nodes_mv_cast_solo import NODES as _mv_cast_node_classes
    from .nodes_external_continuation import NODES as _external_continuation_node_classes
    from .nodes_visible_face_mask import NODES as _visible_face_mask_node_classes
    from .nodes_face_observations import NODES as _face_observation_node_classes
    from .nodes_external_continuation_effects import NODES as _external_effect_node_classes
    from .nodes_hyperflow_curve_exp import NODES as _curve_node_classes
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
    _modular_node_classes = import_module(f"{_package_name}.modular_sampling").node_classes
    _hyper_vae_2x_node_classes = import_module(
        f"{_package_name}.nodes_hyper_vae_2x"
    ).HYPER_VAE_2X_NODE_CLASSES
    _audio_refine_effect_node_classes = import_module(
        f"{_package_name}.modular_sampling.audio_refine_effect_nodes"
    ).NODES
    _ltx_rgb_source_node_classes = import_module(
        f"{_package_name}.modular_sampling.ltx_rgb_source_nodes"
    ).NODES
    _serial_video_io_node_classes = import_module(
        f"{_package_name}.modular_sampling.video_io_nodes"
    ).NODES
    _ltx_effect_node_classes = import_module(
        f"{_package_name}.modular_sampling.ltx_effect_nodes"
    ).NODES
    _ltx_relay_node_classes = import_module(
        f"{_package_name}.modular_sampling.ltx_relay_nodes"
    ).NODES
    _veda_sparse_node_classes = import_module(
        f"{_package_name}.nodes_veda_sparse_exp"
    ).VEDA_SPARSE_NODE_CLASSES
    _veda_heuristic_node_classes = import_module(
        f"{_package_name}.nodes_veda_heuristic_exp"
    ).VEDA_HEURISTIC_NODE_CLASSES
    _prepared_ltx_effect_node_classes = import_module(
        f"{_package_name}.modular_sampling.prepared_ltx_effects_nodes"
    ).NODES
    _prepared_ltx_relay_cache_node_classes = import_module(
        f"{_package_name}.modular_sampling.prepared_ltx_relay_cache_nodes"
    ).NODES
    _semantic_bridge_extra_node_classes = import_module(
        f"{_package_name}.nodes_semantic_bridge"
    ).SEMANTIC_BRIDGE_EXTRA_NODE_CLASSES
    _ltx_load_policy_node_classes = import_module(
        f"{_package_name}.modular_sampling.ltx_load_policy_nodes"
    ).NODES
    _face_source_node_classes = import_module(
        f"{_package_name}.modular_sampling.face_source_nodes"
    ).NODES
    _rf_audio_clock_node_classes = import_module(
        f"{_package_name}.modular_sampling.rf_audio_clock_nodes"
    ).NODES
    _union2_node_classes = import_module(f"{_package_name}.nodes_h3_fun_union2").NODES
    _mv_cast_node_classes = import_module(f"{_package_name}.nodes_mv_cast_solo").NODES
    _external_continuation_node_classes = import_module(f"{_package_name}.nodes_external_continuation").NODES
    _visible_face_mask_node_classes = import_module(f"{_package_name}.nodes_visible_face_mask").NODES
    _face_observation_node_classes = import_module(f"{_package_name}.nodes_face_observations").NODES
    _external_effect_node_classes = import_module(f"{_package_name}.nodes_external_continuation_effects").NODES
    _curve_node_classes = import_module(f"{_package_name}.nodes_hyperflow_curve_exp").NODES


class _HyperFlowLongVideoExtension(_BaseExtension):
    async def get_node_list(self):
        return [*(await super().get_node_list()), _HyperFlowLongVideoNode,
                _HyperFlowSingle8LongVideoNode]


class _ModularSamplingExtension(_HyperFlowLongVideoExtension):
    async def get_node_list(self):
        # Keep the complete live legacy prefix, including root-level additions.
        return [*(await super().get_node_list()), *_modular_node_classes(),
                *_hyper_vae_2x_node_classes, *_audio_refine_effect_node_classes,
                *_ltx_rgb_source_node_classes, *_serial_video_io_node_classes,
                *_ltx_effect_node_classes, *_ltx_relay_node_classes,
                *_veda_sparse_node_classes, *_veda_heuristic_node_classes,
                *_prepared_ltx_effect_node_classes, *_prepared_ltx_relay_cache_node_classes,
                *_semantic_bridge_extra_node_classes, *_ltx_load_policy_node_classes,
                *_face_source_node_classes, *_rf_audio_clock_node_classes, *_union2_node_classes,
                *_mv_cast_node_classes, *_external_continuation_node_classes,
                *_visible_face_mask_node_classes, *_face_observation_node_classes,
                *_external_effect_node_classes, *_curve_node_classes]


def comfy_entrypoint():
    return _ModularSamplingExtension()


WEB_DIRECTORY = "./web"


__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
