"""Read-only, content-bound projection of the two native Veda selectors.

This authenticates the connected operator, not trained quality, speed or a
completed sample. Unknown owners remain executable and nonportable. Derived
tile/head caches are checked, never silently discarded from a live MODEL.
"""

from dataclasses import fields
import hashlib
import json
from pathlib import Path

import torch
from comfy.model_patcher import ModelPatcher

from . import h3_core_compat, veda_flex, veda_heuristic, veda_heuristic_runtime, veda_runtime
from .long_video_dual_identity import content_identity
from .patch_stack_policy import UnverifiedModelStack
from .vdn_attention_compat import _factory_closure
from .veda_vendor.veda import bundle, mask, plan, predictor, tiling
from .veda_vendor import tile_gather_triton


def _require(condition, message):
    if not condition:
        raise UnverifiedModelStack(message)


def _selector_target(selector):
    for factory in (getattr(ModelPatcher, "set_model_optimized_attention", None),
                    h3_core_compat._set_legacy_attention_backend):
        state = _factory_closure(selector, factory, "optimized_attention_override")
        if state is not None:
            target = state.get("optimized_attention")
            _require(getattr(selector, "container_function", None)
                     is getattr(target, "container_function", None),
                     "Veda selector has another container owner")
            return target
    return None


def _equal_layout(actual, expected, device):
    _require(type(actual) is type(expected) and set(vars(actual)) == set(vars(expected)),
             "Veda derived tile cache has an unknown owner or state")
    for name, reference in vars(expected).items():
        value = getattr(actual, name)
        if isinstance(reference, torch.Tensor):
            equal = (type(value) is torch.Tensor and value.dtype == reference.dtype
                     and value.shape == reference.shape and value.layout == torch.strided
                     and value.device == device and value.device.type != "meta"
                     and torch.equal(value.detach().cpu(), reference))
        else:
            equal = type(value) is type(reference) and value == reference
        if not equal:
            raise ValueError(f"Veda derived tile cache changed: {name}")


def _validate_cache(cache, *, trained):
    _require(type(cache) is dict, "Veda tile cache must be the native derived map")
    for key, layout in cache.items():
        _require(type(key) is tuple and len(key) == 5 and type(key[-1]) is torch.device,
                 "Veda tile cache key lacks its native geometry")
        if trained:
            shape, sequence, start, grid, _device = key
            _require(type(shape) is tiling.TileShape and type(sequence) is int
                     and type(start) is int and type(grid) is tuple,
                     "Veda trained tile-cache geometry has another owner")
            expected = tiling.build_tile_layout(
                [tiling.TiledSpan(start, grid, shape)], sequence, sequence, "cpu")
        else:
            shape, grid, start, sequence, _device = key
            _require(type(shape) is type(grid) is tuple and type(sequence) is type(start) is int,
                     "Veda heuristic tile-cache geometry has another owner")
            expected = veda_heuristic.build_layout(grid, start, sequence, shape, "cpu")
        _equal_layout(layout, expected, key[-1])


def _loaded_descriptor(loaded, reference):
    _require(type(loaded) is bundle.Bundle and type(reference) is veda_runtime.VedaBundleRef
             and set(vars(loaded)) == {field.name for field in fields(bundle.Bundle)},
             "Veda loaded bundle has another executable owner")
    if (veda_runtime._sha256(reference.path) != reference.sha256
            or bundle.read_metadata(str(reference.path)) != reference.metadata
            or loaded.metadata != reference.metadata):
        raise ValueError("Veda bundle file or bound metadata changed")
    metadata = reference.metadata
    _require(type(loaded.plans) is plan.PlanTable and set(vars(loaded.plans)) == {"plans"},
             "Veda plan table has another owner")
    stored_plans = json.loads(metadata["plans"])
    live_plans = {}
    for name, selected in loaded.plans.plans.items():
        _require(type(name) is str and type(selected) is plan.TilePlan
                 and set(vars(selected)) == {"geometry", "grid", "shapes", "head_shape", "provenance", "_groups"},
                 "Veda selected plan has another executable owner")
        live_plans[name] = selected.to_json()
        rebuilt = plan.TilePlan.from_json(selected.to_json())
        for key, groups in selected._groups.items():
            _require(type(key) is tuple and len(key) == 2 and type(key[0]) is int
                     and type(key[1]) is str and 0 <= key[0] < selected.num_layers
                     and type(groups) is list, "Veda head-group cache has an unknown key")
            expected = rebuilt.head_groups(key[0], "cpu")
            if len(groups) != len(expected):
                raise ValueError("Veda cached head-group count changed")
            for actual, wanted in zip(groups, expected, strict=True):
                _require(type(actual) is plan.HeadGroup and set(vars(actual)) == {"shape", "heads"},
                         "Veda cached head group has another owner")
                if (actual.shape != wanted.shape or type(actual.heads) is not torch.Tensor
                        or actual.heads.dtype != wanted.heads.dtype
                        or actual.heads.device != torch.device(key[1])
                        or not torch.equal(actual.heads.detach().cpu(), wanted.heads)):
                    raise ValueError("Veda cached head assignment changed")
    if live_plans != stored_plans or loaded.keep_ratio != float(metadata["keep_ratio"]):
        raise ValueError("Veda loaded tile plans differ from the bound bundle")
    network = loaded.predictor
    layers, heads, dim = (int(metadata[key]) for key in ("num_layers", "num_heads", "head_dim"))
    _require(type(network) is predictor.TileScorePredictor and type(network.layers) is torch.nn.ModuleList
             and len(network.layers) == layers, "Veda predictor has another executable owner")
    for child in network.modules():
        _require(type(child) in (predictor.TileScorePredictor, predictor.LayerPredictor, torch.nn.ModuleList)
                 and not child.training and not child._buffers
                 and not child._forward_hooks and not child._forward_pre_hooks
                 and not child._backward_hooks and not child._backward_pre_hooks
                 and not any(name in vars(child) for name in ("forward", "embed", "scores")),
                 "Veda predictor contains an unverified callable or hook")
    for layer in network.layers:
        _require(layer.head_dim == dim and set(layer._parameters) == {"proj_q", "proj_k"}
                 and layer.proj_q.shape == layer.proj_k.shape == (heads, dim * 3, dim),
                 "Veda predictor parameters differ from its native format")
    return {"bundle_sha256": reference.sha256, "metadata": content_identity(metadata),
            "predictor_state": content_identity(network.state_dict()),
            "live_plans": content_identity(live_plans)}


def project(model):
    """Remove only an authenticated selector on a clone used for inspection."""
    selector = model.model_options.get("transformer_options", {}).get("optimized_attention_override")
    target = _selector_target(selector)
    trained = _factory_closure(target, veda_runtime.apply_veda, "route")
    heuristic = _factory_closure(target, veda_heuristic_runtime.apply_heuristic, "route")
    if trained is None and heuristic is None:
        return model, None
    state = trained if trained is not None else heuristic
    factory = veda_runtime.apply_veda if trained is not None else veda_heuristic_runtime.apply_heuristic
    runtime_type = veda_runtime.VedaRuntime if trained is not None else veda_heuristic_runtime.HeuristicRuntime
    runtime = state.get("runtime")
    _require(type(runtime) is runtime_type and set(vars(runtime)) == {field.name for field in fields(runtime_type)}
             and runtime.bypass_reason is None, "Veda selector/runtime ownership is unverified")
    delegate = _factory_closure(state.get("delegate"), factory, "delegate")
    _require(delegate is not None and set(delegate) == {"previous"},
             "Veda dense delegate has another callable owner")
    previous = delegate["previous"]
    previous_backend = None if previous is None else h3_core_compat.plain_attention_backend(previous)
    _require(previous is None or previous_backend is not None,
             "Veda previous backend is unverified; retain execution-local identity")
    if runtime.previous_backend != previous_backend or runtime.failure is not None:
        raise ValueError("Veda bound backend or execution failure changed")
    names = ("mode", "max_copy_mib", "fused_tile_io") if trained is not None else (
        "mode", "max_copy_mib", "min_tokens", "pool_mode", "sigma_start", "sigma_end", "sink_conditioning")
    if any(state.get(name) != getattr(runtime, name) for name in names):
        raise ValueError("Veda captured numerical settings differ from its runtime")
    _validate_cache(runtime.tile_cache, trained=trained is not None)
    compiled = veda_flex._COMPILED
    _require(compiled is None or getattr(compiled, "_torchdynamo_orig_callable", None)
             is veda_flex.flex_attention, "Veda compiled Flex kernel has another callable owner")
    if trained is not None:
        if state.get("loaded") is not runtime.loaded or state.get("keep") != runtime.keep_ratio:
            raise ValueError("Veda loaded predictor or keep ratio changed")
        configuration = {name: getattr(runtime, name) for name in names}
        configuration["keep_ratio"] = runtime.keep_ratio
        configuration["bundle"] = _loaded_descriptor(runtime.loaded, runtime.bundle)
        _require(veda_runtime.attend is veda_flex.attend,
                 "Veda trained attention has another callable owner")
        algorithm = "trained_tile128_flex"
    else:
        shapes = state.get("shapes")
        _require(type(shapes) is tuple and all(type(shape) is tuple for shape in shapes),
                 "Veda heuristic tile shapes have another owner")
        if runtime.preset != "custom" and shapes != veda_heuristic.parse_shapes(runtime.preset):
            raise ValueError("Veda heuristic preset and captured shapes differ")
        configuration = {name: getattr(runtime, name) for name in names}
        configuration.update(keep_ratio=runtime.keep_ratio, shapes=shapes,
                             start_percent=runtime.start_percent, end_percent=runtime.end_percent)
        _require(veda_heuristic_runtime.attend is veda_heuristic.attend,
                 "Veda heuristic attention has another callable owner")
        algorithm = "model_free_tripool64_flex"
    modules = (h3_core_compat, veda_runtime, veda_heuristic_runtime, veda_heuristic,
               veda_flex, bundle, mask, plan, predictor, tiling, tile_gather_triton)
    root = Path(__file__).resolve().parent
    implementation = {Path(module.__file__).resolve().relative_to(root).as_posix():
                      hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for module in modules}
    implementation[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    view = model.clone()
    if previous is None:
        view.model_options["transformer_options"].pop("optimized_attention_override")
    else:
        view.model_options["transformer_options"]["optimized_attention_override"] = previous
    return view, {"schema": "t8.veda.selected-operator-content.v1", "algorithm": algorithm,
                  "configuration": content_identity(configuration), "implementation": implementation,
                  "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                  "boundary": "Operator content only; completion/media/quality/speed require separate evidence"}
