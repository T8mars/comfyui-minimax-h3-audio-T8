"""Fresh native pruned H3 loading with declared FP32 precision islands.

No loaded MODEL is copied, repaired or mutated. All layer classes, quantized
loaders, forwards, LoRA handling and patcher lifecycle remain native Core.
The constructor policy is only selected by the explicit Curve Base Loader.
"""
from pathlib import Path

import torch
from safetensors import safe_open

from . import hyperflow_curve_fit_exp as fitting

ISLANDS = fitting.PROJECTIONS + (
    "video_patch_proj", "audio_patch_proj", "final_layer.video_out", "final_layer.audio_out",
)


def precision_identity(model):
    """Bind persistent native dtype hints, never volatile loaded/pinned handles."""
    modules = dict(model.model.diffusion_model.named_modules())
    result = {}
    for name in ISLANDS:
        module = modules[name]
        factory = getattr(module, "factory_kwargs", {})
        result[name] = {"factory_dtype": str(factory.get("dtype")),
                        "weight_hint": str(getattr(module, "weight_comfy_model_dtype", None)),
                        "bias_hint": str(getattr(module, "bias_comfy_model_dtype", None))}
    return result


def _preflight(path):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        prefix = fitting._prefix(handle)
        keys = set(handle.keys())
        if handle.get_slice(prefix + "adaln_t_table").get_shape() != [1025, 8]:
            raise ValueError("Curve loader requires the native 1025x8 pruned H3 table")
        for name in ISLANDS:
            weight, bias = prefix + name + ".weight", prefix + name + ".bias"
            if weight not in keys or bias not in keys or prefix + name + ".comfy_quant" in keys:
                raise ValueError("Curve precision islands must be complete unquantized native layers: " + name)
            w, b = handle.get_slice(weight), handle.get_slice(bias)
            if (w.get_dtype() not in {"F16", "BF16", "F32"} or b.get_dtype() not in {"F16", "BF16", "F32"}
                    or len(w.get_shape()) != 2 or b.get_shape() != [w.get_shape()[0]]):
                raise ValueError("Invalid plain curve precision island: " + name)
            if name in fitting.PROJECTIONS and w.get_shape()[1] != 8:
                raise ValueError("Curve loader requires the eight-coordinate native pruned basis")
    return prefix


def precision_operations(native):
    """Construct stock layer instances; never monkeypatch a Core class.

    MixedPrecisionOps currently ignores a Linear's declared dtype. Configure
    its fresh factory BEFORE state loading. Ordinary/quantized layers retain
    the original factory and disabled-kernel policy. FP32 hints also preserve
    native dynamic geometry when assign=True keeps the disk bias storage.
    """
    if not hasattr(native, "_compute_dtype"):
        return native

    class CurvePrecisionOperations(native):
        @staticmethod
        def Linear(in_features, out_features, bias=True, device=None, dtype=None):
            layer = native.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
            if dtype == torch.float32:
                layer.factory_kwargs = {"device": device, "dtype": torch.float32}
                layer.weight_comfy_model_dtype = torch.float32
                layer.bias_comfy_model_dtype = torch.float32
                if bias:
                    layer.bias = torch.nn.Parameter(torch.empty(out_features, device=device, dtype=torch.float32),
                                                    requires_grad=False)
            return layer

    return CurvePrecisionOperations


def load_curve_base(path, expected_sha256=None, *, disable_dynamic=False):
    import comfy.model_detection
    import comfy.model_management as management
    import comfy.ops
    import comfy.sd
    import comfy.supported_models
    import comfy.utils

    path = Path(path).resolve(strict=True)
    prefix = _preflight(path)  # Reject wrong models before heavy file reads.
    before = fitting._file(path)
    if expected_sha256 is not None and before["sha256"] != expected_sha256:
        raise ValueError("Selected curve base changed before native reconstruction")
    state, metadata = comfy.utils.load_torch_file(str(path), return_metadata=True)
    state = comfy.utils.state_dict_prefix_replace(state, {prefix: ""}, filter_keys=True) if prefix else state
    state, metadata = comfy.utils.convert_old_quants(state, "", metadata=metadata)
    config = comfy.model_detection.model_config_from_unet(state, "", metadata=metadata)
    if (type(config) is not comfy.supported_models.MiniMaxH3 or config.unet_config.get("num_layers") != 50
            or config.unet_config.get("adaln_curve_grid") != 1025 or config.unet_config.get("time_embed_dim") != 8):
        raise ValueError("Curve loader only accepts the native fifty-block pruned H3 config")
    device = management.get_torch_device()
    supported = config.supported_inference_dtypes
    weight_dtype = None if config.quant_config is not None else comfy.utils.weight_dtype(state)
    dtype = management.unet_dtype(model_params=comfy.utils.calculate_parameters(state),
                                  supported_dtypes=supported, weight_dtype=weight_dtype)
    cast = management.unet_manual_cast(None if config.quant_config is not None else dtype, device, supported)
    native = comfy.ops.pick_operations(dtype, cast, load_device=device, model_config=config)
    operations = precision_operations(native)
    result = comfy.sd.load_diffusion_model_state_dict(state, metadata=metadata,
        model_options={"dtype": dtype, "load_device": device, "custom_operations": operations},
        disable_dynamic=disable_dynamic)
    if result is None:
        raise ValueError("Native Core could not load the selected pruned H3")
    stat = path.stat()
    if [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns] != before["stat"]:
        raise ValueError("Curve base changed during native loading")
    # Fresh reconstruction executes this same policy, not the ordinary UNET
    # loader. No warm cache, backup, hook or parameter is inherited.
    result.cached_patcher_init = (load_curve_base, (str(path), before["sha256"]), 0)
    report = {"schema": "t8.hyperflow.curve_native_loader.v1", "base_sha256": before["sha256"],
              "declared_FP32_layers": list(ISLANDS), "native_quantization_preserved": True,
              "fresh_model": True, "fit_checked": False, "quality_accepted": False}
    return result, report
