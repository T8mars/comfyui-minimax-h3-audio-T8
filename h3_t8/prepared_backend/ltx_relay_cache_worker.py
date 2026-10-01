"""Explicit actual native INT8 Gemma/connector cache preparation with block leases.

No resident-SM121 guard bypass: this is a distinct explicit serial SM89 trial,
not a claim that the original upstream resident cache-builder was supported.
"""
import gc
import json
from pathlib import Path
import sys
import time

from backend_files import sha
from ltx_effects_contract import validate_plan


def require(condition, message):
    if not condition:
        raise ValueError(message)

request_path = Path(sys.argv[sys.argv.index("--request") + 1])
request = json.loads(request_path.read_text(encoding="utf8"))
root = request_path.parent
sys.path[:0] = request["isolated_paths"]
report = {"status": "incomplete", "scope": "actual native per-event INT8 text/connector encode only, no sampler"}
gemma = connector = None
try:
    require(request["schema"] == "t8_prepared_ltx_native_relay_encode_v1", "Wrong explicit Relay encoder request")
    validate_plan(request["plan"], request["geometry"], request["prompt"])
    for filename, digest in request["identities"].items():
        require(sha(filename) == digest, "Relay encoder input changed: " + filename)
    import torch
    from safetensors.torch import save_file
    from runtime.cache_builder import _text_builders, prompt_stats
    from runtime.prompt_cache import RECIPE, fingerprint, validate_payload, load_cache
    from runtime.cache_ops.int8_policy import DevInt8ConvRotLinear, _comfy_kitchen
    from ltx_text_weight_offload import offload_native_module
    _comfy_kitchen()
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    require(str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-").lower() == request["gpu_uuid"].removeprefix("GPU-").lower(), "Selected GPU identity changed")
    def progress(stage, **extra):
        value = {"stage": stage, "time": time.time(), **extra}
        (root / "worker-live.json").write_text(json.dumps(value), encoding="utf8")
        print(json.dumps(value), flush=True)
    for name, key in (("text_encoder", "gemma"), ("connector", "base")):
        checkpoint = Path(request[key])
        if checkpoint.name != RECIPE[name] or checkpoint.stat().st_size != RECIPE[name + "_bytes"]:
            raise ValueError("Actual selected checkpoint differs from the prepared-cache provider")
    progress("building_actual_INT8_Gemma_and_connector_on_CPU")
    gemma_builder, connector_builder = _text_builders(Path(request["gemma"]), Path(request["base"]))
    gemma = gemma_builder.build(device=torch.device("cpu"), dtype=None).eval().requires_grad_(False)
    connector = connector_builder.build(device=torch.device("cpu"), dtype=None).eval().requires_grad_(False)
    connector.feature_extractor.to(dtype=torch.bfloat16)
    for model, expected in ((gemma, 328), (connector, 96)):
        layers = [module for module in model.modules() if isinstance(module, DevInt8ConvRotLinear)]
        if len(layers) != expected or any(layer.weight.dtype != torch.int8 or layer.weight_scale.dtype != torch.float32 for layer in layers):
            raise ValueError("Native INT8/FP32-scale provider cardinality or dtype differs")
        if any(value.device.type != "cpu" or value.is_meta for value in (*model.parameters(), *model.buffers())):
            raise ValueError("Prepared text weights must start as actual CPU tensors")
    language = getattr(gemma.model.model, "language_model", gemma.model.model)
    blocks = getattr(language, "layers", None)
    if blocks is None or not len(blocks):
        raise ValueError("Actual Gemma text model has no known serial layer producer")
    connector_blocks = [*connector.video_connector.transformer_1d_blocks,
                        *connector.audio_connector.transformer_1d_blocks]
    entries, phase_reports, encoded_tensors = [], [], {}
    for event in request["plan"]["events"]:
        prompt = event["local_prompt"]
        stats = prompt_stats(gemma.tokenizer, prompt)
        progress("actual_Gemma_event_encoding", event_index=event["event_index"])
        started = time.perf_counter()
        with torch.inference_mode(), offload_native_module(gemma, blocks, device,
                minimum_free_bytes=2 * 1024**3) as text_lease:
            values = gemma.encode([prompt])
            require(len(values) == 1, "Native Gemma returned the wrong event batch")
            hidden_states, mask = values[0]
        require(len(text_lease["block_calls"]) == len(blocks), "Native Gemma layer coverage differs")
        progress("actual_AV_connector_event_projection", event_index=event["event_index"])
        with torch.inference_mode(), offload_native_module(connector, connector_blocks, device,
                minimum_free_bytes=2 * 1024**3) as connector_lease:
            processed = connector.process_hidden_states(hidden_states, mask)
            contexts = {name: getattr(processed, name + "_encoding").detach().cpu().contiguous() for name in ("video", "audio")}
        require(len(connector_lease["block_calls"]) == len(connector_blocks), "Native AV connector layer coverage differs")
        torch.cuda.synchronize()
        payload = {"schema_version": 1, "prompt": prompt, "recipe": dict(RECIPE), "fingerprint": fingerprint(prompt),
            "contexts": contexts, "prompt_stats": stats,
            "tensor_manifest": {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)} for name, tensor in contexts.items()},
            "generation": {"gemma_encode_calls": 1, "connector_process_calls": 1,
                "video_model_load_count": 0, "strategy": "explicit_native_INT8_serial_weight_lease",
                "checkpoint_SHA256": {key: request["identities"][request[key]] for key in ("gemma", "base")}}}
        validate_payload(payload, prompt=prompt, torch_module=torch)
        output = root / f"event-{event['event_index']}.pt"
        with output.open("xb") as stream:
            torch.save(payload, stream)
        reloaded = load_cache(output, prompt=prompt, torch_module=torch)
        require(all(torch.equal(reloaded.payload["contexts"][key], value) for key, value in contexts.items()), "Serialized native event cache differs")
        entries.append({"event_index": event["event_index"], "prompt": prompt, "path": str(output.resolve()), "sha256": sha(output)})
        encoded_tensors.update({f"event_{event['event_index']}_{name}": tensor for name, tensor in contexts.items()})
        phase_reports.append({"event_index": event["event_index"], "prompt_stats": stats,
            "native_Gemma_blocks": len(text_lease["block_calls"]), "native_connector_blocks": len(connector_lease["block_calls"]),
            "seconds": time.perf_counter() - started})
        del values, hidden_states, mask, processed, contexts, reloaded, payload
        gc.collect()
        torch.cuda.empty_cache()
        require(all(value.device.type == "cpu" for model in (gemma, connector) for value in (*model.parameters(), *model.buffers())), "Native text/connector weights were not restored to CPU")
    manifest = root / "relay-caches.json"
    manifest.write_text(json.dumps({"schema": "t8_prepared_ltx_relay_caches_v1", "plan_hash": request["plan"]["plan_hash"],
                                  "caches": entries}, indent=2), encoding="utf8")
    output = root / "encoded-contexts.safetensors"
    save_file(encoded_tensors, str(output), metadata={"scope": "actual independent native post-connector event contexts, no sampled latent"})
    for filename, digest in request["identities"].items():
        require(sha(filename) == digest, "Relay encoder input changed: " + filename)
    report.update(status="native_INT8_event_caches_prepared_not_sampling", output_sha256=sha(output), manifest_sha256=sha(manifest),
        caches=entries, phases=phase_reports, model_sampling=False, event_text_encoder_actual=True,
        original_global_cache_unchanged=True, quality_accepted=False)
except BaseException as error:
    report.update(status="failed", error=f"{type(error).__name__}: {error}")
    raise
finally:
    active_error = sys.exc_info()[0] is not None
    try:
        for model in (gemma, connector):
            if model is not None:
                model.dispose()
                model.to_empty(device="meta")
        gc.collect()
        if "torch" in globals():
            torch.cuda.empty_cache()
    except BaseException as error:
        report["cleanup_error"] = f"{type(error).__name__}: {error}"
        if not active_error:
            report.update(status="failed", error=report["cleanup_error"])
            raise
    finally:
        (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
        print(json.dumps({"status": report["status"]}), flush=True)
