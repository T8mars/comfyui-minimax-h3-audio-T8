"""One FreeVideo stage in the explicitly selected isolated Python environment."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time
from contextlib import nullcontext


def atomic_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def main():
    root = Path(sys.argv[1]).resolve(strict=True)
    started = time.monotonic()
    while not (root / "launch-gate").is_file():
        if time.monotonic() - started > 30:
            raise RuntimeError("Owned worker was not assigned to its supervisor")
        time.sleep(.05)
    # -I is deliberate: do not inherit Core's Python search path.
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from freevideo_exp.runtime import digest, verify_files, MODEL_REVISION
    request_path = root / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    split = request.get("split_profile") is not None
    if split:
        from freevideo_exp.split_sampler import PROFILE
        if request["split_profile"] != PROFILE or request["role"] not in ("SPLIT_LOW", "SPLIT_HIGH"):
            raise ValueError("Unsupported explicit FreeVideo split profile")
    if digest(request["config_path"]) != request["config_sha256"]:
        raise ValueError("Runtime config changed after dispatch")
    config = json.loads(Path(request["config_path"]).read_text(encoding="utf-8"))
    if config["model_revision"] != MODEL_REVISION or digest(root / "input.safetensors") != request["input_sha256"]:
        raise ValueError("Worker input/config identity mismatch")
    # Parent verified immutable large weights. Worker independently verifies executable source and metadata.
    source_rows = [row for row in config["files"] if row["kind"] != "weight"]
    verify_files(source_rows)
    sys.path.insert(0, config["source_root"])
    os.environ["FREEVIDEO_VDN_ROOT"] = config["vdn_root"]
    from freevideo_engine.paths import add_vdn
    add_vdn()
    import torch
    from safetensors.torch import load_file, save_file
    from freevideo_engine.locking import runtime_lock
    from freevideo_engine.adaln import schedule_timesteps
    from freevideo_engine.adaln_assets import identity, weight_identity, validate_catalog, asset_path, check_table
    from freevideo_engine.media_conditioning import describe
    from freevideo_engine.hardware import _detect_local
    from freevideo_engine.policy import choose
    from freevideo_engine.gpu_budget import configure, LiveGPUBudget
    from freevideo_engine.runtime import Engine
    import diffusers
    import src.models.hybrid_attention as hybrid_module
    if not Path(diffusers.__file__).resolve().is_relative_to(Path(config["vdn_root"]) / "diffusers/src") or not Path(hybrid_module.__file__).resolve().is_relative_to(Path(config["vdn_root"])):
        raise ValueError("Foreign VDN/Diffusers module won the isolated import")
    torch.set_num_threads(8)
    tensors = load_file(str(root / "input.safetensors"), device="cpu")
    value = dict(prompt_embeds=tensors["embeds"].to(torch.bfloat16), text_token_tags=tensors["tags"],
                 task=request["conditions"]["task"], width=request["geometry"]["width"], height=request["geometry"]["height"])
    if request["conditions"]["keyframes"]:
        value.update(keyframe_anchors=[row["anchor"] for row in request["conditions"]["keyframes"]],
                     condition_latents=[tensors[row["tensor"]].float() for row in request["conditions"]["keyframes"]])
    if request["conditions"]["refs"]:
        value["references"] = []
        for row in request["conditions"]["refs"]:
            ref = dict(kind=row["kind"])
            if "tensor" in row:
                ref["latent"] = tensors[row["tensor"]].float()
            if "audio_tensor" in row:
                ref["audio_latent"] = tensors[row["audio_tensor"]].float()
            value["references"].append(ref)
    canvas = request["geometry"]
    info = describe(value, **canvas)
    audio_reference = info["reference_audio_tokens"] > 0
    if audio_reference and not config.get("audio_reference_cache"):
        raise ValueError("Prepare the explicit owned audio-reference constant tables before using audio references")
    cache = Path(config["audio_reference_cache"] if audio_reference else config["cache"])
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    validate_catalog(manifest, 50)
    times = schedule_timesteps(8, task=info["task"], device="cpu")
    expected = identity(weight_identity(manifest), [t.float().tolist() for t in times], 5376)
    table = next((t for t in manifest.get("adaln_tables", []) if t["identity"] == expected), None)
    if table is None or len(table["files"]) != 50:
        raise ValueError("No complete official eight-step AdaLN table for this task; original projections are not downloaded")
    for row in table["files"]:
        check_table(asset_path(cache, row["file"]), row, expected)
    lora_report = {"enabled": False}
    active_loras = [row for row in request["loras"] if row["strength"] != 0]
    if active_loras:
        from freevideo_engine.lora_cache import index_adapter
        from freevideo_engine.lora_online_cache import _supported, prepare
        verify_files(active_loras)
        targets = {name: dict(shape=spec["weight_shape"]) for name, spec in manifest["linears"].items()}
        if not _supported(active_loras, manifest["linears"]):
            raise ValueError("This FreeVideo entry supports online attention/FF LoRA only, not implicit fused/AdaLN conversion")
        for row in active_loras:
            index_adapter(row, targets)
        with runtime_lock():
            cache, lora_report = prepare(cache, active_loras, output_root=Path(config["home"]) / "loras")
    # Our supported runtime currently has one explicitly probed attention backend.
    probe_path = Path(config["home"]) / "kernel-probes.json"
    probes = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.is_file() else {}
    hardware = _detect_local()
    if any(probes.get(name, {}).get("status") != "complete" or probes[name].get("torch") != str(torch.__version__)
           or tuple(probes[name].get("capability", [])) != tuple(hardware.capability) for name in ("linear", "cudnn")):
        raise ValueError("Run the two pinned FreeVideo linear/cudnn compatibility probes before sampling")
    from freevideo_exp.probe_identity import identity_for, validate_probes
    current_identity = identity_for(config, hardware)
    validate_probes(config, probes, current_identity)
    relay = request.get("effects", {}).get("relay")
    if relay and relay["mode"] == "apply_exp" and len(relay["events"]) > 1:
        from freevideo_exp.masked_probe import validate
        validate(config, current_identity)
    from freevideo_engine.geometry import geometry
    admitted = geometry(**canvas)
    admitted.update({k: info[k] for k in ("reference_video_tokens", "reference_audio_tokens")})
    policy = choose(hardware, attention="cudnn", available_backends={"cudnn"}, canvas=admitted,
                    lora_max_block_bytes=lora_report.get("max_block_bytes", 0), lora_root_bytes=lora_report.get("root_bytes", 0))
    options = dict(policy.engine, steps=8, task=info["task"], canvas=admitted)
    options.update(stream_weights=True, preload_host=False, pin_host_weights=False)
    torch.save(value, root / "conditioning.pt")  # Produced here from safetensors; no user pickle load.
    engine = live = effect_runtime = None
    report = {}
    with runtime_lock():
        try:
            memory = configure(torch, policy.gpu_budget_bytes, reserve_bytes=policy.gpu_system_reserve_bytes)
            if memory.get("windows_allocator_limit_enforced"):
                live = LiveGPUBudget(torch, memory, policy.gpu_budget_bytes, policy.gpu_system_reserve_bytes)
            engine = Engine(cache, base=config["base"], checkpoint=config["checkpoint"], **options)
            from freevideo_exp.worker_effects import install
            effect_runtime = install(engine, request.get("effects", {}), canvas)
            completed = 0
            def step(seconds):
                nonlocal completed
                completed += 1
                atomic_json(root / "progress.json", dict(completed=completed, seconds=seconds))
            initial = (tensors["initial_video"], tensors["initial_audio"]) if request["role"] == "HIGH" else None
            context = nullcontext(None)
            if split:
                from freevideo_exp.split_sampler import install as install_split
                context = install_split(engine, request, tensors)
            with context as captured:
                video, audio, sampled = engine.sample(root / "conditioning.pt", request["seed"], **canvas,
                    initial_latents=initial, refine_steps=request["tail_steps"] if initial is not None else None,
                    step_callback=step, gpu_reserve_bytes=policy.gpu_system_reserve_bytes,
                    budget_refresh=live.refresh if live else None)
            if completed != (4 if split else request["tail_steps"] if initial is not None else 8):
                raise ValueError("FreeVideo did not finish all requested NFE")
            if any(request.get("effects", {}).values()) and (effect_runtime.forwards != completed or effect_runtime.blocks != completed * 50):
                raise ValueError("FreeVideo effects did not observe every actual completed block/NFE")
            out = {"video": video.detach().cpu().contiguous(), "audio": audio.detach().cpu().permute(1, 0, 2).unsqueeze(0).contiguous()}
            split_report = None
            if split:
                if request["role"] == "SPLIT_LOW":
                    out["video_state"] = captured["video_state"].contiguous()
                split_report = {key: value for key, value in captured.items() if key not in ("video_state", "video_x0")}
                if request["role"] == "SPLIT_HIGH":
                    from freevideo_exp.runtime import tensor_record
                    split_report["audio_resume_input"] = tensor_record(tensors["initial_audio"].permute(1, 0, 2).unsqueeze(0).contiguous())
            if not all(bool(torch.isfinite(v).all()) for v in out.values()):
                raise ValueError("FreeVideo returned nonfinite output")
            save_file(out, str(root / "output.safetensors"))
            report = dict(request_sha256=digest(request_path), output_sha256=digest(root / "output.safetensors"),
                          sample=sampled, engine=engine.config, load_seconds=engine.load_seconds,
                          policy=policy.to_dict(), memory=memory, lora=lora_report, effects=effect_runtime.snapshot(),
                          conditioning=info, torch=str(torch.__version__), cuda=torch.version.cuda,
                          python=sys.version, source_inventory_sha256=config["inventory_sha256"])
            if split:
                report["split"] = split_report
        finally:
            if effect_runtime:
                effect_runtime.close()
            if engine:
                engine.close()
            if live:
                live.close()
    verify_files(source_rows)
    verify_files(active_loras)
    if digest(root / "input.safetensors") != request["input_sha256"] or json.loads(request_path.read_text(encoding="utf-8")) != request:
        raise ValueError("Worker input/request changed during sampling")
    if digest(request["config_path"]) != request["config_sha256"]:
        raise ValueError("Runtime config changed during sampling")
    atomic_json(root / "result.json", report)  # Published only after Engine cleanup/lock release succeeded.


if __name__ == "__main__":
    main()
