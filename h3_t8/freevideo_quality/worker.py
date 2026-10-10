"""Owned pinned Quality worker: exact profiles, offline constants, cleanup before receipt."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time


def atomic_json(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf8")
    os.replace(temporary, path)


def main():
    root = Path(sys.argv[1]).resolve(strict=True)
    started = time.monotonic()
    while not (root / "launch-gate").is_file():
        if time.monotonic() - started > 30:
            raise RuntimeError("Quality worker was not assigned to its supervisor")
        time.sleep(.05)
    # -I removes the script directory. Import the sibling isolated packages
    # from h3_t8, not from the repository root; never import ComfyUI here.
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from freevideo_quality.profiles import plan, validate_clock, binding, LEGACY_RUNTIME, AUDIO_RUNTIME
    # Helpers are loaded as their original isolated package; no Core imported.
    from freevideo_exp.runtime import digest, verify_files
    from freevideo_quality.assets import required, verify, offline_binding, evidence_sha
    request_path = root / "request.json"
    request = json.loads(request_path.read_text(encoding="utf8"))
    selected = plan(request["plan"]["profile"], request["plan"]["role"])
    if request.get("schema") not in [binding(s)["request"] for s in (LEGACY_RUNTIME, AUDIO_RUNTIME)] or request["plan"] != selected:
        raise ValueError("Quality request profile/role mismatch")
    if digest(request["config_path"]) != request["config_sha256"] or digest(root / "input.safetensors") != request["input_sha256"]:
        raise ValueError("Quality worker input/config identity mismatch")
    config = json.loads(Path(request["config_path"]).read_text(encoding="utf8"))
    producer = binding(config.get("schema"))
    if config.get("freevideo_revision") != producer["revision"] or request["schema"] != producer["request"]:
        raise ValueError("Wrong quality worker runtime")
    source_rows = [row for row in config["files"] if row["kind"] != "weight"]
    verify_files(source_rows)
    sys.path.insert(0, config["source_root"])
    os.environ["FREEVIDEO_VDN_ROOT"] = config["vdn_root"]
    from freevideo_engine.paths import add_vdn
    add_vdn()
    import torch
    from safetensors.torch import load_file, save_file
    from freevideo_engine.locking import runtime_lock
    from freevideo_engine.adaln_assets import validate_catalog, weight_identity
    from freevideo_engine.media_conditioning import describe
    from freevideo_engine.hardware import _detect_local
    from freevideo_engine.policy import choose
    from freevideo_engine.gpu_budget import configure, LiveGPUBudget
    from freevideo_engine.runtime import Engine
    import diffusers
    import src.models.hybrid_attention as hybrid_module
    if (not Path(diffusers.__file__).resolve().is_relative_to(Path(config["vdn_root"]) / "diffusers/src")
        or not Path(hybrid_module.__file__).resolve().is_relative_to(Path(config["vdn_root"]))):
        raise ValueError("Foreign VDN/Diffusers won isolated import")
    torch.set_num_threads(8)
    tensors = load_file(str(root / "input.safetensors"), device="cpu")
    canvas = request["geometry"]
    value = dict(prompt_embeds=tensors["embeds"].to(torch.bfloat16), text_token_tags=tensors["tags"],
                 task=request["conditions"]["task"], width=canvas["width"], height=canvas["height"])
    if request["conditions"]["keyframes"]:
        value.update(keyframe_anchors=[r["anchor"] for r in request["conditions"]["keyframes"]],
                     condition_latents=[tensors[r["tensor"]].float() for r in request["conditions"]["keyframes"]])
    if request["conditions"]["refs"]:
        value["references"] = []
        for row in request["conditions"]["refs"]:
            ref = dict(kind=row["kind"])
            if "tensor" in row:
                ref["latent"] = tensors[row["tensor"]].float()
            if "audio_tensor" in row:
                ref["audio_latent"] = tensors[row["audio_tensor"]].float()
            value["references"].append(ref)
    info = describe(value, **canvas)
    if info["task"] != request["conditions"]["task"]:
        raise ValueError("Actual packed conditioning task changed")
    validate_clock(request["clock"], selected["profile"], selected["role"], info["task"],
                   reference_audio_t=producer["reference_audio_t"])
    cache = Path(config["cache"])
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf8"))
    validate_catalog(manifest, 50)
    weights = weight_identity(manifest)
    catalog = json.loads((Path(config["source_root"]) / "freevideo_engine/prepared_models.json").read_text(encoding="utf8"))
    tables = [(table, cache if location == "base" else Path(config["sampling_root"]))
              for table, location in required(manifest, catalog, selected["profile"], selected["role"], info["task"],
                                               reference_audio_t=producer["reference_audio_t"])]
    for table, folder in tables:
        verify(table, folder)
    active_loras = [row for row in request["loras"] if row["strength"] != 0]
    lora_report = dict(enabled=False)
    if active_loras:
        from freevideo_engine.lora_cache import index_adapter
        from freevideo_engine.lora_online_cache import _supported, prepare
        verify_files(active_loras)
        if not _supported(active_loras, manifest["linears"]):
            raise ValueError("Actual LoRA targets need unsupported fused/AdaLN conversion, not an unknown-combination prohibition")
        targets = {name: dict(shape=spec["weight_shape"]) for name, spec in manifest["linears"].items()}
        for row in active_loras:
            index_adapter(row, targets)
        with runtime_lock():
            cache, lora_report = prepare(cache, active_loras, output_root=Path(config["home"]) / "loras")
        derived = json.loads((cache / "manifest.json").read_text(encoding="utf8"))
        if weight_identity(derived) != weights:
            raise ValueError("LoRA-derived cache changed modulation weights; tables cannot be reused")
    hardware = _detect_local()
    from freevideo_exp.probe_identity import identity_for, validate_probes
    current_identity = identity_for(config, hardware)
    probes = json.loads((Path(config["home"]) / "kernel-probes.json").read_text(encoding="utf8"))
    validate_probes(config, probes, current_identity)
    relay = request.get("effects", {}).get("relay")
    if relay and relay["mode"] == "apply_exp" and len(relay["events"]) > 1:
        from freevideo_exp.masked_probe import validate
        validate(config, current_identity)
    from freevideo_engine.geometry import geometry
    admitted = geometry(**canvas)
    admitted.update({key: info[key] for key in ("reference_video_tokens", "reference_audio_tokens")})
    policy = choose(hardware, attention="cudnn", available_backends={"cudnn"}, canvas=admitted,
        lora_max_block_bytes=lora_report.get("max_block_bytes", 0), lora_root_bytes=lora_report.get("root_bytes", 0))
    options = dict(policy.engine, steps=selected["base_steps"], task=info["task"], canvas=admitted)
    options.update(stream_weights=True, preload_host=False, pin_host_weights=False)
    torch.save(value, root / "conditioning.pt")  # Own safetensors-derived value only.
    engine = live = effects = None
    report = {}
    with runtime_lock(), offline_binding(config["cache"], config["sampling_root"], catalog, tables) as loaded:
        try:
            memory = configure(torch, policy.gpu_budget_bytes, reserve_bytes=policy.gpu_system_reserve_bytes)
            if memory.get("windows_allocator_limit_enforced"):
                live = LiveGPUBudget(torch, memory, policy.gpu_budget_bytes, policy.gpu_system_reserve_bytes)
            engine = Engine(cache, base=config["base"], checkpoint=config["checkpoint"], **options)
            from freevideo_exp.worker_effects import install
            effects = install(engine, request.get("effects", {}), canvas)
            completed = 0

            def step(seconds):
                nonlocal completed
                completed += 1
                atomic_json(root / "progress.json", dict(completed=completed, total=selected["nfe"], seconds=seconds))

            kwargs = {}
            if selected["role"] == "HIGH":
                kwargs = dict(initial_latents=(tensors["initial_video"], tensors["initial_audio"]),
                              refine_steps=3, refine_schedule=selected["schedule"])
            video, audio, sampled = engine.sample(root / "conditioning.pt", request["seed"], **canvas, **kwargs,
                step_callback=step, gpu_reserve_bytes=policy.gpu_system_reserve_bytes,
                budget_refresh=live.refresh if live else None)
            if completed != selected["nfe"] or len(sampled["step_seconds"]) != completed:
                raise ValueError("Quality worker did not finish all actual NFE")
            if any(request.get("effects", {}).values()) and (effects.forwards != completed or effects.blocks != completed * 50):
                raise ValueError("External quality effects did not observe every actual block/NFE")
            expected_identities = {json.dumps(t["identity"], sort_keys=True) for t, _ in tables}
            if ({json.dumps(r["identity"], sort_keys=True) for r in loaded} != expected_identities
                or len(loaded) != len(tables) * 50
                or any({r["index"] for r in loaded if r["identity"] == t["identity"]} != set(range(50)) for t, _ in tables)):
                raise ValueError("Quality actual table loading coverage incomplete")
            out = dict(video=video.detach().cpu().contiguous(), audio=audio.detach().cpu().permute(1, 0, 2).unsqueeze(0).contiguous())
            if not all(bool(torch.isfinite(v).all()) for v in out.values()):
                raise ValueError("Nonfinite quality output")
            if selected["role"] == "HIGH":
                from freevideo_exp.runtime import tensor_record
                if tensor_record(out["audio"]) != request["low_audio"]:
                    raise ValueError("Community HIGH3 changed the completed LOW audio")
            save_file(out, str(root / "output.safetensors"))
            report = dict(request_sha256=digest(request_path), output_sha256=digest(root / "output.safetensors"),
                sample=sampled, engine=engine.config, load_seconds=engine.load_seconds,
                policy=policy.to_dict(), memory=memory, lora=lora_report, effects=effects.snapshot(),
                conditioning=info, torch=str(torch.__version__), cuda=torch.version.cuda, python=sys.version,
                source_inventory_sha256=config["inventory_sha256"], weight_identity=weights,
                tables=loaded, tables_sha256=evidence_sha(loaded))
        finally:
            if effects:
                effects.close()
            if engine:
                engine.close()
            if live:
                live.close()
    verify_files(source_rows)
    verify_files(active_loras)
    if (digest(root / "input.safetensors") != request["input_sha256"]
        or json.loads(request_path.read_text(encoding="utf8")) != request
        or digest(request["config_path"]) != request["config_sha256"]):
        raise ValueError("Quality input/config/request changed during sampling")
    atomic_json(root / "result.json", report)


if __name__ == "__main__":
    main()
