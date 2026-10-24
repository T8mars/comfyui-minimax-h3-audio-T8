"""Owned v4 production child: real pinned CUDA Engine, never a CPU facade."""
from __future__ import annotations

import hashlib
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import time


def publish(path, value):
    temporary = path.with_suffix(".partial")
    with temporary.open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    root = Path(sys.argv[1]).resolve(strict=True)
    started = time.monotonic()
    while not (root / "launch-gate").is_file():
        if time.monotonic() - started > 30:
            raise RuntimeError("H21 child has no actual owned-process supervisor")
        time.sleep(.05)
    # h3_t8 is a namespace package; no ComfyUI/root initializer is executed.
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from h3_t8.freevideo_h21.runtime import Model, config_for, unpack_condition
    from h3_t8.freevideo_h21.source import Source, canonical, digest
    from h3_t8.freevideo_h21.plans import plan
    from h3_t8.freevideo_h21.execution import execute, controlled_cache_environment
    from h3_t8.freevideo_h21.cache_state import snapshot_history
    from h3_t8.freevideo_h21.effects import pass_canvas, install as install_effects
    from h3_t8.freevideo_exp.runtime import verify_files
    request_path = root / "request.json"
    request = json.loads(request_path.read_text(encoding="utf8"))
    request_sha = digest(request_path)
    if request.get("schema") != "t8.freevideo.h21.request.v4":
        raise ValueError("Wrong v4 owned-worker request producer")
    model = Model(request["config_path"], request["config_sha256"], tuple(request["loras"]))
    config = config_for(model, full=True)
    source = Source(config["source_root"])
    selected = request["plan"]
    actual_plan = plan(source, selected["tier"], **selected["requested_canvas"], task=selected["task"], role=selected["role"])
    if actual_plan != selected or digest(root / "input.safetensors") != request["input_sha256"]:
        raise ValueError("v4 request plan/input content changed")
    os.environ.pop("FREEVIDEO_STEP_CACHE", None)
    os.environ["FREEVIDEO_VDN_ROOT"] = config["vdn_root"]
    sys.path.insert(0, config["source_root"])
    # The real author's initializer is deliberately confined to this owned
    # child. Main-process Source.module never imports it.
    from freevideo_engine.paths import add_vdn
    add_vdn()
    import diffusers
    import torch
    from safetensors.torch import load_file, save_file
    from freevideo_engine.runtime import Engine
    from freevideo_engine import step_cache as engine_step_cache
    from freevideo_engine.locking import runtime_lock
    from freevideo_engine.latent_upscale import upscale
    from freevideo_engine.media_conditioning import describe
    from freevideo_engine.adaln_assets import weight_identity
    from h3_t8.freevideo_h21.assets import offline_tables, model_structure, modulation_strategy, prepare_loras
    from h3_t8.freevideo_h21.native_contract import selected_width
    if not Path(diffusers.__file__).resolve().is_relative_to(Path(config["vdn_root"]) / "diffusers/src"):
        raise ValueError("Foreign Diffusers won the explicit v4 VDN dependency binding")
    torch.set_num_threads(8)
    tensors = load_file(str(root / "input.safetensors"), device="cpu")
    conditioning = unpack_condition(request["conditioning"], tensors)
    canvas = selected["actual_canvas"]
    structure = model_structure(config["base"])
    with selected_width(structure["text_dim"]):
        info = describe(conditioning, **canvas)
    if info["task"] != selected["task"]:
        raise ValueError("Actual packed v4 conditioning task changed")
    manifest = json.loads((Path(config["cache"]) / "manifest.json").read_text(encoding="utf8"))
    catalog = json.loads((source.root / "freevideo_engine/prepared_models.json").read_text(encoding="utf8"))
    sampling = selected["sampling"]
    tables = []
    active_loras = [json.loads(row) for row in model.loras if not json.loads(row).get("disabled")]
    verify_files(active_loras)
    cache = Path(config["cache"])
    lora_report = dict(enabled=False, ordered_slots=list(model.loras))
    options = dict(config["engine_options"])
    reserved = {"steps", "task", "canvas", "step_cache", "fetch_tables", "base", "checkpoint"}
    if reserved.intersection(options):
        raise ValueError("v4 engine options must not overwrite the explicit plan/clock/task/cache ownership")
    options.update(steps=sampling["base_steps"], task=selected["task"], canvas=dict(canvas, **{
        key: info[key] for key in ("reference_video_tokens", "reference_audio_tokens")}),
        step_cache=selected["step_cache"], fetch_tables=False)
    engine = effects = None
    result = None
    with runtime_lock(), controlled_cache_environment(), ExitStack() as table_scopes:
        table_scopes.enter_context(snapshot_history(engine_step_cache, enabled=selected["step_cache"] == "easycache-av"))
        table_scopes.enter_context(selected_width(structure["text_dim"]))
        table_scopes.enter_context(offline_tables(source, []))
        try:
            if active_loras:
                cache, lora_report = prepare_loras(cache, active_loras, options, Path(config["home"]) / "loras-v4")
                manifest = json.loads((cache / "manifest.json").read_text(encoding="utf8"))
            cache, manifest, options, selections, modulation = modulation_strategy(source, cache, manifest,
                catalog, selected, options, structure, sampling_root=config["sampling_root"], home=config["home"])
            tables = [entry for row in selections for entry in row["files"]]
            verify_files(tables)
            table_evidence = table_scopes.enter_context(offline_tables(source, selections))
            actual_groups = [dict(path=str(cache / row["file"]), bytes=row["bytes"], sha256=row["sha256"], kind="weight")
                             for row in manifest["groups"]]
            verify_files(actual_groups)
            engine = Engine(cache, base=config["base"], checkpoint=config["checkpoint"], **options)
            engine.h21_source_root = str(source.root)
            # Existing independently authored effects retain their actual
            # callable delegation. No model/LoRA reset or force-to-known chain.
            if any(value is not None and value.get("mode") != "disabled" for value in request.get("effects", {}).values()):
                sys.path.insert(0, str(Path(__file__).parents[1]))
                effects = install_effects(engine, request["effects"], canvas)
            input_path = root / "conditioning.pt"
            torch.save(conditioning, input_path)  # Own safetensors/JSON-derived data only.
            def progress(completed, total, seconds):
                publish(root / "progress.json", dict(completed=completed, total=total, seconds=seconds))
            def learned(video, target):
                if not config.get("upscaler"):
                    raise FileNotFoundError("The actual v4 plan requires an explicit learned 3D checkpoint")
                return upscale(video, config["upscaler"], target["width"], target["height"])
            low = (tensors["low_video"], tensors["initial_audio"].to(engine.device_backend.device)) if selected["role"] == "HIGH" else None
            lifted = tensors["initial_video"].to(engine.device_backend.device) if selected["role"] == "HIGH" else None
            video, audio, actual = execute(engine, input_path, request["seed"], selected, upscaler=learned,
                low=low, lifted=lifted, progress=progress, pass_context=lambda canvas, role: pass_canvas(effects, canvas, role))
            output = dict(video=video.detach().cpu().contiguous(),
                          audio=audio.detach().cpu().permute(1, 0, 2).unsqueeze(0).contiguous())
            save_file(output, str(root / "output.safetensors"))
            result = dict(actual=actual, output_sha256=digest(root / "output.safetensors"),
                tables=table_evidence, tables_sha256=hashlib.sha256(canonical(table_evidence).encode()).hexdigest(),
                engine=engine.config, load_seconds=engine.load_seconds, conditioning=info, lora=lora_report,
                modulation=modulation, actual_weight_identity=weight_identity(manifest), model_structure=structure,
                actual_group_files=actual_groups,
                effects=effects.snapshot() if effects else {}, requested_effects=request.get("effects", {}),
                torch=str(torch.__version__), cuda=torch.version.cuda,
                qualification="actual_owned_engine_completion_not_human_quality_or_speed_acceptance")
        finally:
            if effects:
                effects.close()
            if engine:
                engine.close()
    source.verify()
    config_for(model, full=True)
    verify_files(active_loras + tables)
    verify_files(actual_groups)
    if digest(request_path) != request_sha or digest(root / "input.safetensors") != request["input_sha256"]:
        raise ValueError("v4 request/input changed during actual sampling; no completion receipt published")
    result["request_sha256"] = request_sha
    publish(root / "result.json", result)


if __name__ == "__main__":
    main()
