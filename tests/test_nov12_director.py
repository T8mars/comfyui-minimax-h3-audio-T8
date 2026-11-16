import pytest

from h3_audio_t8_pkg.director_sampling_settings import normalize_sampling
from h3_audio_t8_pkg.director_generation import build_director_generation_prompt
from h3_audio_t8_pkg.director_project import new_project
from test_director_generation import _patch_director_models, _store


def test_missing_upscale_strategy_keeps_legacy_settings():
    value = normalize_sampling({"mode": "two_pass"})
    assert "upscaler_architecture" not in value and "upscaler_sharpness" not in value


@pytest.mark.parametrize("tier", [1, 3, 6])
def test_two_pass_compiler_uses_explicit_2d_tier_and_original_sampler(tmp_path, monkeypatch, tier):
    from h3_audio_t8_pkg import director_generation
    _patch_director_models(monkeypatch)
    monkeypatch.setattr(director_generation.folder_paths, "get_filename_list", lambda folder: {
        "latent_upscale_models": ["minimax_h3_latent_upscaler_2d_conv_v1_bf16.safetensors"],
    }.get(folder, []))
    monkeypatch.setattr(director_generation.folder_paths, "get_full_path", lambda _folder, name: name)
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A continuous room shot."
    project["doc"]["sampling"] = {"mode": "two_pass", "upscaler_architecture": "2d",
                                  "upscaler_sharpness": tier, "output_mp": .6}
    built = build_director_generation_prompt(project, project["current"], _store(tmp_path))
    graph = built["prompt"]
    ups = next(x for x in graph.values() if x["class_type"] == "MiniMaxH3LatentUpscale2DT8")
    assert ups["inputs"]["sharpness"] == tier
    assert ups["inputs"]["av_latent"] == ["9", 1]
    assert ups["inputs"]["model_name"].endswith("2d_conv_v1_bf16.safetensors")
    assert graph["6"]["inputs"]["steps"] == 8
    assert graph["9"]["inputs"]["sigmas"][1] == 0
    assert not any(x["class_type"] == "MiniMaxH3LearnedLatentUpscaleT8Advanced" for x in graph.values())


@pytest.mark.parametrize("tier", [0, 7, 1.5, True])
def test_invalid_tier_is_not_coerced(tier):
    with pytest.raises(ValueError, match="档位"):
        normalize_sampling({"mode": "two_pass", "upscaler_architecture": "2d", "upscaler_sharpness": tier})


@pytest.mark.parametrize("mode", ["single", "two_pass"])
def test_compiler_keeps_interleaved_loras_in_one_ordered_window_stack(tmp_path, monkeypatch, mode):
    import json
    from h3_audio_t8_pkg import director_generation
    _patch_director_models(monkeypatch)
    monkeypatch.setattr(director_generation.folder_paths, "get_filename_list", lambda folder: {
        "loras": ["a.safetensors", "b.safetensors"],
        "latent_upscale_models": ["minimax_h3_latent_upscaler_3d_fp16.safetensors"],
    }.get(folder, []))
    monkeypatch.setattr(director_generation.folder_paths, "get_full_path", lambda _folder, name: name)
    rows = [{"id":"a", "name":"a.safetensors", "strength":.3, "enabled":True,
             "start_percent":.25, "end_percent":.75},
            {"id":"b", "name":"b.safetensors", "strength":-.2, "enabled":True}]
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A quiet room."
    if mode == "single":
        project["doc"]["generation"].update(lora_mode="manual", loras=rows)
    else:
        project["doc"]["sampling"] = {"mode":"two_pass", "low_loras":rows, "high_loras":rows}
    built = build_director_generation_prompt(project, project["current"], _store(tmp_path))
    stacks = [node for node in built["prompt"].values() if node["class_type"]=="MiniMaxH3AttachLoRAWindowsT8"]
    assert len(stacks) == (1 if mode == "single" else 2)
    for node in stacks:
        observed = json.loads(node["inputs"]["windows_json"])
        assert [(r["name"], r["strength"]) for r in observed] == [("a.safetensors", .3), ("b.safetensors", -.2)]
        assert observed[0]["start_percent"] == .25 and observed[0]["end_percent"] == .75
    assert not any(n["class_type"] == "MiniMaxH3LoRACompatibilityLoaderT8Advanced" for n in built["prompt"].values())


def test_2d_selection_and_tier_survive_real_browser_reload():
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from playwright.sync_api import sync_playwright
    from test_director_sampling_browser import ROOT, launch_browser

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def guess_type(self, path):
            return "text/javascript" if path.endswith(".mjs") else super().guess_type(path)

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(ROOT/"web/director")))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as playwright:
            browser = launch_browser(playwright)
            page = browser.new_page(viewport={"width": 1360, "height": 900})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/models", lambda route: route.fulfill(json={
                "unet": [], "clip": [], "video_vae": [], "audio_vae": [], "lora": [{"value":"a.safetensors","label":"A"}],
                "upscaler": [{"value": "minimax_h3_latent_upscaler_2d_conv_v1_bf16.safetensors", "label": "2D"}]}))
            page.goto(f"http://127.0.0.1:{server.server_port}/index.html", wait_until="networkidle")
            page.locator('.w-header-actions [data-action="model-settings"]').click()
            page.locator('[data-sampling-mode="two_pass"]').click()
            assert page.locator('[data-sampling-upscale-architecture]').input_value() == "3d"
            page.locator('[data-sampling-upscale-architecture]').select_option("2d")
            page.locator('[data-sampling-upscale-tier]').select_option("3")
            page.locator('[data-sampling-action="add"][data-stage="low_loras"]').click()
            row = page.locator('[data-stage="low_loras"][data-index="0"]')
            row.locator('[data-sampling-field="name"]').select_option('a.safetensors')
            row.locator('.o-lora-window summary').click()
            row.locator('[data-sampling-field="start_percent"]').fill('0.25')
            row.locator('[data-sampling-field="end_percent"]').fill('0.75')
            page.locator('[data-action="apply-model-settings"]').click()
            stored = page.evaluate("""() => JSON.parse(localStorage.getItem(Object.keys(localStorage).find(k=>k.startsWith('t8director.draft:'))))""")
            assert stored["doc"]["sampling"]["upscaler_architecture"] == "2d"
            assert stored["doc"]["sampling"]["upscaler_sharpness"] == 3
            assert stored["doc"]["sampling"]["low_loras"][0]["start_percent"] == .25
            assert stored["doc"]["sampling"]["low_loras"][0]["end_percent"] == .75
            page.reload(wait_until="networkidle")
            page.locator('.w-header-actions [data-action="model-settings"]').click()
            assert page.locator('[data-sampling-upscale-architecture]').input_value() == "2d"
            assert page.locator('[data-sampling-upscale-tier]').input_value() == "3"
            row = page.locator('[data-stage="low_loras"][data-index="0"]')
            row.locator('.o-lora-window summary').click()
            assert row.locator('[data-sampling-field="start_percent"]').input_value() == '0.25'
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("variant", ["single8", "continuous4plus4", "continuous4plus4separate",
                                     "upscale8plus4", "upscale4plus4"])
@pytest.mark.parametrize("relay", [False, True])
def test_hyperflow_consumes_window_rows_in_actual_phase_conditioning(tmp_path, monkeypatch, variant, relay):
    import json
    import torch
    from safetensors.torch import save_file
    from h3_audio_t8_pkg import director_generation, director_hyperflow
    _patch_director_models(monkeypatch)
    lora = tmp_path/"content.safetensors"
    save_file({"dummy":torch.ones(1)}, str(lora))
    hyperflow = tmp_path/"hf.safetensors"
    hyperflow.write_bytes(b"graph only")
    monkeypatch.setattr(director_hyperflow, "_resolve", lambda _:hyperflow)
    monkeypatch.setattr(director_generation.folder_paths, "get_filename_list", lambda folder: {
        "loras":["content.safetensors"],
        "latent_upscale_models":["minimax_h3_latent_upscaler_3d_fp16.safetensors"]}.get(folder, []))
    monkeypatch.setattr(director_generation.folder_paths, "get_full_path", lambda folder, name: str(lora) if folder == "loras" else name)
    project = new_project()
    project["doc"]["shots"][0]["simplePrompt"] = "A quiet room."
    project["doc"]["d3"] = {"prompt_relay":{"enabled":relay,"execution_mode":"apply_exp"}}
    rows = [{"id":"window", "name":"content.safetensors", "strength":.4, "enabled":True,
             "start_percent":.25, "end_percent":.75},
            {"id":"disabled", "name":"", "strength":1., "enabled":False},
            {"id":"static", "name":"content.safetensors", "strength":-.2, "enabled":True}]
    project["doc"]["sampling"] = {"mode":"hyperflow", "variant":variant, "hyperflow_file":"hyperflow/hf.safetensors",
                                  "low_loras":rows,"high_loras":rows,"output_mp":.4}
    built = build_director_generation_prompt(project, project["current"], _store(tmp_path))
    graph = built["prompt"]
    stacks = {key:node for key,node in graph.items() if node["class_type"]=="MiniMaxH3AttachLoRAWindowsT8"}
    assert len(stacks) == (1 if variant == "single8" else 2)
    assert not any(node["class_type"]=="MiniMaxH3LoRACompatibilityLoaderT8Advanced" for node in graph.values())
    for key, stack in stacks.items():
        observed = json.loads(stack["inputs"]["windows_json"])
        assert [r["id"] for r in observed] == ["window","static"]
        assert [r["_original_row_index"] for r in observed] == [1,3]
        assert observed[0]["start_percent"] == .25
        # Reach a real consumer (guider or stage bind/split), never a dead node.
        assert any(node_id != key and [key,0] in node["inputs"].values() for node_id,node in graph.items())
    hooks_report = [row for row in built["report"]["lora_loaders"] if row["source"]=="native_window_hooks"]
    assert len(hooks_report) == len(stacks)*2
    assert {r["row_index"] for r in hooks_report} == {1,3}
