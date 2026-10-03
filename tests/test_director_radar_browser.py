"""Real Chromium + actual guarded evidence API and real Store; no GPU/queue."""

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from playwright.sync_api import sync_playwright, expect

from test_director_radar_routes import radar_routes as _radar_fixture, dispatch

from h3_audio_t8_pkg.director_project import compile_project

radar_routes = _radar_fixture

ROOT = Path(__file__).resolve().parents[1]


def test_real_ui_rule_candidate_adoption_save_reload_and_revert(radar_routes):
    store, handlers = radar_routes
    requests, saved, errors = [], [], []

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def guess_type(self, path):
            return (
                "text/javascript" if path.endswith(".mjs") else super().guess_type(path)
            )

        def reply(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            requests.append(("GET", self.path))
            if self.path.endswith("/models"):
                return self.reply(
                    {
                        "unet": [],
                        "clip": [],
                        "video_vae": [],
                        "audio_vae": [],
                        "lora": [],
                    }
                )
            if "/results/" in self.path:
                return self.reply({"results": []})
            if self.path.endswith("/projects"):
                return self.reply({"projects": store.list()})
            if "/projects/" in self.path:
                return self.reply(store.load(self.path.rsplit("/", 1)[-1]))
            return super().do_GET()

        def do_POST(self):
            requests.append(("POST", self.path))
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/radar/operate"):
                code, result = dispatch(handlers[self.path], body)
                return self.reply(result, code)
            if "/projects/" in self.path:
                result = store.save(body["project"], body["expected_revision"])
                saved.append(result)
                return self.reply(result)
            if self.path.endswith("/compile"):
                return self.reply(
                    compile_project(body["project"], store, shot_id=body.get("shot_id"))
                )
            raise AssertionError("Unexpected mutation or queue request: " + self.path)

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(Handler, directory=str(ROOT / "web" / "director"))
    )
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(
                f"http://127.0.0.1:{server.server_port}/index.html",
                wait_until="networkidle",
            )
            page.locator('[data-field="simplePrompt"]').fill("原稿🙂 台词“不要改写”。")
            page.locator('[data-field="simplePrompt"]').press("Tab")

            def open_dialog():
                page.locator(".w-header-actions > details > summary").click()
                page.locator("[data-radar-open]").click()
                expect(page.locator("[data-radar-dialog]")).to_be_visible()

            def await_update():
                expect(page.locator("[data-radar-status]")).to_contain_text(
                    "草稿已更新"
                )

            open_dialog()
            page.locator("[data-radar-name]").fill("全片规则")
            page.locator("[data-radar-rule]").fill("cinematic night")
            page.locator("[data-radar-scope]").select_option("project")
            page.locator("[data-radar-add]").click()
            await_update()
            expect(page.locator("[data-radar-enable]")).to_have_count(1)
            page.locator("[data-radar-name]").fill("本镜规则")
            page.locator("[data-radar-rule]").fill("slow zoom")
            page.locator("[data-radar-scope]").select_option("shot")
            page.locator("[data-radar-add]").click()
            await_update()
            expect(page.locator("[data-radar-enable]")).to_have_count(2)
            page.locator("[data-radar-compile]").click()
            await_update()
            expect(page.locator("[data-radar-candidate]")).to_have_value(
                "原稿🙂 台词“不要改写”。\n\ncinematic night\n\nslow zoom"
            )
            expect(page.locator("[data-radar-candidate-state]")).to_contain_text(
                "待审候选"
            )
            page.locator("[data-radar-adopt]").click()
            await_update()
            expect(page.locator("[data-radar-candidate-state]")).to_contain_text(
                "已显式采用"
            )
            page.locator("[data-radar-local-disable]").check()
            await_update()
            expect(page.locator("[data-radar-status]")).to_contain_text("stale")
            page.locator("[data-radar-compile]").click()
            await_update()
            expect(page.locator("[data-radar-candidate]")).to_have_value(
                "原稿🙂 台词“不要改写”。\n\nslow zoom"
            )
            page.locator("[data-radar-adopt]").click()
            await_update()
            page.locator("[data-radar-library-remove]").first.click()
            await_update()
            page.locator("[data-radar-library-remove]").first.click()
            await_update()
            expect(page.locator("[data-radar-library-remove]")).to_have_count(0)
            expect(page.locator("[data-radar-enable]")).to_have_count(2)
            page.locator("[data-radar-close]").click()
            expect(page.locator('[data-field="simplePrompt"]')).to_have_value(
                "原稿🙂 台词“不要改写”。"
            )
            page.locator('[data-service="save"]').click()
            page.wait_for_function(
                "document.querySelector('[data-save]').textContent.includes('已真实保存')"
            )
            assert len(saved) == 1
            assert (
                compile_project(saved[0])["shots"][0]["local_prompt"]
                == "原稿🙂 台词“不要改写”。\n\nslow zoom"
            )
            page.reload(wait_until="networkidle")
            open_dialog()
            expect(page.locator("[data-radar-candidate-state]")).to_contain_text(
                "已显式采用"
            )
            expect(page.locator("[data-radar-local-disable]")).to_be_checked()
            expect(page.locator("[data-radar-library-remove]")).to_have_count(0)
            page.locator("[data-radar-revert]").click()
            await_update()
            expect(page.locator("[data-radar-candidate-state]")).to_contain_text(
                "已回退"
            )
            page.locator("[data-radar-close]").click()
            page.locator('[data-service="save"]').click()
            page.wait_for_function(
                "document.querySelector('[data-save]').textContent.includes('已真实保存')"
            )
            assert (
                len(saved) == 2
                and compile_project(saved[-1])["shots"][0]["local_prompt"]
                == "原稿🙂 台词“不要改写”。"
            )
            assert not any(
                "/generate" in path or "/batches" in path or "/prompt" in path
                for _method, path in requests
            )
            assert errors == []
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
