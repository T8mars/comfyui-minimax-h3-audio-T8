"""Owned headless browser and real guarded API/Store; no user browser or GPU."""

from copy import deepcopy
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from threading import Thread

from playwright.sync_api import expect, sync_playwright

from h3_audio_t8_pkg.director_project import compile_project
from test_director_radar_routes import radar_routes as _radar_fixture, dispatch


radar_routes = _radar_fixture
ROOT = Path(__file__).resolve().parents[1]


def test_new_template_real_button_explicit_candidate_save_reopen_and_original_preserved(radar_routes, tmp_path):
    store, handlers = radar_routes
    requests, saved, errors, before = [], [], [], []

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def guess_type(self, path):
            return "text/javascript" if path.endswith(".mjs") else super().guess_type(path)

        def reply(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            requests.append(("GET", self.path))
            if self.path.endswith("/models"):
                return self.reply({key: [] for key in ("unet", "clip", "video_vae", "audio_vae", "lora")})
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
            if not before:
                before.append(deepcopy(body["project"]))
            if self.path.endswith("/radar/operate"):
                status, value = dispatch(handlers[self.path], body)
                return self.reply(value, status)
            if "/projects/" in self.path:
                value = store.save(body["project"], body["expected_revision"])
                saved.append(value)
                return self.reply(value)
            raise AssertionError("Unexpected side effect or generation: " + self.path)

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(ROOT / "web/director")))
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{server.server_port}/index.html", wait_until="networkidle")
            page.locator('[data-field="simplePrompt"]').fill("用户原稿：A站在原有箱子旁。")
            page.locator('[data-field="simplePrompt"]').press("Tab")
            page.locator(".w-header-actions > details > summary").click()
            page.locator("[data-radar-open]").click()
            page.locator('[data-radar-recipe="ending_beats"]').click()
            expect(page.locator("[data-radar-name]")).to_have_value("明确结局与局部节拍 · 本镜声明")
            expect(page.locator("[data-radar-scope]")).to_have_value("shot")
            expect(page.locator("[data-radar-status]")).to_contain_text("未添加快照、未采用、未保存或排队")
            assert not any(method == "POST" for method, _ in requests)
            page.locator("[data-radar-rule]").fill(page.locator("[data-radar-rule]").input_value() + "\n我的补充不应被覆盖")
            edited = page.locator("[data-radar-rule]").input_value()
            page.locator('[data-radar-recipe="ending_beats"]').click()
            expect(page.locator("[data-radar-status]")).to_contain_text("已有草稿")
            expect(page.locator("[data-radar-rule]")).to_have_value(edited)
            page.get_by_text("参数声明 / 本次取值", exact=True).click()
            page.locator("[data-radar-parameters]").fill(json.dumps({
                "visible_ending": "A没有钥匙，最后箱盖始终关闭。",
                "local_beats": "本镜局部0–2秒展示空手，2–5秒保持箱盖关闭。",
                "literal_dialogue": "无对白",
            }, ensure_ascii=False))

            def updated():
                expect(page.locator("[data-radar-status]")).to_contain_text("草稿已更新")

            page.locator("[data-radar-add]").click()
            updated()
            page.locator("[data-radar-compile]").click()
            updated()
            expect(page.locator("[data-radar-candidate-state]")).to_contain_text("待审候选")
            expect(page.locator("[data-radar-candidate]")).to_have_value(re.compile(r"[\s\S]*最后箱盖始终关闭[\s\S]*我的补充不应被覆盖"))
            page.locator("[data-radar-adopt]").click()
            updated()
            expect(page.locator("[data-radar-candidate-state]")).to_contain_text("已显式采用")
            page.screenshot(path=str(tmp_path / "ending-beats.png"), full_page=True)
            page.locator("[data-radar-close]").click()
            expect(page.locator('[data-field="simplePrompt"]')).to_have_value("用户原稿：A站在原有箱子旁。")
            page.locator('[data-service="save"]').click()
            page.wait_for_function("document.querySelector('[data-save]').textContent.includes('已真实保存')")
            assert len(saved) == 1
            assert "最后箱盖始终关闭" in compile_project(saved[0])["shots"][0]["local_prompt"]
            assert saved[0]["doc"]["global"] == before[0]["doc"]["global"]
            assert saved[0]["doc"]["shots"][0]["events"] == before[0]["doc"]["shots"][0]["events"]
            assert saved[0]["assets"] == before[0]["assets"]
            page.reload(wait_until="networkidle")
            page.locator(".w-header-actions > details > summary").click()
            page.locator("[data-radar-open]").click()
            expect(page.locator("[data-radar-candidate-state]")).to_contain_text("已显式采用")
            expect(page.locator("[data-radar-candidate]")).to_have_value(re.compile(r"[\s\S]*最后箱盖始终关闭[\s\S]*我的补充不应被覆盖"))
            page.locator("[data-radar-revert]").click()
            updated()
            expect(page.locator("[data-radar-candidate-state]")).to_contain_text("已回退")
            assert errors == []
            assert not any("/generate" in path or "/batches" in path or "/prompt" in path for _, path in requests)
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
