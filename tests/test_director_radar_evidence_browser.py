"""Actual source bytes, guarded API/isolated timing worker and visible evidence UI."""

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import uuid

from playwright.sync_api import sync_playwright, expect
import pytest

from test_director_radar_routes import radar_routes as _radar_fixture, dispatch

from test_director_idempotency import _write_valid_video
from h3_audio_t8_pkg.director_project import new_project, compile_project

radar_routes = _radar_fixture

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("lan_crypto", [False, True])
def test_measured_source_evidence_confirm_override_save_and_reload(
    radar_routes, lan_crypto
):
    store, handlers = radar_routes
    aid = str(uuid.uuid4())
    path = store.input_root / "t8_director" / aid / "source.mp4"
    path.parent.mkdir(parents=True)
    _write_valid_video(path)
    asset = store.register_asset(path, aid, "source.mp4")
    project = new_project()
    project["assets"] = [asset]
    project["doc"]["shots"][0]["simplePrompt"] = "保留原作者稿"
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
            if self.path.endswith("/timing"):
                code, result = dispatch(
                    handlers["/minimax_h3_t8/director/radar/assets/{asset_id}/timing"],
                    {},
                    match_info={"asset_id": aid},
                )
                return self.reply(result, code)
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
            raise AssertionError("Unexpected queue or mutation: " + self.path)

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(Handler, directory=str(ROOT / "web" / "director"))
    )
    Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.on("pageerror", lambda error: errors.append(str(error)))
            if lan_crypto:
                page.add_init_script(
                    "Object.defineProperty(crypto,'subtle',{value:undefined,configurable:true});Object.defineProperty(crypto,'randomUUID',{value:undefined,configurable:true});"
                )
            page.goto(
                f"http://127.0.0.1:{server.server_port}/index.html",
                wait_until="networkidle",
            )
            page.evaluate(
                "project=>window.postMessage({type:'t8-director:init',project},location.origin)",
                project,
            )
            expect(page.locator('[data-field="simplePrompt"]')).to_have_value(
                "保留原作者稿"
            )

            def open_dialog():
                page.locator(".w-header-actions > details > summary").click()
                page.locator("[data-radar-open]").click()
                expect(page.locator("[data-radar-dialog]")).to_be_visible()

            def updated():
                expect(page.locator("[data-radar-status]")).to_contain_text(
                    "草稿已更新"
                )

            open_dialog()
            page.locator("[data-radar-dialog] > details").nth(0).locator(
                "summary"
            ).first.click()
            page.locator("[data-radar-example]").click()
            expect(page.locator("[data-radar-status]")).to_contain_text(
                "源轨道/PTS已实测"
            )
            packet = json.loads(page.locator("[data-radar-packet]").input_value())
            assert packet["source"]["sha256"] == asset["sha256"]
            assert packet["source"]["timebase"] != {"num": 1, "den": 90000}
            assert page.locator("[data-radar-start]").get_attribute("min") is None
            page.locator("[data-radar-start]").fill("-1")
            assert page.locator("[data-radar-start]").evaluate(
                "(input)=>input.checkValidity()"
            )
            page.locator("[data-radar-start]").fill(str(packet["source"]["start_pts"]))
            page.locator("[data-radar-claim]").fill("原帧的衣服是红色（人工观察候选）")
            page.locator("[data-radar-claim-add]").click()
            page.locator("[data-radar-import]").click()
            updated()
            expect(page.locator("[data-radar-status]")).to_contain_text("unreviewed")
            page.locator("[data-radar-claim-id]").check()
            page.locator("[data-radar-confirm]").click()
            updated()
            expect(page.locator("[data-radar-status]")).to_contain_text("confirmed")
            page.locator("[data-radar-dialog] > details").nth(1).locator(
                "summary"
            ).first.click()
            page.locator("[data-radar-intent]").fill("改成蓝色衣服，原作者稿保留")
            page.locator("[data-radar-reason]").fill("用户明确创作修改")
            page.locator("[data-radar-intent-save]").click()
            updated()
            page.locator("[data-radar-facts]").check()
            page.locator("[data-radar-compile]").click()
            updated()
            expect(page.locator("[data-radar-candidate]")).to_have_value(
                "保留原作者稿\n\n[已确认visual证据] 原帧的衣服是红色（人工观察候选）\n\n[用户创作意图] 改成蓝色衣服，原作者稿保留"
            )
            page.locator("[data-radar-adopt]").click()
            updated()
            page.locator("[data-radar-close]").click()
            expect(page.locator('[data-field="simplePrompt"]')).to_have_value(
                "保留原作者稿"
            )
            page.locator('[data-service="save"]').click()
            page.wait_for_function(
                "document.querySelector('[data-save]').textContent.includes('已真实保存')"
            )
            assert len(saved) == 1
            compiled = compile_project(saved[0], store)
            assert (
                compiled["ready"]
                and "红色" in compiled["shots"][0]["local_prompt"]
                and "蓝色" in compiled["shots"][0]["local_prompt"]
            )
            open_dialog()
            page.locator("[data-radar-intent]").fill("改成蓝色衣服，镜头缓慢推近")
            page.locator("[data-radar-reason]").fill("第二次明确创作修改")
            page.locator("[data-radar-intent-save]").click()
            updated()
            page.locator("[data-radar-compile]").click()
            updated()
            page.locator("[data-radar-adopt]").click()
            updated()
            expect(page.locator("[data-radar-history]")).to_contain_text(
                "用户明确创作修改"
            )
            expect(page.locator("[data-radar-history]")).to_contain_text(
                "改成蓝色衣服，原作者稿保留"
            )
            page.locator("[data-radar-close]").click()
            with page.expect_response(
                lambda response: "/projects/" in response.url
                and response.request.method == "POST"
            ) as saved_response:
                page.locator('[data-service="save"]').click()
            assert saved_response.value.status == 200
            page.wait_for_function(
                "document.querySelector('[data-save]').textContent.includes('已真实保存')"
            )
            assert len(saved) == 2
            assert (
                saved[1]["doc"]["shots"][0]["intentHistory"][0]
                == saved[0]["doc"]["shots"][0]["creativeIntent"]
            )
            assert (
                saved[1]["doc"]["shots"][0]["candidateHistory"][0]
                == saved[0]["doc"]["shots"][0]["promptCandidate"]
            )
            page.reload(wait_until="networkidle")
            open_dialog()
            expect(page.locator("[data-radar-candidate-state]")).to_contain_text(
                "已显式采用"
            )
            page.locator("[data-radar-dialog] > details").nth(0).locator(
                "summary"
            ).first.click()
            expect(page.locator("[data-radar-claim-id]")).to_be_checked()
            assert (
                page.locator("[data-radar-intent]").input_value()
                == "改成蓝色衣服，镜头缓慢推近"
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
