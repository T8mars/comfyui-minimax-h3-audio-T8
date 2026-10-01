"""Isolated CPU ComfyUI profile for eight private S27 LTX Relay draft graphs."""

from tools import build_modular_ltx_relay_workflows as relay
from tools import serve_modular_s09_s29_browser as browser


browser.SOURCES = (("S27R", relay.ROOT / "artifacts/development/modular-ltx-relay-20260928/candidate-v1",
                    "*_external*relay.json", 8),)
browser.SOURCE_NAMES = {"S27R": frozenset(name + ".json" for name in relay.generated())}
browser.SOURCE_AUDIT_REQUIRED = False


if __name__ == "__main__":
    raise SystemExit(browser.main())
