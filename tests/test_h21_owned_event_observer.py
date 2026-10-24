"""Observer ownership protocol unit negatives; never opens a socket/process."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h21_owned_observer_test", ROOT / "tools/observe_h21_owned_events.py")
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


@pytest.fixture
def contract(tmp_path, monkeypatch):
    manifest = tmp_path / "owned/manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}", encoding="utf-8")
    script = manifest.parent / "runtime_tools/h21_control_probe.py"
    script.parent.mkdir()
    script.write_text("# protocol only, never executed", encoding="utf-8")
    plan = {"python": sys.executable, "run_root": str(manifest.parent), "port": 49199}
    state = {"exe": sys.executable, "argv": [sys.executable, "-B", "-X", "utf8", str(script), "--_serve", str(manifest)]}
    monkeypatch.setattr(observer.psutil, "Process", lambda pid: SimpleNamespace(exe=lambda: state["exe"], cmdline=lambda: state["argv"], create_time=lambda: 123.5))
    return manifest, plan, state


def test_canonical_exact_server_protocol_is_recognized_without_process_start(contract):
    manifest, plan, _ = contract
    row = observer.process_contract(manifest, plan, 42)
    assert row["pid"] == 42 and row["create_time"] == 123.5


@pytest.mark.parametrize("change", ["wrong_executable", "wrong_script", "wrong_manifest", "no_serve"])
def test_foreign_or_ambiguous_process_is_rejected(contract, change, tmp_path):
    manifest, plan, state = contract
    if change == "wrong_executable":
        state["exe"] = str(manifest)  # Exists but is not the bound Python.
    elif change == "wrong_script":
        state["argv"][4] = str(tmp_path / "foreign.py")
    elif change == "wrong_manifest":
        state["argv"][-1] = str(tmp_path / "foreign.json")
    else:
        state["argv"].remove("--_serve")
    with pytest.raises(RuntimeError):
        observer.process_contract(manifest, plan, 42)


@pytest.mark.parametrize("port", [8189, 8878, 8991, 0, True])
def test_user_and_invalid_ports_are_not_contacted(contract, port, monkeypatch):
    manifest, plan, _ = contract
    plan["port"] = port
    monkeypatch.setattr(observer.psutil, "net_connections", lambda **kwargs: pytest.fail("must not inspect/connect forbidden port"))
    with pytest.raises(ValueError):
        observer.listener_owner(manifest, plan)


def test_foreign_bind_and_multiple_listener_owners_are_not_connected(contract, monkeypatch):
    manifest, plan, _ = contract
    rows = [SimpleNamespace(status=observer.psutil.CONN_LISTEN, laddr=SimpleNamespace(port=49199, ip="0.0.0.0"), pid=42)]
    monkeypatch.setattr(observer.psutil, "net_connections", lambda **kwargs: rows)
    with pytest.raises(RuntimeError):
        observer.listener_owner(manifest, plan)
    rows[0].laddr.ip = "127.0.0.1"
    rows.append(SimpleNamespace(status=observer.psutil.CONN_LISTEN, laddr=SimpleNamespace(port=49199, ip="127.0.0.1"), pid=43))
    with pytest.raises(RuntimeError):
        observer.listener_owner(manifest, plan)


def test_missing_listener_is_a_wait_not_a_connection(contract, monkeypatch):
    manifest, plan, _ = contract
    monkeypatch.setattr(observer.psutil, "net_connections", lambda **kwargs: [])
    assert observer.listener_owner(manifest, plan) is None
