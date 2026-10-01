"""CPU-only guards for the real-weight S22 four-window execution driver."""

import asyncio
from argparse import Namespace
from pathlib import Path

from tools import run_modular_s22_chunked_v5_gpu as s22


def test_full_private_graph_keeps_four_independent_windows_and_effects():
    graph = s22.build_graph("full", width=128, height=128)
    assert graph["7"]["inputs"]["length"] == 192
    assert graph["14"]["inputs"]["temporal_chunk_frames"] == 85
    assert graph["14"]["inputs"]["target_width"] == 256
    assert [graph[key]["inputs"]["window_index"] for key in ("30", "31", "32", "33")] == [0, 1, 2, 3]
    assert graph["33"]["inputs"]["previous_result"] == ["53", 1]
    assert graph["53"]["inputs"]["confirm_save"] is True
    assert graph["55"]["inputs"]["window_result"] == ["33", 1]
    assert graph["18"]["inputs"]["av_latent"] == ["51", 0]
    assert graph["25"]["inputs"]["format"] == "mp4"
    assert graph["25"]["inputs"]["format.codec"] == "auto"
    assert [graph[key]["inputs"]["model"] for key in ("30", "31", "32", "33")] == [
        ["41", 0], ["44", 0], ["47", 0], ["50", 0]]


def test_cold_private_graph_uses_exact_receipts_and_has_no_prior_sampling():
    receipts = {"native": {"path": "partial.h3latent.safetensors",
                           "sha256": "a" * 64, "manifest_json": "{}"},
                "window2": {"path": "v5-window-x/manifest.json", "sha256": "b" * 64}}
    graph = s22.build_graph("cold", width=128, height=128, receipts=receipts)
    assert all(key not in graph for key in ("11", "12", "30", "31", "32"))
    assert graph["52"]["inputs"]["expected_file_sha256"] == "a" * 64
    assert graph["53"]["inputs"]["artifact_sha256"] == "b" * 64
    assert graph["33"]["inputs"]["previous_result"] == ["53", 1]
    assert graph["55"]["inputs"]["partial4_denoised_output"] == ["54", 0]
    assert graph["25"]["inputs"]["format"] == "mp4"


def test_s22_preflight_requires_installed_assets_and_both_private_ports(tmp_path, monkeypatch):
    args = Namespace(comfy_root=tmp_path, port=8873, min_free_vram_mib=12000,
                     min_free_ram_mib=95000)
    monkeypatch.setattr(s22.shared, "gpu_memory_mib", lambda: {"available": True, "free_mib": 14000})
    monkeypatch.setattr(s22.native_gpu, "_free_physical_mib", lambda: 100000)
    monkeypatch.setattr(s22.shared, "port_is_listening", lambda _host, port: port == 8874)
    missing = s22.preflight(args)
    assert not missing["ready"]
    assert not missing["checks"]["all_assets_installed"]
    assert not missing["checks"]["cold_port_free"]
    for asset in missing["assets"].values():
        path = Path(asset)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(s22.shared, "port_is_listening", lambda *_: False)
    assert s22.preflight(args)["ready"]


def test_full_and_cold_private_graphs_validate_in_current_cpu_core(monkeypatch):
    from tools.build_modular_fast_h3_v2_workflow import load_live_info
    from comfy_extras.nodes_lora_debug import LoraLoaderBypassModelOnly
    from comfy_extras.nodes_preview_any import PreviewAny

    load_live_info()
    monkeypatch.syspath_prepend(str(s22.PROJECT.parents[1]))
    import execution
    import nodes
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "PreviewAny", PreviewAny)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS,
                        "LoraLoaderBypassModelOnly", LoraLoaderBypassModelOnly)
    receipts = {"native": {"path": "partial.h3latent.safetensors",
                           "sha256": "a" * 64, "manifest_json": "{}"},
                "window2": {"path": "v5-window-x/manifest.json", "sha256": "b" * 64}}
    for kind in ("full", "cold"):
        graph = s22.build_graph(kind, width=128, height=128, receipts=receipts)
        valid, _errors, outputs, diagnostics = asyncio.run(
            execution.validate_prompt(f"s22-real-{kind}", graph, None))
        assert valid and not diagnostics, (kind, _errors, outputs, diagnostics)
        assert "25" in outputs and "55" in outputs
