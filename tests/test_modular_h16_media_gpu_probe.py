"""Reduced GPU probe must preserve H16's stage and cold-resume topology."""

from tools.run_modular_h16_media_gpu import build_probe_graphs


def test_h16_reduced_probe_freeze_and_resume_are_disjoint():
    freeze, resume = build_probe_graphs()
    assert freeze["7"]["inputs"]["width"] == 224
    assert freeze["7"]["inputs"]["height"] == 224
    assert freeze["7"]["inputs"]["length"] == 124
    assert freeze["13"]["inputs"]["scale_by"] == 2.0
    assert freeze["51"]["inputs"]["confirm_save"] is True
    assert freeze["52"]["inputs"]["confirm_save"] is True
    assert resume["14"]["inputs"]["width"] == 448
    assert resume["14"]["inputs"]["height"] == 448
    assert resume["20"]["inputs"]["av_latent"] == ["50", 0]
    assert resume["21"]["inputs"]["images"] == ["20", 0]
    assert resume["21"]["inputs"]["audio"] == ["20", 1]
    assert sum(node["class_type"] == "MiniMaxH3H16Pass2WindowEXPT8"
               for node in freeze.values()) == 3
    assert sum(node["class_type"] == "MiniMaxH3H16Pass2WindowEXPT8"
               for node in resume.values()) == 4
    assert sum(node["class_type"] == "SamplerCustomAdvanced"
               for node in freeze.values()) == 1
    assert not any(node["class_type"] == "SamplerCustomAdvanced"
                   for node in resume.values())
    assert all(value["inputs"]["checkpoint_path"] == ""
               for value in resume.values()
               if value["class_type"] == "MiniMaxH3NativeLatentCheckpointLoadT8Advanced")


def test_h16_reduced_effect_probe_keeps_per_window_relay_and_eav():
    freeze, resume = build_probe_graphs("effects")
    assert freeze["7"]["inputs"]["width"] == 224
    assert freeze["7"]["inputs"]["height"] == 224
    assert freeze["88"]["inputs"]["confirm_save"] is True
    assert freeze["89"]["inputs"]["confirm_save"] is True
    assert resume["14"]["inputs"]["width"] == 448
    assert resume["14"]["inputs"]["height"] == 448
    assert resume["52"]["inputs"]["width"] == 448
    assert resume["52"]["inputs"]["height"] == 448
    assert resume["20"]["inputs"]["av_latent"] == ["80", 0]
    assert [freeze[key]["inputs"]["mode"] for key in ("60", "63", "66")] == [
        "report_only"] * 3
    assert [resume[key]["inputs"]["mode"] for key in ("69", "72", "75", "78")] == [
        "report_only", "report_only", "report_only", "apply_exp"]
    assert {"81", "82", "83", "88", "89"} <= set(freeze)
    assert {"21", "84", "85", "86", "87", "h16_report"} <= set(resume)
    assert sum(node["class_type"] == "MiniMaxH3H16Pass2WindowEXPT8"
               for node in freeze.values()) == 3
    assert sum(node["class_type"] == "MiniMaxH3H16Pass2WindowEXPT8"
               for node in resume.values()) == 4
    assert sum(node["class_type"] == "SamplerCustomAdvanced"
               for node in freeze.values()) == 1
    assert not any(node["class_type"] == "SamplerCustomAdvanced"
                   for node in resume.values())
