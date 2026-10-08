"""New external-RGB grid/origin/stereo contracts; not trained GPU evidence."""
import json

import comfy.samplers
import pytest
import torch

from h3_audio_t8_pkg import sol_engine_h3_super_advanced as sol
from h3_audio_t8_pkg.audio_ops import trim_av_output
from h3_audio_t8_pkg.modular_sampling.ltx_rgb_stage import bind_ltx_rgb_stage, audit_ltx_rgb_stage
from h3_audio_t8_pkg.source_conform import conform


def _case():
    # The small spatial geometry is only a CPU fixture. Non-24fps source PTS
    # are tested separately using the real file/decoder, not this declaration.
    frames = torch.linspace(0, 1, 190*32*32*3).reshape(190, 32, 32, 3)
    audio = {"sample_rate": 44100,
        "waveform": torch.stack((torch.linspace(-.4, .4, 7*44100),
                                 torch.linspace(.2, -.2, 7*44100))).unsqueeze(0)}
    conformed, conformed_audio, count, duration, frame_map, report = conform(
        frames, 30., 32, 32, 124, 1., "strict", "pad_silence", audio)
    prepared, *_, prep_report = sol.prepare_h3_draft_for_ltx_refiner(conformed, 64, 64, fps=24.)
    source_audio = {"sample_rate": 44100, "waveform": audio["waveform"][:, :, 44100:44100+round(124/24*44100)]}
    latent = {"samples": torch.zeros(1, 128, 16, 2, 2)}
    model = type("Model", (), {"model_options": {}})()
    _, sigmas, _, setup = sol.setup_ltx_identity_preserve_refiner(model, enabled=False)
    args = (conformed, source_audio, prepared, prep_report, latent, model, object(),
            comfy.samplers.CFGGuider(model), comfy.samplers.KSAMPLER(lambda *a, **kw: None), sigmas, setup)
    return frames, audio, conformed_audio, count, duration, frame_map, report, args


def test_new_external_grid_uses_nonzero_source_origin_and_old_ltx_trim():
    frames, _, _, count, duration, frame_map, report, args = _case()
    expected_indices = tuple(round((1 + index/24)*30) for index in range(124))
    assert frame_map.indices == expected_indices
    assert frame_map.clock is None  # A declared fps is not observed CFR.
    assert report["clock_status"] == "declared_fps_unverified"
    assert count == 124 and duration == 124/24
    assert len(frames) == 190 and args[2].shape == (121, 32, 32, 3)
    prep = json.loads(args[3])
    assert prep["dropped_tail_frames"] == 3 and prep["output_duration_seconds"] == 121/24
    assert args[4]["samples"].shape == (1, 128, 16, 2, 2)
    assert len(args[9]) - 1 == 3


def test_external_stereo_original_rate_bypasses_conform_audio_and_trims_to_5s():
    _, audio, conformed_audio, _, _, _, _, args = _case()
    assert conformed_audio["sample_rate"] == 32000
    assert args[1]["sample_rate"] == 44100
    bound = bind_ltx_rgb_stage(*args)
    candidate = {"samples": args[4]["samples"] + .01}
    audited = audit_ltx_rgb_stage(bound[5], *args, candidate)
    assert audited[0] is candidate and audited[1] is args[1]
    frames, final_audio, report = trim_av_output(args[2], 0., 5., audited[1], 24.)
    assert frames.shape[0] == 120 and json.loads(report)["actual_video_duration_seconds"] == 5.
    assert final_audio["sample_rate"] == 44100 and final_audio["waveform"].shape == (1, 2, 220500)
    assert torch.equal(final_audio["waveform"], audio["waveform"][:, :, 44100:264600])
    assert not torch.equal(final_audio["waveform"][:, 0], final_audio["waveform"][:, 1])


def test_external_source_or_audio_change_cannot_reuse_bound_candidate():
    *_, args = _case()
    bound = bind_ltx_rgb_stage(*args)
    for index in (0, 1):
        changed = list(args)
        if index == 0:
            changed[0] = args[0].clone()
            changed[0][0, 0, 0, 0] += .01
        else:
            changed[1] = {"sample_rate": 44100, "waveform": args[1]["waveform"].clone()}
            changed[1]["waveform"][0, 1, 0] += .01
        with pytest.raises(ValueError, match="source, Setup or sampling controls"):
            audit_ltx_rgb_stage(bound[5], *changed, args[4])
    assert json.loads(bound[6])["portable_cache_reuse_authorized"] is False


def test_new_template_checks_half_canvas_full_vae_grid_before_generation(monkeypatch):
    from pathlib import Path

    # conftest exposes h3_t8 for legacy tests; resolve Core's `nodes`, not its
    # same-named package file, for this actual Core-only constructor.
    core_root = Path(comfy.samplers.__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(core_root))
    import nodes
    assert Path(nodes.__file__).resolve() == core_root / "nodes.py"
    from comfy_extras.nodes_lt import EmptyLTXVLatentVideo
    from tools.prepare_h05_ltx_source_workflow import validate_ltx_x2_canvas

    # The actual Core grid constructor, not a trained VAE inference claim.
    width, height = validate_ltx_x2_canvas(256, 448)
    grid = EmptyLTXVLatentVideo.execute(width // 2, height // 2, 121, 1).result[0]["samples"]
    assert tuple(grid.shape) == (1, 128, 16, 7, 4)
    assert tuple(2 * value for value in grid.shape[-2:]) == (height // 32, width // 32)
    bad_grid = EmptyLTXVLatentVideo.execute(144, 256, 121, 1).result[0]["samples"]
    assert tuple(2 * value for value in bad_grid.shape[-2:]) != (512 // 32, 288 // 32)
    for dimensions in ((288, 512), (256, 416), (0, 448), (True, 448)):
        with pytest.raises(ValueError, match="divisible by 64"):
            validate_ltx_x2_canvas(*dimensions)
    assert not torch.cuda.is_initialized()


def test_new_template_vhs_widgets_use_name_map_not_legacy_list():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples/workflows/81-h05-ltx-source/H05_LTX_Source_EXP.json"
    native = json.loads(path.read_bytes())
    output, = (row for row in native["nodes"] if row["id"] == 33)
    assert output["type"] == "VHS_VideoCombine"
    assert isinstance(output["widgets_values"], dict)
    assert output["widgets_values"] == output["widgets_values_named"]
    values = output["widgets_values"]
    assert values["pix_fmt"] == "yuv420p" and values["crf"] == 18
    assert values["pingpong"] is False and values["save_output"] is True
    assert values["trim_to_audio"] is False and values["save_metadata"] is False
    assert not torch.cuda.is_initialized()
