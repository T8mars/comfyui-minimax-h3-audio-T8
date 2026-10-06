"""Exact pinned scheduler and actual packed-layout calls, tiny CPU transformer only."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--vdn-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sampling-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    os.environ["FREEVIDEO_VDN_ROOT"] = str(args.vdn_root)
    from freevideo_engine.paths import add_vdn
    add_vdn()
    import torch
    torch.set_num_threads(2)
    from h3_t8.freevideo_quality.profiles import PROFILES, clock, table_identity, raw_grid
    from h3_t8.freevideo_quality.assets import select
    from h3_t8.freevideo_exp.runtime import write_json_new
    from freevideo_engine.geometry import sampler_for_canvas
    from freevideo_engine.adaln import schedule_timesteps
    from freevideo_engine.refine_schedule import timesteps as high_times, COMMUNITY
    from freevideo_engine.adaln_assets import weight_identity
    from freevideo_engine import refine, reference_sampler
    from src.inference import render
    from diffusers import MiniMaxH3Scheduler
    manifest = json.loads((args.cache / "manifest.json").read_text(encoding="utf8"))
    catalog = json.loads((args.source_root / "freevideo_engine/prepared_models.json").read_text(encoding="utf8"))

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(patch_size=(1, 2, 2), in_channels=24)
            self.transformer_blocks = torch.nn.ModuleList()
            self.calls = []

        def forward(self, hidden_states, audio_hidden_states, **kwargs):
            times, ids = kwargs["timestep"], kwargs["timestep_indices"]
            self.calls.append(dict(times=times.cpu().tolist(), row_times=times[ids].clone(),
                audio=audio_hidden_states.clone(), video=hidden_states.clone(),
                vi=kwargs["video_indices"].clone(), ai=kwargs["audio_indices"].clone()))
            return .03 * hidden_states + .02 * audio_hidden_states.mean(), .02 * audio_hidden_states + .01 * hidden_states.mean()

    cases = [("t2va", None), ("fl2va", (["first", "last"], [torch.full((1, 24, 1, 16, 16), .3), torch.full((1, 24, 1, 16, 16), .7)])),
        ("ref2va_audio", [dict(kind="audio", audio_latent=torch.full((12, 32), .2))]),
        ("ref2va_av", [dict(kind="video", latent=torch.full((1, 24, 1, 16, 16), .4), audio_latent=torch.full((12, 32), .2))])]
    results = []
    original_set = MiniMaxH3Scheduler.set_timesteps

    def set_published(self, num_inference_steps=None, device=None, sigmas=None):
        if num_inference_steps == 20 and sigmas is None:
            raw = raw_grid(20)
            sigmas = self.shift * raw / (1 + (self.shift - 1) * raw)
        return original_set(self, num_inference_steps=num_inference_steps, device=device, sigmas=sigmas)

    MiniMaxH3Scheduler.set_timesteps = set_published
    for quality, n in PROFILES.items():
        vs, aus = MiniMaxH3Scheduler(shift=12), MiniMaxH3Scheduler(shift=3)
        vs.set_timesteps(n, device="cpu")
        aus.set_timesteps(n, device="cpu")
        for task, conditions in cases:
            actual_clock = clock(quality, task=task)
            assert actual_clock["video_sigmas"] == vs.sigmas.tolist()
            assert actual_clock["audio_sigmas"] == aus.sigmas.tolist()
            assert actual_clock["modulation_timesteps"] == [t.tolist() for t in schedule_timesteps(n, task=task, device="cpu")]
            table, _ = select(manifest, catalog, quality, "LOW" if quality == "light" else "SINGLE", task)
            assert table["identity"] == table_identity(weight_identity(manifest), quality, "LOW" if quality == "light" else "SINGLE", task)
            model, prompt, tags = Tiny(), torch.zeros(5, 5120), torch.ones(5, dtype=torch.int64)
            if task in ("fl2va", "ref2va_av"):
                tags[0] = 0
            sampler = reference_sampler.generate_latents if task.startswith("ref2va") else render.generate_latents
            video, audio = sampler_for_canvas(sampler, 256, 256)(model, prompt, tags, 39, n, 171, "cpu", conditions=conditions)
            assert len(model.calls) == n
            assert [row["times"] for row in model.calls] == actual_clock["modulation_timesteps"]
            assert torch.isfinite(video).all() and torch.isfinite(audio).all()
            if quality == "light":
                model.calls = []
                high = sampler_for_canvas(refine.generate_latents, 256, 256)(model, prompt, tags, 39, 8, 172, "cpu",
                    initial_latents=(video, audio), refine_steps=3, refine_schedule=COMMUNITY, conditions=conditions)
                high_clock = clock("light", "HIGH", task)
                assert [row["times"] for row in model.calls] == high_clock["modulation_timesteps"]
                assert high_clock["modulation_timesteps"] == [t.tolist() for t in high_times("cpu", task)]
                assert len(model.calls) == 3 and torch.equal(high[1], audio)
                select(manifest, catalog, "light", "HIGH", task)
            results.append(dict(profile=quality, task=task, nfe=n, actual_clock_layout_exact=True,
                community3_audio_exact=quality == "light"))
    MiniMaxH3Scheduler.set_timesteps = original_set
    from h3_t8.freevideo_quality.assets import offline_binding
    from freevideo_engine import adaln, adaln_assets, runtime
    original_cache, original_restore = adaln.TableCache, adaln_assets.restore_projections
    base_table, _ = select(manifest, catalog, "light", "LOW", "t2va")
    max_table, _ = select(manifest, catalog, "max", "SINGLE", "t2va")
    try:
        optional_root = args.sampling_root or args.output.parent
        with offline_binding(args.cache, optional_root, catalog, [(base_table, args.cache), (max_table, optional_root)]) as evidence:
            # Actual production binding, including a LoRA-derived source_id.
            derived = dict(manifest, source_id="a_distinct_LoRA_cache_source_id")
            table = runtime.TableCache(args.output.parent, derived["source_id"], 8, manifest=derived, device="cpu")
            assert len(table.load(0, 8)) == 8 and len(evidence) == 1
            assert table.asset and table.optional_loaded == set()
            assert table.optional_downloaded == set() and table.optional_download_bytes == 0
            assert table.producer == base_table.get("producer")
            vs.set_timesteps(20, device="cpu")
            aus.set_timesteps(20, device="cpu")
            assert vs.sigmas.tolist() == clock("max")["video_sigmas"]
            assert aus.timesteps.tolist() == clock("max")["audio_timesteps"]
            optional = runtime.TableCache(args.output.parent, derived["source_id"], 20, manifest=derived, device="cpu")
            assert optional.asset is None and optional.optional_asset
            assert optional.optional_loaded == set() and optional.optional_downloaded == set()
            assert optional.optional_download_bytes == 0
            if args.sampling_root is not None:
                assert len(optional.load(0, 20)) == 20 and optional.optional_loaded == {0}
                assert optional.optional_downloaded == set() and optional.optional_download_bytes == 0
            for action in (lambda: table.save(0, None), lambda: adaln_assets.restore_projections(args.cache, manifest),
                           lambda: runtime.TableCache(args.output.parent, derived["source_id"], 16, manifest=derived, device="cpu")):
                try:
                    action()
                except ValueError:
                    pass
                else:
                    raise AssertionError("Unprepared calculation/download was not blocked")
            raise RuntimeError("intentional_binding_cleanup")
    except RuntimeError as error:
        assert str(error) == "intentional_binding_cleanup"
    assert adaln.TableCache is original_cache and adaln_assets.restore_projections is original_restore
    assert MiniMaxH3Scheduler.set_timesteps is original_set
    assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
    value = dict(status="pinned_tiny_CPU_pass_not_GPU_or_quality", cases=results, cuda_initialized=False)
    write_json_new(args.output, value)
    print(json.dumps(value))


if __name__ == "__main__":
    main()
