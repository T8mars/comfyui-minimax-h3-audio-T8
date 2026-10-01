"""Prepared-input variant of the proven three-update worker; no implicit TE/adapter.

The original fixed-case GPU worker and its receipts remain unchanged. This
variant requires CPU-qualified prepared AV/metadata and a matching prompt cache.
"""
import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import sys
import time

from backend_files import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    path = parser.parse_args().request
    root = path.parent
    request = json.loads(path.read_text())
    model = torch = None
    handles = []
    report = {"status": "incomplete", "scope": __doc__}
    def progress(stage, **extra):
        value = {"stage": stage, "time": time.time(), **extra}
        (root / "worker-live.json").write_text(json.dumps(value))
        print(json.dumps(value), flush=True)
    try:
        assert request["schema"] == "t8-ltx-prepared-refinement-v1"
        for filename, digest in request["identities"].items():
            assert sha(filename) == digest, filename
        for asset in request["model_identities"]:
            info = Path(asset["path"]).stat()
            assert (info.st_size, info.st_mtime_ns) == (asset["bytes"], asset["mtime_ns"])
        sys.path[:0] = request["isolated_paths"]
        import torch as torch_module
        torch = torch_module
        from safetensors.torch import save_file
        from torch.nn.attention import SDPBackend
        from ltx_core.model.transformer.attention import PytorchAttention
        from ltx_core.model.transformer.transformer import TransformerOpsConfig
        from ltx_core.model.transformer import X0Model
        from ltx_core.components.diffusion_steps import EulerDiffusionStep
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
        from ltx_core.tools import AudioLatentTools, VideoLatentTools
        from ltx_core.types import AudioLatentShape, VideoLatentShape
        from ltx_pipelines.utils.denoisers import SimpleDenoiser
        from ltx_pipelines.utils.samplers import euler_denoising_loop
        from ltx_prepared_request import load_prepared, geometry
        from ltx_dequantized_lora import load_joint_cpu
        from ltx_text_weight_offload import offload_native_module
        from ltx_gpu_fusion_gate import run_gate
        torch.set_num_threads(4)
        # Fail before CUDA/model preparation for wrong text, shape or provenance.
        inputs, cache = load_prepared(request)
        effects = None
        if "effects" in request:
            from ltx_external_effects import PreparedLTXEffects
            effects = PreparedLTXEffects(request["effects"], request["geometry"],
                                         request["prompt"], request["identities"])
        shape, _, _ = geometry(request['geometry'])
        assert str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-").lower() == request["gpu_uuid"].removeprefix("GPU-").lower()
        # Native SDPA protocol, but forbid a materializing MATH fallback for 20k tokens.
        ops = TransformerOpsConfig.from_functions(
            attention=PytorchAttention([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]),
            masked_attention=PytorchAttention([SDPBackend.EFFICIENT_ATTENTION]))
        progress("bounded_native_CUDA_fusion_preflight")
        report["fusion_gate"] = run_gate()
        (root / "fusion-gate.json").write_text(json.dumps(report["fusion_gate"], indent=2))
        progress("loading_CPU_base_with_one_weight_native_CUDA_LoRA_fusion")
        model, report["weight_preparation"] = load_joint_cpu(request["base"], request["lora"], ops=ops,
            fusion_device="cuda:0",
            progress=lambda counts: progress("preparing_CPU_weights", **counts))
        assert len(model.transformer_blocks) == 48
        dtype, device = torch.bfloat16, torch.device("cuda:0")
        vt = VideoLatentTools(VideoLatentPatchifier(1), VideoLatentShape.from_pixel_shape(shape), fps=24)
        at = AudioLatentTools(AudioPatchifier(1), AudioLatentShape.from_video_pixel_shape(shape))
        calls = []
        def completed(index):
            calls.append(index)
            if index == 47:
                progress("native_joint_AV_forward_complete", complete_forwards=len(calls) // 48)
        handles = [block.register_forward_hook(lambda _m, _a, _o, i=i: completed(i))
                   for i, block in enumerate(model.transformer_blocks)]
        with torch.inference_mode():
            video = vt.create_initial_state(device, dtype, inputs["video"].to(device))
            audio = at.create_initial_state(device, dtype, inputs["audio"].to(device))
            assert torch.all(video.denoise_mask == 1) and torch.all(audio.denoise_mask == 1)
            sigmas = torch.tensor([.909375, .725, .421875, 0], dtype=torch.float32, device=device)
            noiser = GaussianNoiser(torch.Generator(device=device).manual_seed(request["seed"]))
            # Same single generator order as native recipe: video first, then audio.
            video, audio = noiser(video, float(sigmas[0])), noiser(audio, float(sigmas[0]))
            video_context, audio_context = cache.contexts(request['prompt'], device)
            progress("three_joint_native_Euler_updates")
            start = time.perf_counter()
            with (effects.installed(model) if effects is not None else nullcontext()):
                if effects is not None:
                    video_context = effects.bind_context(video_context)
                denoiser = SimpleDenoiser(video_context, audio_context)
                if effects is not None:
                    denoiser = effects.denoiser(denoiser)
                with offload_native_module(model, model.transformer_blocks, device,
                                           minimum_free_bytes=2 * 1024**3) as lease:
                    refined_video, refined_audio = euler_denoising_loop(sigmas, video, audio, EulerDiffusionStep(),
                        X0Model(model), denoiser)
                    torch.cuda.synchronize()
            if effects is not None:
                report["external_video_effects"] = effects.report()
            report["sampling_seconds_including_weight_transfer"] = time.perf_counter() - start
            result_video = vt.unpatchify(refined_video).latent.cpu().contiguous()
            result_audio = at.unpatchify(refined_audio).latent.cpu().contiguous()
        assert calls == list(range(48)) * 3 and len(lease["block_calls"]) == 144
        assert all(c["completed"] for c in lease["block_calls"])
        assert result_video.shape == inputs["video"].shape and result_audio.shape == inputs["audio"].shape
        assert torch.isfinite(result_video).all() and torch.isfinite(result_audio).all()
        assert not torch.equal(result_video, inputs["video"]) and not torch.equal(result_audio, inputs["audio"])
        assert all(p.device.type == "cpu" and not p.is_meta for p in (*model.parameters(), *model.buffers()))
        output = root / "refined-latents.safetensors"
        save_file({"video": result_video, "joint_audio_not_for_delivery": result_audio}, str(output),
            metadata={**{key: str(value) for key, value in request['geometry'].items()},
                      "scope": "native 3-update joint AV; original AAC retained separately"})
        for filename, digest in request["identities"].items():
            assert sha(filename) == digest, filename
        for asset in request["model_identities"]:
            info = Path(asset["path"]).stat()
            assert (info.st_size, info.st_mtime_ns) == (asset["bytes"], asset["mtime_ns"])
        report.update(status="prepared_LTX_three_update_latents_pass", output_sha256=sha(output),
            geometry=request['geometry'], prompt=request['prompt'], prepared_inputs_recomputed=False,
            video_shape=list(result_video.shape), audio_shape=list(result_audio.shape),
            sigmas=[.909375, .725, .421875, 0], seed=request["seed"], lora_strength=.8,
            joint_forwards=3, block_calls=144, weight_lease=lease,
            attention="native Pytorch SDPA Flash/Efficient only; no SOL/FA3/MATH fallback",
            text_cache=cache.receipt(), original_audio_delivery="copy original AAC; discard refined audio",
            limits="Real joint refinement from dequantized existing INT8 base in BF16; not original BF16 checkpoint/INT8 activation/Spark speed parity. No video decode/human review yet.")
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            for handle in handles:
                handle.remove()
            if model is not None:
                model.to_empty(device="meta")
            gc.collect()
            if torch is not None:
                torch.cuda.empty_cache()
        except BaseException as exc:
            report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            if not active_error:
                report.update(status="failed", error=report["cleanup_error"])
                raise
        finally:
            (root / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({"status": report["status"]}), flush=True)


if __name__ == "__main__":
    main()
