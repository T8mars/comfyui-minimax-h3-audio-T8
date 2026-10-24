"""H21-A2 real tiny Core math and explicit external protocol probes, CPU only.

The tiny VAEs have random weights, not model/quality qualification. Owner wrappers
only adapt channel order; no fake attention, decoder or quantization kernel runs.
The prefetch state is synthetic CPU protocol state, not a GPU allocation claim.
"""
from __future__ import annotations

import inspect
import json
from functools import wraps
from types import SimpleNamespace

import pytest
import torch

import comfy.model_prefetch as prefetch
import comfy.ops
from comfy.ldm.minimax import model as native
from comfy.ldm.minimax import vae as video_core
from comfy.ldm.minimax.audio_vae import MiniMaxH3AudioVAE
from comfy.ldm.modules import attention
from comfy.model_management import InterruptProcessingException
from comfy.patcher_extension import WrapperExecutor

from h3_audio_t8_pkg import enhance_a_video_advanced as eav
from h3_audio_t8_pkg.av_decode_safety_advanced import decode_av_safely
from h3_audio_t8_pkg.core import empty_av_latent
from h3_audio_t8_pkg.h3_core_compat import call_h3_final_layer, cleanup_h3_prefetched_modules, prefetch_h3_block
from test_block_cache_core_compat import cache_nodes as _installed_cache_nodes
from test_block_cache_core_compat import cached_model, compose, real_case
from test_prompt_relay_core_compat import model_fixture


@pytest.fixture(scope="module")
def cache_nodes(record_testsuite_property):
    yield from _installed_cache_nodes.__wrapped__(record_testsuite_property)


def initialize(module, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * .04)
    module.requires_grad_(False)
    return module.eval()


class ThirdPartyVideoOwner:
    """A local executable ownership adapter, not a claim about a shipped plugin."""
    latent_channels, latent_dim = 24, 3

    def __init__(self, core):
        self.first_stage_model = core
        self.calls = []

    def decode(self, samples, *, vae_options=None):
        self.calls.append(("regular", vae_options))
        return self.first_stage_model.decode(samples).movedim(1, -1)

    def decode_tiled(self, samples, *, tile_x, tile_y, overlap, tile_t, overlap_t):
        options = dict(tile_x=tile_x, tile_y=tile_y, overlap=overlap, tile_t=tile_t, overlap_t=overlap_t)
        self.calls.append(("tiled", options))
        return self.first_stage_model.decode_tiled(samples, **options).movedim(1, -1)


class ThirdPartyAudioOwner:
    latent_channels, latent_dim = 32, 2
    audio_sample_rate_output = 32000

    def __init__(self, core):
        self.first_stage_model = core

    def decode(self, samples):
        return self.first_stage_model.decode(samples).movedim(1, -1)


@pytest.fixture
def real_tiny_vae_owners(monkeypatch):
    monkeypatch.setattr(video_core, "optimized_attention", attention.attention_pytorch)
    # Keep the real VAE normalization/chunk/tiling/output-buffer methods. Only
    # decoder dimensions are tiny; every projection/attention/FFN is Core code.
    video = video_core.MiniMaxH3VideoVAE(encoder=False, decoder=False, tile_size=64, tile_overlap_min=16)
    video.post_quant_conv = comfy.ops.disable_weight_init.Conv3d(24, 24, 1)
    video.decoder = video_core.ViT3DDecoder(num_layers=1, heads=1, dim_head=8,
                                           operations=comfy.ops.disable_weight_init)
    initialize(video, 7301)
    audio = MiniMaxH3AudioVAE(encoder_dim=8, latent_dim=16, decoder_dim=128)
    initialize(audio, 7302)
    audio.latents_mean.zero_()
    audio.latents_std.fill_(1.)
    return ThirdPartyVideoOwner(video), ThirdPartyAudioOwner(audio)


@pytest.mark.parametrize("mode", ["decode_regular", "decode_tiled_exp"])
def test_real_tiny_av_decode_preserves_external_owners_and_explicit_keyword_binding(real_tiny_vae_owners, mode):
    video, audio = real_tiny_vae_owners
    value = empty_av_latent(64, 32, 5)[0]
    video_input, audio_input = value["samples"].unbind()
    video_input.copy_(torch.randn(video_input.shape, generator=torch.Generator().manual_seed(12)))
    audio_input.copy_(torch.randn(audio_input.shape, generator=torch.Generator().manual_seed(13)))
    before = (video_input.clone(), audio_input.clone())
    with torch.no_grad():
        expected_video = video.first_stage_model.decode(video_input).movedim(1, -1).flatten(0, 1)
        expected_audio = audio.first_stage_model.decode(audio_input)
        expected_audio /= (expected_audio.std(dim=(1, 2), keepdim=True) * 5).clamp(min=1.)
        frames, sound, _, _, _ = decode_av_safely(value, video, audio, mode=mode,
                                                 video_tile_size=4, video_tile_overlap=1, video_tile_temporal=7)
    assert frames.shape == (5, 32, 64, 3)
    assert sound["waveform"].shape == (1, 2, audio_input.shape[-1] * 800)
    torch.testing.assert_close(frames, expected_video, rtol=0, atol=0)
    torch.testing.assert_close(sound["waveform"], expected_audio, rtol=0, atol=0)
    assert frames.std() > 0 and sound["waveform"].std() > 0
    assert sound["sample_rate"] == 32000
    expected_call = ("regular", None) if mode == "decode_regular" else (
        "tiled", {"tile_x": 4, "tile_y": 4, "overlap": 1, "tile_t": 7, "overlap_t": 1})
    assert video.calls == [expected_call]
    torch.testing.assert_close(video_input, before[0], rtol=0, atol=0)
    torch.testing.assert_close(audio_input, before[1], rtol=0, atol=0)
    assert not torch.cuda.is_initialized()


def test_actual_video_core_output_buffer_and_frames_signature_bind_once(real_tiny_vae_owners):
    owner, _ = real_tiny_vae_owners
    stage = owner.first_stage_model
    z = torch.randn(1, 24, 2, 2, 3, generator=torch.Generator().manual_seed(73))
    inspect.signature(stage._adaptive_decode).bind(z, frames=slice(1, 3))
    inspect.signature(stage.tiled_decode).bind(z, frames=slice(1, 3))
    with torch.no_grad():
        direct = stage.decode(z)
        output = torch.empty_like(direct)
        actual = stage.decode(z, output_buffer=output)
        alias = stage.decode_tiled(z, tile_x=1, tile_y=1)
    assert actual is output
    torch.testing.assert_close(actual, direct, rtol=0, atol=0)
    torch.testing.assert_close(alias, direct, rtol=0, atol=0)


@pytest.mark.parametrize("broken", ["video", "audio"])
def test_real_decode_nonfinite_parameter_error_then_restored_next_request_is_independent(real_tiny_vae_owners, broken):
    video, audio = real_tiny_vae_owners
    value = empty_av_latent(64, 32, 5)[0]
    video_input, audio_input = value["samples"].unbind()
    video_input.normal_(generator=torch.Generator().manual_seed(44))
    audio_input.normal_(generator=torch.Generator().manual_seed(45))
    with torch.no_grad():
        baseline = decode_av_safely(value, video, audio, mode="decode_regular")
        parameter = (video.first_stage_model.post_quant_conv.weight if broken == "video"
                     else audio.first_stage_model.dec_in_proj.weight)
        original = parameter.clone()
        parameter.fill_(float("nan"))
        try:
            with pytest.raises(ValueError, match="Video VAE returned invalid|Audio VAE returned invalid"):
                decode_av_safely(value, video, audio, mode="decode_regular")
        finally:
            parameter.copy_(original)
        after = decode_av_safely(value, video, audio, mode="decode_regular")
    torch.testing.assert_close(after[0], baseline[0], rtol=0, atol=0)
    torch.testing.assert_close(after[1]["waveform"], baseline[1]["waveform"], rtol=0, atol=0)
    assert len(video.calls) == 3  # Exactly one delegation, never TypeError retries.
    assert not torch.cuda.is_initialized()


def test_unknown_real_video_core_class_name_keeps_capable_external_owner_executable(real_tiny_vae_owners):
    video, audio = real_tiny_vae_owners
    stage = video.first_stage_model
    stage.__class__ = type("ExternalDecoderVariant", (type(stage),), {})
    value = empty_av_latent(64, 32, 5)[0]
    with torch.no_grad():
        expected = stage.decode(value["samples"].unbind()[0]).movedim(1, -1).flatten(0, 1)
        frames, _, _, _, report_json = decode_av_safely(value, video, audio, mode="decode_regular")
    torch.testing.assert_close(frames, expected, rtol=0, atol=0)
    report = json.loads(report_json)
    assert report["vae_contract"] == {"video": "video", "audio": "audio"}
    assert report["video_decode_route"]["first_stage_class"] == "ExternalDecoderVariant"
    assert report["video_decode_route"]["global_tile_coordinate_contract"]["state"] == "unknown"
    assert report["decoded"] is True


def real_final_case():
    layer = initialize(native.FinalLayer(8, 4, 24, 32, 1e-5, dtype=torch.float32,
                                        device="cpu", operations=comfy.ops.disable_weight_init), 7321)
    args = (torch.randn(4, 8), torch.randn(1, 4), (0, 2, 0), (2, 4, 0))
    schedule = {"sigma": torch.tensor(.5), "sample_sigmas": torch.tensor([1., .5, 0.]), "shifts": (12., 3.)}
    return layer, args, schedule


def test_unhashable_callable_final_owner_delegates_actual_core_math():
    layer, args, schedule = real_final_case()

    class CallableOwner:
        __hash__ = None

        def __init__(self):
            self.calls = 0

        def __call__(self, x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts):
            self.calls += 1
            return layer(x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts)

    owner = CallableOwner()
    with torch.no_grad():
        expected = layer(*args, **schedule)
        actual = call_h3_final_layer(owner, *args, **schedule)
    assert owner.calls == 1
    for reference, result in zip(expected, actual):
        torch.testing.assert_close(reference, result, rtol=0, atol=0)


def test_final_owner_internal_typeerror_is_not_retried_as_an_old_signature():
    layer, args, schedule = real_final_case()
    calls = []

    def owner(x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts):
        calls.append(True)
        layer(x, t_emb, video_seg, audio_seg, sigma, sample_sigmas, shifts)
        raise TypeError("explicit-third-party-owner-body-failure")

    with torch.no_grad(), pytest.raises(TypeError, match="explicit-third-party-owner-body-failure"):
        call_h3_final_layer(owner, *args, **schedule)
    assert calls == [True]


def prefetched_linear(seed):
    linear = initialize(comfy.ops.disable_weight_init.Linear(8, 8), seed)
    linear._prefetch = {"signature": None}
    linear.third_party_owner = object()
    return linear


def test_actual_core_cleanup_arity_and_none_queue_do_not_need_cuda():
    block = torch.nn.Module()
    linear = prefetched_linear(7322)
    owner = linear.third_party_owner
    before = linear.weight.detach().clone()
    inspect.signature(prefetch.cleanup_prefetched_modules).bind(block, [linear])
    prefetch.cleanup_prefetched_modules(block, [linear])
    assert not hasattr(linear, "_prefetch") and linear.third_party_owner is owner
    torch.testing.assert_close(linear.weight, before, rtol=0, atol=0)
    assert prefetch_h3_block(None, torch.device("cpu"), block) is None
    assert prefetch_h3_block(None, torch.device("cpu"), None) is None
    assert not torch.cuda.is_initialized()


def test_unadapted_external_cache_retains_observed_legacy_cleanup_abi_error(cache_nodes, monkeypatch):
    # Real external function + actual Core cleanup API. Queue bookkeeping is
    # synthetic CPU protocol state; no stream, pinning or GPU allocation occurs.
    block = native.DiTBlock(8, 2, 8, 16, 4, 1e-5, 1e-5, dtype=torch.float32,
                           device="cpu", operations=comfy.ops.disable_weight_init)
    linear = prefetched_linear(7323)
    unrelated = [object()]
    own = [(None, (block, [linear]))]
    monkeypatch.setattr(prefetch, "PREFETCH_QUEUES", [unrelated, own])
    signature = inspect.signature(prefetch.cleanup_prefetched_modules)
    try:
        signature.bind([linear])
    except TypeError:
        assert cache_nodes.PREFETCH_CLEANUP_TAKES_MODULE is False
        with pytest.raises(TypeError, match="comfy_modules"):
            cache_nodes._cleanup_short_circuited_prefetch(SimpleNamespace(blocks=[block]), torch.device("cpu"))
        assert own[0][1][1] == [linear] and hasattr(linear, "_prefetch")
    else:
        # A future genuine one-argument Core needs no compatibility workaround.
        cache_nodes._cleanup_short_circuited_prefetch(SimpleNamespace(blocks=[block]), torch.device("cpu"))
        assert own == [None] and not hasattr(linear, "_prefetch")
    assert len(unrelated) == 1 and linear.third_party_owner is not None
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("layout", ["owned_only", "mixed_foreign", "nested_same_model"])
def test_actual_full_hit_scoped_prefetch_preserves_foreign_and_outer_queues(cache_nodes, monkeypatch, layout):
    patched, runtime, diffusion, options, x, t, context, payload = real_case(cache_nodes, monkeypatch, "report_only")
    original_final = diffusion.final_layer
    linear = prefetched_linear(7330)
    foreign_module, foreign_linear = torch.nn.Module(), prefetched_linear(7331)
    own = [(None, (diffusion.blocks[1], [linear])), diffusion.blocks[2],
           (None, (diffusion.blocks[3], None)), (None, (diffusion.blocks[4], []))]
    foreign = (None, (foreign_module, [foreign_linear]))
    outer_linear = prefetched_linear(7332)
    outer = [(None, (diffusion.blocks[1], [outer_linear]))]
    unrelated = [foreign]
    if layout == "mixed_foreign":
        own.insert(1, foreign)
    queues = ([outer] if layout == "nested_same_model" else []) + [unrelated]
    order = []
    actual_pop, actual_cleanup = prefetch.prefetch_queue_pop, prefetch.cleanup_prefetched_modules

    @wraps(actual_pop)
    def record_pop(queue, device, module, *args, **kwargs):
        if queue is None and module is None and kwargs.get("malloc_scope") == "block":
            order.append("close_malloc_scope")
        return actual_pop(queue, device, module, *args, **kwargs)

    @wraps(actual_cleanup)
    def record_cleanup(module, comfy_modules):
        order.append((module, comfy_modules))
        return actual_cleanup(module, comfy_modules)

    with torch.no_grad():
        full = diffusion(x, t, context, transformer_options=options, minimax_payload=payload)
        monkeypatch.setattr(prefetch, "PREFETCH_QUEUES", queues)
        monkeypatch.setattr(prefetch, "prefetch_queue_pop", record_pop)
        monkeypatch.setattr(prefetch, "cleanup_prefetched_modules", record_cleanup)
        # Match Core timing: this queue is born inside the actual executor, not
        # before it. An actual block pre-hook adds bookkeeping, not fake output.
        handle = diffusion.blocks[0].register_forward_pre_hook(lambda *_: prefetch.PREFETCH_QUEUES.append(own))
        try:
            hit = diffusion(x, t, context, transformer_options=options, minimax_payload=payload)
        finally:
            handle.remove()
    for baseline, result in zip(full, hit):
        torch.testing.assert_close(result, baseline, rtol=1e-5, atol=1e-6)
    assert not hasattr(linear, "_prefetch")
    assert own == ([foreign] if layout == "mixed_foreign" else [None])
    assert order == ["close_malloc_scope", (diffusion.blocks[1], [linear]), (diffusion.blocks[4], [])]
    assert unrelated == [foreign] and hasattr(foreign_linear, "_prefetch")
    assert outer == [(None, (diffusion.blocks[1], [outer_linear]))] and hasattr(outer_linear, "_prefetch")
    assert diffusion.final_layer is original_final and patched.model.diffusion_model is diffusion
    assert [item["block_cache_decision"] for item in runtime.snapshot(consume=False)["forwards"]] == ["full", "hit"]


@pytest.mark.parametrize("error", [RuntimeError, InterruptProcessingException])
def test_actual_outer_prefetch_hit_abort_then_fresh_request_restores_owner(cache_nodes, monkeypatch, error):
    patched, runtime, diffusion, options, x, t, context, payload = real_case(cache_nodes, monkeypatch, "report_only")
    prototype = patched.model_options["transformer_options"][eav.BLOCK_CACHE_KEY]
    options[eav.BLOCK_CACHE_KEY] = prototype
    original_options = {"transformer_options": options}
    guider = SimpleNamespace(model_options=original_options, model_patcher=patched)
    outer = patched.get_wrappers("outer_sample", eav.BLOCK_CACHE_WRAPPER_KEY)
    seen, prefetched = [], []

    def sample(abort):
        live = guider.model_options["transformer_options"]
        seen.append(live[eav.BLOCK_CACHE_KEY])
        with torch.no_grad():
            full = diffusion(x, t, context, transformer_options=live, minimax_payload=payload)
            linear = prefetched_linear(7335 + len(seen))
            prefetched.append(linear)
            monkeypatch.setattr(prefetch, "PREFETCH_QUEUES", [])
            queue = [(None, (diffusion.blocks[1], [linear]))]
            handle = diffusion.blocks[0].register_forward_pre_hook(lambda *_: prefetch.PREFETCH_QUEUES.append(queue))
            try:
                hit = diffusion(x, t, context, transformer_options=live, minimax_payload=payload)
            finally:
                handle.remove()
        assert not hasattr(linear, "_prefetch")
        for baseline, result in zip(full, hit):
            torch.testing.assert_close(result, baseline, rtol=1e-5, atol=1e-6)
        if abort:
            raise error("explicit-after-real-prefetch-hit")
        return hit

    executor = WrapperExecutor.new_class_executor(sample, guider, outer)
    with pytest.raises(error, match="explicit-after-real-prefetch-hit"):
        executor.execute(True)
    assert guider.model_options is original_options and not seen[0].streams
    runtime.snapshot(consume=True)
    result = executor.execute(False)
    assert len(result) == 2 and seen[1] is not seen[0] and seen[0] is not prototype
    assert guider.model_options is original_options and all(not state.streams for state in seen)
    assert all(not hasattr(module, "_prefetch") for module in prefetched)
    assert not prototype.streams and prototype.total_forwards == 0


def test_actual_cleanup_body_failure_after_real_cache_hit_propagates_once(cache_nodes, monkeypatch):
    _, _, diffusion, options, x, t, context, payload = real_case(cache_nodes, monkeypatch, "report_only")
    calls = []
    original_error = TypeError("explicit-real-core-clear-prepared-failure")

    def fail_prepared():
        calls.append(True)
        raise original_error

    linear = prefetched_linear(7338)
    linear.weight_lowvram_function = SimpleNamespace(clear_prepared=fail_prepared)
    own = [(None, (diffusion.blocks[1], [linear]))]
    with torch.no_grad():
        diffusion(x, t, context, transformer_options=options, minimax_payload=payload)
        monkeypatch.setattr(prefetch, "PREFETCH_QUEUES", [])
        handle = diffusion.blocks[0].register_forward_pre_hook(lambda *_: prefetch.PREFETCH_QUEUES.append(own))
        try:
            with pytest.raises(TypeError, match="explicit-real-core-clear-prepared-failure") as observed:
                diffusion(x, t, context, transformer_options=options, minimax_payload=payload)
        finally:
            handle.remove()
    assert observed.value is original_error and calls == [True]
    assert hasattr(linear, "_prefetch")  # Failure is reported, not forged successful cleanup.


def test_cleanup_owner_body_typeerror_is_never_retried_with_another_arity(monkeypatch):
    calls = []

    def cleanup(module, comfy_modules):
        calls.append((module, comfy_modules))
        raise TypeError("explicit-cleanup-body-error")

    monkeypatch.setattr(prefetch, "cleanup_prefetched_modules", cleanup)
    owner, modules = object(), [object()]
    with pytest.raises(TypeError, match="explicit-cleanup-body-error"):
        cleanup_h3_prefetched_modules(owner, modules)
    assert calls == [(owner, modules)]


@pytest.mark.parametrize("position", ["before", "after"])
@pytest.mark.parametrize("mode", ["report_only", "apply_exp"])
def test_real_sparse_owners_preserved_when_cache_is_bypassed_and_audit_is_unverified(cache_nodes, monkeypatch, position, mode):
    patched, runtime, diffusion, options, x, t, context, payload = real_case(cache_nodes, monkeypatch, mode, position)
    selected_dit = dict(options["patches_replace"]["dit"])
    selected_attention = options.get("optimized_attention_override")
    with torch.no_grad():
        first = diffusion(x, t, context, transformer_options=options, minimax_payload=payload)
        second = diffusion(x, t, context, transformer_options=options, minimax_payload=payload)
    assert len(first) == 2 and all(torch.isfinite(value).all() for value in first + second)
    for reference, result in zip(first, second):
        torch.testing.assert_close(reference, result, rtol=0, atol=0)
    assert options["patches_replace"]["dit"] == selected_dit
    assert all(options["patches_replace"]["dit"][key] is value for key, value in selected_dit.items())
    assert options.get("optimized_attention_override") is selected_attention
    assert patched.model.diffusion_model is diffusion
    assert eav.EAV_RUNTIME_KEY not in options
    snapshot = runtime.snapshot(consume=False)
    assert [row["block_cache_decision"] for row in snapshot["forwards"]] == ["bypassed", "bypassed"]
    latent = {"samples": object()}
    returned, report_json = eav.finalize_eav_runtime(latent, runtime)
    report = json.loads(report_json)
    assert returned is latent and report["composition_verified"] is False
    assert report["status"] == "executed_user_stack_unverified"
    assert report["block_cache"]["bypassed_forwards"] == 2
    assert report["block_cache"]["full_forwards"] == report["block_cache"]["cache_hits"] == 0
    assert report["block_cache"]["active_measurements_expected_from_decisions"] is None


def test_real_cache_full_then_empty_user_dit_mapping_reports_actual_bypass(cache_nodes, monkeypatch):
    _, runtime, diffusion, options, x, t, context, payload = real_case(cache_nodes, monkeypatch, "report_only")
    with torch.no_grad():
        first = diffusion(x, t, context, transformer_options=options, minimax_payload=payload)
        options["patches_replace"]["dit"].clear()
        second = diffusion(x, t, context, transformer_options=options, minimax_payload=payload)
    assert options["patches_replace"]["dit"] == {}
    for reference, result in zip(first, second):
        torch.testing.assert_close(reference, result, rtol=0, atol=0)
    assert [row["block_cache_decision"] for row in runtime.snapshot(consume=False)["forwards"]] == ["full", "bypassed"]
    _, report_json = eav.finalize_eav_runtime({"samples": object()}, runtime)
    report = json.loads(report_json)
    assert report["composition_verified"] is False and report["block_cache"]["bypassed_forwards"] == 1
    assert report["block_cache"]["full_forwards"] == 1 and report["block_cache"]["cache_hits"] == 0
    assert report["block_cache"]["active_measurements_expected_from_decisions"] is None


def test_authenticated_external_cache_patch_with_wrong_index_still_reports_real_corruption(cache_nodes):
    cached = cached_model(cache_nodes)
    patch = cached.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)]
    assert type(patch) is cache_nodes.H3BlockPatch and patch.block_index == 0
    patch.block_index = 49
    sigmas = torch.cat((torch.linspace(1., .05, 20), torch.zeros(1)))
    with pytest.raises(RuntimeError, match="boundary patch identity is invalid"):
        eav.build_eav_block_cache_model(cached, sigmas, mode="report_only", tau=.25,
                                       start_video_progress=0., end_video_progress=1.,
                                       max_workspace_mib=4, g_hard_limit=1.5)
    assert patch.block_index == 49  # No hidden repair or replacement of user state.


def test_real_cache_contradictory_consumption_counters_still_raise_after_actual_math(cache_nodes, monkeypatch):
    _, runtime, diffusion, options, x, t, context, payload = real_case(cache_nodes, monkeypatch, "report_only")
    cache = options[eav.BLOCK_CACHE_KEY]
    calls = []

    def corrupt_counter(*_):
        calls.append(True)
        cache.total_forwards += 1

    handle = diffusion.final_layer.register_forward_hook(corrupt_counter)
    try:
        with torch.no_grad(), pytest.raises(RuntimeError, match="unauditable cache transition"):
            diffusion(x, t, context, transformer_options=options, minimax_payload=payload)
    finally:
        handle.remove()
    assert calls == [True] and cache.total_forwards == 2 and cache.full_forwards == 1
    assert "unauditable cache transition" in runtime.snapshot(consume=False)["aborted"]
    assert eav.EAV_RUNTIME_KEY not in options


def test_unknown_cache_wrapper_actually_delegates_real_full_and_hit_without_being_replaced(cache_nodes, monkeypatch):
    _, _, diffusion, _, x, t, context, payload = real_case(cache_nodes, monkeypatch, "report_only")
    source = model_fixture()
    source.model.diffusion_model = diffusion
    cached = cache_nodes.MiniMaxH3BlockCacheNode.execute(source, .12, 0., 1., 2, "cpu", 1, False)[0]
    original_wrapper = cached.wrappers["diffusion_model"][eav.BLOCK_CACHE_WRAPPER_KEY][0]
    calls = []

    def foreign_wrapper(executor, *args, **kwargs):
        calls.append(True)
        return original_wrapper(executor, *args, **kwargs)

    cached.wrappers["diffusion_model"][eav.BLOCK_CACHE_WRAPPER_KEY] = [foreign_wrapper]
    patched, runtime, _ = compose(cached)
    patched.prepare_state(torch.tensor(.5), patched.model_options)
    options = patched.model_options["transformer_options"].copy()
    options.update(wrappers=patched.wrappers, sigmas=torch.tensor([.5]), sample_sigmas=torch.tensor([1., .5, 0.]))
    options[eav.BLOCK_CACHE_KEY] = options[eav.BLOCK_CACHE_KEY].clone().prepare(source.model.model_sampling)
    chosen = dict(options["patches_replace"]["dit"])
    with torch.no_grad():
        first = diffusion(x, t, context, transformer_options=options, minimax_payload=payload)
        second = diffusion(x, t, context, transformer_options=options, minimax_payload=payload)
    assert calls == [True, True]
    assert cached.wrappers["diffusion_model"][eav.BLOCK_CACHE_WRAPPER_KEY] == [foreign_wrapper]
    assert options["patches_replace"]["dit"] == chosen
    for baseline, result in zip(first, second):
        torch.testing.assert_close(result, baseline, rtol=1e-5, atol=1e-6)
    assert [row["block_cache_decision"] for row in runtime.snapshot(consume=False)["forwards"]] == ["full", "hit"]


def test_foreign_block_runs_actual_core_block_and_its_original_error_propagates(cache_nodes, monkeypatch):
    _, runtime, diffusion, options, x, t, context, payload = real_case(cache_nodes, monkeypatch, "report_only")
    calls = []
    original_error = RuntimeError("explicit-foreign-block-error-after-real-math")

    def foreign_block(args, extra):
        output = extra["original_block"](args)
        assert torch.isfinite(output["img"]).all()
        calls.append(True)
        raise original_error

    options["patches_replace"]["dit"][("double_block", 25)] = foreign_block
    with torch.no_grad(), pytest.raises(RuntimeError, match="explicit-foreign-block-error-after-real-math") as observed:
        diffusion(x, t, context, transformer_options=options, minimax_payload=payload)
    assert observed.value is original_error and calls == [True]
    assert options["patches_replace"]["dit"][("double_block", 25)] is foreign_block
    assert eav.EAV_RUNTIME_KEY not in options
    assert "explicit-foreign-block-error-after-real-math" in runtime.snapshot(consume=False)["aborted"]
