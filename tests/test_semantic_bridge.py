from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from h3_audio_t8_pkg import semantic_bridge as sb
from h3_audio_t8_pkg.nodes_semantic_bridge import (
    MiniMaxH3SemanticBridgeConfigT8, MiniMaxH3SemanticBridgeApplyT8, model_paths,
)


@pytest.fixture(scope="module")
def weight_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("bridge") / "student.safetensors"
    generator = torch.Generator().manual_seed(90217)
    weights = {key: torch.randn(shape, generator=generator) * 0.02 for key, shape in sb.SHAPES.items()}
    save_file(weights, path)
    return path


def config(path, **kwargs):
    return sb.BridgeConfig(str(path), sb.file_sha(path), **kwargs)


def native_conditioning(dtype=torch.float32):
    return [[torch.randn(2, 7, 5120, generator=torch.Generator().manual_seed(71), dtype=dtype),
             {"minimax_token_tags": torch.tensor([0, 1, 1, 0, 1, 1, 1]), "start_percent": 0.25}]]


def oracle(x, state, mode):
    h = x.float()
    x = h / (h.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    x = F.silu(F.linear(x, state["fc1.weight"], state["fc1.bias"]))
    x = F.silu(F.linear(x, state["fc2.weight"], state["fc2.bias"]))
    x = F.linear(x, state["fc3.weight"], state["fc3.bias"])
    if mode == "per_token":
        x = x * ((h.square().mean(-1, keepdim=True) + 1e-8).sqrt() /
                 (x.square().mean(-1, keepdim=True) + 1e-8).sqrt())
    elif mode == "global":
        x = x * ((h.square().mean() + 1e-8).sqrt() / (x.square().mean() + 1e-8).sqrt())
    return (h + 0.10 * (x - h)).to(dtype=h.dtype)


@pytest.mark.parametrize("mode", ["per_token", "global", "none"])
@pytest.mark.parametrize("chunk", [1, 4, 99])
def test_reference_math_and_chunking(weight_file, mode, chunk):
    items = native_conditioning()
    weights, _ = sb.read_weights(weight_file)
    expected = oracle(items[0][0], weights, mode)
    result, report = sb.apply_bridge(items, config(weight_file, magnitude_match=mode, chunk_tokens=chunk))
    torch.testing.assert_close(result[0][0], expected, atol=1e-6, rtol=1e-5)
    assert result[0][1]["minimax_token_tags"] is items[0][1]["minimax_token_tags"]
    assert result[0][1]["start_percent"] == 0.25
    assert sb.RECEIPT_KEY not in items[0][1]
    assert report["items"][0]["shape"] == [2, 7, 5120]
    assert "path" not in report["identity"]


@pytest.mark.parametrize("cfg", [None, sb.BridgeConfig("MISSING", "", alpha=0),
                                 sb.BridgeConfig("MISSING", "", enabled=False)])
def test_bypass_does_not_touch_conditioning_or_files(cfg, monkeypatch):
    original = object()
    monkeypatch.setattr(sb, "_weights", lambda _: pytest.fail("bypass loaded model"))
    result, report = sb.apply_bridge(original, cfg, cancel=lambda: pytest.fail("bypass ran cancellation"))
    assert result is original and not report["applied"]


def test_text_only_preserves_reference_exactly(weight_file):
    original = native_conditioning(torch.bfloat16)
    result, _ = sb.apply_bridge(original, config(weight_file, token_scope="text_only_preserve_reference"))
    assert torch.equal(result[0][0][:, [0, 3]], original[0][0][:, [0, 3]])
    assert result[0][0].dtype == torch.bfloat16
    assert not torch.equal(result[0][0][:, 1], original[0][0][:, 1])


@pytest.mark.parametrize("metadata", [
    {"minimax_prompt_relay_binding": {}}, {"t8_prompt_relay_binding_hash": "x"},
    {sb.RECEIPT_KEY: {}}, {"sensenova_h3_distilled": True},
])
def test_reject_relay_and_repeat_before_load(weight_file, metadata, monkeypatch):
    monkeypatch.setattr(sb, "_weights", lambda _: pytest.fail("invalid condition loaded model"))
    with pytest.raises(ValueError):
        sb.apply_bridge([[torch.ones(1, 1, 5120), metadata]], config(weight_file))


def test_all_items_prevalidated(weight_file, monkeypatch):
    monkeypatch.setattr(sb, "_weights", lambda _: pytest.fail("invalid condition loaded model"))
    with pytest.raises(ValueError, match="5120"):
        sb.apply_bridge(native_conditioning() + [[torch.zeros(1, 2, 5376), {}]], config(weight_file))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_input(weight_file, bad):
    items = native_conditioning()
    items[0][0][0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        sb.apply_bridge(items, config(weight_file))


def test_unknown_tags_fail_text_scope(weight_file):
    items = native_conditioning()
    items[0][1].pop("minimax_token_tags")
    with pytest.raises(ValueError, match="token_tags"):
        sb.apply_bridge(items, config(weight_file, token_scope="text_only_preserve_reference"))


def test_changed_same_name_same_mtime_model(weight_file, tmp_path):
    target = tmp_path / "replace.safetensors"
    target.write_bytes(weight_file.read_bytes())
    cfg = config(target)
    sb.apply_bridge(native_conditioning(), cfg)
    state, _ = sb.read_weights(target)
    stat = target.stat()
    state["fc1.bias"][0] += 1
    save_file(state, target)
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="content changed"):
        sb.apply_bridge(native_conditioning(), cfg)


def _assert_cpu_fp32_cache_states(states):
    assert states
    for cached in states:
        assert type(cached) in (dict, sb._TransWeights, sb._TrainerMLPWeights)
        weights = cached if type(cached) is dict else cached.state
        assert type(weights) is dict and weights
        for tensor in weights.values():
            assert type(tensor) is torch.Tensor
            assert tensor.device.type == "cpu" and tensor.dtype == torch.float32
            assert torch.isfinite(tensor).all()


def _preload_other_bridge_cache(kind, directory):
    """Populate the real loader cache, not synthetic replacements for _weights."""
    path = directory / (kind + ".safetensors")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(930)
        if kind == "trans":
            from h3_audio_t8_pkg.semantic_bridge_trans import TransBridge, TransSpec
            model = TransBridge(TransSpec(64, 1, 4, 64, True, 1.0, None)).eval()
            state = {"net." + key: value.contiguous() for key, value in model.state_dict().items()}
            metadata = {"arch": "trans", "dim": "5120", "hidden": "64",
                        "layers": "1", "heads": "4", "max_tokens": "64",
                        "residual_skip": "True", "residual_scale": "1.0"}
            expected_type = sb._TransWeights
        else:
            assert kind == "trainer_mlp"
            shapes = {"fc1.weight": (8, 5120), "fc1.bias": (8,),
                      "fc2.weight": (8, 8), "fc2.bias": (8,),
                      "fc3.weight": (5120, 8), "fc3.bias": (5120,)}
            state = {"net." + key: torch.randn(shape) * .002 for key, shape in shapes.items()}
            metadata = {"arch": "mlp", "dim": "5120", "hidden": "8",
                        "residual_skip": "True", "residual_scale": ".1",
                        "extra_json": json.dumps({})}
            expected_type = sb._TrainerMLPWeights
        save_file(state, path, metadata=metadata)
    loaded = sb._weights(config(path))
    assert type(loaded) is expected_type
    return loaded


@pytest.mark.parametrize("other_cache", [None, "trans", "trainer_mlp"])
def test_cancel_never_mutates_input(weight_file, tmp_path, other_cache):
    other = _preload_other_bridge_cache(other_cache, tmp_path) if other_cache else None
    other_before = ({key: tensor.clone() for key, tensor in other.state.items()}
                    if other is not None else None)
    items = native_conditioning()
    before = items[0][0].clone()
    calls = []

    def cancel():
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        sb.apply_bridge(items, config(weight_file, chunk_tokens=1), cancel=cancel)
    assert torch.equal(items[0][0], before)
    assert sb.RECEIPT_KEY not in items[0][1]
    with sb._CACHE_LOCK:
        states = list(sb._CACHE.values())
    _assert_cpu_fp32_cache_states(states)
    if other is not None:
        assert any(cached is other for cached in states)
        assert all(torch.equal(tensor, other_before[key]) for key, tensor in other.state.items())


@pytest.mark.parametrize("damage", ["unknown", "empty", "not_tensor", "meta", "half", "nan"])
def test_cache_assertion_does_not_skip_unknown_or_non_cpu_fp32_values(damage):
    cached = {"weight": torch.ones(1)}
    if damage == "unknown":
        cached = object()
    elif damage == "empty":
        cached = {}
    elif damage == "not_tensor":
        cached["weight"] = 1
    elif damage == "meta":
        cached["weight"] = torch.ones(1, device="meta")
    elif damage == "half":
        cached["weight"] = torch.ones(1, dtype=torch.float16)
    else:
        cached["weight"] = torch.full((1,), float("nan"))
    with pytest.raises(AssertionError):
        _assert_cpu_fp32_cache_states([cached])


def test_separate_scheduled_items_global(weight_file):
    one = native_conditioning()
    two = [[one[0][0] * 30, {"start_percent": 0.8}]]
    cfg = config(weight_file, magnitude_match="global", chunk_tokens=3)
    together, _ = sb.apply_bridge(one + two, cfg)
    separate, _ = sb.apply_bridge(two, cfg)
    assert torch.equal(together[1][0], separate[0][0])


def test_converter_lossless_and_no_overwrite(weight_file, tmp_path):
    script = Path(__file__).resolve().parents[1] / "tools/convert_semantic_bridge.py"
    spec = importlib.util.spec_from_file_location("bridge_converter_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "compat.safetensors"
    receipt = module.convert(weight_file, target)
    assert receipt["tensor_identity_verified"]
    assert receipt["source_sha256"] != receipt["output_sha256"]
    a, _ = sb.apply_bridge(native_conditioning(), config(weight_file))
    b, _ = sb.apply_bridge(native_conditioning(), config(target))
    assert torch.equal(a[0][0], b[0][0])
    with pytest.raises(FileExistsError):
        module.convert(weight_file, target)


def test_model_directory_ambiguity_and_disabled_schema(tmp_path, monkeypatch):
    import folder_paths
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    (first / "same.safetensors").touch()
    (second / "same.safetensors").touch()
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "semantic_bridge", ([str(first), str(second)], set()))
    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path / "empty"))
    choices = model_paths()
    assert len(choices) == 2 and "same.safetensors" not in choices
    result = MiniMaxH3SemanticBridgeConfigT8.execute("not installed", alpha=0)
    assert not result.result[0].active
    assert MiniMaxH3SemanticBridgeApplyT8.define_schema().node_id == "MiniMaxH3SemanticBridgeApplyT8"


def test_converter_racing_destination_is_never_overwritten(weight_file, tmp_path, monkeypatch):
    from tools import convert_semantic_bridge as converter
    target = tmp_path / "racing.safetensors"
    link = converter.os.link

    def race(source, destination):
        target.write_bytes(b"other worker's completed result")
        return link(source, destination)

    monkeypatch.setattr(converter.os, "link", race)
    with pytest.raises(FileExistsError):
        converter.convert(weight_file, target)
    assert target.read_bytes() == b"other worker's completed result"
    assert not list(tmp_path.glob(".bridge-*.partial"))


def test_converter_unsupported_atomic_publish_cleans_partial(weight_file, tmp_path, monkeypatch):
    from tools import convert_semantic_bridge as converter
    target = tmp_path / "unsupported.safetensors"

    def unsupported(*_):
        raise OSError("filesystem does not support hard links")

    monkeypatch.setattr(converter.os, "link", unsupported)
    with pytest.raises(OSError, match="hard links"):
        converter.convert(weight_file, target)
    assert not target.exists() and not list(tmp_path.glob(".bridge-*.partial"))


@pytest.mark.parametrize("kwargs", [{"alpha": float("nan")}, {"alpha": 1.01},
                                    {"chunk_tokens": -1}, {"compute_profile": "fake"}])
def test_invalid_settings(weight_file, kwargs):
    with pytest.raises(ValueError):
        replace(config(weight_file), **kwargs)


class RawH3Clip:
    def __init__(self):
        self.last = None

    def tokenize(self, prompt, **kwargs):
        return {"prompt": prompt, "kwargs": kwargs}

    def encode_from_tokens_scheduled(self, tokens):
        self.last = native_conditioning()
        self.last[0][1]["encoding_tokens"] = tokens
        return self.last


def test_ordinary_internal_and_external_application_match(weight_file):
    from h3_audio_t8_pkg.conditioning import build_conditioning
    from helpers import FakeVideoVAE, FakeAudioVAE
    clip = RawH3Clip()
    args = dict(clip=clip, video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(),
                prompt="Two cups", width=128, height=128, length=73, audio_mode="native")
    plain = build_conditioning(**args)
    internal = build_conditioning(**args, semantic_bridge=config(weight_file))
    external, _ = sb.apply_bridge(plain[0], config(weight_file))
    assert torch.equal(internal[0][0][0], external[0][0])
    assert internal[3:5] == plain[3:5]
    for original, bridged in zip(plain[1]["samples"].unbind(), internal[1]["samples"].unbind()):
        assert torch.equal(original, bridged)
    disabled = build_conditioning(**args, semantic_bridge=sb.BridgeConfig("missing", "", alpha=0))
    assert disabled[0] is clip.last
    assert "semantic_bridge" not in disabled[5]


@pytest.mark.parametrize("continuation", [False, True])
def test_long_context_bridge_applied_once_audio_and_layout_untouched(weight_file, continuation):
    import json
    from h3_audio_t8_pkg.long_video import build_long_video_conditioning, LONG_VIDEO_SCHEMA
    from test_long_video import make_context
    from helpers import FakeVideoVAE, FakeAudioVAE
    context = make_context() if continuation else {"schema": LONG_VIDEO_SCHEMA, "empty": True}
    args = dict(clip=RawH3Clip(), video_vae=FakeVideoVAE(), audio_vae=FakeAudioVAE(), context=context,
                segment_index=int(continuation), context_frames=22 if continuation else 0,
                context_audio="video_and_audio", prompt="same scene", width=128, height=128, length=73)
    plain = build_long_video_conditioning(**args)
    bridged = build_long_video_conditioning(**args, semantic_bridge=config(weight_file))
    report = json.loads(bridged[5])
    assert report["context_active"] is continuation
    assert report["semantic_bridge"]["applied"]
    assert len(report["semantic_bridge"]["items"]) == 1
    assert f"context={continuation}" in bridged[0][0][1][sb.RECEIPT_KEY]["encoding_source"]
    assert plain[3:5] == bridged[3:5]
    for original, changed in zip(plain[1]["samples"].unbind(), bridged[1]["samples"].unbind()):
        assert torch.equal(original, changed)
    if continuation:
        assert context["audio_tail"].data_ptr() == bridged[0][0][1]["minimax_refs"][-1]["audio_latent"].data_ptr() or torch.equal(
            plain[0][0][1]["minimax_refs"][-1]["audio_latent"], bridged[0][0][1]["minimax_refs"][-1]["audio_latent"])


def test_relay_receipt_changes_binding_not_token_ranges(weight_file):
    from test_prompt_relay_advanced import NativeLikeFakeClip
    from h3_audio_t8_pkg.prompt_relay_advanced import build_prompt_relay_plan, build_prompt_relay_binding
    clip = NativeLikeFakeClip()
    plan = build_prompt_relay_plan("room", "left\nright", 73, "auto_equal", "", "paper_v1", .1, False, False)[0]
    tokens = clip.tokenize(plan["compiled_prompt"])
    count = len(tokens["qwen3vl_32b"][0])
    condition = [[torch.randn(1, count, 5120), {"minimax_token_tags": torch.ones(count, dtype=torch.long)}]]
    plain = build_prompt_relay_binding(clip, plan, plan["compiled_prompt"], condition, tokens)
    bridged, _ = sb.apply_bridge(condition, config(weight_file))
    bound = build_prompt_relay_binding(clip, plan, plan["compiled_prompt"], bridged, tokens)
    assert bound["events"] == plain["events"]
    assert bound["prompt_token_sha256"] == plain["prompt_token_sha256"]
    assert bound["binding_hash"] != plain["binding_hash"]
    assert bound["semantic_bridge_receipts"] == [bridged[0][1][sb.RECEIPT_KEY]["receipt_sha256"]]


def test_model_cache_concurrent_first_load_once(weight_file, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    with sb._CACHE_LOCK:
        sb._CACHE.clear()
    original = sb.read_weights
    reads = []

    def counted(*args, **kwargs):
        reads.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(sb, "read_weights", counted)
    cfg = config(weight_file)
    with ThreadPoolExecutor(max_workers=3) as pool:
        states = list(pool.map(lambda _: sb._weights(cfg), range(3)))
    assert len(reads) == 1
    assert all(state is states[0] for state in states)


def test_autocast_cannot_change_fp32_profile(weight_file):
    items, cfg = native_conditioning(), config(weight_file)
    expected, _ = sb.apply_bridge(items, cfg)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual, _ = sb.apply_bridge(items, cfg)
    assert torch.equal(actual[0][0], expected[0][0])


def test_schema_inputs_are_append_only_on_inherited_dual_node():
    from h3_audio_t8_pkg.nodes_long_video_dual_model import MiniMaxH3DualModelLongVideoEXPT8
    from h3_audio_t8_pkg.nodes_long_video_in_node_loop_effects_advanced import MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced
    dual = [value.id for value in MiniMaxH3DualModelLongVideoEXPT8.define_schema().inputs]
    single = [value.id for value in MiniMaxH3LongVideoInNodeLoopEffectsT8Advanced.define_schema().inputs]
    assert dual[-4:] == ["color_match_mode", "semantic_bridge", "semantic_bridge_pass1", "semantic_bridge_pass2"]
    assert single[-1] == "semantic_bridge"
    assert len(dual) == len(set(dual))
