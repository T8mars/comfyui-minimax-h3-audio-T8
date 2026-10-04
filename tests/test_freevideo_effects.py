"""Small CPU effects scope: actual equations/descriptor integrity, not quality."""
import hashlib
from types import SimpleNamespace
import sys
import types

import pytest
import torch

from h3_audio_t8_pkg.freevideo_exp import effects, runtime as fv, worker_effects as worker
from h3_audio_t8_pkg.freevideo_exp.nodes import MiniMaxH3FreeVideoPromptRelayEXPT8


def bound_model():
    embeds, tags = torch.zeros(1, 3, 5120), torch.ones(3, dtype=torch.int64)
    value = dict(schema="t8-freevideo-relay-v1", mode="apply_exp", query_route="video_only_paper", query_chunk_rows=32,
        geometry=fv.geometry(256, 256, 39), events=[dict(text_key_start=1, text_key_end=3, midpoint=1., window=2., sigma=1.)],
        embeddings=fv.tensor_record(embeds), tags_sha256=hashlib.sha256(tags.view(torch.uint8).numpy().tobytes()).hexdigest())
    value["conditions_sha256"] = effects.condition_identity([[embeds, {"minimax_token_tags":tags}]], value["geometry"])
    value["sha256"] = hashlib.sha256(fv.canonical(value).encode()).hexdigest()
    model = fv.FreeVideoModel("unused", "unused", relay=fv.canonical(value))
    return model, [[embeds, {"minimax_token_tags": tags, effects.RELAY_KEY:value}]]


@pytest.mark.parametrize("change", [dict(tau=float("nan")), dict(start=.9,end=.1), dict(mode="unknown"), dict(workspace_mib=0)])
def test_eav_rejects_invalid_descriptor(change):
    with pytest.raises(ValueError):
        effects.with_eav(fv.FreeVideoModel("unused", "unused"), **change)


def test_independent_effect_branches_and_lora_retention():
    base = fv.FreeVideoModel("unused", "unused")
    low = effects.with_eav(base, mode="apply_exp", tau=1)
    high = effects.with_eav(base, mode="report_only", tau=2)
    assert base.eav is None and low.eav != high.eav
    assert fv.with_lora(low, "missing", 0).eav == low.eav
    with pytest.raises(ValueError, match="twice"):
        effects.with_eav(low)


@pytest.mark.parametrize("mutation", ["unpaired", "embeds", "tags", "geometry", "self_hash"])
def test_relay_binding_requires_actual_paired_conditioning(mutation):
    model, cond = bound_model()
    assert effects.transport_effects(model, cond, fv.geometry(256,256,39))["relay"]["mode"] == "apply_exp"
    canvas = fv.geometry(256,256,39)
    if mutation == "unpaired":
        cond[0][1].pop(effects.RELAY_KEY)
    elif mutation == "embeds":
        cond[0][0].add_(1)
    elif mutation == "tags":
        cond[0][1]["minimax_token_tags"][0] = 0
    elif mutation == "geometry":
        canvas = fv.geometry(288,256,39)
    else:
        model = fv.FreeVideoModel("unused", "unused", relay=model.relay.replace('"query_chunk_rows":32', '"query_chunk_rows":64'))
    with pytest.raises(ValueError):
        effects.transport_effects(model, cond, canvas)


def test_reference_audio_native_transport_keeps_exact_channel_time_rows():
    audio = torch.arange(1*32*2*65).reshape(1,32,2,65).float()
    cond = [[torch.zeros(1, 3, 5120), {"minimax_token_tags":torch.ones(3, dtype=torch.int64),
        "minimax_refs":[{"kind":"audio", "audio_latent":audio}]}]]
    tensors, meta = fv.conditioning_transport(cond, fv.geometry(256,256,39))
    assert meta["task"] == "ref2va_audio"
    assert torch.equal(tensors["ref_audio_0"], audio[0].permute(1,2,0).reshape(-1,32))
    visual = torch.zeros(1,24,1,16,16)
    cond[0][1]["minimax_token_tags"][0] = 0
    cond[0][1]["minimax_refs"][0] = {"kind":"video_audio", "latent":visual, "audio_latent":audio}
    tensors, meta = fv.conditioning_transport(cond, fv.geometry(256,256,39))
    assert meta["task"] == "ref2va_av" and meta["refs"][0]["kind"] == "video"
    assert torch.equal(tensors["ref_0"], visual)


def test_relay_schema_does_not_mutate_original_core_schema():
    from h3_audio_t8_pkg.nodes_prompt_relay_advanced import MiniMaxH3PromptRelayConditioningT8Advanced
    schema = MiniMaxH3FreeVideoPromptRelayEXPT8.define_schema()
    assert schema.inputs[0].io_type == "H3_T8_FREEVIDEO_MODEL"
    assert next(i for i in schema.inputs if i.id == "audio_mode").options == ["native", "reference_only"]
    original = MiniMaxH3PromptRelayConditioningT8Advanced.define_schema()
    assert original.inputs[0].io_type == "MODEL"
    assert next(i for i in original.inputs if i.id == "width").default == 1056


def test_all_head_trace_equals_direct_temporal_cfi():
    torch.manual_seed(18)
    frames, spatial, heads, dim = 5, 3, 7, 4
    q, k = [torch.randn(frames*spatial, heads, dim) for _ in range(2)]
    trace = sum(worker.temporal_trace(q[:, start:start+2], k[:, start:start+2], frames, spatial, 4096)[0]
                for start in range(0, heads, 2))
    fullq, fullk = [v.view(frames, spatial, heads, dim).permute(1,2,0,3) for v in (q,k)]
    expected = ((fullq * dim**-.5) @ fullk.transpose(-1,-2)).float().softmax(-1).diagonal(dim1=-2,dim2=-1).sum(dtype=torch.float64)
    torch.testing.assert_close(trace, expected, atol=1e-7, rtol=1e-7)
    with pytest.raises(ValueError, match="one temporal column"):
        worker.temporal_trace(q, k, frames, spatial, 1)


@pytest.mark.parametrize("bridge", ["alpha", "none"])
@pytest.mark.parametrize("rule", ["sana", "vdn"])
def test_query_seed_basis_equals_slow_per_query_recurrence(bridge, rule):
    torch.manual_seed(22)
    frames, heads, dim = 5, 2, 4
    k = torch.randn(frames, heads, 3, dim) * .1
    a = k.transpose(-1,-2) @ k
    b, alpha = torch.randn_like(a), torch.rand(frames,heads,dim)*.3+.5
    class Backend:
        def factor_apply(self, alpha, a, b):
            eye = torch.eye(dim).expand_as(a)
            inverse = torch.linalg.inv(eye + a) if rule == "vdn" else eye - a/3
            return alpha.unsqueeze(-1)*inverse, b@inverse if rule == "vdn" else b/(3**.5)
    backend = Backend()
    basis = worker.scan_basis(backend, alpha, a, b)
    trans, injection = backend.factor_apply(alpha, a, b)
    bounds = [(-1,1),(0,2),(1,3),(2,4),(3,5)]  # Real post-anchor virtual boundaries.
    for f in range(frames):
        seed = torch.randn(heads,dim,dim)
        prefix, suffix = [], [None]*frames
        state = seed
        for j in range(frames):
            state = state@trans[j]+injection[j]
            prefix.append(state)
        state = seed
        for j in range(frames-1,-1,-1):
            state = state@trans[j]+injection[j]
            suffix[j] = state
        lo, hi = bounds[f]
        left, right = (seed if lo <= 0 else prefix[lo-1]), (seed if hi >= frames-1 else suffix[hi+1])
        if bridge == "alpha":
            left = left*alpha[max(lo,0):f+1].prod(0).unsqueeze(-2)
            right = right*alpha[f:hi+1].prod(0).unsqueeze(-2)
        expected = left+right
        torch.testing.assert_close(worker.state_for_frame(basis, alpha, bounds, f, seed, bridge), expected, atol=2e-6, rtol=2e-6)


def test_neutral_policy_delegates_actual_producer_exactly_and_eav_video_only():
    q, k, v = torch.zeros(7,2,4), torch.zeros(7,2,4), torch.ones(7,2,4)
    calls = []
    def original(*args):
        calls.append(args)
        return v.clone()
    state = SimpleNamespace(binding={"mode":"apply_exp", "events":[{},{}]}, neutral=True,
        gain=None, eav={"mode":"report_only"})
    policy = worker.EffectPolicy(original, state)
    layout = SimpleNamespace(video_start=3, video_end=7)
    assert torch.equal(policy(q,k,v,layout,[],.5), v) and len(calls) == 1
    state.gain, state.eav = torch.tensor(1.25), {"mode":"apply_exp"}
    out = policy(q,k,v,layout,[],.5)
    assert torch.equal(out[:3],v[:3]) and torch.equal(out[3:],v[3:]*1.25)
    assert torch.equal(v, torch.ones_like(v))


def test_runtime_routes_native_times_after_reference_audio_and_target_clock(monkeypatch):
    layout = SimpleNamespace(text_start=0, text_len=3, video_start=139, video_end=907,
                             num_frames=12, tokens_per_frame=64)
    hybrids = types.ModuleType("src.models.hybrid_transform")
    hybrids.iter_hybrids = lambda model: iter([SimpleNamespace(layout=layout)])
    monkeypatch.setitem(sys.modules, hybrids.__name__, hybrids)
    constants = types.ModuleType("diffusers.modular_pipelines.minimax_h3.modular_pipeline")
    constants.MINIMAX_H3_TEXT_TAG = 1
    monkeypatch.setitem(sys.modules, constants.__name__, constants)
    binding = dict(mode="apply_exp", query_route="joint_av_exp", embeddings={"shape":[1,3,5120]},
        events=[dict(text_key_start=1,text_key_end=2,midpoint=0.,window=.2,sigma=1.),
                dict(text_key_start=2,text_key_end=3,midpoint=8.,window=.2,sigma=1.)])
    state = worker.Runtime(None, dict(relay=binding), fv.geometry(256,256,39))
    # 6 reference rows before 130 target audio rows. Actual stereo times repeat.
    positions = torch.zeros(907,3,dtype=torch.float64)
    positions[9:139,0] = torch.arange(65).repeat(2) / 10 + 100
    positions[139:,0] = (torch.tensor([0,1,5,9,13,17,18,22,26,30,34,35],dtype=torch.float64)*5/3+200).repeat_interleave(64)
    indices = torch.ones(907,dtype=torch.int64)
    indices[:9] = 0
    args = dict(position_ids=positions, token_tags=torch.cat((torch.ones(3,dtype=torch.int64),torch.zeros(904,dtype=torch.int64))),
        audio_indices=torch.arange(3,139), timestep=torch.tensor([0.,.7]), timestep_indices=indices)
    state.before(None, (), args)
    assert state.progress == pytest.approx(.7) and state.forwards == 1 and not state.neutral
    assert state.query_segments[0]["start"] == 9 and state.query_segments[0]["end"] == 139
    torch.testing.assert_close(state.query_segments[0]["times"], (positions[9:139,0]-positions[9,0]).float())
    torch.testing.assert_close(state.video_times, (positions[139:,0]-positions[139,0]).float())
    args["token_tags"][1] = 0
    with pytest.raises(ValueError, match="non-text"):
        state.before(None, (), args)


def test_batched_text_seed_readout_matches_slow_weighted_nonlinear_reference(monkeypatch):
    torch.manual_seed(34)
    frames, spatial, heads, dim, length = 5,2,3,4,6
    raw = [torch.randn(frames*spatial,heads,dim)*.2 for _ in range(3)]
    text_raw = [torch.randn(length,heads,dim)*.2 for _ in range(3)]
    beta, text_beta = torch.rand(frames*spatial,heads), torch.rand(length,heads)
    alpha = torch.rand(frames,heads,dim)*.2+.7
    times = torch.tensor([0.,1.,5.,9.,13.])*5/3
    bounds = [(-1,1),(0,2),(1,3),(2,4),(3,5)]
    binding = dict(events=[dict(text_key_start=1,text_key_end=3,midpoint=2.,window=.5,sigma=3.),
                           dict(text_key_start=3,text_key_end=6,midpoint=15.,window=1.,sigma=4.)],workspace_mib=4)
    def statistics(k,v,beta,**kwargs):
        a = (k*beta[...,None]).transpose(-1,-2) @ k
        return (a+a.transpose(-1,-2))*.5, (v*beta[...,None]).transpose(-1,-2)@k
    class Backend:
        def factor_apply(self,alpha,a,b):
            inverse = torch.linalg.inv(torch.eye(dim)+a)
            return alpha[...,None]*inverse,b@inverse
    scan, kernels = types.ModuleType("src.models.linear_attention.scan"), types.ModuleType("src.models.linear_attention.kernels")
    scan.frame_statistics = statistics
    kernels.linear_epilogue = lambda output,*args,**kwargs: output
    monkeypatch.setitem(sys.modules, scan.__name__, scan)
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    module = SimpleNamespace(head_dim=dim,a_fp32=True,bridge="alpha",TEXT_STATE_SCALE=.5,
        norm=SimpleNamespace(weight=torch.ones(dim),eps=1e-6), alpha=lambda *args,**kwargs:alpha,
        _delta_backend=lambda *args:Backend(), _feature_one=lambda value,*args,**kwargs:value,
        _features=lambda *args,**kwargs:(raw[0].view(frames,spatial,heads,dim).permute(0,2,1,3),raw[1],raw[2]))
    stats = dict(linear_seed_solver_batches=0,linear_seed_factorizations=0,linear_peak_estimate_bytes=0)
    actual = worker.relay_readout(module,None,frames,spatial,bounds,raw,spatial,None,text_raw,
        heads=slice(0,heads),beta=beta,gate=torch.ones(frames,heads,spatial,dim),frame_mean=None,
        text_beta=text_beta,times=times,binding=binding,stats=stats)
    k,v = [t.view(frames,spatial,heads,dim).permute(0,2,1,3) for t in raw[1:]]
    a,b = statistics(k,v,beta.view(frames,spatial,heads).permute(0,2,1))
    trans,injection = Backend().factor_apply(alpha,a,b)
    expected = torch.empty_like(actual)
    for f in range(frames):
        weights = torch.ones(length)
        for event in binding["events"]:
            weights[event["text_key_start"]:event["text_key_end"]] = (-worker.penalty(times[f],event)).exp()
        ta,tb = statistics(text_raw[1].permute(1,0,2),text_raw[2].permute(1,0,2),text_beta.T*weights)
        seed = Backend().factor_apply(torch.ones(heads,dim),ta,tb)[1]*.5
        left,right = seed,seed
        lo,hi = bounds[f]
        for j in range(max(lo,0)):
            left = left@trans[j]+injection[j]
        for j in range(frames-1,hi,-1):
            right = right@trans[j]+injection[j]
        state = left*alpha[max(lo,0):f+1].prod(0).unsqueeze(-2) + right*alpha[f:hi+1].prod(0).unsqueeze(-2)
        expected[f] = raw[0].view(frames,spatial,heads,dim)[f].permute(1,0,2)@state.transpose(-1,-2)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
    assert stats["linear_seed_factorizations"] == frames*heads
    assert stats["linear_seed_solver_batches"] == 1 and stats["linear_peak_estimate_bytes"] <= 4<<20
