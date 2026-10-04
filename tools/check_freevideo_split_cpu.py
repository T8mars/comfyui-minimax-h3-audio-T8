"""CPU-only numerical check using the actual pinned schedulers and layout builders.

Tiny deterministic joint-AV transformer, no learned weights or CUDA. Compare
original producer8, new full-range8, and carry-state4+4 on one unchanged canvas.
This proves equations/layout/clocks, not lifted video quality or FP8 performance.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"h3_t8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime_config",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    from freevideo_exp.runtime import load_model,config_for,write_json_new
    config = config_for(load_model(args.runtime_config),full=True)
    sys.path.insert(0,config["source_root"])
    os.environ["FREEVIDEO_VDN_ROOT"] = config["vdn_root"]
    from freevideo_engine.paths import add_vdn
    add_vdn()
    import torch
    torch.set_num_threads(2)
    from freevideo_exp.split_sampler import run
    from freevideo_engine.geometry import sampler_for_canvas
    import src.inference.render as render
    import freevideo_engine.reference_sampler as reference
    from diffusers import MiniMaxH3Scheduler
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config=SimpleNamespace(patch_size=(1,2,2),in_channels=24)
            self.transformer_blocks=torch.nn.ModuleList()
            self.calls=[]
        def forward(self,hidden_states,audio_hidden_states,**kw):
            self.calls.append({k:v.detach().clone() for k,v in dict(video=hidden_states,audio=audio_hidden_states,
                clock=kw["timestep"],clock_indices=kw["timestep_indices"],vi=kw["video_indices"],ai=kw["audio_indices"]).items()})
            return (.03*hidden_states+.02*audio_hidden_states.mean(),
                    .02*audio_hidden_states+.01*hidden_states.mean())
    cases=[("text",None),
        ("first_last",(["first","last"],[torch.full((1,24,1,16,16),.3),torch.full((1,24,1,16,16),.7)])),
        ("audio_reference",[dict(kind="audio",audio_latent=torch.full((12,32),.2))]),
        ("video_audio_reference",[dict(kind="video",latent=torch.full((1,24,1,16,16),.4),audio_latent=torch.full((12,32),.2))])]
    results=[]
    for name,conditions in cases:
        model=Tiny()
        prompt=torch.zeros(5,5120)
        tags=torch.ones(5,dtype=torch.int64)
        if name in ("first_last","video_audio_reference"):
            tags[0]=0
        original=reference.generate_latents if name.endswith("reference") else render.generate_latents
        expected=sampler_for_canvas(original,256,256)(model,prompt,tags,39,8,171,"cpu",conditions=conditions)
        old_calls=model.calls.copy()
        model.calls=[]
        canvas=dict(width=256,height=256)
        actual=run(model,prompt,tags,39,8,171,"cpu",canvas=canvas,start=0,end=8,conditions=conditions)
        for a,b in zip(actual,expected):
            assert torch.equal(a,b),name+" original/full range differs"
        model.calls=[]
        captured={}
        low=run(model,prompt,tags,39,8,171,"cpu",canvas=canvas,start=0,end=4,conditions=conditions,capture=captured)
        assert len(model.calls)==4 and not torch.equal(captured["video_state"],low[0])
        low_calls=model.calls.copy()
        model.calls=[]
        high=run(model,prompt,tags,39,8,171,"cpu",canvas=canvas,start=4,end=8,
            initial=(captured["video_state"],low[1]),carry_video=True,conditions=conditions)
        assert len(model.calls)==4
        for a,b in zip(high,expected):
            assert torch.equal(a,b),name+" real 4+4 state carry differs"
        for a,b in zip(low_calls+model.calls,old_calls):
            assert all(torch.equal(a[key],b[key]) for key in a),name+" layout/reference/timestep differs"
        assert not torch.equal(high[1],low[1]),name+" unfinished audio was frozen"
        # Learned-lift route: explicit restart changes only generated video.
        # First HIGH audio input must still equal the native step4 input.
        model.calls=[]
        run(model,prompt,tags,39,8,172,"cpu",canvas=canvas,start=4,end=8,
            initial=(low[0],low[1]),conditions=conditions)
        assert torch.equal(model.calls[0]["audio"],old_calls[4]["audio"])
        vs,aus=MiniMaxH3Scheduler(shift=12),MiniMaxH3Scheduler(shift=3)
        vs.set_timesteps(8,device="cpu")
        aus.set_timesteps(8,device="cpu")
        assert captured["video_sigmas"]==vs.sigmas.tolist() and captured["audio_sigmas"]==aus.sigmas.tolist()
        results.append(dict(case=name,original_full_range_bit_exact=True,same_canvas_carry4plus4_bit_exact=True,
            audio_continued_not_frozen=True,layout_reference_clock_exact=True,restart_audio_boundary_exact=True))
    assert not torch.cuda.is_initialized()
    report=dict(status="pinned_tiny_CPU_equations_pass_not_learned_quality",cases=results,cuda_initialized=False)
    write_json_new(args.output,report)
    print(json.dumps(report))


if __name__=="__main__":
    main()
