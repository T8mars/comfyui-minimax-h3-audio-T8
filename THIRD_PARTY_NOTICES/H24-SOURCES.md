# H24 experimental adapters: fixed sources and changes

These are independently written T8 adapters to the existing ComfyUI runtime.
The entries below identify algorithm and interchange references, not a claim
that the upstream frameworks, example media, evaluation packages or model
weights are bundled. Experimental code availability is separate from trained
model, performance and human quality qualification.

| Area | Fixed source | Code license |
| --- | --- | --- |
| Stacked-hop PTD grid, head-bank fusion and physical audio correction | [Nynxz/ComfyUI-PTD-Compact](https://github.com/Nynxz/ComfyUI-PTD-Compact/tree/a50419a703f0cfe9b9212e01c6a2f7153d0592b1), `pdd.py` | Apache-2.0; [license](H24-APACHE2.txt), [unmodified upstream NOTICE](H24-COMPACT-NOTICE.txt). |
| Schema-1 row quantization interchange | [Jalen-Brunson/ComfyUI-MiniMax-H3-PTD](https://github.com/Jalen-Brunson/ComfyUI-MiniMax-H3-PTD/tree/4f283cc8c78c489dddb9af852cedfae520092b3a), `ptd_core.py` | Apache-2.0; [license](H24-APACHE2.txt). No NOTICE file in the retrieved fixed tree. |
| Periodic AV views and decoder wrapping | [pxl-pshr/ComfyUI-MiniMaxH3-Loop](https://github.com/pxl-pshr/ComfyUI-MiniMaxH3-Loop/tree/7583d64c315ad8ce6e8e74d989c0b730de10012e), `__init__.py` | GPL-3.0; [license](H24-GPL3.txt), fixed license blob `f288702d2fa16d3cdf0035b15a9fcbc552cd88e7`. |
| Short-clip structural rerendering | [Enderfga/SolarRenderer](https://github.com/Enderfga/SolarRenderer/tree/00b89093234a194bb06e12b7df0d2ed354769a8d), `comfyui/nodes.py` | Apache-2.0; [license](H24-SOLAR-LICENSE.txt), [unmodified upstream NOTICE](H24-SOLAR-NOTICE.txt). SolarRenderer authors, Copyright 2026. |

T8 differences include request-scoped state, restoration of user wrappers and
hooks, native packed-layout row gathering, literal signed/zero row-scale decoding, actual per-hop consumption reports,
explicit control-owner contracts, and rational frame/PCM clock audits. The
existing EPFL PTD recipes, ordinary sampling defaults and saved workflows are
retained. Decoding INT8 storage into BF16 is not INT8 inference and does not
establish a runtime memory or speed improvement.

The original MiniMax-H3 architecture and Core native APIs retain their upstream
attribution. EPFL VITA Lab and taobao-content-ai provide the PTD training work
identified in the Compact NOTICE. The Solar NOTICE describes additional parts
of its original repository; PRoPE, Reactor and evaluation dependencies are not
copied by these T8 thin adapters.

## Separate model assets

Code licenses do not replace model terms. MiniMax H3 and these derivative
weights are subject to the publisher's MiniMax H3 Community License Agreement
and applicable use terms. The private development downloads retain their
complete license and NOTICE files; weights are not included here.

- Compact Ref2VA 4-hop: [fixed model revision](https://huggingface.co/nynxz/MiniMax-H3-PTD-LoRAs/tree/d3a95b93196480b1570473ea2d1cfdd53d4fef72), `ptd/ptd_h3_ref2va_4step_int8.safetensors`; expected SHA256 `f12f71bace7759c9bab9c9270976e4289f3c72e673ef3c5a8902429abd788b1b`.
- Solar short-clip pair: [fixed model revision](https://huggingface.co/Enderfga/SolarRenderer/tree/c5512be825f297ed8db390cbeda0289fb0c24779), `solarrenderer_4step_lora.safetensors` and `solarrenderer_control_lora.safetensors`. The causal model pair is a distinct asset and is outside this prototype.
- Solar encoded silence is a separate binary asset from the fixed source commit, Git blob `76c172973af091675fb42b88386319859b0be8fc`. Its presence is not evidence that a source recording drives the model; restoring a source soundtrack at final mux is a separate operation.

Fixed-license provenance: Compact/Jalen Apache text blob `d645695673349e3947e8e5ae42332d0ac3164cd7`; Compact NOTICE `0eaee926d818b4a6b6770deef4b0379f63693bc6`; Solar root LICENSE `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`, root NOTICE `b6eab815a221f977f4050083ef1b6a0a4471d56a`. Retrieved source evidence and model download receipts are kept in the private development record, not shipped as a daily handoff package.
