"""One bounded additive-bias cuDNN probe; no model or alternative backend."""
import json
from pathlib import Path
import sys
import time

from .runtime import digest, write_json_new


def implementation():
    root = Path(__file__).parent
    return {name: digest(root / name) for name in ("worker_effects.py", "masked_probe.py")}


def validate(config, identity):
    root = Path(config["home"]) / "masked-probe-v1"
    result = json.loads((root / "result.json").read_text(encoding="utf8"))
    if (result.get("status") != "complete" or result.get("finite") is not True
            or result.get("identity") != identity or result.get("implementation") != implementation()
            or result.get("rmse", 1) > .01 or result.get("backend") != "CUDNN_ATTENTION"):
        raise ValueError("FreeVideo masked cuDNN probe differs from the current implementation/runtime")
    return result


def main():
    config_path, root = Path(sys.argv[1]), Path(sys.argv[2])
    started = time.monotonic()
    while not (root / "launch-gate").is_file():
        if time.monotonic() - started > 30:
            raise RuntimeError("Masked probe was not assigned to its own supervisor")
        time.sleep(.05)
    config = json.loads(config_path.read_text(encoding="utf8"))
    sys.path.insert(0, config["source_root"])
    import os
    os.environ["FREEVIDEO_VDN_ROOT"] = config["vdn_root"]
    from freevideo_engine.paths import add_vdn
    add_vdn()
    from freevideo_engine.hardware import _detect_local
    from .probe_identity import identity_for
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    identity, source = identity_for(config, _detect_local()), implementation()
    generator = torch.Generator(device="cpu").manual_seed(917)
    # Match the actual sliced-projection view: rows, eight heads, D128.
    q, k, v = [torch.randn(rows, 8, 128, generator=generator).to(torch.bfloat16)
               for rows in (131, 519, 519)]
    bias = -(torch.arange(131)[:, None] / 40 - torch.arange(519)[None, :] / 160).square()
    bias[:, 173:] = 0
    bias = bias.to(torch.bfloat16)
    reference = ((q.double().permute(1, 0, 2) @ k.double().permute(1, 2, 0)) * 128 ** -.5 + bias.double()).softmax(-1) @ v.double().permute(1, 0, 2)
    q, k, v, bias = [t.cuda() for t in (q, k, v, bias)]
    parts = []
    for start in (0, 128):
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            parts.append(torch.nn.functional.scaled_dot_product_attention(q[start:start + 128].transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0),
                attn_mask=bias[start:start + 128], scale=128 ** -.5, dropout_p=0., is_causal=False)[0].cpu())
    output = torch.cat(parts, dim=1).double()
    finite, rmse = bool(torch.isfinite(output).all()), float((output - reference).square().mean().sqrt())
    if not finite or rmse > .01 or identity_for(config, _detect_local()) != identity or implementation() != source:
        raise ValueError("Masked cuDNN probe failed numerical or identity checks")
    result = dict(status="complete", finite=finite, rmse=rmse, backend="CUDNN_ATTENTION",
        identity=identity, implementation=source, query_rows=131, key_rows=519, heads=8, dimension=128,
        query_chunks=[128, 3], dtype="bfloat16", reference="CPU float64 exact same inputs/bias", no_model=True)
    write_json_new(root / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
