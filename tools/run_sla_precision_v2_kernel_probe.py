#!/usr/bin/env python3
"""Run one low-load numeric probe for the vendored SLA Precision V2 kernel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import torch


SCHEMA = "t8.minimax_h3.sla_precision_v2.kernel_probe.v1"
SEED = 2609023201
BATCH = 1
SEQUENCE = 513
HEADS = 2
HEAD_DIM = 128
BLOCK_SIZE = 32
SPARSITY_RATIO = 0.90


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reference_pool(x: torch.Tensor, block: int) -> torch.Tensor:
    values = []
    for start in range(0, int(x.shape[1]), int(block)):
        values.append(x[:, start : start + block].float().mean(dim=1))
    return torch.stack(values, dim=2)


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lut: torch.Tensor,
    block_q: int,
    block_k: int,
) -> torch.Tensor:
    batch, length, heads, dimension = q.shape
    output = torch.empty(
        (batch, length, heads, dimension), device=q.device, dtype=torch.float32
    )
    scale = dimension**-0.5
    for batch_index in range(batch):
        for head_index in range(heads):
            for query_block in range(int(lut.shape[2])):
                query_start = query_block * block_q
                query_stop = min(length, query_start + block_q)
                key_indices = []
                for key_block in lut[batch_index, head_index, query_block].tolist():
                    key_start = int(key_block) * block_k
                    key_indices.extend(range(key_start, min(length, key_start + block_k)))
                index = torch.tensor(key_indices, device=q.device, dtype=torch.long)
                q_part = q[batch_index, query_start:query_stop, head_index].float()
                k_part = k[batch_index, index, head_index].float()
                v_part = v[batch_index, index, head_index].float()
                probability = torch.softmax(q_part @ k_part.transpose(0, 1) * scale, dim=-1)
                output[batch_index, query_start:query_stop, head_index] = probability @ v_part
    return output


def _metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate = candidate.float()
    reference = reference.float()
    difference = candidate - reference
    mean_abs_reference = float(reference.abs().mean().item())
    relative_mae = float(difference.abs().mean().item()) / max(
        mean_abs_reference, 1.0e-12
    )
    rmse = float(torch.sqrt(torch.mean(difference.square())).item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            candidate.reshape(1, -1), reference.reshape(1, -1), dim=1
        ).item()
    )
    return {
        "relative_mae": relative_mae,
        "rmse": rmse,
        "cosine_similarity": cosine,
        "max_abs_error": float(difference.abs().max().item()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/sla-precision-v2-kernel-probe-20260902/report.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else project_root / args.output
    sys.path.insert(0, str(project_root))

    if not torch.cuda.is_available():
        raise RuntimeError("SLA Precision V2 kernel probe requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        raise RuntimeError(f"this release probe is pinned to sm89, observed sm{major}{minor}")

    from sla_precision_v2_vendor.block_map import get_block_map
    from sla_precision_v2_vendor.kernel import (
        block_sparse_attention,
    )

    generator = torch.Generator(device="cuda").manual_seed(SEED)
    shape = (BATCH, SEQUENCE, HEADS, HEAD_DIM)
    q = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16)

    keep_ratio = 1.0 - SPARSITY_RATIO
    lut, topk = get_block_map(q, k, keep_ratio, BLOCK_SIZE, BLOCK_SIZE)
    torch.cuda.synchronize()
    first = block_sparse_attention(
        q, k, v, lut, topk, BLOCK_SIZE, BLOCK_SIZE
    )
    torch.cuda.synchronize()
    second = block_sparse_attention(
        q, k, v, lut, topk, BLOCK_SIZE, BLOCK_SIZE
    )
    torch.cuda.synchronize()

    pooled_q = _reference_pool(q, BLOCK_SIZE)
    pooled_k = _reference_pool(k, BLOCK_SIZE) - k.float().mean(dim=1)[:, :, None, :]
    scores = pooled_q @ pooled_k.transpose(-1, -2)
    expected_topk = max(1, min(int(scores.shape[-1]), int(keep_ratio * scores.shape[-1])))
    expected_lut = torch.topk(scores, expected_topk, dim=-1, sorted=False).indices
    reference = _reference_attention(q, k, v, lut, BLOCK_SIZE, BLOCK_SIZE)

    numeric = _metrics(first, reference)
    checks = {
        "device_is_sm89": (major, minor) == (8, 9),
        "router_is_fp32": pooled_q.dtype == torch.float32 and pooled_k.dtype == torch.float32,
        "router_topk_matches_independent_fp32": topk == expected_topk
        and torch.equal(
            torch.sort(lut.long(), dim=-1).values,
            torch.sort(expected_lut.long(), dim=-1).values,
        ),
        "tail_sequence_exercised": SEQUENCE % BLOCK_SIZE != 0,
        "finite_output": bool(torch.isfinite(first).all()),
        "repeat_is_bit_exact": bool(torch.equal(first, second)),
        "relative_mae_below_0p2_percent": numeric["relative_mae"] < 0.002,
        "cosine_above_0p999": numeric["cosine_similarity"] > 0.999,
    }
    report = {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "contract": {
            "seed": SEED,
            "shape_blhd": list(shape),
            "dtype": str(q.dtype),
            "block_size_q": BLOCK_SIZE,
            "block_size_k": BLOCK_SIZE,
            "sparsity_ratio": SPARSITY_RATIO,
            "selected_key_blocks": topk,
            "total_key_blocks": math.ceil(SEQUENCE / BLOCK_SIZE),
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": f"sm{major}{minor}",
        },
        "checks": checks,
        "numeric": numeric,
        "claim_boundary": (
            "Low-load random-tensor kernel/router evidence only; this does not prove "
            "H3 visual, speech, speed, or VRAM non-inferiority."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
