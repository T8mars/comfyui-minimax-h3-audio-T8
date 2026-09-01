"""Block selection: which key blocks each query block is allowed to see.

Vendored from LightX2V (Apache-2.0):
  lightx2v/common/ops/attn/utils/sla_util_blhd.py
  https://github.com/ModelTC/LightX2V

This is the whole of what "SLA" does at inference time. Mean-pool Q into blocks,
mean-pool a smoothed K into blocks, score the two against each other with one
small matmul, and keep the top ``topk_ratio`` fraction of key blocks per query
block. No weights, nothing trained, nothing to load -- the published SLA LoRA
adapts the *model* to tolerate the resulting sparsity, it does not parameterise
this step.

Three changes against upstream, marked FIX:

1. ``other=0.0`` on the masked load. The masked lanes feed ``tl.sum`` two lines
   later, and Triton leaves them undefined, so the final (partial) block of the
   sequence pooled whatever was in memory. Upstream fixed exactly this in the
   BHSD twin of this file and did not port the fix here.
2. ``max(1, ...)`` on the top-k count, so a short sequence keeps one key block
   rather than zero. Also upstream's BHSD-only fix.
3. Smooth-k is folded into the pooled result instead of being materialised.
   Pooling is a mean over L and the correction is constant along L, so
   ``pool(k - mu) == pool(k) - mu`` exactly. Upstream builds the whole smoothed
   copy of K, which at 768p/15s is a needless ~1.3 GB allocation.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _compress_kernel(
    X,
    XM,
    L: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)

    idx_b = idx_bh // H
    idx_h = idx_bh - idx_b * H

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)

    x_offset = idx_b * L * H * D + idx_h * D
    xm_offset = idx_bh * ((L + BLOCK_L - 1) // BLOCK_L) * D
    # FIX vs upstream: other=0.0 -- these lanes are summed below.
    x = tl.load(
        X + x_offset + offs_l[:, None] * (H * D) + offs_d[None, :],
        mask=offs_l[:, None] < L,
        other=0.0,
    )

    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    x_mean = tl.sum(x, axis=0, dtype=tl.float32) / nx
    tl.store(XM + xm_offset + idx_l * D + offs_d, x_mean.to(XM.dtype.element_ty))


def mean_pool(x, BLK):
    """``(B, L, H, D)`` -> ``(B, H, ceil(L/BLK), D)``, mean over each L block."""
    assert x.is_contiguous()
    B, L, H, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    # fp32, not x.dtype. The Triton reduction already accumulates in fp32; only
    # the store rounds. Upstream stores bf16, which quantises the block *scores*
    # and makes top-k pick different blocks than exact arithmetic would -- and
    # at 85-90% sparsity only 3-4 blocks per query survive, so one wrong pick is
    # a large error. Keeping fp32 here costs a few MB and no measurable time.
    x_mean = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=torch.float32)

    grid = (L_BLOCKS, B * H)
    _compress_kernel[grid](x, x_mean, L, H, D, BLK)
    return x_mean


# How large a nudge a previously-selected block gets, as a fraction of that
# query row's own max score -- relative, not absolute, since raw dot-product
# magnitude varies by layer and by how far into denoising a step is. Not
# exposed as a node parameter: small enough that a genuinely better block
# still wins, big enough to kill a near-tie flip between steps that isn't
# telling you anything real.
_STICKY_BONUS_FRAC = 0.05
# Only selections near the top-k cutoff can realistically flip on the next
# step. Retaining the complete LUT for all 50 H3 layers makes stabilization
# state scale as O(NQ * topk) -- several GB at 768p -- for a nudge that only
# ever matters at the boundary. Eight boundary entries per query row keep
# the intended hysteresis while making the retained state O(NQ) with a
# small constant.
_STICKY_HISTORY_WIDTH = 8


def get_protected_block_ranges(protect_upto, protect_ranges, BLKK, NK):
    """Return merged half-open key-block ranges that must always be selected.

    ``protect_upto`` keeps compatibility with the original prefix-only API.
    ``protect_ranges`` permits precise token spans, notably H3's audio segment,
    without force-selecting intervening text or reference-image tokens.
    """
    token_ranges = []
    if protect_upto > 0:
        token_ranges.append((0, int(protect_upto)))
    for item in protect_ranges or ():
        try:
            start, stop = int(item[0]), int(item[1])
        except (IndexError, TypeError, ValueError):
            continue
        if stop <= start:
            continue
        token_ranges.append((max(0, start), max(0, stop)))

    block_ranges = []
    for start, stop in token_ranges:
        first = min(NK, start // BLKK)
        last = min(NK, (stop + BLKK - 1) // BLKK)
        if first < last:
            block_ranges.append((first, last))

    merged = []
    for first, last in sorted(block_ranges):
        if merged and first <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return tuple(merged)


def get_reference_quota_block_ranges(reference_ranges, protect_upto,
                                     protected_ranges, BLKK, NK):
    """Return reference block ranges excluding blocks already fully pinned.

    Token spans are rounded out to key-block boundaries. Any overlap with the
    language/audio protection is subtracted so the optional reference quota
    never pays for, or counts, a block that is already guaranteed.
    """
    references = get_protected_block_ranges(0, reference_ranges, BLKK, NK)
    protected = get_protected_block_ranges(
        protect_upto, protected_ranges, BLKK, NK
    )
    available = []
    for first, last in references:
        pieces = [(first, last)]
        for protected_first, protected_last in protected:
            next_pieces = []
            for piece_first, piece_last in pieces:
                if protected_last <= piece_first or protected_first >= piece_last:
                    next_pieces.append((piece_first, piece_last))
                    continue
                if piece_first < protected_first:
                    next_pieces.append((piece_first, protected_first))
                if protected_last < piece_last:
                    next_pieces.append((protected_last, piece_last))
            pieces = next_pieces
        available.extend(pieces)
    return tuple(available)


def get_reference_quota_keep_count(block_count, reference_sparsity):
    """Translate reference sparsity into a guaranteed number of key blocks."""
    if reference_sparsity is None or block_count <= 0:
        return 0
    sparsity = max(0.0, min(0.99, float(reference_sparsity)))
    return max(1, min(block_count, math.ceil((1.0 - sparsity) * block_count)))


def get_block_map(q, k, topk_ratio, BLKQ=128, BLKK=128, protect_upto=0,
                  prev_lut=None, protect_ranges=None, return_history=False,
                  stabilize_query_from=0, reference_ranges=None,
                  reference_sparsity=None):
    """Return ``(lut, topk)``: the key blocks each query block should attend to.

    ``q``/``k`` are ``(B, L, H, D)`` contiguous. ``lut`` comes back as
    ``(B, H, ceil(LQ/BLKQ), topk)`` int32, contiguous, ready for the kernel.

    ``protect_upto`` pins the first N tokens into every query block's selection.
    For H3 that is the ``[text | cond | audio]`` prefix, and it exists because
    plain top-k starves audio: at 768p/15s the audio is ~1% of the packed
    sequence (19 key blocks of 1794), so nothing makes a query keep any of it,
    and the smooth-k mean it is scored against is 99% video. The pinned blocks
    are added on top of the top-k budget rather than displacing video, so video
    coverage is unchanged and the extra cost is the prefix itself (~7%).

    ``protect_ranges`` provides the same guarantee for specific half-open token
    spans. H3 uses it for language and audio while excluding large visual-
    reference spans. It is merged with ``protect_upto`` when both are supplied.

    ``reference_ranges`` plus a non-None ``reference_sparsity`` adds a second,
    segment-local selection tier for visual conditioning. For example, 0.80
    guarantees the best-scoring 20% of each reference block range per query,
    while 0.0 guarantees all of it. ``None`` disables the quota. This is
    intentionally additive so reference safety never evicts video blocks that
    the unmodified sparse route would have selected.

    ``prev_lut``, when given the previous call's ``lut`` for this same layer,
    nudges those blocks' scores up before top-k -- pure per-step top-k has no
    memory, so on a near-tie between two similarly-scored blocks the winner
    can flip step to step for no reason tied to actual content, and on fast
    motion that shows up as a faint double-exposure rather than one clean
    pick. The nudge only breaks a close call; a block that's actually a
    better fit this step still wins. Silently ignored if its shape doesn't
    match this call's (e.g. right after a dense step, or the first sparse
    call of a run). ``stabilize_query_from`` is a token offset; query blocks
    before the target-video region are neither retained nor biased.
    """
    pooled_q = mean_pool(q, BLKQ)
    # Smooth-k (SageAttention's trick), folded in rather than materialised.
    mu = k.mean(dim=1, dtype=torch.float32)                  # (B, H, D)
    pooled_k = mean_pool(k, BLKK) - mu[:, :, None, :]

    # GQA, for completeness -- H3 is MHA so this is a no-op there.
    num_q_heads, num_kv_heads = pooled_q.shape[1], pooled_k.shape[1]
    if num_q_heads != num_kv_heads:
        assert num_q_heads % num_kv_heads == 0
        pooled_k = pooled_k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)

    pooled_score = pooled_q @ pooled_k.transpose(-1, -2)      # (B, H, NQ, NK)

    NQ = pooled_score.shape[-2]
    sticky_q_start = min(
        NQ,
        (max(0, int(stabilize_query_from)) + BLKQ - 1) // BLKQ,
    )
    sticky_score = pooled_score[..., sticky_q_start:, :]
    if (
        prev_lut is not None
        and prev_lut.shape[:3] == sticky_score.shape[:3]
        and prev_lut.shape[-1] <= pooled_score.shape[-1]
    ):
        row_scale = sticky_score.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        bonus = (row_scale * _STICKY_BONUS_FRAC).expand(*prev_lut.shape)
        sticky_score.scatter_add_(-1, prev_lut.long(), bonus)

    NK = pooled_score.shape[-1]
    # FIX vs upstream: keep at least one key block.
    topk = max(1, min(NK, int(topk_ratio * NK)))

    protected_blocks = get_protected_block_ranges(
        protect_upto, protect_ranges, BLKK, NK
    )
    n_pinned = sum(last - first for first, last in protected_blocks)
    n_reference_quota = 0
    if reference_sparsity is not None:
        reference_sparsity = float(reference_sparsity)
        reference_blocks = get_reference_quota_block_ranges(
            reference_ranges, protect_upto, protect_ranges, BLKK, NK
        )
        for first, last in reference_blocks:
            block_count = last - first
            keep = get_reference_quota_keep_count(
                block_count, reference_sparsity
            )
            if keep <= 0:
                continue
            selected_reference = torch.topk(
                pooled_score[..., first:last], keep, dim=-1, sorted=False
            ).indices + first
            pooled_score.scatter_(-1, selected_reference, float("inf"))
            n_reference_quota += keep

    # Ranking protected blocks above everything else is what pins them;
    # widening topk by the same amount stops them evicting the blocks that the
    # ordinary top-k selection would otherwise keep.
    for first, last in protected_blocks:
        pooled_score[..., first:last] = float("inf")
    if n_pinned > 0 or n_reference_quota > 0:
        topk = min(NK, topk + n_pinned + n_reference_quota)

    selected = torch.topk(pooled_score, topk, dim=-1, sorted=False)
    lut = selected.indices
    lut_i32 = lut.to(torch.int32).contiguous()

    if not return_history:
        return lut_i32, topk

    # Bound what stabilize_motion carries into the next step to the
    # entries nearest the cutoff -- the only ones that can plausibly flip.
    # Strong selections do not need a sticky bonus, so dropping them from
    # the retained history costs nothing but the VRAM they were sitting on.
    history_indices = lut[..., sticky_q_start:, :]
    history_values = selected.values[..., sticky_q_start:, :]
    history_width = min(_STICKY_HISTORY_WIDTH, topk)
    if history_width == topk:
        history = history_indices
    else:
        boundary_pos = torch.topk(
            history_values, history_width, dim=-1, largest=False, sorted=False
        ).indices
        history = torch.gather(history_indices, -1, boundary_pos)

    return lut_i32, topk, history.to(torch.int32).contiguous()
