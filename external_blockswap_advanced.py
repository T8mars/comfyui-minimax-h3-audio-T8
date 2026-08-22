from __future__ import annotations

from dataclasses import asdict, dataclass
import json

import torch


SCHEMA = "t8.minimax_h3.external_blockswap_bridge.v1"
UPSTREAM = "https://github.com/xiaolibai-sys/ComfyUI-MiniMaxH3"
UPSTREAM_REVISION = "099aa38c122cea030ce45a51eb1d83208b16a363"
TOTAL_BLOCKS = 50


@dataclass(frozen=True)
class H3ExternalBlockSwapConfig:
    """Public duck-typed contract consumed by the external H3 streaming sampler."""

    enabled: bool = True
    block_to_swap: int = 47
    hot_blocks: int = 0
    prefetch: bool = True
    prefetch_count: int = 2
    pin_memory: bool = True
    disk_workers: int = 2
    auto_vram: bool = True
    vram_reserve_mb: float = 0.0
    runtime_lora_total_mb: float = 0.0
    runtime_lora_fixed_mb: float = 0.0
    offload_dit: bool = False
    dtype: str = "bfloat16"

    def window_size(self, total_blocks: int) -> int:
        return max(1, min(int(total_blocks) - self.block_to_swap, int(total_blocks)))

    @property
    def torch_dtype(self) -> torch.dtype:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.dtype]


def external_blockswap_available() -> bool:
    try:
        import nodes

        required = {
            "MiniMaxH3Loader",
            "MiniMaxH3KSampler",
            "MiniMaxH3BlockSwapArgs",
        }
        return required <= set(nodes.NODE_CLASS_MAPPINGS)
    except Exception:
        return False


def build_external_blockswap_config(
    profile: str,
    block_to_swap: int,
    hot_blocks: int,
    prefetch: bool,
    prefetch_count: int,
    pin_memory: bool,
    disk_workers: int,
    auto_vram: bool,
    vram_reserve_mb: float,
    offload_dit: bool,
    dtype: str,
    require_external_runtime: bool,
):
    if require_external_runtime and not external_blockswap_available():
        raise RuntimeError(
            "The external streaming MiniMax H3 runtime is not installed. This bridge only "
            f"connects to {UPSTREAM}; it does not alter ComfyUI's official MODEL loader."
        )
    if profile not in {"upstream_default_auto", "manual"}:
        raise ValueError(f"Unknown BlockSwap profile {profile!r}")
    if profile == "upstream_default_auto":
        block_to_swap = 47
        hot_blocks = 0
        prefetch = True
        prefetch_count = 2
        pin_memory = True
        disk_workers = 2
        auto_vram = True
        vram_reserve_mb = 0.0
        offload_dit = False
        dtype = "bfloat16"

    block_to_swap = int(block_to_swap)
    hot_blocks = int(hot_blocks)
    prefetch_count = int(prefetch_count)
    disk_workers = int(disk_workers)
    if not 0 <= block_to_swap <= TOTAL_BLOCKS:
        raise ValueError("block_to_swap must be between 0 and 50")
    if not 0 <= hot_blocks <= TOTAL_BLOCKS:
        raise ValueError("hot_blocks must be between 0 and 50")
    if not 1 <= prefetch_count <= 8:
        raise ValueError("prefetch_count must be between 1 and 8")
    if not 1 <= disk_workers <= 16:
        raise ValueError("disk_workers must be between 1 and 16")
    if float(vram_reserve_mb) < 0:
        raise ValueError("vram_reserve_mb cannot be negative")
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"Unsupported BlockSwap dtype {dtype!r}")

    config = H3ExternalBlockSwapConfig(
        enabled=True,
        block_to_swap=block_to_swap,
        hot_blocks=hot_blocks,
        prefetch=bool(prefetch),
        prefetch_count=prefetch_count,
        pin_memory=bool(pin_memory),
        disk_workers=disk_workers,
        auto_vram=bool(auto_vram),
        vram_reserve_mb=float(vram_reserve_mb),
        offload_dit=bool(offload_dit),
        dtype=dtype,
    )
    window = config.window_size(TOTAL_BLOCKS)
    report = {
        "schema": SCHEMA,
        "status": "external_blockswap_config_ready",
        "profile": profile,
        "upstream": UPSTREAM,
        "upstream_revision_reviewed": UPSTREAM_REVISION,
        "external_runtime_detected": external_blockswap_available(),
        "config": asdict(config),
        "derived": {
            "total_blocks": TOTAL_BLOCKS,
            "requested_resident_window": window,
            "requested_gpu_slots_before_auto_vram": window + prefetch_count,
            "hot_blocks_capped_by_upstream_to": min(hot_blocks, max(0, window - 1)),
        },
        "compatibility": {
            "output_socket": "MINIMAX_H3_SWAP",
            "consumer": "external MiniMaxH3KSampler only",
            "official_comfy_model_supported": False,
            "stable_t8_workflow_changed": False,
        },
        "claims": {
            "universal_oom_prevention": False,
            "validated_16gb_profile": False,
            "upstream_auto_vram_remains_authoritative": bool(auto_vram),
        },
        "notes": [
            "The upstream project currently does not declare a source-code license, so this node implements only its public payload contract and does not copy its runtime.",
            "BlockSwap applies only to the external streaming loader and sampler path; official ComfyUI MODEL workflows remain unchanged.",
        ],
    }
    return config, json.dumps(report, ensure_ascii=False, indent=2)
