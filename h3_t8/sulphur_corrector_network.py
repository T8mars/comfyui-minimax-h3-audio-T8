"""T8 inference implementation of the public Sulphur-3 corrector tensor ABI.

Layer names and tensor layouts follow SulphurAI's published checkpoint. The
original node package is not imported or required. See docs/SULPHUR3_CORRECTOR.md
for pinned sources, model licensing and numerical comparison boundaries.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CorrectorConfig:
    video_channels: int = 24
    text_dim: int = 5120
    width: int = 1280
    depth: int = 55
    heads: int = 20
    head_dim: int = 64
    mlp_ratio: int = 4
    time_embed_dim: int = 256
    patch: tuple[int, int, int] = (1, 2, 2)
    norm_eps: float = 1e-5

    def __post_init__(self):
        for key in ('video_channels', 'text_dim', 'width', 'depth', 'heads',
                    'head_dim', 'mlp_ratio', 'time_embed_dim'):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f'Corrector {key} must be a positive integer')
        if self.width != self.heads * self.head_dim or self.head_dim % 2:
            raise ValueError('Corrector width must equal heads * head_dim; rotary head_dim must be even')
        if self.time_embed_dim < 2:
            raise ValueError('Corrector time_embed_dim needs at least one sin/cos pair')
        if len(self.patch) != 3 or any(type(x) is not int or x <= 0 for x in self.patch):
            raise ValueError('Corrector patch must contain three positive integers')
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError('Corrector norm_eps must be finite and positive')

    @classmethod
    def from_metadata(cls, metadata):
        if not metadata or 'h3_corrector' not in metadata:
            raise ValueError("Corrector safetensors requires 'h3_corrector' architecture metadata")
        try:
            values = json.loads(metadata['h3_corrector'])
            if not isinstance(values, dict):
                raise TypeError('architecture must be a JSON object')
            if 'patch' in values:
                values['patch'] = tuple(values['patch'])
            return cls(**values)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f'Invalid corrector architecture metadata: {error}') from error

    def to_metadata(self):
        return {'h3_corrector': json.dumps(asdict(self), separators=(',', ':'))}


def _embed_time(value, dimensions):
    half = dimensions // 2
    exponent = torch.arange(half, device=value.device, dtype=torch.float32)
    frequencies = torch.exp(-math.log(10000.) * exponent / half)
    phase = value.float().reshape(-1, 1) * frequencies.reshape(1, -1) * 1000.
    return F.pad(torch.cat([phase.cos(), phase.sin()], -1), (0, dimensions - 2 * half))


def _rotate_axes(tensor, sizes):
    frames, height, width = sizes
    positions = (
        torch.arange(frames, device=tensor.device).repeat_interleave(height * width),
        torch.arange(height, device=tensor.device).repeat_interleave(width).repeat(frames),
        torch.arange(width, device=tensor.device).repeat(frames * height),
    )
    dimensions = tensor.shape[-1]
    temporal = 2 * (dimensions // 8)
    vertical = 2 * ((dimensions - temporal) // 4)
    segments = tensor.split((temporal, vertical, dimensions - temporal - vertical), -1)
    result = []
    for segment, position in zip(segments, positions):
        n = segment.shape[-1] // 2
        if not n:
            result.append(segment)
            continue
        frequency = torch.exp(-math.log(10000.)
                              * torch.arange(n, device=tensor.device, dtype=torch.float32) / n)
        angle = position.float().unsqueeze(-1) * frequency
        cosine = angle.cos().to(tensor.dtype)[None, None]
        sine = angle.sin().to(tensor.dtype)[None, None]
        even, odd = segment[..., ::2], segment[..., 1::2]
        result.append(torch.stack([even * cosine - odd * sine,
                                   odd * cosine + even * sine], -1).flatten(-2))
    return torch.cat(result, -1)


class VideoAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, config.width * 3, dtype=torch.float32)
        self.out = nn.Linear(config.width, config.width, dtype=torch.float32)

    def forward(self, hidden):
        batch, frames, height, width, dim = hidden.shape
        qkv = self.qkv(hidden.flatten(1, 3)).reshape(batch, -1, 3, self.heads, dim // self.heads)
        q, k, v = (item.transpose(1, 2) for item in qkv.unbind(2))
        q, k = (_rotate_axes(item, (frames, height, width)) for item in (q, k))
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        return self.out(attended.transpose(1, 2).reshape_as(hidden))


class CaptionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        for name in ('q', 'k', 'v', 'out'):
            setattr(self, name, nn.Linear(config.width, config.width, dtype=torch.float32))

    def forward(self, hidden, caption):
        batch, *_, dim = hidden.shape
        def heads(value):
            return value.reshape(batch, -1, self.heads, dim // self.heads).transpose(1, 2)
        q, k, v = heads(self.q(hidden.flatten(1, 3))), heads(self.k(caption)), heads(self.v(caption))
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.)
        return self.out(attended.transpose(1, 2).reshape_as(hidden))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.width, config.width * config.mlp_ratio, dtype=torch.float32)
        self.fc2 = nn.Linear(config.width * config.mlp_ratio, config.width, dtype=torch.float32)

    def forward(self, value):
        return self.fc2(F.gelu(self.fc1(value), approximate='tanh'))


class CorrectorBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        for name in ('norm_self', 'norm_cross', 'norm_mlp'):
            setattr(self, name, nn.LayerNorm(config.width, eps=config.norm_eps, dtype=torch.float32))
        self.self_attention = VideoAttention(config)
        self.cross_attention = CaptionAttention(config)
        self.mlp = MLP(config)
        self.modulation = nn.Linear(config.width, 6 * config.width, dtype=torch.float32)

    def forward(self, hidden, time, caption):
        modulation = self.modulation(time).reshape(time.shape[0], 1, 1, 1, -1)
        sa_shift, sa_scale, sa_gate, ff_shift, ff_scale, ff_gate = modulation.chunk(6, -1)
        hidden = hidden + (1 + sa_gate) * self.self_attention(self.norm_self(hidden) * (1 + sa_scale) + sa_shift)
        hidden = hidden + self.cross_attention(self.norm_cross(hidden), caption)
        return hidden + (1 + ff_gate) * self.mlp(self.norm_mlp(hidden) * (1 + ff_scale) + ff_shift)


class H3Corrector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_in = nn.Conv3d(2 * config.video_channels, config.width,
                                  config.patch, stride=config.patch, dtype=torch.float32)
        self.text_norm = nn.LayerNorm(config.text_dim, eps=config.norm_eps, dtype=torch.float32)
        self.text_projection = nn.Linear(config.text_dim, config.width, dtype=torch.float32)
        self.time_mlp = nn.Sequential(nn.Linear(3 * config.time_embed_dim, config.width, dtype=torch.float32),
                                      nn.SiLU(), nn.Linear(config.width, config.width, dtype=torch.float32))
        self.blocks = nn.ModuleList(CorrectorBlock(config) for _ in range(config.depth))
        self.final_norm = nn.LayerNorm(config.width, eps=config.norm_eps, dtype=torch.float32)
        self.final_modulation = nn.Linear(config.width, 2 * config.width, dtype=torch.float32)
        self.patch_out = nn.Linear(config.width, config.video_channels * math.prod(config.patch), dtype=torch.float32)

    def forward(self, video, velocity, sigma, sigma_next, caption):
        if video.ndim != 5 or velocity.shape != video.shape:
            raise ValueError('Corrector needs matching [B,C,T,H,W] video and velocity tensors')
        batch, channels, frames, height, width = video.shape
        pt, ph, pw = self.config.patch
        if channels != self.config.video_channels or any(s % p for s, p in zip(video.shape[2:], self.config.patch)):
            raise ValueError(f'Video latent {tuple(video.shape)} does not fit corrector channels/patch {self.config.patch}')
        if caption.ndim != 3 or caption.shape[0] != batch or caption.shape[-1] != self.config.text_dim:
            raise ValueError(f'Corrector caption must be [B,L,{self.config.text_dim}] with the video batch size')
        before = torch.as_tensor(sigma, device=video.device, dtype=torch.float32).reshape(-1).expand(batch)
        after = torch.as_tensor(sigma_next, device=video.device, dtype=torch.float32).reshape(-1).expand(batch)
        time = torch.cat([_embed_time(value, self.config.time_embed_dim)
                          for value in (before, after, before - after)], -1)
        hidden = self.patch_in(torch.cat([video, velocity], 1)).permute(0, 2, 3, 4, 1)
        time = self.time_mlp(time.to(hidden.dtype))
        caption = self.text_projection(self.text_norm(caption))
        for block in self.blocks:
            hidden = block(hidden, time, caption)
        shift, scale = self.final_modulation(time).chunk(2, -1)
        hidden = self.final_norm(hidden) * (1 + scale[:, None, None, None]) + shift[:, None, None, None]
        patches = self.patch_out(hidden).reshape(batch, frames // pt, height // ph, width // pw,
                                                channels, pt, ph, pw)
        return patches.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape_as(video)


def bound_residual(residual, velocity, budget_fraction, velocity_floor):
    """Author's per-video soft RMS budget; strength is applied afterwards."""
    residual, velocity = residual.float(), velocity.float()
    dims = tuple(range(1, residual.ndim))
    base = velocity.square().mean(dims).sqrt()
    budget = base.clamp_min(velocity_floor) * budget_fraction
    factor = torch.rsqrt(1 + residual.square().mean(dims) / budget.square().clamp_min(1e-12))
    applied = residual * factor.reshape(-1, *([1] * (residual.ndim - 1)))
    return applied, applied.square().mean(dims).sqrt() / base.clamp_min(1e-12)
