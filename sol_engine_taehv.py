"""Minimal TAEHV wide runtime used by NVIDIA's LTX-2.5 Stage-2 refiner.

Adapted from madebyollin/taehv commit
32ac0146b11007cda5a57b60a3b35653361fb8a4 (MIT).  Only the batch
encode/decode path required by H3 Super Acceleration is included.

Copyright (c) 2025 Ollin Boer Bohan. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))


def _conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            _conv(n_in * 2, n_out),
            nn.ReLU(inplace=True),
            _conv(n_out, n_out),
            nn.ReLU(inplace=True),
            _conv(n_out, n_out),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class WideMemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        groups = max(1, n_out // 64)
        if n_out % groups:
            raise ValueError(f"invalid WideMemBlock groups: {n_out=} {groups=}")
        self.conv = nn.Sequential(
            nn.Conv2d(n_in * 2, n_out, 1),
            nn.ReLU(inplace=True),
            _conv(n_out, n_out, groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_out, n_out, 1),
            nn.ReLU(inplace=True),
            _conv(n_out, n_out, groups=groups),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        _nt, channels, height, width = x.shape
        return self.conv(x.reshape(-1, self.stride * channels, height, width))


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _nt, channels, height, width = x.shape
        x = self.conv(x)
        return x.reshape(-1, channels, height, width)


def _apply_parallel(model, x):
    if x.ndim != 5:
        raise ValueError(f"TAEHV expects NTCHW, got {tuple(x.shape)}")
    batch, frames, channels, height, width = x.shape
    x = x.reshape(batch * frames, channels, height, width)
    for block in model:
        if isinstance(block, (MemBlock, WideMemBlock)):
            nt, channels, height, width = x.shape
            frames = nt // batch
            temporal = x.reshape(batch, frames, channels, height, width)
            memory = F.pad(temporal, (0, 0, 0, 0, 0, 0, 1, 0), value=0)
            x = block(x, memory[:, :frames].reshape(x.shape))
        else:
            x = block(x)
    nt, channels, height, width = x.shape
    return x.view(batch, nt // batch, channels, height, width)


def _apply_sequential_single_step(model, memory, work_queue):
    while work_queue:
        xt, index = work_queue.pop(0)
        if index == len(model):
            return xt.unsqueeze(1)
        block = model[index]
        if isinstance(block, (MemBlock, WideMemBlock)):
            previous = xt * 0 if memory[index] is None else memory[index]
            next_xt = block(xt, previous)
            memory[index] = xt
            work_queue.insert(0, TWorkItem(next_xt, index + 1))
        elif isinstance(block, TPool):
            if memory[index] is None:
                memory[index] = []
            memory[index].append(xt)
            if len(memory[index]) > block.stride:
                raise ValueError("TAEHV TPool memory overflow")
            if len(memory[index]) == block.stride:
                batch, channels, height, width = xt.shape
                pooled = block(
                    torch.cat(memory[index], 1).view(
                        batch * block.stride,
                        channels,
                        height,
                        width,
                    )
                )
                memory[index] = []
                work_queue.insert(0, TWorkItem(pooled, index + 1))
        elif isinstance(block, TGrow):
            grown = block(xt)
            nt, channels, height, width = grown.shape
            chunks = grown.view(nt // block.stride, block.stride * channels, height, width).chunk(
                block.stride,
                1,
            )
            for next_xt in reversed(chunks):
                work_queue.insert(0, TWorkItem(next_xt, index + 1))
        else:
            work_queue.insert(0, TWorkItem(block(xt), index + 1))
    return None


def _apply_sequential(model, x):
    if x.ndim != 5:
        raise ValueError(f"TAEHV expects NTCHW, got {tuple(x.shape)}")
    work_queue = [TWorkItem(xt, 0) for xt in x.unbind(1)]
    memory = [None] * len(model)
    outputs = []
    while work_queue:
        xt = _apply_sequential_single_step(model, memory, work_queue)
        if xt is not None:
            outputs.append(xt)
    return torch.cat(outputs, 1)


def _apply(model, x, parallel):
    return _apply_parallel(model, x) if parallel else _apply_sequential(model, x)


class TAEHVLTX23Wide(nn.Module):
    """The explicit LTX-2/2.3 wide architecture; filename-independent."""

    patch_size = 4
    latent_channels = 128
    image_channels = 3

    def __init__(self, checkpoint_path=None):
        super().__init__()
        self.encoder = nn.Sequential(
            _conv(self.image_channels * self.patch_size**2, 64),
            nn.ReLU(inplace=True),
            TPool(64, 2),
            _conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            TPool(64, 2),
            _conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            TPool(64, 2),
            _conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            _conv(64, self.latent_channels),
        )
        n_f = [1024, 512, 256, 64]
        self.decoder = nn.Sequential(
            Clamp(),
            _conv(self.latent_channels, n_f[0]),
            nn.ReLU(inplace=True),
            WideMemBlock(n_f[0], n_f[0]),
            WideMemBlock(n_f[0], n_f[0]),
            WideMemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2),
            TGrow(n_f[0], 2),
            _conv(n_f[0], n_f[1], bias=False),
            WideMemBlock(n_f[1], n_f[1]),
            WideMemBlock(n_f[1], n_f[1]),
            WideMemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=2),
            TGrow(n_f[1], 2),
            _conv(n_f[1], n_f[2], bias=False),
            WideMemBlock(n_f[2], n_f[2]),
            WideMemBlock(n_f[2], n_f[2]),
            WideMemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=2),
            TGrow(n_f[2], 2),
            _conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True),
            _conv(n_f[3], self.image_channels * self.patch_size**2),
        )
        self.t_downscale = 8
        self.t_upscale = 8
        self.frames_to_trim = 7
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.load_state_dict(self._patch_tgrow_layers(state))

    def _patch_tgrow_layers(self, state):
        current = self.state_dict()
        for index, layer in enumerate(self.decoder):
            if not isinstance(layer, TGrow):
                continue
            key = f"decoder.{index}.conv.weight"
            if state[key].shape[0] > current[key].shape[0]:
                state[key] = state[key][-current[key].shape[0] :]
        return state

    def _preprocess(self, x):
        return F.pixel_unshuffle(x, self.patch_size)

    def _postprocess(self, x):
        return F.pixel_shuffle(x, self.patch_size).clamp_(0, 1)

    def encode_video(self, x, *, parallel):
        x = self._preprocess(x)
        remainder = x.shape[1] % self.t_downscale
        if remainder:
            pad = self.t_downscale - remainder
            x = torch.cat([x, x[:, -1:].repeat_interleave(pad, dim=1)], 1)
        return _apply(self.encoder, x, parallel)

    def decode_video(self, x, *, parallel):
        decoded = self._postprocess(_apply(self.decoder, x, parallel))
        return decoded[:, self.frames_to_trim :]
