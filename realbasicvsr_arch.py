"""Dependency-light RealBasicVSR inference architecture.

Adapted from OpenMMLab MMagic's Apache-2.0 implementations of RealBasicVSR,
BasicVSR and SPyNet. Training/registry/MMEngine integration is intentionally
omitted so loading this optional node does not alter ComfyUI's dependencies.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _make_layer(block, count: int, **kwargs) -> nn.Sequential:
    return nn.Sequential(*(block(**kwargs) for _ in range(count)))


class _ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels: int = 64, res_scale: float = 1.0):
        super().__init__()
        self.res_scale = float(res_scale)
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value
        value = self.conv2(self.relu(self.conv1(value)))
        return identity + value * self.res_scale


def _flow_warp(
    value: torch.Tensor,
    flow: torch.Tensor,
    *,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Warp NCHW ``value`` with NHW2 flow expressed in source pixels."""

    if value.shape[-2:] != flow.shape[1:3]:
        raise ValueError(
            f"flow spatial shape {tuple(flow.shape[1:3])} does not match "
            f"feature shape {tuple(value.shape[-2:])}"
        )
    height, width = value.shape[-2:]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=value.device, dtype=value.dtype),
        torch.arange(width, device=value.device, dtype=value.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    grid = grid + flow.to(dtype=value.dtype)
    scale_x = max(width - 1, 1)
    scale_y = max(height - 1, 1)
    grid_x = 2.0 * grid[..., 0] / float(scale_x) - 1.0
    grid_y = 2.0 * grid[..., 1] / float(scale_y) - 1.0
    normalized = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(
        value,
        normalized,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )


class _ConvModule(nn.Module):
    """Small shape-compatible subset of MMCV ConvModule used by SPyNet."""

    def __init__(self, in_channels: int, out_channels: int, *, activate: bool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 7, 1, 3, bias=True)
        self.activate = bool(activate)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv(value)
        return F.relu(value, inplace=True) if self.activate else value


class SPyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            _ConvModule(8, 32, activate=True),
            _ConvModule(32, 64, activate=True),
            _ConvModule(64, 32, activate=True),
            _ConvModule(32, 16, activate=True),
            _ConvModule(16, 2, activate=False),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.basic_module(value)


class SPyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.basic_module = nn.ModuleList(SPyNetBasicModule() for _ in range(6))
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def compute_flow(self, reference: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = reference.shape
        references = [(reference - self.mean) / self.std]
        supports = [(support - self.mean) / self.std]
        for _ in range(5):
            references.append(
                F.avg_pool2d(references[-1], 2, 2, count_include_pad=False)
            )
            supports.append(F.avg_pool2d(supports[-1], 2, 2, count_include_pad=False))
        references.reverse()
        supports.reverse()
        flow = reference.new_zeros(batch, 2, height // 32, width // 32)
        for level, (ref_level, support_level) in enumerate(zip(references, supports)):
            if level:
                flow_up = F.interpolate(
                    flow, scale_factor=2, mode="bilinear", align_corners=True
                ) * 2.0
            else:
                flow_up = flow
            warped = _flow_warp(
                support_level, flow_up.permute(0, 2, 3, 1), padding_mode="border"
            )
            flow = flow_up + self.basic_module[level](
                torch.cat((ref_level, warped, flow_up), dim=1)
            )
        return flow

    def forward(self, reference: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        height, width = reference.shape[-2:]
        height_up = ((height + 31) // 32) * 32
        width_up = ((width + 31) // 32) * 32
        reference_up = F.interpolate(
            reference, size=(height_up, width_up), mode="bilinear", align_corners=False
        )
        support_up = F.interpolate(
            support, size=(height_up, width_up), mode="bilinear", align_corners=False
        )
        flow = F.interpolate(
            self.compute_flow(reference_up, support_up),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        flow[:, 0] *= float(width) / float(width_up)
        flow[:, 1] *= float(height) / float(height_up)
        return flow


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64, num_blocks: int = 20):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            _make_layer(_ResidualBlockNoBN, num_blocks, mid_channels=out_channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.main(value)


class PixelShufflePack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.upsample_conv = nn.Conv2d(
            in_channels,
            out_channels * scale_factor * scale_factor,
            3,
            1,
            1,
            bias=True,
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.pixel_shuffle(self.upsample_conv(value))


class BasicVSRNet(nn.Module):
    def __init__(self, mid_channels: int = 64, num_blocks: int = 20):
        super().__init__()
        self.mid_channels = int(mid_channels)
        self.spynet = SPyNet()
        self.backward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks
        )
        self.forward_resblocks = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks
        )
        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0, bias=True)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(
            scale_factor=4, mode="bilinear", align_corners=False
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def compute_flow(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count, channels, height, width = frames.shape
        first = frames[:, :-1].reshape(-1, channels, height, width)
        second = frames[:, 1:].reshape(-1, channels, height, width)
        backward = self.spynet(first, second).view(batch, count - 1, 2, height, width)
        forward = self.spynet(second, first).view(batch, count - 1, 2, height, width)
        return forward, backward

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, count, _, height, width = frames.shape
        if count < 2:
            raise ValueError("RealBasicVSR requires at least two frames per inference window")
        flows_forward, flows_backward = self.compute_flow(frames)
        outputs: list[torch.Tensor] = []
        feature = frames.new_zeros(batch, self.mid_channels, height, width)
        for index in range(count - 1, -1, -1):
            if index < count - 1:
                feature = _flow_warp(
                    feature, flows_backward[:, index].permute(0, 2, 3, 1)
                )
            feature = self.backward_resblocks(
                torch.cat((frames[:, index], feature), dim=1)
            )
            outputs.append(feature)
        outputs.reverse()

        feature = torch.zeros_like(feature)
        for index in range(count):
            current = frames[:, index]
            if index:
                feature = _flow_warp(
                    feature, flows_forward[:, index - 1].permute(0, 2, 3, 1)
                )
            feature = self.forward_resblocks(torch.cat((current, feature), dim=1))
            output = self.lrelu(self.fusion(torch.cat((outputs[index], feature), dim=1)))
            output = self.lrelu(self.upsample1(output))
            output = self.lrelu(self.upsample2(output))
            output = self.lrelu(self.conv_hr(output))
            output = self.conv_last(output) + self.img_upsample(current)
            outputs[index] = output
        return torch.stack(outputs, dim=1)


class RealBasicVSRNet(nn.Module):
    """Official RealBasicVSR x4 generator architecture, inference-only."""

    def __init__(
        self,
        mid_channels: int = 64,
        num_propagation_blocks: int = 20,
        num_cleaning_blocks: int = 20,
        dynamic_refine_threshold: float = 255.0,
        sequential_cleaning: bool = True,
    ):
        super().__init__()
        self.dynamic_refine_threshold = float(dynamic_refine_threshold) / 255.0
        self.sequential_cleaning = bool(sequential_cleaning)
        self.image_cleaning = nn.Sequential(
            ResidualBlocksWithInputConv(3, mid_channels, num_cleaning_blocks),
            nn.Conv2d(mid_channels, 3, 3, 1, 1, bias=True),
        )
        self.basicvsr = BasicVSRNet(mid_channels, num_propagation_blocks)
        self.basicvsr.spynet.requires_grad_(False)

    def forward(self, frames: torch.Tensor, *, return_cleaned: bool = False):
        batch, count, channels, height, width = frames.shape
        frames = frames.clone()
        for _ in range(3):
            if self.sequential_cleaning:
                residues = []
                for index in range(count):
                    residue = self.image_cleaning(frames[:, index])
                    frames[:, index] = frames[:, index] + residue
                    residues.append(residue)
                residue_batch = torch.stack(residues, dim=1)
            else:
                flat = frames.reshape(-1, channels, height, width)
                residue_batch = self.image_cleaning(flat)
                frames = (flat + residue_batch).view(batch, count, channels, height, width)
            if torch.mean(torch.abs(residue_batch)) < self.dynamic_refine_threshold:
                break
        output = self.basicvsr(frames)
        return (output, frames) if return_cleaned else output
