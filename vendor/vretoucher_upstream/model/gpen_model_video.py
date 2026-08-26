'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import math
import random
import functools
import operator
import itertools
from turtle import up

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=0, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2, device='cpu'):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)
        self.device = device

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad, device=self.device)

        return out

class Downsample(nn.Module):
    def __init__(self, kernel, factor=2, device='cpu'):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)
        self.device = device

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad, device=self.device)

        return out

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1, device='cpu'):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad
        self.device = device

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad, device=self.device)

        return out

class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})')

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None, device='cpu'
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation
        self.device = device

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul, device=self.device)

        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')

class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)

class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        device='cpu', 
        factor = 2
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor, device=device)

        if downsample:
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), device=device)

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out

class ModulatedConv2d_new(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        device='cpu', 
        factor = 2
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor, device=device)

        if downsample:
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), device=device)

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )
        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input):
        batch, in_channel, height, width = input.shape
        weight = self.scale * self.weight 
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out

class NoiseInjection(nn.Module):
    def __init__(self, isconcat=True):
        super().__init__()

        self.isconcat = isconcat
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, channel, height, width = image.shape
            noise = image.new_empty(batch, channel, height, width).normal_()

        if self.isconcat:
            return torch.cat((image, self.weight * noise), dim=1)
        else:
            return image + self.weight * noise

class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        isconcat=True,
        device='cpu'
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            device=device
        )

        self.noise = NoiseInjection(isconcat)
        #self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        #self.activate = ScaledLeakyReLU(0.2)
        feat_multiplier = 2 if isconcat else 1
        self.activate = FusedLeakyReLU(out_channel*feat_multiplier, device=device)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out

class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], device='cpu'):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel, device=device)
        else:
            self.upsample = None

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False, device=device)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if self.upsample is not None:
            skip = self.upsample(skip)
        if skip is not None:
            out = out + skip
        return out

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=2,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        device='cpu'
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1), device=device))
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel, device=device))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)



class Encoder(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            channel_multiplier=2,
            narrow=1,
            device='cuda'
    ):
        super().__init__()
        self.enc = nn.Sequential(
            ConvLayer(3, 64, 1, device=device),
            ConvLayer(64, 128, 3, downsample=True, device=device),
            ConvLayer(128, 256, 3, downsample=True, device=device),
            ConvLayer(256, 512, 3, downsample=True, device=device),
            ConvLayer(512, 512, 3, downsample=False, device=device),
            ConvLayer(512, 512, 3, downsample=False, device=device)
        )


    def forward(self, x_list):
        noises = []
        for x in x_list:
            noises.append(self.enc(x))
        return noises



class Encoder_new(nn.Module):
    def __init__(
            self,
            size,
            channel_multiplier=2,
            narrow=1,
            device='cuda'
    ):
        super().__init__()
        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow)
        }
        self.log_size = int(math.log(size, 2))
        conv = [ConvLayer(3, channels[size], 1, device=device)]
        self.ecd0 = nn.Sequential(*conv)
        in_channel = channels[size]
        self.names = ['ecd%d' % i for i in range(self.log_size - 1)]
        for i in range(self.log_size, 4, -1):
            out_channel = channels[2 ** (i - 1)]
            if i > 6 :
                conv = [ConvLayer(in_channel, out_channel, 3, downsample=True, device=device)]
            else:
                conv = [ConvLayer(in_channel, out_channel, 3, downsample=False, device=device)]
            setattr(self, self.names[self.log_size - i + 1], nn.Sequential(*conv))
            in_channel = out_channel

    def forward(self, x_list):
        noises = []  # noises代表最后一帧的多尺度特征
        frame_list = []
        for frame_num in range(len(x_list)):
            frame = x_list[frame_num]
            for i in range(self.log_size - 3):
                ecd = getattr(self, self.names[i])
                frame = ecd(frame)
                if frame_num == len(x_list)-1:
                    noises.append(frame)
            frame_list.append(frame)
        return noises, frame_list


class Encoder_8(nn.Module):
    def __init__(
            self,
            size,
            channel_multiplier=2,
            narrow=1,
            device='cuda'
    ):
        super().__init__()
        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow)
        }
        self.log_size = int(math.log(size, 2))
        conv = [ConvLayer(3, channels[size], 1, device=device)]
        self.ecd0 = nn.Sequential(*conv)
        in_channel = channels[size]
        self.names = ['ecd%d' % i for i in range(self.log_size - 1)]
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            if i > 6 :
                conv = [ConvLayer(in_channel, out_channel, 3, downsample=True, device=device)]
            else:
                conv = [ConvLayer(in_channel, out_channel, 3, downsample=False, device=device)]
            setattr(self, self.names[self.log_size - i + 1], nn.Sequential(*conv))
            in_channel = out_channel

    def forward(self, x_list):
        noises = []  # noises代表最后一帧的多尺度特征
        frame_list = []
        for frame_num in range(len(x_list)):
            frame = x_list[frame_num]
            for i in range(self.log_size - 1):
                ecd = getattr(self, self.names[i])
                frame = ecd(frame)
                if frame_num == len(x_list)-1:
                    noises.append(frame)
            frame_list.append(frame)
        return noises, frame_list


class Decoder(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            isconcat=True,
            narrow=1,
            device='cuda'
    ):
        super().__init__()

        self.size = size
        self.style_dim = style_dim
        self.feat_multiplier = 2 if isconcat else 1

        self.channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow)
        }

        self.input = nn.Conv2d(self.channels[4], self.channels[4], kernel_size=1)
        self.conv0 = StyledConv(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel, isconcat=isconcat, device=device)
        self.to_rgb0 = ToRGB(self.channels[4] * self.feat_multiplier, style_dim, upsample=False, device=device)
        self.log_size = int(math.log(size, 2))
        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channel = self.channels[4]

        for i in range(5, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            if i < 7:
                self.convs.append(
                StyledConv(in_channel * self.feat_multiplier, out_channel, 3, style_dim, upsample=False, blur_kernel=blur_kernel,
                    isconcat=isconcat, device=device)) 
                self.to_rgbs.append(ToRGB(out_channel * self.feat_multiplier, style_dim, upsample=False, device=device))
            else:
                self.convs.append(
                StyledConv(in_channel * self.feat_multiplier, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel,
                    isconcat=isconcat, device=device))
                self.to_rgbs.append(ToRGB(out_channel * self.feat_multiplier, style_dim, upsample=True, device=device))
            self.convs.append(
                StyledConv(
                    out_channel * self.feat_multiplier, out_channel, 3, style_dim, blur_kernel=blur_kernel,
                    isconcat=isconcat, device=device))
            in_channel = out_channel


    def forward(self, noise, style=None):
        batch = noise[0].shape[0]
        out = self.input(noise[0])
        if style == None:
            style = torch.ones([batch, self.style_dim], device=noise[0].device).float() # override style with const
        
        out = self.conv0(out, style, noise=noise[0])
        skip = self.to_rgb0(out, style)

        for conv1, conv2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], self.to_rgbs):
            out = conv1(out, style, noise=None)
            out = conv2(out, style, noise=None)
            skip = to_rgb(out, style, skip)      
        return skip


class Decoder_8(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            isconcat=True,
            narrow=1,
            device='cuda'
    ):
        super().__init__()

        self.size = size
        self.style_dim = style_dim
        self.feat_multiplier = 2 if isconcat else 1

        self.channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow)
        }

        self.input = nn.Conv2d(self.channels[4], self.channels[4], kernel_size=1)
        self.conv0 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel, isconcat=isconcat, device=device
        )
        # self.channels_changer0 = nn.Conv2d(self.channels[4] * self.feat_multiplier, self.channels[4], kernel_size=1) # For conv1
        self.to_rgb0 = ToRGB(self.channels[4] * self.feat_multiplier, style_dim, upsample=False, device=device)

        self.log_size = int(math.log(size, 2))

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        # self.channel_changers = nn.ModuleList()

        in_channel = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            if i < 7:
                self.convs.append(
                    StyledConv(in_channel * self.feat_multiplier, out_channel, 3, style_dim, upsample=False,
                               blur_kernel=blur_kernel,
                               isconcat=isconcat, device=device))
                self.to_rgbs.append(ToRGB(out_channel * self.feat_multiplier, style_dim, upsample=False, device=device))
            else:
                self.convs.append(
                    StyledConv(in_channel * self.feat_multiplier, out_channel, 3, style_dim, upsample=True,
                               blur_kernel=blur_kernel,
                               isconcat=isconcat, device=device))
                self.to_rgbs.append(ToRGB(out_channel * self.feat_multiplier, style_dim, upsample=True, device=device))

            self.convs.append(
                StyledConv(
                    out_channel * self.feat_multiplier, out_channel, 3, style_dim, blur_kernel=blur_kernel,
                    isconcat=isconcat, device=device))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def forward(self, noise, style=None):
        batch = noise[0].shape[0]
        out = self.input(noise[0])
        if style == None:
            style = torch.ones([batch, self.style_dim], device=noise[0].device).float()  # override style with const

        out = self.conv0(out, style, noise=noise[0])
        skip = self.to_rgb0(out, style)

        for conv1, conv2, noise, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1:], self.to_rgbs):
            out = conv1(out, style, noise=noise)
            out = conv2(out, style, noise=noise)
            skip = to_rgb(out, style, skip)
        return skip


class Decoder_noise(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            channel_multiplier=2,
            blur_kernel=[1, 3, 3, 1],
            isconcat=True,
            narrow=1,
            device='cuda'
    ):
        super().__init__()

        self.size = size
        self.style_dim = style_dim
        self.feat_multiplier = 2 if isconcat else 1

        self.channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow)
        }

        self.input = nn.Conv2d(self.channels[4], self.channels[4], kernel_size=1)
        self.conv0 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel, isconcat=isconcat, device=device
        )
        # self.channels_changer0 = nn.Conv2d(self.channels[4] * self.feat_multiplier, self.channels[4], kernel_size=1) # For conv1
        self.to_rgb0 = ToRGB(self.channels[4] * self.feat_multiplier, style_dim, upsample=False, device=device)

        self.log_size = int(math.log(size, 2))

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        # self.channel_changers = nn.ModuleList()

        in_channel = self.channels[4]

        for i in range(5, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            if i < 7:
                self.convs.append(
                StyledConv(in_channel * self.feat_multiplier, out_channel, 3, style_dim, upsample=False, blur_kernel=blur_kernel,
                    isconcat=isconcat, device=device)) 
                self.to_rgbs.append(ToRGB(out_channel * self.feat_multiplier, style_dim, upsample=False, device=device))
            else:
                self.convs.append(
                StyledConv(in_channel * self.feat_multiplier, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel,
                    isconcat=isconcat, device=device))
                self.to_rgbs.append(ToRGB(out_channel * self.feat_multiplier, style_dim, upsample=True, device=device))

            self.convs.append(
                StyledConv(
                    out_channel * self.feat_multiplier, out_channel, 3, style_dim, blur_kernel=blur_kernel,
                    isconcat=isconcat, device=device))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def forward(self, noise, style=None):
        batch = noise[0].shape[0]
        out = self.input(noise[0])
        if style == None:
            style = torch.ones([batch, self.style_dim], device=noise[0].device).float() # override style with const
        
        out = self.conv0(out, style, noise=noise[0])
        skip = self.to_rgb0(out, style)

        for conv1, conv2, noise, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1:], self.to_rgbs):
            out = conv1(out, style, noise=noise)
            out = conv2(out, style, noise=noise)
            skip = to_rgb(out, style, skip)      
        return skip


