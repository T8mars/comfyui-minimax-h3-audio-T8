"""LBH 2D model components, fixed upstream 87d8822.

Copyright (c) 2026 LBH-123-AI; MIT. See THIRD_PARTY_NOTICES/LBH-2D-LICENSE.txt.
Only model definitions and checkpoint shape discovery are extracted. No UI,
folder registration, device selection or unsafe pickle loader is imported.
Source: https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
"""
import gc
import math
import re
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

USE_LOCAL_ENSEMBLE = False
CHUNK_FRAMES = 32
_UP_PATH_LOGGED = set()

def _log_up_path(desc, s):
    key = round(float(s), 4)
    if key in _UP_PATH_LOGGED:
        return
    _UP_PATH_LOGGED.add(key)
    print(f"[MinimaxH3-2D] 上采样路径: {desc} | 目标倍率 {float(s):.4f}")

def local_ensemble_upsample(feat, target_size):
    """
    Local Ensemble 上采样：消除任意倍率下的锯齿 (LIIF 核心)
    feat: [N, C, H, W]
    target_size: (H_new, W_new)
    """
    N, C, H, W = feat.shape
    H_new, W_new = target_size

    grid_y = torch.linspace(-1, 1, H_new, device=feat.device, dtype=feat.dtype)
    grid_x = torch.linspace(-1, 1, W_new, device=feat.device, dtype=feat.dtype)
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(N, -1, -1, -1)

    cell_W = 1.0 / W_new if W_new > 1 else 0.0
    cell_H = 1.0 / H_new if H_new > 1 else 0.0

    preds = [F.grid_sample(feat, grid, mode='bilinear', padding_mode='border', align_corners=False)]

    for dy in [-cell_H, cell_H]:
        for dx in [-cell_W, cell_W]:
            grid_shift = grid.clone()
            grid_shift[..., 0] += dx
            grid_shift[..., 1] += dy
            preds.append(F.grid_sample(feat, grid_shift, mode='bilinear', padding_mode='border', align_corners=False))

    return torch.stack(preds, dim=0).mean(dim=0)

def normalization(channels):
    return nn.GroupNorm(32 if channels >= 32 else channels, channels)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class FourierScaleEmbedding2D(nn.Module):
    """scale_h / scale_w 的 Fourier 编码，可选拼接【锐度档位】的 Fourier 编码。

    【2026-11-05 新增 tier_dim】与训练脚本 Q-latent-2d-v15 的 EnhancedScaleEmbedding 逐位对齐：
      输入 = [ sin/cos(scale_h) *16 频 → 32 维 ]
             [ sin/cos(scale_w) *16 频 → 32 维 ]
             [ sin/cos((tier-1)) *16 频 → 32 维 ]   ← tier_dim=32 时才存在
    tier_dim=0 时输入仍是 4*freq（64 维），**旧权重行为完全不变**；
    tier_dim=32 时是 96 维（e25 及之后带档位的权重）。
    """

    def __init__(self, embed_dim, num_frequencies=16, tier_dim=0):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.tier_dim = int(tier_dim)
        in_dim = 4 * num_frequencies + self.tier_dim
        self.linear = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # 仅对"新建模型"有意义：档位通道零初始化 → 等价于无档位。加载权重后这些列会被覆盖。
        if self.tier_dim > 0:
            with torch.no_grad():
                self.linear[0].weight[:, 4 * num_frequencies:].zero_()

    def forward(self, scale_h, scale_w, tier=None):
        # 【对齐 v12】与训练脚本 EnhancedScaleEmbedding 完全一致：
        #   1) 支持 float/int 输入（训练里 scale_h 是 Python float）
        #   2) 相位不乘 π —— 乘 π 后 scale*freq*π 在「倍率相差偶数整数」时恰为 π 的整数倍，
        #      sin 恒 0、cos 恒 ±1，高频分量整体塌缩（1.2x/3.2x、2.0x/4.0x 无法区分）
        #   3) 三角函数在 fp32 下计算，算完再回到权重 dtype
        dev = self.linear[0].weight.device
        if isinstance(scale_h, (float, int)): scale_h = torch.tensor([float(scale_h)], device=dev)
        if isinstance(scale_w, (float, int)): scale_w = torch.tensor([float(scale_w)], device=dev)
        scale_h = scale_h.to(device=dev, dtype=torch.float32)
        scale_w = scale_w.to(device=dev, dtype=torch.float32)
        if scale_h.dim() == 0: scale_h = scale_h.unsqueeze(0)
        if scale_w.dim() == 0: scale_w = scale_w.unsqueeze(0)

        freqs = 2.0 ** torch.arange(self.num_frequencies, device=dev).float()
        emb_h = torch.cat([torch.sin(scale_h.unsqueeze(1) * freqs.unsqueeze(0)),
                           torch.cos(scale_h.unsqueeze(1) * freqs.unsqueeze(0))], dim=1)
        emb_w = torch.cat([torch.sin(scale_w.unsqueeze(1) * freqs.unsqueeze(0)),
                           torch.cos(scale_w.unsqueeze(1) * freqs.unsqueeze(0))], dim=1)

        emb = torch.cat([emb_h, emb_w], dim=1)

        if self.tier_dim > 0:
            # 与训练脚本一致：相位用 (tier-1)，档位 1 是规范零点。
            # tier=None 表示"调用方没指定" → 按 t1 处理（相位 0），
            # 不能按 0 处理（那会变成 tier=0，是训练里根本不存在的档位，属于分布外输入）。
            if tier is None:
                t = torch.ones(1, device=dev)
            elif isinstance(tier, (float, int)):
                t = torch.tensor([float(tier)], device=dev)
            else:
                t = tier.to(device=dev, dtype=torch.float32)
                if t.dim() == 0: t = t.unsqueeze(0)
            t = t - 1.0
            emb = torch.cat([emb,
                             torch.sin(t.unsqueeze(1) * freqs.unsqueeze(0)),
                             torch.cos(t.unsqueeze(1) * freqs.unsqueeze(0))], dim=1)

        # 拼接后转回输入 dtype（如 fp16），以匹配 self.linear 的权重 dtype
        return self.linear(emb.to(self.linear[0].weight.dtype))

class CoordConv2D(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        yc = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype).view(1, 1, H, 1).expand(B, 1, H, W)
        xc = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, yc, xc], dim=1)

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        mid = max(ch // reduction, 16)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, mid),
            nn.SiLU(True),
            nn.Linear(mid, ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)

class LargeKernelBlock(nn.Module):
    def __init__(self, ch, ks=7):
        super().__init__()
        self.norm = normalization(ch)
        self.dw = nn.Conv2d(ch, ch, ks, padding=ks // 2, groups=ch)
        self.pw = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

    def forward(self, x):
        return x + self.pw(self.dw(F.silu(self.norm(x))))

class PostUpsampleRefine(nn.Module):
    def __init__(self, ch, n=2):
        super().__init__()
        layers = []
        for i in range(n):
            layers.append(nn.Conv2d(ch, ch, 3, padding=1))
            if i < n - 1:
                layers.append(nn.SiLU(True))
        self.refine = nn.Sequential(*layers)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, x):
        return x + self.refine(x)

class MultiScaleTemporalConv(nn.Module):
    def __init__(self, ch, kernel_sizes=(3, 5, 9)):
        super().__init__()
        self.norm = normalization(ch)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, ch, (ks, 1, 1), padding=(ks // 2, 0, 0), groups=ch),
                nn.Conv3d(ch, ch, 1)
            ) for ks in kernel_sizes
        ])
        self.fusion = nn.Conv3d(ch * len(kernel_sizes), ch, 1)
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, x):
        B, C, T, H, W = x.shape
        h = rearrange(x, "b c t h w -> (b t) c h w")
        h = self.norm(h)
        h = rearrange(h, "(b t) c h w -> b c t h w", b=B, t=T)
        return x + self.fusion(torch.cat([br(F.silu(h)) for br in self.branches], dim=1))

class TemporalConv(nn.Module):
    # 【对齐 v12】参数名必须是 dwconv / pwconv，否则 strict=False 下加载不到权重
    # （dwconv 是随机初始化，静默 mismatch 会直接毁掉输出）
    def __init__(self, ch, ks=5):
        super().__init__()
        self.norm = normalization(ch)
        self.dwconv = nn.Conv3d(ch, ch, (ks, 1, 1), padding=(ks // 2, 0, 0), groups=ch)
        self.pwconv = nn.Conv3d(ch, ch, 1)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def forward(self, x):
        B, C, T, H, W = x.shape
        h = rearrange(x, "b c t h w -> (b t) c h w")
        h = self.norm(h)
        h = rearrange(h, "(b t) c h w -> b c t h w", b=B, t=T)
        return x + self.pwconv(self.dwconv(F.silu(h)))

class ResBlockEmb(nn.Module):
    def __init__(self, channels, emb_channels, dropout=0, out_channels=None, use_se=True, se_reduction=8):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Conv2d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )
        self.se = SEBlock(self.out_channels, se_reduction) if use_se else None

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        if self.se is not None:
            h = self.se(h)
        return self.skip(x) + h

def _bilinear_taps(k):
    """k 个长度为 3 的 1D 插值 tap（对应输入索引 i-1, i, i+1），align_corners=False 口径。"""
    taps = []
    for a in range(k):
        t = (a + 0.5) / k - 0.5
        lo = math.floor(t)
        frac = t - lo
        v = torch.zeros(3)
        v[lo + 1] = 1.0 - frac
        v[lo + 2] = frac
        taps.append(v)
    return taps

class PSUpsample(nn.Module):
    """单倍率 PixelShuffle(k) 学习型上采样（独占式，不与插值叠加）。"""

    def __init__(self, channels, k=2):
        super().__init__()
        self.k = int(k)
        assert self.k >= 2, f"PS 倍数必须 >= 2，收到 {k}"
        # padding_mode 必须 'replicate'：与训练侧一致，且使初始化输出与
        # F.interpolate(bilinear) 的边界 clamp 行为对齐（'zeros' 会让边缘系统性偏低）
        self.conv = nn.Conv2d(channels, self.k * self.k * channels, 3, padding=1,
                              padding_mode='replicate')
        self.ps = nn.PixelShuffle(self.k)
        self._init_bilinear_kernel()

    def _init_bilinear_kernel(self):
        """双线性核初始化（与训练侧完全一致）。正常会被权重覆盖，仅作未加载时的合理兜底。"""
        k, C = self.k, self.conv.in_channels
        taps = _bilinear_taps(k)
        with torch.no_grad():
            w = torch.zeros_like(self.conv.weight)
            for a in range(k):
                for b in range(k):
                    ker = torch.outer(taps[a], taps[b])
                    idx = a * k + b
                    for c in range(C):
                        w[c * k * k + idx, c] = ker
            self.conv.weight.copy_(w)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.ps(self.conv(x))

class PSUpsampleBank(nn.Module):
    """一组 PS 支路，按目标倍率精确选 k。"""

    def __init__(self, channels, ks):
        super().__init__()
        self.ks = sorted({int(k) for k in ks})
        assert self.ks and all(k >= 2 for k in self.ks), f"learned_upsample_ks 非法: {ks}"
        self.branches = nn.ModuleDict({str(k): PSUpsample(channels, k) for k in self.ks})

    def select_k(self, scale):
        for k in self.ks:                       # 精确匹配优先
            if abs(k - scale) <= 1e-3:
                return k
        for k in self.ks:                       # 兜底：最小的 k > scale
            if k > scale:
                return k
        return self.ks[-1]

    def forward(self, x, scale):
        return self.branches[str(self.select_k(scale))](x)

class LatentResizer(nn.Module):
    """2D Ultimate Backbone with Local Ensemble"""
    def __init__(self, in_channels=24, in_blocks=8, out_blocks=8, channels=768,
                 dropout=0.1, attn=False, use_dual_scale=True, use_coord=True,
                 use_se=True, se_reduction=8, use_large_kernel=True, large_kernel_size=7,
                 large_kernel_every=3, use_post_refine=True, post_refine_layers=2,
                 dual_embed_freq=16, embed_dim=128, learned_upsample_ks=None, tier_dim=0):
        super().__init__()
        # 【对齐 v12】EMBED_DIM=128，必须由权重推断后传入，不能写死
        self.use_dual_scale = use_dual_scale
        self.use_coord = use_coord
        # 【2026-11-05】锐度档位通道宽度：0 = 旧权重无档位，32 = 带档位权重（e25 及之后）
        self.tier_dim = int(tier_dim)

        if use_dual_scale:
            self.embed = FourierScaleEmbedding2D(embed_dim, dual_embed_freq, tier_dim=self.tier_dim)
        else:
            self.embed = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        conv_in_ch = in_channels + 2 if use_coord else in_channels
        self.coord_conv = CoordConv2D() if use_coord else None
        self.conv_in = nn.Conv2d(conv_in_ch, channels, 3, padding=1)

        # 【2026-09-21】学习型上采样支路（与训练脚本 v16 同款）。
        # 仅当权重里存在 learned_up.* 时才构建（由 _detect_arch 推断），否则为 None → 纯插值，
        # 保证旧权重仍能被 load_state_dict 严格校验通过。
        self.learned_up = PSUpsampleBank(channels, learned_upsample_ks) if learned_upsample_ks else None

        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            self.in_blocks.append(ResBlockEmb(channels, embed_dim, dropout, use_se=use_se, se_reduction=se_reduction))
            if use_large_kernel and b % large_kernel_every == (large_kernel_every - 1):
                self.in_blocks.append(LargeKernelBlock(channels, large_kernel_size))

        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            self.out_blocks.append(ResBlockEmb(channels, embed_dim, dropout, use_se=use_se, se_reduction=se_reduction))
            if use_large_kernel and b % large_kernel_every == (large_kernel_every - 1):
                self.out_blocks.append(LargeKernelBlock(channels, large_kernel_size))

        self.post_up = PostUpsampleRefine(channels, post_refine_layers) if use_post_refine else None
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, in_channels, 3, padding=1)

    def _upsample_mid(self, x, size):
        """中段上采样：学习型 PixelShuffle 与插值 bilinear【互斥二选一】。

        - learned_up 存在且目标倍率精确等于某个 k → 完全用学习型输出（不叠加任何插值）
        - 否则（无支路 / 倍率非整数，如 1.5x）→ 回退 F.interpolate(bilinear)
        注意：训练脚本只在「倍率精确等于 k」的桶上训练学习型支路，推理端非整数倍率
        必须回退插值，否则会因尺寸不符而裁掉内容。
        """
        if self.learned_up is not None:
            s = min(size[0] / x.shape[-2], size[1] / x.shape[-1])
            k = self.learned_up.select_k(s)
            if abs(k - s) <= 1e-3 and x.shape[-2] * k >= size[0] and x.shape[-1] * k >= size[1]:
                y = self.learned_up(x, s)
                if y.shape[-2] != size[0] or y.shape[-1] != size[1]:
                    y = y[..., :size[0], :size[1]]      # 仅裁剪对齐，不引入插值
                _log_up_path(f"学习型 PixelShuffle(k={k})", s)
                return y
            _log_up_path(f"回退插值：目标倍率不匹配整数 k={self.learned_up.ks}（学习型支路未生效）", s)
            return local_ensemble_upsample(x, size) if USE_LOCAL_ENSEMBLE else \
                F.interpolate(x, size=size, mode="bilinear")
        s = min(size[0] / x.shape[-2], size[1] / x.shape[-1])
        _log_up_path("回退插值：该权重不含学习型支路（旧权重）", s)
        return local_ensemble_upsample(x, size) if USE_LOCAL_ENSEMBLE else \
            F.interpolate(x, size=size, mode="bilinear")

    def forward(self, x, scale_h=None, scale_w=None, target_hw=None, tier=None):
        if target_hw is not None:
            size = target_hw
        elif scale_h is not None and scale_w is not None:
            size = (int(round(x.shape[-2] * scale_h)), int(round(x.shape[-1] * scale_w)))
        else:
            return x

        if size == x.shape[-2:]:
            return x

        if scale_h is None: scale_h = size[0] / x.shape[-2]
        if scale_w is None: scale_w = size[1] / x.shape[-1]

        # 【对齐 v12】嵌入输入是原始倍率（不减 1）
        if self.use_dual_scale:
            emb = self.embed(scale_h, scale_w, tier=tier)
        else:
            emb = self.embed((scale_h + scale_w) / 2.0, (scale_h + scale_w) / 2.0)

        if self.use_coord:
            x = self.coord_conv(x)
        x = self.conv_in(x)

        for b in self.in_blocks:
            if isinstance(b, ResBlockEmb):
                x = b(x, emb.expand(x.shape[0], -1))
            else:
                x = b(x)

        # 【2026-09-21】中段上采样：学习型 / 插值【互斥二选一】（见 _upsample_mid）
        x = self._upsample_mid(x, size)
        if self.post_up:
            x = self.post_up(x)

        for b in self.out_blocks:
            if isinstance(b, ResBlockEmb):
                x = b(x, emb.expand(x.shape[0], -1))
            else:
                x = b(x)

        return self.conv_out(F.silu(self.norm_out(x)))

class VideoLatentResizer(nn.Module):
    """5D Ultimate Wrapper with Temporal Blocks + Global Skip Local Ensemble"""
    def __init__(self, in_channels=24, in_blocks=8, out_blocks=8, channels=768,
                 dropout=0.1, attn=False, temporal_every=2, use_multiscale_temporal=True,
                 temporal_kernel_sizes=(3, 5, 9), temporal_kernel=5, **resizer_kwargs):
        super().__init__()
        self.resizer = LatentResizer(
            in_channels=in_channels, in_blocks=in_blocks, out_blocks=out_blocks,
            channels=channels, dropout=dropout, attn=attn, **resizer_kwargs
        )
        self.temporal_every = temporal_every
        self.temporal_blocks = nn.ModuleList()

        if temporal_every > 0:
            if use_multiscale_temporal:
                self.temporal_blocks.append(MultiScaleTemporalConv(channels, temporal_kernel_sizes))
                self.temporal_blocks.append(MultiScaleTemporalConv(channels, temporal_kernel_sizes))
            else:
                self.temporal_blocks.append(TemporalConv(channels, temporal_kernel))
                self.temporal_blocks.append(TemporalConv(channels, temporal_kernel))

    def forward(self, x, scale=None, target_hw=None, enable_chunking=True,
                chunk_frames=None, tier=None):
        B, C, T, H, W = x.shape

        if target_hw is not None:
            size = target_hw
        elif scale is not None:
            size = (int(round(H * scale)), int(round(W * scale)))
        else:
            size = (H, W)

        # Temporal Chunking for long videos
        # 【上下文宽度 = 时间感受野，不是单个卷积核】
        # temporal conv 在 in_blocks / out_blocks 里各插入 (blocks // temporal_every) 次，
        # 每次影响 ±(k//2) 帧 → v12（k=5、每 2 层插一次、12+12 层）总感受野 = 2*(6+6) = 24 帧。
        # 按 kernel 长度（5）给上下文会少 19 帧，块边界末段帧的邻居被当成 0 填充 → 接缝闪烁。
        tk = 0
        for b in self.temporal_blocks:
            if isinstance(b, TemporalConv):
                tk = b.dwconv.weight.shape[2]
                break
            elif isinstance(b, MultiScaleTemporalConv):
                tk = max(br[0].weight.shape[2] for br in b.branches)
                break

        radius = max(tk // 2, 1)
        n_temporal = sum(1 for i in range(len(self.resizer.in_blocks)) if i % self.temporal_every == 0)
        n_temporal += sum(1 for i in range(len(self.resizer.out_blocks)) if i % self.temporal_every == 0)
        overlap = radius * max(n_temporal, 1)
        chunk = max(1, int(chunk_frames)) if chunk_frames else CHUNK_FRAMES

        if not enable_chunking or T <= chunk or len(self.temporal_blocks) == 0:
            return self._forward_full(x, size, tier=tier)

        # Chunked inference：每块只负责输出 [seg_start, seg_end)，输入额外带 ±overlap 的**真实**帧。
        # 不补显式 0 帧：显式 0 帧经 GroupNorm/bias 后 ≠「不存在的帧」（conv 的隐式填充才是真 0 贡献），
        # 会在视频首尾引入偏差。分段边缘交给 Conv3d 自带 padding，行为与训练时完全一致。
        # 上下文给足 overlap 后，分块结果与整段一次前向只差浮点噪声，纯粹是省显存。
        out_full = torch.zeros(B, C, T, size[0], size[1], device=x.device, dtype=x.dtype)

        start = 0
        while start < T:
            seg_start = start
            seg_end = min(T, start + chunk)
            lo = max(0, seg_start - overlap)          # 左侧多给 overlap 帧真实上下文
            hi = min(T, seg_end + overlap)            # 右侧同理

            seg = x[:, :, lo:hi].contiguous()
            seg_out = self._forward_full(seg, size, tier=tier)

            s0 = seg_start - lo                       # seg 内 s0 帧 = 视频第 seg_start 帧
            out_full[:, :, seg_start:seg_end] = seg_out[:, :, s0:s0 + (seg_end - seg_start)]

            start += chunk
            del seg, seg_out
            if start % (chunk * 4) == 0:
                gc.collect()

        return out_full

    def _forward_full(self, x, size, tier=None):
        """与 v12 训练脚本 VideoLatentResizer.forward 逐行对齐（无全局残差支路）。"""
        B, C, T, H, W = x.shape
        x_flat = rearrange(x, "b c t h w -> (b t) c h w")

        scale_h = size[0] / H
        scale_w = size[1] / W

        if len(self.temporal_blocks) == 0:
            out = self.resizer(x_flat, scale_h=scale_h, scale_w=scale_w, target_hw=size, tier=tier)
            return rearrange(out, "(b t) c h w -> b c t h w", b=B, t=T)

        if self.resizer.use_dual_scale:
            emb = self.resizer.embed(scale_h, scale_w, tier=tier)
        else:
            emb = self.resizer.embed((scale_h + scale_w) / 2.0, (scale_h + scale_w) / 2.0)

        feat = self.resizer.coord_conv(x_flat) if self.resizer.use_coord else x_flat
        out = self.resizer.conv_in(feat)

        for i, block in enumerate(self.resizer.in_blocks):
            if isinstance(block, ResBlockEmb):
                out = block(out, emb.expand(B * T, -1))
            else:
                out = block(out)
            if i % self.temporal_every == 0 and len(self.temporal_blocks) > 0:
                out_3d = rearrange(out, "(b t) c h w -> b c t h w", b=B, t=T)
                out_3d = self.temporal_blocks[0](out_3d)
                out = rearrange(out_3d, "b c t h w -> (b t) c h w")

        # 【2026-09-21】中段上采样：学习型 / 插值【互斥二选一】，训练与推理必须同款
        out = self.resizer._upsample_mid(out, size)
        if self.resizer.post_up:
            out = self.resizer.post_up(out)

        for i, block in enumerate(self.resizer.out_blocks):
            if isinstance(block, ResBlockEmb):
                out = block(out, emb.expand(B * T, -1))
            else:
                out = block(out)
            if i % self.temporal_every == 0 and len(self.temporal_blocks) > 1:
                out_3d = rearrange(out, "(b t) c h w -> b c t h w", b=B, t=T)
                out_3d = self.temporal_blocks[1](out_3d)
                out = rearrange(out_3d, "b c t h w -> (b t) c h w")

        out = self.resizer.norm_out(out)
        out = F.silu(out)
        out = self.resizer.conv_out(out)
        return rearrange(out, "(b t) c h w -> b c t h w", b=B, t=T)

def _detect_arch(sd):
    """Auto-detect Ultimate architecture parameters from state_dict"""
    cfg = {
        "in_channels": 24, "in_blocks": 8, "out_blocks": 8, "channels": 768,
        "dropout": 0.1, "attn": False, "temporal_every": 2,
        "use_dual_scale": True, "use_coord": True, "use_se": True, "se_reduction": 8,
        "use_large_kernel": True, "large_kernel_size": 7, "large_kernel_every": 3,
        "use_post_refine": True, "post_refine_layers": 2,
        "embed_dim": 128, "dual_embed_freq": 16,
        "tier_dim": 0,   # 【2026-11-05】锐度档位通道宽度：0 = 旧权重无档位；32 = 带档位（e25+）
        "use_multiscale_temporal": True, "temporal_kernel_sizes": (3, 5, 9), "temporal_kernel": 5,
        "learned_upsample_ks": [],   # 【2026-09-21】学习型上采样 PixelShuffle 的 k 列表；空 = 纯插值
    }

    # Detect channels from conv_in
    conv_key = 'resizer.conv_in.weight'
    if conv_key in sd:
        w = sd[conv_key]
        cfg["channels"] = w.shape[0]
        conv_in_ch = w.shape[1]
        
        # 【关键修复】CoordConv2D 无参数，不在 state_dict 中
        # 通过 conv_in 输入通道数判断：26 = 24(latent) + 2(coord)
        # 同时检查 conv_out 输出通道数确认真实 latent 通道数
        conv_out_key = 'resizer.conv_out.weight'
        if conv_out_key in sd:
            real_in_channels = sd[conv_out_key].shape[0]  # conv_out 输出 = 真实 latent 通道数
        else:
            real_in_channels = 24  # fallback
        
        if conv_in_ch == real_in_channels + 2:
            cfg["use_coord"] = True
            cfg["in_channels"] = real_in_channels
        else:
            cfg["use_coord"] = False
            cfg["in_channels"] = conv_in_ch

    # Count blocks
    in_ids, out_ids = set(), set()
    for k in sd.keys():
        m = re.match(r'resizer\.in_blocks\.(\d+)\.', k)
        if m: in_ids.add(int(m.group(1)))
        m = re.match(r'resizer\.out_blocks\.(\d+)\.', k)
        if m: out_ids.add(int(m.group(1)))
    if in_ids: cfg["in_blocks"] = max(in_ids) + 1
    if out_ids: cfg["out_blocks"] = max(out_ids) + 1

    # Detect SE Block
    cfg["use_se"] = any('.se.fc.' in k for k in sd)

    # Detect Large Kernel
    cfg["use_large_kernel"] = any('.dw.weight' in k and 'resizer.in_blocks' in k
                                   and 'temporal' not in k for k in sd)
    if cfg["use_large_kernel"]:
        for k in sd.keys():
            if '.dw.weight' in k and 'resizer.in_blocks' in k and 'temporal' not in k:
                cfg["large_kernel_size"] = sd[k].shape[2]
                break

    # Detect Post Refine
    cfg["use_post_refine"] = any('resizer.post_up.refine.' in k for k in sd)

    # 【2026-09-21】Detect 学习型上采样（PixelShuffle）支路：从键名 resizer.learned_up.branches.<k>. 推断。
    # 权重中没有该前缀 → 空列表 → 模型不构建该支路 → 走纯插值（旧权重完全兼容）。
    _lu_ks = set()
    for k in sd.keys():
        m = re.match(r'resizer\.learned_up\.branches\.(\d+)\.', k)
        if m:
            _lu_ks.add(int(m.group(1)))
    cfg["learned_upsample_ks"] = sorted(_lu_ks)

    # Detect Dual Scale Embedding（【v12】embed_dim / 频率数都由权重形状决定，不能写死）
    cfg["use_dual_scale"] = any('resizer.embed.linear.' in k for k in sd)
    emb_k = sd.get('resizer.embed.linear.0.weight')
    if emb_k is not None:
        cfg["embed_dim"] = int(emb_k.shape[0])          # Linear(in_dim, embed_dim)
        _in_dim = int(emb_k.shape[1])
        # 【2026-11-05】输入宽度判定：
        #   64 = 4*16            → 只有 scale_h/scale_w，无档位（v12 ~ e24）
        #   96 = 4*16 + 32       → 额外 32 维锐度档位 Fourier 编码（e25 及之后的 tier 权重）
        # 训练脚本 num_frequencies 固定为 16，所以只认这两种；其它宽度按"无档位"兜底。
        if _in_dim == 96:
            cfg["dual_embed_freq"], cfg["tier_dim"] = 16, 32
        else:
            cfg["dual_embed_freq"], cfg["tier_dim"] = _in_dim // 4, 0

    # Detect MultiScale Temporal
    cfg["use_multiscale_temporal"] = any('temporal_blocks.0.branches.' in k for k in sd)
    if cfg["use_multiscale_temporal"]:
        ks = set()
        for k in sd.keys():
            m = re.match(r'temporal_blocks\.0\.branches\.(\d+)\.0\.weight', k)
            if m:
                ks.add(sd[k].shape[2])
        if ks:
            cfg["temporal_kernel_sizes"] = tuple(sorted(ks))
    else:
        # 【v12】TemporalConv 的参数名是 dwconv/pwconv（旧版为 dw/pw），两种都要识别
        for k in sd.keys():
            if 'temporal_blocks.0.dwconv.weight' in k or 'temporal_blocks.0.dw.weight' in k:
                cfg["temporal_kernel"] = sd[k].shape[2]
                break

    cfg["attn"] = False  # Force off for inference
    return cfg
