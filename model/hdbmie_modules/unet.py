# =========================
# FIXED STABLE DDPM / SR3 UNET
# =========================

import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction


# -------------------------
# helpers
# -------------------------

def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else (d() if isfunction(d) else d)


# -------------------------
# positional encoding
# -------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        device = t.device

        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)

        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        return emb


# -------------------------
# FiLM conditioning
# -------------------------

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.SiLU()
        )

    def forward(self, x, t):
        h = self.net(t).view(x.shape[0], -1, 1, 1)
        return x + h


# -------------------------
# basic blocks
# -------------------------

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(dim, dim, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Conv2d(dim, dim, 3, stride=2, padding=1)

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32):
        super().__init__()

        groups = min(groups, dim)
        while dim % groups != 0:
            groups -= 1

        self.net = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# ResBlock
# -------------------------

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, tdim=None):
        super().__init__()

        self.time_mlp = FeatureWiseAffine(tdim, dim_out) if tdim else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)

        self.res = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, t=None):
        h = self.block1(x)

        if self.time_mlp and t is not None:
            h = self.time_mlp(h, t)

        h = self.block2(h)

        return h + self.res(x)


# -------------------------
# Attention (FIXED)
# -------------------------

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.GroupNorm(32, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)
        q, k, v = self.qkv(x).chunk(3, dim=1)

        q = q.reshape(b, c, h * w)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)

        attn = torch.einsum("bci,bcj->bij", q, k)
        attn = attn / math.sqrt(c // 1)   # stable scalar scaling
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum("bij,bcj->bci", attn, v)
        out = out.reshape(b, c, h, w)

        return self.proj(out) + x


# -------------------------
# ResBlock wrapper
# -------------------------

class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, tdim=None, with_attn=False):
        super().__init__()

        self.block = ResnetBlock(dim, dim_out, tdim)
        self.attn = SelfAttention(dim_out) if with_attn else None

    def forward(self, x, t):
        x = self.block(x, t)
        if self.attn:
            x = self.attn(x)
        return x


# -------------------------
# UNet (FIXED)
# -------------------------

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        base=32,
        channel_mults=(1, 2, 4, 8),
        res_blocks=2,
        image_size=128,
        with_time=True
    ):
        super().__init__()

        self.image_size = image_size

        self.time_mlp = None
        if with_time:
            self.time_mlp = nn.Sequential(
                PositionalEncoding(base),
                nn.Linear(base, base * 4),
                Swish(),
                nn.Linear(base * 4, base)
            )

        # ---------------- encoder ----------------
        self.downs = nn.ModuleList()
        self.skips = []

        ch = base
        self.downs.append(nn.Conv2d(in_channel, ch, 3, padding=1))

        for mult in channel_mults:
            out_ch = base * mult

            for _ in range(res_blocks):
                self.downs.append(ResnetBlocWithAttn(ch, out_ch, base))
                ch = out_ch

            self.downs.append(Downsample(ch))

        # ---------------- middle ----------------
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(ch, ch, base, True),
            ResnetBlocWithAttn(ch, ch, base, False)
        ])

        # ---------------- decoder ----------------
        self.ups = nn.ModuleList()

        for mult in reversed(channel_mults):
            out_ch = base * mult

            for _ in range(res_blocks):
                self.ups.append(
                    ResnetBlocWithAttn(ch + out_ch, out_ch, base)
                )
                ch = out_ch

            self.ups.append(Upsample(ch))

        self.final = Block(ch, out_channel)

    # ---------------- forward ----------------

    def forward(self, x, t):
        t = self.time_mlp(t) if self.time_mlp else None

        skips = []

        # encoder
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
                skips.append(x)
            else:
                x = layer(x)

        # middle
        for layer in self.mid:
            x = layer(x, t)

        # decoder
        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = layer(x, t)
            else:
                x = layer(x)

        return self.final(x)
