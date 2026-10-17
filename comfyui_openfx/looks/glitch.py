import math

import torch
import torch.nn.functional as F

from .. import ops as O
from ..registry import look


def rgb_split(x, dx, dy=0.0):
    r = O.shift(x[..., 0:1], -dx, -dy)
    b = O.shift(x[..., 2:3], dx, dy)
    return torch.cat([r, x[..., 1:2], b], -1)


def _blocks(x, bs):
    H, W = x.shape[1], x.shape[2]
    return max(1, round(H / bs)), max(1, round(W / bs))


def slice_shift(x, t, rng, fps, amount, salt, n_max=14, burst_rate=6.0, burst_p=0.55):
    """Horizontal slices shifted sideways, in short random bursts."""
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    s = O.S(x)
    idx = O.frame_index(t, fps)
    keys = [int(k / fps * burst_rate) for k in idx]
    b = O.key_rand(rng, keys, (3,), salt, "burst", device=x.device)
    on = (b[:, 0] < burst_p).to(x.dtype) * (0.4 + 0.6 * b[:, 1])
    ys = torch.arange(H, device=x.device, dtype=x.dtype) / H
    p = O.frame_rand(rng, t, fps, (n_max, 3), salt, device=x.device)
    dx = torch.zeros(Fn, H, device=x.device, dtype=x.dtype)
    for k in range(n_max):
        y0 = p[:, k, 0]
        hgt = 0.01 + 0.12 * p[:, k, 1] ** 2
        inb = ((ys[None] >= y0[:, None]) & (ys[None] < (y0 + hgt)[:, None])).to(x.dtype)
        dx = dx + inb * (p[:, k, 2:3] - 0.5) * 2 * s * 0.18
    dx = dx * on.view(-1, 1) * amount
    uu, vv, _, _ = O.grid(x)
    gu = uu[None] - dx[:, :, None] * 2 / W
    y = O.warp(x, gu, vv[None].expand_as(gu), "reflection")
    moved = (dx.abs() > 0.5).to(x.dtype)[..., None, None]
    # colour fringe on displaced slices
    y = y * (1 - moved) + rgb_split(y, s * 0.008, 0) * moved
    return y, on


def macroblock(x, t, rng, fps, amount, salt, rate=8.0):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    s = O.S(x)
    bs = max(4, s / 22)
    bh, bw = _blocks(x, bs)
    idx = O.frame_index(t, fps)
    keys = [int(k / fps * rate) for k in idx]
    # clustered corruption: low-res random field thresholded
    r = O.key_rand(rng, keys, (1, max(2, bh // 3), max(2, bw // 3)), salt, "cl", device=x.device)
    r = F.interpolate(r, size=(bh, bw), mode="bilinear", align_corners=False)
    r2 = O.key_rand(rng, keys, (1, bh, bw), salt, "fine", device=x.device)
    sel = ((r * 0.7 + r2 * 0.3) > (1 - 0.32 * amount)).to(x.dtype)
    mode = O.key_rand(rng, keys, (1, bh, bw), salt, "mode", device=x.device)
    dc = F.interpolate(F.adaptive_avg_pool2d(O.nchw(x), (bh, bw)), size=(H, W), mode="nearest")
    # smeared copy: blocks repeat content from a block above (stale motion vectors)
    smear = O.nchw(O.shift(x, 0, bs * 2.0))
    small = F.adaptive_avg_pool2d(O.nchw(x), (bh * 2, bw * 2))
    coarse = F.interpolate(small, size=(H, W), mode="nearest")
    big = lambda m: F.interpolate(m, size=(H, W), mode="nearest")
    selb, modeb = big(sel), big(mode)
    xn = O.nchw(x)
    tint = torch.tensor([0.12, -0.05, 0.10], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    alt = torch.where(modeb < 0.35, dc + tint * (modeb - 0.17) * 2,
                      torch.where(modeb < 0.7, smear, coarse))
    y = xn * (1 - selb) + alt * selb
    # subtle 8x8 'jpeg' quantisation everywhere
    q = F.interpolate(F.adaptive_avg_pool2d(xn, (max(1, H // 8), max(1, W // 8))), size=(H, W), mode="nearest")
    y = y * 0.9 + (y * 0.6 + q * 0.4) * 0.1
    return O.nhwc(y)


@look("rgb-split", "RGB Split", "glitch", "Red and blue channels pulled apart for a jittery chromatic offset.",
      tags=["chromatic", "glitch", "cyberpunk"], synonyms=["chromatic aberration", "rgb shift", "channel shift", "color split"], animated=True)
def rgb_split_look(x, t, i, rng, fps):
    s = O.S(x)
    j = O.frame_rand(rng, t, fps, (2,), "rgbj", device=x.device) - 0.5
    dx = s * (0.012 + j[:, 0] * 0.006)
    dy = s * j[:, 1] * 0.004
    y = rgb_split(x, dx, dy)
    return O.finish(x, y, i)


@look("pixel-sort", "Pixel Sort", "glitch", "Bright runs of pixels sorted by brightness into smeared horizontal streaks.",
      tags=["glitch-art", "databending", "streaks"], synonyms=["pixel sorting", "sorted pixels", "glitch art"])
def pixel_sort(x, t, i, rng, fps):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    l = O.luma(O.blur(x, O.S(x) * 0.0015))[..., 0]
    q = torch.quantile(l.flatten(1)[:, :: max(1, l[0].numel() // 20000)], 0.45, dim=1).view(-1, 1, 1)
    mask = (l > q.clamp(0.2, 0.6)) & (l < 0.985)
    pos = torch.arange(W, device=x.device, dtype=torch.float64).view(1, 1, W).expand(Fn, H, W)
    start = mask & ~torch.cat([torch.zeros_like(mask[..., :1]), mask[..., :-1]], -1)
    seg_start = torch.where(start, pos, torch.full_like(pos, -1.0))
    seg_start = torch.cummax(seg_start, -1).values
    key = torch.where(mask, seg_start + l.double().clamp(0, 1) * 0.999, pos)
    order = torch.argsort(key, dim=-1, stable=True)
    y = torch.gather(x, 2, order[..., None].expand(-1, -1, -1, 3))
    return O.finish(x, y, i)


@look("macroblock", "Macroblock", "glitch", "Broken digital stream: blocky compression artefacts, stale blocks and colour-shifted squares.",
      tags=["compression", "datamosh", "digital"], synonyms=["datamosh", "compression", "blocky", "jpeg", "bad stream"], animated=True)
def macroblock_look(x, t, i, rng, fps):
    return O.finish(x, macroblock(x, t, rng, fps, 1.0, "mb"), i)


@look("slice-shift", "Slice Shift", "glitch", "Bursts of horizontal slices tearing sideways.",
      tags=["glitch", "displacement", "digital"], synonyms=["slice glitch", "displacement glitch", "glitch slices"], animated=True)
def slice_shift_look(x, t, i, rng, fps):
    y, _ = slice_shift(x, t, rng, fps, 1.0, "ss", burst_p=0.7)
    return O.finish(x, y, i)


@look("frame-repeat", "Frame Repeat", "glitch", "Stuttering playback: frames freeze, repeat and jump back.",
      tags=["stutter", "lag", "temporal"], synonyms=["stutter", "lag", "buffering", "frame stutter"], animated=True, temporal=True)
def frame_repeat(x, t, i, rng, fps):
    Fn = x.shape[0]
    if Fn < 2 or i <= 0:
        return x
    g = O.gen(rng, "cpu", "frep")
    idx = []
    k = 0
    while len(idx) < Fn:
        r = float(torch.rand(1, generator=g))
        if r < 0.55 * i:
            hold = 2 + int(torch.randint(0, 5, (1,), generator=g))
            if float(torch.rand(1, generator=g)) < 0.3 and k > 4:
                k = max(0, k - int(torch.randint(2, 6, (1,), generator=g)))
            idx.extend([min(k, Fn - 1)] * hold)
            k += hold
        else:
            idx.append(min(k, Fn - 1))
            k += 1
    idx = torch.tensor(idx[:Fn], device=x.device)
    return x[idx]


@look("digital-tear", "Digital Tear", "glitch", "The frame tears along a few lines, with sections slipping sideways and fringing.",
      tags=["glitch", "tearing", "vsync"], synonyms=["screen tear", "tearing", "vsync tear"], animated=True)
def digital_tear(x, t, i, rng, fps):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    s = O.S(x)
    p = O.frame_rand(rng, t, fps, (3, 3), "tear", device=x.device)
    ys = torch.arange(H, device=x.device, dtype=x.dtype) / H
    dx = torch.zeros(Fn, H, device=x.device, dtype=x.dtype)
    dy = torch.zeros(Fn, H, device=x.device, dtype=x.dtype)
    line = torch.zeros(Fn, H, device=x.device, dtype=x.dtype)
    for k in range(3):
        y0 = p[:, k, 0:1]
        below = (ys[None] > y0).to(x.dtype)
        dx = dx + below * (p[:, k, 1:2] - 0.5) * s * 0.16
        dy = dy + below * (p[:, k, 2:3] - 0.5) * s * 0.03
        line = torch.maximum(line, O.smoothstep(max(1.0, s / 400) / H, 0.0, (ys[None] - y0).abs()) * (p[:, k, 1:2] > 0.5).to(x.dtype))
    uu, vv, _, _ = O.grid(x)
    gu = uu[None] - dx[:, :, None] * 2 / W
    gv = vv[None] - dy[:, :, None] * 2 / H
    y = O.warp(x, gu, gv, "reflection")
    y = rgb_split(y, s * 0.004)
    y = y + line[..., None, None] * 0.3
    return O.finish(x, y, i)


_BAYER4 = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]


@look("bit-crush", "Bit Crush", "glitch", "Reduced colour depth with ordered dithering, like an 8-bit computer.",
      tags=["8-bit", "dither", "retro-computer"], synonyms=["8 bit", "dither", "low color", "retro game", "8-bit"])
def bit_crush(x, t, i, rng, fps):
    H, W = x.shape[1], x.shape[2]
    s = O.S(x)
    px = max(1, int(round(s / 240)))
    hs, ws = max(1, H // px), max(1, W // px)
    small = O.resize(x, hs, ws, "area")
    b = (torch.tensor(_BAYER4, device=x.device, dtype=x.dtype) + 0.5) / 16 - 0.5
    b = b.repeat((hs + 3) // 4, (ws + 3) // 4)[:hs, :ws]
    lv = 4
    q = torch.floor(O.saturate(small, 1.15).clamp(0, 1) * (lv - 1) + 0.5 + b[None, ..., None]) / (lv - 1)
    y = O.resize(q.clamp(0, 1), H, W, "nearest")
    return O.finish(x, y, i)


@look("pixelate", "Pixelate", "glitch", "Big chunky pixels.",
      tags=["mosaic", "8-bit", "censor"], synonyms=["pixels", "pixelated", "mosaic", "censor", "8-bit"])
def pixelate(x, t, i, rng, fps):
    H, W = x.shape[1], x.shape[2]
    bs = max(1.0, O.S(x) / 40)
    bh, bw = _blocks(x, bs)
    y = F.interpolate(F.adaptive_avg_pool2d(O.nchw(x), (bh, bw)), size=(H, W), mode="nearest")
    return O.finish(x, O.nhwc(y), i)


@look("glitch-bursts", "Glitch Bursts", "glitch", "Mostly clean picture hit by short random bursts of RGB split, slice tearing and broken blocks.",
      tags=["glitch", "cyberpunk", "digital", "music-video"], synonyms=["glitch", "glitchy", "corrupted", "cyberpunk", "error"], animated=True)
def glitch_bursts(x, t, i, rng, fps):
    s = O.S(x)
    idx = O.frame_index(t, fps)
    keys = [int(k / fps * 5) for k in idx]
    b = O.key_rand(rng, keys, (4,), "gb", device=x.device)
    on = (b[:, 0] < 0.6).to(x.dtype)
    y = rgb_split(x, s * (0.005 + 0.02 * on * b[:, 1]), 0)
    sl, _ = slice_shift(y, t, rng, fps, 1.0, "gbs", burst_rate=5, burst_p=1.0)
    y = O.mix(y, sl, (on * (b[:, 2] < 0.75).to(x.dtype)).view(-1, 1, 1, 1))
    mb = macroblock(y, t, rng, fps, 1.2, "gbm", rate=10)
    y = O.mix(y, mb, (on * (b[:, 3] < 0.5).to(x.dtype)).view(-1, 1, 1, 1))
    # thin scanline flash during bursts
    H = x.shape[1]
    ln = (torch.arange(H, device=x.device) % max(2, int(s / 180)) == 0).to(x.dtype).view(1, H, 1, 1)
    y = y * (1 - 0.15 * ln * on.view(-1, 1, 1, 1))
    return O.finish(x, y, i)
