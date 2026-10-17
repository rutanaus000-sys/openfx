import math

import torch
import torch.nn.functional as F

from .. import font
from .. import ops as O
from ..registry import look


def _row_noise(rng, t, fps, x, size_x, *salt):
    """Horizontally streaky per-frame noise [F,H,W,1] (analog video noise)."""
    H, W = x.shape[1], x.shape[2]
    gw = max(2, int(W / size_x))
    n = O.frame_rand(rng, t, fps, (1, H, gw), *salt, device=x.device, normal=True)
    n = F.interpolate(n, size=(H, W), mode="bilinear", align_corners=False)
    return n.permute(0, 2, 3, 1)


def _row_shift(x, dx):
    """Shift each row horizontally by dx [F,H] pixels."""
    H, W = x.shape[1], x.shape[2]
    uu, vv, _, _ = O.grid(x)
    gu = uu[None] - dx[:, :, None] * 2.0 / W
    return O.warp(x, gu, vv[None].expand_as(gu), "border")


def chroma_bleed(x, amount=1.0, sat=1.0):
    s = O.S(x)
    y = O.rgb_to_yiq(x)
    lum = y[..., :1]
    c = y[..., 1:]
    c = O.blur(c, s * 0.002 * amount, s * 0.012 * amount)
    c = O.shift(c, s * 0.008 * amount, 0)
    out = O.yiq_to_rgb(torch.cat([lum, c * sat], -1))
    return out


def scanlines(x, pitch, depth):
    H = x.shape[1]
    ys = torch.arange(H, device=x.device, dtype=x.dtype)
    ph = (ys / pitch) * 2 * math.pi
    l = O.luma(x)
    # bright lines bloom wider (beam spread)
    prof = 0.5 + 0.5 * torch.cos(ph)
    prof = prof.view(1, H, 1, 1)
    width = 1 - 0.6 * l
    m = 1 - depth * (1 - prof.pow(0.5 + width))
    return x * m * (1 + depth * 0.45)


@look("vhs", "VHS", "retro", "Worn VHS tape: soft picture, bleeding chroma, tape noise, wobbling lines and tracking bands.",
      tags=["90s", "tape", "analog", "nostalgic"], synonyms=["vhs tape", "videotape", "90s", "80s", "home video"], animated=True)
def vhs(x, t, i, rng, fps):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    s = O.S(x)
    dev = x.device
    # line wobble (time-base error)
    ys = torch.arange(H, device=dev, dtype=x.dtype) / H
    wob = O.frame_rand(rng, t, fps, (max(2, H // 6),), "vwob", device=dev, normal=True)
    wob = F.interpolate(wob[:, None], size=H, mode="linear", align_corners=False)[:, 0] * s * 0.0015
    # head-switching noise at the bottom
    hs = O.smoothstep(0.965, 1.0, ys)[None] * (O.frame_rand(rng, t, fps, (1,), "vhs-hs", device=dev) * 0.5 + 0.5) * s * 0.03
    # tracking band slowly drifting
    by = torch.remainder(t * 0.11 + float(O.urand(rng, dev, (1,), "vband")[0]), 1.3) - 0.15
    band = torch.exp(-((ys[None] - by.view(-1, 1)) / 0.018) ** 2)
    bandj = O.frame_rand(rng, t, fps, (H,), "vbj", device=dev, normal=True) * band * s * 0.01
    y = _row_shift(x, wob + hs + bandj)
    # soft luma, smeared chroma
    y = O.blur(y, s * 0.0008, s * 0.0028)
    y = chroma_bleed(y, 1.0, 0.92)
    # slight sharpening ringing (VHS 'edge enhancement')
    y = y + (y - O.blur(y, 0.1, s * 0.004)) * 0.6
    # colour shift / lifted blacks
    y = O.scurve(y.clamp(0, 1), 3.5) * O.col((1.03, 0.97, 1.04), x) + O.col((0.035, 0.02, 0.04), x)
    n = _row_noise(rng, t, fps, x, max(2.0, s / 180), "vnoise")
    y = y + n * 0.035
    # tracking band snow
    sn = (O.frame_rand(rng, t, fps, (H, max(2, W // 3)), "vsnow", device=dev) > 0.92).to(x.dtype)
    sn = F.interpolate(sn[:, None], size=(H, W), mode="nearest")[:, 0]
    y = y + (sn * band[:, :, None])[..., None] * 0.5
    y = y + O.smoothstep(0.97, 1.0, ys).view(1, H, 1, 1) * _row_noise(rng, t, fps, x, 3, "vhsn").abs() * 0.4
    return O.finish(x, y, i)


@look("tracking-errors", "Tracking Errors", "retro", "Mis-tracked tape: rolling bands of torn, noisy, sideways-shifted lines.",
      tags=["tape", "glitch", "analog"], synonyms=["bad tracking", "tracking", "vhs glitch"], animated=True)
def tracking_errors(x, t, i, rng, fps):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    s = O.S(x)
    dev = x.device
    ys = torch.arange(H, device=dev, dtype=x.dtype) / H
    off = float(O.urand(rng, dev, (1,), "te")[0])
    dx = torch.zeros(Fn, H, device=dev, dtype=x.dtype)
    bandsum = torch.zeros(Fn, H, device=dev, dtype=x.dtype)
    for k, (speed, wdt) in enumerate([(0.23, 0.05), (-0.13, 0.03), (0.41, 0.02)]):
        by = torch.remainder(t * speed + off + k * 0.37, 1.2) - 0.1
        band = O.smoothstep(wdt, 0.0, (ys[None] - by.view(-1, 1)).abs())
        jit = O.frame_rand(rng, t, fps, (max(2, H // 3),), "tej", k, device=dev, normal=True)
        jit = F.interpolate(jit[:, None], size=H, mode="nearest")[:, 0]
        dx = dx + band * (jit * 0.4 + 1.0) * s * 0.05 * (1 if k % 2 == 0 else -1)
        bandsum = bandsum + band
    y = _row_shift(x, dx)
    bandsum = bandsum.clamp(0, 1)[..., None, None]
    n = _row_noise(rng, t, fps, x, max(2.0, s / 120), "ten")
    y = y + n * 0.25 * bandsum
    y = O.mix(y, O.saturate(y, 0.3) * 1.15, bandsum)
    sn = (O.frame_rand(rng, t, fps, (H, max(2, W // 2)), "tes", device=dev) > 0.9).to(x.dtype)
    sn = F.interpolate(sn[:, None], size=(H, W), mode="nearest")[:, 0][..., None]
    y = y + sn * bandsum * 0.6
    y = chroma_bleed(y.clamp(0, 1), 0.6)
    return O.finish(x, y, i)


@look("tape-dropout", "Tape Dropout", "retro", "Flecks and streaks of white where the magnetic tape lost signal.",
      tags=["tape", "glitch", "damage"], synonyms=["dropout", "tape damage", "signal loss"], animated=True)
def tape_dropout(x, t, i, rng, fps):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    s = O.S(x)
    dev = x.device
    n = 26
    p = O.frame_rand(rng, t, fps, (n, 4), "dropout", device=dev)
    ys = torch.arange(H, device=dev, dtype=x.dtype)
    xs = torch.arange(W, device=dev, dtype=x.dtype)
    mask = torch.zeros(Fn, H, W, device=dev, dtype=x.dtype)
    th = max(1.0, s / 360)
    for k in range(n):
        cy = p[:, k, 0] * H
        x0 = p[:, k, 1] * W
        ln = (0.02 + 0.3 * p[:, k, 2] ** 2) * W
        on = (p[:, k, 3] < 0.6).to(x.dtype)
        my = O.smoothstep(th * (1 + (k % 3)), 0.0, (ys[None] - cy[:, None]).abs())
        mx = O.smoothstep(0, s * 0.01, xs[None] - x0[:, None]) * O.smoothstep(0, s * 0.03, x0[:, None] + ln[:, None] - xs[None])
        mask = torch.maximum(mask, my[:, :, None] * mx[:, None, :] * on.view(-1, 1, 1))
    nz = O.frame_rand(rng, t, fps, (H, max(2, W // 2)), "dropn", device=dev)
    nz = F.interpolate(nz[:, None], size=(H, W), mode="nearest")[:, 0]
    m = mask[..., None]
    y = x * (1 - m) + m * (0.75 + 0.25 * nz[..., None])
    # trailing dark smear after each dropout
    smear = O.shift(m, s * 0.02, 0)
    y = y * (1 - 0.3 * (smear - m).clamp(0, 1))
    return O.finish(x, y, i)


@look("chroma-bleed", "Chroma Bleed", "retro", "Colour smears and lags to the right of edges, like composite video.",
      tags=["analog", "color", "composite"], synonyms=["color bleed", "composite video", "chroma smear"])
def chroma_bleed_look(x, t, i, rng, fps):
    y = chroma_bleed(x, 1.6, 1.25)
    return O.finish(x, y, i)


@look("crt-scanlines", "CRT Scanlines", "retro", "Visible scanlines of a tube TV, with brighter lines blooming wider.",
      tags=["crt", "tv", "arcade"], synonyms=["scanlines", "crt", "old tv", "arcade"])
def crt_scanlines(x, t, i, rng, fps):
    pitch = max(2.0, O.S(x) / 200)
    y = scanlines(O.blur(x, pitch * 0.25), pitch, 0.55)
    y = O.screen(y.clamp(0, 1), O.blur(x, pitch * 2) * 0.15)
    return O.finish(x, y, i)


def _barrel(x, k):
    uu, vv, ax, ay = O.grid(x)
    r2 = (uu * uu) * 0.6 + (vv * vv) * 0.6
    f = 1 + k * r2
    gu, gv = uu * f, vv * f
    return gu, gv


@look("crt-curvature", "CRT Curvature", "retro", "A curved tube-TV screen: bulging glass, rounded corners, scanlines and glow.",
      tags=["crt", "tv", "retro", "arcade"], synonyms=["crt tv", "tube tv", "old television", "curved screen"])
def crt_curvature(x, t, i, rng, fps):
    s = O.S(x)
    H, W = x.shape[1], x.shape[2]
    k = 0.14 * i
    gu, gv = _barrel(x, k)
    sc = 1 + 0.0 * i
    gu, gv = gu * sc, gv * sc
    y = O.warp(x, gu, gv, "zeros")
    pitch = max(2.0, s / 200)
    y = scanlines(y, pitch, 0.5 * i)
    y = O.screen(y.clamp(0, 1), O.blur(y, s * 0.012) * 0.25 * i)
    # rounded screen mask, in warped space
    edge = 0.03
    mu = O.smoothstep(1.0, 1.0 - edge, gu.abs())
    mv = O.smoothstep(1.0, 1.0 - edge * W / H, gv.abs())
    corner = O.smoothstep(1.0, 0.95, (gu.abs() ** 10 + gv.abs() ** 10) ** (1 / 10))
    m = (mu * mv * corner)[None, ..., None]
    y = y * m * O.vignette_mask(x, 0.35, 0.45, 1.4)[None, ..., None] + (1 - m) * 0.015
    return O.finish(x, y, i)


@look("phosphor-mask", "Phosphor Mask", "retro", "Close-up of a CRT: the picture breaks into glowing red, green and blue phosphor stripes.",
      tags=["crt", "rgb", "texture"], synonyms=["aperture grille", "shadow mask", "rgb pixels", "crt mask"])
def phosphor_mask(x, t, i, rng, fps):
    s = O.S(x)
    H, W = x.shape[1], x.shape[2]
    p = max(1.0, s / 300)  # width of one phosphor stripe
    xs = torch.arange(W, device=x.device, dtype=x.dtype)
    ys = torch.arange(H, device=x.device, dtype=x.dtype)
    ch = torch.floor(xs / p)
    mask = torch.stack([(torch.remainder(ch - c, 3) == 0).to(x.dtype) for c in range(3)], -1)
    mask = O.mix(torch.full_like(mask, 0.32), torch.ones_like(mask), mask)
    # stagger + horizontal gaps between cells
    cell = torch.floor(xs / (3 * p))
    rowp = 3 * p
    yy = torch.remainder(ys[:, None] + (torch.remainder(cell, 2) * rowp / 2)[None], rowp)
    gap = O.smoothstep(0, max(0.8, p * 0.5), yy) * O.smoothstep(rowp, rowp - max(0.8, p * 0.5), yy)
    m = mask[None] * (0.35 + 0.65 * gap[None, ..., None])
    src = O.blur(x, p * 0.5)
    y = src * m * 1.75
    y = O.screen(y.clamp(0, 1), O.blur(src, s * 0.01) * 0.25)
    return O.finish(x, y, i)


@look("interlace", "Interlace", "retro", "Interlaced video: alternate lines from different moments, with combing on motion.",
      tags=["video", "tv", "combing"], synonyms=["interlaced", "combing", "fields", "deinterlace"], animated=True)
def interlace(x, t, i, rng, fps):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    s = O.S(x)
    lw = max(1, int(round(s / 360)))
    ys = torch.arange(H, device=x.device)
    odd = (torch.div(ys, lw, rounding_mode="floor") % 2 == 1).to(x.dtype).view(1, H, 1, 1)
    jit = (O.frame_rand(rng, t, fps, (), "ilj", device=x.device) - 0.5) * s * 0.016 + s * 0.01
    other = O.shift(x, jit, 0)
    y = x * (1 - odd) + other * odd
    y = y * (1 - 0.12 * odd)
    return O.finish(x, y, i)


@look("analog-snow", "Analog Snow", "retro", "Weak broadcast signal buried in hissing TV static.",
      tags=["static", "noise", "tv"], synonyms=["static", "tv static", "snow", "noise", "no signal"], animated=True)
def analog_snow(x, t, i, rng, fps):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    s = O.S(x)
    sz = max(1.0, s / 300)
    gh, gw = max(2, int(H / sz)), max(2, int(W / (sz * 1.6)))
    n = O.frame_rand(rng, t, fps, (1, gh, gw), "snow", device=x.device)
    n = F.interpolate(n, size=(H, W), mode="nearest")[:, 0][..., None]
    n = n.pow(1.6)
    lines = O.frame_rand(rng, t, fps, (H,), "snowl", device=x.device).view(Fn, H, 1, 1)
    sig = O.scurve(O.saturate(x, 0.6), 3.0)
    sig = O.shift(sig, s * 0.02, 0) * 0.18 + sig * 0.82  # ghosting
    y = sig * 0.6 + n * 0.55 + (lines - 0.5) * 0.12 - 0.1
    return O.finish(x, y, i)


@look("vertical-roll", "Vertical Roll", "retro", "The picture loses vertical hold and rolls, with the black blanking bar sliding through.",
      tags=["tv", "glitch", "analog"], synonyms=["rolling", "vertical hold", "roll", "v-hold"], animated=True)
def vertical_roll(x, t, i, rng, fps):
    H = x.shape[1]
    s = O.S(x)
    uu, vv, _, _ = O.grid(x)
    speed = 0.45
    ph0 = float(O.urand(rng, x.device, (1,), "vroll")[0])
    off = (t * speed + ph0 + 0.08 * torch.sin(t * 1.7)) * i  # in screen heights
    yy = (vv[None] + 1) / 2 + off.view(-1, 1, 1)
    yy = torch.remainder(yy, 1.0)
    gv = yy * 2 - 1
    y = O.warp(x, uu[None].expand_as(gv), gv, "border")
    bar = 0.06
    d = torch.remainder((vv[None] + 1) / 2 + off.view(-1, 1, 1), 1.0)
    # near the seam (d close to 0 or 1) the blanking interval shows
    dist = torch.minimum(d, 1 - d)
    m = O.smoothstep(bar * 0.6, 0.0, dist)[..., None] * min(1.0, i * 4)
    y = y * (1 - m * 0.97)
    y = y + O.smoothstep(bar, bar * 0.4, dist)[..., None] * (1 - m) * 0.08 * min(1.0, i * 4)
    return y.clamp(0, 1)


@look("ghosting", "Ghosting", "retro", "Multipath broadcast ghost: a faint offset echo of every edge.",
      tags=["tv", "antenna", "echo"], synonyms=["ghost image", "multipath", "antenna ghost"])
def ghosting(x, t, i, rng, fps):
    s = O.S(x)
    g1 = O.shift(x, s * 0.035, 0)
    g2 = O.shift(x, s * 0.075, 0)
    y = x * 0.72 + O.blur(g1, s * 0.002) * 0.3 + O.blur(g2, s * 0.004) * 0.12 - 0.06
    y = O.saturate(y, 0.9)
    return O.finish(x, y, i)


@look("camcorder-overlay", "Camcorder", "retro", "Home-video viewfinder: blinking REC dot, running timecode, battery icon and frame marks.",
      tags=["home-video", "90s", "overlay", "found-footage"], synonyms=["camcorder", "rec", "handycam", "found footage", "viewfinder"], animated=True)
def camcorder_overlay(x, t, i, rng, fps):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    s = O.S(x)
    y = O.saturate(x, 1.08)
    y = y.clone()
    sc = max(1, int(round(s / 150)))
    m = int(s * 0.06)
    white = (0.97, 0.97, 0.95)
    rec = font.text_mask("REC", sc, x.device, x.dtype)
    rh, rw = rec.shape
    ys = torch.arange(rh, device=x.device, dtype=x.dtype)
    yd, xd = ys[:, None] - rh / 2 + 0.5, ys[None, :] - rh / 2 + 0.5
    dot = (rh * 0.42 + 0.5 - torch.sqrt(yd * yd + xd * xd)).clamp(0, 1)
    idx = O.frame_index(t, fps)
    tw = 3 * sc
    for f, k in enumerate(idx):
        fr = y[f]
        sec = k / fps
        if (sec % 1.0) < 0.6:
            font.draw_mask(fr, dot, m, m, (0.95, 0.08, 0.06))
        font.draw_mask(fr, rec, m, m + rh + 2 * sc, white)
        hh, mm, ss, ff = int(sec // 3600), int(sec // 60) % 60, int(sec) % 60, int(k % max(1, round(fps)))
        tc = font.text_mask(f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}", sc, x.device, x.dtype)
        font.draw_mask(fr, tc, H - m - tc.shape[0], W - m - tc.shape[1], white)
        sp = font.text_mask("SP", sc, x.device, x.dtype)
        font.draw_mask(fr, sp, H - m - sp.shape[0], m, white)
        # battery icon
        bw, bh = 9 * sc, 4 * sc + 2
        bat = torch.zeros(bh, bw + sc, device=x.device, dtype=x.dtype)
        bat[:, :bw] = 1
        bat[sc:bh - sc, sc:bw - sc] = 0
        bat[sc + 1:bh - sc - 1, sc + 1:int(sc + 1 + (bw - 2 * sc - 2) * 0.66)] = 1
        bat[bh // 3:bh - bh // 3, bw:] = 1
        font.draw_mask(fr, bat, m, W - m - bat.shape[1], white)
    # viewfinder corner brackets
    L, th = int(s * 0.07), max(1, sc)
    br = torch.zeros(H, W, device=x.device, dtype=x.dtype)
    p = int(s * 0.03)
    for (yy, xx, dy, dx) in [(p, p, 1, 1), (p, W - p, 1, -1), (H - p, p, -1, 1), (H - p, W - p, -1, -1)]:
        ya, yb = sorted((yy, yy + dy * L))
        xa, xb = sorted((xx, xx + dx * L))
        ty = sorted((yy, yy + dy * th))
        tx = sorted((xx, xx + dx * th))
        br[ty[0]:ty[1], xa:xb] = 1
        br[ya:yb, tx[0]:tx[1]] = 1
    y = y * (1 - 0.85 * br[None, ..., None]) + 0.85 * br[None, ..., None] * O.col(white, x)
    return O.finish(x, y, i)
