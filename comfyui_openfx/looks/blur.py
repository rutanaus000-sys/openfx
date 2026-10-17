import math

import torch

from .. import ops as O
from ..registry import look

_G = 2.2


def _lin(x):
    return x.clamp(0, 1).pow(_G)


def _gam(x):
    return x.clamp(0, 1).pow(1 / _G)


def _tap_blur(x, coords_fn, n):
    """Average n warped copies; coords_fn(k in -1..1) -> (gu, gv)."""
    acc = torch.zeros_like(x)
    for j in range(n):
        k = (j / (n - 1)) * 2 - 1 if n > 1 else 0.0
        gu, gv = coords_fn(k)
        acc = acc + O.warp(x, gu, gv, "border")
    return acc / n


@look("gaussian-blur", "Gaussian Blur", "blur", "Smooth, even defocus.", tags=["soft", "defocus"], synonyms=["blur", "blurry", "soft", "out of focus"])
def gaussian_blur(x, t, i, rng, fps):
    return O.blur(x, O.S(x) * 0.012 * i).clamp(0, 1)


@look("box-blur", "Box Blur", "blur", "Flat, boxy averaging blur with a slightly digital character.", tags=["soft", "digital"], synonyms=["average blur", "mean blur"])
def box_blur(x, t, i, rng, fps):
    r = O.S(x) * 0.014 * i
    if r < 0.5:
        return x
    return O.box_blur(x, r).clamp(0, 1)


@look("lens-blur", "Lens Blur", "blur", "Optical out-of-focus blur with a round aperture and luminous highlights.",
      tags=["defocus", "optical", "cinematic"], synonyms=["defocus", "out of focus", "camera blur", "dof"])
def lens_blur(x, t, i, rng, fps):
    r = O.S(x) * 0.016 * i
    y = _gam(O.nhwc(O.disc_conv_nchw(O.nchw(_lin(x)), r)))
    return y.clamp(0, 1)


@look("bokeh", "Bokeh", "blur", "Dreamy defocus where highlights bloom into bright round discs.",
      tags=["dreamy", "defocus", "lights"], synonyms=["bokeh balls", "background blur", "portrait blur", "fairy lights"])
def bokeh(x, t, i, rng, fps):
    s = O.S(x)
    r = s * 0.024 * i
    lin = _lin(x)
    l = O.luma(x)
    w = 1 + 8 * O.smoothstep(0.75, 1.0, l)
    num = O.nhwc(O.disc_conv_nchw(O.nchw(lin * w), r, 9))
    den = O.nhwc(O.disc_conv_nchw(O.nchw(w), r, 9))
    y = num / den
    hl = O.nhwc(O.disc_conv_nchw(O.nchw(lin * O.smoothstep(0.75, 1.0, l)), r, 9))
    y = _gam(y + hl * 0.8 * i)
    return O.finish(x, y, 1.0)


@look("directional-blur", "Directional Blur", "blur", "Motion blur streaking along a diagonal, as if panning fast.",
      tags=["motion", "speed"], synonyms=["motion blur", "speed blur", "streak"])
def directional_blur(x, t, i, rng, fps):
    H, W = x.shape[1], x.shape[2]
    L = O.S(x) * 0.05 * i
    if L < 0.5:
        return x
    ang = math.radians(12)
    uu, vv, _, _ = O.grid(x)
    n = int(min(32, max(4, L / 2)))
    dx, dy = math.cos(ang) * L, math.sin(ang) * L
    return _tap_blur(x, lambda k: (uu + k * dx / W, vv + k * dy / H), n).clamp(0, 1)


@look("spin-blur", "Spin Blur", "blur", "Rotational blur swirling around the centre.",
      tags=["motion", "rotation"], synonyms=["radial blur", "rotation blur", "twist blur"])
def spin_blur(x, t, i, rng, fps):
    a = math.radians(5) * i
    if a <= 1e-4:
        return x
    uu, vv, ax, ay = O.grid(x)
    U, V = uu * ax, vv * ay

    def c(k):
        ca, sa = math.cos(k * a), math.sin(k * a)
        return (U * ca - V * sa) / ax, (U * sa + V * ca) / ay

    return _tap_blur(x, c, 24).clamp(0, 1)


@look("zoom-blur", "Zoom Blur", "blur", "Explosive radial blur rushing out from the centre.",
      tags=["motion", "speed", "impact"], synonyms=["radial blur", "warp speed", "rush"])
def zoom_blur(x, t, i, rng, fps):
    z = 0.09 * i
    if z <= 1e-4:
        return x
    uu, vv, _, _ = O.grid(x)
    y = _tap_blur(x, lambda k: (uu * (1 - z * (k + 1) / 2), vv * (1 - z * (k + 1) / 2)), 24)
    return y.clamp(0, 1)


@look("tilt-shift", "Tilt Shift", "blur", "Miniature-model effect: a sharp horizontal band, blurred top and bottom, punchier colour.",
      tags=["miniature", "toy", "diorama"], synonyms=["miniature", "toy town", "diorama", "fake miniature"])
def tilt_shift(x, t, i, rng, fps):
    H = x.shape[1]
    s = O.S(x)
    ys = (torch.arange(H, device=x.device, dtype=x.dtype) + 0.5) / H
    d = ((ys - 0.56).abs() - 0.09).clamp_min(0) / 0.3
    m = d.clamp(0, 1).view(1, H, 1, 1)
    b1 = O.nhwc(O.disc_conv_nchw(O.nchw(x), s * 0.006))
    b2 = O.nhwc(O.disc_conv_nchw(O.nchw(x), s * 0.016))
    y = torch.where(m < 0.5, O.mix(x, b1, m * 2), O.mix(b1, b2, (m - 0.5) * 2))
    y = O.mix(y, O.scurve(O.saturate(y, 1.25).clamp(0, 1), 4.5), 0.8)
    return O.finish(x, y, i)


@look("soft-focus", "Soft Focus", "blur", "Dreamy diffusion filter: a soft glow blooms around everything.",
      tags=["dreamy", "glow", "romantic", "portrait"], synonyms=["diffusion", "dreamy", "glow", "pro-mist", "soft glow"])
def soft_focus(x, t, i, rng, fps):
    s = O.S(x)
    g = O.blur(x, s * 0.02)
    y = O.screen(x, g * 0.55)
    y = O.mix(y, O.blur(x, s * 0.004), 0.35)
    y = O.contrast(y, 0.92, 0.6)
    return O.finish(x, y, i)


@look("sharpen", "Sharpen", "blur", "Crisper edges and fine detail.", tags=["crisp", "detail"], synonyms=["sharp", "crisp", "detail", "clarity"])
def sharpen(x, t, i, rng, fps):
    s = O.S(x)
    b = O.blur(x, max(0.8, s * 0.0025))
    y = x + (x - b) * 1.4
    b2 = O.blur(x, s * 0.012)
    y = y + (x - b2) * 0.25
    return O.finish(x, y, i)
