import math

import torch

from .. import ops as O
from ..registry import look


def _polar_warp(x, fn, padding="reflection"):
    """fn(U, V, r, a) -> (r_src, a_src) in aspect units; may return [F,H,W] tensors."""
    uu, vv, ax, ay = O.grid(x)
    U, V = uu * ax, vv * ay
    r = torch.sqrt(U * U + V * V)
    a = torch.atan2(V, U)
    rs, as_ = fn(U, V, r, a)
    gu = rs * torch.cos(as_) / ax
    gv = rs * torch.sin(as_) / ay
    if gu.dim() == 2:
        gu, gv = gu[None], gv[None]
    return O.warp(x, gu, gv, padding)


@look("wave", "Wave", "distort", "Gentle sideways waves rolling down the picture, like a reflection in water.",
      tags=["water", "wobble", "dreamy"], synonyms=["wavy", "water reflection", "wobble", "underwater"], animated=True)
def wave(x, t, i, rng, fps):
    uu, vv, ax, ay = O.grid(x)
    T = t.view(-1, 1, 1)
    A = 0.03 * i
    gu = uu[None] + A * torch.sin(vv[None] * ay * 9.0 + T * 2.5) / ax
    gv = vv[None] + A * 0.4 * torch.sin(uu[None] * ax * 6.0 + T * 1.7) / ay
    return O.warp(x, gu, gv, "reflection").clamp(0, 1)


@look("ripple", "Ripple", "distort", "Concentric ripples spreading out from the centre, like a stone dropped in a pond.",
      tags=["water", "pond", "rings"], synonyms=["water ripple", "pond", "shockwave", "drop"], animated=True)
def ripple(x, t, i, rng, fps):
    A = 0.035 * i
    T = t.view(-1, 1, 1)

    def fn(U, V, r, a):
        r = r[None]
        d = A * torch.sin(r * 22 - T * 6.0) * torch.exp(-r * 1.2)
        return r + d, a[None].expand_as(r)

    return _polar_warp(x, fn).clamp(0, 1)


@look("turbulence", "Turbulence", "distort", "Churning heat-haze displacement that never sits still.",
      tags=["heat-haze", "warp", "liquid"], synonyms=["heat haze", "heat wave", "mirage", "liquid", "warp"], animated=True)
def turbulence(x, t, i, rng, fps):
    H, W = x.shape[1], x.shape[2]
    uu, vv, ax, ay = O.grid(x)
    nx = O.fractal_evo(rng, t, fps, 1.5, 4, H, W, "turbx", device=x.device, octaves=2, aspect=W / H) - 0.5
    ny = O.fractal_evo(rng, t, fps, 1.5, 4, H, W, "turby", device=x.device, octaves=2, aspect=W / H) - 0.5
    A = 0.25 * i
    return O.warp(x, uu[None] + nx * A / ax, vv[None] + ny * A / ay, "reflection").clamp(0, 1)


@look("bulge", "Bulge", "distort", "The centre swells toward the camera like a fish-bowl.",
      tags=["funny", "big-head", "fishbowl"], synonyms=["big head", "fish bowl", "magnify", "bubble"])
def bulge(x, t, i, rng, fps):
    R = 0.85
    k = 0.55 * i

    def fn(U, V, r, a):
        q = (r / R).clamp(max=1)
        return r * (1 - k * (1 - q) ** 2), a

    return _polar_warp(x, fn).clamp(0, 1)


@look("pinch", "Pinch", "distort", "The centre is sucked inward, squeezing the picture to a point.",
      tags=["funny", "squeeze"], synonyms=["squeeze", "suck in", "implode"])
def pinch(x, t, i, rng, fps):
    R = 0.9
    k = 0.6 * i

    def fn(U, V, r, a):
        q = (r / R).clamp(max=1)
        return r * (1 + k * (1 - q) ** 1.5 * 1.4), a

    return _polar_warp(x, fn).clamp(0, 1)


@look("twirl", "Twirl", "distort", "A swirl that twists the middle of the frame.",
      tags=["swirl", "trippy"], synonyms=["swirl", "twist"])
def twirl(x, t, i, rng, fps):
    R = 0.9
    ang = 2.6 * i

    def fn(U, V, r, a):
        q = (r / R).clamp(max=1)
        return r, a + ang * (1 - q) ** 2

    return _polar_warp(x, fn).clamp(0, 1)


@look("vortex", "Vortex", "distort", "A spinning whirlpool that keeps twisting the centre of the frame.",
      tags=["whirlpool", "trippy", "portal"], synonyms=["whirlpool", "portal", "spiral", "black hole"], animated=True)
def vortex(x, t, i, rng, fps):
    R = 1.0
    T = t.view(-1, 1, 1)

    def fn(U, V, r, a):
        q = (r / R).clamp(max=1)[None]
        w = (1 - q) ** 2
        spin = (1.8 + 0.8 * torch.sin(T * 1.3)) * w + T * 1.6 * w
        return r[None] * (1 - 0.15 * w * i), a[None] + spin * i

    return _polar_warp(x, fn).clamp(0, 1)


@look("mirror", "Mirror", "distort", "Left half reflected onto the right for perfect symmetry.",
      tags=["symmetry", "reflection"], synonyms=["symmetry", "reflect", "mirrored"])
def mirror(x, t, i, rng, fps):
    W = x.shape[2]
    h = (W + 1) // 2
    left = x[:, :, :h]
    y = torch.cat([left, left.flip(2)[:, :, (2 * h - W):]], 2)
    return O.finish(x, y, i)


@look("tile", "Tile", "distort", "A seamless 2x2 mirrored tiling of the frame.",
      tags=["pattern", "repeat"], synonyms=["repeat", "tiles", "grid"])
def tile(x, t, i, rng, fps):
    H, W = x.shape[1], x.shape[2]
    h2, w2 = (H + 1) // 2, (W + 1) // 2
    s = O.resize(x, h2, w2, "area")
    top = torch.cat([s, s.flip(2)], 2)
    y = torch.cat([top, top.flip(1)], 1)[:, :H, :W]
    return O.finish(x, y, i)


@look("polar", "Polar", "distort", "Wraps the frame into a circle: a tiny-planet view with the ground at the centre.",
      tags=["tiny-planet", "circle"], synonyms=["tiny planet", "little planet", "polar coordinates", "globe"])
def polar(x, t, i, rng, fps):
    uu, vv, ax, ay = O.grid(x)
    U, V = uu * ax, vv * ay
    r = torch.sqrt(U * U + V * V)
    a = torch.atan2(U, V)  # 0 pointing down
    rmax = math.sqrt(ax * ax + ay * ay) * 0.92
    gu = (a.abs() / math.pi) * 2 - 1  # mirrored so the wrap has no seam
    gv = 1 - 2 * (r / rmax).clamp(0, 1)
    y = O.warp(x, gu, gv, "border")
    return O.finish(x, y, i)


@look("perspective-tilt", "Perspective Tilt", "distort", "The frame tips back in 3D, receding into the distance.",
      tags=["3d", "tilt", "star-wars"], synonyms=["3d tilt", "keystone", "tilted", "star wars crawl"])
def perspective_tilt(x, t, i, rng, fps):
    uu, vv, ax, ay = O.grid(x)
    k = 0.5 * i
    yn = (vv + 1) / 2  # 0 top .. 1 bottom
    # homography of a plane tipped away from the camera: top recedes
    z = (1 + k) / (1 + k * yn)
    gu = uu * z
    gv = (yn * z) * 2 - 1
    if i <= 1e-3:
        return x
    H, W = x.shape[1], x.shape[2]
    y = O.warp(x, gu, gv, "zeros")
    bg = O.col((0.03, 0.03, 0.04), x)
    edge = (O.smoothstep(1 + 1.0 / W, 1 - 1.0 / W, gu.abs()) * O.smoothstep(1 + 1.0 / H, 1 - 1.0 / H, gv.abs()))[None, ..., None]
    y = y * edge + bg * (1 - edge)
    return y.clamp(0, 1)
