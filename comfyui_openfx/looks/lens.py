import math

import torch
import torch.nn.functional as F

from .. import ops as O
from ..registry import look


def _highlights(x, lo=0.72, hi=1.0, adaptive=False):
    l = O.luma(x)
    if adaptive:
        # dim footage still gets a few sparkle points: threshold follows the brightest pixels
        q = torch.quantile(l.flatten(1)[:, :: max(1, l[0].numel() // 20000)], 0.99, dim=1).view(-1, 1, 1, 1)
        lo = torch.minimum(torch.full_like(q, lo), q - 0.05)
        return x.clamp(0, 1) * O.smoothstep(lo, lo + (hi - lo), l)
    return x.clamp(0, 1) * O.smoothstep(lo, hi, l)


def _bright_spot(x, t, rng, fps, drift=0.0):
    """Per-frame location of the brightest soft region, in aspect units (U, V)."""
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    sh, sw = max(4, H // 16), max(4, W // 16)
    small = O.luma(O.resize(x, sh, sw, "area"))[..., 0]
    small = O.blur_map(small, max(0.8, min(sh, sw) * 0.06))
    # prefer the upper part of the frame a little (lights/sun are usually up)
    ys = torch.linspace(1.0, 0.9, sh, device=x.device, dtype=x.dtype).view(1, sh, 1)
    small = small * ys
    a = small.flatten(1).argmax(1)
    py = (torch.div(a, sw, rounding_mode="floor").to(x.dtype) + 0.5) / sh * 2 - 1
    px = (torch.remainder(a, sw).to(x.dtype) + 0.5) / sw * 2 - 1
    s = min(H, W)
    U, V = px * W / s, py * H / s
    if drift:
        U = U + (O.fbm1(t, 0.3, rng, "flare", "dx") - 0.5) * drift
        V = V + (O.fbm1(t, 0.3, rng, "flare", "dy") - 0.5) * drift * 0.6
    return U, V


def _spot(Ux, Vy, cx, cy, r, soft=1.0):
    d = torch.sqrt((Ux - cx) ** 2 + (Vy - cy) ** 2)
    return d, torch.exp(-(d / r) ** 2 * soft)


@look("lens-flare", "Lens Flare", "lens", "A drifting sun flare anchored to the brightest light: glow, rays, halo and a chain of coloured ghosts.",
      tags=["sun", "flare", "cinematic", "summer"], synonyms=["flare", "sun flare", "jj abrams", "lens glare"], animated=True)
def lens_flare(x, t, i, rng, fps):
    Fn = x.shape[0]
    uu, vv, ax, ay = O.grid(x)
    U, V = (uu * ax)[None], (vv * ay)[None]
    lx, ly = _bright_spot(x, t, rng, fps, drift=0.25)
    cx, cy = lx.view(-1, 1, 1), ly.view(-1, 1, 1)
    d, core = _spot(U, V, cx, cy, 0.09)
    glow = torch.exp(-d / 0.35) * 0.55 + core
    ang = torch.atan2(V - cy, U - cx)
    rot = (t * 0.15).view(-1, 1, 1)
    rays = (torch.cos(ang * 6 + rot) ** 2 * 0.6 + torch.cos(ang * 15 - rot * 2) ** 8 * 0.4) * torch.exp(-d / 0.45) * 0.45
    warm = O.col((1.0, 0.85, 0.6), x)
    acc = (glow + rays)[..., None] * warm
    # rainbow halo ring
    hr = 0.62
    ring = torch.exp(-((d - hr) / 0.035) ** 2) * 0.22
    hue = ((d - hr) / 0.07).clamp(-1, 1)
    rc = torch.stack([O.smoothstep(-0.2, 0.8, hue), 1 - hue.abs() * 0.7, O.smoothstep(0.2, -0.8, hue)], -1)
    acc = acc + ring[..., None] * rc
    # ghosts along the line through the centre
    ghosts = [(-0.35, 0.05, (0.5, 1.0, 0.6)), (0.25, 0.09, (1.0, 0.6, 0.3)), (0.55, 0.035, (0.6, 0.8, 1.0)),
              (0.8, 0.14, (0.4, 0.6, 1.0)), (1.15, 0.06, (1.0, 0.4, 0.7)), (1.45, 0.2, (0.5, 1.0, 0.8))]
    for k, r, c in ghosts:
        gx, gy = cx - (cx * 2) * k, cy - (cy * 2) * k
        dg = torch.sqrt((U - gx) ** 2 + (V - gy) ** 2)
        disc = O.smoothstep(r, r * 0.75, dg) * (0.6 + 0.4 * dg / r)
        acc = acc + (disc * 0.12)[..., None] * O.col(c, x)
    # faint horizontal streak
    acc = acc + (torch.exp(-((V - cy) / 0.012) ** 2) * torch.exp(-(U - cx).abs() / 0.9) * 0.35)[..., None] * O.col((0.7, 0.8, 1.0), x)
    y = O.screen(x, acc.clamp(0, 1))
    return O.finish(x, y, i)


@look("anamorphic-streaks", "Anamorphic Streaks", "lens", "Long blue horizontal streaks flaring off every bright light.",
      tags=["cinematic", "sci-fi", "lights"], synonyms=["anamorphic flare", "blue streak", "horizontal flare", "jj abrams"])
def anamorphic_streaks(x, t, i, rng, fps):
    s = O.S(x)
    W = x.shape[2]
    hl = _highlights(x, 0.8, adaptive=True)
    a = O.blur(hl, s * 0.003, W * 0.06)
    b = O.blur(hl, s * 0.002, W * 0.2)
    streak = (a * 1.6 + b * 2.4)
    tint = O.col((0.35, 0.6, 1.0), x)
    sl = O.luma(streak)
    y = O.screen(x, (sl * tint * 1.2).clamp(0, 1))
    return O.finish(x, y, i)


def _streak(m, ang, length, n=24):
    """Directional decaying smear of map m (NHWC) in both directions."""
    H, W = m.shape[1], m.shape[2]
    acc = torch.zeros_like(m)
    for j in range(1, n + 1):
        f = j / n
        w = math.exp(-f * 4.0)
        dx, dy = math.cos(ang) * f * length, math.sin(ang) * f * length
        acc = acc + (O.shift(m, dx, dy, "zeros") + O.shift(m, -dx, -dy, "zeros")) * w
    return acc


@look("starburst", "Starburst", "lens", "Star-filter sparkle: bright points burst into six-pointed stars.",
      tags=["sparkle", "glamour", "night"], synonyms=["star filter", "sparkle", "twinkle", "star cross"])
def starburst(x, t, i, rng, fps):
    s = O.S(x)
    H, W = x.shape[1], x.shape[2]
    small = O.resize(x, max(4, H // 2), max(4, W // 2), "area")
    hl = _highlights(small, 0.8, adaptive=True) ** 1.5
    # rays come from point-like lights: subtract the local average (top-hat) so big bright areas don't flood
    hl = (hl - O.blur(hl, max(1.0, s * 0.006))).clamp(0, 1) * 1.6
    L = s * 0.09
    acc = torch.zeros_like(hl)
    for k in range(3):
        acc = acc + _streak(hl, math.radians(30 + 60 * k), L / 2, 28)
    acc = O.resize(acc, H, W) * 0.7
    acc = 1 - torch.exp(-acc)
    y = O.screen(x, (acc * O.col((1.0, 0.96, 0.9), x)).clamp(0, 1))
    y = O.screen(y, O.blur(_highlights(x, 0.85, adaptive=True), s * 0.008) * 0.35)
    return O.finish(x, y, i)


@look("bloom", "Bloom", "lens", "Highlights glow and spill softly into their surroundings.",
      tags=["glow", "dreamy", "bright"], synonyms=["glow", "bright glow", "dreamy glow"])
def bloom(x, t, i, rng, fps):
    s = O.S(x)
    hl = _highlights(x, 0.55)
    g = O.blur(hl, s * 0.008) * 0.5 + O.blur(hl, s * 0.025) * 0.6 + O.blur(hl, s * 0.07) * 0.6
    y = O.screen(x, g.clamp(0, 1))
    return O.finish(x, y, i)


@look("vignette", "Vignette", "lens", "Darkened corners that draw the eye to the centre.", tags=["frame", "focus"], synonyms=["dark corners", "dark edges"])
def vignette(x, t, i, rng, fps):
    m = O.vignette_mask(x, 0.7, 0.35, 1.35, 0.6)[None, ..., None]
    y = O.saturate(x, 1.0) * m
    y = O.mix(y, O.saturate(y, 0.8), 1 - m)
    return O.finish(x, y, i)


@look("chromatic-aberration", "Chromatic Aberration", "lens", "Cheap-lens colour fringing that grows towards the edges.",
      tags=["fringe", "lens", "lo-fi"], synonyms=["color fringing", "ca", "lens fringe", "prism fringe"])
def chromatic_aberration(x, t, i, rng, fps):
    uu, vv, ax, ay = O.grid(x)
    k = 0.012 * i
    r = O.warp(x[..., 0:1], uu * (1 - k), vv * (1 - k), "border")
    b = O.warp(x[..., 2:3], uu * (1 + k), vv * (1 + k), "border")
    g = O.warp(x[..., 1:2], uu * (1 + k * 0.3), vv * (1 + k * 0.3), "border")
    y = torch.cat([r, g, b], -1)
    return y.clamp(0, 1)


def _radial(x, fn):
    uu, vv, ax, ay = O.grid(x)
    U, V = uu * ax, vv * ay
    r = torch.sqrt(U * U + V * V)
    rs = fn(r)
    f = rs / r.clamp_min(1e-6)
    return uu * f, vv * f


@look("barrel", "Barrel Distortion", "lens", "Wide-angle bulge: straight lines bow outward.",
      tags=["wide-angle", "distortion"], synonyms=["bulge", "wide angle", "gopro"])
def barrel(x, t, i, rng, fps):
    uu, vv, ax, ay = O.grid(x)
    rc2 = ax * ax + ay * ay
    k = 0.35 * i
    gu, gv = _radial(x, lambda r: r * (1 + k * r * r / rc2) / (1 + k))
    return O.warp(x, gu, gv, "border").clamp(0, 1)


@look("pincushion", "Pincushion", "lens", "Telephoto pinch: straight lines bow inward toward the centre.",
      tags=["distortion"], synonyms=["pinch distortion", "inward distortion"])
def pincushion(x, t, i, rng, fps):
    uu, vv, ax, ay = O.grid(x)
    rc2 = ax * ax + ay * ay
    k = 0.3 * i
    gu, gv = _radial(x, lambda r: r * (1 - k * r * r / rc2))
    return O.warp(x, gu, gv, "border").clamp(0, 1)


@look("fisheye", "Fisheye", "lens", "Extreme ultra-wide fisheye lens with a strongly curved world.",
      tags=["skate", "extreme", "wide-angle"], synonyms=["fish eye", "skate video", "ultra wide", "peephole"])
def fisheye(x, t, i, rng, fps):
    uu, vv, ax, ay = O.grid(x)
    rc = math.sqrt(ax * ax + ay * ay)
    if i <= 1e-3:
        return x
    th = math.radians(10 + 55 * i)  # half field of view at the corner
    # output follows an equidistant fisheye; source is rectilinear (corner -> corner)
    gu, gv = _radial(x, lambda r: torch.tan(r / rc * th) / math.tan(th) * rc)
    y = O.warp(x, gu, gv, "border")
    y = y * O.vignette_mask(x, 0.5 * i, 0.5, 1.3, 0.0)[None, ..., None]
    return y.clamp(0, 1)


@look("prism", "Prism", "lens", "Shooting through a prism: translucent, rainbow-tinted copies float beside the image.",
      tags=["dreamy", "kaleidoscopic", "music-video"], synonyms=["prism lens", "refraction", "double image", "prism filter"])
def prism(x, t, i, rng, fps):
    s = O.S(x)
    off = s * 0.09
    c1 = O.shift(x, -off, -off * 0.25) * O.col((1.0, 0.45, 0.55), x)
    c2 = O.shift(x, off, off * 0.25) * O.col((0.45, 0.75, 1.0), x)
    c3 = O.shift(x, 0, off * 0.9) * O.col((0.7, 1.0, 0.5), x)
    ghosts = (c1 + c2 + c3) / 3
    y = O.screen(x * 0.8, O.blur(ghosts, s * 0.002) * 0.85)
    return O.finish(x, y, i)


@look("kaleidoscope", "Kaleidoscope", "lens", "Eight mirrored wedges turning slowly, like looking through a kaleidoscope.",
      tags=["psychedelic", "symmetry", "trippy"], synonyms=["mirror symmetry", "mandala", "trippy"], animated=True)
def kaleidoscope(x, t, i, rng, fps):
    uu, vv, ax, ay = O.grid(x)
    U, V = (uu * ax)[None], (vv * ay)[None]
    seg = 2 * math.pi / 8
    r = torch.sqrt(U * U + V * V)
    a = torch.atan2(V, U) + (t * 0.25).view(-1, 1, 1)
    a = torch.remainder(a, seg)
    a = torch.minimum(a, seg - a)  # mirror
    a = a + math.pi / 2 - seg / 2 + 0.2
    rr = r * 0.85
    gu = (rr * torch.cos(a)) / ax
    gv = (rr * torch.sin(a) - 0.25) / ay
    y = O.warp(x, gu, gv, "reflection")
    return O.finish(x, y, i)


@look("lens-fog", "Lens Fog", "lens", "A misted-up lens: milky haze drifting across, highlights glowing.",
      tags=["haze", "mist", "dreamy", "moody"], synonyms=["fog", "mist", "haze", "foggy lens", "steam"], animated=True)
def lens_fog(x, t, i, rng, fps):
    H, W = x.shape[1], x.shape[2]
    s = O.S(x)
    n = O.fractal_evo(rng, t, fps, 0.4, 2, H, W, "fog", device=x.device, octaves=3, aspect=W / H)
    edge = 1 - O.vignette_mask(x, 1.0, 0.0, 1.4, 0.3)[None]
    m = (O.smoothstep(0.35, 0.8, n) * 0.45 + edge * 0.45).clamp(0, 1)[..., None]
    b = O.blur(x, s * 0.012)
    glow = O.blur(_highlights(x, 0.6), s * 0.05)
    haze = O.col((0.86, 0.88, 0.9), x)
    y = O.mix(x, b, m)
    y = O.mix(y, haze * (0.55 + 0.45 * O.luma(b)), m * 0.4 + 0.06)
    y = O.screen(y, glow * 0.7)
    y = O.contrast(y, 0.94, 0.7)
    return O.finish(x, y, i)
