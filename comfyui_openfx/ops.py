"""Shared tensor helpers for OpenFX looks.

Every helper works on NHWC float tensors in 0..1 (ComfyUI IMAGE layout) unless
it says otherwise, and runs on whatever device the input lives on.

Randomness: looks never draw from the shared generator directly. They derive
fresh generators from ``rng.initial_seed()`` plus a salt (and, for per-frame
noise, the absolute frame index ``round(t * fps)``). That makes every look
deterministic for a seed *and* independent of how the frames are chunked.
"""

from __future__ import annotations

import math
import zlib

import torch
import torch.nn.functional as F

LUMA = (0.2126, 0.7152, 0.0722)

# ---------------------------------------------------------------- basics


def S(x: torch.Tensor) -> int:
    """Reference size of a frame batch: min(H, W)."""
    return min(x.shape[1], x.shape[2])


def nchw(x):
    return x.permute(0, 3, 1, 2)


def nhwc(x):
    return x.permute(0, 2, 3, 1).contiguous()


def col(c, ref):
    return torch.tensor(c, device=ref.device, dtype=ref.dtype)


def luma(x):
    w = col(LUMA, x)
    return (x * w).sum(-1, keepdim=True)


def mix(a, b, t):
    return a + (b - a) * t


def finish(x, out, intensity):
    """Blend the effect with the source by intensity and clamp."""
    if intensity >= 1.0:
        return out.clamp(0, 1)
    return mix(x, out, float(intensity)).clamp(0, 1)


def smoothstep(e0, e1, x):
    t = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def screen(a, b):
    return 1 - (1 - a) * (1 - b)


def overlay(a, b):
    return torch.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))


def soft_light(a, b):
    return (1 - 2 * b) * a * a + 2 * b * a


def saturate(x, s):
    l = luma(x)
    return l + (x - l) * s


def contrast(x, c, pivot=0.5):
    return (x - pivot) * c + pivot


def scurve(x, k=6.0):
    """Normalised logistic S-curve, k ~ 4 (gentle) .. 12 (hard)."""
    lo = 1 / (1 + math.exp(k * 0.5))
    hi = 1 / (1 + math.exp(-k * 0.5))
    return ((torch.sigmoid(k * (x - 0.5)) - lo) / (hi - lo))


def gradient_map(v, stops, ref=None):
    """Map a [..., 1] value in 0..1 through piecewise-linear colour stops.

    stops: list of (position, (r, g, b)), positions increasing.
    """
    ref = v if ref is None else ref
    out = col(stops[0][1], ref).expand(*v.shape[:-1], 3).clone()
    for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
        w = ((v - p0) / max(p1 - p0, 1e-6)).clamp(0, 1)
        out = out + w * (col(c1, ref) - col(c0, ref))
    return out


def tint_luma(x, color, amount):
    """Push colour toward `color` while keeping luminance."""
    c = col(color, x)
    cl = (c * col(LUMA, x)).sum()
    l = luma(x)
    target = (l * c / cl.clamp_min(1e-4))
    return mix(x, target, amount)


def levels(x, black=0.0, white=1.0, gamma=1.0):
    y = ((x - black) / max(white - black, 1e-4)).clamp(0, 1)
    if gamma != 1.0:
        y = y.pow(1.0 / gamma)
    return y


# ---------------------------------------------------------------- colour spaces


def rgb_to_hsv(x):
    r, g, b = x.unbind(-1)
    mx = x.max(-1).values
    mn = x.min(-1).values
    d = mx - mn
    dd = d.clamp_min(1e-6)
    h = torch.where(
        mx == r,
        torch.remainder((g - b) / dd, 6.0),
        torch.where(mx == g, (b - r) / dd + 2.0, (r - g) / dd + 4.0),
    ) / 6.0
    h = torch.where(d > 1e-6, h, torch.zeros_like(h))
    s = torch.where(mx > 1e-6, d / mx.clamp_min(1e-6), torch.zeros_like(mx))
    return torch.stack([h, s, mx], -1)


def hsv_to_rgb(hsv):
    h, s, v = hsv.unbind(-1)
    out = []
    for n in (5.0, 3.0, 1.0):
        k = torch.remainder(n + h * 6.0, 6.0)
        f = v - v * s * torch.clamp(torch.minimum(k, 4.0 - k), 0, 1)
        out.append(f)
    return torch.stack(out, -1)


_YIQ = ((0.299, 0.587, 0.114), (0.596, -0.274, -0.322), (0.211, -0.523, 0.312))


def rgb_to_yiq(x):
    return x @ col(_YIQ, x).T


def yiq_to_rgb(y):
    m = torch.linalg.inv(torch.tensor(_YIQ, dtype=torch.float64)).to(y.device, y.dtype)
    return y @ m.T


def hue_rotate(x, angle):
    """Rotate hue by `angle` radians (scalar or [F] tensor)."""
    y = rgb_to_yiq(x)
    if not torch.is_tensor(angle):
        angle = torch.tensor(float(angle), device=x.device)
    a = angle.to(x.dtype).reshape(-1, *([1] * (x.dim() - 1)))
    c, s = torch.cos(a), torch.sin(a)
    i, q = y[..., 1:2], y[..., 2:3]
    y = torch.cat([y[..., :1], i * c - q * s, i * s + q * c], -1)
    return yiq_to_rgb(y)


# ---------------------------------------------------------------- filtering


def _gk(sigma, device, dtype):
    r = max(1, int(math.ceil(sigma * 3.0)))
    xs = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (xs / sigma) ** 2)
    return k / k.sum(), r


def _conv1d_sep(x, k, r, axis):
    c = x.shape[1]
    if axis == "x":
        w = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
        x = F.pad(x, (r, r, 0, 0), mode="replicate")
    else:
        w = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
        x = F.pad(x, (0, 0, r, r), mode="replicate")
    return F.conv2d(x, w, groups=c)


def blur_nchw(x, sy, sx=None):
    """Separable gaussian blur on NCHW; large sigmas run on a downsampled copy."""
    sx = sy if sx is None else sx
    h, w = x.shape[-2:]
    fy = max(1.0, sy / 3.0) if sy > 6 else 1.0
    fx = max(1.0, sx / 3.0) if sx > 6 else 1.0
    small = fy > 1 or fx > 1
    y = x
    if small:
        hh, ww = max(1, int(round(h / fy))), max(1, int(round(w / fx)))
        fy, fx = h / hh, w / ww
        y = F.interpolate(y, size=(hh, ww), mode="area")
    if sx / fx > 0.2:
        k, r = _gk(sx / fx, x.device, x.dtype)
        y = _conv1d_sep(y, k, r, "x")
    if sy / fy > 0.2:
        k, r = _gk(sy / fy, x.device, x.dtype)
        y = _conv1d_sep(y, k, r, "y")
    if small:
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
    return y


def blur(x, sigma, sigma_x=None):
    """Gaussian blur of an NHWC tensor (sigma in pixels)."""
    if sigma <= 0.2 and (sigma_x is None or sigma_x <= 0.2):
        return x
    return nhwc(blur_nchw(nchw(x), sigma, sigma_x))


def blur_map(m, sigma):
    """Blur a [F,H,W] map."""
    return blur_nchw(m.unsqueeze(1), sigma)[:, 0]


def box_blur(x, r):
    r = int(round(r))
    if r < 1:
        return x
    y = F.pad(nchw(x), (r, r, r, r), mode="replicate")
    y = F.avg_pool2d(y, 2 * r + 1, stride=1)
    return nhwc(y)


def disc_kernel(radius, device, dtype):
    r = int(math.ceil(radius))
    xs = torch.arange(-r, r + 1, device=device, dtype=dtype)
    d = torch.sqrt(xs[None, :] ** 2 + xs[:, None] ** 2)
    k = (radius + 0.5 - d).clamp(0, 1)
    return k / k.sum(), r


def disc_conv_nchw(x, radius, max_r=7.0):
    """Convolve NCHW with an antialiased disc (lens aperture)."""
    if radius < 0.6:
        return x
    h, w = x.shape[-2:]
    f = max(1.0, radius / max_r)
    y = x
    if f > 1:
        y = F.interpolate(y, size=(max(1, int(round(h / f))), max(1, int(round(w / f)))), mode="area")
    k, r = disc_kernel(radius / f, x.device, x.dtype)
    c = y.shape[1]
    y = F.pad(y, (r, r, r, r), mode="replicate")
    y = F.conv2d(y, k[None, None].repeat(c, 1, 1, 1), groups=c)
    if f > 1:
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
    return y


def resize(x, h, w, mode="bilinear"):
    y = nchw(x)
    if mode == "area":
        y = F.interpolate(y, size=(h, w), mode="area")
    elif mode == "nearest":
        y = F.interpolate(y, size=(h, w), mode="nearest")
    else:
        y = F.interpolate(y, size=(h, w), mode=mode, align_corners=False, antialias=(h < x.shape[1] or w < x.shape[2]) and mode in ("bilinear", "bicubic"))
    return nhwc(y)


def sobel(l):
    """Gradient magnitude of a [F,H,W,1] map."""
    y = F.pad(nchw(l), (1, 1, 1, 1), mode="replicate")
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=l.device, dtype=l.dtype) / 4
    gx = F.conv2d(y, kx[None, None])
    gy = F.conv2d(y, kx.T.contiguous()[None, None])
    return nhwc(torch.sqrt(gx * gx + gy * gy + 1e-8))


def xdog(l, sigma, k=1.6, tau=0.98, eps=0.0, phi=60.0):
    """Extended difference of gaussians -> ink-like lines (1 = paper, 0 = ink)."""
    g1 = blur(l, sigma)
    g2 = blur(l, sigma * k)
    d = g1 - tau * g2
    e = torch.where(d >= eps, torch.ones_like(d), 1 + torch.tanh(phi * (d - eps)))
    return e.clamp(0, 1)


# ---------------------------------------------------------------- geometry


def grid(x):
    """Return (u, v) sampling coords in -1..1 of shape [H, W] and aspect scale (ax, ay).

    Aspect-correct coordinates are (u * ax, v * ay): one unit = half of min(H, W).
    """
    h, w = x.shape[1], x.shape[2]
    v = (torch.arange(h, device=x.device, dtype=x.dtype) + 0.5) / h * 2 - 1
    u = (torch.arange(w, device=x.device, dtype=x.dtype) + 0.5) / w * 2 - 1
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    s = min(h, w)
    return uu, vv, w / s, h / s


def warp(x, gu, gv, padding="border"):
    """Sample x at normalised coords (gu, gv); each is [H,W] or [F,H,W]."""
    if gu.dim() == 2:
        gu = gu[None]
        gv = gv[None]
    g = torch.stack([gu, gv], -1)
    if g.shape[0] == 1 and x.shape[0] > 1:
        g = g.expand(x.shape[0], -1, -1, -1)
    y = F.grid_sample(nchw(x), g.to(x.dtype), mode="bilinear", padding_mode=padding, align_corners=False)
    return nhwc(y)


def inside(gu, gv, soft=0.0):
    m = ((gu.abs() <= 1) & (gv.abs() <= 1)).to(gu.dtype)
    return m


def shift(x, dx, dy=0.0, padding="border"):
    """Translate by (dx, dy) pixels (floats ok). dx may be an [F] tensor."""
    h, w = x.shape[1], x.shape[2]
    uu, vv, _, _ = grid(x)
    if torch.is_tensor(dx):
        dx = dx.view(-1, 1, 1)
    if torch.is_tensor(dy):
        dy = dy.view(-1, 1, 1)
    gu = uu - dx * 2.0 / w
    gv = vv - dy * 2.0 / h
    if gu.dim() == 2:
        gu = gu.expand(1, -1, -1)
    if gv.dim() == 2:
        gv = gv.expand(1, -1, -1)
    n = max(gu.shape[0], gv.shape[0])
    return warp(x, gu.expand(n, -1, -1), gv.expand(n, -1, -1), padding)


def cover(x, h, w):
    """Resize x to cover (h, w) and centre-crop."""
    H, W = x.shape[1], x.shape[2]
    s = max(h / H, w / W)
    nh, nw = max(h, int(math.ceil(H * s))), max(w, int(math.ceil(W * s)))
    y = resize(x, nh, nw)
    y0, x0 = (nh - h) // 2, (nw - w) // 2
    return y[:, y0:y0 + h, x0:x0 + w]


def paste(dst, src, y0, x0):
    """Paste src [F,h,w,3] into dst at (y0, x0), clipping at borders."""
    H, W = dst.shape[1], dst.shape[2]
    h, w = src.shape[1], src.shape[2]
    ya, xa = max(0, y0), max(0, x0)
    yb, xb = min(H, y0 + h), min(W, x0 + w)
    if yb <= ya or xb <= xa:
        return dst
    dst[:, ya:yb, xa:xb] = src[:, ya - y0:yb - y0, xa - x0:xb - x0]
    return dst


def rounded_rect_mask(h, w, y0, x0, rh, rw, radius, device, dtype, soft=1.0):
    """Antialiased rounded-rectangle mask [h, w] for a box at (y0, x0) of size (rh, rw)."""
    ys = torch.arange(h, device=device, dtype=dtype) + 0.5
    xs = torch.arange(w, device=device, dtype=dtype) + 0.5
    cy, cx = y0 + rh / 2, x0 + rw / 2
    qy = (ys[:, None] - cy).abs() - (rh / 2 - radius)
    qx = (xs[None, :] - cx).abs() - (rw / 2 - radius)
    d = torch.sqrt(qx.clamp_min(0) ** 2 + qy.clamp_min(0) ** 2) + torch.minimum(torch.maximum(qx, qy), torch.zeros_like(qx)) - radius
    return (0.5 - d / max(soft, 1e-3)).clamp(0, 1)


def vignette_mask(x, amount=0.5, start=0.35, end=1.25, roundness=0.5):
    uu, vv, ax, ay = grid(x)
    # blend between circular (aspect-correct) and elliptical (frame-fitting)
    cu = mix(uu * ax, uu * math.sqrt(ax * ay), roundness)
    cv = mix(vv * ay, vv * math.sqrt(ax * ay), roundness)
    r = torch.sqrt(cu * cu + cv * cv) / math.sqrt(ax * ay)
    return 1 - amount * smoothstep(start, end, r)


# ---------------------------------------------------------------- randomness

_M64 = 0xFFFFFFFFFFFFFFFF


def mixseed(*vals) -> int:
    h = 0x9E3779B97F4A7C15
    flat = []
    stack = list(vals)
    while stack:
        v = stack.pop(0)
        if isinstance(v, (tuple, list)):
            stack = list(v) + stack
        else:
            flat.append(v)
    for v in flat:
        if isinstance(v, str):
            v = zlib.crc32(v.encode())
        elif isinstance(v, float):
            v = int(v * 1000003)
        h ^= int(v) & _M64
        h = (h * 0xBF58476D1CE4E5B9) & _M64
        h ^= h >> 31
        h = (h * 0x94D049BB133111EB) & _M64
        h ^= h >> 29
    return h & ((1 << 63) - 1)


def seed_of(rng) -> int:
    return int(rng.initial_seed()) if rng is not None else 0


def gen(rng, device, *salt):
    g = torch.Generator(device=device)
    g.manual_seed(mixseed(seed_of(rng), *salt))
    return g


def urand(rng, device, shape, *salt):
    """Global (frame-independent) uniform randoms."""
    return torch.rand(shape, generator=gen(rng, device, *salt), device=device)


def nrand(rng, device, shape, *salt):
    return torch.randn(shape, generator=gen(rng, device, *salt), device=device)


def frame_index(t, fps):
    return [int(round(v)) for v in (t.double() * float(fps)).tolist()]


def key_rand(rng, keys, shape, *salt, device, normal=False):
    """Stack of random tensors, one per integer key (e.g. absolute frame index)."""
    fn = torch.randn if normal else torch.rand
    out = [fn(shape, generator=gen(rng, device, *salt, k), device=device) for k in keys]
    return torch.stack(out, 0)


def frame_rand(rng, t, fps, shape, *salt, device, normal=False):
    return key_rand(rng, frame_index(t, fps), shape, *salt, device=device, normal=normal)


def hash1(n, seed):
    """Deterministic hash of float64 tensor n -> 0..1."""
    s = float(seed % 9973) * 0.6180339887
    v = torch.sin(n * 12.9898 + s * 78.233 + 0.5) * 43758.5453
    return v - torch.floor(v)


def vnoise1(t, freq, rng, *salt):
    """Smooth 1-D value noise over time t (seconds) -> [F] in 0..1."""
    seed = mixseed(seed_of(rng), *salt)
    x = t.double() * freq + (seed % 1000) * 1.37
    i = torch.floor(x)
    f = x - i
    f = f * f * (3 - 2 * f)
    a = hash1(i, seed)
    b = hash1(i + 1, seed)
    return (a + (b - a) * f).to(t.dtype)


def fbm1(t, freq, rng, *salt, octaves=3):
    v, amp, tot = 0, 1.0, 0.0
    for o in range(octaves):
        v = v + amp * vnoise1(t, freq * (2 ** o), rng, *salt, o)
        tot += amp
        amp *= 0.5
    return v / tot


def noise2(rng, n, gh, gw, H, W, *salt, device, mode="bicubic"):
    """Static smooth noise map(s) [n,H,W], 0..1-ish."""
    g = urand(rng, device, (n, 1, gh, gw), *salt)
    return F.interpolate(g, size=(H, W), mode=mode, align_corners=False)[:, 0]


def evo_noise(rng, t, fps, rate, gh, gw, H, W, *salt, device):
    """Smooth 2-D noise that evolves over time: [F,H,W] in ~0..1.

    rate: keyframes per second. Keys are absolute, so chunking is irrelevant.
    """
    x = t.double() * rate
    k0 = torch.floor(x)
    f = (x - k0).to(torch.float32)
    f = (f * f * (3 - 2 * f)).to(device)
    keys = [int(k) for k in k0.tolist()]
    a = key_rand(rng, keys, (gh, gw), *salt, device=device)
    b = key_rand(rng, [k + 1 for k in keys], (gh, gw), *salt, device=device)
    m = a + (b - a) * f.view(-1, 1, 1)
    return F.interpolate(m[:, None], size=(H, W), mode="bicubic", align_corners=False)[:, 0]


def fractal_evo(rng, t, fps, rate, base, H, W, *salt, device, octaves=3, aspect=1.0):
    acc, amp, tot = 0, 1.0, 0.0
    for o in range(octaves):
        gh = max(2, int(base * (2 ** o)))
        gw = max(2, int(base * (2 ** o) * aspect))
        acc = acc + amp * evo_noise(rng, t, fps, rate * (1.5 ** o), gh, gw, H, W, *salt, o, device=device)
        tot += amp
        amp *= 0.5
    return acc / tot


def grain(rng, t, fps, H, W, size, *salt, device, mono=True):
    """Per-frame film grain field, zero-mean unit-ish std. Shape [F,H,W,1 or 3]."""
    gh, gw = max(2, int(round(H / size))), max(2, int(round(W / size)))
    c = 1 if mono else 3
    n = frame_rand(rng, t, fps, (c, gh, gw), *salt, device=device, normal=True)
    if gh != H or gw != W:
        n = F.interpolate(n, size=(H, W), mode="bilinear", align_corners=False)
        n = n * min(2.0, 0.6 + 0.4 * size)  # bilinear upsampling reduces variance
    return n.permute(0, 2, 3, 1)


def apply_grain(x, g, amount):
    """Luminance-dependent grain: strongest in mid-tones, softer in deep shadows/highlights."""
    l = luma(x)
    w = 0.35 + 2.6 * l * (1 - l)
    return x + g * amount * w
