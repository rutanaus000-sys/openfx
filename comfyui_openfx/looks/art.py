import math

import torch
import torch.nn.functional as F

from .. import font
from .. import ops as O
from ..registry import look


def _rot_cells(H, W, cell, ang, device, dtype):
    """For a rotated square grid: per-pixel (local offset from cell centre, cell centre in pixels)."""
    ys = torch.arange(H, device=device, dtype=dtype) + 0.5
    xs = torch.arange(W, device=device, dtype=dtype) + 0.5
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    c, s = math.cos(ang), math.sin(ang)
    u = (X * c + Y * s) / cell
    v = (-X * s + Y * c) / cell
    cu, cv = torch.floor(u) + 0.5, torch.floor(v) + 0.5
    lu, lv = u - cu, v - cv
    # centre back in image space
    cx = (cu * c - cv * s) * cell
    cy = (cu * s + cv * c) * cell
    return lu, lv, cx, cy


def _sample_px(img, cx, cy):
    """Sample NHWC img at pixel coords (cx, cy) [H,W]."""
    H, W = img.shape[1], img.shape[2]
    gu = cx / W * 2 - 1
    gv = cy / H * 2 - 1
    return O.warp(img, gu, gv, "border")


def _halftone(x, cell, angles, src=None, cmy=True):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    src = O.blur(x, cell * 0.35) if src is None else src
    inks = []
    for ch, ang in enumerate(angles):
        lu, lv, cx, cy = _rot_cells(H, W, cell, math.radians(ang), x.device, x.dtype)
        val = _sample_px(src[..., ch:ch + 1] if cmy else src, cx, cy)[..., 0]
        cov = (1 - val).clamp(0, 1)
        r = torch.sqrt(cov / math.pi) * 1.05  # dot area ~ ink coverage
        d = torch.sqrt(lu * lu + lv * lv)[None]
        aa = 1.0 / cell
        inks.append(O.smoothstep(r + aa, r - aa, d))
    return inks


@look("halftone", "Halftone", "art", "Printed-magazine CMY dot screen with angled rosettes.",
      tags=["print", "pop-art", "dots"], synonyms=["dots", "ben day", "print", "newspaper", "magazine"])
def halftone(x, t, i, rng, fps):
    s = O.S(x)
    cell = max(3.0, s / 70)
    src = O.scurve(O.saturate(x, 1.4).clamp(0, 1), 4.0)
    inks = _halftone(src, cell, [15, 75, 0], O.blur(src, cell * 0.3))
    paper = O.col((0.97, 0.96, 0.92), x)
    y = paper * torch.stack([1 - inks[0] * 0.95, 1 - inks[1] * 0.95, 1 - inks[2] * 0.95], -1)
    return O.finish(x, y, i)


@look("comic", "Comic Book", "art", "Comic-book panel: flat bold colours, inked outlines and halftone shadows.",
      tags=["cartoon", "pop-art", "ink"], synonyms=["comic book", "cartoon", "graphic novel", "pop art", "marvel"])
def comic(x, t, i, rng, fps):
    s = O.S(x)
    b = O.blur(x, s * 0.004)
    hsv = O.rgb_to_hsv(O.saturate(b, 1.5).clamp(0, 1))
    v = hsv[..., 2:3]
    vq = torch.round(v.clamp(0, 1) * 3) / 3
    vq = 0.18 + vq * 0.82
    hsv = torch.cat([hsv[..., :1], (hsv[..., 1:2] * 1.1).clamp(0, 1), vq], -1)
    flat = O.hsv_to_rgb(hsv)
    l = O.luma(b)
    ink = O.xdog(l, max(0.8, s * 0.0028), tau=0.985, phi=80)
    # mono halftone dots in the darker tones
    cell = max(3.0, s / 110)
    lu, lv, cx, cy = _rot_cells(x.shape[1], x.shape[2], cell, math.radians(45), x.device, x.dtype)
    val = _sample_px(l, cx, cy)[..., 0]
    r = torch.sqrt((0.6 - val).clamp(0, 1) / 0.6) * 0.5
    d = torch.sqrt(lu * lu + lv * lv)[None]
    dots = O.smoothstep(r + 1 / cell, r - 1 / cell, d)[..., None]
    y = flat * (1 - 0.55 * dots)
    y = y * ink
    return O.finish(x, y, i)


@look("pencil-sketch", "Pencil Sketch", "art", "Graphite drawing on paper with soft shading and cross-hatching.",
      tags=["drawing", "sketch", "graphite"], synonyms=["sketch", "drawing", "pencil", "graphite", "hand drawn"])
def pencil_sketch(x, t, i, rng, fps):
    s = O.S(x)
    H, W = x.shape[1], x.shape[2]
    g = O.luma(x)
    inv = 1 - g
    bl = O.blur(inv, s * 0.012)
    dodge = (g / (1 - bl).clamp_min(0.02)).clamp(0, 1)
    dodge = dodge.pow(2.2)
    # hatching in mid/dark tones
    ys = torch.arange(H, device=x.device, dtype=x.dtype)
    xs = torch.arange(W, device=x.device, dtype=x.dtype)
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    p = max(3.0, s / 120)
    h1 = 0.5 + 0.5 * torch.sin((X + Y) / p * math.pi)
    h2 = 0.5 + 0.5 * torch.sin((X - Y) / p * math.pi)
    tone = O.blur(g, s * 0.006)
    hatch = 1 - (O.smoothstep(0.55, 0.15, tone) * (1 - h1[None, ..., None]) ** 3 * 0.35
                 + O.smoothstep(0.32, 0.05, tone) * (1 - h2[None, ..., None]) ** 3 * 0.35)
    n = O.noise2(rng, 1, max(2, H // 2), max(2, W // 2), H, W, "paper", device=x.device)[0]
    paper = 0.94 + 0.05 * n[None, ..., None]
    y = dodge * hatch * paper
    y = y * O.col((0.98, 0.97, 0.94), x)
    return O.finish(x, y.expand_as(x), i)


@look("ink-outline", "Ink Outline", "art", "Crisp black ink linework over pale watercolour washes.",
      tags=["line-art", "illustration", "ink"], synonyms=["line art", "outline", "ink drawing", "illustration", "lineart"])
def ink_outline(x, t, i, rng, fps):
    s = O.S(x)
    l = O.luma(O.blur(x, s * 0.002))
    ink = O.xdog(l, max(0.7, s * 0.0022), tau=0.99, phi=120)
    wash = O.blur(O.saturate(x, 0.75), s * 0.006)
    wash = 1 - (1 - wash) * 0.45
    y = wash * ink
    return O.finish(x, y, i)


@look("mosaic", "Mosaic", "art", "Glossy ceramic tiles with grout lines, each tile a single colour.",
      tags=["tiles", "ceramic", "craft"], synonyms=["tiles", "tile mosaic", "roman mosaic"])
def mosaic(x, t, i, rng, fps):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    s = O.S(x)
    cell = max(4.0, s / 48)
    bh, bw = max(1, round(H / cell)), max(1, round(W / cell))
    avg = F.adaptive_avg_pool2d(O.nchw(x), (bh, bw))
    var = O.urand(rng, x.device, (1, 1, bh, bw), "mosaic") * 0.14 + 0.93
    avg = avg * var
    tiles = O.nhwc(F.interpolate(avg, size=(H, W), mode="nearest"))
    ys = (torch.arange(H, device=x.device, dtype=x.dtype) + 0.5) * bh / H
    xs = (torch.arange(W, device=x.device, dtype=x.dtype) + 0.5) * bw / W
    fy, fx = ys - torch.floor(ys), xs - torch.floor(xs)
    gw = 0.08
    gy = O.smoothstep(0, gw, fy) * O.smoothstep(1, 1 - gw, fy)
    gx = O.smoothstep(0, gw, fx) * O.smoothstep(1, 1 - gw, fx)
    m = (gy[:, None] * gx[None, :])[None, ..., None]
    bevel = (1 + 0.12 * (0.5 - fy)[:, None] + 0.06 * (0.5 - fx)[None, :])[None, ..., None]
    grout = O.col((0.22, 0.21, 0.2), x)
    y = (O.saturate(tiles, 1.15) * bevel) * m + grout * (1 - m)
    return O.finish(x, y, i)


def _voronoi(H, W, cell, rng, salt, device, dtype, jitter=0.9):
    """Jittered-grid voronoi: nearest/second-nearest distances and nearest seed coords (pixels)."""
    gh, gw = int(math.ceil(H / cell)) + 2, int(math.ceil(W / cell)) + 2
    jit = O.urand(rng, device, (2, gh, gw), salt)
    ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) / cell
    xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) / cell
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    iy, ix = torch.floor(Y).long(), torch.floor(X).long()
    d1 = torch.full((H, W), 1e9, device=device, dtype=dtype)
    d2 = d1.clone()
    sx = torch.zeros(H, W, device=device, dtype=dtype)
    sy = torch.zeros(H, W, device=device, dtype=dtype)
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            cy, cx = iy + oy, ix + ox
            jy = jit[0][(cy + 1).clamp(0, gh - 1), (cx + 1).clamp(0, gw - 1)]
            jx = jit[1][(cy + 1).clamp(0, gh - 1), (cx + 1).clamp(0, gw - 1)]
            py = cy.to(dtype) + 0.5 + (jy - 0.5) * jitter
            px = cx.to(dtype) + 0.5 + (jx - 0.5) * jitter
            d = torch.sqrt((Y - py) ** 2 + (X - px) ** 2)
            closer = d < d1
            d2 = torch.where(closer, d1, torch.minimum(d2, d))
            sx = torch.where(closer, px, sx)
            sy = torch.where(closer, py, sy)
            d1 = torch.where(closer, d, d1)
    return d1, d2, sx * cell, sy * cell


@look("stained-glass", "Stained Glass", "art", "Jewel-toned glass cells joined by dark lead came, glowing with light.",
      tags=["church", "glass", "jewel"], synonyms=["stained glass window", "voronoi", "cathedral glass"])
def stained_glass(x, t, i, rng, fps):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    s = O.S(x)
    cell = max(4.0, s / 22)
    d1, d2, sx, sy = _voronoi(H, W, cell, rng, "glass", x.device, x.dtype)
    src = O.blur(x, cell * 0.3)
    colr = _sample_px(src, sx, sy)
    colr = O.saturate(colr, 1.6).clamp(0, 1)
    colr = 0.08 + colr * 0.95
    edge = (d2 - d1) * cell  # pixels to the cell border
    lead_w = max(1.0, s * 0.006)
    lead = O.smoothstep(lead_w, lead_w + 1.2, edge)[None, ..., None]
    # light falloff toward the leading and glass texture
    glow = (0.78 + 0.22 * O.smoothstep(0, cell * 0.5, edge))[None, ..., None]
    tex = O.noise2(rng, 1, max(2, H // 6), max(2, W // 6), H, W, "glass-tex", device=x.device)[0]
    y = colr * glow * (0.92 + 0.16 * tex[None, ..., None])
    y = O.screen(y.clamp(0, 1), O.blur(y.clamp(0, 1) * lead, s * 0.02) * 0.25)
    y = y * lead + O.col((0.04, 0.035, 0.03), x) * (1 - lead)
    return O.finish(x, y, i)


@look("ascii", "ASCII Art", "art", "The video rebuilt from coloured text characters on a black terminal.",
      tags=["text", "terminal", "hacker", "code"], synonyms=["ascii art", "text art", "terminal", "matrix", "characters"])
def ascii_art(x, t, i, rng, fps):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    s = O.S(x)
    cw = max(3, int(round(s / 75)))
    sc = max(1, cw // 6)
    cw = 6 * sc if cw >= 6 else cw
    gw_, gh_ = 6 * sc, 9 * sc  # glyph cell incl. spacing
    cols, rows = max(1, W // gw_ + 1), max(1, H // gh_ + 1)
    small = F.adaptive_avg_pool2d(O.nchw(x), (rows, cols))
    l = (small * torch.tensor([0.2126, 0.7152, 0.0722], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)).sum(1)
    l = 0.12 + 0.88 * O.scurve(l.clamp(0, 1), 3.0)
    glyphs = font.ramp_glyphs(device=x.device, dtype=x.dtype)  # [N,7,5]
    N = glyphs.shape[0]
    cellg = torch.zeros(N, 9, 6, device=x.device, dtype=x.dtype)
    cellg[:, 1:8, 0:5] = glyphs
    cellg = cellg.repeat_interleave(sc, 1).repeat_interleave(sc, 2)  # [N, gh_, gw_]
    idx = (l * (N - 1) + 0.5).long().clamp(0, N - 1)  # [F,rows,cols]
    tilemap = cellg[idx]  # [F,rows,cols,gh_,gw_]
    m = tilemap.permute(0, 1, 3, 2, 4).reshape(Fn, rows * gh_, cols * gw_)[:, :H, :W]
    colr = F.interpolate(small, size=(rows * gh_, cols * gw_), mode="nearest")[:, :, :H, :W]
    colr = O.nhwc(colr)
    colr = O.saturate(colr, 1.3).clamp(0, 1)
    colr = O.saturate(colr, 1.3).clamp(0, 1)
    colr = colr / colr.max(-1, keepdim=True).values.clamp_min(0.08)
    colr = O.mix(colr, torch.ones_like(colr), 0.12)
    y = m[..., None] * colr + (1 - m[..., None]) * O.col((0.02, 0.025, 0.03), x)
    return O.finish(x, y, i)


@look("cel-shade", "Cel Shade", "art", "Anime/cartoon cel shading: flat light bands with clean dark outlines.",
      tags=["anime", "cartoon", "toon"], synonyms=["toon", "anime", "cartoon", "cel shading", "toon shader"])
def cel_shade(x, t, i, rng, fps):
    s = O.S(x)
    b = O.blur(x, s * 0.003)
    l = O.luma(b)
    chroma = b / l.clamp_min(0.03)
    bands = torch.tensor([0.22, 0.5, 0.78, 1.0], device=x.device, dtype=x.dtype)
    lq = torch.where(l < 0.25, bands[0], torch.where(l < 0.5, bands[1], torch.where(l < 0.78, bands[2], bands[3])))
    # soften band edges slightly
    lq = O.blur(lq, max(0.6, s * 0.001))
    y = O.saturate((chroma * lq).clamp(0, 1), 1.3)
    ink = O.xdog(l, max(0.8, s * 0.003), tau=0.98, phi=70)
    y = y * (0.15 + 0.85 * ink)
    return O.finish(x, y, i)


@look("pointillism", "Pointillism", "art", "Painted entirely in small dabs of pure colour, like Seurat.",
      tags=["painting", "impressionist", "dots"], synonyms=["seurat", "dot painting", "stippling", "impressionist"])
def pointillism(x, t, i, rng, fps):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    s = O.S(x)
    cell = max(3.0, s / 85)
    src = O.saturate(O.blur(x, cell * 0.3), 1.45).clamp(0, 1)
    paper = O.col((0.95, 0.93, 0.87), x)
    y = O.mix(paper, O.blur(src, cell * 1.5), 0.55).expand_as(x).clone()
    for layer in range(4):
        d1, d2, sx, sy = _voronoi(H, W, cell, rng, ("pt", layer), x.device, x.dtype, jitter=1.0)
        colr = _sample_px(src, sx, sy)
        j = O.urand(rng, x.device, (3,), "ptj", layer) - 0.5
        colr = (colr + j.view(1, 1, 1, 3) * 0.12).clamp(0, 1)
        r = 0.5 + 0.06 * layer
        m = O.smoothstep(r, r - 1.2 / cell, d1)[None, ..., None]
        y = y * (1 - m) + colr * m
    return O.finish(x, y, i)


@look("oil-paint", "Oil Paint", "art", "Painterly Kuwahara filter: smooth brushy patches, kept edges and a hint of impasto.",
      tags=["painting", "painterly", "artistic"], synonyms=["painting", "oil painting", "painterly", "kuwahara", "van gogh"])
def oil_paint(x, t, i, rng, fps):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    s = O.S(x)
    r = max(2, int(round(s / 65)))
    # work at reduced resolution for big frames
    f = max(1.0, s / 540)
    h2, w2 = int(round(H / f)), int(round(W / f))
    src = O.resize(x, h2, w2, "area") if f > 1 else x
    r = max(2, int(round(r / f)))
    xn = O.nchw(src)
    k = r + 1
    pad = F.pad(xn, (r, r, r, r), mode="replicate")
    mean = F.avg_pool2d(pad, k, stride=1)
    l = (pad * torch.tensor([0.2126, 0.7152, 0.0722], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)).sum(1, keepdim=True)
    lm = F.avg_pool2d(l, k, stride=1)
    lv = F.avg_pool2d(l * l, k, stride=1) - lm * lm
    # quadrant windows: top-left, top-right, bottom-left, bottom-right
    hh, ww = h2, w2
    offs = [(0, 0), (0, r), (r, 0), (r, r)]
    means = torch.stack([mean[:, :, oy:oy + hh, ox:ox + ww] for oy, ox in offs], 0)
    vars_ = torch.stack([lv[:, :, oy:oy + hh, ox:ox + ww] for oy, ox in offs], 0)
    # smooth weighting (generalised Kuwahara) instead of hard argmin
    w = 1.0 / (1e-4 + vars_.clamp_min(0) * 40) ** 2
    out = (means * w).sum(0) / w.sum(0)
    y = O.nhwc(out)
    y = O.saturate(y, 1.2)
    if f > 1:
        y = O.resize(y, H, W, "bicubic")
    # impasto: light from the top-left on the luma relief
    lum = O.luma(y)
    gx = lum - O.shift(lum, 1.5 * max(1, f), 0)
    gy = lum - O.shift(lum, 0, 1.5 * max(1, f))
    y = y * (1 + (gx + gy) * 2.0)
    return O.finish(x, y, i)
