import math

import torch

from .. import ops as O
from ..registry import look


def _offset(x, d):
    """Frames delayed by d frames (clamped to the first)."""
    Fn = x.shape[0]
    idx = (torch.arange(Fn, device=x.device) - d).clamp(0, Fn - 1)
    return x[idx]


def _bg_blur(x):
    s = O.S(x)
    return O.blur(x, s * 0.04) * 0.45


def _shadow(h, w, y0, x0, rh, rw, radius, soft, device, dtype):
    m = O.rounded_rect_mask(h, w, y0 + soft * 0.3, x0, rh, rw, radius, device, dtype, soft=soft)
    return m


@look("letterbox", "Letterbox", "layout", "Cinemascope 2.39:1 black bars for an instant movie feel.",
      tags=["cinematic", "widescreen", "bars"], synonyms=["cinemascope", "black bars", "widescreen", "movie bars", "cinematic bars"])
def letterbox(x, t, i, rng, fps):
    H, W = x.shape[1], x.shape[2]
    target = W / 2.39
    bar = max(0.0, (H - target) / 2) * i
    ys = torch.arange(H, device=x.device, dtype=x.dtype) + 0.5
    m = (O.smoothstep(bar - 0.5, bar + 0.5, ys) * O.smoothstep(H - bar + 0.5, H - bar - 0.5, ys)).view(1, H, 1, 1)
    return (x * m).clamp(0, 1)


@look("polaroid", "Polaroid", "layout", "An instant photo: cream frame with the thick bottom border and faded, warm colours.",
      tags=["instant", "vintage", "frame"], synonyms=["instant photo", "instax", "photo frame", "polaroid frame"])
def polaroid(x, t, i, rng, fps):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    s = O.S(x)
    side = int(round(s * 0.06 * i))
    top = side
    bottom = int(round(s * 0.22 * i))
    ih, iw = H - top - bottom, W - 2 * side
    if ih < 4 or iw < 4 or i <= 1e-3:
        return x
    img = O.cover(x, ih, iw)
    img = O.mix(img, 0.06 + O.saturate(img, 0.85) * 0.86 * O.col((1.03, 1.0, 0.92), x), i)
    # slight inner shadow at the photo edge
    m = O.rounded_rect_mask(ih, iw, 0, 0, ih, iw, 1, x.device, x.dtype, soft=max(1.0, s * 0.006))
    img = img * (0.85 + 0.15 * m[None, ..., None])
    paper = O.col((0.95, 0.94, 0.90), x)
    out = paper.expand(Fn, H, W, 3).clone()
    # faint paper texture
    tex = O.noise2(rng, 1, max(2, H // 3), max(2, W // 3), H, W, "paper", device=x.device)[0]
    out = out * (0.97 + 0.03 * tex[None, ..., None])
    out[:, top:top + ih, side:side + iw] = img
    return out.clamp(0, 1)


@look("filmstrip", "Filmstrip", "layout", "The clip runs on a strip of film, sprocket holes sliding past.",
      tags=["film", "frame", "vintage"], synonyms=["film strip", "sprockets", "negative strip", "35mm frame"], animated=True)
def filmstrip(x, t, i, rng, fps):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    if i <= 1e-3:
        return x
    vertical = H > W  # holes along the long edges
    y = x if vertical else x.transpose(1, 2)
    h, w = y.shape[1], y.shape[2]
    s = min(h, w)
    band = int(round(w * 0.11 * i))
    iw = w - 2 * band
    gap = max(2, int(round(s * 0.02)))
    base = O.col((0.07, 0.05, 0.04), x)
    out = base.expand(Fn, h, w, 3).clone()
    img = O.cover(y, h, iw)
    # frame lines (the strip moves, so the gap between frames scrolls)
    speed = h * 0.25
    ys = torch.arange(h, device=x.device, dtype=x.dtype)
    off = (t * speed).view(-1, 1)
    period = h * 1.0
    lines = (torch.remainder(ys[None] + off, period) < gap).to(x.dtype)[..., None, None]
    img = img * (1 - lines) + base * lines
    out[:, :, band:band + iw] = img
    # sprocket holes
    hp = band * 0.95  # hole pitch
    hh, hw = hp * 0.55, band * 0.42
    xs = torch.arange(w, device=x.device, dtype=x.dtype) + 0.5
    yy = torch.remainder(ys[None] + 0.5 + off, hp)  # [F,h]
    qy = (yy - hp / 2).abs() - (hh / 2 - hh * 0.2)
    holes = torch.zeros(Fn, h, w, device=x.device, dtype=x.dtype)
    for cx in (band / 2, w - band / 2):
        qx = (xs - cx).abs() - (hw / 2 - hh * 0.2)
        qxx, qyy = qx[None, None, :], qy[:, :, None]
        d = torch.sqrt(qxx.clamp_min(0) ** 2 + qyy.clamp_min(0) ** 2) + torch.minimum(torch.maximum(qxx, qyy), torch.zeros(1, device=x.device)) - hh * 0.2
        holes = torch.maximum(holes, (0.5 - d).clamp(0, 1))
    out = out * (1 - holes[..., None]) + holes[..., None] * O.col((0.93, 0.92, 0.88), x)
    if not vertical:
        out = out.transpose(1, 2)
    return out.contiguous().clamp(0, 1)


@look("rounded-frame", "Rounded Frame", "layout", "The clip floats as a rounded card with a soft shadow over a blurred backdrop.",
      tags=["card", "modern", "frame"], synonyms=["rounded corners", "card", "floating frame", "frame"])
def rounded_frame(x, t, i, rng, fps):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    s = O.S(x)
    if i <= 1e-3:
        return x
    mg = s * 0.06 * i
    rh, rw = H - 2 * mg, W - 2 * mg
    rad = s * 0.06
    bg = O.blur(x, s * 0.05) * 0.5
    card = O.cover(x, int(round(rh)), int(round(rw)))
    full = bg.clone()
    y0, x0 = int(round(mg)), int(round(mg))
    full = O.paste(full, card, y0, x0)
    m = O.rounded_rect_mask(H, W, y0, x0, card.shape[1], card.shape[2], rad, x.device, x.dtype)[None, ..., None]
    sh = O.rounded_rect_mask(H, W, y0 + s * 0.015, x0, card.shape[1], card.shape[2], rad, x.device, x.dtype, soft=s * 0.04)[None, ..., None]
    bg = bg * (1 - 0.5 * sh)
    out = bg * (1 - m) + full * m
    return out.clamp(0, 1)


@look("split-screen", "Split Screen", "layout", "Three vertical panels of the same clip, the middle one running half a second behind.",
      tags=["panels", "triptych", "music-video"], synonyms=["triptych", "three panels", "split", "multi screen"], animated=True, temporal=True)
def split_screen(x, t, i, rng, fps):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    if i <= 1e-3:
        return x
    pw = W // 3
    widths = [pw, W - 2 * pw - 0, pw]
    widths[1] = W - 2 * pw
    d = int(round(fps * 0.5))
    out = torch.empty_like(x)
    xo = 0
    gap = max(1, int(round(O.S(x) * 0.006)))
    for k, w in enumerate(widths):
        src = _offset(x, d) if k == 1 else x
        panel = O.cover(src, H, w)
        if k == 1:
            panel = panel.flip(2)
        out[:, :, xo:xo + w] = panel
        xo += w
    for xb in (pw, W - pw):
        out[:, :, max(0, xb - gap // 2 - 1):xb + gap // 2 + 1] = 0.02
    return out.clamp(0, 1)


@look("grid-montage", "Grid Montage", "layout", "A 2x2 grid of the clip, each panel a beat later than the last.",
      tags=["grid", "montage", "panels"], synonyms=["2x2", "quad", "grid", "four up", "brady bunch"], animated=True, temporal=True)
def grid_montage(x, t, i, rng, fps):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    if i <= 1e-3:
        return x
    g = max(1, int(round(O.S(x) * 0.01)))
    h2, w2 = (H - g) // 2, (W - g) // 2
    out = torch.full_like(x, 0.03)
    d = max(1, int(round(fps * 0.25)))
    tints = [(1.0, 1.0, 1.0), (1.04, 0.98, 0.92), (0.94, 0.98, 1.05), (1.02, 0.95, 1.02)]
    for k, (yy, xx) in enumerate([(0, 0), (0, W - w2), (H - h2, 0), (H - h2, W - w2)]):
        panel = O.cover(_offset(x, d * k), h2, w2) * O.col(tints[k], x)
        out[:, yy:yy + h2, xx:xx + w2] = panel
    return out.clamp(0, 1)


@look("picture-in-picture", "Picture in Picture", "layout", "A framed inset in the corner shows a zoomed-in detail, a moment behind.",
      tags=["inset", "reaction", "pip"], synonyms=["pip", "inset", "reaction cam", "zoom inset"], animated=True, temporal=True)
def picture_in_picture(x, t, i, rng, fps):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    s = O.S(x)
    if i <= 1e-3:
        return x
    ih, iw = int(round(H * 0.32)), int(round(W * 0.32))
    src = _offset(x, int(round(fps * 0.4)))
    cy, cx = H // 2, W // 2
    ch, cw = max(2, int(H * 0.45)), max(2, int(W * 0.45))
    crop = src[:, cy - ch // 2:cy - ch // 2 + ch, cx - cw // 2:cx - cw // 2 + cw]
    inset = O.cover(crop, ih, iw)
    m = int(round(s * 0.04))
    b = max(1, int(round(s * 0.008)))
    y0, x0 = m, W - m - iw
    out = x.clone()
    sh = O.rounded_rect_mask(H, W, y0 + s * 0.012, x0 - s * 0.004, ih, iw, s * 0.02, x.device, x.dtype, soft=s * 0.03)[None, ..., None]
    out = out * (1 - 0.55 * sh)
    out[:, y0 - b:y0 + ih + b, x0 - b:x0 + iw + b] = 0.97
    out[:, y0:y0 + ih, x0:x0 + iw] = inset
    return out.clamp(0, 1)


def _droste(x, levels=6, ratio=0.62, border=None):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    s = O.S(x)
    b = max(1, int(round(s * 0.006))) if border is None else border
    out = x
    for _ in range(levels):
        h, w = max(2, int(round(H * ratio))), max(2, int(round(W * ratio)))
        small = O.resize(out, h, w, "area")
        nxt = x.clone()
        y0, x0 = (H - h) // 2, (W - w) // 2
        # soft shadow + white border
        sh = O.rounded_rect_mask(H, W, y0 + s * 0.01, x0, h, w, 1, x.device, x.dtype, soft=s * 0.03)[None, ..., None]
        nxt = nxt * (1 - 0.5 * sh)
        nxt[:, max(0, y0 - b):y0 + h + b, max(0, x0 - b):x0 + w + b] = 0.96
        nxt[:, y0:y0 + h, x0:x0 + w] = small
        out = nxt
    return out


@look("droste", "Droste", "layout", "The picture contains itself, over and over, in nested frames.",
      tags=["recursive", "infinite", "mise-en-abyme"], synonyms=["recursive", "infinite frame", "picture in picture loop", "mise en abyme"])
def droste(x, t, i, rng, fps):
    if i <= 1e-3:
        return x
    return _droste(x, 6, 0.62).clamp(0, 1)


@look("recursive-zoom", "Recursive Zoom", "layout", "An endless zoom into nested copies of the frame that loops seamlessly.",
      tags=["infinite", "zoom", "loop", "trippy"], synonyms=["infinite zoom", "endless zoom", "zoom loop", "droste zoom"], animated=True)
def recursive_zoom(x, t, i, rng, fps):
    if i <= 1e-3:
        return x
    ratio = 0.5
    d = _droste(x, 7, ratio)
    phase = torch.remainder(t * 0.45, 1.0)
    z = torch.pow(torch.tensor(ratio, device=x.device), phase).view(-1, 1, 1)  # 1 -> ratio
    uu, vv, _, _ = O.grid(x)
    return O.warp(d, uu[None] * z, vv[None] * z, "border").clamp(0, 1)
