"""A tiny built-in 5x7 bitmap font (no font files) and text rendering helpers."""

from __future__ import annotations

import torch

_G = {
    " ": ["00000"] * 7,
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "-": ["00000", "00000", "00000", "01110", "00000", "00000", "00000"],
    "=": ["00000", "00000", "11111", "00000", "11111", "00000", "00000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "*": ["00000", "10101", "01110", "11111", "01110", "10101", "00000"],
    "o": ["00000", "00000", "01110", "10001", "10001", "10001", "01110"],
    "#": ["01010", "01010", "11111", "01010", "11111", "01010", "01010"],
    "%": ["11001", "11010", "00010", "00100", "01000", "01011", "10011"],
    "@": ["01110", "10001", "10111", "10101", "10111", "10000", "01111"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11100", "10010", "10001", "10001", "10001", "10010", "11100"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
}

GLYPH_W, GLYPH_H = 5, 7


def glyph(ch, device=None, dtype=torch.float32):
    rows = _G.get(ch, _G.get(ch.upper(), _G[" "]))
    return torch.tensor([[1.0 if c == "1" else 0.0 for c in r] for r in rows], device=device, dtype=dtype)


def text_mask(text, scale, device=None, dtype=torch.float32, spacing=1):
    """Render `text` into a [7*scale, len*(5+spacing)*scale] 0/1 mask."""
    scale = max(1, int(scale))
    cols = []
    for ch in text:
        g = glyph(ch, device, dtype)
        cols.append(g)
        if spacing:
            cols.append(torch.zeros(GLYPH_H, spacing, device=device, dtype=dtype))
    m = torch.cat(cols, 1) if cols else torch.zeros(GLYPH_H, 1, device=device, dtype=dtype)
    return m.repeat_interleave(scale, 0).repeat_interleave(scale, 1)


def draw_mask(frame, mask, y0, x0, color, shadow=True, alpha=1.0):
    """Composite a 0..1 mask onto a single HWC frame (in place) at (y0, x0)."""
    H, W = frame.shape[0], frame.shape[1]
    h, w = mask.shape
    c = torch.tensor(color, device=frame.device, dtype=frame.dtype)
    layers = []
    if shadow:
        o = max(1, h // 14)
        layers.append((mask, y0 + o, x0 + o, torch.zeros_like(c), 0.6 * alpha))
    layers.append((mask, y0, x0, c, alpha))
    for m, yy, xx, cc, a in layers:
        ya, xa, yb, xb = max(0, yy), max(0, xx), min(H, yy + h), min(W, xx + w)
        if yb <= ya or xb <= xa:
            continue
        mm = m[ya - yy:yb - yy, xa - xx:xb - xx, None] * a
        frame[ya:yb, xa:xb] = frame[ya:yb, xa:xb] * (1 - mm) + cc * mm
    return frame


def ramp_glyphs(chars=" .:-=+*o#%@", device=None, dtype=torch.float32):
    """Glyph stack [N,7,5] sorted by ink density (light -> dense)."""
    gs = [glyph(c, device, dtype) for c in chars]
    gs.sort(key=lambda g: float(g.sum()))
    return torch.stack(gs, 0)
