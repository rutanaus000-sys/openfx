"""Render every OpenFX look on a procedural test scene into a labelled contact sheet.

Usage: python tests/render_contact_sheet.py [out.png] [--size HxW] [--only id1,id2]
Pillow is used only here (for labels and PNG writing), never in the node pack.
"""

import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfyui_openfx.nodes import still_to_clip  # noqa: E402
from comfyui_openfx.registry import CATEGORIES, LOOKS, apply_look, load_all  # noqa: E402


def _ss(e0, e1, x):
    t = ((x - e0) / (e1 - e0)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _noise(h, w, cell, seed):
    g = torch.Generator().manual_seed(seed)
    n = torch.rand(1, 1, max(2, int(h / cell)), max(2, int(w / cell)), generator=g)
    return F.interpolate(n, size=(h, w), mode="bicubic", align_corners=False)[0, 0]


def make_scene(H=480, W=320):
    """A sunset landscape with hills, a lake, a tree, string lights and a person in a red coat."""
    ys = (torch.arange(H) + 0.5) / H
    xs = (torch.arange(W) + 0.5) / W
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    asp = W / H
    img = torch.zeros(H, W, 3)

    def put(mask, color):
        nonlocal img
        c = torch.tensor(color)
        img = img * (1 - mask[..., None]) + c * mask[..., None]

    # sky gradient: deep blue -> violet -> warm horizon
    horizon = 0.56
    sky_t = (Y / horizon).clamp(0, 1)
    top, mid, low = torch.tensor([0.10, 0.20, 0.45]), torch.tensor([0.52, 0.42, 0.62]), torch.tensor([1.0, 0.68, 0.40])
    a = sky_t[..., None]
    sky = torch.where(a < 0.6, top + (mid - top) * (a / 0.6), mid + (low - mid) * ((a - 0.6) / 0.4))
    img = sky
    # clouds
    cl = _noise(H, W, 40, 1) * 0.6 + _noise(H, W, 14, 2) * 0.4
    cmask = _ss(0.55, 0.8, cl) * _ss(horizon, 0.1, Y) * 0.55
    put(cmask, (1.0, 0.82, 0.78))
    # sun with glow
    sx, sy = 0.68, horizon - 0.07
    d = torch.sqrt(((X - sx) * asp) ** 2 + (Y - sy) ** 2)
    img = img + torch.exp(-(d / 0.12) ** 2)[..., None] * torch.tensor([0.6, 0.35, 0.12])
    put(_ss(0.045, 0.04, d), (1.0, 0.97, 0.88))
    # far hills (hazy) and near hills
    far = horizon - 0.06 + 0.04 * torch.sin(X * 9 + 1) + 0.025 * torch.sin(X * 23)
    put(_ss(far - 0.002, far + 0.002, Y), (0.42, 0.35, 0.50))
    near = horizon + 0.02 * torch.sin(X * 5 + 2) - 0.01
    put(_ss(near - 0.002, near + 0.002, Y), (0.20, 0.24, 0.26))
    # lake reflecting the sky
    lake = (Y > horizon + 0.02) & (Y < horizon + 0.12)
    ry = (2 * (horizon + 0.02) - Y).clamp(0, 1)
    rimg = img[(ry * (H - 1)).long().clamp(0, H - 1), torch.arange(W)[None, :].expand(H, W)]
    ripple = 0.85 + 0.15 * torch.sin(Y * 400 + _noise(H, W, 8, 3) * 6)
    img = torch.where(lake[..., None], rimg * 0.75 * ripple[..., None] + torch.tensor([0.0, 0.03, 0.06]), img)
    # meadow with texture
    meadow = Y >= horizon + 0.12
    g = _noise(H, W, 6, 4) * 0.5 + _noise(H, W, 2, 5) * 0.5
    grass = torch.tensor([0.20, 0.42, 0.12]) * (0.7 + 0.6 * g[..., None]) + (Y[..., None] - 0.7) * torch.tensor([0.1, 0.05, 0.0])
    img = torch.where(meadow[..., None], grass, img)
    # yellow flowers
    fl = _ss(0.86, 0.9, _noise(H, W, 1.5, 6)) * meadow.float()
    put(fl, (1.0, 0.85, 0.2))
    # tree on the left
    trunk = (((X - 0.18) * asp).abs() < 0.018) & (Y > 0.45) & (Y < 0.72)
    put(trunk.float(), (0.25, 0.16, 0.10))
    for (cx, cy, r) in [(0.18, 0.40, 0.11), (0.10, 0.46, 0.08), (0.27, 0.46, 0.08), (0.18, 0.32, 0.08)]:
        dd = torch.sqrt(((X - cx) * asp) ** 2 + (Y - cy) ** 2) + (_noise(H, W, 5, 7) - 0.5) * 0.03
        leaf = torch.tensor([0.10, 0.30, 0.10]) * (0.7 + 0.7 * _noise(H, W, 3, 8)[..., None])
        m = _ss(r, r - 0.006, dd)
        img = img * (1 - m[..., None]) + leaf * m[..., None]
    # string lights (bright highlights for bloom/bokeh/starburst)
    for k in range(9):
        lx = 0.30 + k * 0.08
        ly = 0.20 + 0.04 * math.cos((lx - 0.62) * 6) + 0.02 * k / 9
        dd = torch.sqrt(((X - lx) * asp) ** 2 + (Y - ly) ** 2)
        col = [(1.0, 0.9, 0.6), (1.0, 0.5, 0.4), (0.6, 0.9, 1.0)][k % 3]
        put(_ss(0.011, 0.007, dd), col)
    wire = ((Y - (0.20 + 0.04 * torch.cos((X - 0.62) * 6) + 0.02 * (X - 0.3) / 0.72)).abs() < 0.002) & (X > 0.28)
    put(wire.float() * 0.7, (0.1, 0.1, 0.12))
    # person: red coat, skin face, dark hair, jeans
    px = 0.56
    head = torch.sqrt(((X - px) * asp) ** 2 + (Y - 0.585) ** 2)
    body_w = 0.055 + (Y - 0.62).clamp(0, 1) * 0.25
    coat = (((X - px) * asp).abs() < body_w) & (Y > 0.615) & (Y < 0.80)
    legs = (((X - px) * asp).abs() < 0.04) & (Y >= 0.80) & (Y < 0.93) & (((X - px) * asp).abs() > 0.006)
    put(legs.float(), (0.12, 0.18, 0.35))
    coatc = torch.tensor([0.80, 0.07, 0.08]) * (0.8 + 0.3 * (1 - (X - px).abs() * 8))[..., None]
    img = torch.where(coat[..., None], coatc.clamp(0, 1), img)
    put(_ss(0.04, 0.036, head), (0.86, 0.62, 0.48))
    hair = _ss(0.042, 0.038, head) * (Y < 0.575).float()
    put(hair, (0.14, 0.08, 0.05))
    # white house with lit window
    house = (X > 0.80) & (X < 0.97) & (Y > 0.50) & (Y < 0.60)
    put(house.float(), (0.88, 0.86, 0.80))
    roof = (Y > 0.44) & (Y <= 0.50) & ((X - 0.885).abs() < (Y - 0.44) / 0.06 * 0.1)
    put(roof.float(), (0.35, 0.12, 0.10))
    win = (X > 0.85) & (X < 0.90) & (Y > 0.53) & (Y < 0.57)
    put(win.float(), (1.0, 0.85, 0.45))
    # fine sensor texture
    g = torch.Generator().manual_seed(11)
    img = img + (torch.rand(H, W, 1, generator=g) - 0.5) * 0.02
    return img.clamp(0, 1)[None]


def _font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def to_pil(x):
    return Image.fromarray((x.clamp(0, 1) * 255 + 0.5).byte().cpu().numpy())


def sheet(entries, cols, tw, th, title):
    lab = 22
    rows = math.ceil(len(entries) / cols)
    pad = 6
    W = cols * (tw + pad) + pad
    H = rows * (th + lab + pad) + pad + 34
    im = Image.new("RGB", (W, H), (24, 24, 28))
    d = ImageDraw.Draw(im)
    d.text((pad, 6), title, fill=(240, 240, 240), font=_font(20))
    f = _font(14)
    for k, (name, frame) in enumerate(entries):
        r, c = divmod(k, cols)
        x0 = pad + c * (tw + pad)
        y0 = 34 + pad + r * (th + lab + pad)
        im.paste(to_pil(frame).resize((tw, th), Image.LANCZOS), (x0, y0))
        d.text((x0 + 2, y0 + th + 3), name, fill=(230, 230, 230), font=f)
    return im


def main():
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "contact_sheet.png"
    H, W = 480, 320
    only = None
    for a in sys.argv[1:]:
        if a.startswith("--size="):
            H, W = map(int, a.split("=")[1].split("x"))
        if a.startswith("--only="):
            only = a.split("=")[1].split(",")
    load_all()
    scene = make_scene(H, W)
    clip = still_to_clip(scene, seconds=1.0, fps=25, motion="slow push-in")
    mid = clip.shape[0] // 2
    entries = [("original", clip[mid])]
    by_cat = {c: [] for c in CATEGORIES}
    timing = []
    order = sorted(LOOKS.values(), key=lambda l: (CATEGORIES.index(l.category), l.id))
    for lk in order:
        if only and lk.id not in only:
            continue
        t0 = time.time()
        y = apply_look(clip, lk.id, 1.0, seed=42, fps=25)
        timing.append((time.time() - t0, lk.id))
        entries.append((lk.id, y[mid]))
        by_cat[lk.category].append((lk.id, y[mid]))
    base, ext = os.path.splitext(out)
    sheet(entries, 10, W // 2, H // 2, f"OpenFX looks ({len(entries) - 1}) - middle frame of a 25-frame push-in clip").save(out)
    for c, ents in by_cat.items():
        if ents:
            sheet([("original", clip[mid])] + ents, 5, W, H, f"category: {c}").save(f"{base}_{c}{ext}")
    timing.sort(reverse=True)
    print("slowest:", ", ".join(f"{n} {s:.2f}s" for s, n in timing[:8]))
    print("wrote", out)


if __name__ == "__main__":
    main()
