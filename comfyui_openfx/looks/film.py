import math

import torch
import torch.nn.functional as F

from .. import ops as O
from ..registry import look

# ---------------------------------------------------------------- building blocks


def halation(x, strength=0.35, radius=0.02, thresh=0.62):
    l = O.luma(x)
    hl = O.smoothstep(thresh, 1.0, l) * x.clamp(0, 1)
    glow = O.blur(hl, O.S(x) * radius)
    tint = O.col((1.0, 0.33, 0.12), x)
    return O.screen(x, (glow * tint * strength * 2.2).clamp(0, 1))


def gate_weave(x, t, rng, fps, amount, salt="weave"):
    """Smooth, slightly jittery frame wander (as from a worn camera gate)."""
    s = O.S(x)
    jx = (O.fbm1(t, 2.5, rng, salt, "x") - 0.5) * 2 + (O.frame_rand(rng, t, fps, (), salt, "jx", device=x.device) - 0.5) * 0.5
    jy = (O.fbm1(t, 3.1, rng, salt, "y") - 0.5) * 2 + (O.frame_rand(rng, t, fps, (), salt, "jy", device=x.device) - 0.5) * 0.6
    dx = jx * s * 0.006 * amount
    dy = jy * s * 0.009 * amount
    h, w = x.shape[1], x.shape[2]
    uu, vv, _, _ = O.grid(x)
    zoom = 1.0 / (1.0 + 0.025 * amount)  # slight overscan hides edges
    gu = uu[None] * zoom - (dx * 2 / w).view(-1, 1, 1)
    gv = vv[None] * zoom - (dy * 2 / h).view(-1, 1, 1)
    return O.warp(x, gu, gv, "border")


def flicker_gain(t, rng, fps, amount, salt="flicker"):
    """Per-frame exposure multiplier around 1."""
    fr = O.frame_rand(rng, t, fps, (), salt, device=t.device) - 0.5
    slow = O.fbm1(t, 4.0, rng, salt, "slow") - 0.5
    return 1 + amount * (fr * 1.2 + slow * 0.9)


def flicker(x, t, rng, fps, amount, salt="flicker"):
    g = flicker_gain(t, rng, fps, amount, salt).view(-1, 1, 1, 1)
    # uneven exposure: a soft gradient that moves frame to frame
    uu, vv, _, _ = O.grid(x)
    ang = O.frame_rand(rng, t, fps, (), salt, "ang", device=x.device) * 2 * math.pi
    grad = (uu[None] * torch.cos(ang).view(-1, 1, 1) + vv[None] * torch.sin(ang).view(-1, 1, 1))[..., None]
    return x * g * (1 + grad * amount * 0.35)


def scratches(x, t, rng, fps, n=3, strength=1.0, salt="scratch", dark_bias=0.4):
    """Vertical scratches that persist a few frames and wander slightly."""
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    s = O.S(x)
    idx = O.frame_index(t, fps)
    dev = x.device
    xs = torch.arange(W, device=dev, dtype=x.dtype)
    ys = torch.arange(H, device=dev, dtype=x.dtype) / H
    out = x
    for k in range(n):
        # each scratch lives for a 'segment' of frames
        seg = [i // 6 for i in idx]
        p = O.key_rand(rng, seg, (6,), salt, k, device=dev)  # x, width, alive, polarity, phase, length
        fr = O.frame_rand(rng, t, fps, (2,), salt, k, "j", device=dev)
        alive = (p[:, 2] < 0.65).to(x.dtype)
        px = p[:, 0] * W + (fr[:, 0] - 0.5) * s * 0.004 + (torch.tensor(idx, device=dev, dtype=x.dtype) % 6) * (p[:, 4] - 0.5) * s * 0.002
        wd = (0.4 + p[:, 1] * 1.6) * max(1.0, s / 500)
        prof = torch.exp(-((xs[None] - px[:, None]) / wd[:, None]) ** 2)  # [F,W]
        # along-y modulation: breaks + varying opacity
        ph = p[:, 4:5] * 50
        along = 0.55 + 0.45 * torch.sin(ys[None] * (8 + p[:, 5:6] * 20) + ph)
        along = along * O.smoothstep(0.2, 0.5, 0.5 + 0.5 * torch.sin(ys[None] * 3.0 + ph * 1.3))
        m = (prof[:, None, :] * along[:, :, None] * alive.view(-1, 1, 1))[..., None] * strength
        dark = (p[:, 3] < dark_bias).to(x.dtype).view(-1, 1, 1, 1)
        out = out * (1 - m * 0.8 * dark) + m * (1 - dark) * 0.85 * (1 - out) + m * (1 - dark) * 0.0
    return out


def dust(x, t, rng, fps, amount=1.0, salt="dust", hair=True):
    """Dark specks (dust on the print) that change every frame, plus occasional hairs."""
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    s = O.S(x)
    dev = x.device
    cell = max(2, int(s / 90))
    gh, gw = max(2, H // cell), max(2, W // cell)
    r = O.frame_rand(rng, t, fps, (1, gh, gw), salt, device=dev)
    spec = O.smoothstep(1 - 0.0012 * amount, 1.0, r) * 3
    spec = F.interpolate(spec, size=(H, W), mode="bilinear", align_corners=False)
    shape = O.frame_rand(rng, t, fps, (1, max(2, H // 3), max(2, W // 3)), salt, "shape", device=dev)
    shape = F.interpolate(shape, size=(H, W), mode="bilinear", align_corners=False)
    spec = (spec * (0.6 + 0.8 * shape)).clamp(0, 1)
    mask = spec[:, 0]
    # a few specks are light (dirt on the negative prints white)
    pol = O.frame_rand(rng, t, fps, (1, gh, gw), salt, "pol", device=dev)
    pol = F.interpolate((pol < 0.3).to(x.dtype), size=(H, W), mode="nearest")[:, 0]
    light = mask * pol
    mask = mask * (1 - pol)
    if hair:
        idx = O.frame_index(t, fps)
        hm = torch.zeros(Fn, H * W, device=dev, dtype=x.dtype)
        for fi, k in enumerate(idx):
            g = O.gen(rng, "cpu", salt, "hair", k // 4)
            q = torch.rand(12, generator=g)
            if q[0] > 0.6:
                continue
            n = 1200
            tt = torch.linspace(0, 1, n)
            cx, cy = q[1] * W, q[2] * H
            L = (0.05 + 0.12 * q[3]) * s
            a0 = q[4] * 2 * math.pi
            curl = (q[5] - 0.5) * 8
            ang = a0 + curl * tt + 0.6 * torch.sin(tt * (4 + 6 * q[6]) + q[7] * 6)
            dx = torch.cumsum(torch.cos(ang), 0) * L / n
            dy = torch.cumsum(torch.sin(ang), 0) * L / n
            px = (cx + dx).round().long().clamp(0, W - 1)
            py = (cy + dy).round().long().clamp(0, H - 1)
            hm[fi, (py * W + px).to(dev)] = 1.0
        hm = hm.view(Fn, H, W)
        thick = max(0.5, s / 900)
        hm = (O.blur_map(hm, thick) * (2.2 / max(thick, 0.7))).clamp(0, 1)
        mask = torch.maximum(mask, hm * 0.9)
    y = x * (1 - 0.85 * mask[..., None])
    return y + (1 - y) * 0.7 * light[..., None]


def film_grade(x, warm=0.04, fade=0.03, sat=0.92, curve=4.5):
    y = O.scurve(x.clamp(0, 1), curve)
    y = O.mix(x, y, 0.7)
    l = O.luma(y)
    y = y + O.smoothstep(0.5, 1.0, l) * O.col((warm, warm * 0.4, -warm), x)
    y = y + (1 - O.smoothstep(0.0, 0.4, l)) * O.col((-0.01, 0.01, 0.025), x)
    y = O.saturate(y, sat)
    return fade + y * (1 - fade * 1.6)


# ---------------------------------------------------------------- looks


@look("film-35mm", "35mm Film", "film", "Cinematic 35mm print: gentle S-curve, warm highlights, fine moving grain and a soft red halation glow.",
      tags=["cinematic", "analog", "grain"], synonyms=["35mm", "film look", "kodak", "cinematic film", "analog"], animated=True)
def film_35mm(x, t, i, rng, fps):
    s = O.S(x)
    y = film_grade(x, 0.045, 0.025, 0.93)
    y = halation(y, 0.28, 0.018)
    g = O.grain(rng, t, fps, x.shape[1], x.shape[2], max(1.0, s / 650), "g35", device=x.device)
    y = O.apply_grain(y, g, 0.032)
    y = y * O.vignette_mask(x, 0.18, 0.5, 1.4)[None, ..., None]
    return O.finish(x, y, i)


@look("film-16mm", "16mm Film", "film", "Indie 16mm: chunky grain, softer detail, faded warmth, gentle weave and flicker.",
      tags=["indie", "analog", "grain", "documentary"], synonyms=["16mm", "bolex", "indie film"], animated=True)
def film_16mm(x, t, i, rng, fps):
    s = O.S(x)
    y = gate_weave(x, t, rng, fps, 0.5 * i, "w16")
    y = O.blur(y, s * 0.0016)
    y = film_grade(y, 0.05, 0.05, 0.88, 4.0)
    y = halation(y, 0.25, 0.02)
    y = flicker(y, t, rng, fps, 0.04, "f16")
    g = O.grain(rng, t, fps, x.shape[1], x.shape[2], max(1.3, s / 380), "g16", device=x.device)
    y = O.apply_grain(y, g, 0.06)
    y = y * O.vignette_mask(x, 0.32, 0.4, 1.3)[None, ..., None]
    return O.finish(x, y, i)


@look("super-8", "Super 8", "film", "Home-movie Super 8: heavy grain, flicker, gate weave, warm faded colour, soft focus and dark corners.",
      tags=["home-movie", "vintage", "nostalgic", "70s"], synonyms=["super8", "8mm", "home movie", "vintage film", "70s"], animated=True)
def super_8(x, t, i, rng, fps):
    s = O.S(x)
    y = gate_weave(x, t, rng, fps, 1.2 * i, "w8")
    y = O.blur(y, s * 0.0035)
    y = O.scurve(y, 3.5)
    y = O.saturate(y, 0.9)
    y = y * O.col((1.08, 0.98, 0.78), x) + O.col((0.07, 0.04, 0.02), x)
    y = halation(y.clamp(0, 1), 0.35, 0.025, 0.55)
    y = flicker(y, t, rng, fps, 0.08, "f8")
    g = O.grain(rng, t, fps, x.shape[1], x.shape[2], max(1.6, s / 260), "g8", device=x.device, mono=False)
    y = O.apply_grain(y, g, 0.07)
    y = y * O.vignette_mask(x, 0.65, 0.25, 1.15, 0.2)[None, ..., None]
    return O.finish(x, y, i)


@look("silent-film", "Silent Film", "film", "1920s silent picture: flickering black and white, scratches, dust, weave and heavy vignette.",
      tags=["1920s", "black-and-white", "vintage", "chaplin"], synonyms=["old movie", "1920s", "chaplin", "nickelodeon", "silent movie"], animated=True)
def silent_film(x, t, i, rng, fps):
    s = O.S(x)
    y = gate_weave(x, t, rng, fps, 1.4 * i, "wsil")
    y = O.blur(y, s * 0.002)
    l = (y * O.col((0.3, 0.6, 0.1), x)).sum(-1, keepdim=True)
    l = O.scurve(O.levels(l, 0.0, 0.85), 5.5) * 0.94 + 0.04
    l = l * O.col((1.0, 0.98, 0.93), x)
    l = flicker(l, t, rng, fps, 0.14, "fsil")
    g = O.grain(rng, t, fps, x.shape[1], x.shape[2], max(1.4, s / 320), "gsil", device=x.device)
    l = O.apply_grain(l, g, 0.07)
    l = scratches(l, t, rng, fps, 4, 1.0, "ssil", 0.35)
    l = dust(l, t, rng, fps, 1.4, "dsil")
    l = l * O.vignette_mask(x, 0.6, 0.3, 1.25, 0.3)[None, ..., None]
    return O.finish(x, l, i)


@look("film-grain", "Film Grain", "film", "Organic, moving film grain that is strongest in the mid-tones.",
      tags=["grain", "texture", "analog"], synonyms=["grain", "noise", "grainy"], animated=True)
def film_grain(x, t, i, rng, fps):
    s = O.S(x)
    g = O.grain(rng, t, fps, x.shape[1], x.shape[2], max(1.2, s / 450), "grain", device=x.device)
    gc = O.grain(rng, t, fps, x.shape[1], x.shape[2], max(1.5, s / 300), "grainc", device=x.device, mono=False)
    y = O.apply_grain(x, g * 0.85 + gc * 0.35, 0.075)
    return O.finish(x, y, i)


@look("film-scratches", "Film Scratches", "film", "Worn-print vertical scratches that come and go.",
      tags=["damage", "vintage", "worn"], synonyms=["scratches", "scratched film", "damaged film"], animated=True)
def film_scratches(x, t, i, rng, fps):
    y = scratches(x, t, rng, fps, 5, 1.0, "fscr", 0.4)
    return O.finish(x, y, i)


@look("dust-and-hair", "Dust & Hair", "film", "Flecks of dust and the odd stray hair dancing on the projected print.",
      tags=["damage", "vintage", "projector"], synonyms=["dust", "specks", "hair in gate", "dirty film"], animated=True)
def dust_and_hair(x, t, i, rng, fps):
    y = dust(x, t, rng, fps, 2.0, "dh")
    return O.finish(x, y, i)


@look("gate-weave", "Gate Weave", "film", "The frame wanders slightly, like film moving through a worn projector gate.",
      tags=["wobble", "projector", "analog"], synonyms=["weave", "jitter", "film wobble", "projector shake"], animated=True)
def gate_weave_look(x, t, i, rng, fps):
    return gate_weave(x, t, rng, fps, 1.5 * i, "gw").clamp(0, 1)


@look("exposure-flicker", "Exposure Flicker", "film", "Uneven, flickering exposure from an old hand-cranked camera.",
      tags=["flicker", "vintage"], synonyms=["flicker", "flickering", "strobing light"], animated=True)
def exposure_flicker(x, t, i, rng, fps):
    y = flicker(x, t, rng, fps, 0.16, "ef")
    return O.finish(x, y, i)


@look("halation", "Halation", "film", "Red-orange glow bleeding around bright highlights, as on cinema film stock.",
      tags=["glow", "film", "highlights"], synonyms=["red glow", "cinestill", "film glow"])
def halation_look(x, t, i, rng, fps):
    y = halation(x, 0.6, 0.022, 0.55)
    return O.finish(x, y, i)


_LEAK_COLS = [(1.0, 0.45, 0.10), (1.0, 0.20, 0.08), (1.0, 0.75, 0.25), (0.95, 0.25, 0.45), (1.0, 0.55, 0.30)]


@look("light-leaks", "Light Leaks", "film", "Warm, soft washes of light drifting across the frame like a leaky camera.",
      tags=["warm", "dreamy", "vintage", "glow"], synonyms=["leak", "light leak", "film leak", "lens leak"], animated=True)
def light_leaks(x, t, i, rng, fps):
    H, W = x.shape[1], x.shape[2]
    uu, vv, ax, ay = O.grid(x)
    U, V = uu * ax, vv * ay
    acc = torch.zeros(x.shape[0], H, W, 3, device=x.device, dtype=x.dtype)
    p = O.urand(rng, x.device, (4, 6), "leaks")
    for k in range(4):
        side = int(p[k, 0] * 4)
        e = float(p[k, 1]) * 2 - 1
        bx, by = [(-ax * 1.1, e * ay), (ax * 1.1, e * ay), (e * ax, -ay * 1.1), (e * ax, ay * 1.1)][side]
        mx = (O.fbm1(t, 0.25, rng, "lk", k, "x") - 0.5) * 1.4
        my = (O.fbm1(t, 0.25, rng, "lk", k, "y") - 0.5) * 1.4
        cx, cy = bx + mx.view(-1, 1, 1), by + my.view(-1, 1, 1)
        rad = 0.55 + 0.6 * float(p[k, 2])
        stretch = 0.5 + float(p[k, 3])
        d2 = ((U[None] - cx) / stretch) ** 2 + ((V[None] - cy) * stretch) ** 2
        b = torch.exp(-d2 / (rad * rad))
        pulse = 0.35 + 0.65 * O.smoothstep(0.25, 0.75, O.fbm1(t, 0.4, rng, "lk", k, "p")).view(-1, 1, 1)
        c = O.col(_LEAK_COLS[int(p[k, 4] * len(_LEAK_COLS)) % len(_LEAK_COLS)], x)
        acc = acc + (b * pulse)[..., None] * c
    leak = (1 - torch.exp(-acc * 1.3))
    y = O.screen(x, leak * 0.95)
    return O.finish(x, y, i)


@look("film-burn", "Film Burn", "film", "The film melts and burns: a molten orange-white front creeps in from the edge and flickers.",
      tags=["burn", "transition", "fire", "vintage"], synonyms=["burn", "burning film", "film melt", "burnout"], animated=True)
def film_burn(x, t, i, rng, fps):
    H, W = x.shape[1], x.shape[2]
    uu, vv, ax, ay = O.grid(x)
    p = O.urand(rng, x.device, (3,), "burn")
    ang = float(p[0]) * 2 * math.pi
    ex, ey = math.cos(ang) * ax, math.sin(ang) * ay
    # distance from the burning edge (0 = edge it starts from)
    d = ((uu * ax) * math.cos(ang) + (vv * ay) * math.sin(ang)) / math.sqrt(ax * ax + ay * ay)
    d = (1 - (d + 1) / 2)[None]
    n = O.fractal_evo(rng, t, fps, 1.2, 3, H, W, "burnn", device=x.device, octaves=4, aspect=W / H)
    prog = 0.10 + 0.22 * O.fbm1(t, 0.35, rng, "burnp").view(-1, 1, 1) + 0.03 * torch.sin(t * 7.0).view(-1, 1, 1)
    f = d + (n - 0.5) * 0.55
    heat = O.smoothstep(prog + 0.18, prog - 0.12, f)  # 1 = burnt through
    core = O.smoothstep(prog + 0.02, prog - 0.10, f)
    rim = (heat - core).clamp(0, 1)
    burn = O.gradient_map(heat[..., None], [(0.0, (0.0, 0.0, 0.0)), (0.35, (0.55, 0.06, 0.0)), (0.6, (1.0, 0.42, 0.05)),
                                             (0.85, (1.0, 0.85, 0.45)), (1.0, (1.0, 0.98, 0.9))], ref=x)
    y = O.screen(x, burn)
    y = y * (1 - 0.35 * rim[..., None] * (1 - core[..., None]))
    y = O.mix(y, O.col((1.0, 0.95, 0.82), x).expand_as(y), core[..., None] * 0.7)
    return O.finish(x, y, i)
