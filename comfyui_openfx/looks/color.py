import math

import torch

from .. import ops as O
from ..registry import look


def _gray(x, weights=(0.30, 0.59, 0.11)):
    return (x * O.col(weights, x)).sum(-1, keepdim=True)


def _norm(l, lo=0.02, hi=0.98):
    """Stretch luminance so maps use the full range."""
    a = torch.quantile(l.flatten(1)[:, :: max(1, l[0].numel() // 20000)], lo, dim=1).view(-1, 1, 1, 1)
    b = torch.quantile(l.flatten(1)[:, :: max(1, l[0].numel() // 20000)], hi, dim=1).view(-1, 1, 1, 1)
    a = a.clamp(max=0.2)
    b = b.clamp(min=0.8)
    return ((l - a) / (b - a).clamp_min(0.2)).clamp(0, 1)


@look("black-and-white", "Black & White", "color", "Rich monochrome with punchy contrast, like a red-filtered B&W film stock.",
      tags=["monochrome", "classic", "noir"], synonyms=["bw", "b&w", "grayscale", "greyscale", "monochrome", "noir"])
def black_and_white(x, t, i, rng, fps):
    l = _gray(x, (0.36, 0.54, 0.10))
    l = O.scurve(l, 5.0)
    return O.finish(x, l.expand_as(x), i)


@look("sepia", "Sepia", "color", "Warm brown antique toning, like an old photographic print.",
      tags=["vintage", "warm", "antique"], synonyms=["old photo", "brown", "antique"])
def sepia(x, t, i, rng, fps):
    l = O.scurve(_gray(x), 4.0)
    out = O.gradient_map(l, [(0.0, (0.08, 0.05, 0.03)), (0.45, (0.50, 0.35, 0.21)), (0.8, (0.85, 0.72, 0.52)), (1.0, (1.0, 0.95, 0.84))])
    return O.finish(x, out, i)


@look("duotone", "Duotone", "color", "Two-colour gradient map from deep navy-teal shadows to soft peach highlights.",
      tags=["graphic", "poster", "gradient-map"], synonyms=["two tone", "gradient map", "spotify"])
def duotone(x, t, i, rng, fps):
    l = O.smoothstep(0.0, 1.0, _norm(O.luma(x)))
    out = O.gradient_map(l, [(0.0, (0.04, 0.12, 0.22)), (0.5, (0.36, 0.45, 0.48)), (1.0, (1.0, 0.80, 0.66))])
    return O.finish(x, out, i)


@look("tritone", "Tritone", "color", "Three-colour map: plum shadows, rose mid-tones and golden highlights.",
      tags=["graphic", "gradient-map", "vibrant"], synonyms=["three tone", "gradient map"])
def tritone(x, t, i, rng, fps):
    l = _norm(O.luma(x))
    out = O.gradient_map(l, [(0.0, (0.10, 0.03, 0.18)), (0.5, (0.86, 0.30, 0.42)), (1.0, (1.0, 0.94, 0.62))])
    return O.finish(x, out, i)


@look("monochrome-tint", "Monochrome Tint", "color", "Single-hue cyanotype-style tint from deep blue to paper white.",
      tags=["tint", "blue", "cyanotype"], synonyms=["cyanotype", "blue tint", "tinted"])
def monochrome_tint(x, t, i, rng, fps):
    l = O.scurve(_gray(x), 4.0)
    out = O.gradient_map(l, [(0.0, (0.02, 0.08, 0.20)), (0.55, (0.18, 0.45, 0.70)), (1.0, (0.92, 0.97, 1.0))])
    return O.finish(x, out, i)


@look("split-toning", "Split Toning", "color", "Cool teal shadows and warm golden highlights, keeping the original colours.",
      tags=["grade", "cinematic"], synonyms=["split tone", "toning"])
def split_toning(x, t, i, rng, fps):
    l = O.luma(x)
    sh = (1 - O.smoothstep(0.0, 0.55, l))
    hi = O.smoothstep(0.45, 1.0, l)
    out = x + sh * (O.col((-0.06, 0.02, 0.09), x)) + hi * O.col((0.08, 0.03, -0.07), x)
    out = O.saturate(out, 1.05)
    return O.finish(x, out, i)


@look("bleach-bypass", "Bleach Bypass", "color", "Desaturated, high-contrast silver-retention look with gritty metallic tones.",
      tags=["gritty", "desaturated", "war-film"], synonyms=["silver retention", "skip bleach", "saving private ryan"])
def bleach_bypass(x, t, i, rng, fps):
    l = O.luma(x)
    ov = O.overlay(x, l.expand_as(x))
    out = O.mix(ov, l.expand_as(x), 0.45)
    out = O.scurve(out.clamp(0, 1), 5.5)
    out = O.mix(out, x, 0.15)
    return O.finish(x, out, i)


@look("cross-processing", "Cross Processing", "color", "Slide film in negative chemistry: punchy contrast, yellow-green highlights and cyan shadows.",
      tags=["lomo", "vintage", "vibrant"], synonyms=["xpro", "cross process", "lomo"])
def cross_processing(x, t, i, rng, fps):
    r, g, b = x.unbind(-1)
    r = O.scurve(r, 7.0)
    g = O.scurve(g, 5.0) * 1.03
    b = 0.16 + 0.66 * b
    out = torch.stack([r, g, b], -1)
    out = O.saturate(out.clamp(0, 1), 1.2)
    return O.finish(x, out, i)


@look("teal-orange", "Teal & Orange", "color", "Blockbuster grade: warm skin and highlights against teal shadows.",
      tags=["cinematic", "blockbuster", "grade"], synonyms=["orange and teal", "hollywood", "movie look"])
def teal_orange(x, t, i, rng, fps):
    l = O.luma(x)
    hsv = O.rgb_to_hsv(x.clamp(0, 1))
    h = hsv[..., :1]
    # how 'warm' each pixel already is (reds-oranges-yellows)
    dist = torch.minimum((h - 0.08).abs(), 1 - (h - 0.08).abs())
    warm = (1 - O.smoothstep(0.06, 0.2, dist)) * O.smoothstep(0.05, 0.25, hsv[..., 1:2])
    w = torch.maximum(warm, O.smoothstep(0.55, 0.95, l) * 0.7)
    teal = O.tint_luma(x, (0.0, 0.55, 0.62), 0.55)
    orange = O.tint_luma(x, (1.0, 0.55, 0.25), 0.5)
    out = O.mix(teal, orange, w)
    out = O.scurve(out.clamp(0, 1), 5.0)
    out = O.mix(x, out, 0.85)
    return O.finish(x, O.saturate(out, 1.1), i)


@look("day-for-night", "Day for Night", "color", "Turns daylight into moonlit night: dark, desaturated and blue.",
      tags=["night", "blue", "moonlight"], synonyms=["night", "moonlight", "american night"])
def day_for_night(x, t, i, rng, fps):
    l = O.luma(x)
    d = O.saturate(x, 0.35)
    d = d * O.col((0.55, 0.72, 1.05), x)
    d = d.clamp(0, 1).pow(1.25) * 0.95
    # keep a bit of 'moonlight' on the very brightest areas
    d = d + O.smoothstep(0.8, 1.0, l) * O.col((0.10, 0.14, 0.2), x)
    d = d * O.vignette_mask(x, 0.4, 0.35, 1.35)[None, ..., None]
    return O.finish(x, d, i)


@look("infrared", "Infrared", "color", "False-colour infrared film: foliage glows pink-red, skies turn deep blue.",
      tags=["surreal", "false-color", "aerochrome"], synonyms=["aerochrome", "ir", "false colour"])
def infrared(x, t, i, rng, fps):
    r, g, b = x.unbind(-1)
    ir = (1.25 * g + 0.35 * r - 0.45 * b).clamp(0, 1)
    out = torch.stack([0.15 + 0.9 * ir, 0.85 * r + 0.1 * ir, 0.25 * g + 0.75 * b], -1)
    out = O.saturate(out.clamp(0, 1), 1.25)
    out = O.scurve(out.clamp(0, 1), 4.5)
    return O.finish(x, out, i)


@look("thermal", "Thermal", "color", "Heat-camera palette: cold purples through hot yellow-white.",
      tags=["heatmap", "predator", "sci-fi"], synonyms=["heat vision", "infrared camera", "flir", "heat map"])
def thermal(x, t, i, rng, fps):
    l = _norm(O.blur(O.luma(x), O.S(x) * 0.004))
    out = O.gradient_map(l, [(0.0, (0.02, 0.0, 0.10)), (0.2, (0.22, 0.02, 0.55)), (0.42, (0.75, 0.05, 0.50)),
                             (0.62, (0.98, 0.35, 0.05)), (0.82, (1.0, 0.80, 0.10)), (1.0, (1.0, 1.0, 0.9))])
    return O.finish(x, out, i)


@look("invert", "Invert", "color", "Photographic negative.", tags=["negative"], synonyms=["negative", "inverse"])
def invert(x, t, i, rng, fps):
    return O.finish(x, 1 - x, i)


@look("solarize", "Solarize", "color", "Sabattier effect: highlights fold back into dark, surreal tones.",
      tags=["surreal", "psychedelic", "darkroom"], synonyms=["sabattier", "man ray"])
def solarize(x, t, i, rng, fps):
    y = torch.where(x < 0.5, x, 1 - x) * 2
    y = O.mix(y, O.luma(y).expand_as(y), 0.25)
    y = O.scurve(y.clamp(0, 1), 4.0)
    return O.finish(x, y, i)


@look("posterize", "Posterize", "color", "Flat bands of colour, like a screen print.",
      tags=["graphic", "pop-art"], synonyms=["poster", "banding", "pop art"])
def posterize(x, t, i, rng, fps):
    y = O.blur(x, O.S(x) * 0.002)
    n = 5
    y = torch.round(O.saturate(y, 1.2).clamp(0, 1) * (n - 1)) / (n - 1)
    return O.finish(x, y, i)


@look("hue-cycle", "Hue Cycle", "color", "Colours rotate continuously around the colour wheel.",
      tags=["psychedelic", "rainbow", "trippy"], synonyms=["rainbow", "color cycle", "hue shift", "psychedelic"], animated=True)
def hue_cycle(x, t, i, rng, fps):
    ph = float(O.urand(rng, x.device, (1,), "hue-cycle")[0]) * 0.3 + 0.35
    ang = (t * 0.35 + ph) * 2 * math.pi
    y = O.hue_rotate(O.saturate(x, 1.25), ang)
    return O.finish(x, y, i)


@look("crushed-blacks", "Crushed Blacks", "color", "Deep, inky shadows with rich contrast.",
      tags=["contrast", "moody"], synonyms=["deep blacks", "high contrast", "moody"])
def crushed_blacks(x, t, i, rng, fps):
    y = O.levels(x, 0.10, 1.0, 0.85)
    y = O.scurve(y, 4.5)
    y = O.saturate(y, 1.08)
    return O.finish(x, y, i)


@look("faded-blacks", "Faded Blacks", "color", "Matte, lifted shadows and soft highlights, like a faded print.",
      tags=["matte", "vintage", "instagram"], synonyms=["matte", "faded", "washed out", "lifted blacks"])
def faded_blacks(x, t, i, rng, fps):
    y = 0.11 + x * 0.82
    y = O.saturate(y, 0.85)
    l = O.luma(x)
    y = y + (1 - l) * O.col((0.015, 0.0, 0.03), x)
    return O.finish(x, y, i)


@look("selective-color", "Selective Color", "color", "Everything turns black and white except the reds.",
      tags=["color-splash", "red"], synonyms=["color splash", "sin city", "keep red"])
def selective_color(x, t, i, rng, fps):
    hsv = O.rgb_to_hsv(x.clamp(0, 1))
    h, s, v = hsv[..., 0:1], hsv[..., 1:2], hsv[..., 2:3]
    d = torch.minimum(h, 1 - h)  # distance to red hue
    keep = (1 - O.smoothstep(0.035, 0.085, d)) * O.smoothstep(0.18, 0.35, s) * O.smoothstep(0.08, 0.2, v)
    keep = O.blur(keep, O.S(x) * 0.002)
    g = O.scurve(_gray(x), 4.5).expand_as(x)
    red = O.saturate(x, 1.25)
    out = O.mix(g, red, keep)
    return O.finish(x, out, i)
