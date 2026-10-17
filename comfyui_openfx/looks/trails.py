"""Trail looks. They read previous frames, so they see the whole clip; each output
frame is built in a loop so memory stays ~2x the clip even for long videos."""

import torch

from .. import ops as O
from ..registry import look


def _p(x, k):
    return x[max(0, k)]


@look("echo", "Echo", "trails", "Fading copies of the last moments trail behind anything that moves.",
      tags=["trail", "motion", "dreamy"], synonyms=["echo trail", "motion trail", "afterimage", "ghost frames"], animated=True)
def echo(x, t, i, rng, fps):
    gap = max(1, int(round(fps * 0.12)))
    out = torch.empty_like(x)
    ws = [0.62 ** j for j in range(6)]
    tot = sum(ws)
    for k in range(x.shape[0]):
        acc = sum(_p(x, k - j * gap) * w for j, w in enumerate(ws)) / tot
        out[k] = O.mix(x[k], O.mix(acc, x[k], 0.25), i)
    return out.clamp(0, 1)


@look("ghost-trail", "Ghost Trail", "trails", "Glowing, cool-tinted afterimages that linger on bright moving shapes.",
      tags=["trail", "glow", "spooky"], synonyms=["afterimage", "light trail", "ghosting trail", "phantom"], animated=True)
def ghost_trail(x, t, i, rng, fps):
    gap = max(1, int(round(fps * 0.06)))
    tint = O.col((0.7, 0.85, 1.15), x)
    s = O.S(x)
    out = torch.empty_like(x)
    for k in range(x.shape[0]):
        y = x[k]
        for j in range(1, 10):
            y = torch.maximum(y, (_p(x, k - j * gap) * tint * (0.8 ** j)).clamp(0, 1))
        glow = O.blur((y - x[k]).clamp(0, 1)[None], s * 0.01)[0]
        y = O.screen(y, glow * 0.6)
        out[k] = O.mix(x[k], y, i)
    return out.clamp(0, 1)


@look("long-exposure", "Long Exposure", "trails", "Motion smears into silky streaks while lights paint bright trails.",
      tags=["light-painting", "smooth", "night"], synonyms=["light trails", "slow shutter", "light painting", "motion smear"], animated=True)
def long_exposure(x, t, i, rng, fps):
    Fn = x.shape[0]
    win = max(2, int(round(fps * 1.0)))
    out = torch.empty_like(x)
    run_sum = torch.zeros_like(x[0])
    run_max = x[0].clone()
    for k in range(Fn):
        run_sum = run_sum + x[k]
        if k - win >= 0:
            run_sum = run_sum - x[k - win]
        n = min(k + 1, win)
        mean = run_sum / n
        run_max = torch.maximum(run_max * 0.97, x[k])
        y = mean * 0.7 + run_max * 0.3
        hl = O.smoothstep(0.65, 1.0, O.luma(run_max))
        y = O.mix(y, run_max, hl * 0.7)
        out[k] = O.mix(x[k], y, i)
    return out.clamp(0, 1)


@look("strobe", "Strobe", "trails", "Stroboscopic multiple exposure: crisp, distinct stages of the motion frozen together.",
      tags=["multiple-exposure", "sports", "dance"], synonyms=["stroboscope", "multiple exposure", "stroboscopic", "chronophotography"], animated=True)
def strobe(x, t, i, rng, fps):
    Fn = x.shape[0]
    step = max(2, int(round(fps * 0.16)))
    out = torch.empty_like(x)
    for k in range(Fn):
        held = (k // step) * step
        cur = x[held]
        y = cur.clone()
        for j, w in [(1, 0.65), (2, 0.42), (3, 0.25)]:
            y = torch.maximum(y, O.mix(cur * 0.55, _p(x, held - j * step), w))
        if k == held:
            y = O.screen(y, torch.full_like(y, 0.1))
        out[k] = O.mix(x[k], y, i)
    return out.clamp(0, 1)


@look("color-trails", "Color Trails", "trails", "Red, green and blue are taken from different moments, leaving rainbow fringes on motion.",
      tags=["rgb", "psychedelic", "trail"], synonyms=["rgb delay", "chromatic trails", "rainbow trail", "channel delay"], animated=True)
def color_trails(x, t, i, rng, fps):
    d = max(1, int(round(fps * 0.15)))
    out = torch.empty_like(x)
    for k in range(x.shape[0]):
        y = torch.stack([x[k, ..., 0], _p(x, k - d)[..., 1], _p(x, k - 2 * d)[..., 2]], -1)
        out[k] = O.mix(x[k], O.saturate(y, 1.1), i)
    return out.clamp(0, 1)
