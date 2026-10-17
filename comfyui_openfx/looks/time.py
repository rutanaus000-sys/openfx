import math

import torch

from .. import ops as O
from ..registry import look


def _resample(x, pos):
    """Sample frames at fractional positions pos [F'] with linear frame blending."""
    Fn = x.shape[0]
    pos = pos.clamp(0, Fn - 1)
    a = torch.floor(pos).long()
    b = (a + 1).clamp(max=Fn - 1)
    f = (pos - a.to(pos.dtype)).tolist()
    out = torch.empty((len(f),) + tuple(x.shape[1:]), device=x.device, dtype=x.dtype)
    for j, (ia, ib, fj) in enumerate(zip(a.tolist(), b.tolist(), f)):
        out[j] = x[ia] * (1 - fj) + x[ib] * fj if fj > 1e-6 else x[ia]
    return out


@look("reverse", "Reverse", "time", "Plays the clip backwards.", tags=["rewind", "backwards"], synonyms=["rewind", "backwards", "reverse playback"])
def reverse(x, t, i, rng, fps):
    if i <= 1e-3:
        return x
    return x.flip(0)


@look("ping-pong", "Ping-Pong", "time", "Plays forward at double speed, then bounces back, for a seamless loop.",
      tags=["loop", "boomerang"], synonyms=["boomerang", "bounce", "loop", "back and forth"])
def ping_pong(x, t, i, rng, fps):
    Fn = x.shape[0]
    if Fn < 3 or i <= 1e-3:
        return x
    k = torch.arange(Fn, device=x.device, dtype=torch.float32)
    u = k / (Fn - 1)
    tri = 1 - (2 * u - 1).abs()  # 0 -> 1 -> 0
    pos = tri * (Fn - 1)
    pos = O.mix(k, pos, i)
    return _resample(x, pos)


@look("freeze-frame", "Freeze Frame", "time", "Plays the first half, then freezes on the middle frame with a quick flash.",
      tags=["freeze", "pause", "dramatic"], synonyms=["freeze", "pause", "still frame", "record scratch"])
def freeze_frame(x, t, i, rng, fps):
    Fn = x.shape[0]
    if Fn < 2 or i <= 1e-3:
        return x
    mid = Fn // 2
    idx = torch.arange(Fn, device=x.device).clamp(max=mid)
    y = x[idx].clone()
    # short exposure 'pop' on the freeze, then a slightly desaturated hold
    k = torch.arange(Fn, device=x.device, dtype=x.dtype)
    flash = torch.exp(-((k - mid) / max(1.0, fps * 0.05)).clamp(min=0)) * (k >= mid).to(x.dtype)
    held = (k >= mid).to(x.dtype).view(-1, 1, 1, 1)
    y = O.mix(y, O.saturate(y, 0.85), held * 0.6)
    y = O.screen(y, flash.view(-1, 1, 1, 1) * 0.35)
    return O.mix(x, y, i).clamp(0, 1) if i < 1 else y.clamp(0, 1)


@look("stop-motion", "Stop Motion", "time", "Choppy, handmade stop-motion: frames held at an 8 fps cadence with tiny jitters.",
      tags=["claymation", "choppy", "handmade"], synonyms=["claymation", "choppy", "low frame rate", "8fps", "animation"], animated=True)
def stop_motion(x, t, i, rng, fps):
    Fn = x.shape[0]
    if Fn < 2 or i <= 1e-3:
        return x
    step = max(1.0, fps / 8.0)
    k = torch.arange(Fn, device=x.device, dtype=torch.float32)
    held = torch.floor(torch.floor(k / step) * step).long().clamp(max=Fn - 1)
    y = x[held]
    s = O.S(x)
    keys = [int(v) for v in torch.floor(k / step).tolist()]
    j = O.key_rand(rng, keys, (3,), "stopm", device=x.device) - 0.5
    y = O.shift(y, j[:, 0] * s * 0.004, j[:, 1] * s * 0.004)
    y = y * (1 + j[:, 2] * 0.06).view(-1, 1, 1, 1)
    return y.clamp(0, 1)


@look("speed-ramp", "Speed Ramp", "time", "Slow-fast-slow time remapping, like an action-sports edit.",
      tags=["slow-motion", "action", "edit"], synonyms=["slow motion", "time ramp", "velocity edit", "slowmo"])
def speed_ramp(x, t, i, rng, fps):
    Fn = x.shape[0]
    if Fn < 3 or i <= 1e-3:
        return x
    u = torch.arange(Fn, device=x.device, dtype=torch.float32) / (Fn - 1)
    # slow at the ends, fast in the middle
    s = u - torch.sin(2 * math.pi * u) / (2 * math.pi)
    pos = O.mix(u, s, i) * (Fn - 1)
    return _resample(x, pos)


@look("time-displace", "Time Displacement", "time", "Each row of the picture lags further behind in time, bending anything that moves.",
      tags=["temporal", "warp", "experimental"], synonyms=["time warp", "rolling shutter", "scan warp"])
def time_displace(x, t, i, rng, fps):
    Fn, H = x.shape[0], x.shape[1]
    if Fn < 2 or i <= 1e-3:
        return x
    D = min(Fn - 1, fps * 1.0) * i
    k = torch.arange(Fn, device=x.device, dtype=torch.float32).view(-1, 1)
    ys = (torch.arange(H, device=x.device, dtype=torch.float32) / max(1, H - 1)).view(1, -1)
    src = (k - ys * D).clamp(0, Fn - 1)
    a = torch.floor(src).long()
    b = (a + 1).clamp(max=Fn - 1)
    f = (src - a.float()).to(x.dtype)[..., None, None]
    rows = torch.arange(H, device=x.device)
    out = torch.empty_like(x)
    for j in range(Fn):  # per-frame to keep memory flat on long clips
        out[j] = x[a[j], rows] * (1 - f[j]) + x[b[j], rows] * f[j]
    return out.clamp(0, 1)


@look("slit-scan", "Slit Scan", "time", "Columns are sampled from different moments, sweeping time across the frame like a slit-scan camera.",
      tags=["temporal", "experimental", "2001"], synonyms=["slitscan", "time slice", "stargate"], animated=True)
def slit_scan(x, t, i, rng, fps):
    Fn, H, W = x.shape[0], x.shape[1], x.shape[2]
    if Fn < 2 or i <= 1e-3:
        return x
    D = min(Fn - 1, fps * 1.5) * i
    k = torch.arange(Fn, device=x.device, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(W, device=x.device, dtype=torch.float32).view(1, -1) / max(1, W - 1)
    # delay is a moving wave across the columns
    ph = (k / max(1.0, fps)) * 1.2
    delay = (0.5 + 0.5 * torch.sin(xs * 2 * math.pi * 1.0 - ph)) * D
    src = (k - delay).clamp(0, Fn - 1)
    a = torch.floor(src).long()
    b = (a + 1).clamp(max=Fn - 1)
    f = (src - a.float()).to(x.dtype)
    cols = torch.arange(W, device=x.device).view(1, -1)
    rows = torch.arange(H, device=x.device).view(-1, 1)
    out = torch.empty_like(x)
    for j in range(Fn):
        fa = x[a[j].view(1, -1), rows, cols]  # [H, W, 3]
        fb = x[b[j].view(1, -1), rows, cols]
        fj = f[j].view(1, -1, 1)
        out[j] = fa * (1 - fj) + fb * fj
    return out.clamp(0, 1)
