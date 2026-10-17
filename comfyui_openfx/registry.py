"""Look registry. Pure torch; never imports ComfyUI so it can be used standalone.

Register a look with the ``@look`` decorator::

    @look("my-look", "My Look", "color", "What it does.", tags=["warm"], synonyms=["cozy"])
    def my_look(frames, t, intensity, rng, fps):
        ...  # frames: [F,H,W,3] float 0..1, t: [F] seconds
        return out  # [F,H,W,3]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch

CATEGORIES = ["color", "film", "retro", "glitch", "blur", "lens", "distort", "time", "trails", "layout", "art"]

# categories whose looks read across frames, so they must see the whole clip at once
TEMPORAL_CATEGORIES = {"time", "trails"}


@dataclass
class Look:
    id: str
    name: str
    category: str
    description: str
    fn: Callable
    tags: list = field(default_factory=list)
    synonyms: list = field(default_factory=list)
    animated: bool = False
    temporal: bool = False

    def meta(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "tags": list(self.tags),
            "synonyms": list(self.synonyms),
            "animated": bool(self.animated),
        }


LOOKS: dict[str, Look] = {}


def look(id, name, category, description, tags=(), synonyms=(), animated=False, temporal=None):
    """Decorator that registers a look function under `id`."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r} for look {id!r}")

    def deco(fn):
        if id in LOOKS:
            raise ValueError(f"duplicate look id {id!r}")
        tmp = (category in TEMPORAL_CATEGORIES) if temporal is None else bool(temporal)
        LOOKS[id] = Look(id, name, category, description, fn, list(tags), list(synonyms), bool(animated), tmp)
        return fn

    return deco


_loaded = False


def load_all():
    global _loaded
    if not _loaded:
        from . import looks  # noqa: F401  (registers everything)

        _loaded = True
    return LOOKS


def get_look(look_id: str) -> Look:
    load_all()
    if look_id not in LOOKS:
        raise ValueError(f"Unknown OpenFX look id: {look_id!r}")
    return LOOKS[look_id]


def look_ids() -> list[str]:
    load_all()
    order = {c: i for i, c in enumerate(CATEGORIES)}
    return [l.id for l in sorted(LOOKS.values(), key=lambda l: order[l.category])]


def _chunk_size(h, w):
    # ~16 frames at 720x1280 keeps the heaviest looks well inside 12 GB
    return max(1, int(16 * (720 * 1280) / max(1, h * w)))


@torch.no_grad()
def apply_look(frames, look_id, intensity=1.0, seed=0, fps=25.0, device=None, chunk=None):
    """Run one look on an IMAGE batch. Returns a tensor on the input's device.

    device: where to compute (default: the input's device).
    """
    lk = get_look(look_id)
    src_dev = frames.device
    dev = torch.device(device) if device is not None else src_dev
    intensity = float(min(1.0, max(0.0, intensity)))
    fps = float(fps) if fps and fps > 0 else 25.0
    if frames.dim() == 3:
        frames = frames[None]
    frames = frames[..., :3].float()
    n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    t_all = torch.arange(n, dtype=torch.float32) / fps

    def run(x, t):
        x = x.to(dev)
        rng = torch.Generator(device=dev)
        rng.manual_seed(seed)
        y = lk.fn(x, t.to(dev), intensity, rng, fps)
        return y.clamp(0, 1).nan_to_num(0.0).to(src_dev)

    if lk.temporal:
        return run(frames, t_all)
    cs = chunk or _chunk_size(h, w)
    if n <= cs:
        return run(frames, t_all)
    out = torch.empty_like(frames)
    for a in range(0, n, cs):
        out[a:a + cs] = run(frames[a:a + cs], t_all[a:a + cs])
    return out


def apply_stack(frames, stack, seed=0, fps=25.0, device=None):
    """stack: list of {"look": id, "intensity": float?, "seed": int?} (or bare ids). Validates all ids first."""
    steps = [{"look": st} if isinstance(st, str) else dict(st) for st in stack]
    for st in steps:
        get_look(str(st.get("look") or st.get("id")))
    for k, st in enumerate(steps):
        s = int(st.get("seed", (int(seed) + k * 7919) & 0xFFFFFFFFFFFFFFFF))
        frames = apply_look(frames, str(st.get("look") or st.get("id")), float(st.get("intensity", 1.0)), s, fps, device=device)
    return frames
