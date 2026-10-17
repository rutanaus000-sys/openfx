"""ComfyUI node classes. ComfyUI is imported lazily so the pack also works standalone."""

from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F

from .registry import apply_look, apply_stack, get_look, look_ids

MOTIONS = ["none", "slow push-in", "slow pull-out", "drift"]


def _compute_device(images):
    """Prefer ComfyUI's torch device (usually the GPU); fall back to CUDA or the input's device."""
    try:
        import comfy.model_management as mm  # type: ignore

        return mm.get_torch_device()
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return images.device


def _seed_widget():
    return ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF})


def _fps_widget():
    return ("FLOAT", {"default": 25.0, "min": 1.0, "max": 240.0, "step": 0.5})


def parse_stack(stack):
    """Parse the Look Stack JSON. Empty string or [] -> []. Validates every look id."""
    if stack is None:
        return []
    if isinstance(stack, str):
        txt = stack.strip()
        if not txt:
            return []
        data = json.loads(txt)
    else:
        data = stack
    if isinstance(data, dict):
        data = data.get("stack", data.get("looks", [data]))
    steps = []
    for item in data:
        if isinstance(item, str):
            item = {"look": item}
        lid = item.get("look") or item.get("id")
        get_look(str(lid))  # ValueError naming the unknown id
        steps.append({"look": str(lid), "intensity": float(item.get("intensity", 1.0)), **({"seed": int(item["seed"])} if "seed" in item else {})})
    return steps


class OpenFXLook:
    CATEGORY = "OpenFX"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    DESCRIPTION = "Apply one deterministic OpenFX look (video filter) to an IMAGE batch."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "look": (look_ids(),),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": _seed_widget(),
                "fps": _fps_widget(),
            }
        }

    def apply(self, images, look, intensity, seed, fps):
        out = apply_look(images, look, intensity, seed, fps, device=_compute_device(images))
        return (out,)


class OpenFXLookStack:
    CATEGORY = "OpenFX"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    DESCRIPTION = 'Apply several looks in order. stack: JSON list, e.g. [{"look":"vhs","intensity":0.8},{"look":"film-grain"}].'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "stack": ("STRING", {"default": '[{"look": "film-35mm", "intensity": 1.0}]', "multiline": True}),
                "seed": _seed_widget(),
                "fps": _fps_widget(),
            }
        }

    def apply(self, images, stack, seed, fps):
        steps = parse_stack(stack)
        if not steps:
            return (images,)
        return (apply_stack(images, steps, seed, fps, device=_compute_device(images)),)


def still_to_clip(image, seconds=4.0, fps=25.0, motion="none", device=None):
    """Turn the first frame of `image` into an [F,H,W,3] clip with an optional gentle camera move."""
    img = image[:1, ..., :3].float() if image.dim() == 4 else image[None, ..., :3].float()
    H, W = img.shape[1] - img.shape[1] % 2, img.shape[2] - img.shape[2] % 2
    img = img[:, :H, :W]
    n = max(1, int(round(float(seconds) * float(fps))))
    dev = device if device is not None else img.device
    src = img.to(dev)
    if motion == "none":
        return src.expand(n, -1, -1, -1).contiguous().to(img.device)
    u = torch.linspace(0, 1, n, device=dev) if n > 1 else torch.zeros(1, device=dev)
    ease = u * u * (3 - 2 * u) * 0.35 + u * 0.65  # mostly linear, soft start/end
    if motion == "slow push-in":
        scale, dx, dy = 1 + 0.12 * ease, 0 * u, 0 * u
    elif motion == "slow pull-out":
        scale, dx, dy = 1.12 - 0.12 * ease, 0 * u, 0 * u
    elif motion == "drift":
        scale = 1.08 + 0 * u
        dx = (ease - 0.5) * 0.06
        dy = (ease - 0.5) * -0.015
    else:
        raise ValueError(f"unknown motion {motion!r}; expected one of {MOTIONS}")
    v = (torch.arange(H, device=dev, dtype=torch.float32) + 0.5) / H * 2 - 1
    h = (torch.arange(W, device=dev, dtype=torch.float32) + 0.5) / W * 2 - 1
    vv, uu = torch.meshgrid(v, h, indexing="ij")
    out = torch.empty(n, H, W, 3, device=img.device)
    src_n = src.permute(0, 3, 1, 2)
    for a in range(0, n, 16):
        b = min(n, a + 16)
        sc = scale[a:b].view(-1, 1, 1)
        gu = uu[None] / sc + dx[a:b].view(-1, 1, 1)
        gv = vv[None] / sc + dy[a:b].view(-1, 1, 1)
        g = torch.stack([gu, gv], -1)
        y = F.grid_sample(src_n.expand(b - a, -1, -1, -1), g, mode="bicubic", padding_mode="border", align_corners=False)
        out[a:b] = y.permute(0, 2, 3, 1).clamp(0, 1).to(img.device)
    return out


class OpenFXStillToClip:
    CATEGORY = "OpenFX"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "make"
    DESCRIPTION = "Turn a still photo into a short clip (with an optional gentle camera move) so animated looks can move."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "seconds": ("FLOAT", {"default": 4.0, "min": 0.04, "max": 60.0, "step": 0.1}),
                "fps": _fps_widget(),
                "motion": (MOTIONS,),
            }
        }

    def make(self, image, seconds, fps, motion):
        return (still_to_clip(image, seconds, fps, motion, device=_compute_device(image)),)


NODE_CLASS_MAPPINGS = {
    "OpenFXLook": OpenFXLook,
    "OpenFXLookStack": OpenFXLookStack,
    "OpenFXStillToClip": OpenFXStillToClip,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenFXLook": "OpenFX Look",
    "OpenFXLookStack": "OpenFX Look Stack",
    "OpenFXStillToClip": "OpenFX Still To Clip",
}
