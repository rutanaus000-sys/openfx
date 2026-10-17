"""Tests for the OpenFX ComfyUI look pack (pure torch, CPU)."""

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfyui_openfx import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402
from comfyui_openfx.catalog import export_catalog  # noqa: E402
from comfyui_openfx.nodes import OpenFXLookStack, still_to_clip  # noqa: E402
from comfyui_openfx.registry import CATEGORIES, LOOKS, apply_look, load_all  # noqa: E402

load_all()
IDS = sorted(LOOKS)
EXEMPT_FROM_IDENTITY = {"time", "layout"}


def _clip(frames, h=96, w=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(frames, h // 8, w // 8, 3, generator=g)
    x = torch.nn.functional.interpolate(base.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False)
    x = x.permute(0, 2, 3, 1)
    x = x * 0.8 + torch.rand(frames, h, w, 3, generator=g) * 0.2
    return x.clamp(0, 1).contiguous()


CLIP = _clip(6)
STILL = _clip(1, seed=1)


def _check(out, ref):
    assert out.shape == ref.shape
    assert out.dtype == torch.float32
    assert not torch.isnan(out).any()
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize("look_id", IDS)
def test_look_clip(look_id):
    a = apply_look(CLIP, look_id, 1.0, seed=123, fps=25)
    _check(a, CLIP)
    b = apply_look(CLIP, look_id, 1.0, seed=123, fps=25)
    assert torch.equal(a, b), "not deterministic"


@pytest.mark.parametrize("look_id", IDS)
def test_look_single_frame(look_id):
    a = apply_look(STILL, look_id, 1.0, seed=5, fps=25)
    _check(a, STILL)
    assert torch.equal(a, apply_look(STILL, look_id, 1.0, seed=5, fps=25))


@pytest.mark.parametrize("look_id", IDS)
def test_intensity_zero_is_identity(look_id):
    lk = LOOKS[look_id]
    if lk.category in EXEMPT_FROM_IDENTITY:
        pytest.skip("time/layout looks are exempt")
    out = apply_look(CLIP, look_id, 0.0, seed=1, fps=25)
    assert (out - CLIP).abs().max() < 2e-3


@pytest.mark.parametrize("look_id", IDS)
def test_look_changes_image(look_id):
    """At full intensity every look must visibly do something on a moving clip."""
    clip = still_to_clip(_clip(1, seed=3), seconds=0.4, fps=25, motion="drift")
    out = apply_look(clip, look_id, 1.0, seed=2, fps=25)
    assert (out - clip).abs().mean() > 1e-3


@pytest.mark.parametrize("look_id", [l for l in IDS if LOOKS[l].animated and not LOOKS[l].temporal])
def test_animated_looks_move(look_id):
    """Animated looks must change over time even on a perfectly still clip (no frozen grain)."""
    still = still_to_clip(_clip(1, seed=4), seconds=1.0, fps=25, motion="none")
    out = apply_look(still, look_id, 1.0, seed=9, fps=25)
    diffs = [(out[k] - out[k - 1]).abs().mean().item() for k in range(1, out.shape[0])]
    assert max(diffs) > 1e-4, "animated look is frozen on a still clip"


@pytest.mark.parametrize("look_id", [l for l in IDS if not LOOKS[l].temporal])
def test_chunking_invariant(look_id):
    a = apply_look(CLIP, look_id, 1.0, seed=7, fps=25)
    b = apply_look(CLIP, look_id, 1.0, seed=7, fps=25, chunk=2)
    assert (a - b).abs().max() < 1e-4


def test_seed_changes_random_looks():
    a = apply_look(CLIP, "film-grain", 1.0, seed=1)
    b = apply_look(CLIP, "film-grain", 1.0, seed=2)
    assert not torch.equal(a, b)


def test_catalog():
    cat = export_catalog()
    assert len(cat) == len(LOOKS) >= 100
    for e in cat:
        assert set(e) == {"id", "name", "category", "description", "tags", "synonyms", "animated"}
        assert e["category"] in CATEGORIES
        assert e["id"] == e["id"].lower() and " " not in e["id"]
    json.dumps(cat)
    assert {e["category"] for e in cat} == set(CATEGORIES)


def test_nodes_registered():
    assert set(NODE_CLASS_MAPPINGS) == {"OpenFXLook", "OpenFXLookStack", "OpenFXStillToClip"}
    assert NODE_DISPLAY_NAME_MAPPINGS["OpenFXLook"] == "OpenFX Look"
    looks = NODE_CLASS_MAPPINGS["OpenFXLook"].INPUT_TYPES()["required"]["look"][0]
    assert set(looks) == set(LOOKS)


def test_look_node():
    (out,) = NODE_CLASS_MAPPINGS["OpenFXLook"]().apply(CLIP, "vhs", 1.0, 3, 25.0)
    _check(out, CLIP)


def test_stack():
    node = OpenFXLookStack()
    (out,) = node.apply(CLIP, '[{"look":"vhs","intensity":0.8}, {"look":"film-grain"}]', 0, 25.0)
    _check(out, CLIP)
    (same,) = node.apply(CLIP, "", 0, 25.0)
    assert torch.equal(same, CLIP)
    (same,) = node.apply(CLIP, "[]", 0, 25.0)
    assert torch.equal(same, CLIP)
    with pytest.raises(ValueError, match="no-such-look"):
        node.apply(CLIP, '[{"look":"sepia"},{"look":"no-such-look"}]', 0, 25.0)


@pytest.mark.parametrize("motion", ["none", "slow push-in", "slow pull-out", "drift"])
def test_still_to_clip(motion):
    img = _clip(2, h=97, w=63)
    (out,) = NODE_CLASS_MAPPINGS["OpenFXStillToClip"]().make(img, 1.0, 10.0, motion)
    assert out.shape == (10, 96, 62, 3)
    assert out.min() >= 0 and out.max() <= 1
    if motion != "none":
        assert (out[0] - out[-1]).abs().mean() > 1e-3
