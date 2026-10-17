# OpenFX Looks for ComfyUI

A ComfyUI custom node pack with 116 deterministic "Looks" (video filters) for IMAGE batches: film stocks, VHS, glitch, lenses, distortions, time effects, trails, layouts and art styles. It's written in pure PyTorch, with no extra pip dependencies. It runs on CUDA or CPU and gives the same output for the same seed.

## Install

Copy or symlink this folder into `ComfyUI/custom_nodes/`, then restart ComfyUI:

```bash
ln -s /path/to/openfx/comfyui_openfx ComfyUI/custom_nodes/comfyui_openfx
```

## Nodes

| Node | Inputs | Output |
|---|---|---|
| **OpenFX Look** | `images`, `look` (dropdown), `intensity` 0-1, `seed`, `fps` | IMAGE |
| **OpenFX Look Stack** | `images`, `stack` (JSON, e.g. `[{"look":"vhs","intensity":0.8},{"look":"film-grain"}]`), `seed`, `fps` | IMAGE |
| **OpenFX Still To Clip** | `image` (uses the first frame), `seconds`, `fps`, `motion` (`none`, `slow push-in`, `slow pull-out`, `drift`) | IMAGE batch |

- An IMAGE batch is treated as video frames `[F, H, W, 3]`.
- Run Still To Clip before a Look to animate a photo. Grain, flicker and leaks will move, and the camera makes a gentle move.
- `intensity` 0 returns the input, and 1 applies the full, tuned effect.
- The Look Stack node passes the input through unchanged for an empty or `[]` stack. It raises `ValueError` for an unknown look id.
- Work runs on ComfyUI's torch device (the GPU when one is available). Heavy looks process the clip in chunks of frames.

## Catalog

```bash
python -m comfyui_openfx.catalog > looks.json   # run from the repo root; ComfyUI is not needed
```

Each entry has these fields: `id`, `name`, `category`, `description`, `tags`, `synonyms`, `animated`.

## Adding a look

Put a function in the matching file under `looks/` (or add a new file and import it in `looks/__init__.py`):

```python
from .. import ops as O
from ..registry import look

@look("warm-glow", "Warm Glow", "color", "Golden-hour warmth with soft highlights.",
      tags=["warm"], synonyms=["golden hour"], animated=False)
def warm_glow(frames, t, intensity, rng, fps):
    # frames: [F,H,W,3] float 0..1   t: [F] seconds   rng: torch.Generator
    out = O.screen(frames * O.col((1.06, 1.0, 0.9), frames), O.blur(frames, O.S(frames) * 0.02) * 0.3)
    return O.finish(frames, out, intensity)  # blends by intensity and clamps
```

Rules:

- Scale sizes by `O.S(frames)`, which is min(H, W), so the look works at any resolution.
- Keep the output shape the same as the input.
- Use the `ops` random helpers (`O.urand` for per-clip values, `O.frame_rand` / `O.evo_noise` / `O.vnoise1` for per-frame values). They key off the seed and the absolute frame index, which keeps a look deterministic and gives the same result however the frames are chunked.
- A look that reads other frames needs `temporal=True`. This is automatic for the `time` and `trails` categories.

Tests: `pytest tests/test_looks.py`. Visual QA: `python tests/render_contact_sheet.py out.png`.
