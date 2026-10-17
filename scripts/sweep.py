"""Render a grid of variants for AI effects so we can pick the best settings.

    python scripts/sweep.py --effects disintegrate on-fire --images test_images \
        --sizes 576x1024 720x1280 --seeds 2 --enhance off on

Writes videos to outputs/sweep/<effect>/ and a results.csv with timings.
Needs ComfyUI + the Comfy API proxy running (COMFY_BASE_URL, default :8189).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("COMFY_BASE_URL", "http://127.0.0.1:8189")

from comfy_sdk import Comfy  # noqa: E402

from openfx.effects import ROOT, build_prompt, load_bindings, load_effects  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--effects", nargs="+", required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--sizes", nargs="+", default=["576x1024"])
    ap.add_argument("--seconds", type=int, default=None, help="override the manifest length")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--enhance", nargs="+", default=["off"], choices=["off", "on"])
    ap.add_argument("--prompt-file", type=Path, help="text file whose contents replace the manifest prompt")
    ap.add_argument("--tag", default="", help="label added to output names")
    args = ap.parse_args()

    effects = load_effects()
    client = Comfy()
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    out_root = ROOT / "outputs" / "sweep"
    log = out_root / "results.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    new_log = not log.exists()

    with log.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if new_log:
            writer.writerow(["file", "effect", "image", "size", "seconds", "seed", "enhance", "tag", "render_s", "status"])
        for eid, image, size, seed_i, enhance in itertools.product(
            args.effects, images, args.sizes, range(args.seeds), args.enhance
        ):
            effect = effects[eid]
            width, height = (int(x) for x in size.split("x"))
            seconds = args.seconds or effect.defaults.get("seconds", 4)
            seed = 1000 + seed_i
            prompt = args.prompt_file.read_text(encoding="utf-8").strip() if args.prompt_file else effect.prompt
            effect_prompt = build_prompt(replace(effect, prompt=prompt), None)

            bindings = load_bindings(effect.workflow)
            wf = client.workflows.from_file(effect.workflow)
            values = {
                "image": client.assets.from_file(image),
                "prompt": effect_prompt,
                "width": width,
                "height": height,
                "seconds": seconds,
                "seed": seed,
                "enhance_prompt": enhance == "on",
            }
            if effect.negative_prompt is not None:
                values["negative_prompt"] = effect.negative_prompt
            for name, value in values.items():
                if name in bindings:
                    wf.set_input(*bindings[name], value)

            name = f"{image.stem}_{size}_{seconds}s_seed{seed}_enh-{enhance}{'_' + args.tag if args.tag else ''}.mp4"
            dest = out_root / eid / name
            dest.parent.mkdir(exist_ok=True)
            start = time.time()
            try:
                job = client.run(wf)
                video = next(o for o in job.outputs if o.content_type.startswith("video/"))
                video.to_file(dest)
                status = "ok"
            except Exception as exc:  # keep sweeping; the CSV records the failure
                status = f"failed: {exc}"
            elapsed = round(time.time() - start, 1)
            writer.writerow([dest.relative_to(ROOT), eid, image.name, size, seconds, seed, enhance, args.tag, elapsed, status])
            fh.flush()
            print(f"{status:6} {elapsed:6.1f}s  {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
