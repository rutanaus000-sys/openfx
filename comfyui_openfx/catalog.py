"""Export the look catalogue as JSON: `python -m comfyui_openfx.catalog > looks.json`."""

from __future__ import annotations

import json
import sys

from .registry import CATEGORIES, LOOKS, load_all


def export_catalog() -> list[dict]:
    load_all()
    order = {c: i for i, c in enumerate(CATEGORIES)}
    return [lk.meta() for lk in sorted(LOOKS.values(), key=lambda l: order[l.category])]


def main() -> None:
    json.dump(export_catalog(), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
