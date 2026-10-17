"""Effect registry.

An effect is a folder under ``effects/`` holding a ``manifest.json``. The
manifest names a base workflow (an API-format ComfyUI graph under
``workflows/``) and the prompt and settings that turn that workflow into one
look. Each base workflow ships a ``<name>.bindings.json`` next to it that maps
the app's inputs (image, prompt, width, height, seconds, ...) to node inputs, so
an effect never has to know node ids.

Adding an effect is adding a folder: no code changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EFFECTS_DIR = ROOT / "effects"

# Inputs a user may override per run. Everything else comes from the manifest.
USER_PARAMS = ("width", "height", "seconds", "seed")


@dataclass
class Effect:
    id: str
    name: str
    description: str
    workflow: Path
    prompt: str
    negative_prompt: str | None = None
    defaults: dict[str, Any] = field(default_factory=dict)
    extra_inputs: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    preview: str | None = None

    def public(self) -> dict[str, Any]:
        """What the web UI is allowed to see."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "preview": f"/effects/{self.id}/{self.preview}" if self.preview else None,
            "defaults": self.defaults,
        }


def _load(folder: Path) -> Effect:
    data = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    workflow = (ROOT / data["workflow"]).resolve()
    if not workflow.is_file():
        raise FileNotFoundError(f"{folder.name}: workflow not found: {data['workflow']}")
    return Effect(
        id=folder.name,
        name=data["name"],
        description=data.get("description", ""),
        workflow=workflow,
        prompt=data["prompt"],
        negative_prompt=data.get("negative_prompt"),
        defaults=data.get("defaults", {}),
        extra_inputs=data.get("inputs", {}),
        tags=data.get("tags", []),
        preview=data.get("preview"),
    )


def load_effects() -> dict[str, Effect]:
    effects: dict[str, Effect] = {}
    for folder in sorted(EFFECTS_DIR.iterdir()):
        if (folder / "manifest.json").is_file():
            try:
                effects[folder.name] = _load(folder)
            except (KeyError, ValueError, FileNotFoundError) as exc:
                print(f"[openfx] skipping effect {folder.name}: {exc}")
    return effects


def load_bindings(workflow: Path) -> dict[str, list[str]]:
    """``{"image": ["269", "image"], ...}`` for one base workflow."""
    path = workflow.with_name(workflow.name.replace("_api.json", "") + ".bindings.json")
    return json.loads(path.read_text(encoding="utf-8"))


def build_prompt(effect: Effect, detail: str | None) -> str:
    detail = (detail or "").strip()
    return effect.prompt.replace("{detail}", f" {detail}" if detail else "").strip()
