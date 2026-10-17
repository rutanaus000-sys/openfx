"""Runs effects through the Comfy SDK, one at a time.

The SDK talks to whatever ``COMFY_BASE_URL`` points at: the local API proxy in
front of your own ComfyUI (``http://127.0.0.1:8189``, free, no key), or a Comfy
API deployment (set ``COMFY_API_KEY`` too). The code is the same either way.

A job is an optional AI effect plus an optional stack of Looks. With an AI
effect, the Looks are spliced into its workflow between the video decoder and
the encoder, so it is still one render. Without one, a small Look-only workflow
turns the uploaded photo or video into the clip.

A single worker thread drains the queue because one GPU renders one video at a
time; extra requests wait their turn instead of fighting for VRAM.
"""

from __future__ import annotations

import json
import queue
import random
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from comfy_sdk import Comfy

from .effects import ROOT, Effect, build_prompt, load_bindings

OUTPUT_DIR = ROOT / "outputs"
LOOK_STILL_WORKFLOW = ROOT / "workflows" / "look_still_api.json"
LOOK_VIDEO_WORKFLOW = ROOT / "workflows" / "look_video_api.json"
LOOK_STACK_NODE = "openfx_looks"


@dataclass
class Job:
    id: str
    effect: Effect | None
    looks: list[dict[str, Any]]
    upload: Path
    is_video: bool
    detail: str | None
    params: dict[str, Any]
    status: str = "queued"  # queued | running | done | failed
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    output: Path | None = None
    error: str | None = None

    @property
    def label(self) -> str:
        parts = [self.effect.id] if self.effect else []
        parts += [look["look"] for look in self.looks]
        return "+".join(parts) or "clip"

    def public(self, position: int | None) -> dict[str, Any]:
        return {
            "id": self.id,
            "effect": self.effect.id if self.effect else None,
            "looks": self.looks,
            "status": self.status,
            "position": position,
            "elapsed": round((self.finished or time.time()) - (self.started or self.created), 1),
            "video": f"/outputs/{self.output.name}" if self.output else None,
            "error": self.error,
        }


class Runner:
    def __init__(self) -> None:
        self._client: Comfy | None = None
        self._queue: queue.Queue[Job] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._waiting: list[str] = []
        self._lock = threading.Lock()
        OUTPUT_DIR.mkdir(exist_ok=True)
        threading.Thread(target=self._work, daemon=True, name="openfx-runner").start()

    @property
    def client(self) -> Comfy:
        if self._client is None:
            self._client = Comfy()
        return self._client

    def submit(
        self,
        effect: Effect | None,
        looks: list[dict[str, Any]],
        upload: Path,
        is_video: bool,
        detail: str | None,
        params: dict[str, Any],
    ) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            effect=effect,
            looks=looks,
            upload=upload,
            is_video=is_video,
            detail=detail,
            params=params,
        )
        with self._lock:
            self._jobs[job.id] = job
            self._waiting.append(job.id)
        self._queue.put(job)
        return job

    def status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            position = self._waiting.index(job_id) + 1 if job_id in self._waiting else None
            return job.public(position)

    def _work(self) -> None:
        while True:
            job = self._queue.get()
            with self._lock:
                self._waiting.remove(job.id)
                job.status, job.started = "running", time.time()
            try:
                job.output = self._render(job)
                job.status = "done"
            except Exception as exc:  # surfaced to the UI, full trace to the console
                traceback.print_exc()
                job.status, job.error = "failed", str(exc) or type(exc).__name__
            finally:
                job.finished = time.time()

    def _render(self, job: Job) -> Path:
        seed = job.params.get("seed", random.randint(0, 2**31 - 1))
        if job.effect is not None:
            workflow = job.effect.workflow
            values = self._effect_values(job, seed)
        else:
            workflow = LOOK_VIDEO_WORKFLOW if job.is_video else LOOK_STILL_WORKFLOW
            values = {**job.params, "seed": seed}
            values["video" if job.is_video else "image"] = self.client.assets.from_file(job.upload)

        bindings = load_bindings(workflow)
        wf = self.client.workflows.from_file(workflow)
        for name, value in values.items():
            if name in bindings:
                node_id, field_name = bindings[name]
                wf.set_input(node_id, field_name, value)
        if job.effect is not None:
            for node_id, inputs in job.effect.extra_inputs.items():
                for field_name, value in inputs.items():
                    wf.set_input(node_id, field_name, value)
        self._apply_looks(wf.json, bindings, job.looks, seed)

        result = self.client.run(wf)
        videos = [o for o in result.outputs if o.content_type.startswith("video/")]
        if not videos:
            raise RuntimeError("the workflow finished without producing a video")
        out = OUTPUT_DIR / f"{job.id}-{job.label}{Path(videos[0].name).suffix or '.mp4'}"
        videos[0].to_file(out)
        return out

    def _effect_values(self, job: Job, seed: int) -> dict[str, Any]:
        effect = job.effect
        values: dict[str, Any] = {**effect.defaults, **job.params, "seed": seed}
        values["prompt"] = build_prompt(effect, job.detail)
        if effect.negative_prompt is not None:
            values["negative_prompt"] = effect.negative_prompt
        values["image"] = self.client.assets.from_file(job.upload)
        return values

    @staticmethod
    def _apply_looks(graph: dict[str, Any], bindings: dict[str, Any], looks: list[dict[str, Any]], seed: int) -> None:
        """Splice an OpenFX Look Stack between the decoded frames and the encoder.

        ``bindings["frames"]`` names where frames come from and which input
        consumes them, e.g. ``{"from": ["320:315", 0], "into": ["320:310", "images"]}``.
        Look-only workflows already contain the stack node; there we only set it.
        """
        stack = json.dumps(looks)
        if LOOK_STACK_NODE in graph:
            graph[LOOK_STACK_NODE]["inputs"].update(stack=stack, seed=seed)
            return
        if not looks:
            return
        frames = bindings.get("frames")
        if frames is None:
            raise RuntimeError("this effect's workflow does not support Looks")
        (src_node, src_slot), (dst_node, dst_input) = frames["from"], frames["into"]
        graph[LOOK_STACK_NODE] = {
            "class_type": "OpenFXLookStack",
            "inputs": {"images": [src_node, src_slot], "stack": stack, "seed": seed, "fps": 25.0},
        }
        graph[dst_node]["inputs"][dst_input] = [LOOK_STACK_NODE, 0]
