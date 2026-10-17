"""Runs effects through the Comfy SDK, one at a time.

The SDK talks to whatever ``COMFY_BASE_URL`` points at: the local API proxy in
front of your own ComfyUI (``http://127.0.0.1:8189``, free, no key), or a Comfy
API deployment (set ``COMFY_API_KEY`` too). The code is the same either way.

A single worker thread drains the queue because one GPU renders one video at a
time; extra requests wait their turn instead of fighting for VRAM.
"""

from __future__ import annotations

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


@dataclass
class Job:
    id: str
    effect: Effect
    image: Path
    detail: str | None
    params: dict[str, Any]
    status: str = "queued"  # queued | running | done | failed
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    output: Path | None = None
    error: str | None = None

    def public(self, position: int | None) -> dict[str, Any]:
        return {
            "id": self.id,
            "effect": self.effect.id,
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

    def submit(self, effect: Effect, image: Path, detail: str | None, params: dict[str, Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], effect=effect, image=image, detail=detail, params=params)
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
        effect = job.effect
        bindings = load_bindings(effect.workflow)
        wf = self.client.workflows.from_file(effect.workflow)

        values: dict[str, Any] = {**effect.defaults, **job.params}
        values.setdefault("seed", random.randint(0, 2**31 - 1))
        values["prompt"] = build_prompt(effect, job.detail)
        if effect.negative_prompt is not None:
            values["negative_prompt"] = effect.negative_prompt
        values["image"] = self.client.assets.from_file(job.image)

        for name, value in values.items():
            if name in bindings:
                node_id, field_name = bindings[name]
                wf.set_input(node_id, field_name, value)
        for node_id, inputs in effect.extra_inputs.items():
            for field_name, value in inputs.items():
                wf.set_input(node_id, field_name, value)

        result = self.client.run(wf)
        videos = [o for o in result.outputs if o.content_type.startswith("video/")]
        if not videos:
            raise RuntimeError("the workflow finished without producing a video")
        out = OUTPUT_DIR / f"{job.id}-{effect.id}{Path(videos[0].name).suffix or '.mp4'}"
        videos[0].to_file(out)
        return out
