"""OpenFX web server: effect catalog, uploads, job status, and the web UI."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .effects import EFFECTS_DIR, ROOT, load_effects
from .runner import OUTPUT_DIR, Runner

UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
IMAGE_TYPES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}

# Upper bounds that keep a 12 GB card out of trouble; manifests set the defaults.
LIMITS = {"width": (256, 1280), "height": (256, 1280), "seconds": (1, 10)}

app = FastAPI(title="OpenFX")
effects = load_effects()
runner = Runner()


@app.get("/api/effects")
def list_effects() -> list[dict]:
    return [e.public() for e in effects.values()]


@app.post("/api/generate")
async def generate(
    effect: str = Form(...),
    image: UploadFile = File(...),
    detail: str | None = Form(None),
    width: int | None = Form(None),
    height: int | None = Form(None),
    seconds: int | None = Form(None),
) -> dict:
    chosen = effects.get(effect)
    if chosen is None:
        raise HTTPException(404, f"unknown effect: {effect}")
    suffix = IMAGE_TYPES.get(image.content_type or "")
    if suffix is None:
        raise HTTPException(415, "upload a PNG, JPEG or WebP image")
    data = await image.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image is larger than 15 MB")
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(data)

    params = {}
    for name, value in (("width", width), ("height", height), ("seconds", seconds)):
        if value is not None:
            low, high = LIMITS[name]
            params[name] = min(max(value, low), high)
    if detail is not None:
        detail = detail[:300]

    job = runner.submit(chosen, path, detail, params)
    return runner.status(job.id)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    status = runner.status(job_id)
    if status is None:
        raise HTTPException(404, "unknown job")
    return status


app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
app.mount("/effects", StaticFiles(directory=EFFECTS_DIR), name="effects")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(ROOT / "web" / "index.html"))
