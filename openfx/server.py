"""OpenFX web server: effect catalog, uploads, job status, and the web UI."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .effects import EFFECTS_DIR, ROOT, load_effects
from .runner import OUTPUT_DIR, Runner

UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
LOOKS_CATALOG = ROOT / "web" / "looks.json"
MAX_UPLOAD_BYTES = 60 * 1024 * 1024
IMAGE_TYPES = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
VIDEO_TYPES = {"video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm"}
MAX_LOOKS = 4

# Upper bounds that keep a 12 GB card out of trouble; manifests set the defaults.
LIMITS = {"width": (256, 1280), "height": (256, 1280), "seconds": (1, 10)}

app = FastAPI(title="OpenFX")
effects = load_effects()
looks = {look["id"]: look for look in json.loads(LOOKS_CATALOG.read_text(encoding="utf-8"))} if LOOKS_CATALOG.is_file() else {}
runner = Runner()


@app.get("/api/catalog")
def catalog() -> dict:
    return {"effects": [e.public() for e in effects.values()], "looks": list(looks.values())}


def _parse_looks(raw: str | None) -> list[dict]:
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(422, "looks must be a JSON list") from None
    if not isinstance(items, list) or len(items) > MAX_LOOKS:
        raise HTTPException(422, f"pick up to {MAX_LOOKS} looks")
    stack = []
    for item in items:
        look_id = item.get("look") if isinstance(item, dict) else None
        if look_id not in looks:
            raise HTTPException(404, f"unknown look: {look_id}")
        intensity = float(item.get("intensity", 1.0))
        stack.append({"look": look_id, "intensity": min(max(intensity, 0.0), 1.0)})
    return stack


@app.post("/api/generate")
async def generate(
    upload: UploadFile = File(...),
    effect: str | None = Form(None),
    looks_json: str | None = Form(None, alias="looks"),
    detail: str | None = Form(None),
    width: int | None = Form(None),
    height: int | None = Form(None),
    seconds: int | None = Form(None),
) -> dict:
    chosen = None
    if effect:
        chosen = effects.get(effect)
        if chosen is None:
            raise HTTPException(404, f"unknown effect: {effect}")
    stack = _parse_looks(looks_json)
    if chosen is None and not stack:
        raise HTTPException(422, "pick an effect or at least one look")

    content_type = upload.content_type or ""
    is_video = content_type in VIDEO_TYPES
    suffix = VIDEO_TYPES.get(content_type) or IMAGE_TYPES.get(content_type)
    if suffix is None:
        raise HTTPException(415, "upload a PNG, JPEG or WebP image, or an MP4, MOV or WebM video")
    if is_video and chosen is not None and chosen.input == "image":
        raise HTTPException(422, f"{chosen.name} starts from a photo; upload an image")
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file is larger than 60 MB")
    path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(data)

    params = {}
    for name, value in (("width", width), ("height", height), ("seconds", seconds)):
        if value is not None:
            low, high = LIMITS[name]
            params[name] = min(max(value, low), high)
    if detail is not None:
        detail = detail[:300]

    job = runner.submit(chosen, stack, path, is_video, detail, params)
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
