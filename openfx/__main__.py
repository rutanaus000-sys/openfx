"""``python -m openfx`` starts the web app on http://127.0.0.1:7860."""

import os

import uvicorn

if __name__ == "__main__":
    os.environ.setdefault("COMFY_BASE_URL", "http://127.0.0.1:8189")
    uvicorn.run(
        "openfx.server:app",
        host=os.environ.get("OPENFX_HOST", "127.0.0.1"),
        port=int(os.environ.get("OPENFX_PORT", "7860")),
    )
