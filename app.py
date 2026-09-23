"""ML Bench Studio — one place to try every model in ml-bench.

    uvicorn app:app --reload --port 8010      (from the studio/ directory)

Plugins are discovered from plugins/*.py at startup (see registry.py). The UI is a single
static page that renders whatever the plugin contract describes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from registry import Registry  # noqa: E402

app = FastAPI(title="ML Bench Studio", version="0.1.0")
registry = Registry()


@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    """Local dev tool: never let the browser serve a stale app.js / styles.css after an edit."""
    resp = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


class RunRequest(BaseModel):
    values: dict[str, Any]


@app.get("/api/plugins")
def plugins() -> dict[str, Any]:
    return {"plugins": [p.public() for p in registry.plugins.values()], "load_errors": registry.load_errors}


@app.post("/api/run/{plugin_id}")
def run(plugin_id: str, req: RunRequest) -> dict[str, Any]:
    if plugin_id not in registry.plugins:
        raise HTTPException(404, "unknown plugin")
    try:
        return registry.run(plugin_id, req.values)
    except ValueError as e:                       # validation problems -> 400 with a readable message
        raise HTTPException(400, str(e)) from None
    except FileNotFoundError as e:
        raise HTTPException(409, f"artifact missing: {e}") from None
    except Exception as e:                        # model errors surface to the UI instead of a blank 500
        raise HTTPException(500, f"{type(e).__name__}: {e}") from None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
