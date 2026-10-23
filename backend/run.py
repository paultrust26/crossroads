"""Serve the Crossroads API and the built frontend on one origin.

Usage (from the repository root):
    export OPENAI_API_KEY=...            # only needed for the LLM judge
    export OPENAI_BASE_URL=...           # optional, any OpenAI compatible endpoint
    python backend/run.py --port 8000
"""

import sys

if sys.version_info < (3, 10):
    raise SystemExit(f"Crossroads needs Python 3.10 or newer; this is Python {sys.version.split()[0]}. "
                     "Create the virtual environment with a newer python3.")

import os
from pathlib import Path


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into the environment, without overriding existing variables.

    Args:
        path: The .env file. A missing file is ignored.
    """
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file(Path(__file__).resolve().parent.parent / ".env")

import argparse
from pathlib import Path

import uvicorn
from fastapi.responses import FileResponse, JSONResponse

from crossroads.app import app

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if FRONTEND_DIR.is_dir():

    @app.get("/")
    async def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/assets/{path:path}")
    async def serve_assets(path: str):
        asset_path = FRONTEND_DIR / "assets" / path
        if asset_path.is_file():
            return FileResponse(str(asset_path))
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        if path.startswith("api/"):
            return JSONResponse({"error": "not found"}, status_code=404)
        file_path = FRONTEND_DIR / path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, reload=False)
