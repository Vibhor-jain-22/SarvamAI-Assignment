from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from .embedding import get_llm_provider
from .rag_pipeline import FALLBACK_ANSWER, answer_query, ingest_pdf


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("api")


app = FastAPI(title="Bike Troubleshooting Assistant (RAG)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def no_cache_for_ui_and_static(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path or ""
    if path == "/" or path.startswith("/static"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        return JSONResponse({"message": "Frontend not found. Ensure the frontend/ folder exists next to backend/."})
    return FileResponse(index_path)


@app.get("/ui")
def serve_ui():
    return RedirectResponse(url="/", status_code=307)


@app.post("/upload")
async def upload(pdf: UploadFile = File(...)):
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file. Please upload a PDF.")

    try:
        get_llm_provider()
    except Exception as e:
        # Provider misconfiguration (e.g. missing API key) should be surfaced clearly.
        raise HTTPException(status_code=400, detail=str(e))

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
        contents = await pdf.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty PDF upload.")
        tmp.write(contents)

    try:
        manifest = ingest_pdf(pdf_path=tmp_path, original_filename=pdf.filename)
        return JSONResponse({"status": "ok", "manifest": manifest})
    except Exception as e:
        logger.exception("Upload/ingest failed")
        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.post("/query")
async def query(request: Request, question: Optional[str] = Form(default=None)):
    try:
        get_llm_provider()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Support both JSON { "question": "..."} and multipart/form-data (question).
    q = (question or "").strip()
    content_type = (request.headers.get("content-type") or "").lower()
    if not q and "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                q = str(body.get("question") or "").strip()
        except Exception:
            q = ""

    result = answer_query(question=q)
    return JSONResponse(result)

