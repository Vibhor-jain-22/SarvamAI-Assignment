from __future__ import annotations

import logging
import mimetypes
import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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


FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def serve_index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        return JSONResponse({"message": "Frontend not found. Open frontend/index.html directly."})
    return FileResponse(index_path)


@app.get("/ui")
def serve_ui():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        return JSONResponse({"message": "Frontend not found. Open frontend/index.html directly."})
    return FileResponse(index_path)


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
async def query(request: Request, question: Optional[str] = Form(default=None), image: Optional[UploadFile] = File(default=None)):
    try:
        provider = get_llm_provider()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Support both JSON { "question": "..."} and multipart/form-data (question + optional image).
    q = (question or "").strip()
    content_type = (request.headers.get("content-type") or "").lower()
    if not q and image is None and "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                q = str(body.get("question") or "").strip()
        except Exception:
            q = ""

    if image is not None:
        mime = image.content_type or mimetypes.guess_type(image.filename or "")[0] or "image/png"
        if not mime.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid image upload.")
        img_bytes = await image.read()
        if not img_bytes:
            raise HTTPException(status_code=400, detail="Empty image upload.")
        try:
            q = provider.vision_to_text(image_bytes=img_bytes, mime_type=mime)
            q = (q or "").strip()
            if not q:
                raise RuntimeError("Image produced an empty query.")
        except NotImplementedError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception("Vision-to-text failed")
            raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    result = answer_query(question=q)
    return JSONResponse(result)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
