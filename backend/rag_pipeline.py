from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import pdfplumber
from pypdf import PdfReader

from .embedding import get_llm_provider
from .retriever import RetrievedChunk, reset_collection, similarity_search, upsert_chunks


logger = logging.getLogger("rag")

FALLBACK_ANSWER = "Sorry, this information is not available in the manual."

SYSTEM_PROMPT = """You are a document question-answering assistant.

Answer ONLY using the provided context.
If the answer is not present in the context, say:
'Sorry, this information is not available in the manual.'

Do NOT make assumptions.
Do NOT use external knowledge.
Always be grounded in the provided document.

If the user asks for a derived value (e.g. totals, sums, durations), you MAY do basic arithmetic
ONLY using numbers/dates explicitly present in the provided context. Show the calculation briefly.
"""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    page: int
    section: Optional[str]


def _storage_root() -> str:
    return os.path.join(os.path.dirname(__file__), "storage")


def _ensure_storage_dirs() -> None:
    os.makedirs(os.path.join(_storage_root(), "chroma"), exist_ok=True)


def _manifest_path() -> str:
    return os.path.join(_storage_root(), "manifest.json")


def _eval_log_path() -> str:
    return os.path.join(_storage_root(), "eval_logs.jsonl")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(data: dict[str, Any]) -> None:
    _ensure_storage_dirs()
    with open(_manifest_path(), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_manifest() -> Optional[dict[str, Any]]:
    if not os.path.exists(_manifest_path()):
        return None
    try:
        with open(_manifest_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        logger.exception("Failed to read manifest")
    return None


def has_ingested_manual() -> bool:
    return os.path.exists(_manifest_path())


def log_evaluation_event(payload: dict[str, Any]) -> None:
    _ensure_storage_dirs()
    payload = dict(payload)
    payload.setdefault("ts", _utc_now_iso())
    with open(_eval_log_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _normalize_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_pdf_pages(pdf_path: str) -> list[tuple[int, str]]:
    """
    Returns list of (page_number_1_indexed, page_text).
    Tries pdfplumber first; falls back to pypdf if needed.
    """
    pages: list[tuple[int, str]] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append((idx, _normalize_text(text)))
        if any(t for _, t in pages):
            return pages
        # if everything is empty, fall through to pypdf
    except Exception:
        logger.exception("pdfplumber failed; falling back to pypdf")

    reader = PdfReader(pdf_path)
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((idx, _normalize_text(text)))
    return pages


_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+)$")


def detect_section(text: str) -> Optional[str]:
    """
    Best-effort section detection.
    Heuristics:
    - First non-empty line that matches numbering like '2.1 Something'
    - Or a short ALL-CAPS line
    """
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _HEADING_RE.match(line)
        if m:
            return f"{m.group(1)} {m.group(2)}".strip()
        if len(line) <= 80 and line.isupper() and any(c.isalpha() for c in line):
            return line
        return None
    return None


_SENTENCE_END_RE = re.compile(r"[.!?]\s")


def _find_best_sentence_boundary(text: str, target: int, window: int = 200) -> int:
    """
    Try to move a cut point close to target onto a sentence boundary.
    Returns an index in [0, len(text)].
    """
    n = len(text)
    if n <= target:
        return n
    lo = max(0, target - window)
    hi = min(n, target + window)
    segment = text[lo:hi]

    best = None
    best_dist = None
    for m in _SENTENCE_END_RE.finditer(segment):
        cut = lo + m.end()
        dist = abs(cut - target)
        if best is None or dist < (best_dist or 10**9):
            best, best_dist = cut, dist
    if best is not None:
        return best
    return target


def recursive_chunk_text(
    *,
    text: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Recursive-ish chunking:
    - Prefer splitting by paragraphs, then lines, then sentence boundaries
    - Finally fall back to near sentence boundary cuts around chunk_size
    Overlap is applied on the final chunk list.
    """
    text = _normalize_text(text)
    if not text:
        return []

    separators = ["\n\n", "\n", ". "]

    def split_with_sep(t: str, sep: str) -> list[str]:
        parts = [p.strip() for p in t.split(sep)]
        return [p for p in parts if p]

    def pack(parts: list[str]) -> list[str]:
        chunks: list[str] = []
        buf = ""
        for p in parts:
            candidate = (buf + ("\n\n" if buf else "") + p).strip()
            if len(candidate) <= chunk_size:
                buf = candidate
                continue
            if buf:
                chunks.append(buf)
                buf = p
            else:
                # single part too large; cut it
                start = 0
                while start < len(p):
                    tgt = start + chunk_size
                    cut = _find_best_sentence_boundary(p, tgt)
                    chunks.append(p[start:cut].strip())
                    start = max(cut - chunk_overlap, start + 1)
                buf = ""
        if buf:
            chunks.append(buf)
        return [c for c in (c.strip() for c in chunks) if c]

    parts = [text]
    for sep in separators:
        next_parts: list[str] = []
        for part in parts:
            if len(part) <= chunk_size:
                next_parts.append(part)
            else:
                next_parts.extend(split_with_sep(part, sep))
        parts = next_parts

    # Pack to target size and apply overlap at text level
    packed = pack(parts)
    if not packed:
        return []

    overlapped: list[str] = []
    prev = ""
    for chunk in packed:
        if prev and chunk_overlap > 0:
            prefix = prev[-chunk_overlap:]
            combined = (prefix + "\n" + chunk).strip()
            overlapped.append(combined)
        else:
            overlapped.append(chunk)
        prev = chunk
    return overlapped


def build_chunks_from_pages(
    pages: list[tuple[int, str]],
    *,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_num, page_text in pages:
        if not page_text:
            continue
        section = detect_section(page_text)
        page_chunks = recursive_chunk_text(text=page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, c in enumerate(page_chunks):
            chunk_id = f"p{page_num:04d}_c{i:04d}"
            chunks.append(Chunk(chunk_id=chunk_id, text=c, page=page_num, section=section))
    return chunks


def ingest_pdf(
    *,
    pdf_path: str,
    original_filename: str,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> dict[str, Any]:
    _ensure_storage_dirs()
    doc_id = uuid.uuid4().hex
    # Ensure a clean index per upload so results never mix across documents.
    reset_collection()
    pages = extract_pdf_pages(pdf_path)
    num_pages = len(pages)
    chunks = build_chunks_from_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info("Extracted %s pages", num_pages)
    logger.info("Created %s chunks (chunk_size=%s overlap=%s)", len(chunks), chunk_size, chunk_overlap)

    client = get_llm_provider()
    ids: list[str] = []
    docs: list[str] = []
    embeds: list[list[float]] = []
    metas: list[dict[str, Any]] = []

    for ch in chunks:
        emb = client.embed_text(ch.text)
        ids.append(ch.chunk_id)
        docs.append(ch.text)
        embeds.append(emb)
        metas.append({"page": ch.page, "section": ch.section, "chunk_id": ch.chunk_id, "doc_id": doc_id})

    if ids:
        upsert_chunks(ids=ids, documents=docs, embeddings=embeds, metadatas=metas)

    manifest = {
        "doc_id": doc_id,
        "last_uploaded_filename": original_filename,
        "ingested_at": _utc_now_iso(),
        "chunking": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        "counts": {"pages": num_pages, "chunks": len(chunks)},
    }
    write_manifest(manifest)
    return manifest


def _format_context(chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        label = f"Source {i} (page {ch.page})"
        if ch.section:
            label += f" [section: {ch.section}]"
        lines.append(f"{label}:\n{ch.text}")
    return "\n\n---\n\n".join(lines)


_SUMMARY_HINTS = (
    "summary",
    "summarize",
    "summarise",
    "overview",
    "outline",
    "tl;dr",
    "tldr",
    "key points",
    "high level",
)

_AGGREGATION_HINTS = (
    "total",
    "sum",
    "add up",
    "overall",
    "years",
    "months",
    "duration",
    "tenure",
    "experience",
    "worked",
    "how long",
)


def _is_summary_question(q: str) -> bool:
    q = (q or "").strip().lower()
    return any(h in q for h in _SUMMARY_HINTS)


def _is_aggregation_question(q: str) -> bool:
    q = (q or "").strip().lower()
    return any(h in q for h in _AGGREGATION_HINTS)


def answer_query(
    *,
    question: str,
    k: int = 5,
    distance_threshold: float = 0.35,
) -> dict[str, Any]:
    question = (question or "").strip()
    if not question:
        return {"answer": FALLBACK_ANSWER, "sources": []}
    if not has_ingested_manual():
        return {"answer": "Please upload a bike manual PDF first.", "sources": []}

    manifest = read_manifest() or {}
    doc_id = manifest.get("doc_id")

    client = get_llm_provider()
    q_emb = client.embed_text(question)
    is_summary = _is_summary_question(question)
    is_agg = _is_aggregation_question(question)
    # Summary-style questions often don't match a single chunk strongly; retrieve more.
    retrieve_k = 25 if (is_summary or is_agg) else max(k, 10)
    where = {"doc_id": doc_id} if isinstance(doc_id, str) and doc_id else None
    retrieved = similarity_search(query_embedding=q_emb, k=retrieve_k, where=where)
    # Backward-compat / safety net: if doc_id filtering yields nothing (e.g. old indexes),
    # retry without a where-clause.
    if where is not None and not retrieved:
        retrieved = similarity_search(query_embedding=q_emb, k=retrieve_k, where=None)

    logger.info(
        "Retrieved %s chunks: %s",
        len(retrieved),
        [{"id": r.chunk_id, "page": r.page, "dist": round(r.distance, 4)} for r in retrieved],
    )

    # Chroma cosine distance: 0 is most similar, larger is worse.
    # For summary-style questions, we intentionally include the top chunks even if the
    # embedding similarity isn't super tight, otherwise we incorrectly fall back.
    if is_summary:
        filtered = [r for r in retrieved if r.text.strip()]
    else:
        filtered = [r for r in retrieved if r.distance <= distance_threshold and r.text.strip()]
        # If thresholding filtered everything out, fall back to the top chunks anyway.
        # This improves recall for questions like "bike name" or "submerged in water"
        # where the embedding match can be weaker, but the answer exists in the doc.
        if not filtered:
            filtered = [r for r in retrieved if r.text.strip()]
    if not filtered:
        log_evaluation_event(
            {
                "query": question,
                "retrieved": [{"chunk_id": r.chunk_id, "page": r.page, "distance": r.distance} for r in retrieved],
                "answer": FALLBACK_ANSWER,
                "fallback": True,
            }
        )
        return {"answer": FALLBACK_ANSWER, "sources": []}

    context = _format_context(filtered)
    user_prompt = (
        "Use ONLY the context below to answer the question.\n"
        "If the user asks for a summary/overview, produce a concise, structured summary grounded in the context.\n\n"
        "If the user asks for a total / duration / years of experience, extract the relevant dates or numbers from the context and do the math.\n"
        "If required dates/numbers are missing or ambiguous, ask for clarification or respond with the fallback.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}"
    )
    answer = client.chat_answer(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
    answer = answer.strip()

    # Guardrail: if model didn't comply, enforce fallback.
    if not answer or FALLBACK_ANSWER.lower() in answer.lower():
        final_answer = FALLBACK_ANSWER
    else:
        final_answer = answer

    # Return more sources for summaries since they naturally span more of the doc.
    sources_k = 10 if is_summary else k
    sources = [{"text": r.text, "page": r.page} for r in filtered[:sources_k]]

    log_evaluation_event(
        {
            "query": question,
            "retrieved": [
                {
                    "chunk_id": r.chunk_id,
                    "page": r.page,
                    "section": r.section,
                    "distance": r.distance,
                    "text_preview": r.text[:400],
                }
                for r in retrieved
            ],
            "answer": final_answer,
            "fallback": final_answer == FALLBACK_ANSWER,
        }
    )

    return {"answer": final_answer, "sources": sources}

