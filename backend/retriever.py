from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import chromadb
from chromadb.config import Settings
from chromadb.errors import InvalidArgumentError


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    page: int
    section: Optional[str]
    chunk_id: str
    distance: float


def _storage_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "storage", "chroma")


def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=_storage_dir(),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(name: str = "bike_manual_chunks"):
    client = _get_client()
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def upsert_chunks(
    *,
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
) -> None:
    col_name = "bike_manual_chunks"
    col = get_collection(name=col_name)
    try:
        col.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        return
    except InvalidArgumentError as e:
        # Most common cause: switching embedding providers changes vector dimension.
        # The persisted collection is then incompatible with new embeddings.
        msg = str(e).lower()
        if "dimension" not in msg:
            raise

        client = _get_client()
        try:
            client.delete_collection(name=col_name)
        except Exception:
            # If deletion fails, re-raise original error.
            raise e

        col = client.get_or_create_collection(name=col_name, metadata={"hnsw:space": "cosine"})
        col.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)


def similarity_search(
    *,
    query_embedding: list[float],
    k: int = 5,
) -> list[RetrievedChunk]:
    col = get_collection()
    res = col.query(query_embeddings=[query_embedding], n_results=k, include=["documents", "metadatas", "distances"])
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: list[RetrievedChunk] = []
    for doc, meta, dist in zip(docs, metas, dists):
        if not isinstance(meta, dict):
            continue
        out.append(
            RetrievedChunk(
                text=str(doc or ""),
                page=int(meta.get("page", -1)),
                section=(str(meta["section"]) if meta.get("section") else None),
                chunk_id=str(meta.get("chunk_id", "")),
                distance=float(dist),
            )
        )
    return out

