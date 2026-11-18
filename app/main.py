"""Thin API: retrieval stays inspectable and answer generation comes later."""

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.store import collection_name, get_client, get_model

app = FastAPI(title="PostgreSQL Docs Retrieval", version="0.1.0")


class SearchResult(BaseModel):
    chunk_id: str
    section_id: str
    title: str
    url: str
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    strategy: str
    results: list[SearchResult]


@lru_cache(maxsize=1)
def resources() -> tuple[QdrantClient, SentenceTransformer]:
    # Lazy initialization: unit tests and /health do not download model weights.
    return get_client(), get_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/search", response_model=SearchResponse)
def search(q: str, strategy: str = "fixed", k: int = Query(default=5, ge=1, le=20),):
    if strategy not in {"fixed", "paragraph"}:
        raise HTTPException(status_code=422, detail="strategy must be fixed or paragraph")
    client, model = resources()
    collection = collection_name(strategy)
    if not client.collection_exists(collection):
        raise HTTPException(status_code=503, detail=f"Collection {collection} is missing; run ingestion")
    vector = model.encode(q, normalize_embeddings=True).tolist()
    matches = client.query_points(collection_name=collection, query=vector, limit=k,
                                  with_payload=True).points
    results = []
    for point in matches:
        payload = point.payload or {}
        results.append(SearchResult(
            chunk_id=payload["chunk_id"], section_id=payload["section_id"],
            title=payload["title"], url=payload["url"], text=payload["text"],
            score=point.score,
        ))
    return SearchResponse(query=q, strategy=strategy, results=results)

from app.answering import install_ask_route

install_ask_route(app, search)
