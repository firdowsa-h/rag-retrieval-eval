import uuid

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from app.chunking import Chunk
from app.config import settings


def collection_name(strategy: str) -> str:
    if strategy not in {"fixed", "paragraph"}:
        raise ValueError("Unknown chunking strategy")
    return f"{settings.collection_prefix}_{strategy}"


def get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def get_client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, timeout=30)


def ingest(client: QdrantClient, model: SentenceTransformer,
           chunks: list[Chunk], collection: str) -> None:
    """Recreate a collection for reproducible local experiments."""
    if not chunks:
        raise ValueError("Refusing to create an empty collection")
    dimension = model.get_sentence_embedding_dimension()
    if dimension is None:
        raise ValueError("Embedding model did not report its vector dimension")
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(collection_name=collection,
                             vectors_config=models.VectorParams(
                                 size=dimension, distance=models.Distance.COSINE))
    for offset in range(0, len(chunks), 64):
        batch = chunks[offset:offset + 64]
        vectors = model.encode([c.text for c in batch], normalize_embeddings=True).tolist()
        points = [models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
            vector=vector,
            payload={"chunk_id": c.chunk_id, "section_id": c.section_id,
                     "title": c.title, "url": c.url, "text": c.text},
        ) for c, vector in zip(batch, vectors)]
        client.upsert(collection_name=collection, points=points, wait=True)
