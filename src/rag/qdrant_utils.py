"""Shared Qdrant helpers: deterministic point ids and vector search.

Point ids are the numeric hash of a textual key (chunk_id / paragraph_id /
doc_id), so re-indexing the same document overwrites the existing points
without creating duplicates (idempotent upsert).
"""

import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import Filter


def point_id(key: str) -> int:
    """Deterministic Qdrant point id (unsigned 64-bit) from a textual key."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def vector_search(
    client: QdrantClient,
    collection_name: str,
    vector: list[float],
    limit: int,
    query_filter: Filter | None = None,
    with_payload: bool = True,
) -> list:
    """Vector search preferring query_points, with a search fallback."""
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=with_payload,
        )
        return list(response.points)
    return list(
        client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=with_payload,
        )
    )
