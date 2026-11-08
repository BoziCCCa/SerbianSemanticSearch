#All Qdrant interaction; import this, not qdrant_client.
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm


class QdrantUnreachable(RuntimeError):
    pass


def get_client(host: str = "localhost", port: int = 6333, timeout: int = 10) -> QdrantClient:
    #Probe the server so callers can fail fast.
    client = QdrantClient(host=host, port=port, timeout=timeout)
    try:
        client.get_collections()
    except Exception as e:
        raise QdrantUnreachable(f"Qdrant not reachable at {host}:{port}: {e}") from e
    return client


def recreate_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    #Cosine distance + payload indices on article_id and category.
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    client.create_payload_index(name, field_name="article_id",
                                field_schema=PayloadSchemaType.INTEGER)
    client.create_payload_index(name, field_name="category",
                                field_schema=PayloadSchemaType.KEYWORD)


def upsert_points(client: QdrantClient, name: str,
                  points: list[PointStruct], batch_size: int = 128) -> None:
    for i in tqdm(range(0, len(points), batch_size), desc="upsert", unit="batch"):
        client.upsert(collection_name=name, points=points[i:i + batch_size])


def _category_filter(category: str | None) -> Filter | None:
    if not category:
        return None
    return Filter(must=[FieldCondition(key="category",
                                       match=MatchValue(value=category))])


def semantic_search_grouped(client: QdrantClient, name: str,
                            query_vector: list[float],
                            limit: int = 5,
                            category: str | None = None):
    #Group by article_id so results are distinct articles.
    return client.query_points_groups(
        collection_name=name,
        query=query_vector,
        group_by="article_id",
        limit=limit,
        group_size=1,
        with_payload=True,
        query_filter=_category_filter(category),
    )


def collection_point_count(client: QdrantClient, name: str) -> int:
    return client.get_collection(name).points_count
