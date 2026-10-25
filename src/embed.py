"""
src/embed.py

Reads data/metadata.csv, chunks every article, embeds each chunk with
intfloat/multilingual-e5-base, and upserts the whole thing into Qdrant.

Why chunking: the E5 encoder has a 512-token input limit. Our Serbian
articles routinely run several thousand tokens — if we just truncated them,
only the lead section would be searchable. Splitting into ~350-word chunks
with 50-word overlap keeps every paragraph reachable, and Qdrant's grouping
query (see vector_store.semantic_search_grouped) collapses the chunks back
to one hit per article at search time.

Why the "passage: " prefix: the E5 family was trained with asymmetric
prefixes — passages at ingest, "query: " at search. Skipping them silently
tanks retrieval quality. See https://huggingface.co/intfloat/multilingual-e5-base
"""
from __future__ import annotations

import csv
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

import vector_store as vs

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]


# ============================================================================
# Config
# ============================================================================

MODEL_NAME = "intfloat/multilingual-e5-base"
PASSAGE_PREFIX = "passage: "            # required by E5 for documents
QUERY_PREFIX = "query: "                # exported for search.py / compare.py

CHUNK_WORDS = 350                       # roughly the 512-token budget
CHUNK_OVERLAP = 50                      # context carry-over between chunks
ENCODE_BATCH = 32                       # bump to 64 on GPU, drop to 8 on tight CPU


# ============================================================================
# Chunking
# ============================================================================

def chunk_text(text: str,
               size: int = CHUNK_WORDS,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split on whitespace and emit overlapping fixed-width windows."""
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        if i + size >= len(words):
            break
        i += step
    return chunks


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    load_dotenv()  # read QDRANT_HOST/PORT/COLLECTION from .env if present
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection = os.getenv("QDRANT_COLLECTION", "serbian_wiki")

    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    meta_path = root / "data" / "metadata.csv"
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model -----------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device} (first run downloads ~450 MB)…")
    model = SentenceTransformer(MODEL_NAME, device=device)
    vector_size = model.get_sentence_embedding_dimension()
    print(f"  vector size: {vector_size}")

    # --- Build chunk list -----------------------------------------------
    print(f"Reading {meta_path.name} and chunking articles…")
    with meta_path.open(encoding="utf-8-sig", newline="") as f:
        articles = list(csv.DictReader(f))
    print(f"  {len(articles)} articles on disk")

    chunks: list[dict] = []
    for row in articles:
        body = (raw_dir / row["filename"]).read_text(encoding="utf-8")
        for idx, piece in enumerate(chunk_text(body)):
            chunks.append({
                "article_id": int(row["id"]),
                "title": row["title"],
                "category": row["category"],
                "filename": row["filename"],
                "chunk_index": idx,
                "text": piece,
            })
    print(f"  {len(chunks)} chunks total")

    # --- Embed ----------------------------------------------------------
    print("Embedding chunks…")
    passages = [PASSAGE_PREFIX + c["text"] for c in chunks]
    embeddings = model.encode(
        passages,
        batch_size=ENCODE_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,         # cosine distance needs unit vectors
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)
    print(f"  embeddings shape: {embeddings.shape}")

    # --- Connect to Qdrant ---------------------------------------------
    print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}…")
    try:
        client = vs.get_client(qdrant_host, qdrant_port)
    except vs.QdrantUnreachable as e:
        print(f"ERROR: {e}")
        print("Run `docker compose up -d` from the project root and try again.")
        sys.exit(1)

    print(f"Recreating collection '{collection}' (dim={vector_size}, cosine)…")
    vs.recreate_collection(client, collection, vector_size)

    # --- Build + upsert points -----------------------------------------
    print("Building point structs…")
    point_ids: list[str] = []
    points: list[PointStruct] = []
    for chunk, vec in zip(chunks, embeddings):
        pid = str(uuid.uuid4())
        point_ids.append(pid)
        points.append(PointStruct(
            id=pid,
            vector=vec.tolist(),
            payload={
                "article_id": chunk["article_id"],
                "title": chunk["title"],
                "category": chunk["category"],
                "filename": chunk["filename"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            },
        ))

    print(f"Upserting {len(points)} points…")
    vs.upsert_points(client, collection, points)

    # --- Export embeddings + id map ------------------------------------
    print("Exporting embeddings.npy and embedding_ids.csv to results/…")
    np.save(results_dir / "embeddings.npy", embeddings)
    with (results_dir / "embedding_ids.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qdrant_point_id", "article_id", "chunk_index"])
        for pid, chunk in zip(point_ids, chunks):
            w.writerow([pid, chunk["article_id"], chunk["chunk_index"]])

    final_count = vs.collection_point_count(client, collection)
    print(f"\nDone. Collection '{collection}' now has {final_count} points.")


if __name__ == "__main__":
    main()
