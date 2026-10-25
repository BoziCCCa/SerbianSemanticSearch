"""
src/search.py

Interactive search CLI.

Usage:
  python src/search.py                 # REPL loop; "exit" or Ctrl+C quits
  python src/search.py "nikola tesla"  # single query, one-shot

For each query it prints top-5 from semantic search (E5 + Qdrant) and top-5
from BM25. The query is accepted in either script — tokenize() transliterates
Cyrillic to Latin before BM25 scoring, and the E5 encoder handles both
scripts natively.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import vector_store as vs
from embed import MODEL_NAME, QUERY_PREFIX
from keyword_search import BM25Index, SearchHit

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")   # type: ignore[attr-defined]


def semantic_search(client, collection_name: str,
                    model: SentenceTransformer,
                    query: str, k: int = 5) -> list[SearchHit]:
    """Encode with E5's required 'query: ' prefix, hit Qdrant grouped search."""
    vec = model.encode([QUERY_PREFIX + query],
                       normalize_embeddings=True)[0].tolist()
    response = vs.semantic_search_grouped(client, collection_name, vec, limit=k)
    hits: list[SearchHit] = []
    for group in response.groups:
        best = group.hits[0]
        p = best.payload or {}
        hits.append(SearchHit(
            article_id=int(p.get("article_id", 0)),
            title=p.get("title", "?"),
            category=p.get("category", "?"),
            filename=p.get("filename", "?"),
            score=float(best.score),
            snippet=(p.get("text") or "")[:220],
        ))
    return hits


def _print_hits(label: str, hits: list[SearchHit]) -> None:
    print(f"\n-- {label} --")
    if not hits:
        print("  (no results)")
        return
    for i, h in enumerate(hits, 1):
        print(f"  {i}. [{h.score:7.3f}]  ({h.category:<12}) {h.title}")


def run_query(query: str, bm25: BM25Index,
              model: SentenceTransformer, client, collection: str) -> None:
    _print_hits("Semantic (E5 + Qdrant)", semantic_search(client, collection, model, query))
    _print_hits("BM25", bm25.search(query))


def main() -> None:
    load_dotenv()
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    collection = os.getenv("QDRANT_COLLECTION", "serbian_wiki")

    root = Path(__file__).resolve().parents[1]
    meta = root / "data" / "metadata.csv"
    raw = root / "data" / "raw"

    print("Building BM25 index…")
    bm25 = BM25Index.build(meta, raw)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}…")
    model = SentenceTransformer(MODEL_NAME, device=device)

    try:
        client = vs.get_client(host, port)
    except vs.QdrantUnreachable as e:
        print(f"ERROR: {e}")
        print("Run `docker compose up -d` from the project root and try again.")
        sys.exit(1)

    # Single-query mode: everything after the script name is the query.
    if len(sys.argv) > 1:
        run_query(" ".join(sys.argv[1:]), bm25, model, client, collection)
        return

    # REPL mode
    print("\nReady. Type a query (or 'exit' to quit).")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        run_query(query, bm25, model, client, collection)


if __name__ == "__main__":
    main()
