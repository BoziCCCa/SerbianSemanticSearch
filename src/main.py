from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import vector_store as vs
from embed import MODEL_NAME, QUERY_PREFIX
from keyword_search import BM25Index, SearchHit, TfidfIndex

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


# Hand-picked queries

QUERIES: list[dict] = [
    #exact article-title matches (BM25 territory)
    {"q": "Beograd",                            "type": "exact_title",   "intent": "geografija"},
    {"q": "Rambo Amadeus",                      "type": "exact_title",   "intent": "muzika"},
    {"q": "Burek",                              "type": "exact_title",   "intent": "hrana"},
    {"q": "Programski jezik Lua",               "type": "exact_title",   "intent": "tehnologija"},

    #paraphrase: different words than the article uses
    {"q": "glavni grad Srbije",                 "type": "paraphrase",    "intent": "geografija"},
    {"q": "srpski kantautor i tekstopisac",     "type": "paraphrase",    "intent": "muzika"},
    {"q": "jelo od lisnatog testa",             "type": "paraphrase",    "intent": "hrana"},
    {"q": "jezik za skriptovanje",              "type": "paraphrase",    "intent": "tehnologija"},

    # --- conceptual / descriptive questions ---
    {"q": "ko je osvojio Balkan u srednjem veku",       "type": "conceptual", "intent": "istorija"},
    {"q": "najveći grad u Srbiji",                      "type": "conceptual", "intent": "geografija"},
    {"q": "srpski naučnik koji je radio na elektrotehnici", "type": "conceptual", "intent": "nauka"},
    {"q": "slatko jelo sa orasima",                     "type": "conceptual", "intent": "hrana"},

    #same query, both scripts (transliteration sanity check)
    {"q": "Ђорђе Балашевић",                    "type": "cyrillic",      "intent": "muzika"},
    {"q": "Đorđe Balašević",                    "type": "latin",         "intent": "muzika"},

    #misspelled
    {"q": "Rabmo Admeus",                      "type": "misspelled",    "intent": "muzika"},
    {"q": "Beogrogad",                           "type": "misspelled",    "intent": "geografija"},

    #topics absent from the corpus
    {"q": "kvantna mehanika",                   "type": "out_of_corpus", "intent": None},
    {"q": "veštačka inteligencija",             "type": "out_of_corpus", "intent": None},

    #exact phrase from an article + multi-concept
    {"q": "austrougarski ultimatum Srbiji 1913", "type": "exact_phrase", "intent": "istorija"},
    {"q": "srpski fudbaler rođen u Beogradu",   "type": "multi_concept", "intent": "sport"},
]



# Backend adapters


def semantic_search(client, collection_name: str,
                    model: SentenceTransformer,
                    query: str, k: int = 5) -> list[SearchHit]:
    """Encode with the required E5 query prefix, hit Qdrant's grouped search."""
    vec = model.encode([QUERY_PREFIX + query],
                       normalize_embeddings=True)[0].tolist()
    response = vs.semantic_search_grouped(client, collection_name, vec, limit=k)
    hits: list[SearchHit] = []
    for group in response.groups:
        best = group.hits[0]              # group_size=1 → single best chunk
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



# Pretty-printing
def _fmt(hits: list[SearchHit]) -> str:
    if not hits:
        return "  (no results)"
    return "\n".join(
        f"  {i+1}. [{h.score:7.3f}]  ({h.category:<12}) {h.title}"
        for i, h in enumerate(hits)
    )


def print_query_block(idx: int, total: int, q: dict,
                      sem: list[SearchHit],
                      bm: list[SearchHit],
                      tf: list[SearchHit]) -> None:
    intent = q["intent"] or "none"
    print("\n" + "=" * 100)
    print(f"[{idx:2d}/{total}]  type={q['type']:<15}  intent={intent:<12}  query={q['q']!r}")
    print("=" * 100)
    print("\n-- Semantic (E5 + Qdrant) --")
    print(_fmt(sem))
    print("\n-- BM25 --")
    print(_fmt(bm))
    print("\n-- TF-IDF --")
    print(_fmt(tf))


# Main
def main() -> None:
    load_dotenv()
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    collection = os.getenv("QDRANT_COLLECTION", "serbian_wiki")

    root = Path(__file__).resolve().parents[1]
    meta = root / "data" / "metadata.csv"
    raw = root / "data" / "raw"

    print("Building BM25 and TF-IDF indices…")
    bm25 = BM25Index.build(meta, raw)
    tfidf = TfidfIndex.build(meta, raw)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}…")
    model = SentenceTransformer(MODEL_NAME, device=device)

    try:
        client = vs.get_client(host, port)
    except vs.QdrantUnreachable as e:
        print(f"ERROR: {e}")
        print("Run `docker compose up -d` from the project root and try again.")
        sys.exit(1)

    for i, q in enumerate(QUERIES, 1):
        sem = semantic_search(client, collection, model, q["q"])
        bm = bm25.search(q["q"])
        tf = tfidf.search(q["q"])
        print_query_block(i, len(QUERIES), q, sem, bm, tf)

    print("\n" + "=" * 100)
    print(f"Done. {len(QUERIES)} queries, 3 backends each.")


if __name__ == "__main__":
    main()
